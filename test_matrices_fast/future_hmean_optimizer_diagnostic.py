import argparse
import csv
import json
import time
from collections import defaultdict

import numpy as np

import cex_restricted_space_probe as probe
import half_window_sliding_hmean_experiment as hm
from diagnose_future_hmean_retention import rowspace_mass
from second_slot_tail_bias_diagnostic import make_state


def rowspace_basis(A, tol=1e-10):
    A = np.asarray(A, dtype=np.float64)
    if A.size == 0:
        return np.zeros((A.shape[1], 0), dtype=np.float64)
    _, s, Vh = np.linalg.svd(A, full_matrices=False)
    if s.size == 0:
        return np.zeros((A.shape[1], 0), dtype=np.float64)
    keep = s > max(float(s[0]) * tol, 1e-30)
    return np.ascontiguousarray(Vh[keep, :].T, dtype=np.float64)


def orth_basis_against(B, q):
    B = np.asarray(B, dtype=np.float64)
    if B.size == 0:
        return B
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    nq = float(np.linalg.norm(q))
    if nq > 1e-30:
        q = q / nq
        B = B - q[:, None] @ (q[None, :] @ B)
    Q, R = np.linalg.qr(B)
    diag = np.abs(np.diag(R))
    if diag.size == 0:
        return np.zeros((B.shape[0], 0), dtype=np.float64)
    keep = diag > max(float(diag.max()) * 1e-10, 1e-30)
    return np.ascontiguousarray(Q[:, keep], dtype=np.float64)


def future_hmean_value_grad(A_cur, A_fut, v):
    A_cur = np.asarray(A_cur, dtype=np.float64)
    A_fut = np.asarray(A_fut, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    y = A_cur @ v
    z = A_fut @ v
    a = max(float(np.dot(y, y)), 0.0)
    b = max(float(np.dot(z, z)), 0.0)
    denom = max(a + b, 1e-30)
    h = 2.0 * a * b / denom
    grad_h = (
        (2.0 * b * b / (denom * denom)) * (2.0 * (A_cur.T @ y))
        + (2.0 * a * a / (denom * denom)) * (2.0 * (A_fut.T @ z))
    )

    e = y * y
    S = max(float(np.sum(e)), 1e-30)
    p = e / S
    p_pos = np.maximum(p, 1e-300)
    H = -float(np.sum(p * np.log(p_pos)))
    rel = max(H / np.log(max(len(e), 2)), 0.0)
    dH_de = -(np.log(p_pos) + H) / S
    grad_H = A_cur.T @ (2.0 * y * dH_de)
    grad_rel = grad_H / np.log(max(len(e), 2))
    val = h * rel
    grad = rel * grad_h + h * grad_rel
    return float(val), np.ascontiguousarray(grad, dtype=np.float64), float(a), float(b), float(rel)


def optimize_future_hmean_in_basis(A_cur, A_fut, B, starts, rng, maxit=60, tol=1e-8, random_starts=8):
    B = np.asarray(B, dtype=np.float64)
    q = B.shape[1]
    if q <= 0:
        return None

    z_starts = []
    for v0 in starts:
        if v0 is None:
            continue
        z = B.T @ np.asarray(v0, dtype=np.float64).reshape(-1)
        nz = float(np.linalg.norm(z))
        if nz > 1e-12:
            z_starts.append(z / nz)
    for _ in range(max(0, int(random_starts))):
        z = rng.standard_normal(q)
        nz = float(np.linalg.norm(z))
        if nz > 1e-12:
            z_starts.append(z / nz)

    best = None
    for z0 in z_starts:
        z = np.ascontiguousarray(z0, dtype=np.float64)
        val, grad_full, a, b, rel = future_hmean_value_grad(A_cur, A_fut, B @ z)
        stop = {"reason": "maxit", "iters": int(maxit), "grad_norm": np.nan}
        for it in range(int(maxit)):
            gz = B.T @ grad_full
            gtan = gz - z * float(z @ gz)
            gnorm = float(np.linalg.norm(gtan))
            if gnorm <= tol:
                stop = {"reason": "grad_tol", "iters": it, "grad_norm": gnorm}
                break
            alpha = 1.0
            accepted = False
            for ls_iter in range(30):
                zt = z + alpha * gtan
                nz = float(np.linalg.norm(zt))
                if nz > 1e-30:
                    zt = zt / nz
                    vt = B @ zt
                    val_t, grad_t, a_t, b_t, rel_t = future_hmean_value_grad(A_cur, A_fut, vt)
                    if np.isfinite(val_t) and val_t >= val + 1e-4 * alpha * float(gtan @ gtan):
                        z = np.ascontiguousarray(zt)
                        val, grad_full, a, b, rel = val_t, grad_t, a_t, b_t, rel_t
                        accepted = True
                        stop = {
                            "reason": "progress",
                            "iters": it + 1,
                            "grad_norm": gnorm,
                            "line_search_alpha": alpha,
                            "line_search_steps": ls_iter + 1,
                        }
                        break
                alpha *= 0.5
            if not accepted:
                stop = {"reason": "line_search_fail", "iters": it + 1, "grad_norm": gnorm}
                break
        rec = {
            "vec": np.ascontiguousarray(B @ z, dtype=np.float64),
            "score": float(val),
            "gain1": float(a),
            "gain2": float(b),
            "relH1": float(rel),
            "stop": stop,
        }
        if best is None or rec["score"] > best["score"]:
            best = rec
    return best


def future_score(A_cur, A_fut, v):
    val, _, a, b, rel = future_hmean_value_grad(A_cur, A_fut, v)
    return val, a, b, rel


def combined_score(M_gain, A_cur, v, rows_ref, state, old_row_memory):
    return probe.combined_score_component_details(
        M_gain, A_cur, v, rows_ref, state_prev=state, old_row_memory=old_row_memory
    )["score_total"]


def run_matrix(args, matrix):
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=matrix,
        n=args.n,
        preset=args.preset,
        seed=args.seed,
        r_sig=args.r_sig,
        alpha_sig=args.alpha_sig,
        alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale,
        sigma1=args.sigma1,
        v_type=args.v_type,
        shuffle_rows=args.shuffle_rows,
        row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, dtype=np.float64)
    V_exact = np.asarray(V_exact, dtype=np.float64)
    rank = int(args.rank)
    half_win = int(args.half_win)
    state = None
    old_row_memory = None
    rows = []

    for block_id, start0 in enumerate(range(0, A.shape[0] - half_win, half_win), start=1):
        if args.max_pairs is not None and block_id > args.max_pairs:
            break
        mid0 = start0 + half_win
        end0 = min(mid0 + half_win, A.shape[0])
        if end0 - mid0 < half_win:
            break
        A_cur = np.asarray(A[start0:mid0, :], dtype=work_dtype)
        A_fut = np.asarray(A[mid0:end0, :], dtype=work_dtype)
        if state is None:
            M_sketch = None
            M_gain = A_cur
            V_init = probe.row_norm_seed(A_cur, rank)
            rows_seen = A_cur.shape[0]
        else:
            B_top = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
            M_sketch = B_top
            M_gain = np.vstack([B_top, A_cur]).astype(work_dtype, copy=False)
            V_init = probe.row_norm_seed(A_cur, rank)
            rows_seen = state["rows_seen"] + A_cur.shape[0]

        V_score, _, _, _, diag = probe.entropy_iter_basis_forget(
            M_gain=M_gain,
            active_r=rank,
            rows_ref=A.shape[0],
            V_init=np.asarray(V_init, dtype=work_dtype),
            q0=args.q0,
            qmax=args.qmax,
            krylov_depth=args.krylov_depth,
            residual_tol=args.residual_tol,
            expansion_maxit=args.expansion_maxit,
            num_restarts=args.num_restarts,
            maxit=args.maxit,
            tol=args.tol,
            rng=np.random.default_rng(args.seed),
            verbose=False,
            state_prev=state,
            A_block=A_cur,
            rows_total=rows_seen,
            reduced_optimizer="cex",
            basis_selection="greedy",
            work_dtype=work_dtype,
            expansion_direction="residual",
            reuse_line_search_grad=True,
            expansion_warm_start=True,
            post_expansion_maxit=args.post_expansion_maxit,
            score_variant="combined",
            old_row_memory=old_row_memory,
            combined_rank=None,
            patience=args.patience,
            patience_rel_tol=args.patience_rel_tol,
        )
        V_default = np.ascontiguousarray(np.asarray(V_score[:, :rank], dtype=np.float64))
        union = np.vstack([A_cur, A_fut]).astype(np.float64, copy=False)
        B_union = orth_basis_against(rowspace_basis(union), V_default[:, 0])
        starts = [V_default[:, 1]]
        if V_exact.shape[1] >= 2:
            starts.append(V_exact[:, 1])
        if diag.get("Vbasis_final") is not None:
            Vbasis = np.asarray(diag["Vbasis_final"], dtype=np.float64)
            for j in range(min(Vbasis.shape[1], 8)):
                starts.append(Vbasis[:, j])
        best = optimize_future_hmean_in_basis(
            A_cur, A_fut, B_union, starts, np.random.default_rng(args.seed + 1009 * block_id),
            maxit=args.union_maxit, tol=args.union_tol, random_starts=args.union_random_starts,
        )
        default_score, default_g1, default_g2, default_rel = future_score(A_cur, A_fut, V_default[:, 1])
        if best is None:
            best_v = None
            best_score = np.nan
            best_g1 = np.nan
            best_g2 = np.nan
            best_rel = np.nan
            best_stop = {}
        else:
            best_v = best["vec"]
            best_score = best["score"]
            best_g1 = best["gain1"]
            best_g2 = best["gain2"]
            best_rel = best["relH1"]
            best_stop = best["stop"]

        default_combined = combined_score(M_gain, A_cur, V_default[:, 1], A.shape[0], state, old_row_memory)
        best_combined = (
            np.nan
            if best_v is None
            else combined_score(M_gain, A_cur, best_v, A.shape[0], state, old_row_memory)
        )
        rows.append({
            "matrix": matrix,
            "block": block_id,
            "rows_seen": mid0,
            "union_dim_after_v1": int(B_union.shape[1]),
            "optimizer_subspace_dim_v2": int(np.asarray(diag.get("subspace_dims", [np.nan]))[1])
            if len(np.asarray(diag.get("subspace_dims", []))) > 1 else np.nan,
            "optimizer_expansions_v2": int(np.asarray(diag.get("expansion_iters", [np.nan]))[1])
            if len(np.asarray(diag.get("expansion_iters", []))) > 1 else np.nan,
            "optimizer_grad_perp_v2": float(np.asarray(diag.get("grad_perp_ratio", [np.nan]))[1])
            if len(np.asarray(diag.get("grad_perp_ratio", []))) > 1 else np.nan,
            "default_future_score": default_score,
            "best_union_future_score": best_score,
            "future_score_ratio_best_over_default": best_score / max(default_score, 1e-300),
            "default_gain1": default_g1,
            "default_gain2": default_g2,
            "default_relH1": default_rel,
            "best_gain1": best_g1,
            "best_gain2": best_g2,
            "best_relH1": best_rel,
            "default_combined_score": default_combined,
            "best_union_combined_score": best_combined,
            "combined_score_ratio_best_over_default": best_combined / max(default_combined, 1e-300),
            "default_union_mass": rowspace_mass(union, V_default[:, 1]),
            "best_mgain_rowspace_mass": np.nan if best_v is None else rowspace_mass(M_gain, best_v),
            "best_state_space_mass": np.nan if state is None or best_v is None else float(
                np.linalg.norm(state["V"] @ (state["V"].T @ best_v)) ** 2
            ),
            "best_exact_align2": np.nan if best_v is None else float(abs(np.dot(best_v, V_exact[:, 1])) ** 2),
            "default_exact_align2": float(abs(np.dot(V_default[:, 1], V_exact[:, 1])) ** 2),
            "best_stop_reason": best_stop.get("reason", ""),
            "best_stop_grad_norm": best_stop.get("grad_norm", np.nan),
            "sketch_present": int(M_sketch is not None),
        })

        score_selected = np.zeros(rank, dtype=float)
        H_selected = np.zeros(rank, dtype=float)
        for j in range(rank):
            score_selected[j], _, H_selected[j] = probe.score_full_vector_details_forget(
                M_gain,
                A_cur,
                V_default[:, j],
                A.shape[0],
                state_prev=state,
                score_variant="combined",
                old_row_memory=old_row_memory,
            )
        state, V_r, _ = make_state(M_gain, V_default, H_selected, score_selected, rows_seen)
        old_row_memory, _ = probe.select_old_row_memory(
            np.asarray(A[:mid0, :], dtype=work_dtype),
            V_r.astype(work_dtype, copy=False),
            args.old_memory_size if args.old_memory_size > 0 else half_win,
            np.random.default_rng(args.seed + end0),
            return_indices=True,
        )
    return rows


def summarize(rows):
    out = []
    by_matrix = defaultdict(list)
    for row in rows:
        by_matrix[row["matrix"]].append(row)
    for matrix, rs in sorted(by_matrix.items()):
        after_first = [r for r in rs if r["block"] > 1]
        target = after_first if after_first else rs
        out.append({
            "matrix": matrix,
            "blocks": len(rs),
            "mean_future_ratio": float(np.nanmean([r["future_score_ratio_best_over_default"] for r in target])),
            "median_future_ratio": float(np.nanmedian([r["future_score_ratio_best_over_default"] for r in target])),
            "mean_combined_ratio": float(np.nanmean([r["combined_score_ratio_best_over_default"] for r in target])),
            "median_combined_ratio": float(np.nanmedian([r["combined_score_ratio_best_over_default"] for r in target])),
            "mean_default_union_mass": float(np.nanmean([r["default_union_mass"] for r in target])),
            "mean_best_mgain_mass": float(np.nanmean([r["best_mgain_rowspace_mass"] for r in target])),
            "mean_optimizer_grad_perp_v2": float(np.nanmean([r["optimizer_grad_perp_v2"] for r in target])),
            "mean_union_dim_after_v1": float(np.nanmean([r["union_dim_after_v1"] for r in target])),
            "mean_optimizer_subspace_dim_v2": float(np.nanmean([r["optimizer_subspace_dim_v2"] for r in target])),
            "mean_best_exact_align2": float(np.nanmean([r["best_exact_align2"] for r in target])),
            "mean_default_exact_align2": float(np.nanmean([r["default_exact_align2"] for r in target])),
        })
    return out


def write_text(path, summaries):
    fields = [
        "matrix",
        "blocks",
        "mean_future_ratio",
        "median_future_ratio",
        "mean_combined_ratio",
        "median_combined_ratio",
        "mean_default_union_mass",
        "mean_best_mgain_mass",
        "mean_optimizer_grad_perp_v2",
        "mean_union_dim_after_v1",
        "mean_optimizer_subspace_dim_v2",
        "mean_best_exact_align2",
        "mean_default_exact_align2",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("Future-HM optimizer diagnostic\n")
        f.write("==============================\n\n")
        f.write("For each block, optimize HM(||A_w v||^2, ||A_{w+1} v||^2) * relH(A_w v) directly inside rowspan([A_w; A_{w+1}]) after orthogonalizing against optimizer v1.\n")
        f.write("Compare that union optimum with the streaming combined-score optimizer's v2.\n\n")
        f.write("Summary by matrix\n")
        f.write("-----------------\n")
        f.write(" ".join(f"{x:<34}" for x in fields) + "\n")
        for s in summaries:
            vals = []
            for field in fields:
                v = s[field]
                vals.append(f"{v:<34.4f}" if isinstance(v, float) else f"{str(v):<34}")
            f.write(" ".join(vals) + "\n")
        f.write("\nInterpretation\n")
        f.write("--------------\n")
        f.write("future_ratio > 1 means there are better future-HM points in the two-half union than the optimizer's v2.\n")
        f.write("combined_ratio < 1 means those points are worse under the optimizer's actual combined forgetting objective, so this is objective mismatch rather than failed convergence.\n")
        f.write("best_mgain_mass near 1 means the better union point is already visible to M_gain; low values indicate lookahead-only directions that the streaming optimizer cannot see.\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices", nargs="+", default=[
        "mixed-tail-sharp",
        "mixed-tail-balanced",
        "mixed-tail-soft",
        "diffuse-diffuse",
        "static-cex",
        "etf-basket-basis",
        "residual-spiky-shocks",
        "risk-residual-panel",
    ])
    parser.add_argument("--out-prefix", default="summary/future_hmean_optimizer_diagnostic")
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--half-win", type=int, default=32)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-rows", action="store_true", default=True)
    parser.add_argument("--row-shuffle-seed", type=int, default=0)
    parser.add_argument("--old-memory-size", type=int, default=32)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--q0", type=int, default=8)
    parser.add_argument("--qmax", type=int, default=48)
    parser.add_argument("--krylov-depth", type=int, default=2)
    parser.add_argument("--residual-tol", type=float, default=0.01)
    parser.add_argument("--expansion-maxit", type=int, default=8)
    parser.add_argument("--num-restarts", type=int, default=3)
    parser.add_argument("--maxit", type=int, default=120)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--post-expansion-maxit", type=int, default=80)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--patience-rel-tol", type=float, default=1e-5)
    parser.add_argument("--union-maxit", type=int, default=60)
    parser.add_argument("--union-tol", type=float, default=1e-8)
    parser.add_argument("--union-random-starts", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--r-sig", type=int, default=2)
    parser.add_argument("--alpha-sig", type=float, default=0.003)
    parser.add_argument("--alpha-tail", type=float, default=0.0145)
    parser.add_argument("--tail-scale", type=float, default=0.99)
    parser.add_argument("--sigma1", type=float, default=0.991)
    parser.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    rows = []
    for matrix in args.matrices:
        mat_rows = run_matrix(args, matrix)
        rows.extend(mat_rows)
        print(f"done {matrix} blocks={len(mat_rows)}")
    csv_path = args.out_prefix + ".csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summaries = summarize(rows)
    json_path = args.out_prefix + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summaries": summaries, "rows": rows}, f, indent=2, sort_keys=True)
    text_path = args.out_prefix + ".txt"
    write_text(text_path, summaries)
    print(f"wrote {csv_path} {json_path} {text_path} elapsed={time.time() - t0:.3f}")


if __name__ == "__main__":
    main()
