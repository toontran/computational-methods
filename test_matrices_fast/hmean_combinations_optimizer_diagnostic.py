import argparse
import csv
import json
import time
from collections import defaultdict

import numpy as np

import cex_restricted_space_probe as probe
import half_window_sliding_hmean_experiment as hm
from diagnose_future_hmean_retention import rowspace_mass
from future_hmean_optimizer_diagnostic import (
    combined_score,
    optimize_future_hmean_in_basis,
    orth_basis_against,
    rowspace_basis,
)
from second_slot_tail_bias_diagnostic import make_state, raw_oracle_columns


POLICIES = (
    "future_hmean_triplet_online",
    "future_hmean_nested_online",
    "future_hmean_pairwise_online",
    "future_hmean_weighted_online",
)


def hmean_many_value_grad(items, eps=1e-30):
    valid = [(float(w), float(x), g) for w, x, g in items if w > 0.0 and np.isfinite(x)]
    if not valid:
        return np.nan, None
    for _, x, _ in valid:
        if x <= eps:
            return 0.0, np.zeros_like(valid[0][2])
    weight_sum = sum(w for w, _, _ in valid)
    recip = sum(w / x for w, x, _ in valid)
    val = weight_sum / max(recip, eps)
    grad = np.zeros_like(valid[0][2])
    scale = weight_sum / max(recip * recip, eps)
    for w, x, g in valid:
        grad = grad + scale * w * g / max(x * x, eps)
    return float(val), np.ascontiguousarray(grad, dtype=np.float64)


def entropy_value_grad(A_cur, v):
    A_cur = np.asarray(A_cur, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    y = A_cur @ v
    e = y * y
    S = max(float(np.sum(e)), 1e-30)
    p = e / S
    p_pos = np.maximum(p, 1e-300)
    H_nat = -float(np.sum(p * np.log(p_pos)))
    rel = max(H_nat / np.log(max(len(e), 2)), 0.0)
    dH_de = -(np.log(p_pos) + H_nat) / (S * np.log(max(len(e), 2)))
    grad = A_cur.T @ (2.0 * y * dH_de)
    return float(rel), np.ascontiguousarray(grad, dtype=np.float64)


def quad_share_value_grad(A, v, denom):
    if A is None:
        return 1.0, np.zeros_like(v)
    A = np.asarray(A, dtype=np.float64)
    y = A @ v
    raw = float(np.dot(y, y))
    den = max(float(denom), 1e-30)
    return raw / den, np.ascontiguousarray(2.0 * (A.T @ y) / den, dtype=np.float64)


def combination_value_grad(policy, A_cur, A_fut, A_sketch, denoms, weights, v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    H1, grad_H1 = entropy_value_grad(A_cur, v)
    sk, grad_sk = quad_share_value_grad(A_sketch, v, denoms["sketch"])
    g1, grad_g1 = quad_share_value_grad(A_cur, v, denoms["gain1"])
    g2, grad_g2 = quad_share_value_grad(A_fut, v, denoms["gain2"])
    c1, grad_c1 = quad_share_value_grad(None, v, 1.0)
    c2, grad_c2 = quad_share_value_grad(None, v, 1.0)
    if A_sketch is None:
        c1 = g1
        grad_c1 = grad_g1
        c2 = g2
        grad_c2 = grad_g2
    else:
        sk_raw_denom = max(float(denoms["sketch_raw_for_concat"]), 1e-30)
        # The concat denominators are fixed separately from the sketch/gain shares.
        sk_y = np.asarray(A_sketch, dtype=np.float64) @ v
        cur_y = np.asarray(A_cur, dtype=np.float64) @ v
        fut_y = np.asarray(A_fut, dtype=np.float64) @ v
        sk_raw = float(np.dot(sk_y, sk_y))
        g1_raw = float(np.dot(cur_y, cur_y))
        g2_raw = float(np.dot(fut_y, fut_y))
        c1 = (sk_raw + g1_raw) / max(float(denoms["sketch_gain1"]), 1e-30)
        c2 = (sk_raw + g2_raw) / max(float(denoms["sketch_gain2"]), 1e-30)
        grad_sk_raw = 2.0 * (np.asarray(A_sketch, dtype=np.float64).T @ sk_y)
        grad_g1_raw = 2.0 * (np.asarray(A_cur, dtype=np.float64).T @ cur_y)
        grad_g2_raw = 2.0 * (np.asarray(A_fut, dtype=np.float64).T @ fut_y)
        grad_c1 = (grad_sk_raw + grad_g1_raw) / max(float(denoms["sketch_gain1"]), 1e-30)
        grad_c2 = (grad_sk_raw + grad_g2_raw) / max(float(denoms["sketch_gain2"]), 1e-30)
        del sk_raw_denom

    if policy == "future_hmean_triplet_online":
        hm_val, hm_grad = hmean_many_value_grad([(1.0, sk, grad_sk), (1.0, g1, grad_g1), (1.0, g2, grad_g2)])
    elif policy == "future_hmean_nested_online":
        hm_val, hm_grad = hmean_many_value_grad([(1.0, c1, grad_c1), (1.0, g2, grad_g2)])
    elif policy == "future_hmean_pairwise_online":
        hm_val, hm_grad = hmean_many_value_grad([(1.0, c1, grad_c1), (1.0, c2, grad_c2)])
    elif policy == "future_hmean_weighted_online":
        hm_val, hm_grad = hmean_many_value_grad([
            (float(weights[0]), sk, grad_sk),
            (float(weights[1]), g1, grad_g1),
            (float(weights[2]), g2, grad_g2),
        ])
    else:
        raise ValueError(f"Unknown policy: {policy}")

    if hm_grad is None or not np.isfinite(hm_val):
        return np.nan, np.zeros_like(v), {}
    val = hm_val * H1
    grad = H1 * hm_grad + hm_val * grad_H1
    parts = {"sketch_share": sk, "gain1_share": g1, "gain2_share": g2, "c1_share": c1, "c2_share": c2, "relH1": H1}
    return float(val), np.ascontiguousarray(grad, dtype=np.float64), parts


def make_combination_optimizer(policy, A_cur, A_fut, A_sketch, denoms, weights):
    def value_grad(A_unused_cur, A_unused_fut, v):
        del A_unused_cur, A_unused_fut
        val, grad, parts = combination_value_grad(policy, A_cur, A_fut, A_sketch, denoms, weights, v)
        return val, grad, parts.get("gain1_share", np.nan), parts.get("gain2_share", np.nan), parts.get("relH1", np.nan)

    return value_grad


def optimize_combination_in_basis(policy, A_cur, A_fut, A_sketch, denoms, weights, B, starts, rng, maxit, tol, random_starts):
    original = optimize_future_hmean_in_basis.__globals__["future_hmean_value_grad"]
    optimize_future_hmean_in_basis.__globals__["future_hmean_value_grad"] = make_combination_optimizer(
        policy, A_cur, A_fut, A_sketch, denoms, weights
    )
    try:
        return optimize_future_hmean_in_basis(
            A_cur, A_fut, B, starts, rng, maxit=maxit, tol=tol, random_starts=random_starts
        )
    finally:
        optimize_future_hmean_in_basis.__globals__["future_hmean_value_grad"] = original


def candidate_denoms(candidates, A_cur, A_fut, A_sketch):
    records = hm.score_half_candidates(candidates, A_cur, A_fut, A_sketch_prior=A_sketch)
    finite = [r for r in records.values() if r is not None]
    finite_sketch = [r for r in finite if np.isfinite(r["sketch_gain"])]
    return {
        "sketch": max((r["sketch_gain"] for r in finite_sketch), default=1.0),
        "gain1": max((r["gain1"] for r in finite), default=1.0),
        "gain2": max((r["gain2"] for r in finite), default=1.0),
        "sketch_gain1": max((r["sketch_gain1"] for r in finite), default=1.0),
        "sketch_gain2": max((r["sketch_gain2"] for r in finite), default=1.0),
        "sketch_raw_for_concat": 1.0,
    }, records


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
    prev_opt2 = None
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
            A_sketch = None
            M_gain = A_cur
            V_init = probe.row_norm_seed(A_cur, rank)
            rows_seen = A_cur.shape[0]
        else:
            B_top = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
            A_sketch = B_top
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
        Q_oracle, raw_oracle = raw_oracle_columns(M_gain, V_exact, rank, np.float64)
        candidates = hm.build_candidates(V_default, Q_oracle, raw_oracle, M_gain, A_cur, A_fut, prev_opt2=prev_opt2)
        candidates = {k: candidates.get(k) for k in hm.ONLINE_POOL}
        prior_seen = 0 if state is None else int(state["rows_seen"])
        weights = (prior_seen, A_cur.shape[0], A_fut.shape[0])
        denoms, _ = candidate_denoms(candidates, A_cur, A_fut, A_sketch)

        union = np.vstack([A_cur, A_fut]).astype(np.float64, copy=False)
        B_union = orth_basis_against(rowspace_basis(union), V_default[:, 0])
        starts = [V_default[:, 1]]
        starts.extend([v for v in candidates.values() if v is not None])
        if diag.get("Vbasis_final") is not None:
            Vbasis = np.asarray(diag["Vbasis_final"], dtype=np.float64)
            for j in range(min(Vbasis.shape[1], 8)):
                starts.append(Vbasis[:, j])

        default_combined = combined_score(M_gain, A_cur, V_default[:, 1], A.shape[0], state, old_row_memory)
        for policy in POLICIES:
            default_obj, _, default_parts = combination_value_grad(
                policy, A_cur, A_fut, A_sketch, denoms, weights, V_default[:, 1]
            )
            default_g1 = default_parts.get("gain1_share", np.nan)
            default_g2 = default_parts.get("gain2_share", np.nan)
            default_rel = default_parts.get("relH1", np.nan)
            best = optimize_combination_in_basis(
                policy, A_cur, A_fut, A_sketch, denoms, weights, B_union, starts,
                np.random.default_rng(args.seed + 1009 * block_id + 37 * POLICIES.index(policy)),
                args.union_maxit, args.union_tol, args.union_random_starts,
            )
            if best is None:
                best_v = None
                best_obj = np.nan
                best_g1 = np.nan
                best_g2 = np.nan
                best_rel = np.nan
                best_stop = {}
            else:
                best_v = best["vec"]
                best_obj = best["score"]
                best_g1 = best["gain1"]
                best_g2 = best["gain2"]
                best_rel = best["relH1"]
                best_stop = best["stop"]
            best_combined = np.nan if best_v is None else combined_score(
                M_gain, A_cur, best_v, A.shape[0], state, old_row_memory
            )
            rows.append({
                "matrix": matrix,
                "policy": policy,
                "block": block_id,
                "rows_seen": mid0,
                "union_dim_after_v1": int(B_union.shape[1]),
                "optimizer_subspace_dim_v2": int(np.asarray(diag.get("subspace_dims", [np.nan]))[1])
                if len(np.asarray(diag.get("subspace_dims", []))) > 1 else np.nan,
                "optimizer_grad_perp_v2": float(np.asarray(diag.get("grad_perp_ratio", [np.nan]))[1])
                if len(np.asarray(diag.get("grad_perp_ratio", []))) > 1 else np.nan,
                "default_policy_score": default_obj,
                "best_union_policy_score": best_obj,
                "policy_score_ratio_best_over_default": best_obj / max(default_obj, 1e-300),
                "default_gain1_share": default_g1,
                "default_gain2_share": default_g2,
                "default_relH1": default_rel,
                "best_gain1_share": best_g1,
                "best_gain2_share": best_g2,
                "best_relH1": best_rel,
                "default_combined_score": default_combined,
                "best_union_combined_score": best_combined,
                "combined_score_ratio_best_over_default": best_combined / max(default_combined, 1e-300),
                "default_union_mass": rowspace_mass(union, V_default[:, 1]),
                "best_mgain_rowspace_mass": np.nan if best_v is None else rowspace_mass(M_gain, best_v),
                "best_exact_align2": np.nan if best_v is None else float(abs(np.dot(best_v, V_exact[:, 1])) ** 2),
                "default_exact_align2": float(abs(np.dot(V_default[:, 1], V_exact[:, 1])) ** 2),
                "best_stop_reason": best_stop.get("reason", ""),
                "best_stop_grad_norm": best_stop.get("grad_norm", np.nan),
                "sketch_present": int(A_sketch is not None),
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
        prev_opt2 = np.ascontiguousarray(V_default[:, 1].copy())
    return rows


def summarize(rows):
    out = []
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row["matrix"], row["policy"])].append(row)
    for (matrix, policy), rs in sorted(by_key.items()):
        target = [r for r in rs if r["block"] > 1] or rs
        out.append({
            "matrix": matrix,
            "policy": policy,
            "blocks": len(rs),
            "mean_policy_ratio": float(np.nanmean([r["policy_score_ratio_best_over_default"] for r in target])),
            "median_policy_ratio": float(np.nanmedian([r["policy_score_ratio_best_over_default"] for r in target])),
            "mean_combined_ratio": float(np.nanmean([r["combined_score_ratio_best_over_default"] for r in target])),
            "median_combined_ratio": float(np.nanmedian([r["combined_score_ratio_best_over_default"] for r in target])),
            "mean_default_union_mass": float(np.nanmean([r["default_union_mass"] for r in target])),
            "mean_best_mgain_mass": float(np.nanmean([r["best_mgain_rowspace_mass"] for r in target])),
            "mean_optimizer_grad_perp_v2": float(np.nanmean([r["optimizer_grad_perp_v2"] for r in target])),
            "mean_best_exact_align2": float(np.nanmean([r["best_exact_align2"] for r in target])),
            "mean_default_exact_align2": float(np.nanmean([r["default_exact_align2"] for r in target])),
        })
    return out


def write_text(path, summaries):
    fields = [
        "matrix",
        "policy",
        "blocks",
        "mean_policy_ratio",
        "median_policy_ratio",
        "mean_combined_ratio",
        "median_combined_ratio",
        "mean_default_union_mass",
        "mean_best_mgain_mass",
        "mean_optimizer_grad_perp_v2",
        "mean_best_exact_align2",
        "mean_default_exact_align2",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("HM-combination optimizer diagnostic\n")
        f.write("===================================\n\n")
        f.write("For each block and HM-combination policy, optimize the policy objective inside rowspan([A_w; A_{w+1}]) after orthogonalizing against optimizer v1.\n")
        f.write("Policy normalizers are fixed from the online candidate pool, matching the hmean-combination implementation notes.\n\n")
        f.write("Summary by matrix and policy\n")
        f.write("----------------------------\n")
        f.write(" ".join(f"{x:<34}" for x in fields) + "\n")
        for s in summaries:
            vals = []
            for field in fields:
                v = s[field]
                vals.append(f"{v:<34.4f}" if isinstance(v, float) else f"{str(v):<34}")
            f.write(" ".join(vals) + "\n")
        f.write("\nInterpretation\n")
        f.write("--------------\n")
        f.write("policy_ratio > 1 means the HM-combination objective has better in-union points than optimizer v2.\n")
        f.write("combined_ratio < 1 means those points still lose under the optimizer's combined forgetting objective.\n")


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
    parser.add_argument("--out-prefix", default="summary/hmean_combinations_optimizer_diagnostic")
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
        print(f"done {matrix} rows={len(mat_rows)}")
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
