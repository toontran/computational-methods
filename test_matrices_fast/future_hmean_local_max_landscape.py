"""Probe the multi-local-max landscape of combined-score around v_carry.

For each block:
  - Compute score and gradient at v_carry (optimizer's output) and at
    proj(v_carry, rowspan([A_w; A_{w+1}])) — the projection onto the window.
  - Also score v_inwindow (in-window future-HM peak — NOT a projection of
    v_carry; a separately found optimum) and v_inwindow_proj (that vector
    projected onto M_gain rowspace).
  - Run many random-restart full-sphere ascent and collect ALL terminal
    points with score within 1% of the best. Report:
      * spread of terminal scores
      * pairwise inner products with v_carry and with each other
      * top-k clusters
"""
import argparse
import csv
import json

import numpy as np

import cex_restricted_space_probe as probe
from second_slot_tail_bias_diagnostic import make_state
from future_hmean_optimizer_diagnostic import (
    optimize_future_hmean_in_basis,
    rowspace_basis,
    orth_basis_against,
)
from future_hmean_local_optimum_probe import (
    project_to_v1_perp,
    combined_score_and_grad_full,
    tangent_grad_norm,
    gradient_ascent_full,
)


def run_one(args):
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=args.matrix, n=args.n, preset=args.preset, seed=args.seed,
        r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
        shuffle_rows=True, row_shuffle_seed=args.row_shuffle_seed,
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
            M_gain = A_cur
            V_init = probe.row_norm_seed(A_cur, rank)
            rows_seen = A_cur.shape[0]
            sketch_block = None
        else:
            B_top = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
            sketch_block = B_top
            M_gain = np.vstack([B_top, A_cur]).astype(work_dtype, copy=False)
            V_init = probe.row_norm_seed(A_cur, rank)
            rows_seen = state["rows_seen"] + A_cur.shape[0]

        V_score, _, _, _, diag = probe.entropy_iter_basis_forget(
            M_gain=M_gain, active_r=rank, rows_ref=A.shape[0],
            V_init=np.asarray(V_init, dtype=work_dtype),
            q0=args.q0, qmax=args.qmax, krylov_depth=args.krylov_depth,
            residual_tol=args.residual_tol, expansion_maxit=args.expansion_maxit,
            num_restarts=args.num_restarts, maxit=args.maxit, tol=args.tol,
            rng=np.random.default_rng(args.seed), verbose=False,
            state_prev=state, A_block=A_cur, rows_total=rows_seen,
            reduced_optimizer="cex", basis_selection="greedy", work_dtype=work_dtype,
            expansion_direction="residual", reuse_line_search_grad=True,
            expansion_warm_start=True,
            post_expansion_maxit=args.post_expansion_maxit,
            score_variant="combined", old_row_memory=old_row_memory,
            combined_rank=None, patience=args.patience,
            patience_rel_tol=args.patience_rel_tol,
        )
        V_default = np.ascontiguousarray(np.asarray(V_score[:, :rank], dtype=np.float64))
        v1 = V_default[:, 0]
        v2 = V_default[:, 1] if V_default.shape[1] >= 2 else None

        if state is not None and v2 is not None:
            # In-window future-HM optimum
            union = np.vstack([A_cur, A_fut]).astype(np.float64, copy=False)
            Q_window = rowspace_basis(union)
            B_union = orth_basis_against(Q_window, v1)
            starts = [V_default[:, 1]]
            if V_exact.shape[1] >= 2:
                starts.append(V_exact[:, 1])
            if diag.get("Vbasis_final") is not None:
                Vbasis = np.asarray(diag["Vbasis_final"], dtype=np.float64)
                for j in range(min(Vbasis.shape[1], 8)):
                    starts.append(Vbasis[:, j])
            best = optimize_future_hmean_in_basis(
                A_cur, A_fut, B_union, starts,
                np.random.default_rng(args.seed + 1009 * block_id),
                maxit=80, tol=1e-9, random_starts=12,
            )
            v_inwindow = None if best is None else best["vec"]

            # v_carry projected onto window space (rowspan([A_w; A_{w+1}]))
            v_carry_proj_to_window = Q_window @ (Q_window.T @ v2)
            nrm = float(np.linalg.norm(v_carry_proj_to_window))
            v_carry_proj_to_window = v_carry_proj_to_window / nrm if nrm > 1e-30 else None
            mass_v_carry_in_window = float(nrm ** 2)

            # v_inwindow projected onto M_gain rowspace
            Q_mgain = rowspace_basis(np.asarray(M_gain, dtype=np.float64))
            v_inwindow_proj = None
            if v_inwindow is not None:
                vp = Q_mgain @ (Q_mgain.T @ v_inwindow)
                npv = float(np.linalg.norm(vp))
                if npv > 1e-30:
                    v_inwindow_proj = vp / npv

            rec = {"matrix": args.matrix, "block": block_id, "rows_seen": mid0,
                   "mass_v_carry_in_window_before_renorm": mass_v_carry_in_window}

            def _eval(name, v):
                if v is None:
                    rec[f"{name}_score"] = np.nan
                    rec[f"{name}_grad_norm"] = np.nan
                    return
                sc, gr, _, _, _ = combined_score_and_grad_full(
                    M_gain, A_cur, v, A.shape[0], state, old_row_memory)
                gn, _ = tangent_grad_norm(gr, v, v1)
                rec[f"{name}_score"] = sc
                rec[f"{name}_grad_norm"] = gn

            _eval("v_carry", v2)
            _eval("v_carry_proj_window", v_carry_proj_to_window)
            _eval("v_inwindow", v_inwindow)
            _eval("v_inwindow_proj", v_inwindow_proj)

            # Many random restarts
            n = A.shape[1]
            rng = np.random.default_rng(args.seed + 10007 * block_id)
            terminals = []  # list of (score, v_unit)
            for _ in range(args.random_restarts):
                v0 = rng.standard_normal(n)
                v0 /= max(float(np.linalg.norm(v0)), 1e-30)
                v_end, sc_end, gn_end = gradient_ascent_full(
                    v0, v1, M_gain, A_cur, A.shape[0], state, old_row_memory,
                    maxit=400, tol=1e-10)
                if v_end is not None and np.isfinite(sc_end):
                    terminals.append((float(sc_end), float(gn_end), np.asarray(v_end, dtype=np.float64)))

            # Also include warm starts from v_carry, v_inwindow, v_inwindow_proj, v_carry_proj_window
            for nm, vstart in [("v_carry", v2), ("v_inwindow", v_inwindow),
                                ("v_inwindow_proj", v_inwindow_proj),
                                ("v_carry_proj_window", v_carry_proj_to_window)]:
                if vstart is None:
                    continue
                v_end, sc_end, gn_end = gradient_ascent_full(
                    vstart, v1, M_gain, A_cur, A.shape[0], state, old_row_memory,
                    maxit=400, tol=1e-10)
                if v_end is not None and np.isfinite(sc_end):
                    terminals.append((float(sc_end), float(gn_end), np.asarray(v_end, dtype=np.float64)))

            terminals.sort(key=lambda x: -x[0])
            if terminals:
                top_score = terminals[0][0]
                rec["best_terminal_score"] = top_score
                rec["best_terminal_grad_norm"] = terminals[0][1]
                rec["best_terminal_align_v_carry"] = float(abs(np.dot(terminals[0][2], v2)) ** 2)
                # Cluster: count terminals within 1% of best
                near_best = [t for t in terminals if t[0] >= top_score * 0.99]
                rec["count_within_1pct_of_best"] = len(near_best)
                rec["count_total_terminals"] = len(terminals)
                # Score percentiles
                scs = sorted([t[0] for t in terminals], reverse=True)
                rec["score_p100"] = scs[0]
                rec["score_p90"] = scs[max(0, int(len(scs) * 0.10))]
                rec["score_p50"] = scs[len(scs) // 2]
                rec["score_p10"] = scs[max(0, int(len(scs) * 0.90))]
                # Pairwise alignments among top-5 distinct terminals (greedy dedupe by alignment > 0.99)
                kept = []
                for sc, gn, vv in terminals:
                    if any(abs(np.dot(vv, k_v)) > 0.99 for _, _, k_v in kept):
                        continue
                    kept.append((sc, gn, vv))
                    if len(kept) >= 5:
                        break
                rec["distinct_top5_count"] = len(kept)
                rec["distinct_top5_scores"] = [k[0] for k in kept]
                rec["distinct_top5_align_v_carry"] = [
                    float(abs(np.dot(k[2], v2)) ** 2) for k in kept
                ]
                # Pairwise alignments among kept top-5
                pa = []
                for i in range(len(kept)):
                    for j in range(i + 1, len(kept)):
                        pa.append(float(abs(np.dot(kept[i][2], kept[j][2])) ** 2))
                rec["distinct_top5_pairwise_align"] = pa
            rows.append(rec)

        # Update state
        score_selected = np.zeros(rank, dtype=float)
        H_selected = np.zeros(rank, dtype=float)
        for j in range(rank):
            score_selected[j], _, H_selected[j] = probe.score_full_vector_details_forget(
                M_gain, A_cur, V_default[:, j], A.shape[0], state_prev=state,
                score_variant="combined", old_row_memory=old_row_memory,
            )
        state, V_r, _ = make_state(M_gain, V_default, H_selected, score_selected, rows_seen)
        old_row_memory, _ = probe.select_old_row_memory(
            np.asarray(A[:mid0, :], dtype=work_dtype),
            V_r.astype(work_dtype, copy=False),
            args.old_memory_size if args.old_memory_size > 0 else half_win,
            np.random.default_rng(args.seed + end0), return_indices=True,
        )
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="mixed-tail-sharp")
    p.add_argument("--out-prefix", default="summary/future_hmean_local_max_landscape")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--half-win", type=int, default=32)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--preset", default="fast")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--row-shuffle-seed", type=int, default=0)
    p.add_argument("--old-memory-size", type=int, default=32)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--q0", type=int, default=8)
    p.add_argument("--qmax", type=int, default=48)
    p.add_argument("--krylov-depth", type=int, default=2)
    p.add_argument("--residual-tol", type=float, default=0.01)
    p.add_argument("--expansion-maxit", type=int, default=8)
    p.add_argument("--num-restarts", type=int, default=3)
    p.add_argument("--maxit", type=int, default=120)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--post-expansion-maxit", type=int, default=80)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--patience-rel-tol", type=float, default=1e-5)
    p.add_argument("--random-restarts", type=int, default=24)
    p.add_argument("--max-pairs", type=int, default=8)
    p.add_argument("--r-sig", type=int, default=2)
    p.add_argument("--alpha-sig", type=float, default=0.003)
    p.add_argument("--alpha-tail", type=float, default=0.0145)
    p.add_argument("--tail-scale", type=float, default=0.99)
    p.add_argument("--sigma1", type=float, default=0.991)
    p.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    return p.parse_args()


def main():
    args = parse_args()
    rows = run_one(args)
    if not rows:
        print("no rows")
        return
    json_path = args.out_prefix + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True, default=float)
    flat = []
    for r in rows:
        rcopy = {k: v for k, v in r.items() if not isinstance(v, list)}
        for k in ("distinct_top5_scores", "distinct_top5_align_v_carry", "distinct_top5_pairwise_align"):
            v = r.get(k, None)
            rcopy[k] = json.dumps(v) if v is not None else ""
        flat.append(rcopy)
    csv_path = args.out_prefix + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    print(f"wrote {csv_path} {json_path}")


if __name__ == "__main__":
    main()
