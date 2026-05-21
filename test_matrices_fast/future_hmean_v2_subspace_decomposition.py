"""Decompose the streaming optimizer's v2 into orthogonal subspaces.

For each block (after the first), we partition R^n into four orthogonal pieces:
  S1 = state V         (carried 2-dim subspace from the prior block)
  S2 = sketch tail     (M_gain rowspace minus S1)
  S3 = A_w residual    (rowspan(A_w) minus S1+S2)  -- typically ~empty since A_w is in M_gain
  S4 = A_{w+1} residual (rowspan(A_{w+1}) minus S1+S2+S3)

We project v2 (streaming) and v_best (in-window future-HM optimum) onto each
piece and report the squared mass. We also compute alignment of v2 with the
second right-singular vector of the sketch and with V_exact[:,1] (global
oracle).

The point is to verify: the streaming v2 is mostly the second sketch
direction (S1 + nearby S2), which is why future-HM scores it as ~zero
on A_{w+1} but it inherits long-horizon structure from prior blocks.
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


def _onb(M, tol=1e-10):
    M = np.asarray(M, dtype=np.float64)
    if M.size == 0 or M.shape[1] == 0:
        return np.zeros((M.shape[0] if M.ndim == 2 else 0, 0), dtype=np.float64)
    Q, R = np.linalg.qr(M)
    diag = np.abs(np.diag(R))
    if diag.size == 0:
        return np.zeros((M.shape[0], 0), dtype=np.float64)
    keep = diag > max(float(diag.max()) * tol, 1e-30)
    return np.ascontiguousarray(Q[:, keep], dtype=np.float64)


def deflate(B_target, *Q_remove, tol=1e-10):
    M = B_target
    for Q in Q_remove:
        if Q is None or Q.size == 0 or Q.shape[1] == 0:
            continue
        M = M - Q @ (Q.T @ M)
    return _onb(M, tol)


def mass(v, Q):
    if Q is None or Q.size == 0 or Q.shape[1] == 0:
        return 0.0
    return float(np.linalg.norm(Q.T @ v) ** 2)


def run_matrix(args, matrix):
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=matrix, n=args.n, preset=args.preset, seed=args.seed,
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
        if v2 is None or block_id == 1:
            # Update state and continue without recording (no sketch yet for block 1)
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
            continue

        # Build orthogonal subspaces
        Q_state = _onb(state["V"])  # carried 2D
        Q_sketch_full = _onb(sketch_block.T) if sketch_block is not None else _onb(np.empty((A.shape[1], 0)))
        # Actually sketch is rows x n; rowspace = Q_sketch_full from Vh side
        # Use rowspace_basis utility for clarity
        if sketch_block is not None:
            Q_sketch = rowspace_basis(np.asarray(sketch_block, dtype=np.float64))
        else:
            Q_sketch = np.zeros((A.shape[1], 0))
        Q_sketch_tail = deflate(Q_sketch, Q_state)
        Q_Aw = rowspace_basis(np.asarray(A_cur, dtype=np.float64))
        Q_Aw_resid = deflate(Q_Aw, Q_state, Q_sketch_tail)
        Q_Afut = rowspace_basis(np.asarray(A_fut, dtype=np.float64))
        Q_Afut_resid = deflate(Q_Afut, Q_state, Q_sketch_tail, Q_Aw_resid)

        # Search for in-window future-HM optimum (orthogonal to v1)
        union = np.vstack([A_cur, A_fut]).astype(np.float64, copy=False)
        B_union = orth_basis_against(rowspace_basis(union), v1)
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
            maxit=60, tol=1e-8, random_starts=8,
        )
        v_best = best["vec"] if best is not None else None

        # Sketch's second right-singular direction (deflate v1 first)
        if sketch_block is not None and sketch_block.shape[0] >= 2:
            S = np.asarray(sketch_block, dtype=np.float64)
            v1n = v1 / max(float(np.linalg.norm(v1)), 1e-30)
            S_def = S - (S @ v1n)[:, None] * v1n[None, :]
            _, _, Vh_s = np.linalg.svd(S_def, full_matrices=False)
            sketch_v2 = Vh_s[0]  # leading singular dir AFTER deflation against v1
        else:
            sketch_v2 = None

        rec = {
            "matrix": matrix,
            "block": block_id,
            "rows_seen": mid0,
            # v2 mass decomposition
            "v2_mass_state": mass(v2, Q_state),
            "v2_mass_sketch_tail": mass(v2, Q_sketch_tail),
            "v2_mass_Aw_resid": mass(v2, Q_Aw_resid),
            "v2_mass_Afut_resid": mass(v2, Q_Afut_resid),
            # in-window optimum mass decomposition
            "best_mass_state": mass(v_best, Q_state) if v_best is not None else np.nan,
            "best_mass_sketch_tail": mass(v_best, Q_sketch_tail) if v_best is not None else np.nan,
            "best_mass_Aw_resid": mass(v_best, Q_Aw_resid) if v_best is not None else np.nan,
            "best_mass_Afut_resid": mass(v_best, Q_Afut_resid) if v_best is not None else np.nan,
            # alignment with sketch's second singular direction
            "v2_align_sketch_v2": float(abs(np.dot(v2, sketch_v2)) ** 2) if sketch_v2 is not None else np.nan,
            "best_align_sketch_v2": float(abs(np.dot(v_best, sketch_v2)) ** 2) if (v_best is not None and sketch_v2 is not None) else np.nan,
            # alignment with global oracle V_exact[:,1]
            "v2_align_exact": float(abs(np.dot(v2, V_exact[:, 1])) ** 2) if V_exact.shape[1] > 1 else np.nan,
            "best_align_exact": float(abs(np.dot(v_best, V_exact[:, 1])) ** 2) if (v_best is not None and V_exact.shape[1] > 1) else np.nan,
            # sketch's own alignment with V_exact[:,1] (does the carried direction track the truth?)
            "sketch_v2_align_exact": float(abs(np.dot(sketch_v2, V_exact[:, 1])) ** 2) if (sketch_v2 is not None and V_exact.shape[1] > 1) else np.nan,
        }
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
    p.add_argument("--matrices", nargs="+", default=[
        "mixed-tail-sharp", "mixed-tail-balanced", "mixed-tail-soft",
        "diffuse-diffuse", "static-cex", "etf-basket-basis",
        "residual-spiky-shocks", "risk-residual-panel",
    ])
    p.add_argument("--out-prefix", default="summary/future_hmean_v2_subspace_decomposition")
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
    p.add_argument("--max-pairs", type=int, default=16)
    p.add_argument("--r-sig", type=int, default=2)
    p.add_argument("--alpha-sig", type=float, default=0.003)
    p.add_argument("--alpha-tail", type=float, default=0.0145)
    p.add_argument("--tail-scale", type=float, default=0.99)
    p.add_argument("--sigma1", type=float, default=0.991)
    p.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    for matrix in args.matrices:
        mat_rows = run_matrix(args, matrix)
        rows.extend(mat_rows)
        print(f"done {matrix} blocks={len(mat_rows)}")
    if not rows:
        return
    csv_path = args.out_prefix + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json_path = args.out_prefix + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2, sort_keys=True)
    print(f"wrote {csv_path} {json_path}")


if __name__ == "__main__":
    main()
