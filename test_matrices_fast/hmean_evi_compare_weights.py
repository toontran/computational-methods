"""Compare HM-evi score under three weight schemes on identical candidates.

For each block, we:
  1. Stream forward (combined-greedy carry).
  2. Build fixed candidate set: combined slot-2, oracle_v1/v2 projections,
     HM-triplet (norm/raw) optima, and HM-evi optima for each weight scheme.
  3. Score every candidate under fixed / c / c-on-u weights.
"""

import argparse
import json
import numpy as np

import cex_restricted_space_probe as probe
import half_window_sliding_hmean_experiment as hm
from future_hmean_optimizer_diagnostic import (
    combined_score,
    orth_basis_against,
    rowspace_basis,
)
from hmean_combinations_optimizer_diagnostic import (
    candidate_denoms,
    optimize_combination_in_basis,
)
from second_slot_tail_bias_diagnostic import raw_oracle_columns
from hmean_evidence_score import (
    hm_evi_value_grad,
    optimize_hm_evi_in_basis,
    per_block_constants,
    stream_to_block,
)


def weight_triple(scheme, rank, half_win, c_sk, c_g1, c_g2):
    if scheme == "c":
        return float(c_sk * c_sk), float(c_g1 * c_g1), float(c_g2 * c_g2)
    if scheme == "c-on-u":
        return float(c_sk), float(c_g1), float(c_g2)
    return float(rank), float(half_win), float(half_win)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="mixed-tail-sharp")
    p.add_argument("--blocks", nargs="+", type=int, default=[2, 6, 12])
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--half-win", type=int, default=32)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--preset", default="fast")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-rows", action="store_true", default=True)
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
    p.add_argument("--union-maxit", type=int, default=120)
    p.add_argument("--union-tol", type=float, default=1e-9)
    p.add_argument("--union-random-starts", type=int, default=24)
    p.add_argument("--r-sig", type=int, default=2)
    p.add_argument("--alpha-sig", type=float, default=0.003)
    p.add_argument("--alpha-tail", type=float, default=0.0145)
    p.add_argument("--tail-scale", type=float, default=0.99)
    p.add_argument("--sigma1", type=float, default=0.991)
    p.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    p.add_argument("--out", default="summary/hmean_evi_compare_weights.txt")
    return p.parse_args()


def main():
    args = parse_args()
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    rank = int(args.rank)
    half_win = int(args.half_win)

    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=args.matrix, n=args.n, preset=args.preset, seed=args.seed,
        r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
        shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, np.float64)
    V_exact = np.asarray(V_exact, np.float64)

    blocks = sorted(set(args.blocks))
    snapshots = stream_to_block(args, A, V_exact, work_dtype, rank, max(blocks), set(blocks))

    schemes = ("fixed", "c", "c-on-u")
    out_lines = []

    for block_id in blocks:
        snap = snapshots[block_id]
        A_cur = snap["A_cur"]
        A_fut = snap["A_fut"]
        A_sketch = snap["A_sketch"]
        M_gain = snap["M_gain"]
        state = snap["state"]
        old_row_memory = snap["old_row_memory"]
        V_default = snap["V_default"]
        diag = snap["diag"]

        consts = per_block_constants(A, block_id, half_win)
        c_sk, c_g1, c_g2 = consts["c_sk"], consts["c_g1"], consts["c_g2"]
        A_sketch_for_evi = A_sketch if A_sketch.size else None

        # Subspace bases.
        union_stack = np.vstack([A_sketch, A_cur, A_fut]) if A_sketch.size else np.vstack([A_cur, A_fut])
        B_union = rowspace_basis(union_stack)

        def project_unit(vec, B):
            if vec is None or B is None or B.size == 0:
                return None
            p = B @ (B.T @ vec)
            nv = float(np.linalg.norm(p))
            return None if nv <= 1e-30 else p / nv

        oracle_v1 = V_exact[:, 0] / max(np.linalg.norm(V_exact[:, 0]), 1e-30)
        oracle_v2 = V_exact[:, 1] / max(np.linalg.norm(V_exact[:, 1]), 1e-30)
        oracle_v1_proj = project_unit(oracle_v1, B_union)
        oracle_v2_proj = project_unit(oracle_v2, B_union)

        # HM-triplet pool optimizers.
        Q_oracle, raw_oracle = raw_oracle_columns(M_gain, V_exact, rank, np.float64)
        pool = hm.build_candidates(V_default, Q_oracle, raw_oracle, M_gain, A_cur, A_fut)
        pool = {k: pool.get(k) for k in hm.ONLINE_POOL}
        weights_existing = (state["rows_seen"] if state is not None else 0,
                            A_cur.shape[0], A_fut.shape[0])
        denoms, _ = candidate_denoms(pool, A_cur, A_fut, A_sketch if A_sketch.size else None)
        union_for_search = (np.vstack([A_sketch, A_cur, A_fut]) if A_sketch.size
                            else np.vstack([A_cur, A_fut])).astype(np.float64, copy=False)
        B_search = orth_basis_against(rowspace_basis(union_for_search), V_default[:, 0])

        starts = [V_default[:, 1]]
        starts.extend([v for v in pool.values() if v is not None])
        Vbasis = diag.get("Vbasis_final")
        if Vbasis is not None:
            Vb = np.asarray(Vbasis, dtype=np.float64)
            for j in range(min(Vb.shape[1], 8)):
                starts.append(Vb[:, j])
        starts_with_oracle = list(starts) + (
            [oracle_v1_proj, oracle_v2_proj] if oracle_v1_proj is not None else [])

        triplet_norm = optimize_combination_in_basis(
            "future_hmean_triplet_online",
            A_cur, A_fut, A_sketch if A_sketch.size else None, denoms, weights_existing,
            B_search, starts,
            np.random.default_rng(args.seed + 7777 + block_id),
            args.union_maxit, args.union_tol, args.union_random_starts,
        )
        denoms_raw = {k: 1.0 for k in ("sketch", "gain1", "gain2",
                                        "sketch_gain1", "sketch_gain2",
                                        "sketch_raw_for_concat")}
        triplet_raw = optimize_combination_in_basis(
            "future_hmean_triplet_online",
            A_cur, A_fut, A_sketch if A_sketch.size else None, denoms_raw, weights_existing,
            B_search, starts_with_oracle,
            np.random.default_rng(args.seed + 9001 + block_id),
            args.union_maxit, args.union_tol, args.union_random_starts,
        )

        candidates = {
            "combined_optimizer_v1": V_default[:, 0],
            "combined_optimizer_v2": V_default[:, 1],
            "hm_triplet_raw_best": None if triplet_raw is None else triplet_raw["vec"],
            "oracle_v1_proj": oracle_v1_proj,
            "oracle_v2_proj": oracle_v2_proj,
        }

        n_dim = A.shape[1]
        # E[raw_k] for v ~ uniform on unit sphere = ||A_k||_F^2 / n
        E_raw_sk = consts["sk_F2"] / n_dim if consts["N_sk"] > 0 else 0.0
        E_raw_g1 = consts["cur_F2"] / n_dim
        E_raw_g2 = consts["fut_F2"] / n_dim
        # Per-block constants the user's score depends on:
        sum_c = (c_sk if consts["N_sk"] > 0 else 0.0) + c_g1 + c_g2
        sum_c2 = (c_sk * c_sk if consts["N_sk"] > 0 else 0.0) + c_g1 * c_g1 + c_g2 * c_g2

        out_lines.append(f"== block {block_id}  matrix={args.matrix}  N_sk={consts['N_sk']} ==")
        out_lines.append(
            f"  c_sk={c_sk:.4e}  c_g1={c_g1:.4e}  c_g2={c_g2:.4e}   "
            f"sk_F2={consts['sk_F2']:.3e}  cur_F2={consts['cur_F2']:.3e}  fut_F2={consts['fut_F2']:.3e}"
        )
        out_lines.append(
            f"  E[raw_sk]={E_raw_sk:.4e}  E[raw_g1]={E_raw_g1:.4e}  E[raw_g2]={E_raw_g2:.4e}   "
            f"Σc={sum_c:.4e}  Σc²={sum_c2:.4e}"
        )
        header = (
            f"  {'label':<24} "
            f"{'raw_sk':>9} {'raw_g1':>9} {'raw_g2':>9}  "
            f"{'u_sk':>6} {'u_g1':>6} {'u_g2':>6}  "
            f"{'1/u_sk':>8} {'1/r_g1':>8} {'1/r_g2':>8}  "
            f"{'D_skonly':>9} {'sc_skonly':>10}  "
            f"{'sc_fixed':>10} {'sc_c':>10}  "
            f"{'a_v1':>5} {'a_v2':>5}"
        )
        out_lines.append(header)
        for label, v in candidates.items():
            if v is None:
                out_lines.append(f"  {label:<24}  (n/a)")
                continue
            v = np.asarray(v, dtype=np.float64).reshape(-1)
            nv = float(np.linalg.norm(v))
            if nv <= 1e-30:
                continue
            v = v / nv

            # Compute raws directly.
            if A_sketch_for_evi is not None:
                y_sk = A_sketch_for_evi @ v
                raw_sk = float(np.dot(y_sk, y_sk))
            else:
                raw_sk = 0.0
            y_c = A_cur @ v
            raw_g1 = float(np.dot(y_c, y_c))
            y_f = A_fut @ v
            raw_g2 = float(np.dot(y_f, y_f))

            # User's exact raw-domain HM with weights = c:
            #   HM_c_raw = (Σc) / Σ (c_k / raw_k)
            cr_sk = (c_sk / raw_sk) if (raw_sk > 1e-30 and consts["N_sk"] > 0) else 0.0
            cr_g1 = (c_g1 / raw_g1) if raw_g1 > 1e-30 else float("inf")
            cr_g2 = (c_g2 / raw_g2) if raw_g2 > 1e-30 else float("inf")
            D_c = cr_sk + cr_g1 + cr_g2
            hm_c_raw = sum_c / D_c if D_c > 0 else 0.0

            # User's "penalize only the sketch" score:
            #   sc_skonly = 1 / (1/u_sk + 1/raw_g1 + 1/raw_g2)
            u_sk_local = c_sk * raw_sk if consts["N_sk"] > 0 else 0.0
            inv_u_sk = (1.0 / max(u_sk_local, 1e-30)) if consts["N_sk"] > 0 else 0.0
            inv_raw_g1 = (1.0 / raw_g1) if raw_g1 > 1e-30 else float("inf")
            inv_raw_g2 = (1.0 / raw_g2) if raw_g2 > 1e-30 else float("inf")
            D_skonly = inv_u_sk + inv_raw_g1 + inv_raw_g2
            sc_skonly = 1.0 / D_skonly if D_skonly > 0 and np.isfinite(D_skonly) else 0.0

            scores = {}
            u_sk = u_g1 = u_g2 = None
            for sc in schemes:
                w_sk, w_g1, w_g2 = weight_triple(sc, rank, half_win, c_sk, c_g1, c_g2)
                s, _, u_sk, u_g1, u_g2, _, _ = hm_evi_value_grad(
                    A_sketch_for_evi, A_cur, A_fut,
                    c_sk, c_g1, c_g2, w_sk, w_g1, w_g2, v,
                )
                scores[sc] = s
            comb = combined_score(M_gain, A_cur, v, A.shape[0], state, old_row_memory)
            align_v1 = float(np.dot(v, oracle_v1) ** 2)
            align_v2 = float(np.dot(v, oracle_v2) ** 2)
            out_lines.append(
                f"  {label:<24} "
                f"{raw_sk:>9.4f} {raw_g1:>9.4f} {raw_g2:>9.4f}  "
                f"{u_sk:>6.3f} {u_g1:>6.3f} {u_g2:>6.3f}  "
                f"{inv_u_sk:>8.3f} {inv_raw_g1:>8.3f} {inv_raw_g2:>8.3f}  "
                f"{D_skonly:>9.3f} {sc_skonly:>10.4e}  "
                f"{scores['fixed']:>10.4e} {scores['c']:>10.4e}  "
                f"{align_v1:>5.3f} {align_v2:>5.3f}"
            )
        out_lines.append("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
