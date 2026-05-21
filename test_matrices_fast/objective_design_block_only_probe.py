import argparse
import time

import numpy as np

import cex_restricted_space_probe as probe
from second_slot_tail_bias_diagnostic import (
    fmt,
    make_state,
    orth_against,
    raw_oracle_columns,
    svd_complement,
)


def block_only_component(v, A_block, rows_ref, state_prev, old_row_memory):
    """Combined score with current-block gain only and pooled entropy rows."""
    if v is None:
        return None
    comp = probe.combined_score_component_details(
        A_block,
        A_block,
        v,
        rows_ref,
        state_prev=state_prev,
        old_row_memory=old_row_memory,
    )
    return {
        "score": float(comp["score_total"]),
        "gain2": float(comp["gain2"]),
        "phi": float(comp["phi"]),
        "relH": float(comp["pooled_rel_H"]),
        "new_y2": float(comp["new_y2_sq"]),
        "old_y2": float(comp["old_y2_sq"]) if np.isfinite(comp["old_y2_sq"]) else np.nan,
        "rows_entropy": int(comp["rows_entropy"]),
    }


def full_component(v, M_gain, A_block, rows_ref, state_prev, old_row_memory):
    if v is None:
        return None
    comp = probe.combined_score_component_details(
        M_gain,
        A_block,
        v,
        rows_ref,
        state_prev=state_prev,
        old_row_memory=old_row_memory,
    )
    return {
        "score": float(comp["score_total"]),
        "gain2": float(comp["gain2"]),
        "phi": float(comp["phi"]),
        "relH": float(comp["pooled_rel_H"]),
    }


def best_label(records, key):
    vals = [(label, rec[key]) for label, rec in records.items() if rec is not None]
    if not vals:
        return "", np.nan
    return max(vals, key=lambda item: item[1])


def ratio(num, den):
    return float(num / max(den, 1e-30))


def run_matrix(matrix, args):
    np.random.seed(args.seed)
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
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    n = A.shape[0]
    rank = int(args.rank)
    state = None
    old_row_memory = None
    old_row_memory_idx = None
    V_r = None
    prev_opt2 = None
    rows = []
    t0 = time.time()

    print(f"block_only_matrix_start matrix={matrix} n={n} win={args.win} rank={rank}")
    for block_idx, start0 in enumerate(range(0, n, args.win), start=1):
        end0 = min(start0 + args.win, n)
        A_block = np.asarray(A[start0:end0, :], dtype=work_dtype)
        if state is None:
            M_gain = A_block
            V_init = probe.row_norm_seed(A_block, rank)
            rows_seen = A_block.shape[0]
        else:
            B_top = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
            M_gain = np.vstack([B_top, A_block]).astype(work_dtype, copy=False)
            V_init = probe.row_norm_seed(A_block, rank)
            rows_seen = state["rows_seen"] + A_block.shape[0]

        V_score, s_score, H_score, score_score, _ = probe.entropy_iter_basis_forget(
            M_gain=M_gain,
            active_r=rank,
            rows_ref=n,
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
            A_block=A_block,
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
        )

        Q_oracle, raw_oracle = raw_oracle_columns(M_gain, V_exact, rank, np.float64)
        V_selected = np.ascontiguousarray(np.asarray(V_score[:, :rank], dtype=np.float64))
        v1 = V_selected[:, :1]
        candidates = {
            "opt2": V_selected[:, 1],
            "q2_vs_v1opt": orth_against(raw_oracle[1] if len(raw_oracle) > 1 else Q_oracle[:, 1], v1),
            "q2_vs_q1oracle": orth_against(raw_oracle[1] if len(raw_oracle) > 1 else Q_oracle[:, 1], Q_oracle[:, :1]),
            "prev_opt2": prev_opt2,
            "svd_complement": svd_complement(M_gain, v1),
        }
        block_records = {
            label: block_only_component(v, A_block, n, state, old_row_memory)
            for label, v in candidates.items()
        }
        full_records = {
            label: full_component(v, M_gain, A_block, n, state, old_row_memory)
            for label, v in candidates.items()
        }
        best_score_label, best_score = best_label(block_records, "score")
        best_gain_label, best_gain = best_label(block_records, "gain2")
        q2 = block_records.get("q2_vs_q1oracle")
        opt2 = block_records.get("opt2")
        full_q2 = full_records.get("q2_vs_q1oracle")
        full_opt2 = full_records.get("opt2")
        cos = probe.subspace_principal_cosines(V_selected, Q_oracle)
        q2_over_opt_score = np.nan if q2 is None or opt2 is None else ratio(q2["score"], opt2["score"])
        q2_over_opt_gain = np.nan if q2 is None or opt2 is None else ratio(q2["gain2"], opt2["gain2"])
        full_q2_over_opt_score = (
            np.nan if full_q2 is None or full_opt2 is None else ratio(full_q2["score"], full_opt2["score"])
        )
        rows.append(
            {
                "block": block_idx,
                "best_score": best_score_label,
                "best_gain": best_gain_label,
                "q2_score_win": bool(q2 is not None and opt2 is not None and q2["score"] > opt2["score"]),
                "q2_gain_win": bool(q2 is not None and opt2 is not None and q2["gain2"] > opt2["gain2"]),
                "q2_over_opt_score": q2_over_opt_score,
                "q2_over_opt_gain": q2_over_opt_gain,
                "full_q2_over_opt_score": full_q2_over_opt_score,
                "cos2": float(cos[1]) if len(cos) > 1 else np.nan,
                "opt2_phi": np.nan if opt2 is None else opt2["phi"],
                "q2_phi": np.nan if q2 is None else q2["phi"],
                "opt2_relH": np.nan if opt2 is None else opt2["relH"],
                "q2_relH": np.nan if q2 is None else q2["relH"],
            }
        )

        if block_idx in args.print_blocks:
            parts = []
            for label in ["opt2", "q2_vs_v1opt", "q2_vs_q1oracle", "prev_opt2", "svd_complement"]:
                rec = block_records.get(label)
                if rec is None:
                    continue
                parts.append(
                    f"{label}:score={rec['score']:.6f},gain2={rec['gain2']:.6f},"
                    f"phi={rec['phi']:.6f},relH={rec['relH']:.6f}"
                )
            print(
                f"block_only_scores matrix={matrix} block={block_idx} best_score={best_score_label} "
                f"best_gain={best_gain_label} q2_over_opt_score={q2_over_opt_score:.6f} "
                f"q2_over_opt_gain={q2_over_opt_gain:.6f} full_q2_over_opt_score={full_q2_over_opt_score:.6f} "
                + " | ".join(parts)
            )

        score_selected = np.zeros(rank, dtype=float)
        H_selected = np.zeros(rank, dtype=float)
        for j in range(rank):
            score_selected[j], _, H_selected[j] = probe.score_full_vector_details_forget(
                M_gain,
                A_block,
                V_selected[:, j],
                n,
                state_prev=state,
                score_variant="combined",
                old_row_memory=old_row_memory,
            )
        state, V_r, _ = make_state(M_gain, V_selected, H_selected, score_selected, rows_seen)
        old_row_memory, old_row_memory_idx = probe.select_old_row_memory(
            A[:end0, :].astype(work_dtype, copy=False),
            V_r.astype(work_dtype, copy=False),
            args.old_memory_size,
            np.random.default_rng(args.seed + end0),
            return_indices=True,
        )
        del old_row_memory_idx
        prev_opt2 = np.ascontiguousarray(V_selected[:, 1].copy())

    q2_score_wins = sum(r["q2_score_win"] for r in rows)
    q2_gain_wins = sum(r["q2_gain_win"] for r in rows)
    score_ratios = np.asarray([r["q2_over_opt_score"] for r in rows if np.isfinite(r["q2_over_opt_score"])])
    gain_ratios = np.asarray([r["q2_over_opt_gain"] for r in rows if np.isfinite(r["q2_over_opt_gain"])])
    full_score_ratios = np.asarray([r["full_q2_over_opt_score"] for r in rows if np.isfinite(r["full_q2_over_opt_score"])])
    opt2_phi = np.asarray([r["opt2_phi"] for r in rows if np.isfinite(r["opt2_phi"])])
    q2_phi = np.asarray([r["q2_phi"] for r in rows if np.isfinite(r["q2_phi"])])
    print(
        f"block_only_summary matrix={matrix} blocks={len(rows)} q2_score_wins={q2_score_wins}/{len(rows)} "
        f"q2_gain_wins={q2_gain_wins}/{len(rows)} mean_q2_over_opt_score={np.mean(score_ratios):.6f} "
        f"min_q2_over_opt_score={np.min(score_ratios):.6f} mean_q2_over_opt_gain={np.mean(gain_ratios):.6f} "
        f"mean_full_q2_over_opt_score={np.mean(full_score_ratios):.6f} "
        f"mean_opt2_phi={np.mean(opt2_phi):.6f} mean_q2_phi={np.mean(q2_phi):.6f} "
        f"elapsed={time.time() - t0:.3f}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices", nargs="+", default=["static-cex", "diffuse-diffuse", "mixed-tail-sharp"])
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--win", type=int, default=32)
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
    parser.add_argument("--num-restarts", type=int, default=8)
    parser.add_argument("--maxit", type=int, default=120)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--post-expansion-maxit", type=int, default=80)
    parser.add_argument("--r-sig", type=int, default=2)
    parser.add_argument("--alpha-sig", type=float, default=0.003)
    parser.add_argument("--alpha-tail", type=float, default=0.0145)
    parser.add_argument("--tail-scale", type=float, default=0.99)
    parser.add_argument("--sigma1", type=float, default=0.991)
    parser.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    parser.add_argument("--print-blocks", type=int, nargs="*", default=[1, 2, 3, 12, 24, 32])
    return parser.parse_args()


def main():
    args = parse_args()
    for matrix in args.matrices:
        run_matrix(matrix, args)


if __name__ == "__main__":
    main()
