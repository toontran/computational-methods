import argparse
import time

import numpy as np

import cex_restricted_space_probe as probe


def fmt(vals, precision=6):
    arr = np.asarray(vals, dtype=float).reshape(-1)
    if arr.size == 0:
        return ""
    return " ".join(f"{x:.{precision}f}" for x in arr)


def make_state(M_gain, V_score, H_score, score_score, rows_seen, carry="left"):
    if carry == "left":
        _, s_new, Vt_new, _ = probe.left_projected_operator_svd_factors(V_score.T, M_gain)
        V_r = Vt_new.T
    else:
        V_r, s_new = probe.projected_subspace_svd(M_gain.astype(np.float64), V_score.astype(np.float64))
    return {
        "V": np.ascontiguousarray(V_r.astype(np.float32, copy=False)),
        "s": np.asarray(s_new, dtype=np.float32),
        "s2": np.asarray(s_new, dtype=np.float32) ** 2,
        "H": np.asarray(H_score[: len(s_new)], dtype=np.float32),
        "score": np.asarray(score_score[: len(s_new)], dtype=np.float32),
        "rows_seen": int(rows_seen),
    }, V_r, s_new


def raw_oracle_columns(M_gain, V_exact, rank, dtype, row_samples=None):
    Q_oracle, Q_row = probe.projected_true_span_oracle(
        np.asarray(M_gain, dtype=dtype),
        np.asarray(V_exact, dtype=dtype)[:, : int(rank)],
        int(rank),
        dtype=dtype,
        row_samples=row_samples,
    )
    raw = []
    for j in range(min(int(rank), np.asarray(V_exact).shape[1])):
        v = probe.project_onto_span(np.asarray(V_exact, dtype=dtype)[:, j], Q_row).reshape(-1)
        nv = float(np.linalg.norm(v))
        if nv > 1e-30:
            raw.append(np.ascontiguousarray(v / nv, dtype=dtype))
    return Q_oracle, raw


def oracle_projection_norms(M_gain, V_exact, rank, dtype):
    _, Q_row = probe.projected_true_span_oracle(
        np.asarray(M_gain, dtype=dtype),
        np.asarray(V_exact, dtype=dtype)[:, : int(rank)],
        int(rank),
        dtype=dtype,
    )
    norms = []
    for j in range(min(int(rank), np.asarray(V_exact).shape[1])):
        v = np.asarray(V_exact, dtype=dtype)[:, j]
        norms.append(float(np.linalg.norm(probe.project_onto_span(v, Q_row))))
    return np.asarray(norms, dtype=float)


def orth_against(v, Q):
    out = np.asarray(v, dtype=np.float64).reshape(-1)
    if Q is not None and np.asarray(Q).size:
        Qq = probe.orthonormalize_columns(np.asarray(Q, dtype=np.float64), dtype=np.float64)
        out = out - Qq @ (Qq.T @ out)
    nrm = float(np.linalg.norm(out))
    if nrm <= 1e-30:
        return None
    return np.ascontiguousarray(out / nrm, dtype=np.float64)


def svd_complement(M_gain, Q):
    _, _, Vh = np.linalg.svd(np.asarray(M_gain, dtype=np.float64), full_matrices=False)
    best = None
    best_gain = -np.inf
    for row in Vh[: min(16, Vh.shape[0])]:
        cand = orth_against(row, Q)
        if cand is None:
            continue
        gain = float(np.linalg.norm(np.asarray(M_gain, dtype=np.float64) @ cand) ** 2)
        if gain > best_gain:
            best_gain = gain
            best = cand
    return best


def score_candidate(label, v, M_gain, A_block, rows_ref, state_prev, old_row_memory):
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
        "label": label,
        "score": float(comp["score_total"]),
        "gain2": float(comp["gain2"]),
        "phi": float(comp["phi"]),
        "relH": float(comp["pooled_rel_H"]),
    }


def candidate_line(block, context, records):
    parts = []
    for rec in records:
        if rec is None:
            continue
        parts.append(
            f"{rec['label']}:score={rec['score']:.6f},gain2={rec['gain2']:.6f},"
            f"phi={rec['phi']:.6f},relH={rec['relH']:.6f}"
        )
    return f"second_slot_scores block={block} context={context} " + " | ".join(parts)


def intervention_frame(mode, block_idx, V_score, Q_oracle, raw_oracle, rank):
    V = np.asarray(V_score, dtype=np.float64).copy()
    if Q_oracle is None or np.asarray(Q_oracle).shape[1] < int(rank):
        return V
    if mode == "normal":
        return V
    if mode == "force-b1" and block_idx == 1:
        return np.asarray(Q_oracle[:, :rank], dtype=np.float64)
    if mode == "force-b1b2" and block_idx in {1, 2}:
        return np.asarray(Q_oracle[:, :rank], dtype=np.float64)
    if mode == "force-second-b1" and block_idx == 1:
        q2 = None
        if len(raw_oracle) >= 2:
            q2 = orth_against(raw_oracle[1], V[:, :1])
        if q2 is None:
            q2 = orth_against(Q_oracle[:, 1], V[:, :1])
        if q2 is not None:
            V[:, 1] = q2
            V[:, :rank] = probe.orthonormalize_columns(V[:, :rank], dtype=np.float64)[:, :rank]
    return V


def run_trajectory(matrix, mode, args, emit_second_slot=False):
    np.random.seed(args.seed)
    A, V_exact, _, sigma1 = probe.generate_matrix_input(
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
    oracle_state = None
    V_r = None
    old_row_memory = None
    old_row_memory_idx = None
    oracle_old_row_memory = None
    prev_V_score = None
    prev_opt2 = None
    rows = []
    t0 = time.time()

    for block_idx, start0 in enumerate(range(0, n, args.win), start=1):
        end0 = min(start0 + args.win, n)
        A_block = A[start0:end0, :]
        A_block_work = A_block.astype(work_dtype, copy=False)
        if state is None:
            M_gain = A_block_work
            V_init = probe.row_norm_seed(A_block_work, rank)
            rows_seen = A_block.shape[0]
        else:
            B_top = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
            M_gain = np.vstack([B_top, A_block_work]).astype(work_dtype, copy=False)
            V_init = probe.row_norm_seed(A_block_work, rank)
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
            A_block=A_block_work,
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
        V_selected = intervention_frame(mode, block_idx, V_score[:, :rank], Q_oracle, raw_oracle, rank)
        if mode == "normal":
            V_selected = np.ascontiguousarray(np.asarray(V_score[:, :rank], dtype=np.float64))
        else:
            V_selected = probe.orthonormalize_columns(V_selected[:, :rank], dtype=np.float64)[:, :rank]
        score_selected = np.zeros(rank, dtype=float)
        H_selected = np.zeros(rank, dtype=float)
        s_selected = np.zeros(rank, dtype=float)
        for j in range(rank):
            score_selected[j], s_selected[j], H_selected[j] = probe.score_full_vector_details_forget(
                M_gain,
                A_block_work,
                V_selected[:, j],
                n,
                state_prev=state,
                score_variant="combined",
                old_row_memory=old_row_memory,
            )

        cos = probe.subspace_principal_cosines(V_selected, Q_oracle)
        exact_cos = probe.subspace_principal_cosines(V_selected, V_exact[:, :rank])
        oracle_proj_norm = oracle_projection_norms(M_gain, V_exact, rank, np.float64)
        V_sig = probe.orthonormalize_columns(V_exact[:, :rank], dtype=np.float64)
        sig_mass = float(np.linalg.norm(V_sig @ (V_sig.T @ V_selected), ord="fro") ** 2 / rank)
        tail_mass = max(0.0, 1.0 - sig_mass)
        survive_subspace = np.nan
        survive_v2 = np.nan
        prev2_score_ratio = np.nan
        if prev_opt2 is not None:
            survive_subspace = float(np.linalg.norm(V_selected @ (V_selected.T @ prev_opt2)))
            survive_v2 = abs(float(V_selected[:, 1] @ prev_opt2))
            prev_score = probe.combined_score_component_details(
                M_gain, A_block_work, prev_opt2, n, state_prev=state, old_row_memory=old_row_memory
            )["score_total"]
            curr_score = probe.combined_score_component_details(
                M_gain, A_block_work, V_selected[:, 1], n, state_prev=state, old_row_memory=old_row_memory
            )["score_total"]
            prev2_score_ratio = float(prev_score / max(curr_score, 1e-30))

        if emit_second_slot and mode == "normal":
            v1 = V_selected[:, :1]
            candidates = [
                ("opt2", V_selected[:, 1]),
                ("q2_vs_v1opt", orth_against(raw_oracle[1] if len(raw_oracle) > 1 else Q_oracle[:, 1], v1)),
                ("q2_vs_q1oracle", orth_against(raw_oracle[1] if len(raw_oracle) > 1 else Q_oracle[:, 1], Q_oracle[:, :1])),
                ("prev_opt2", prev_opt2),
                ("svd_complement", svd_complement(M_gain, v1)),
            ]
            actual_recs = [
                score_candidate(label, v, M_gain, A_block_work, n, state, old_row_memory)
                for label, v in candidates
            ]
            zero_recs = [
                score_candidate(label, v, A_block_work, A_block_work, n, state, old_row_memory)
                for label, v in candidates
            ]
            if oracle_state is not None:
                B_oracle = oracle_state["s"].astype(work_dtype)[:, None] * oracle_state["V"].astype(work_dtype).T
                M_oracle = np.vstack([B_oracle, A_block_work]).astype(work_dtype, copy=False)
                oracle_recs = [
                    score_candidate(label, v, M_oracle, A_block_work, n, oracle_state, old_row_memory)
                    for label, v in candidates
                ]
            else:
                oracle_recs = [
                    score_candidate(label, v, A_block_work, A_block_work, n, state, old_row_memory)
                    for label, v in candidates
                ]
            print(candidate_line(block_idx, "actual_B", actual_recs))
            print(candidate_line(block_idx, "zero_B", zero_recs))
            print(candidate_line(block_idx, "oracle_B", oracle_recs))
            print(
                f"tail_survival block={block_idx} prev_opt2_in_curr_subspace={survive_subspace:.6f} "
                f"prev_opt2_dot_curr_opt2={survive_v2:.6f} prev2_score_ratio={prev2_score_ratio:.6f}"
            )

        state, V_r, _ = make_state(M_gain, V_selected, H_selected, score_selected, rows_seen)

        # Parallel oracle-forced state used only as a scoring context.
        if oracle_state is None:
            M_gain_oracle = A_block_work
            oracle_rows_seen = A_block.shape[0]
        else:
            B_oracle = oracle_state["s"].astype(work_dtype)[:, None] * oracle_state["V"].astype(work_dtype).T
            M_gain_oracle = np.vstack([B_oracle, A_block_work]).astype(work_dtype, copy=False)
            oracle_rows_seen = oracle_state["rows_seen"] + A_block.shape[0]
        Q_oracle_state, _ = probe.projected_true_span_oracle(M_gain_oracle, V_exact[:, :rank], rank, dtype=np.float64)
        if Q_oracle_state.shape[1] >= rank:
            oracle_scores = np.zeros(rank, dtype=float)
            oracle_H = np.zeros(rank, dtype=float)
            for j in range(rank):
                oracle_scores[j], _, oracle_H[j] = probe.score_full_vector_details_forget(
                    M_gain_oracle,
                    A_block_work,
                    Q_oracle_state[:, j],
                    n,
                    state_prev=oracle_state,
                    score_variant="combined",
                    old_row_memory=old_row_memory,
                )
            oracle_state, _, _ = make_state(M_gain_oracle, Q_oracle_state[:, :rank], oracle_H, oracle_scores, oracle_rows_seen)

        old_row_memory, old_row_memory_idx = probe.select_old_row_memory(
            A[:end0, :].astype(work_dtype, copy=False),
            V_r.astype(work_dtype, copy=False),
            args.old_memory_size,
            np.random.default_rng(args.seed + end0),
            return_indices=True,
        )
        oracle_old_row_memory = old_row_memory
        prev_V_score = V_selected
        prev_opt2 = np.ascontiguousarray(V_selected[:, 1].copy())
        rows.append({
            "block": block_idx,
            "cos": cos,
            "exact_cos": exact_cos,
            "oracle_proj_norm": oracle_proj_norm,
            "tail_mass": tail_mass,
            "survive_subspace": survive_subspace,
            "survive_v2": survive_v2,
            "prev2_score_ratio": prev2_score_ratio,
        })

    align = float(np.linalg.norm((V_r @ V_r.T) @ V_exact[:, :1], "fro"))
    return {
        "matrix": matrix,
        "mode": mode,
        "align": align,
        "elapsed": time.time() - t0,
        "rows": rows,
        "sigma1": float(sigma1),
    }


def summarize(result):
    rows = result["rows"]
    final = rows[-1]
    blocks = {r["block"]: r for r in rows}
    print(
        f"trajectory_summary matrix={result['matrix']} mode={result['mode']} "
        f"mean_align={result['align']:.6f} elapsed={result['elapsed']:.3f}"
    )
    for b in [1, 2, 3, 5, 8, 12, 16, 24, 32]:
        if b not in blocks:
            continue
        r = blocks[b]
        print(
            f"  b{b:02d} block_sketch_oracle_cos={fmt(r['cos'])} "
            f"exact_oracle_cos={fmt(r['exact_cos'])} "
            f"oracle_proj_norm={fmt(r['oracle_proj_norm'])} "
            f"tail_mass={r['tail_mass']:.6f} "
            f"survive_subspace={r['survive_subspace']:.6f} "
            f"survive_v2={r['survive_v2']:.6f} "
            f"prev2_score_ratio={r['prev2_score_ratio']:.6f}"
        )
    print(
        f"  final block_sketch_oracle_cos={fmt(final['cos'])} "
        f"exact_oracle_cos={fmt(final['exact_cos'])} "
        f"oracle_proj_norm={fmt(final['oracle_proj_norm'])} "
        f"tail_mass={final['tail_mass']:.6f}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices", nargs="+", default=["mixed-tail-sharp", "diffuse-diffuse", "static-cex"])
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
    parser.add_argument(
        "--interventions",
        nargs="+",
        default=["normal", "force-b1", "force-b1b2", "force-second-b1"],
        choices=("normal", "force-b1", "force-b1b2", "force-second-b1"),
    )
    parser.add_argument("--emit-second-slot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for matrix in args.matrices:
        for mode in args.interventions:
            result = run_trajectory(matrix, mode, args, emit_second_slot=args.emit_second_slot)
            summarize(result)


if __name__ == "__main__":
    main()
