import argparse
import time

import numpy as np

import cex_restricted_space_probe as probe


def fmt(vals, precision=6):
    arr = np.asarray(vals, dtype=float).reshape(-1)
    if arr.size == 0:
        return ""
    return " ".join(f"{x:.{precision}f}" for x in arr)


def frame_tail_mass(V, V_exact, rank):
    Vq = probe.orthonormalize_columns(np.asarray(V, dtype=np.float64)[:, :rank], dtype=np.float64)
    Qsig = probe.orthonormalize_columns(np.asarray(V_exact, dtype=np.float64)[:, :rank], dtype=np.float64)
    sig_mass = float(np.linalg.norm(Qsig @ (Qsig.T @ Vq), ord="fro") ** 2 / max(rank, 1))
    return max(0.0, 1.0 - sig_mass)


def oracle_frame(M_gain, V_exact, rank):
    Q_oracle, _ = probe.projected_true_span_oracle(
        np.asarray(M_gain, dtype=np.float64),
        np.asarray(V_exact, dtype=np.float64)[:, :rank],
        rank,
        dtype=np.float64,
    )
    return Q_oracle[:, :rank]


def abs_matrix(A):
    return np.abs(np.asarray(A, dtype=np.float64))


def row_line(matrix, block, kind, **items):
    parts = [f"matrix={matrix}", f"block={block}", f"kind={kind}"]
    for key, val in items.items():
        if isinstance(val, str):
            parts.append(f"{key}={val}")
        elif np.asarray(val).ndim > 0:
            parts.append(f"{key}=[{fmt(val)}]")
        else:
            parts.append(f"{key}={float(val):.6f}")
    return " ".join(parts)


def run_matrix(matrix, args):
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
    n = A.shape[0]
    rank = int(args.rank)
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    state = None
    old_row_memory = None
    V_r = None
    prev_carried = None
    prev_selected = None
    t0 = time.time()

    print(f"rotation_handoff_start matrix={matrix} A={A.shape} sigma1={sigma1:.12g}")
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
            rows_seen = int(state["rows_seen"]) + A_block.shape[0]

        V_score, s_score, H_score, score_score, diag = probe.entropy_iter_basis_forget(
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
        V_sel = np.ascontiguousarray(np.asarray(V_score[:, :rank], dtype=np.float64))
        _, s_new, Vt_new, _ = probe.left_projected_operator_svd_factors(V_score.T, M_gain)
        V_car = np.ascontiguousarray(Vt_new.T[:, :rank], dtype=np.float64)
        Q_oracle = oracle_frame(M_gain, V_exact, rank)

        sel_oracle_cos = probe.subspace_principal_cosines(V_sel, Q_oracle)
        car_oracle_cos = probe.subspace_principal_cosines(V_car, Q_oracle)
        sel_car_cos = probe.subspace_principal_cosines(V_sel, V_car)
        sel_tail = frame_tail_mass(V_sel, V_exact, rank)
        car_tail = frame_tail_mass(V_car, V_exact, rank)
        selected_to_carried_cols = abs_matrix(V_sel.T @ V_car)
        oracle_coords_sel = abs_matrix(Q_oracle.T @ V_sel)
        oracle_coords_car = abs_matrix(Q_oracle.T @ V_car)

        prev_car_v2_in_sel = np.nan
        prev_car_v2_dot_sel2 = np.nan
        prev_sel_v2_in_sel = np.nan
        if prev_carried is not None:
            prev_car_v2 = prev_carried[:, 1]
            prev_car_v2_in_sel = float(np.linalg.norm(V_sel @ (V_sel.T @ prev_car_v2)))
            prev_car_v2_dot_sel2 = abs(float(V_sel[:, 1] @ prev_car_v2))
        if prev_selected is not None:
            prev_sel_v2 = prev_selected[:, 1]
            prev_sel_v2_in_sel = float(np.linalg.norm(V_sel @ (V_sel.T @ prev_sel_v2)))

        print(row_line(
            matrix,
            block_idx,
            "subspace",
            sel_oracle_cos=sel_oracle_cos,
            car_oracle_cos=car_oracle_cos,
            sel_car_cos=sel_car_cos,
            sel_tail=sel_tail,
            car_tail=car_tail,
            s=s_new[:rank],
        ))
        print(row_line(
            matrix,
            block_idx,
            "coords",
            sel_to_car=selected_to_carried_cols.reshape(-1),
            oracle_coords_sel=oracle_coords_sel.reshape(-1),
            oracle_coords_car=oracle_coords_car.reshape(-1),
        ))
        print(row_line(
            matrix,
            block_idx,
            "survival",
            prev_car_v2_in_sel=prev_car_v2_in_sel,
            prev_car_v2_dot_sel2=prev_car_v2_dot_sel2,
            prev_sel_v2_in_sel=prev_sel_v2_in_sel,
        ))

        state = {
            "V": V_car.astype(np.float32, copy=False),
            "s": np.asarray(s_new[:rank], dtype=np.float32),
            "s2": np.asarray(s_new[:rank], dtype=np.float32) ** 2,
            "H": np.asarray(H_score[:rank], dtype=np.float32),
            "score": np.asarray(score_score[:rank], dtype=np.float32),
            "rows_seen": rows_seen,
            "diag": diag,
        }
        V_r = V_car
        old_row_memory = probe.select_old_row_memory(
            A[:end0, :].astype(work_dtype, copy=False),
            V_r.astype(work_dtype, copy=False),
            args.old_memory_size,
            np.random.default_rng(args.seed + end0),
            return_indices=False,
        )
        prev_carried = V_car
        prev_selected = V_sel

    align = float(np.linalg.norm((V_r @ V_r.T) @ V_exact[:, :1], "fro"))
    print(f"rotation_handoff_done matrix={matrix} mean_align={align:.6f} elapsed={time.time() - t0:.3f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices", nargs="+", default=["mixed-tail-sharp", "static-cex"])
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
    return parser.parse_args()


def main():
    args = parse_args()
    for matrix in args.matrices:
        run_matrix(matrix, args)


if __name__ == "__main__":
    main()
