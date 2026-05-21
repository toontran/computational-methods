import argparse

import numpy as np

import cex_restricted_space_probe as probe


def normed(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(x))
    if n <= 1e-30:
        return None
    return np.ascontiguousarray(x / n)


def cos_abs(a, b):
    if a is None or b is None:
        return np.nan
    return abs(float(np.asarray(a).reshape(-1) @ np.asarray(b).reshape(-1)))


def angle_deg(c):
    if np.isnan(c):
        return np.nan
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))


def fmt(x):
    if np.isnan(float(x)):
        return "nan"
    return f"{float(x):.6f}"


def orth_against(v, Q):
    out = np.asarray(v, dtype=np.float64).reshape(-1)
    if Q is not None and np.asarray(Q).size:
        Qq = probe.orthonormalize_columns(np.asarray(Q, dtype=np.float64), dtype=np.float64)
        out = out - Qq @ (Qq.T @ out)
    return normed(out)


def run_matrix(matrix, args):
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
    prev = None
    rows = []

    print(f"outside_rotation_start matrix={matrix} A={A.shape} sigma1={sigma1:.12g}")
    for block_idx, start0 in enumerate(range(0, n, args.win), start=1):
        end0 = min(start0 + args.win, n)
        A_block = A[start0:end0, :]
        A_block_work = A_block.astype(work_dtype, copy=False)
        if state is None:
            M_gain = A_block_work
            rows_seen = A_block.shape[0]
        else:
            B_top = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
            M_gain = np.vstack([B_top, A_block_work]).astype(work_dtype, copy=False)
            rows_seen = int(state["rows_seen"]) + A_block.shape[0]
        V_init = probe.row_norm_seed(A_block_work, rank)

        V_score, _, H_score, score_score, _ = probe.entropy_iter_basis_forget(
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
        V_sel = np.asarray(V_score[:, :rank], dtype=np.float64)
        _, s_new, Vt_new, _ = probe.left_projected_operator_svd_factors(V_score.T, M_gain)
        V_car = np.asarray(Vt_new.T[:, :rank], dtype=np.float64)
        Q_oracle, _ = probe.projected_true_span_oracle(M_gain, V_exact[:, :rank], rank, dtype=np.float64)
        Q_exact = probe.orthonormalize_columns(V_exact[:, :rank], dtype=np.float64)

        # Outside means outside the fixed exact top-r subspace. This isolates tail
        # direction movement from the changing projected-oracle frame.
        P_exact = Q_exact @ Q_exact.T
        sel2 = normed(V_sel[:, 1])
        car2 = normed(V_car[:, 1])
        sel2_out = normed(sel2 - P_exact @ sel2)
        car2_out = normed(car2 - P_exact @ car2)
        sel2_in = normed(P_exact @ sel2)
        car2_in = normed(P_exact @ car2)
        q1 = normed(Q_oracle[:, 0])
        q2 = normed(Q_oracle[:, 1])
        q2_out = normed(q2 - P_exact @ q2)

        if prev is None:
            metrics = {
                "sel2_out_cos": np.nan,
                "car2_out_cos": np.nan,
                "sel2_in_cos": np.nan,
                "car2_in_cos": np.nan,
                "q1_cos": np.nan,
                "q2_cos": np.nan,
                "q2_out_cos": np.nan,
            }
        else:
            metrics = {
                "sel2_out_cos": cos_abs(prev["sel2_out"], sel2_out),
                "car2_out_cos": cos_abs(prev["car2_out"], car2_out),
                "sel2_in_cos": cos_abs(prev["sel2_in"], sel2_in),
                "car2_in_cos": cos_abs(prev["car2_in"], car2_in),
                "q1_cos": cos_abs(prev["q1"], q1),
                "q2_cos": cos_abs(prev["q2"], q2),
                "q2_out_cos": cos_abs(prev["q2_out"], q2_out),
            }

        sel_oracle_cos = probe.subspace_principal_cosines(V_sel, Q_oracle)
        car_oracle_cos = probe.subspace_principal_cosines(V_car, Q_oracle)
        print(
            f"outside_rotation matrix={matrix} block={block_idx} "
            f"sel2_out_cos={fmt(metrics['sel2_out_cos'])} sel2_out_angle={fmt(angle_deg(metrics['sel2_out_cos']))} "
            f"car2_out_cos={fmt(metrics['car2_out_cos'])} car2_out_angle={fmt(angle_deg(metrics['car2_out_cos']))} "
            f"sel2_in_cos={fmt(metrics['sel2_in_cos'])} "
            f"q1_cos={fmt(metrics['q1_cos'])} q2_cos={fmt(metrics['q2_cos'])} "
            f"q2_out_cos={fmt(metrics['q2_out_cos'])} q2_out_angle={fmt(angle_deg(metrics['q2_out_cos']))} "
            f"sel_oracle_cos=[{fmt(sel_oracle_cos[0])} {fmt(sel_oracle_cos[1])}] "
            f"car_oracle_cos=[{fmt(car_oracle_cos[0])} {fmt(car_oracle_cos[1])}]"
        )
        rows.append({"block": block_idx, **metrics})
        state = {
            "V": V_car.astype(np.float32, copy=False),
            "s": np.asarray(s_new[:rank], dtype=np.float32),
            "s2": np.asarray(s_new[:rank], dtype=np.float32) ** 2,
            "H": np.asarray(H_score[:rank], dtype=np.float32),
            "score": np.asarray(score_score[:rank], dtype=np.float32),
            "rows_seen": rows_seen,
        }
        old_row_memory = probe.select_old_row_memory(
            A[:end0, :].astype(work_dtype, copy=False),
            V_car.astype(work_dtype, copy=False),
            args.old_memory_size,
            np.random.default_rng(args.seed + end0),
            return_indices=False,
        )
        prev = {
            "sel2_out": sel2_out,
            "car2_out": car2_out,
            "sel2_in": sel2_in,
            "car2_in": car2_in,
            "q1": q1,
            "q2": q2,
            "q2_out": q2_out,
        }

    usable = [r for r in rows if r["block"] > 1]
    for key in ["sel2_out_cos", "car2_out_cos", "sel2_in_cos", "q1_cos", "q2_cos", "q2_out_cos"]:
        vals = np.asarray([r[key] for r in usable if not np.isnan(r[key])], dtype=float)
        if vals.size:
            print(
                f"outside_rotation_summary matrix={matrix} metric={key} "
                f"mean_cos={float(np.mean(vals)):.6f} min_cos={float(np.min(vals)):.6f} "
                f"mean_angle={float(np.mean([angle_deg(v) for v in vals])):.6f}"
            )


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
