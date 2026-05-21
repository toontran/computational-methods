import argparse

import numpy as np

import cex_restricted_space_probe as probe
import init_geometry_probe as geom


def parse_lambdas(text):
    vals = []
    for part in text.split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    if not vals:
        raise ValueError("At least one lambda is required.")
    return vals


def fmt(x):
    if isinstance(x, str):
        return x
    if not np.isfinite(float(x)):
        return "nan"
    return f"{float(x):.6g}"


def run_for_lambda(A, V_exact, sigma1, args, lam):
    work_dtype = np.float64 if args.dtype == "float64" else np.float32
    state = None
    final_align = np.nan
    final_relerr = np.nan
    final_found_qoracle = "n/a"
    final_score_sum = np.nan
    final_reg_score_sum = np.nan
    outside_norms = []

    for block_idx, start0 in enumerate(range(0, A.shape[0], args.win)):
        end0 = min(start0 + args.win, A.shape[0])
        A_block = A[start0:end0, :]
        A_block_work = A_block.astype(work_dtype, copy=False)
        if state is None:
            M_gain = A_block_work
            V_init = None
            rows_seen = A_block.shape[0]
        else:
            B_top = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
            M_gain = np.vstack([B_top, A_block_work]).astype(work_dtype, copy=False)
            V_init = state["V"].astype(work_dtype, copy=False)
            rows_seen = state["rows_seen"] + A_block.shape[0]

        V_score, _, H_score, score_score, diag = probe.entropy_iter_basis_forget(
            M_gain=M_gain,
            active_r=args.rank,
            rows_ref=A.shape[0],
            V_init=V_init,
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
            work_dtype=work_dtype,
            expansion_direction="residual",
            reuse_line_search_grad=True,
            expansion_warm_start=True,
            post_expansion_maxit=args.post_expansion_maxit,
            basis_selection=args.basis_selection,
            joint_warm_start_greedy=args.joint_warm_start_greedy,
            joint_warm_start_oracle=args.joint_warm_start_oracle,
            oracle_warm_start_target=V_exact,
            joint_solver=args.joint_solver,
            row_concentration_lambda=lam,
        )

        _, s_new, Vt_new, _ = probe.left_projected_operator_svd_factors(V_score.T, M_gain)
        V_r = np.asarray(Vt_new.T, dtype=np.float64)
        Q_oracle = geom.oracle_frame(M_gain, V_exact, args.rank)
        if Q_oracle.shape[1] >= args.rank:
            grad_row = geom.dominant_outside_projection_summary(
                Q_oracle[:, : args.rank], state, A_block_work, M_gain, V_exact, A.shape[0],
                row_concentration_lambda=lam,
            )
            outside_norms.append(grad_row["outside_norm"])
            final_found_qoracle = geom.fmt_cos(geom.subspace_cosines(V_r[:, : args.rank], Q_oracle))

        align = np.linalg.norm((V_r @ V_r.T) @ V_exact[:, :1], "fro")
        final_align = float(align)
        final_relerr = float(abs(s_new[0] - sigma1) / sigma1)
        final_score_sum = float(np.sum(score_score[: args.rank]))
        final_reg_score_sum = float(diag["regularized_score_sum"])
        state = {
            "V": V_r,
            "s": s_new,
            "s2": s_new ** 2,
            "H": np.asarray(H_score[: len(s_new)], dtype=np.float32),
            "score": np.asarray(score_score[: len(s_new)], dtype=np.float32),
            "rows_seen": rows_seen,
            "diag": diag,
        }

    return {
        "lambda": lam,
        "final_align": final_align,
        "final_relerr": final_relerr,
        "found_vs_Q_oracle": final_found_qoracle,
        "outside_norm_first": outside_norms[0] if outside_norms else np.nan,
        "outside_norm_last": outside_norms[-1] if outside_norms else np.nan,
        "score_sum": final_score_sum,
        "regularized_score_sum": final_reg_score_sum,
    }


def print_table(rows):
    headers = [
        "lambda",
        "final_align",
        "final_relerr",
        "found_vs_Q_oracle",
        "outside_norm_first",
        "outside_norm_last",
        "score_sum",
        "regularized_score_sum",
    ]
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(fmt(row[h])))
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print(" | ".join("-" * widths[h] for h in headers))
    for row in rows:
        print(" | ".join(fmt(row[h]).ljust(widths[h]) for h in headers))


def main():
    parser = argparse.ArgumentParser(description="Sweep row-concentration regularization lambdas.")
    parser.add_argument("--matrix", default="static-cex")
    parser.add_argument("--preset", default="cex-replicate")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--win", type=int, default=32)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--q0", type=int, default=4)
    parser.add_argument("--qmax", type=int, default=32)
    parser.add_argument("--krylov-depth", type=int, default=2)
    parser.add_argument("--residual-tol", type=float, default=1e-4)
    parser.add_argument("--expansion-maxit", type=int, default=8)
    parser.add_argument("--num-restarts", type=int, default=64)
    parser.add_argument("--maxit", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--post-expansion-maxit", type=int, default=500)
    parser.add_argument("--basis-selection", choices=("greedy", "joint"), default="joint")
    parser.add_argument("--joint-warm-start-greedy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--joint-warm-start-oracle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--joint-solver", choices=("riemannian", "slsqp"), default="riemannian")
    parser.add_argument("--lambdas", default="0,1e-4,3e-4,1e-3,3e-3,1e-2")
    args = parser.parse_args()

    np.random.seed(args.seed)
    A, V_exact, _, sigma1 = probe.generate_matrix_input(
        matrix=args.matrix,
        n=args.n,
        preset=args.preset,
        seed=args.seed,
    )
    A = np.asarray(A, dtype=np.float64)
    rows = [run_for_lambda(A, V_exact, sigma1, args, lam) for lam in parse_lambdas(args.lambdas)]
    print_table(rows)


if __name__ == "__main__":
    main()
