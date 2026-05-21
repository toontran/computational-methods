import argparse
import time

import numpy as np

import cex_restricted_space_probe as probe


def fmt(x):
    if isinstance(x, str):
        return x
    if isinstance(x, int):
        return str(x)
    return f"{float(x):.6g}"


def make_base_params(args):
    return {
        "q0": args.q0,
        "qmax": args.qmax,
        "krylov_depth": args.krylov_depth,
        "residual_tol": args.residual_tol,
        "expansion_maxit": args.expansion_maxit,
        "num_restarts": args.num_restarts,
        "maxit": args.maxit,
        "tol": args.tol,
        "reduced_optimizer": args.reduced_optimizer,
        "work_dtype": np.float64 if args.dtype == "float64" else np.float32,
        "expansion_direction": args.expansion_direction,
        "reuse_line_search_grad": args.reuse_line_search_grad,
        "expansion_warm_start": args.expansion_warm_start,
        "post_expansion_maxit": args.post_expansion_maxit,
        "joint_warm_start_oracle": args.joint_warm_start_oracle,
    }


def run_one_block(A, V_exact, sigma1, state_prev, block_idx, params, candidate, seed):
    win = candidate.get("win")
    rank = candidate.get("rank")
    start0 = block_idx * win
    end0 = min(start0 + win, A.shape[0])
    work_dtype = params["work_dtype"]
    A_block = A[start0:end0, :]
    A_block_work = A_block.astype(work_dtype, copy=False)

    if state_prev is None:
        M_gain = A_block_work
        V_init = None
        rows_seen = A_block.shape[0]
    else:
        B_top = state_prev["s"].astype(work_dtype)[:, None] * state_prev["V"].astype(work_dtype).T
        M_gain = np.vstack([B_top, A_block_work]).astype(work_dtype, copy=False)
        V_init = state_prev["V"].astype(work_dtype, copy=False)
        rows_seen = state_prev["rows_seen"] + A_block.shape[0]

    run_params = dict(params)
    run_params.update(candidate["overrides"])
    t0 = time.time()
    V_score, s_score, H_score, score_score, diag = probe.entropy_iter_basis_forget(
        M_gain=M_gain,
        active_r=rank,
        rows_ref=A.shape[0],
        V_init=V_init,
        q0=run_params["q0"],
        qmax=run_params["qmax"],
        krylov_depth=run_params["krylov_depth"],
        residual_tol=run_params["residual_tol"],
        expansion_maxit=run_params["expansion_maxit"],
        num_restarts=run_params["num_restarts"],
        maxit=run_params["maxit"],
        tol=run_params["tol"],
        rng=np.random.default_rng(seed),
        verbose=False,
        state_prev=state_prev,
        A_block=A_block_work,
        rows_total=rows_seen,
        reduced_optimizer=run_params["reduced_optimizer"],
        work_dtype=work_dtype,
        expansion_direction=run_params["expansion_direction"],
        reuse_line_search_grad=run_params["reuse_line_search_grad"],
        expansion_warm_start=run_params["expansion_warm_start"],
        post_expansion_maxit=run_params["post_expansion_maxit"],
        basis_selection=run_params.get("basis_selection", "greedy"),
        joint_warm_start_greedy=run_params.get("joint_warm_start_greedy", False),
        joint_warm_start_oracle=run_params.get("joint_warm_start_oracle", False),
        oracle_warm_start_target=V_exact,
        joint_warm_start_rotations=run_params.get("joint_warm_start_rotations", 0),
        joint_warm_start_rotation_angle=run_params.get("joint_warm_start_rotation_angle", np.pi / 4),
        joint_warm_start_perturbations=run_params.get("joint_warm_start_perturbations", 0),
        joint_warm_start_perturb_scale=run_params.get("joint_warm_start_perturb_scale", 1e-2),
        joint_default_svd_start=run_params.get("joint_default_svd_start", True),
        joint_oversample=run_params.get("joint_oversample", 0),
        joint_oversample_rotate=run_params.get("joint_oversample_rotate", "svd"),
        joint_solver=run_params.get("joint_solver", "riemannian"),
    )
    opt_elapsed = time.time() - t0

    _, s_new, Vt_new, _ = probe.left_projected_operator_svd_factors(V_score.T, M_gain)
    V_r = Vt_new.T
    S_r = np.diag(s_new)
    align = np.linalg.norm((V_r @ V_r.T) @ V_exact[:, :1], "fro")
    rel_err = abs(S_r[0, 0] - sigma1) / sigma1
    return {
        "name": candidate["name"],
        "block": f"{start0 + 1}:{end0}",
        "score_sum": float(np.sum(score_score[:rank])),
        "scores": ",".join(f"{x:.6g}" for x in score_score[:rank]),
        "align": float(align),
        "rel_err": float(rel_err),
        "svals": ",".join(f"{x:.6g}" for x in s_new[:rank]),
        "subspace_dims": ",".join(str(int(x)) for x in diag["subspace_dims"][:rank]),
        "grad_perp_max": float(np.max(diag["grad_perp_ratio"][:rank])),
        "elapsed": opt_elapsed,
    }


def advance_reference_state(A, args, params, target_block):
    state = None
    for block_idx in range(target_block):
        result = run_one_block(
            A=A,
            V_exact=args["_V_exact"],
            sigma1=args["_sigma1"],
            state_prev=state,
            block_idx=block_idx,
            params=params,
            candidate={
                "name": "reference-greedy",
                "win": args["win"],
                "rank": args["rank"],
                "overrides": {"basis_selection": "greedy"},
            },
            seed=args["seed"],
        )

        start0 = block_idx * args["win"]
        end0 = min(start0 + args["win"], A.shape[0])
        A_block = A[start0:end0, :]
        work_dtype = params["work_dtype"]
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
            active_r=args["rank"],
            rows_ref=A.shape[0],
            V_init=V_init,
            q0=params["q0"],
            qmax=params["qmax"],
            krylov_depth=params["krylov_depth"],
            residual_tol=params["residual_tol"],
            expansion_maxit=params["expansion_maxit"],
            num_restarts=params["num_restarts"],
            maxit=params["maxit"],
            tol=params["tol"],
            rng=np.random.default_rng(args["seed"]),
            verbose=False,
            state_prev=state,
            A_block=A_block_work,
            rows_total=rows_seen,
            reduced_optimizer=params["reduced_optimizer"],
            work_dtype=work_dtype,
            expansion_direction=params["expansion_direction"],
            reuse_line_search_grad=params["reuse_line_search_grad"],
            expansion_warm_start=params["expansion_warm_start"],
            post_expansion_maxit=params["post_expansion_maxit"],
            basis_selection="greedy",
            joint_warm_start_oracle=params.get("joint_warm_start_oracle", False),
            oracle_warm_start_target=args["_V_exact"],
        )
        _, s_new, Vt_new, _ = probe.left_projected_operator_svd_factors(V_score.T, M_gain)
        V_r = Vt_new.T
        state = {
            "V": V_r,
            "s": s_new,
            "s2": s_new ** 2,
            "H": np.asarray(H_score[: len(s_new)], dtype=np.float32),
            "score": np.asarray(score_score[: len(s_new)], dtype=np.float32),
            "rows_seen": rows_seen,
            "diag": diag,
        }
        _ = result
    return state


def print_table(rows):
    headers = [
        "name",
        "block",
        "score_sum",
        "scores",
        "align",
        "rel_err",
        "svals",
        "subspace_dims",
        "grad_perp_max",
        "elapsed",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="static-cex")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--win", type=int, default=32)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--preset", default="cex-replicate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-block", type=int, default=3, help="0-indexed block to compare.")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--q0", type=int, default=4)
    parser.add_argument("--qmax", type=int, default=32)
    parser.add_argument("--krylov-depth", type=int, default=2)
    parser.add_argument("--residual-tol", type=float, default=1e-4)
    parser.add_argument("--expansion-maxit", type=int, default=8)
    parser.add_argument("--num-restarts", type=int, default=8)
    parser.add_argument("--maxit", type=int, default=250)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--post-expansion-maxit", type=int, default=120)
    parser.add_argument("--reduced-optimizer", default="cex")
    parser.add_argument("--expansion-direction", default="residual")
    parser.add_argument("--reuse-line-search-grad", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expansion-warm-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--joint-warm-start-oracle", action=argparse.BooleanOptionalAction, default=False)
    parsed = parser.parse_args()

    np.random.seed(parsed.seed)
    A, V_exact, _, sigma1 = probe.generate_matrix_input(
        matrix=parsed.matrix,
        n=parsed.n,
        preset=parsed.preset,
        seed=parsed.seed,
    )
    A = np.asarray(A, dtype=np.float64)

    args_dict = vars(parsed)
    args_dict["_V_exact"] = V_exact
    args_dict["_sigma1"] = sigma1
    params = make_base_params(parsed)
    state_prev = advance_reference_state(A, args_dict, params, parsed.target_block)

    candidates = [
        {"name": "greedy-old", "win": parsed.win, "rank": parsed.rank, "overrides": {"basis_selection": "greedy"}},
        {
            "name": "joint-old",
            "win": parsed.win,
            "rank": parsed.rank,
            "overrides": {"basis_selection": "joint"},
        },
        {
            "name": "joint-greedywarm",
            "win": parsed.win,
            "rank": parsed.rank,
            "overrides": {"basis_selection": "joint", "joint_warm_start_greedy": True},
        },
        {
            "name": "joint-high-random",
            "win": parsed.win,
            "rank": parsed.rank,
            "overrides": {
                "basis_selection": "joint",
                "num_restarts": 64,
                "maxit": 1000,
                "tol": 1e-10,
                "post_expansion_maxit": 500,
            },
        },
        {
            "name": "joint-slsqp",
            "win": parsed.win,
            "rank": parsed.rank,
            "overrides": {"basis_selection": "joint", "joint_solver": "slsqp", "tol": 1e-10},
        },
        {
            "name": "joint-over1-svd",
            "win": parsed.win,
            "rank": parsed.rank,
            "overrides": {
                "basis_selection": "joint",
                "joint_warm_start_greedy": True,
                "joint_oversample": 1,
                "joint_oversample_rotate": "svd",
            },
        },
    ]
    rows = [
        run_one_block(A, V_exact, sigma1, state_prev, parsed.target_block, params, candidate, parsed.seed)
        for candidate in candidates
    ]
    print(f"Fixed previous state: reference greedy through block {parsed.target_block - 1}")
    print_table(rows)


if __name__ == "__main__":
    main()
