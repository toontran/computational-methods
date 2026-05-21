import argparse
import contextlib
import io
import time
from types import SimpleNamespace

import cex_restricted_space_probe as probe


MATRICES = [
    "static-cex",
    "diffuse-diffuse",
    "mixed-tail-soft",
    "mixed-tail-balanced",
    "mixed-tail-sharp",
    "residual-spiky-shocks",
    "alternative-data-signals",
    "futures-term-structure",
    "crowded-strategy",
    "rates-cross-currency",
    "options-vol-surface",
    "risk-residual-panel",
    "macro-factor-panel",
    "realized-vol-corr",
    "etf-basket-basis",
    "execution-cost-slippage",
    "stat-arb-spreads",
    "intraday-liquidity-shape",
]


METHODS = {
    "greedy-medium": {
        "basis_selection": "greedy",
        "joint_warm_start_greedy": False,
        "joint_warm_start_oracle": False,
        "rownorm_seed_first_block": True,
        "rownorm_seed_all_blocks": True,
    },
    "hybrid-half": {
        "basis_selection": "greedy",
        "joint_warm_start_greedy": False,
        "joint_warm_start_oracle": False,
        "rownorm_seed_first_block": True,
        "rownorm_seed_all_blocks": True,
        "combined_rank_ratio": 0.5,
    },
    "greedy-oracle": {
        "basis_selection": "greedy",
        "joint_warm_start_greedy": False,
        "joint_warm_start_oracle": False,
        "rownorm_seed_first_block": True,
        "rownorm_seed_all_blocks": True,
        "oracle_candidate_check": True,
    },
    "joint-greedywarm-medium": {
        "basis_selection": "joint",
        "joint_warm_start_greedy": True,
        "joint_warm_start_oracle": False,
    },
    "joint-oraclewarm-medium": {
        "basis_selection": "joint",
        "joint_warm_start_greedy": False,
        "joint_warm_start_oracle": True,
    },
    "isvd": {
        "mode": "isvd",
        "basis_selection": "greedy",
        "joint_warm_start_greedy": False,
        "joint_warm_start_oracle": False,
    },
    "fd": {
        "mode": "fd",
        "basis_selection": "greedy",
        "joint_warm_start_greedy": False,
        "joint_warm_start_oracle": False,
    },
}


def _resolve_combined_rank(overrides, rank):
    if "combined_rank" in overrides:
        return int(overrides["combined_rank"])
    ratio = overrides.get("combined_rank_ratio")
    if ratio is None:
        return None
    return max(0, min(int(rank), int(round(float(ratio) * int(rank)))))


def make_args(matrix, args, overrides):
    return SimpleNamespace(
        mat_input=None,
        matrix=matrix,
        n=args.n,
        r_sig=2,
        alpha_sig=0.003,
        alpha_tail=0.0145,
        tail_scale=0.99,
        sigma1=0.991,
        v_type="rand",
        mode=overrides.get("mode", "restricted"),
        rank=args.rank,
        win=args.win,
        preset=args.matrix_preset,
        shuffle_rows=args.shuffle_rows,
        row_shuffle_seed=args.row_shuffle_seed,
        cex_replicate=False,
        q0=args.q0,
        qmax=args.qmax,
        krylov_depth=args.krylov_depth,
        residual_tol=args.residual_tol,
        expansion_maxit=args.expansion_maxit,
        num_restarts=args.num_restarts,
        maxit=args.maxit,
        tol=args.tol,
        seed=args.seed,
        normalize_by_sigma=False,
        carry="left",
        reduced_optimizer="cex",
        basis_selection=overrides["basis_selection"],
        joint_warm_start_greedy=overrides["joint_warm_start_greedy"],
        joint_warm_start_oracle=overrides["joint_warm_start_oracle"],
        rownorm_seed_first_block=overrides.get("rownorm_seed_first_block", False),
        rownorm_seed_all_blocks=overrides.get("rownorm_seed_all_blocks", False),
        joint_warm_start_rotations=0,
        joint_warm_start_rotation_angle=0.7853981633974483,
        joint_warm_start_perturbations=0,
        joint_warm_start_perturb_scale=1e-2,
        joint_default_svd_start=True,
        joint_oversample=0,
        joint_oversample_rotate="svd",
        joint_solver="riemannian",
        row_concentration_lambda=args.row_concentration_lambda,
        row_leverage_lambda=0.0,
        row_leverage_mode="none",
        row_leverage_rank=2,
        score_variant="combined",
        old_memory_size=args.old_memory_size if args.old_memory_size is not None else args.win,
        debug_mode="off",
        combined_rank=_resolve_combined_rank(overrides, args.rank),
        oracle_candidate_check=overrides.get("oracle_candidate_check", False),
        oracle_sketch_all_seen_rows=False,
        dump_score_components=False,
        dump_consecutive_tail_diagnostics=False,
        old_memory_holdout_size=None,
        dump_oracle_old_row_responses=False,
        dump_oracle_old_row_response_block=3,
        dtype=args.dtype,
        expansion_direction="residual",
        reuse_line_search_grad=True,
        expansion_warm_start=True,
        post_expansion_maxit=args.post_expansion_maxit,
        benchmark_output=None,
        benchmark_append=False,
        verbose=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Medium-budget restricted benchmark sweep.")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--win", type=int, default=32)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-rows", action="store_true")
    parser.add_argument("--row-shuffle-seed", type=int)
    parser.add_argument("--matrix-preset", default="small")
    parser.add_argument("--q0", type=int, default=8)
    parser.add_argument("--qmax", type=int, default=48)
    parser.add_argument("--krylov-depth", type=int, default=2)
    parser.add_argument("--residual-tol", type=float, default=1e-2)
    parser.add_argument("--expansion-maxit", type=int, default=8)
    parser.add_argument("--num-restarts", type=int, default=8)
    parser.add_argument("--maxit", type=int, default=120)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--post-expansion-maxit", type=int, default=80)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--old-memory-size", type=int)
    parser.add_argument("--row-concentration-lambda", type=float, default=0.0)
    parser.add_argument("--matrices", default=",".join(MATRICES))
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()

    matrices = [x.strip() for x in args.matrices.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    unknown_methods = sorted(set(methods) - set(METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown methods: {', '.join(unknown_methods)}")
    print(
        f"# n={args.n}, rank={args.rank}, win={args.win}, matrix_preset={args.matrix_preset}, "
        f"seed={args.seed}, shuffle_rows={args.shuffle_rows}, "
        f"row_shuffle_seed={args.seed if args.row_shuffle_seed is None else args.row_shuffle_seed}"
    )
    print(
        "# medium params: "
        f"q0={args.q0}, qmax={args.qmax}, restarts={args.num_restarts}, "
        f"maxit={args.maxit}, expansion_maxit={args.expansion_maxit}, "
        f"post_expansion_maxit={args.post_expansion_maxit}, dtype={args.dtype}"
    )
    print("matrix\tmethod\tmean_align\tmean_relerr_sval\telapsed")
    for matrix in matrices:
        for method in methods:
            overrides = METHODS[method]
            t0 = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                result = probe.run(make_args(matrix, args, overrides))
            elapsed = time.time() - t0
            print(
                f"{matrix}\t{method}\t"
                f"{result['mean_align']:.6f}\t{result['mean_relerr_sval']:.8f}\t{elapsed:.3f}"
            )


if __name__ == "__main__":
    main()
