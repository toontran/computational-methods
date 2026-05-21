"""Small-config driver for the combined and hybrid restricted-space methods.

This is the test_matrices_small analogue of run_joint_warmstart_benchmarks.py.
It runs a small sweep of (matrix, win, rank) configurations through
cex_restricted_space_probe.run() with score_variant='combined' and
basis_selection='greedy'. Two methods are exposed:

    combined  - full rank-r basis from combined-score greedy (no combined_rank override).
    hybrid    - first ceil(combined_rank_ratio*r) directions from combined-score greedy;
                remainder from top right singular vectors of M_gain restricted to the
                orthogonal complement of the greedy part (combined_rank_ratio=0.5 by default,
                which at rank=2 gives r1=1, r2=1).

The CLI defaults follow the medium budget: q0=8, qmax=48, num_restarts=8,
maxit=120, expansion_maxit=8, post_expansion_maxit=80, dtype=float32,
old_memory_size=win. Row-normalized SVD seeding is enabled on the first
block and every subsequent block; joint warm starts and oracle checks
are disabled. Diagnostic dumps (score components, oracle old-row
responses, debug_mode) are exposed as CLI flags for summary-style runs.
"""

import argparse
import contextlib
import io
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


_HERE = Path(__file__).resolve().parent
_PROBE_DIR = _HERE.parent / "test_matrices_fast"
if str(_PROBE_DIR) not in sys.path:
    sys.path.insert(0, str(_PROBE_DIR))

import cex_restricted_space_probe as probe  # noqa: E402


DEFAULT_MATRICES = [
    "static-cex",
    "diffuse-diffuse",
    "residual-spiky-shocks",
    "crowded-strategy",
    "macro-factor-panel",
]

DEFAULT_WINS = [32]
DEFAULT_RANKS = [2]


METHODS = {
    "combined": {
        "combined_rank_ratio": None,
    },
    "hybrid": {
        "combined_rank_ratio": 0.5,
    },
}


def _resolve_combined_rank(overrides, rank):
    if "combined_rank" in overrides and overrides["combined_rank"] is not None:
        return int(overrides["combined_rank"])
    ratio = overrides.get("combined_rank_ratio")
    if ratio is None:
        return None
    return max(0, min(int(rank), int(round(float(ratio) * int(rank)))))


def make_args(matrix, win, rank, args, overrides):
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
        mode="restricted",
        rank=int(rank),
        win=int(win),
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
        basis_selection="greedy",
        joint_warm_start_greedy=False,
        joint_warm_start_oracle=False,
        rownorm_seed_first_block=True,
        rownorm_seed_all_blocks=True,
        joint_warm_start_rotations=0,
        joint_warm_start_rotation_angle=0.7853981633974483,
        joint_warm_start_perturbations=0,
        joint_warm_start_perturb_scale=1e-2,
        joint_default_svd_start=True,
        joint_oversample=0,
        joint_oversample_rotate="svd",
        joint_solver="riemannian",
        row_concentration_lambda=0.0,
        row_leverage_lambda=0.0,
        row_leverage_mode="none",
        row_leverage_rank=2,
        score_variant="combined",
        old_memory_size=args.old_memory_size if args.old_memory_size is not None else int(win),
        debug_mode=args.debug_mode,
        combined_rank=_resolve_combined_rank(overrides, rank),
        oracle_candidate_check=False,
        oracle_sketch_all_seen_rows=False,
        dump_score_components=args.dump_score_components,
        dump_oracle_old_row_responses=args.dump_oracle_old_row_responses,
        dump_oracle_old_row_response_block=args.dump_oracle_old_row_response_block,
        dtype=args.dtype,
        expansion_direction="residual",
        reuse_line_search_grad=True,
        expansion_warm_start=True,
        post_expansion_maxit=args.post_expansion_maxit,
        benchmark_output=None,
        benchmark_append=False,
        verbose=args.verbose,
    )


def _parse_int_list(text, name):
    vals = [x.strip() for x in text.split(",") if x.strip()]
    try:
        return [int(v) for v in vals]
    except ValueError as exc:
        raise ValueError(f"Invalid integer in --{name}: {text!r}") from exc


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Small-config restricted-space sweep for combined and hybrid methods. "
            "Runs a few (matrix, win, rank) triples through the probe with medium-budget "
            "defaults and supports summary/dump-style diagnostics."
        )
    )
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=2,
                        help="Default target rank (used when --ranks is not given).")
    parser.add_argument("--win", type=int, default=32,
                        help="Default block size (used when --wins is not given).")
    parser.add_argument(
        "--ranks",
        help="Comma-separated target ranks to sweep. Overrides --rank if given.",
    )
    parser.add_argument(
        "--wins",
        help="Comma-separated block sizes to sweep. Overrides --win if given.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shuffle-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply a deterministic row permutation after matrix generation (default: on).",
    )
    parser.add_argument(
        "--row-shuffle-seed",
        type=int,
        default=0,
        help="Seed for --shuffle-rows. Defaults to 0.",
    )
    parser.add_argument("--matrix-preset", default="fast",
                        help="Preset passed to generate_matrix_input (default: fast).")

    # Medium-budget optimizer defaults.
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
    parser.add_argument(
        "--old-memory-size",
        type=int,
        default=None,
        help="Row-memory size. Defaults to --win for each run.",
    )

    # Diagnostic knobs.
    parser.add_argument(
        "--debug-mode",
        choices=("off", "combined", "summary"),
        default="off",
        help="Bundled diagnostic workflow: 'summary' enables oracle dumps for every block.",
    )
    parser.add_argument(
        "--dump-score-components",
        action="store_true",
        help="Print score-component diagnostics for each block.",
    )
    parser.add_argument(
        "--dump-oracle-old-row-responses",
        action="store_true",
        help="Print old_row_memory responses to oracle-projected directions.",
    )
    parser.add_argument(
        "--dump-oracle-old-row-response-block",
        type=int,
        default=0,
        help="1-based block index for old-row response dumps; 0 means every block.",
    )

    parser.add_argument(
        "--matrices",
        default=",".join(DEFAULT_MATRICES),
        help="Comma-separated matrix families to run.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="Comma-separated method names; choose from: " + ", ".join(METHODS),
    )
    parser.add_argument(
        "--benchmark-output",
        help="Optional tab-separated benchmark file; one row per (matrix, method, win, rank).",
    )
    parser.add_argument(
        "--benchmark-append",
        action="store_true",
        help="Append to --benchmark-output instead of overwriting.",
    )
    parser.add_argument(
        "--show-run-output",
        action="store_true",
        help="Stream the probe's stdout for each run (otherwise it is captured).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass --verbose through to the probe.",
    )
    args = parser.parse_args()

    matrices = [x.strip() for x in args.matrices.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    unknown_methods = sorted(set(methods) - set(METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown methods: {', '.join(unknown_methods)}")

    wins = _parse_int_list(args.wins, "wins") if args.wins else [int(args.win)]
    ranks = _parse_int_list(args.ranks, "ranks") if args.ranks else [int(args.rank)]

    # 'summary'/'combined' debug_mode implies diagnostic dumps.
    if args.debug_mode in {"summary", "combined"}:
        args.dump_score_components = True
        args.dump_oracle_old_row_responses = True
        if args.dump_oracle_old_row_response_block == 0:
            args.dump_oracle_old_row_response_block = 0

    print(
        f"# n={args.n}, matrix_preset={args.matrix_preset}, "
        f"seed={args.seed}, shuffle_rows={args.shuffle_rows}, "
        f"row_shuffle_seed={args.seed if args.row_shuffle_seed is None else args.row_shuffle_seed}"
    )
    print(
        "# medium params: "
        f"q0={args.q0}, qmax={args.qmax}, restarts={args.num_restarts}, "
        f"maxit={args.maxit}, expansion_maxit={args.expansion_maxit}, "
        f"post_expansion_maxit={args.post_expansion_maxit}, dtype={args.dtype}, "
        f"debug_mode={args.debug_mode}, "
        f"dump_score_components={args.dump_score_components}, "
        f"dump_oracle_old_row_responses={args.dump_oracle_old_row_responses}"
    )
    print(f"# wins={wins}, ranks={ranks}")
    print("matrix\tmethod\twin\trank\tmean_align\tmean_relerr_sval\telapsed")

    rows = []
    for matrix in matrices:
        for win in wins:
            for rank in ranks:
                for method in methods:
                    overrides = METHODS[method]
                    run_args = make_args(matrix, win, rank, args, overrides)
                    t0 = time.time()
                    if args.show_run_output or args.verbose:
                        result = probe.run(run_args)
                    else:
                        with contextlib.redirect_stdout(io.StringIO()):
                            result = probe.run(run_args)
                    elapsed = time.time() - t0
                    row = (
                        matrix,
                        method,
                        int(win),
                        int(rank),
                        float(result["mean_align"]),
                        float(result["mean_relerr_sval"]),
                        float(elapsed),
                    )
                    rows.append(row)
                    print(
                        f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t"
                        f"{row[4]:.6f}\t{row[5]:.8f}\t{row[6]:.3f}"
                    )

    if args.benchmark_output:
        output = Path(args.benchmark_output)
        if not output.is_absolute():
            output = _HERE / output
        output.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.benchmark_append else "w"
        with output.open(mode, encoding="utf-8") as f:
            if not args.benchmark_append:
                f.write("matrix\tmethod\twin\trank\tmean_align\tmean_relerr_sval\telapsed\n")
            for row in rows:
                f.write(
                    f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t"
                    f"{row[4]:.6f}\t{row[5]:.8f}\t{row[6]:.3f}\n"
                )
        print(f"# wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
