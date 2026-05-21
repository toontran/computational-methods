import argparse
import json
import os
import time
from itertools import combinations

import numpy as np

import cex_restricted_space_probe as base


MATRIX_CHOICES = (
    "alternative-data-signals",
    "static-cex",
    "crowded-strategy",
    "execution-cost-slippage",
    "etf-basket-basis",
    "futures-term-structure",
    "intraday-liquidity-shape",
    "macro-factor-panel",
    "rates-cross-currency",
    "options-vol-surface",
    "realized-vol-corr",
    "risk-residual-panel",
    "stat-arb-spreads",
)


def parse_angle_grid(angle_grid):
    if isinstance(angle_grid, str):
        vals = [float(tok.strip()) for tok in angle_grid.split(",") if tok.strip()]
    else:
        vals = [float(v) for v in angle_grid]
    if not vals:
        raise ValueError("Angle grid cannot be empty.")
    return tuple(vals)


def rotate_pair(V, i, j, theta_deg):
    V_rot = np.array(V, copy=True)
    theta = np.deg2rad(float(theta_deg))
    c = np.cos(theta)
    s = np.sin(theta)
    zi = np.array(V[:, i], copy=True)
    zj = np.array(V[:, j], copy=True)
    V_rot[:, i] = c * zi + s * zj
    V_rot[:, j] = -s * zi + c * zj
    return np.ascontiguousarray(V_rot, dtype=V.dtype)


def basis_score_details(M_gain, A_block, state_prev, rows_ref, V):
    V_work = np.ascontiguousarray(V)
    q = V_work.shape[1]
    scores = np.zeros(q, dtype=float)
    s_vals = np.zeros(q, dtype=float)
    H_vals = np.zeros(q, dtype=float)

    if state_prev is None:
        B = np.ascontiguousarray(A_block @ V_work, dtype=V_work.dtype)
        for k in range(q):
            z = np.zeros(q, dtype=V_work.dtype)
            z[k] = 1.0
            score, _, sval, hval = base.entropyscore_forget_score_grad_reduced(
                B=B,
                z=z,
                rows_block=A_block.shape[0],
                rows_ref=rows_ref,
                c_scale=1.0,
            )
            scores[k] = float(score)
            s_vals[k] = float(sval)
            H_vals[k] = float(hval)
    else:
        prev_basis = np.ascontiguousarray(np.asarray(state_prev["V"], dtype=V_work.dtype))
        prev_s2 = np.asarray(state_prev["s2"], dtype=V_work.dtype)
        B_gain = np.ascontiguousarray(M_gain @ V_work, dtype=V_work.dtype)
        B_block = np.ascontiguousarray(A_block @ V_work, dtype=V_work.dtype)
        C_prev = np.ascontiguousarray(prev_basis.T @ V_work, dtype=V_work.dtype)
        for k in range(q):
            z = np.zeros(q, dtype=V_work.dtype)
            z[k] = 1.0
            score, _, sval, hval = base.entropyscore_forget_streaming_score_grad_reduced(
                B_gain=B_gain,
                B_block=B_block,
                C_prev=C_prev,
                s2_old=prev_s2,
                z=z,
                rows_block=A_block.shape[0],
                rows_ref=rows_ref,
                c_scale=1.0,
            )
            scores[k] = float(score)
            s_vals[k] = float(sval)
            H_vals[k] = float(hval)

    return {
        "scores": scores,
        "s": s_vals,
        "H": H_vals,
        "total_score": float(np.sum(scores)),
    }


def candidate_pairs(rank, scores, grad_perp_ratio, mode, pair_window, max_pairs):
    limit = min(int(rank), max(2, int(pair_window)))
    pairs = list(combinations(range(limit), 2))

    if mode == "adjacent":
        pairs = [(i, i + 1) for i in range(limit - 1)]
    elif mode == "unstable":
        if grad_perp_ratio is not None and len(grad_perp_ratio) >= limit:
            pairs.sort(key=lambda ij: -(float(grad_perp_ratio[ij[0]]) + float(grad_perp_ratio[ij[1]])))
        else:
            pairs.sort(key=lambda ij: abs(float(scores[ij[0]]) - float(scores[ij[1]])))
    elif mode == "top":
        pass
    elif mode == "all":
        pairs = list(combinations(range(int(rank)), 2))
    else:
        raise ValueError(f"Unknown pair mode: {mode}")

    return pairs[: max(0, int(max_pairs))]


def optimizer_kwargs(args, work_dtype, reopt=False):
    if reopt:
        num_restarts = args.escape_num_restarts
        maxit = args.escape_maxit
        warm_start_greedy = True
        warm_start_perturbations = 0
    else:
        num_restarts = args.num_restarts
        maxit = args.maxit
        warm_start_greedy = args.warm_start_greedy
        warm_start_perturbations = args.warm_start_perturbations

    return dict(
        q0=args.q0,
        qmax=args.qmax,
        krylov_depth=args.krylov_depth,
        residual_tol=args.residual_tol,
        expansion_maxit=args.expansion_maxit,
        num_restarts=num_restarts,
        maxit=maxit,
        tol=args.tol,
        verbose=args.verbose,
        reduced_optimizer=args.reduced_optimizer,
        work_dtype=work_dtype,
        expansion_direction=args.expansion_direction,
        reuse_line_search_grad=args.reuse_line_search_grad,
        expansion_warm_start=args.expansion_warm_start,
        post_expansion_maxit=args.post_expansion_maxit,
        warm_start_greedy=warm_start_greedy,
        warm_start_perturbations=warm_start_perturbations,
        warm_start_perturb_scale=args.warm_start_perturb_scale,
        continuation=args.continuation,
        continuation_schedule=args.continuation_schedule,
    )


def run_local_optimizer(args, M_gain, A_block, state_prev, rows_seen, rows_ref, V_init, work_dtype, reopt=False):
    return base.entropy_iter_basis_forget(
        M_gain=M_gain,
        active_r=args.rank,
        rows_ref=rows_ref,
        V_init=V_init,
        rng=np.random.default_rng(args.seed),
        state_prev=state_prev,
        A_block=A_block,
        rows_total=rows_seen,
        **optimizer_kwargs(args, work_dtype, reopt=reopt),
    )


def probe_pair_rotations(args, M_gain, A_block, state_prev, rows_seen, rows_ref, V_base, base_details, base_diag, work_dtype):
    pairs = candidate_pairs(
        rank=V_base.shape[1],
        scores=base_details["scores"],
        grad_perp_ratio=base_diag.get("grad_perp_ratio"),
        mode=args.pair_mode,
        pair_window=args.pair_window,
        max_pairs=args.max_pairs,
    )
    angles = parse_angle_grid(args.angle_grid)
    trials = []

    for i, j in pairs:
        for theta in angles:
            V_rot = rotate_pair(V_base, i, j, theta)
            details = basis_score_details(M_gain, A_block, state_prev, rows_ref, V_rot)
            trials.append(
                {
                    "pair": (int(i), int(j)),
                    "theta_deg": float(theta),
                    "pre_score": details["total_score"],
                    "pre_gain": details["total_score"] - base_details["total_score"],
                    "pre_scores": details["scores"],
                    "V_rot": V_rot,
                }
            )

    trials.sort(key=lambda item: item["pre_gain"], reverse=True)
    promising = [item for item in trials if item["pre_gain"] >= args.min_pre_gain]
    if not promising and args.try_best_even_if_negative and trials:
        promising = [trials[0]]
    promising = promising[: max(0, int(args.keep_top))]

    accepted = None
    reopt_trials = []
    best_total = base_details["total_score"]

    for item in promising:
        V_opt, s_opt, H_opt, score_opt, diag_opt = run_local_optimizer(
            args=args,
            M_gain=M_gain,
            A_block=A_block,
            state_prev=state_prev,
            rows_seen=rows_seen,
            rows_ref=rows_ref,
            V_init=item["V_rot"],
            work_dtype=work_dtype,
            reopt=True,
        )
        total_opt = float(np.sum(score_opt))
        record = {
            "pair": item["pair"],
            "theta_deg": item["theta_deg"],
            "pre_score": item["pre_score"],
            "pre_gain": item["pre_gain"],
            "post_score": total_opt,
            "post_gain": total_opt - base_details["total_score"],
            "scores": np.asarray(score_opt, dtype=float),
            "s": np.asarray(s_opt, dtype=float),
            "H": np.asarray(H_opt, dtype=float),
            "diag": diag_opt,
            "V": V_opt,
        }
        reopt_trials.append(record)
        if total_opt > best_total + args.accept_tol:
            best_total = total_opt
            accepted = record

    return {
        "pairs": pairs,
        "angles": angles,
        "trials": trials,
        "promising": promising,
        "reopt_trials": reopt_trials,
        "accepted": accepted,
    }


def compact_rotation_record(record):
    return {
        "pair": list(record["pair"]),
        "theta_deg": float(record["theta_deg"]),
        "pre_score": float(record["pre_score"]),
        "pre_gain": float(record["pre_gain"]),
        "post_score": float(record["post_score"]),
        "post_gain": float(record["post_gain"]),
        "scores": np.asarray(record["scores"], dtype=float).tolist(),
        "s": np.asarray(record["s"], dtype=float).tolist(),
        "H": np.asarray(record["H"], dtype=float).tolist(),
    }


def update_streaming_state(args, M_gain, V_score, H_score, score_score, rows_seen, work_dtype):
    if args.carry == "left":
        _, s_new, Vt_new, _ = base.left_projected_operator_svd_factors(V_score.T, M_gain)
        V_r = Vt_new.T
    else:
        V_r, s_new = base.projected_subspace_svd(M_gain.astype(np.float64), V_score.astype(np.float64))
        s_new = s_new.astype(work_dtype, copy=False)
        V_r = V_r.astype(work_dtype, copy=False)
    return {
        "V": V_r,
        "s": s_new,
        "s2": s_new ** 2,
        "H": np.asarray(H_score[: len(s_new)], dtype=work_dtype),
        "score": np.asarray(score_score[: len(s_new)], dtype=work_dtype),
        "rows_seen": rows_seen,
    }, V_r, s_new


def load_input(args):
    if args.mat_input:
        A, V_exact, _, sigma1 = base.load_matlab_cex_input(args.mat_input)
        source_desc = args.mat_input
    else:
        A, V_exact, _, sigma1 = base.generate_matrix_input(
            matrix=args.matrix,
            n=args.n,
            preset=args.preset,
            seed=args.seed,
            r_sig=args.r_sig,
            alpha_sig=args.alpha_sig,
            alpha_tail=args.alpha_tail,
            tail_scale=args.tail_scale,
            sigma1=args.sigma1,
            v_type=args.v_type,
        )
        source_desc = f"generated {args.matrix} (n={args.n}, preset={args.preset}, seed={args.seed})"
    A = np.asarray(A, dtype=np.float64)
    if args.normalize_by_sigma:
        A = A / sigma1
    return A, V_exact, sigma1, source_desc


def run(args):
    np.random.seed(args.seed)
    t0 = time.time()
    A, V_exact, sigma1, source_desc = load_input(args)
    n = A.shape[0]
    work_dtype = np.float64 if args.dtype == "float64" else np.float32
    state = None
    V_r = None
    all_blocks = []

    print(f"Input: {source_desc}: A={A.shape}, sigma1={sigma1:.12g}, normalize_by_sigma={args.normalize_by_sigma}")
    print(
        "Pair-rotation escape params: "
        f"rank={args.rank}, win={args.win}, pair_mode={args.pair_mode}, pair_window={args.pair_window}, "
        f"max_pairs={args.max_pairs}, angles={parse_angle_grid(args.angle_grid)}, "
        f"keep_top={args.keep_top}, min_pre_gain={args.min_pre_gain}, "
        f"escape_num_restarts={args.escape_num_restarts}, escape_maxit={args.escape_maxit}"
    )

    for start0 in range(0, n, args.win):
        end0 = min(start0 + args.win, n)
        A_block = A[start0:end0, :]
        A_block_work = A_block.astype(work_dtype, copy=False)

        if state is None:
            M_gain = A_block_work
            V_init = None
            rows_seen = A_block.shape[0]
            print(f"\n===== block rows {start0 + 1}:{end0} (initial restricted score) =====")
        else:
            B_top = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
            M_gain = np.vstack([B_top, A_block_work]).astype(work_dtype, copy=False)
            V_init = state["V"].astype(work_dtype, copy=False)
            rows_seen = state["rows_seen"] + A_block.shape[0]
            print(f"\n===== block rows {start0 + 1}:{end0} (streaming restricted score) =====")

        V_score, s_score, H_score, score_score, diag = run_local_optimizer(
            args=args,
            M_gain=M_gain,
            A_block=A_block_work,
            state_prev=state,
            rows_seen=rows_seen,
            rows_ref=n,
            V_init=V_init,
            work_dtype=work_dtype,
            reopt=False,
        )

        base_details = basis_score_details(M_gain, A_block_work, state, n, V_score)
        escape = probe_pair_rotations(
            args=args,
            M_gain=M_gain,
            A_block=A_block_work,
            state_prev=state,
            rows_seen=rows_seen,
            rows_ref=n,
            V_base=V_score,
            base_details=base_details,
            base_diag=diag,
            work_dtype=work_dtype,
        )

        accepted = escape["accepted"]
        if accepted is not None:
            V_score = accepted["V"]
            s_score = accepted["s"]
            H_score = accepted["H"]
            score_score = accepted["scores"]
            diag = accepted["diag"]

        state, V_r, s_new = update_streaming_state(args, M_gain, V_score, H_score, score_score, rows_seen, work_dtype)
        state["diag"] = diag

        print(f"rows {start0 + 1}:{end0}")
        print(f"base_scores: {base.fmt_row(base_details['scores'])}")
        print(f"base_total_score: {base_details['total_score']:.6f}")
        print(f"best_pre_rotation_gain: {escape['trials'][0]['pre_gain']:.6g}" if escape["trials"] else "best_pre_rotation_gain: n/a")
        for record in escape["reopt_trials"]:
            print(
                "rotation_reopt: "
                f"pair={record['pair']}, theta={record['theta_deg']:.6g}, "
                f"pre_gain={record['pre_gain']:.6g}, post_gain={record['post_gain']:.6g}"
            )
        if accepted is None:
            print("accepted_rotation: none")
        else:
            print(
                "accepted_rotation: "
                f"pair={accepted['pair']}, theta={accepted['theta_deg']:.6g}, "
                f"post_total_score={accepted['post_score']:.6f}, post_gain={accepted['post_gain']:.6g}"
            )
        print(f"carried_s: {base.fmt_row(s_new)}")

        block_result = {
            "rows": [int(start0 + 1), int(end0)],
            "base_total_score": float(base_details["total_score"]),
            "base_scores": base_details["scores"].tolist(),
            "best_pre_rotation": None if not escape["trials"] else {
                "pair": list(escape["trials"][0]["pair"]),
                "theta_deg": float(escape["trials"][0]["theta_deg"]),
                "pre_score": float(escape["trials"][0]["pre_score"]),
                "pre_gain": float(escape["trials"][0]["pre_gain"]),
            },
            "reopt_trials": [compact_rotation_record(record) for record in escape["reopt_trials"]],
            "accepted_rotation": None if accepted is None else compact_rotation_record(accepted),
            "carried_s": np.asarray(s_new, dtype=float).tolist(),
        }
        all_blocks.append(block_result)

    align = np.linalg.norm((V_r @ V_r.T) @ V_exact[:, :1], "fro")
    top_sval_est = float(state["s"][0])
    rel_err_sval = abs(top_sval_est - sigma1) / sigma1
    elapsed = time.time() - t0
    print("sigma1    mean_align    mean_relerr_sval    elapsed")
    print(f"{sigma1:.3f}      {align:.6f}           {rel_err_sval:.8f}          {elapsed:.3f}")

    result = {
        "matrix": args.matrix if not args.mat_input else os.path.basename(args.mat_input),
        "method": "pair_rotation_escape",
        "mean_align": float(align),
        "mean_relerr_sval": float(rel_err_sval),
        "elapsed": float(elapsed),
        "blocks": all_blocks,
    }
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe pairwise column-rotation escapes for the restricted-space entropy-forget optimizer."
    )
    parser.add_argument("--mat-input")
    parser.add_argument("--matrix", choices=MATRIX_CHOICES, default="static-cex")
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--r-sig", type=int, default=2)
    parser.add_argument("--alpha-sig", type=float, default=0.003)
    parser.add_argument("--alpha-tail", type=float, default=0.0145)
    parser.add_argument("--tail-scale", type=float, default=0.99)
    parser.add_argument("--sigma1", type=float, default=0.991)
    parser.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--win", type=int, default=100)
    parser.add_argument("--preset", choices=sorted(base.PRESETS), default="fast")
    parser.add_argument("--cex-replicate", action="store_true")
    parser.add_argument("--q0", type=int)
    parser.add_argument("--qmax", type=int)
    parser.add_argument("--krylov-depth", type=int)
    parser.add_argument("--residual-tol", type=float)
    parser.add_argument("--expansion-maxit", type=int)
    parser.add_argument("--num-restarts", type=int)
    parser.add_argument("--maxit", type=int)
    parser.add_argument("--tol", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normalize-by-sigma", action="store_true")
    parser.add_argument("--carry", choices=("left", "right"))
    parser.add_argument("--reduced-optimizer", choices=("legacy", "cex"))
    parser.add_argument("--dtype", choices=("float32", "float64"))
    parser.add_argument("--expansion-direction", choices=("krylov_v", "residual"))
    parser.add_argument("--reuse-line-search-grad", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--expansion-warm-start", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--post-expansion-maxit", type=int)
    parser.add_argument("--warm-start-greedy", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--warm-start-perturbations", type=int)
    parser.add_argument("--warm-start-perturb-scale", type=float)
    parser.add_argument("--continuation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--continuation-schedule")
    parser.add_argument("--pair-mode", choices=("top", "adjacent", "unstable", "all"), default="top")
    parser.add_argument("--pair-window", type=int, default=6)
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--angle-grid", default="-20,-10,-3,-1,1,3,10,20")
    parser.add_argument("--keep-top", type=int, default=3)
    parser.add_argument("--min-pre-gain", type=float, default=0.0)
    parser.add_argument("--try-best-even-if-negative", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--accept-tol", type=float, default=1e-12)
    parser.add_argument("--escape-num-restarts", type=int, default=1)
    parser.add_argument("--escape-maxit", type=int, default=60)
    parser.add_argument("--json-output")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.cex_replicate:
        args.preset = "cex-replicate"

    preset_values = base.PRESETS[args.preset]
    for name, value in preset_values.items():
        if hasattr(args, name) and getattr(args, name) is None:
            setattr(args, name, value)

    if args.warm_start_greedy is None:
        args.warm_start_greedy = True
    args.continuation_schedule = base.parse_continuation_schedule(args.continuation_schedule)
    args.angle_grid = parse_angle_grid(args.angle_grid)
    return args


if __name__ == "__main__":
    run(parse_args())
