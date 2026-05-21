import argparse
import csv
import json
import time

import numpy as np

from cex_joint_diag_math import (
    complete_from_first,
    coupling_matrix,
    curvature_probe,
    optimize_joint,
    perturbation_probe,
    principal_angles,
    random_stiefel,
    reduced_score_grad,
    stiefel_retract,
)
from cex_restricted_space_probe import (
    PRESETS,
    entropy_iter_basis_forget,
    generate_structured_cex_input,
    left_projected_operator_svd_factors,
    load_matlab_cex_input,
    orthonormalize_columns,
    projected_subspace_svd,
)


def fmt(x, nd=12):
    return f"{float(x):.{nd}g}"


def make_problem(M_gain, A_block, Vbasis, state_prev, rows_ref, dtype):
    B_gain = np.ascontiguousarray(M_gain @ Vbasis, dtype=dtype)
    B_block = np.ascontiguousarray(A_block @ Vbasis, dtype=dtype)
    C_prev = None
    s2_old = None
    if state_prev is not None:
        C_prev = np.ascontiguousarray(state_prev["V"].astype(dtype, copy=False).T @ Vbasis, dtype=dtype)
        s2_old = np.asarray(state_prev["s2"], dtype=dtype)
    return {
        "B_gain": B_gain,
        "B_block": B_block,
        "C_prev": C_prev,
        "s2_old": s2_old,
        "state_prev": state_prev,
        "rows_block": int(A_block.shape[0]),
        "rows_ref": int(rows_ref),
        "Qz": np.zeros((Vbasis.shape[1], 0), dtype=dtype),
        "dtype": np.dtype(dtype),
    }


def greedy_reduced_frame(Vbasis, V_score, r, dtype):
    Z = np.ascontiguousarray(Vbasis.T @ np.asarray(V_score[:, :r], dtype=dtype), dtype=dtype)
    Z = stiefel_retract(Z, None)
    if Z is None:
        raise RuntimeError("Could not map greedy output into final restricted basis.")
    return Z


def projected_true_frame(Vbasis, V_exact, r, dtype):
    if V_exact is None:
        return None
    Z = np.ascontiguousarray(Vbasis.T @ np.asarray(V_exact[:, :r], dtype=dtype), dtype=dtype)
    return stiefel_retract(Z, None)


def summarize_history(history):
    if not history:
        return {"iters": 0, "min_alpha": np.nan, "last_alpha": np.nan, "rejects": 0, "last_improvement": np.nan}
    alphas = [h["alpha"] for h in history]
    return {
        "iters": int(len(history)),
        "min_alpha": float(np.min(alphas)),
        "last_alpha": float(alphas[-1]),
        "rejects": int(sum(h["rejected"] for h in history)),
        "last_improvement": float(history[-1]["improvement"]),
    }


def run(args):
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    if args.mat_input:
        A, V_exact, _, sigma1 = load_matlab_cex_input(args.mat_input)
        source = args.mat_input
    else:
        A, V_exact, _, sigma1 = generate_structured_cex_input(
            n=args.n,
            r_sig=args.r_sig,
            alpha_sig=args.alpha_sig,
            alpha_tail=args.alpha_tail,
            tail_scale=args.tail_scale,
            sigma1=args.sigma1,
            v_type=args.v_type,
        )
        source = f"generated n={args.n} r_sig={args.r_sig} v_type={args.v_type}"
    A = np.asarray(A, dtype=np.float64)
    if args.normalize_by_sigma:
        A = A / sigma1

    preset = dict(PRESETS[args.preset])
    for name in (
        "q0",
        "qmax",
        "krylov_depth",
        "residual_tol",
        "expansion_maxit",
        "num_restarts",
        "maxit",
        "tol",
        "carry",
        "reduced_optimizer",
        "dtype",
        "expansion_direction",
        "reuse_line_search_grad",
        "expansion_warm_start",
        "post_expansion_maxit",
    ):
        value = getattr(args, name)
        if value is not None:
            preset[name] = value

    dtype = np.float64 if preset["dtype"] == "float64" else np.float32
    r = args.rank
    n = A.shape[0]
    state = None
    rows = []

    print(f"Input: {source}; A={A.shape}; normalize_by_sigma={args.normalize_by_sigma}")
    print(f"Preset: {args.preset}; rank={r}; win={args.win}; dtype={np.dtype(dtype)}")
    print(f"Joint diagnostics: multistarts={args.joint_starts}; joint_maxit={args.joint_maxit}; seed={args.seed}")

    for block_idx, start0 in enumerate(range(0, n, args.win), start=1):
        end0 = min(start0 + args.win, n)
        A_block = np.asarray(A[start0:end0, :], dtype=dtype)
        if state is None:
            M_gain = A_block
            V_init = None
            rows_seen = A_block.shape[0]
        else:
            B_top = state["s"].astype(dtype)[:, None] * state["V"].astype(dtype).T
            M_gain = np.vstack([B_top, A_block]).astype(dtype, copy=False)
            V_init = state["V"].astype(dtype, copy=False)
            rows_seen = state["rows_seen"] + A_block.shape[0]

        V_score, s_score, H_score, score_score, diag = entropy_iter_basis_forget(
            M_gain=M_gain,
            active_r=r,
            rows_ref=n,
            V_init=V_init,
            q0=preset["q0"],
            qmax=preset["qmax"],
            krylov_depth=preset["krylov_depth"],
            residual_tol=preset["residual_tol"],
            expansion_maxit=preset["expansion_maxit"],
            num_restarts=preset["num_restarts"],
            maxit=preset["maxit"],
            tol=preset["tol"],
            rng=np.random.default_rng(args.seed),
            verbose=False,
            state_prev=state,
            A_block=A_block,
            rows_total=rows_seen,
            reduced_optimizer=preset["reduced_optimizer"],
            work_dtype=dtype,
            expansion_direction=preset["expansion_direction"],
            reuse_line_search_grad=preset["reuse_line_search_grad"],
            expansion_warm_start=preset["expansion_warm_start"],
            post_expansion_maxit=preset["post_expansion_maxit"],
        )

        Vbasis = np.asarray(diag["Vbasis_final"], dtype=dtype)
        problem = make_problem(M_gain, A_block, Vbasis, state, n, dtype)
        Z_greedy = greedy_reduced_frame(Vbasis, V_score, r, dtype)
        F_greedy, _, G_greedy, _, _ = reduced_score_grad(problem, Z_greedy)

        joint_runs = []
        for k in range(args.joint_starts):
            Z0 = random_stiefel(Vbasis.shape[1], r, rng, dtype=dtype)
            result = optimize_joint(problem, Z0, maxit=args.joint_maxit, tol=args.joint_tol)
            result["label"] = f"random_{k + 1}"
            joint_runs.append(result)

        greedy_init = optimize_joint(problem, Z_greedy, maxit=args.joint_maxit, tol=args.joint_tol)
        greedy_init["label"] = "greedy_frame"
        joint_runs.append(greedy_init)

        Z_cont = complete_from_first(Z_greedy[:, 0], r, rng)
        cont = optimize_joint(problem, Z_cont, maxit=args.joint_maxit, tol=args.joint_tol)
        cont["label"] = "greedy_first_completion"
        joint_runs.append(cont)

        best = max(joint_runs, key=lambda item: item["F"])
        random_runs = [item for item in joint_runs if item["label"].startswith("random_")]
        best_random = max(random_runs, key=lambda item: item["F"]) if random_runs else None
        spread = max(item["F"] for item in joint_runs) - min(item["F"] for item in joint_runs)
        hist = summarize_history(best["history"])
        angles = principal_angles(best["Z"], Z_greedy)
        C = coupling_matrix(problem, best["Z"])
        offdiag = C - np.diag(np.diag(C))
        coupling_ratio = float(np.linalg.norm(offdiag, "fro") / max(np.linalg.norm(np.diag(np.diag(C)), "fro"), 1e-30))
        perturb = perturbation_probe(problem, best["Z"], args.probe_eps, args.probe_trials, rng)
        curve = curvature_probe(problem, best["Z"], args.probe_eps, args.probe_trials, rng)

        oracle_angle_max = np.nan
        oracle_F = np.nan
        Z_true = projected_true_frame(Vbasis, V_exact, r, dtype)
        if Z_true is not None:
            oracle_F, _, _, _, _ = reduced_score_grad(problem, Z_true)
            true_angles = principal_angles(best["Z"], Z_true)
            if true_angles.size:
                oracle_angle_max = float(np.max(true_angles))

        row = {
            "block": block_idx,
            "rows": f"{start0 + 1}:{end0}",
            "q": int(Vbasis.shape[1]),
            "F_greedy": float(F_greedy),
            "F_joint_best": float(best["F"]),
            "F_joint_best_random": np.nan if best_random is None else float(best_random["F"]),
            "joint_minus_greedy": float(best["F"] - F_greedy),
            "best_random_minus_greedy": np.nan if best_random is None else float(best_random["F"] - F_greedy),
            "joint_spread": float(spread),
            "best_label": best["label"],
            "best_stop": best["stop"]["reason"],
            "best_grad_norm": float(best["grad_norm"]),
            "min_alpha": hist["min_alpha"],
            "last_alpha": hist["last_alpha"],
            "line_search_rejects": hist["rejects"],
            "last_improvement": hist["last_improvement"],
            "max_angle_to_greedy_deg": float(np.max(angles)) if angles.size else np.nan,
            "coupling_offdiag_ratio": coupling_ratio,
            "oracle_F": float(oracle_F),
            "max_angle_to_oracle_deg": oracle_angle_max,
            "perturb": perturb,
            "curvature": curve,
            "joint_values": [float(item["F"]) for item in joint_runs],
            "joint_stops": [item["stop"]["reason"] for item in joint_runs],
        }
        rows.append(row)

        verdict = "OK" if row["joint_minus_greedy"] >= -args.failure_tol else "FAIL"
        print(
            f"block {block_idx:02d} rows {row['rows']} q={row['q']} "
            f"F_greedy={fmt(row['F_greedy'])} F_joint={fmt(row['F_joint_best'])} "
            f"delta={fmt(row['joint_minus_greedy'])} {verdict} "
            f"best_random_delta={fmt(row['best_random_minus_greedy'])} "
            f"spread={fmt(row['joint_spread'], 6)} grad={fmt(row['best_grad_norm'], 6)} "
            f"alpha_min={fmt(row['min_alpha'], 4)} rejects={row['line_search_rejects']} "
            f"angle_greedy_max={fmt(row['max_angle_to_greedy_deg'], 6)} "
            f"coupling={fmt(row['coupling_offdiag_ratio'], 6)}"
        )
        if args.log_history:
            for item in joint_runs:
                print(f"  {item['label']}: F={fmt(item['F'])} stop={item['stop']} values={json.dumps(summarize_history(item['history']))}")
            print(f"  GtG_best={np.array2string(C, precision=6, suppress_small=False)}")
            print(f"  perturb={json.dumps(perturb)}")
            print(f"  curvature={json.dumps(curve)}")

        if preset["carry"] == "left":
            _, s_new, Vt_new, _ = left_projected_operator_svd_factors(V_score.T, M_gain)
            V_r = Vt_new.T
        else:
            V_r, s_new = projected_subspace_svd(M_gain.astype(np.float64), V_score.astype(np.float64))
            s_new = s_new.astype(dtype, copy=False)
            V_r = V_r.astype(dtype, copy=False)
        state = {
            "V": V_r,
            "s": s_new,
            "s2": s_new ** 2,
            "H": np.asarray(H_score[: len(s_new)], dtype=np.float32),
            "score": np.asarray(score_score[: len(s_new)], dtype=np.float32),
            "rows_seen": rows_seen,
            "diag": diag,
        }

    if args.csv:
        scalar_fields = [
            "block",
            "rows",
            "q",
            "F_greedy",
            "F_joint_best",
            "F_joint_best_random",
            "joint_minus_greedy",
            "best_random_minus_greedy",
            "joint_spread",
            "best_label",
            "best_stop",
            "best_grad_norm",
            "min_alpha",
            "last_alpha",
            "line_search_rejects",
            "last_improvement",
            "max_angle_to_greedy_deg",
            "coupling_offdiag_ratio",
            "oracle_F",
            "max_angle_to_oracle_deg",
        ]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=scalar_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row[name] for name in scalar_fields})
        print(f"Wrote scalar summary CSV: {args.csv}")

    print(f"elapsed={time.time() - t0:.3f}s")
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone diagnostics for restricted CEX joint score-sum optimization.")
    parser.add_argument("--mat-input")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--r-sig", type=int, default=2)
    parser.add_argument("--alpha-sig", type=float, default=0.003)
    parser.add_argument("--alpha-tail", type=float, default=0.0145)
    parser.add_argument("--tail-scale", type=float, default=0.99)
    parser.add_argument("--sigma1", type=float, default=0.991)
    parser.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    parser.add_argument("--normalize-by-sigma", action="store_true")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--win", type=int, default=32)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="cex-replicate")
    parser.add_argument("--q0", type=int)
    parser.add_argument("--qmax", type=int)
    parser.add_argument("--krylov-depth", type=int)
    parser.add_argument("--residual-tol", type=float)
    parser.add_argument("--expansion-maxit", type=int)
    parser.add_argument("--num-restarts", type=int)
    parser.add_argument("--maxit", type=int)
    parser.add_argument("--tol", type=float)
    parser.add_argument("--carry", choices=("left", "right"))
    parser.add_argument("--reduced-optimizer", choices=("legacy", "cex"))
    parser.add_argument("--dtype", choices=("float32", "float64"))
    parser.add_argument("--expansion-direction", choices=("krylov_v", "residual"))
    parser.add_argument("--reuse-line-search-grad", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--expansion-warm-start", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--post-expansion-maxit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--joint-starts", type=int, default=8)
    parser.add_argument("--joint-maxit", type=int, default=250)
    parser.add_argument("--joint-tol", type=float, default=1e-8)
    parser.add_argument("--failure-tol", type=float, default=1e-10)
    parser.add_argument("--probe-eps", type=float, nargs="+", default=[1e-3, 1e-2])
    parser.add_argument("--probe-trials", type=int, default=8)
    parser.add_argument("--log-history", action="store_true")
    parser.add_argument("--csv")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
