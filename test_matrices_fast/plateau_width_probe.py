"""Plateau-width quantifier (INFRA-05).

At a fixed (matrix, block, score variant), sample N random V's on the Stiefel
manifold St(d, r) within `score(V) >= tau * score(V_opt)` (default tau=0.95)
and report principal angles to V_oracle. The probe quantifies whether the
S6 (or any score variant) basin is narrow (cos² near 1, tight) or wide-but-
multimodal (cos² spread).

Outputs (under summary/infra_plateau_width/):
  {matrix}_b{block}_{variant}_r{rank}.csv      per-sample principal cos² + angles
  summary_plot.png                              boxplot of cos² distribution per
                                                matrix (terminal block)
  synthesis.md                                  written separately

For r=1 (default), Stiefel(d, 1) reduces to the sphere; the lone "principal
angle" is the scalar angle to V_oracle's first column.

Sampling: rejection on the score-restricted set. If acceptance rate falls
below --rejection-min-rate (default 0.01) on any matrix, we automatically fall
back to a random-walk MCMC on the sublevel set and flag the matrix in the
acceptance-rate column. Acceptance rate is reported per (matrix, block).

Entry point: python plateau_width_probe.py [...]

Backlog: INFRA-05 in summary/overview/score_family_workflow.txt §5.
Resolves: Q2 in summary/overview/score_design_overview.txt §7.
"""

import argparse
import csv
import json
import os
import time

import numpy as np

import cex_restricted_space_probe as probe
from future_hmean_optimizer_diagnostic import (
    orth_basis_against,
    rowspace_basis,
)
from hmean_evidence_score import per_block_constants, stream_to_block
from r_sk_g_score import (
    r_sk_g_value_grad,
    optimize_r_sk_g_in_basis,
)


# ---------------------------------------------------------------------------
# Score evaluation
# ---------------------------------------------------------------------------


def score_value(V, A_sketch, A_cur, A_fut, c_sk, variant, alpha, beta, gamma,
                V_state, cur_F2, fut_F2, sk_F2_low):
    """Score for an n-by-r frame V. Aggregates per-column scores by sum.

    For r=1 this collapses to the standard r_sk_g_value_grad return value.
    For r>1 this is a column-wise sum (compatible with the rank-r lift in
    score_design_overview.txt §5: HM3 of (||A_sk V||_F^2/sk_F2_low,
    ||A_cur V||_F^2/cur_F2, ||A_fut V||_F^2/fut_F2)). For S6 specifically the
    Frobenius variant differs — we compute it explicitly below to match the
    §5 form.
    """
    V = np.asarray(V, dtype=np.float64)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    eps = 1e-30
    if variant == "S6":
        # Match the §5 Stiefel-rank-r lift: HM3(F-norm^2 / total).
        if cur_F2 is None or fut_F2 is None:
            raise ValueError("S6 score needs cur_F2 and fut_F2")
        W_c = float(cur_F2)
        W_f = float(fut_F2)
        if W_c <= eps or W_f <= eps:
            return 0.0
        Y_c = A_cur @ V
        Y_f = A_fut @ V
        u_g1 = float(np.sum(Y_c * Y_c)) / W_c
        u_g2 = float(np.sum(Y_f * Y_f)) / W_f
        have_sketch = (
            A_sketch is not None
            and sk_F2_low is not None
            and float(sk_F2_low) > eps
            and np.asarray(A_sketch).size
        )
        if have_sketch:
            W_sk = float(sk_F2_low)
            Y_sk = np.asarray(A_sketch) @ V
            u_sk = float(np.sum(Y_sk * Y_sk)) / W_sk
            if u_sk <= eps or u_g1 <= eps or u_g2 <= eps:
                return 0.0
            D = 1.0 / u_sk + 1.0 / u_g1 + 1.0 / u_g2
            return 3.0 / D
        else:
            if u_g1 <= eps or u_g2 <= eps:
                return 0.0
            D2 = 1.0 / u_g1 + 1.0 / u_g2
            return 2.0 / D2
    if variant == "S6_GM":
        if cur_F2 is None or fut_F2 is None:
            raise ValueError("S6_GM score needs cur_F2 and fut_F2")
        W_c = float(cur_F2)
        W_f = float(fut_F2)
        if W_c <= eps or W_f <= eps:
            return 0.0
        Y_c = A_cur @ V
        Y_f = A_fut @ V
        u_g1 = float(np.sum(Y_c * Y_c)) / W_c
        u_g2 = float(np.sum(Y_f * Y_f)) / W_f
        have_sketch = (
            A_sketch is not None
            and sk_F2_low is not None
            and float(sk_F2_low) > eps
            and np.asarray(A_sketch).size
        )
        if have_sketch:
            W_sk = float(sk_F2_low)
            Y_sk = np.asarray(A_sketch) @ V
            u_sk = float(np.sum(Y_sk * Y_sk)) / W_sk
            if u_sk <= eps or u_g1 <= eps or u_g2 <= eps:
                return 0.0
            return float((u_sk * u_g1 * u_g2) ** (1.0 / 3.0))
        else:
            if u_g1 <= eps or u_g2 <= eps:
                return 0.0
            return float((u_g1 * u_g2) ** 0.5)
    # S1..S5 fall back to the column-wise sum of r_sk_g_value_grad for r>=1.
    total = 0.0
    for j in range(V.shape[1]):
        s, *_ = r_sk_g_value_grad(
            A_sketch, A_cur, A_fut, c_sk, V[:, j],
            variant=variant, alpha=alpha, beta=beta, gamma=gamma,
            V_state=V_state, cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
        )
        total += float(s)
    return total


# ---------------------------------------------------------------------------
# Stiefel sampling
# ---------------------------------------------------------------------------


def random_stiefel(d, r, rng):
    if r <= 0:
        return np.zeros((d, 0), dtype=np.float64)
    G = rng.standard_normal((d, r))
    Q, _ = np.linalg.qr(G)
    if Q.shape[1] < r:
        # extremely rare; pad if QR thinned
        return Q
    return np.ascontiguousarray(Q[:, :r], dtype=np.float64)


def random_stiefel_in_subspace(B, r, rng):
    """Random orthonormal frame in span(B) (B has orthonormal cols)."""
    q = B.shape[1]
    if q < r:
        return None
    Z = rng.standard_normal((q, r))
    Qz, _ = np.linalg.qr(Z)
    if Qz.shape[1] < r:
        return None
    return np.ascontiguousarray(B @ Qz[:, :r], dtype=np.float64)


# ---------------------------------------------------------------------------
# Principal angles / cos²
# ---------------------------------------------------------------------------


def principal_cos2_and_angles(V, V_oracle):
    """Return (cos2_array, angle_rad_array) of principal angles between
    span(V) and span(V_oracle). Length = min(r1, r2)."""
    V = np.asarray(V, dtype=np.float64)
    Vor = np.asarray(V_oracle, dtype=np.float64)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    if Vor.ndim == 1:
        Vor = Vor.reshape(-1, 1)
    if V.size == 0 or Vor.size == 0:
        return np.zeros(0), np.zeros(0)
    # Both should be orthonormal; orthonormalize defensively.
    Q1, _ = np.linalg.qr(V)
    Q2, _ = np.linalg.qr(Vor)
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    cos2 = s ** 2
    angles = np.arccos(s)
    return cos2, angles


# ---------------------------------------------------------------------------
# Per-snapshot probe
# ---------------------------------------------------------------------------


def probe_block(args, matrix, A, V_exact, snap, block_id):
    """Sample N random Stiefel frames; return per-sample records + summary."""
    rank = int(args.rank)
    half_win = int(args.half_win)
    A_cur = np.asarray(snap["A_cur"], dtype=np.float64)
    A_fut = np.asarray(snap["A_fut"], dtype=np.float64)
    A_sketch_arr = np.asarray(snap["A_sketch"], dtype=np.float64)
    A_sketch = A_sketch_arr if A_sketch_arr.size else None
    M_gain = np.asarray(snap["M_gain"], dtype=np.float64)
    state = snap["state"]
    V_default = snap["V_default"]
    diag = snap["diag"]

    consts = per_block_constants(A, block_id, half_win)
    c_sk = consts["c_sk"]
    cur_F2 = float(consts["cur_F2"])
    fut_F2 = float(consts["fut_F2"])
    if A_sketch is not None:
        sk_F2_low = float(np.sum(A_sketch ** 2))
    else:
        sk_F2_low = 0.0

    V_state = None
    if state is not None and state.get("V") is not None:
        Vs = np.asarray(state["V"], dtype=np.float64)
        if Vs.size:
            V_state = Vs

    variant = args.variant
    alpha, beta, gamma = float(args.alpha), float(args.beta), float(args.gamma)

    # Subspace where the sublevel set lives (union span = rowspan(A_sk + A_cur + A_fut)).
    if A_sketch is not None:
        union_stack = np.vstack([A_sketch, A_cur, A_fut])
    else:
        union_stack = np.vstack([A_cur, A_fut])
    B_union = rowspace_basis(union_stack)
    union_dim = int(B_union.shape[1])

    # Oracle subspace (top-rank columns of V_exact projected into B_union, then
    # orthonormalized). For rank=1 this is the projected leading right SV; for
    # rank>1 it's the projected oracle frame, deflated.
    Vex = np.asarray(V_exact, dtype=np.float64)[:, :rank]
    V_or_proj = B_union @ (B_union.T @ Vex)
    # Orthonormalize the projected oracle frame for principal-angle computation.
    Q_or, _ = np.linalg.qr(V_or_proj)
    # Trim if QR collapsed; principal_cos2 handles mismatched ranks.
    V_oracle = np.ascontiguousarray(Q_or[:, :rank], dtype=np.float64)

    # ---- Compute V_opt (optimizer's best frame inside the union) ----
    starts = [V_default[:, 0]]
    if V_default.shape[1] >= 2:
        starts.append(V_default[:, 1])
    if V_state is not None:
        for j in range(min(V_state.shape[1], 4)):
            starts.append(V_state[:, j])
    if not getattr(args, "no_oracle_warmstart", False):
        # Diagnostic-only oracle warmstart for V_opt; off by default per workflow §9.
        for j in range(rank):
            starts.append(Vex[:, j])

    rng_opt = np.random.default_rng(args.seed + 70001 + block_id)
    # Greedy rank-r: optimize v1 over B_union (variant), then deflate.
    V_opt_cols = []
    B_search = B_union
    for k in range(rank):
        rsk = optimize_r_sk_g_in_basis(
            A_cur, A_fut, A_sketch, c_sk,
            B_search,
            [s for s in starts if s is not None],
            rng_opt,
            args.union_maxit, args.union_tol, args.union_random_starts,
            variant=variant, alpha=alpha, beta=beta, gamma=gamma, V_state=V_state,
            cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
        )
        if rsk is None:
            break
        v_k = np.asarray(rsk["vec"], dtype=np.float64).reshape(-1)
        nrm = float(np.linalg.norm(v_k))
        if nrm <= 1e-30:
            break
        v_k = v_k / nrm
        V_opt_cols.append(v_k)
        if k + 1 < rank:
            B_search = orth_basis_against(B_search, v_k)
            if B_search.shape[1] < 1:
                break
    if not V_opt_cols:
        raise RuntimeError(f"optimizer returned no vector at b{block_id}")
    V_opt = np.column_stack(V_opt_cols)
    # Re-orthonormalize for safety.
    V_opt, _ = np.linalg.qr(V_opt)
    V_opt = np.ascontiguousarray(V_opt[:, :len(V_opt_cols)], dtype=np.float64)

    score_opt = score_value(V_opt, A_sketch, A_cur, A_fut, c_sk, variant,
                            alpha, beta, gamma, V_state, cur_F2, fut_F2, sk_F2_low)
    score_oracle = score_value(V_oracle, A_sketch, A_cur, A_fut, c_sk, variant,
                               alpha, beta, gamma, V_state, cur_F2, fut_F2, sk_F2_low)
    threshold = float(args.tau) * score_opt

    # ---- Sample N frames within {V : score(V) >= threshold} ----
    N_target = int(args.num_samples)
    rng_samp = np.random.default_rng(args.seed + 80003 + block_id)

    samples = []
    n_drawn = 0
    n_accepted = 0
    method = "rejection"

    # Cap rejection at this many draws; if we don't reach N, switch to MCMC.
    max_draws = int(args.max_rejection_draws)

    while n_accepted < N_target and n_drawn < max_draws:
        n_drawn += 1
        if args.sample_in_union:
            V_samp = random_stiefel_in_subspace(B_union, len(V_opt_cols), rng_samp)
        else:
            V_samp = random_stiefel(A.shape[1], len(V_opt_cols), rng_samp)
        if V_samp is None:
            continue
        sc = score_value(V_samp, A_sketch, A_cur, A_fut, c_sk, variant,
                         alpha, beta, gamma, V_state, cur_F2, fut_F2, sk_F2_low)
        if sc >= threshold:
            cos2, angles = principal_cos2_and_angles(V_samp, V_oracle)
            samples.append({
                "score": float(sc),
                "cos2": cos2.tolist(),
                "angles_rad": angles.tolist(),
                "max_angle_rad": float(np.max(angles)) if angles.size else float("nan"),
                "min_cos2": float(np.min(cos2)) if cos2.size else float("nan"),
                "mean_cos2": float(np.mean(cos2)) if cos2.size else float("nan"),
                "method": "rejection",
            })
            n_accepted += 1

    rejection_acceptance = n_accepted / max(n_drawn, 1)

    # MCMC fallback
    if n_accepted < N_target:
        method = "rejection+mcmc" if n_accepted > 0 else "mcmc"
        # Random-walk on Stiefel: perturb current accepted V by a small Gaussian
        # in the tangent space, retract by QR, accept if score >= threshold.
        if not samples:
            # Need an initial sample: V_opt itself is in the sublevel set.
            V_cur = V_opt.copy()
            sc_cur = score_opt
        else:
            last = samples[-1]
            V_cur = np.column_stack(
                [V_oracle[:, 0]]) if False else V_opt.copy()
            sc_cur = score_opt
        step_sigma = float(args.mcmc_step)
        n_mcmc_drawn = 0
        max_mcmc = int(args.max_mcmc_steps)
        while n_accepted < N_target and n_mcmc_drawn < max_mcmc:
            n_mcmc_drawn += 1
            G = rng_samp.standard_normal(V_cur.shape) * step_sigma
            V_prop = V_cur + G
            try:
                Q_prop, _ = np.linalg.qr(V_prop)
            except np.linalg.LinAlgError:
                continue
            V_prop = np.ascontiguousarray(Q_prop[:, :V_cur.shape[1]], dtype=np.float64)
            sc_prop = score_value(V_prop, A_sketch, A_cur, A_fut, c_sk, variant,
                                  alpha, beta, gamma, V_state, cur_F2, fut_F2, sk_F2_low)
            if sc_prop >= threshold:
                cos2, angles = principal_cos2_and_angles(V_prop, V_oracle)
                samples.append({
                    "score": float(sc_prop),
                    "cos2": cos2.tolist(),
                    "angles_rad": angles.tolist(),
                    "max_angle_rad": float(np.max(angles)) if angles.size else float("nan"),
                    "min_cos2": float(np.min(cos2)) if cos2.size else float("nan"),
                    "mean_cos2": float(np.mean(cos2)) if cos2.size else float("nan"),
                    "method": "mcmc",
                })
                n_accepted += 1
                V_cur = V_prop
                sc_cur = sc_prop
        n_drawn += n_mcmc_drawn
        rejection_acceptance_combined = n_accepted / max(n_drawn, 1)
    else:
        rejection_acceptance_combined = rejection_acceptance

    summary = {
        "matrix": matrix,
        "block_id": int(block_id),
        "variant": variant,
        "rank": int(rank),
        "tau": float(args.tau),
        "n_target": int(N_target),
        "n_accepted": int(n_accepted),
        "n_drawn": int(n_drawn),
        "acceptance_rate": float(rejection_acceptance),
        "acceptance_rate_combined": float(rejection_acceptance_combined),
        "method": method,
        "score_opt": float(score_opt),
        "score_oracle": float(score_oracle),
        "score_threshold": float(threshold),
        "union_dim": union_dim,
    }
    # Cos² of the optimizer itself vs oracle (a single reference point).
    cos2_opt, ang_opt = principal_cos2_and_angles(V_opt, V_oracle)
    summary["opt_min_cos2"] = float(np.min(cos2_opt)) if cos2_opt.size else float("nan")
    summary["opt_mean_cos2"] = float(np.mean(cos2_opt)) if cos2_opt.size else float("nan")
    summary["opt_max_angle_rad"] = float(np.max(ang_opt)) if ang_opt.size else float("nan")

    return summary, samples


# ---------------------------------------------------------------------------
# Per-matrix runner
# ---------------------------------------------------------------------------


def run_matrix(args, matrix):
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=matrix, n=args.n, preset=args.preset, seed=args.seed,
        r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
        shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, dtype=np.float64)
    V_exact = np.asarray(V_exact, dtype=np.float64)
    blocks = sorted(set(int(b) for b in args.blocks))
    target = max(blocks)
    snapshots = stream_to_block(args, A, V_exact, work_dtype, int(args.rank), target, set(blocks))

    out = {}
    for b in blocks:
        if b not in snapshots:
            print(f"  [{matrix}] block {b}: no snapshot; skipped")
            continue
        t0 = time.time()
        summary, samples = probe_block(args, matrix, A, V_exact, snapshots[b], b)
        elapsed = time.time() - t0
        summary["elapsed_s"] = elapsed
        out[b] = (summary, samples)
        print(
            f"  [{matrix}] b{b:>2} {args.variant} r={args.rank}: "
            f"accepted {summary['n_accepted']}/{summary['n_target']} of "
            f"{summary['n_drawn']} draws (rate={summary['acceptance_rate_combined']*100:.2f}%, "
            f"method={summary['method']}); "
            f"opt min_cos2={summary['opt_min_cos2']:.3f}; "
            f"score_opt={summary['score_opt']:.3e} thr={summary['score_threshold']:.3e}; "
            f"{elapsed:.1f}s"
        )
    return out


# ---------------------------------------------------------------------------
# Output: CSV per (matrix, block), JSON summary, summary plot
# ---------------------------------------------------------------------------


def write_csv(path, summary, samples):
    """Write per-sample CSV with columns:
        sample_id, score, max_principal_angle_rad, min_cos2, mean_cos2,
        cos2_0, cos2_1, ..., cos2_{r-1}, method, in_sublevel
    """
    rank = summary["rank"]
    fieldnames = [
        "sample_id", "score", "max_principal_angle_rad",
        "min_cos2", "mean_cos2",
    ]
    fieldnames += [f"cos2_{j}" for j in range(rank)]
    fieldnames += [f"angle_rad_{j}" for j in range(rank)]
    fieldnames += ["method", "in_sublevel"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, s in enumerate(samples):
            row = {
                "sample_id": i,
                "score": s["score"],
                "max_principal_angle_rad": s["max_angle_rad"],
                "min_cos2": s["min_cos2"],
                "mean_cos2": s["mean_cos2"],
                "method": s["method"],
                "in_sublevel": 1,
            }
            for j in range(rank):
                row[f"cos2_{j}"] = s["cos2"][j] if j < len(s["cos2"]) else float("nan")
                row[f"angle_rad_{j}"] = s["angles_rad"][j] if j < len(s["angles_rad"]) else float("nan")
            w.writerow(row)


def write_summary_plot(path, results_by_matrix, block_for_plot):
    """Boxplot of min_cos2 distribution per matrix at the chosen block."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = []
    data = []
    opt_points = []
    for matrix, by_block in results_by_matrix.items():
        if block_for_plot not in by_block:
            continue
        summary, samples = by_block[block_for_plot]
        if not samples:
            continue
        labels.append(matrix)
        data.append([s["min_cos2"] for s in samples])
        opt_points.append(summary["opt_min_cos2"])
    if not data:
        return False

    fig, ax = plt.subplots(figsize=(max(6, 2.0 * len(labels)), 5.0))
    bp = ax.boxplot(data, labels=labels, showfliers=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9ecae1")
        patch.set_alpha(0.7)
    # Overlay optimizer's min_cos2 as a red star.
    for i, p in enumerate(opt_points, start=1):
        ax.plot([i], [p], marker="*", color="red", markersize=12,
                markeredgecolor="black", linestyle="none", zorder=10,
                label="optimizer" if i == 1 else None)
    ax.set_ylabel("min principal cos²(V_sample, V_oracle)")
    ax.set_title(
        f"Plateau-width probe at block {block_for_plot} "
        f"(samples within score >= tau · score(V_opt))"
    )
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def write_json_summary(path, summaries_per_matrix, args):
    blob = {
        "args": {
            "variant": args.variant,
            "rank": int(args.rank),
            "tau": float(args.tau),
            "num_samples": int(args.num_samples),
            "blocks": list(args.blocks),
            "matrices": list(args.matrices) if args.matrices else [args.matrix],
            "half_win": int(args.half_win),
            "n": int(args.n),
            "seed": int(args.seed),
            "sample_in_union": bool(args.sample_in_union),
        },
        "per_matrix_block": {
            matrix: {str(b): summary for (b, summary) in by_block.items()}
            for matrix, by_block in summaries_per_matrix.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2, sort_keys=True, default=float)


# ---------------------------------------------------------------------------
# Argument parsing / main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="INFRA-05 plateau-width probe")
    p.add_argument("--matrix", default="mixed-tail-sharp")
    p.add_argument(
        "--matrices", nargs="*", default=None,
        help="If given, run multiple matrices (default: probe set).",
    )
    p.add_argument(
        "--out-dir", default="summary/infra_plateau_width",
        help="Output directory for CSVs / plot / JSON.",
    )
    p.add_argument(
        "--blocks", nargs="+", type=int, default=[6, 12, 31],
        help="Blocks to probe. Default covers mid-stream and terminal.",
    )
    p.add_argument(
        "--summary-plot-block", type=int, default=31,
        help="Block to use for the cross-matrix summary plot (default 31).",
    )
    p.add_argument(
        "--variant", choices=("S1", "S2", "S3", "S4", "S5", "S6", "S6_GM"),
        default="S6",
    )
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--tau", type=float, default=0.95)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument(
        "--max-rejection-draws", type=int, default=20000,
        help="Hard cap on rejection draws before MCMC fallback.",
    )
    p.add_argument(
        "--max-mcmc-steps", type=int, default=50000,
        help="Hard cap on MCMC proposals.",
    )
    p.add_argument(
        "--mcmc-step", type=float, default=0.05,
        help="Std-dev of Gaussian tangent perturbation in MCMC fallback.",
    )
    p.add_argument(
        "--sample-in-union", action="store_true", default=True,
        help="Sample frames in the union rowspan (B_union) instead of full R^n.",
    )
    p.add_argument(
        "--no-sample-in-union", dest="sample_in_union", action="store_false",
        help="Sample on the full sphere instead.",
    )
    p.add_argument(
        "--no-oracle-warmstart", action="store_true",
        help="Drop oracle from V_opt warm-starts (matches §9 streaming workflow).",
    )

    # Optimizer / streaming args (mirrored from r_sk_g_score.py)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--half-win", type=int, default=32)
    p.add_argument("--preset", default="fast")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-rows", action="store_true", default=True)
    p.add_argument("--row-shuffle-seed", type=int, default=0)
    p.add_argument("--old-memory-size", type=int, default=32)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--q0", type=int, default=8)
    p.add_argument("--qmax", type=int, default=48)
    p.add_argument("--krylov-depth", type=int, default=2)
    p.add_argument("--residual-tol", type=float, default=0.01)
    p.add_argument("--expansion-maxit", type=int, default=8)
    p.add_argument("--num-restarts", type=int, default=3)
    p.add_argument("--maxit", type=int, default=120)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--post-expansion-maxit", type=int, default=80)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--patience-rel-tol", type=float, default=1e-5)
    p.add_argument("--union-maxit", type=int, default=120)
    p.add_argument("--union-tol", type=float, default=1e-9)
    p.add_argument("--union-random-starts", type=int, default=24)
    p.add_argument("--r-sig", type=int, default=2)
    p.add_argument("--alpha-sig", type=float, default=0.003)
    p.add_argument("--alpha-tail", type=float, default=0.0145)
    p.add_argument("--tail-scale", type=float, default=0.99)
    p.add_argument("--sigma1", type=float, default=0.991)
    p.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    return p.parse_args()


def main():
    args = parse_args()
    matrices = args.matrices if args.matrices else (
        ["static-cex", "mixed-tail-sharp", "diffuse-diffuse"] if args.matrix == "mixed-tail-sharp" and args.matrices is None
        else [args.matrix]
    )
    # Allow user to override single-matrix run with --matrix foo (bypasses the
    # default 3-matrix probe set when --matrices is not provided).
    if args.matrices is None and args.matrix != "mixed-tail-sharp":
        matrices = [args.matrix]
    elif args.matrices is None:
        matrices = ["static-cex", "mixed-tail-sharp", "diffuse-diffuse"]

    os.makedirs(args.out_dir, exist_ok=True)
    print(
        f"plateau_width_probe: variant={args.variant} rank={args.rank} "
        f"tau={args.tau} N={args.num_samples} blocks={args.blocks} "
        f"matrices={matrices}"
    )

    summaries_per_matrix = {}
    results_per_matrix = {}
    for matrix in matrices:
        print(f"== {matrix} ==")
        per_block = run_matrix(args, matrix)
        results_per_matrix[matrix] = per_block
        summaries_per_matrix[matrix] = {b: summary for b, (summary, _) in per_block.items()}

        for b, (summary, samples) in per_block.items():
            csv_path = os.path.join(
                args.out_dir,
                f"{matrix}_b{b:02d}_{args.variant}_r{args.rank}.csv",
            )
            write_csv(csv_path, summary, samples)
            print(f"  wrote {csv_path}")

    # Cross-matrix summary plot at the chosen block.
    plot_path = os.path.join(args.out_dir, "summary_plot.png")
    plotted = write_summary_plot(plot_path, results_per_matrix, args.summary_plot_block)
    if plotted:
        print(f"wrote {plot_path}")
    else:
        print(f"summary plot not produced (no samples at block {args.summary_plot_block})")

    json_path = os.path.join(args.out_dir, "summary.json")
    write_json_summary(json_path, summaries_per_matrix, args)
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
