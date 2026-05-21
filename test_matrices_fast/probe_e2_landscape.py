"""S6_E2 landscape investigation — probes A/B/C/D for AB-03 phase 1 closure.

Question: why does DIAG-04b's E2 oracle-balance prediction anti-correlate
with T3 sliding-bench cos1² on diffuse-diffuse and residual-spiky-shocks?

For each matrix in {diffuse-diffuse, residual-spiky-shocks, mixed-tail-soft},
at block 31 we run:

  Probe A — oracle-vs-winner score gap. Score(oracle_*) vs Score(opt_winner)
            under S6 and S6_E2; if the S6_E2 winner score exceeds the S6_E2
            oracle score, hypothesis 1 (landscape argmax ≠ oracle) holds.

  Probe B — geodesic landscape between oracle_v1_proj and oracle_v2_proj
            (great-circle), evaluated under S6 and S6_E2. Looks for a
            non-oracle peak inserted by E2.

  Probe C — argmax map k_sk(v(t)), k_g1(v(t)), k_g2(v(t)) along the same
            geodesic. Score discontinuities at boundaries support
            hypothesis 2 (Voronoi step changes).

  Probe D — slot-1 attractor strength: gradient projection onto
            V_state[:,0] under tiny perturbation off V_state[:,0]. If the
            projection collapses under S6_E2 vs S6, hypothesis 3 holds.

Outputs:
  summary/score_family_aggregator_ablation/landscape_<matrix>_b31.json
  summary/score_family_aggregator_ablation/landscape_<matrix>_b31.txt
  summary/score_family_aggregator_ablation/landscape_summary.txt
"""
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np

import cex_restricted_space_probe as probe
import half_window_sliding_hmean_experiment as hm  # noqa: F401  (import side-effects)
from future_hmean_optimizer_diagnostic import orth_basis_against, rowspace_basis
from hmean_evidence_score import per_block_constants, stream_to_block

from r_sk_g_score import (
    _per_direction_w_e2,
    _state_V,
    build_e2_data,
    optimize_r_sk_g_in_basis,
    r_sk_g_value_grad,
)


def make_default_args(matrix: str, block: int = 31) -> SimpleNamespace:
    """Build an argparse-Namespace with the defaults expected by stream_to_block
    and the analyze_block helpers, matching r_sk_g_score.parse_args() defaults.
    """
    return SimpleNamespace(
        matrix=matrix,
        matrices=None,
        out_prefix="summary/score_family_aggregator_ablation/landscape",
        blocks=[block],
        variant="S6_E2",
        alpha=1.0, beta=2.0, gamma=1.0,
        n=1024,
        half_win=32,
        rank=2,
        preset="fast",
        seed=0,
        shuffle_rows=True,
        row_shuffle_seed=0,
        old_memory_size=32,
        dtype="float32",
        q0=8, qmax=48, krylov_depth=2,
        residual_tol=0.01, expansion_maxit=8,
        num_restarts=3,
        maxit=120, tol=1e-8,
        post_expansion_maxit=80,
        patience=5, patience_rel_tol=1e-5,
        union_maxit=120, union_tol=1e-9, union_random_starts=24,
        r_sig=2, alpha_sig=0.003, alpha_tail=0.0145,
        tail_scale=0.99, sigma1=0.991,
        v_type="rand",
        gradient_check=False,
        no_oracle_warmstart=False,
    )


def _project_unit(vec, B):
    if vec is None or B is None or B.size == 0:
        return None
    p = B @ (B.T @ vec)
    nv = float(np.linalg.norm(p))
    return None if nv <= 1e-30 else p / nv


def _argmax_k(V_top, v):
    if V_top is None or V_top.size == 0:
        return -1
    proj = V_top.T @ v
    return int(np.argmax(proj * proj))


def _score_at(v, *, A_sk_for, A_cur, A_fut, c_sk, V_state,
              cur_F2, fut_F2, sk_F2_low, e2_data):
    """Return dict with both S6 and S6_E2 scores at v (already unit-norm)."""
    s6 = r_sk_g_value_grad(
        A_sk_for, A_cur, A_fut, c_sk, v,
        variant="S6", V_state=V_state,
        cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
    )
    se2 = r_sk_g_value_grad(
        A_sk_for, A_cur, A_fut, c_sk, v,
        variant="S6_E2", V_state=V_state, e2_data=e2_data,
        cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
    )
    # tuples: (score, grad, r_sk, raw_g1, raw_g2, hm_g, sat_term, s_align)
    return {
        "score_S6": float(s6[0]),
        "score_S6_E2": float(se2[0]),
        "grad_S6": np.ascontiguousarray(s6[1]),
        "grad_S6_E2": np.ascontiguousarray(se2[1]),
        "raw_sk": float(s6[2]),
        "raw_g1": float(s6[3]),
        "raw_g2": float(s6[4]),
    }


def run_probe_for_matrix(matrix: str, *, block: int = 31, n_geodesic: int = 121):
    args = make_default_args(matrix, block=block)
    work_dtype = np.float32 if args.dtype == "float32" else np.float64

    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=matrix, n=args.n, preset=args.preset, seed=args.seed,
        r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
        shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, np.float64)
    V_exact = np.asarray(V_exact, np.float64)

    snapshots = stream_to_block(args, A, V_exact, work_dtype, int(args.rank), block, {block})
    snap = snapshots[block]
    consts = per_block_constants(A, block, args.half_win)

    A_cur = np.asarray(snap["A_cur"], dtype=np.float64)
    A_fut = np.asarray(snap["A_fut"], dtype=np.float64)
    A_sketch = np.asarray(snap["A_sketch"], dtype=np.float64)
    A_sk_for = A_sketch if A_sketch.size else None
    state = snap["state"]
    V_state = _state_V(state)
    V_default = snap["V_default"]

    c_sk = float(consts["c_sk"])
    cur_F2 = float(consts["cur_F2"])
    fut_F2 = float(consts["fut_F2"])
    sk_F2_low = float(np.sum(A_sketch ** 2)) if A_sk_for is not None else 0.0

    e2_data = build_e2_data(A_cur, A_fut, A_sketch, state, int(args.rank))

    # Bases
    if A_sk_for is not None:
        union_stack = np.vstack([A_sketch, A_cur, A_fut])
    else:
        union_stack = np.vstack([A_cur, A_fut])
    B_union = rowspace_basis(union_stack)

    # Reference vectors
    oracle_v1 = V_exact[:, 0] / max(np.linalg.norm(V_exact[:, 0]), 1e-30)
    oracle_v2 = V_exact[:, 1] / max(np.linalg.norm(V_exact[:, 1]), 1e-30)
    oracle_v1_proj = _project_unit(oracle_v1, B_union)
    oracle_v2_proj = _project_unit(oracle_v2, B_union)
    sketch_v1 = (V_state[:, 0] / max(np.linalg.norm(V_state[:, 0]), 1e-30)
                 if V_state is not None and V_state.shape[1] >= 1 else None)
    sketch_v2 = (V_state[:, 1] / max(np.linalg.norm(V_state[:, 1]), 1e-30)
                 if V_state is not None and V_state.shape[1] >= 2 else None)

    # ---- Probe A: rank-2 sequential optimizer winners under S6 and S6_E2 ----
    starts = []
    if oracle_v1_proj is not None:
        starts.append(oracle_v1_proj)
    if oracle_v2_proj is not None:
        starts.append(oracle_v2_proj)
    if sketch_v1 is not None:
        starts.append(sketch_v1)
    if sketch_v2 is not None:
        starts.append(sketch_v2)
    starts.append(V_default[:, 0])
    starts.append(V_default[:, 1])

    def opt(variant, basis, seed_off, **extra):
        best = optimize_r_sk_g_in_basis(
            A_cur, A_fut, A_sk_for, c_sk,
            basis, starts,
            np.random.default_rng(args.seed + seed_off + block),
            args.union_maxit, args.union_tol, args.union_random_starts,
            variant=variant, alpha=args.alpha, beta=args.beta, gamma=args.gamma,
            V_state=V_state,
            cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
            **extra,
        )
        if best is None:
            return None
        v = np.asarray(best["vec"], dtype=np.float64)
        v = v / max(np.linalg.norm(v), 1e-30)
        return v

    v1_S6 = opt("S6", B_union, 57000)
    v2_S6 = (opt("S6", orth_basis_against(B_union, v1_S6), 58000)
             if v1_S6 is not None else None)
    v1_E2 = opt("S6_E2", B_union, 65000, e2_data=e2_data)
    v2_E2 = (opt("S6_E2", orth_basis_against(B_union, v1_E2), 66000, e2_data=e2_data)
             if v1_E2 is not None else None)

    candidates = {
        "oracle_v1_proj": oracle_v1_proj,
        "oracle_v2_proj": oracle_v2_proj,
        "sketch_v1": sketch_v1,
        "sketch_v2": sketch_v2,
        "S6_v1_winner": v1_S6,
        "S6_v2_winner": v2_S6,
        "S6_E2_v1_winner": v1_E2,
        "S6_E2_v2_winner": v2_E2,
    }

    score_table = {}
    for label, v in candidates.items():
        if v is None:
            continue
        v = v / max(np.linalg.norm(v), 1e-30)
        s = _score_at(v, A_sk_for=A_sk_for, A_cur=A_cur, A_fut=A_fut, c_sk=c_sk,
                      V_state=V_state, cur_F2=cur_F2, fut_F2=fut_F2,
                      sk_F2_low=sk_F2_low, e2_data=e2_data)
        # Drop gradients in the table to keep JSON small.
        score_table[label] = {
            "score_S6": s["score_S6"],
            "score_S6_E2": s["score_S6_E2"],
            "raw_sk": s["raw_sk"],
            "raw_g1": s["raw_g1"],
            "raw_g2": s["raw_g2"],
            "k_sk": _argmax_k(e2_data.get("V_sk"), v),
            "k_g1": _argmax_k(e2_data.get("V_cur"), v),
            "k_g2": _argmax_k(e2_data.get("V_fut"), v),
            "cos2_oracle_v1": float(np.dot(v, oracle_v1_proj) ** 2)
                if oracle_v1_proj is not None else float("nan"),
            "cos2_oracle_v2": float(np.dot(v, oracle_v2_proj) ** 2)
                if oracle_v2_proj is not None else float("nan"),
            "cos2_sketch_v1": float(np.dot(v, sketch_v1) ** 2)
                if sketch_v1 is not None else float("nan"),
        }

    # ---- Probe B/C: geodesic between oracle_v1_proj and oracle_v2_proj ----
    geodesic = []
    if oracle_v1_proj is not None and oracle_v2_proj is not None:
        v1 = oracle_v1_proj
        # Orthonormal pair in span(oracle_v1_proj, oracle_v2_proj):
        v2_orth = oracle_v2_proj - float(np.dot(oracle_v1_proj, oracle_v2_proj)) * oracle_v1_proj
        v2_norm = float(np.linalg.norm(v2_orth))
        if v2_norm > 1e-12:
            v2 = v2_orth / v2_norm
            for t in np.linspace(0.0, np.pi / 2.0, n_geodesic):
                v = np.cos(t) * v1 + np.sin(t) * v2
                v = v / max(np.linalg.norm(v), 1e-30)
                s = _score_at(v, A_sk_for=A_sk_for, A_cur=A_cur, A_fut=A_fut,
                              c_sk=c_sk, V_state=V_state, cur_F2=cur_F2,
                              fut_F2=fut_F2, sk_F2_low=sk_F2_low, e2_data=e2_data)
                geodesic.append({
                    "t": float(t),
                    "score_S6": s["score_S6"],
                    "score_S6_E2": s["score_S6_E2"],
                    "k_sk": _argmax_k(e2_data.get("V_sk"), v),
                    "k_g1": _argmax_k(e2_data.get("V_cur"), v),
                    "k_g2": _argmax_k(e2_data.get("V_fut"), v),
                    "cos2_oracle_v1": float(np.dot(v, oracle_v1_proj) ** 2),
                    "cos2_oracle_v2": float(np.dot(v, oracle_v2_proj) ** 2),
                    "cos2_sketch_v1": float(np.dot(v, sketch_v1) ** 2)
                        if sketch_v1 is not None else float("nan"),
                })

    # ---- Probe D: slot-1 attractor strength near V_state[:,0] ----
    attractor = None
    if sketch_v1 is not None:
        # Build a unit perturbation direction inside B_union, orthogonal to sketch_v1.
        rng = np.random.default_rng(args.seed + 99000 + block)
        d = B_union @ (B_union.T @ rng.standard_normal(B_union.shape[0]))
        d = d - float(np.dot(d, sketch_v1)) * sketch_v1
        d_norm = float(np.linalg.norm(d))
        rows = []
        if d_norm > 1e-12:
            d = d / d_norm
            for theta in (1e-3, 1e-2, 5e-2, 1e-1):
                v = np.cos(theta) * sketch_v1 + np.sin(theta) * d
                v = v / max(np.linalg.norm(v), 1e-30)
                s = _score_at(v, A_sk_for=A_sk_for, A_cur=A_cur, A_fut=A_fut,
                              c_sk=c_sk, V_state=V_state, cur_F2=cur_F2,
                              fut_F2=fut_F2, sk_F2_low=sk_F2_low, e2_data=e2_data)
                # Project gradient onto the tangent direction sketch_v1 (back-pull
                # component on the sphere): tangent at v toward sketch_v1 is
                # sketch_v1 - <sketch_v1,v> v; but for a small perturbation the
                # raw gradient projection onto sketch_v1 is a sufficient proxy.
                pull_S6 = float(np.dot(s["grad_S6"], sketch_v1))
                pull_E2 = float(np.dot(s["grad_S6_E2"], sketch_v1))
                rows.append({
                    "theta": float(theta),
                    "score_S6": s["score_S6"],
                    "score_S6_E2": s["score_S6_E2"],
                    "grad_S6_norm": float(np.linalg.norm(s["grad_S6"])),
                    "grad_S6_E2_norm": float(np.linalg.norm(s["grad_S6_E2"])),
                    "pull_to_sketch_v1_S6": pull_S6,
                    "pull_to_sketch_v1_S6_E2": pull_E2,
                })
        attractor = rows

    # Per-direction weight diagnostics at oracle_v2_proj and sketch_v1:
    W_diag = {}
    for label, v in (("oracle_v2_proj", oracle_v2_proj),
                     ("oracle_v1_proj", oracle_v1_proj),
                     ("sketch_v1", sketch_v1)):
        if v is None:
            continue
        W_diag[label] = {
            "W_sk_E2": float(_per_direction_w_e2(e2_data.get("V_sk"),
                                                e2_data.get("s2_sk"), v, sk_F2_low)),
            "W_c_E2": float(_per_direction_w_e2(e2_data.get("V_cur"),
                                               e2_data.get("s2_cur"), v, cur_F2)),
            "W_f_E2": float(_per_direction_w_e2(e2_data.get("V_fut"),
                                               e2_data.get("s2_fut"), v, fut_F2)),
            "k_sk": _argmax_k(e2_data.get("V_sk"), v),
            "k_g1": _argmax_k(e2_data.get("V_cur"), v),
            "k_g2": _argmax_k(e2_data.get("V_fut"), v),
        }
    W_diag["F_norm_weights"] = {
        "sk_F2_low": sk_F2_low, "cur_F2": cur_F2, "fut_F2": fut_F2,
        "s2_sk_top": list(map(float, np.asarray(e2_data.get("s2_sk")))) if e2_data.get("s2_sk") is not None else None,
        "s2_cur_top": list(map(float, np.asarray(e2_data.get("s2_cur")))) if e2_data.get("s2_cur") is not None else None,
        "s2_fut_top": list(map(float, np.asarray(e2_data.get("s2_fut")))) if e2_data.get("s2_fut") is not None else None,
    }

    return {
        "matrix": matrix,
        "block": block,
        "score_table": score_table,
        "geodesic": geodesic,
        "attractor": attractor,
        "W_diag": W_diag,
    }


def _fmt_float(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "  nan  "
    return f"{x:>{nd+5}.{nd}f}"


def write_text_report(result: dict, out_path: str) -> None:
    matrix = result["matrix"]
    block = result["block"]
    st = result["score_table"]
    geo = result["geodesic"]
    att = result.get("attractor") or []
    W = result["W_diag"]
    lines = []
    lines.append(f"== matrix={matrix} block={block}")
    lines.append("")
    lines.append("# Probe A — score under S6 vs S6_E2 at named candidates")
    lines.append(f"  {'label':<20} {'score_S6':>10} {'score_S6_E2':>12}  "
                 f"{'cos2_o1':>8} {'cos2_o2':>8} {'cos2_sV1':>9} "
                 f"{'k_sk':>4} {'k_g1':>4} {'k_g2':>4}")
    for label, row in st.items():
        lines.append(
            f"  {label:<20} {row['score_S6']:>10.4f} {row['score_S6_E2']:>12.4f}  "
            f"{row['cos2_oracle_v1']:>8.3f} {row['cos2_oracle_v2']:>8.3f} {row['cos2_sketch_v1']:>9.3f} "
            f"{row['k_sk']:>4d} {row['k_g1']:>4d} {row['k_g2']:>4d}"
        )
    if "S6_E2_v2_winner" in st and "oracle_v2_proj" in st:
        gap = st["S6_E2_v2_winner"]["score_S6_E2"] - st["oracle_v2_proj"]["score_S6_E2"]
        lines.append(f"  Δ_S6_E2(slot2_winner − oracle_v2_proj) = {gap:+.4e}")
    if "S6_v2_winner" in st and "oracle_v2_proj" in st:
        gap = st["S6_v2_winner"]["score_S6"] - st["oracle_v2_proj"]["score_S6"]
        lines.append(f"  Δ_S6   (slot2_winner − oracle_v2_proj) = {gap:+.4e}")
    if "S6_E2_v1_winner" in st and "sketch_v1" in st:
        cosw = st["S6_E2_v1_winner"]["cos2_sketch_v1"]
        coss6 = st.get("S6_v1_winner", {}).get("cos2_sketch_v1", float("nan"))
        lines.append(f"  cos²(S6_v1, sketch_v1) = {coss6:.3f}   "
                     f"cos²(S6_E2_v1, sketch_v1) = {cosw:.3f}")
    lines.append("")
    lines.append("# Probe B/C — geodesic from oracle_v1_proj → oracle_v2_proj")
    lines.append("# t in [0, π/2]; logging score, argmax of each source, cos² to oracle vectors")
    if geo:
        lines.append(f"  {'t/π':>6} {'score_S6':>10} {'score_E2':>10} "
                     f"{'k_sk':>4} {'k_g1':>4} {'k_g2':>4} "
                     f"{'cos2_o1':>8} {'cos2_o2':>8} {'cos2_sV1':>9}")
        # Print every Nth point + every transition.
        n = len(geo)
        prev = None
        for i, p in enumerate(geo):
            transition = (
                prev is not None and (
                    prev["k_sk"] != p["k_sk"]
                    or prev["k_g1"] != p["k_g1"]
                    or prev["k_g2"] != p["k_g2"]
                )
            )
            if i % max(1, n // 12) == 0 or transition or i == n - 1:
                marker = " *" if transition else "  "
                lines.append(
                    f"  {p['t']/np.pi:>6.3f} {p['score_S6']:>10.4f} {p['score_S6_E2']:>10.4f} "
                    f"{p['k_sk']:>4d} {p['k_g1']:>4d} {p['k_g2']:>4d} "
                    f"{p['cos2_oracle_v1']:>8.3f} {p['cos2_oracle_v2']:>8.3f} "
                    f"{p['cos2_sketch_v1']:>9.3f}{marker}"
                )
            prev = p
        # Score-discontinuity check: for each transition point, compare scores
        # with neighbors.
        max_jump_S6 = 0.0
        max_jump_E2 = 0.0
        for i in range(1, len(geo)):
            jp = geo[i]; jq = geo[i - 1]
            if (jp["k_sk"] != jq["k_sk"] or jp["k_g1"] != jq["k_g1"]
                    or jp["k_g2"] != jq["k_g2"]):
                max_jump_S6 = max(max_jump_S6,
                                  abs(jp["score_S6"] - jq["score_S6"]))
                max_jump_E2 = max(max_jump_E2,
                                  abs(jp["score_S6_E2"] - jq["score_S6_E2"]))
        lines.append(f"  max-Δscore at any argmax transition: S6={max_jump_S6:.4e}  "
                     f"S6_E2={max_jump_E2:.4e}")
        # Find peaks (local argmax) under each score.
        s6_arr = np.array([p["score_S6"] for p in geo])
        e2_arr = np.array([p["score_S6_E2"] for p in geo])
        i6 = int(np.argmax(s6_arr)); i2 = int(np.argmax(e2_arr))
        lines.append(
            f"  argmax(S6) at t/π={geo[i6]['t']/np.pi:.3f} "
            f"score={s6_arr[i6]:.4f}  cos²(o1,o2)=({geo[i6]['cos2_oracle_v1']:.3f},"
            f"{geo[i6]['cos2_oracle_v2']:.3f})"
        )
        lines.append(
            f"  argmax(S6_E2) at t/π={geo[i2]['t']/np.pi:.3f} "
            f"score={e2_arr[i2]:.4f}  cos²(o1,o2)=({geo[i2]['cos2_oracle_v1']:.3f},"
            f"{geo[i2]['cos2_oracle_v2']:.3f})"
        )
    else:
        lines.append("  (geodesic skipped: missing oracle projection)")
    lines.append("")
    lines.append("# Probe D — slot-1 attractor near sketch_v1 (V_state[:,0])")
    if att:
        lines.append(f"  {'theta':>8} {'score_S6':>10} {'score_E2':>10} "
                     f"{'|grad_S6|':>10} {'|grad_E2|':>10} "
                     f"{'pullS6':>10} {'pullE2':>10}")
        for r in att:
            lines.append(
                f"  {r['theta']:>8.4f} {r['score_S6']:>10.4f} {r['score_S6_E2']:>10.4f} "
                f"{r['grad_S6_norm']:>10.3e} {r['grad_S6_E2_norm']:>10.3e} "
                f"{r['pull_to_sketch_v1_S6']:>10.3e} {r['pull_to_sketch_v1_S6_E2']:>10.3e}"
            )
        lines.append("  (positive 'pull' = gradient leans toward sketch_v1, i.e., "
                     "score increases when v moves back to sketch_v1)")
    else:
        lines.append("  (attractor probe skipped: no sketch_v1 available)")
    lines.append("")
    lines.append("# E2 per-direction weight diagnostics")
    for label, row in W.items():
        if label == "F_norm_weights":
            f = row
            lines.append(f"  F-norm weights: cur_F2={f['cur_F2']:.3e}  fut_F2={f['fut_F2']:.3e}  sk_F2_low={f['sk_F2_low']:.3e}")
            for k in ("s2_sk_top", "s2_cur_top", "s2_fut_top"):
                if f[k] is not None:
                    s = "  ".join(f"{x:.3e}" for x in f[k])
                    lines.append(f"  {k}: [{s}]")
            continue
        lines.append(
            f"  {label:<20} W_sk_E2={row['W_sk_E2']:.3e}  W_c_E2={row['W_c_E2']:.3e}  "
            f"W_f_E2={row['W_f_E2']:.3e}  k_sk={row['k_sk']} k_g1={row['k_g1']} k_g2={row['k_g2']}"
        )
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", nargs="+", default=[
        "diffuse-diffuse", "residual-spiky-shocks", "mixed-tail-soft",
    ])
    ap.add_argument("--block", type=int, default=31)
    ap.add_argument("--n-geodesic", type=int, default=121)
    ap.add_argument("--out-dir", default="summary/score_family_aggregator_ablation")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    summary_lines = []
    summary_lines.append("# S6_E2 landscape probes — summary across §6 matrices")
    summary_lines.append(f"# block={args.block}; geodesic from oracle_v1_proj→oracle_v2_proj")
    summary_lines.append("")
    summary_lines.append(
        f"{'matrix':<24} {'S6_e2(o2)':>10} {'S6_e2(v2*)':>11} "
        f"{'gap_E2':>9} {'gap_S6':>9} {'arg_t/π_E2':>11} {'arg_t/π_S6':>11} "
        f"{'cos²(v1*,sV1)_S6':>17} {'cos²(v1*,sV1)_E2':>17}"
    )
    for matrix in args.matrices:
        print(f"== {matrix} ==", flush=True)
        result = run_probe_for_matrix(matrix, block=args.block,
                                      n_geodesic=args.n_geodesic)
        json_path = os.path.join(args.out_dir, f"landscape_{matrix}_b{args.block}.json")
        txt_path = os.path.join(args.out_dir, f"landscape_{matrix}_b{args.block}.txt")
        with open(json_path, "w") as fh:
            json.dump(result, fh, indent=2, default=float)
        write_text_report(result, txt_path)
        st = result["score_table"]
        s_o2 = st.get("oracle_v2_proj", {}).get("score_S6_E2", float("nan"))
        s_v2 = st.get("S6_E2_v2_winner", {}).get("score_S6_E2", float("nan"))
        gap_E2 = (s_v2 - s_o2) if (not np.isnan(s_o2) and not np.isnan(s_v2)) else float("nan")
        s_o2_S6 = st.get("oracle_v2_proj", {}).get("score_S6", float("nan"))
        s_v2_S6 = st.get("S6_v2_winner", {}).get("score_S6", float("nan"))
        gap_S6 = (s_v2_S6 - s_o2_S6) if (not np.isnan(s_o2_S6) and not np.isnan(s_v2_S6)) else float("nan")
        # Geodesic argmax positions
        arg_t_E2 = float("nan"); arg_t_S6 = float("nan")
        if result["geodesic"]:
            geo = result["geodesic"]
            arg_t_E2 = geo[int(np.argmax([p["score_S6_E2"] for p in geo]))]["t"] / np.pi
            arg_t_S6 = geo[int(np.argmax([p["score_S6"] for p in geo]))]["t"] / np.pi
        cos2_v1_E2 = st.get("S6_E2_v1_winner", {}).get("cos2_sketch_v1", float("nan"))
        cos2_v1_S6 = st.get("S6_v1_winner", {}).get("cos2_sketch_v1", float("nan"))
        summary_lines.append(
            f"{matrix:<24} {s_o2:>10.4f} {s_v2:>11.4f} "
            f"{gap_E2:>+9.4f} {gap_S6:>+9.4f} {arg_t_E2:>11.3f} {arg_t_S6:>11.3f} "
            f"{cos2_v1_S6:>17.3f} {cos2_v1_E2:>17.3f}"
        )
        print(f"  wrote {json_path}\n  wrote {txt_path}", flush=True)

    summary_path = os.path.join(args.out_dir,
                                f"landscape_summary_b{args.block}.txt")
    with open(summary_path, "w") as fh:
        fh.write("\n".join(summary_lines) + "\n")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
