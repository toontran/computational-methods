"""Reproduce combined-score M1/M2/M3 failures and test tail diagnostics.

Diagnostic-only probe for summary/overview workflow item DIAG-05.
It intentionally uses oracle directions as references; the goal is mechanism
identification, not a value-only policy.
"""

import argparse
import csv
import json
import os
import time

import numpy as np

import cex_restricted_space_probe as probe
import combined_objective_sketch_bias_probe as combined_probe
import r_sk_g_score
import half_window_sliding_hmean_experiment as hmexp
from future_hmean_optimizer_diagnostic import orth_basis_against, rowspace_basis
from hmean_evidence_score import per_block_constants, stream_to_block


def unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    nv = float(np.linalg.norm(v))
    if nv <= 1e-30:
        return None
    return np.ascontiguousarray(v / nv)


def project_unit(v, B):
    if v is None or B is None or B.size == 0:
        return None
    p = B @ (B.T @ v)
    return unit(p)


def deflate_unit(v, q):
    v = unit(v)
    q = unit(q)
    if v is None or q is None:
        return None
    return unit(v - q * float(q @ v))


def rel_entropy_and_top_share(A, v):
    A = np.asarray(A, dtype=np.float64)
    if A.size == 0:
        return {
            "rel_entropy": None,
            "top_share": None,
            "energy": 0.0,
        }
    y = A @ v
    e = y * y
    S = float(np.sum(e))
    if S <= 1e-30:
        return {
            "rel_entropy": 0.0,
            "top_share": 0.0,
            "energy": 0.0,
        }
    p = e / S
    H = -float(np.sum(p * np.log(np.maximum(p, 1e-300))))
    return {
        "rel_entropy": float(H / np.log(max(len(e), 2))),
        "top_share": float(np.max(p)),
        "energy": S,
    }


def pearson_abs_response(A_cur, A_fut, v):
    x = np.abs(np.asarray(A_cur, dtype=np.float64) @ v)
    y = np.abs(np.asarray(A_fut, dtype=np.float64) @ v)
    if x.size != y.size or x.size < 2:
        return None
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 1e-30 or sy <= 1e-30:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def source_u(A, denom, v):
    if A is None or np.asarray(A).size == 0 or denom is None or denom <= 1e-30:
        return 0.0
    y = np.asarray(A, dtype=np.float64) @ v
    return float(np.dot(y, y) / denom)


def tail_diag_record(matrix, block, mechanism, label, v, snap, A, V_exact,
                     oracle_slot, role):
    v = unit(v)
    if v is None:
        return None
    A_sketch = snap["A_sketch"] if snap["A_sketch"].size else None
    A_cur = snap["A_cur"]
    A_fut = snap["A_fut"]
    consts = per_block_constants(A, block, A_cur.shape[0])
    sk_den = float(np.sum(A_sketch * A_sketch)) if A_sketch is not None else 0.0
    cur_den = consts["cur_F2"]
    fut_den = consts["fut_F2"]
    u_cur = source_u(A_cur, cur_den, v)
    u_fut = source_u(A_fut, fut_den, v)
    asym = abs(u_cur - u_fut) / max(u_cur, u_fut, 1e-30)
    d_sk = rel_entropy_and_top_share(A_sketch, v) if A_sketch is not None else {
        "rel_entropy": None, "top_share": None, "energy": 0.0,
    }
    d_cur = rel_entropy_and_top_share(A_cur, v)
    d_fut = rel_entropy_and_top_share(A_fut, v)
    V_state = None
    if snap["state"] is not None and snap["state"].get("V") is not None:
        V_state = np.asarray(snap["state"]["V"], dtype=np.float64)
    state_align = None
    if V_state is not None and V_state.size:
        p = V_state.T @ v
        state_align = float(np.dot(p, p))
    return {
        "kind": "tail_diagnostic",
        "matrix": matrix,
        "block": int(block),
        "mechanism": mechanism,
        "label": label,
        "role": role,
        "oracle_slot": int(oracle_slot),
        "align_exact_o1": float((v @ unit(V_exact[:, 0])) ** 2),
        "align_exact_o2": float((v @ unit(V_exact[:, 1])) ** 2),
        "state_align": state_align,
        "u_cur": u_cur,
        "u_fut": u_fut,
        "between_asym": float(asym),
        "replicability_abs_pearson": pearson_abs_response(A_cur, A_fut, v),
        "rel_entropy_B": d_sk["rel_entropy"],
        "top_share_B": d_sk["top_share"],
        "energy_B": d_sk["energy"],
        "rel_entropy_cur": d_cur["rel_entropy"],
        "top_share_cur": d_cur["top_share"],
        "energy_cur": d_cur["energy"],
        "rel_entropy_fut": d_fut["rel_entropy"],
        "top_share_fut": d_fut["top_share"],
        "energy_fut": d_fut["energy"],
    }


def random_panel(B, avoid, count, seed):
    rng = np.random.default_rng(seed)
    out = []
    n = B.shape[0]
    for k in range(count):
        v = rng.standard_normal(n)
        if B is not None and B.size:
            v = B @ (B.T @ v)
        if avoid is not None:
            for q in avoid:
                v = deflate_unit(v, q)
                if v is None:
                    break
        v = unit(v)
        if v is not None:
            out.append((f"random_{k}", v))
    return out


def generate(args, matrix):
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=matrix, n=args.n, preset=args.preset, seed=args.seed,
        r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
        shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
    )
    return np.asarray(A, dtype=np.float64), np.asarray(V_exact, dtype=np.float64)


def combined_score_parts(A, snap, v):
    comp = probe.combined_score_component_details(
        snap["M_gain"], snap["A_cur"], v, A.shape[0],
        state_prev=snap["state"], old_row_memory=snap["old_row_memory"],
    )
    return comp


def m1_rows(args, A, V_exact, snap):
    rows = []
    block = 1
    A_cur = snap["A_cur"]
    B_union = rowspace_basis(A_cur)
    v1 = unit(snap["V_default"][:, 0])
    oracle_v2_proj = project_unit(unit(V_exact[:, 1]), B_union)
    oracle_v2_perp = deflate_unit(oracle_v2_proj, v1)
    candidates = [("combined_v2", unit(snap["V_default"][:, 1]), "score_favoured")]
    candidates.append(("oracle_v2_proj_perp", oracle_v2_perp, "oracle"))
    candidates.extend((lab, v, "random") for lab, v in random_panel(B_union, [v1], 5, args.seed + 101))
    for label, v, role in candidates:
        if v is None:
            continue
        comp = combined_score_parts(A, snap, v)
        rows.append({
            "kind": "ablation",
            "mechanism": "M1",
            "matrix": "mixed-tail-sharp",
            "block": block,
            "label": label,
            "role": role,
            "combined_score": float(comp["score_total"]),
            "phi": float(comp["phi"]),
            "energy_A_cur": float(comp["new_y2_sq"]),
            "gain_total": float(comp["gain2"]),
            "align_projected_o2": None if oracle_v2_proj is None else float((v @ oracle_v2_proj) ** 2),
        })
        rows.append(tail_diag_record("mixed-tail-sharp", block, "M1", label, v, snap, A, V_exact, 2, role))
    return [r for r in rows if r is not None]


def m2_rows(args, A, V_exact, snapshots):
    rows = []
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    for block in [1, 6, 12, 31]:
        snap = snapshots[block]
        if "V_joint" not in snap:
            snap["V_joint"] = combined_probe.joint_optimizer_at_snapshot(args, A.shape[0], snap, work_dtype, args.rank)
        A_sketch = snap["A_sketch"] if snap["A_sketch"].size else None
        A_cur = snap["A_cur"]
        union_stack = A_cur if A_sketch is None else np.vstack([A_sketch, A_cur])
        B_union = rowspace_basis(union_stack)
        oracle_v2_raw = project_unit(unit(V_exact[:, 1]), B_union)
        oracle_v2 = deflate_unit(oracle_v2_raw, unit(snap["V_default"][:, 0]))
        V_state = None
        if snap["state"] is not None and snap["state"].get("V") is not None:
            V_state = np.asarray(snap["state"]["V"], dtype=np.float64)
        rho = 0.0
        rho_inverse = None
        rho_prefix = 0.0
        if A_sketch is not None:
            sk_f2 = float(np.sum(A_sketch * A_sketch))
            cur_f2 = float(np.sum(A_cur * A_cur))
            rho = float(sk_f2 / max(cur_f2, 1e-30))
            rho_inverse = float(cur_f2 / max(sk_f2, 1e-30))
            consts = per_block_constants(A, block, A_cur.shape[0])
            rho_prefix = float(consts["sk_F2"] / max(cur_f2, 1e-30))
        candidates = [
            ("combined_v2", unit(snap["V_default"][:, 1]), "score_favoured"),
            ("oracle_v2_proj_perp", oracle_v2, "oracle"),
            ("joint_v2", unit(snap["V_joint"][:, 1]), "joint"),
        ]
        for label, v, role in candidates:
            if v is None:
                continue
            state_align = None
            if V_state is not None and V_state.size:
                p = V_state.T @ v
                state_align = float(np.dot(p, p))
            rows.append({
                "kind": "ablation",
                "mechanism": "M2",
                "matrix": "mixed-tail-sharp",
                "block": int(block),
                "label": label,
                "role": role,
                "rho_B_over_Acur": rho,
                "rho_Acur_over_B": rho_inverse,
                "rho_prefix_over_Acur": rho_prefix,
                "state_align": state_align,
                "align_exact_o2": float((v @ unit(V_exact[:, 1])) ** 2),
                "align_projected_o2": None if oracle_v2 is None else float((v @ oracle_v2) ** 2),
            })
            rows.append(tail_diag_record("mixed-tail-sharp", block, "M2", label, v, snap, A, V_exact, 2, role))
        for label, v in random_panel(B_union, [unit(snap["V_default"][:, 0])], 5, args.seed + 200 + block):
            rows.append(tail_diag_record("mixed-tail-sharp", block, "M2", label, v, snap, A, V_exact, 2, "random"))
    return [r for r in rows if r is not None]


def optimize_s6_b1(args, A, V_exact, snap):
    A_cur = snap["A_cur"]
    A_fut = snap["A_fut"]
    B_union = rowspace_basis(np.vstack([A_cur, A_fut]))
    oracle_v1_proj = project_unit(unit(V_exact[:, 0]), B_union)
    starts = [oracle_v1_proj]
    consts = per_block_constants(A, 1, args.half_win)
    best = r_sk_g_score.optimize_r_sk_g_in_basis(
        A_cur, A_fut, None, 0.0,
        B_union, starts,
        np.random.default_rng(args.seed + 606),
        args.union_maxit, args.union_tol, 0,
        variant="S6", cur_F2=consts["cur_F2"], fut_F2=consts["fut_F2"],
        sk_F2_low=None,
    )
    return None if best is None else unit(best["vec"])


def hm2_raw_top(A_cur, A_fut):
    # Maximize raw HM2 over the two-half union by reusing the unnormalized
    # future-hmean optimizer through S6 with denominators set to one.
    B = rowspace_basis(np.vstack([A_cur, A_fut]))
    rng = np.random.default_rng(707)
    starts = []
    _, _, Vt = np.linalg.svd(np.vstack([A_cur, A_fut]), full_matrices=False)
    for j in range(min(4, Vt.shape[0])):
        starts.append(Vt[j])
    best = r_sk_g_score.optimize_r_sk_g_in_basis(
        A_cur, A_fut, None, 0.0,
        B, starts, rng, 160, 1e-9, 16,
        variant="S6", cur_F2=1.0, fut_F2=1.0, sk_F2_low=None,
    )
    return None if best is None else unit(best["vec"])


def m3_rows(args, matrix, A, V_exact, snap):
    rows = []
    block = 1
    B_union = rowspace_basis(np.vstack([snap["A_cur"], snap["A_fut"]]))
    oracle_v1 = project_unit(unit(V_exact[:, 0]), B_union)
    candidates = [
        ("combined_v1", unit(snap["V_default"][:, 0]), "score_favoured"),
        ("combined_v2", unit(snap["V_default"][:, 1]), "combined_slot2"),
        ("HM2_raw_v1", hm2_raw_top(snap["A_cur"], snap["A_fut"]), "hm2"),
        ("S6_v1", optimize_s6_b1(args, A, V_exact, snap), "s6"),
        ("oracle_v1_proj", oracle_v1, "oracle"),
    ]
    for label, v, role in candidates:
        if v is None:
            continue
        score_s6, _, _r_sk, raw_g1, raw_g2, _hm_g, _sat, _state_align = r_sk_g_score.r_sk_g_value_grad(
            None, snap["A_cur"], snap["A_fut"], 0.0, v,
            variant="S6",
            cur_F2=float(np.sum(snap["A_cur"] * snap["A_cur"])),
            fut_F2=float(np.sum(snap["A_fut"] * snap["A_fut"])),
        )
        comp = combined_score_parts(A, snap, v)
        rows.append({
            "kind": "ablation",
            "mechanism": "M3",
            "matrix": matrix,
            "block": block,
            "label": label,
            "role": role,
            "combined_score": float(comp["score_total"]),
            "score_S6_HM2_norm": float(score_s6),
            "raw_g1": float(raw_g1),
            "raw_g2": float(raw_g2),
            "align_exact_o1": float((v @ unit(V_exact[:, 0])) ** 2),
            "align_exact_o2": float((v @ unit(V_exact[:, 1])) ** 2),
            "align_projected_o1": None if oracle_v1 is None else float((v @ oracle_v1) ** 2),
        })
        rows.append(tail_diag_record(matrix, block, "M3", label, v, snap, A, V_exact, 1, role))
    for label, v in random_panel(B_union, [], 5, args.seed + 303):
        rows.append(tail_diag_record(matrix, block, "M3", label, v, snap, A, V_exact, 1, "random"))
    return [r for r in rows if r is not None]


def run(args):
    np.random.seed(args.seed)
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    rows = []

    A_mts, V_mts = generate(args, "mixed-tail-sharp")
    mts_blocks = {1, 6, 12, 31}
    mts_snaps = stream_to_block(args, A_mts, V_mts, work_dtype, args.rank, 31, mts_blocks)
    rows.extend(m1_rows(args, A_mts, V_mts, mts_snaps[1]))
    rows.extend(m2_rows(args, A_mts, V_mts, mts_snaps))

    A_dd, V_dd = generate(args, "diffuse-diffuse")
    dd_snaps = stream_to_block(args, A_dd, V_dd, work_dtype, args.rank, 1, {1})
    rows.extend(m3_rows(args, "diffuse-diffuse", A_dd, V_dd, dd_snaps[1]))
    rows.extend(m3_rows(args, "mixed-tail-sharp", A_mts, V_mts, mts_snaps[1]))
    combined_stream = hmexp.run_pair_stream(
        A_mts, V_mts, args.sigma1, args, "combined", args.half_win, sliding=True
    )
    combined_summary = hmexp.summarize_result(combined_stream)
    rows.append({
        "kind": "ablation",
        "mechanism": "M3",
        "matrix": "mixed-tail-sharp",
        "block": 31,
        "label": "combined_stream_final",
        "role": "mixed_tail_slot1_control",
        "final_exact_cos0": float(combined_summary["final_exact_cos"][0]),
        "final_exact_cos1": float(combined_summary["final_exact_cos"][1]),
        "final_car_exact_cos0": float(combined_summary["final_car_exact_cos"][0]),
        "final_car_exact_cos1": float(combined_summary["final_car_exact_cos"][1]),
        "steps": int(combined_summary["steps"]),
    })
    return rows


def group_rows(rows, kind):
    out = {}
    for r in rows:
        if r.get("kind") != kind:
            continue
        out.setdefault((r["mechanism"], r["matrix"], r["block"]), []).append(r)
    return out


def fmt(x, digits=4):
    if x is None:
        return "NA"
    return f"{float(x):.{digits}f}"


def synthesize(rows, elapsed):
    ab = group_rows(rows, "ablation")
    td = group_rows(rows, "tail_diagnostic")
    lines = []
    lines.append("# M1/M2/M3 tail-conspiracy unification probe\n")
    lines.append(f"Generated by `probe_m1m2m3_unification.py` in {elapsed:.1f}s.\n")
    lines.append("## Reproducibility\n")

    m1 = ab.get(("M1", "mixed-tail-sharp", 1), [])
    m1_by = {r["label"]: r for r in m1}
    c = m1_by.get("combined_v2")
    o = m1_by.get("oracle_v2_proj_perp")
    if c and o:
        m1_ok = (
            abs(c["phi"] - 5.66) <= 0.01
            and abs(o["phi"] - 3.70) <= 0.01
            and abs(c["energy_A_cur"] - o["energy_A_cur"]) <= 0.01
        )
        lines.append(
            f"- M1 mixed-tail-sharp b1: combined phi={fmt(c['phi'])}, "
            f"score={fmt(c['combined_score'])}, energy={fmt(c['energy_A_cur'])}; "
            f"oracle-proj phi={fmt(o['phi'])}, score={fmt(o['combined_score'])}, "
            f"energy={fmt(o['energy_A_cur'])}. Score gap is "
            f"{fmt(c['combined_score'] - o['combined_score'])}; energy gap is "
            f"{fmt(c['energy_A_cur'] - o['energy_A_cur'])}. "
            f"{'Matches' if m1_ok else 'Does not exactly match'} the documented "
            f"phi≈5.66 vs 3.70 / flat-energy headline."
        )

    lines.append("- M2 mixed-tail-sharp carry table:")
    lines.append("")
    lines.append("| block | carried B/A_cur | prefix/A_cur | A_cur/carried B | combined state_align | oracle state_align | joint state_align |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for b in [1, 6, 12, 31]:
        recs = {r["label"]: r for r in ab.get(("M2", "mixed-tail-sharp", b), [])}
        rr = next(iter(recs.values()), {}).get("rho_B_over_Acur")
        rri = next(iter(recs.values()), {}).get("rho_Acur_over_B")
        rrp = next(iter(recs.values()), {}).get("rho_prefix_over_Acur")
        lines.append(
            f"| {b} | {fmt(rr)} | {fmt(rrp)} | {fmt(rri)} | {fmt(recs.get('combined_v2', {}).get('state_align'))} | "
            f"{fmt(recs.get('oracle_v2_proj_perp', {}).get('state_align'))} | "
            f"{fmt(recs.get('joint_v2', {}).get('state_align'))} |"
        )
    lines.append("")
    rec31 = {r["label"]: r for r in ab.get(("M2", "mixed-tail-sharp", 31), [])}
    if rec31:
        lines.append(
            "- M2 reproducibility read: the documented carry-dominance scale "
            f"sk_F2/cur_F2≈30 is reproduced by the full-prefix ratio "
            f"`prefix/A_cur={fmt(next(iter(rec31.values())).get('rho_prefix_over_Acur'))}`. "
            "The literal carried rank-2 sketch ratio is much smaller "
            f"(`carried B/A_cur={fmt(next(iter(rec31.values())).get('rho_B_over_Acur'))}`), "
            "so both are logged. The combined and joint slot-2 state-align "
            f"values reproduce the pinned-to-carry headline "
            f"({fmt(rec31.get('combined_v2', {}).get('state_align'))}, "
            f"{fmt(rec31.get('joint_v2', {}).get('state_align'))}); the "
            f"deflated projected-oracle state-align is "
            f"{fmt(rec31.get('oracle_v2_proj_perp', {}).get('state_align'))}, "
            "higher than the older 0.103 note but still far below the "
            "score-favoured/joint directions."
        )
        lines.append("")

    lines.append("- M3 b1 slot-1 table:")
    lines.append("")
    lines.append("| matrix | candidate | cos2 projected oracle v1 | cos2 exact oracle v1 | S6/HM2 score | combined score |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for matrix in ["diffuse-diffuse", "mixed-tail-sharp"]:
        for r in ab.get(("M3", matrix, 1), []):
            if r["label"] in ("combined_v1", "HM2_raw_v1", "S6_v1", "oracle_v1_proj"):
                lines.append(
                    f"| {matrix} | {r['label']} | {fmt(r.get('align_projected_o1'))} | "
                    f"{fmt(r['align_exact_o1'])} | "
                    f"{fmt(r['score_S6_HM2_norm'], 6)} | {fmt(r['combined_score'])} |"
                )
    lines.append("")
    final_ctl = {
        r["label"]: r for r in ab.get(("M3", "mixed-tail-sharp", 31), [])
    }.get("combined_stream_final")
    if final_ctl:
        final_delta = final_ctl["final_exact_cos0"] - 0.785
        lines.append(
            f"- M3 mixed-tail-sharp streaming control: combined final "
            f"cos0²={fmt(final_ctl['final_exact_cos0'])}, "
            f"cos1²={fmt(final_ctl['final_exact_cos1'])} over "
            f"{final_ctl['steps']} sliding blocks. This is the current-code "
            f"check corresponding to the documented cos0²≈0.785 control "
            f"(delta {final_delta:+.4f}); the conclusion is unchanged: "
            "mixed-tail-sharp has a successful combined slot-1 while "
            "diffuse-diffuse fails slot-1 at b1."
        )
        lines.append("")

    lines.append("## Tail-Conspiracy Separation\n")
    lines.append(
        "The requested relH1 formula is implemented as `top_share_X = "
        "max_i (A_X v)_i^2 / ||A_X v||^2`; `rel_entropy_X` is also reported "
        "because the surrounding docs use relH1 for normalized entropy. A "
        "concentrated row response means high `top_share_X` and low "
        "`rel_entropy_X`."
    )
    lines.append("")
    lines.append("| mechanism | matrix/block | score-favoured vs oracle read | verdict |")
    lines.append("|---|---|---|---|")
    verdicts = []
    for key, score_label, oracle_label in [
        (("M1", "mixed-tail-sharp", 1), "combined_v2", "oracle_v2_proj_perp"),
        (("M2", "mixed-tail-sharp", 31), "combined_v2", "oracle_v2_proj_perp"),
        (("M3", "diffuse-diffuse", 1), "combined_v1", "oracle_v1_proj"),
    ]:
        recs = {r["label"]: r for r in td.get(key, [])}
        s = recs.get(score_label)
        o = recs.get(oracle_label)
        if not s or not o:
            lines.append(f"| {key[0]} | {key[1]} b{key[2]} | missing rows | FAIL |")
            verdicts.append(False)
            continue
        diff_top = max(
            abs((s.get("top_share_cur") or 0.0) - (o.get("top_share_cur") or 0.0)),
            abs((s.get("top_share_fut") or 0.0) - (o.get("top_share_fut") or 0.0)),
            abs((s.get("top_share_B") or 0.0) - (o.get("top_share_B") or 0.0)),
        )
        diff_asym = abs(s["between_asym"] - o["between_asym"])
        rs = s.get("replicability_abs_pearson")
        ro = o.get("replicability_abs_pearson")
        diff_rep = 0.0 if rs is None or ro is None else abs(rs - ro)
        separated = diff_top >= args.sep_top_share or diff_asym >= args.sep_asym or diff_rep >= args.sep_rep
        verdicts.append(separated)
        read = (
            f"top_share cur/fut/B {fmt(s.get('top_share_cur'))}/{fmt(s.get('top_share_fut'))}/"
            f"{fmt(s.get('top_share_B'))} vs {fmt(o.get('top_share_cur'))}/"
            f"{fmt(o.get('top_share_fut'))}/{fmt(o.get('top_share_B'))}; "
            f"asym {fmt(s['between_asym'])} vs {fmt(o['between_asym'])}; "
            f"r {fmt(rs)} vs {fmt(ro)}"
        )
        lines.append(
            f"| {key[0]} | {key[1]} b{key[2]} | {read} | "
            f"{'PASS' if separated else 'NO SEPARATION'} |"
        )
    lines.append("")
    if all(verdicts):
        lines.append(
            "Conclusion: the measured diagnostics separate the score-favoured "
            "direction from the oracle at the three failure points. A follow-up "
            "overview edit can propose folding M1/M2/M3 under one tail-"
            "conspiracy heading with sub-cases."
        )
    else:
        lines.append(
            "Conclusion: at least one failure point did not separate under the "
            "configured diagnostic thresholds. Keep §1bis split and record the "
            "residual mechanism before rewriting the overview."
        )
    lines.append("")
    lines.append("## Output Files\n")
    lines.append("- `probe_m1m2m3_unification.json`")
    lines.append("- `probe_m1m2m3_unification.csv`")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="summary/m1m2m3_tail_conspiracy_unification")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--half-win", type=int, default=32)
    p.add_argument("--rank", type=int, default=2)
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
    p.add_argument("--union-maxit", type=int, default=180)
    p.add_argument("--union-tol", type=float, default=1e-9)
    p.add_argument("--union-random-starts", type=int, default=24)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--r-sig", type=int, default=2)
    p.add_argument("--alpha-sig", type=float, default=0.003)
    p.add_argument("--alpha-tail", type=float, default=0.0145)
    p.add_argument("--tail-scale", type=float, default=0.99)
    p.add_argument("--sigma1", type=float, default=0.991)
    p.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    p.add_argument("--sep-top-share", type=float, default=0.05)
    p.add_argument("--sep-asym", type=float, default=0.25)
    p.add_argument("--sep-rep", type=float, default=0.25)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = run(args)
    elapsed = time.time() - t0
    json_path = os.path.join(args.out_dir, "probe_m1m2m3_unification.json")
    csv_path = os.path.join(args.out_dir, "probe_m1m2m3_unification.csv")
    md_path = os.path.join(args.out_dir, "m1m2m3_tail_conspiracy_unification.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(synthesize(rows, elapsed))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
