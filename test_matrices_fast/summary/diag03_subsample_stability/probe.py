"""DIAG-03 subsample-stability probe.

Backlog: summary/overview/score_family_workflow.txt [DIAG-03] (~line 1742).
Resolves Q15 (score_design_overview.txt) and toolkit §8(p).
Cross-blocks: gates [FAM-09] (stability-weighted evidence).

Goal:
    For each (matrix, block, slot, candidate), estimate row-subsample
    mean / variance of evidence (u_g1) under random row subsampling at
    p in {0.50, 0.75}. Then ask whether subsample instability predicts
    (a) next-window replication failure or (b) carry-alignment decay
    BETTER than raw u_X (= u_g1) or relH1 alone.

Candidate panel (per slot):
    oracle_v_proj   ←  V_exact[:, k] projected into B_union
    s6_opt          ←  c_evi_v[k]  (the F-weighted HM-evi optimizer; also
                       the S6 optimizer-in-basis used by f_hm_score)
    mgain_svd       ←  top-k right SV of M_gain = [B_top; A_cur]
                       (also the `combined_v[k]` produced by the
                        streaming combined-step optimizer, in practice;
                        we keep both for clarity)
    combined        ←  V_default[:, k]  (combined-step optimizer)
    rowcheat        ←  rowcheat_v[k]   (rank-r row-cheat baseline)
    sketch          ←  state.V[:, k]    (carry direction; only for k>=0
                                        and only when sketch present)
    [high-scoring non-oracle candidates: in the score-family, the
     primary high-scoring non-oracle candidate is `combined`/`mgain_svd`
     (when score peak misses the oracle). We tag it by slot via the
     winner among combined / mgain_svd that is NOT oracle.]

Outcome metrics (per candidate v at block b, slot k):
    rep_fail(b)        := |u_g1_next - u_g2_cur|
                          (i.e., how much the next-window cur differs
                           from the predicted A_fut response on v)
    rep_fail_norm(b)   := rep_fail / max(u_g2_cur, eps)
    carry_decay(b)     := 1 - max_j cos^2(v, V_state^{b+1}[:, j])
                          (i.e., 1 - alignment to next block's carry
                          frame; high = `v` is not retained.)

Subsample-instability metrics (per candidate v at block b, slot k):
    For a fixed sampling fraction p in {0.5, 0.75}, draw 30 random row
    subsets S_t of A_cur and 30 of A_fut. For each:
        u_g1_S = ||A_cur[S] v||^2 / ||A_cur[S]||_F^2
        u_g2_S = ||A_fut[S] v||^2 / ||A_fut[S]||_F^2
    We track (mean, std, var, IQR, CV) of {u_g1_S} and {u_g2_S}, and
    a JOINT instability metric:
        instab_g1g2_p := std(u_g1_S - u_g2_S) / max(mean(...), eps)
    The reasoning: a candidate that has well-replicated current vs
    future evidence WHEN sampled the same rows out, but not otherwise,
    is "fragile" in the row-subsample sense.

    We also report relH1 of A_cur on the unit candidate (as a baseline
    predictor; it's the alternative scalar already in the score-family).

Verdict rule (per the backlog acceptance):
    Compute Pearson r over the (matrix, block, slot, candidate) flat
    table for each of the four predictors {u_g1, relH1, instab_g1_p50,
    instab_g1_p75, instab_g1g2_p50, instab_g1g2_p75} against each of
    the two outcomes {rep_fail_norm, carry_decay}. If max_p |r_instab|
    > max(|r_u_g1|, |r_relH1|) + 0.10 on at least one outcome and is
    consistent in sign, declare SIGNAL. If the gap is in [-0.05, +0.10]
    declare WEAK-SIGNAL. Otherwise NO-SIGNAL.

Output:
    summary/diag03_subsample_stability/raw.csv
    summary/diag03_subsample_stability/synthesis.md

Run from test_matrices_fast/:
    python summary/diag03_subsample_stability/probe.py
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

# Make repo-root imports work when this is invoked from test_matrices_fast/.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import cex_restricted_space_probe as probe
from frob_hm3_score_diagnostic import collect_candidates as fhm3_collect_candidates
from frob_norm_diagnostic import collect_candidates as fnorm_collect_candidates
from hmean_evidence_score import (
    entropy_relH1_value_grad,
    per_block_constants,
    stream_to_block,
)


# §6 high-entropy + reference suite. Per the backlog: pick >=4 covering both
# high- and low-entropy regimes. Here we cover all 8 §6 matrices for
# completeness; the verdict aggregation handles the size.
DEFAULT_MATRICES = [
    "etf-basket-basis",
    "residual-spiky-shocks",
    "risk-residual-panel",
    "diffuse-diffuse",
    "static-cex",
    "mixed-tail-sharp",
    "mixed-tail-balanced",
    "mixed-tail-soft",
]

# DIAG-03 spec: blocks {1, 6, 12, 31} at half_win=32. We also need the
# next-block snapshot for the outcome metrics (replication / carry decay),
# so we report on {1,2,6,7,12,13,31,32} and only use 1/6/12/31 as the
# anchor blocks.
ANCHOR_BLOCKS = [1, 6, 12, 31]
SUBSAMPLE_FRACTIONS = (0.5, 0.75)
N_SUBSAMPLES = 30
SLOTS = (0, 1)  # slot-1 (idx 0) and slot-2 (idx 1)
EPS = 1e-30


# ---------------------------------------------------------------------------
# Candidate panel
# ---------------------------------------------------------------------------


def _unit(v):
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    nv = float(np.linalg.norm(v))
    if nv <= 1e-30:
        return None
    return v / nv


def collect_panel(args, snap, V_exact, slot, c_sk, c_g1, c_g2):
    """Return {label: unit-vector v} for the named slot.

    The panel covers (per the DIAG-03 spec):
      oracle, s6_opt, mgain, combined, rowcheat, sketch (if present).
    """
    fnorm = fnorm_collect_candidates(snap, V_exact)
    fhm3, _, _ = fhm3_collect_candidates(args, snap, V_exact, c_sk, c_g1, c_g2)

    suffix = "v1" if slot == 0 else "v2"
    panel = {
        "oracle":   fnorm.get(f"oracle_{suffix}_proj"),
        "s6_opt":   fhm3.get(f"c_evi_{suffix}"),
        "mgain":    fnorm.get(f"mgain_svd_{suffix}"),
        "combined": fnorm.get(f"combined_{suffix}"),
        "rowcheat": fhm3.get(f"rowcheat_{suffix}"),
        "sketch":   fnorm.get(f"sketch_{suffix}"),
    }
    return {k: _unit(v) for k, v in panel.items() if v is not None}


# ---------------------------------------------------------------------------
# Subsample-stability statistics
# ---------------------------------------------------------------------------


def _u_full(A, v):
    """Return ||A v||^2 / ||A||_F^2 for unit v on dense A. NaN if A empty."""
    if A is None or A.size == 0 or v is None:
        return float("nan")
    y = A @ v
    F2 = float(np.sum(A * A))
    return float(np.dot(y, y)) / max(F2, EPS)


def _subsample_u(A, v, frac, n_draws, rng):
    """Return array of length n_draws of u_p values on row subsets of A."""
    if A is None or A.size == 0 or v is None:
        return np.full(n_draws, np.nan)
    m = A.shape[0]
    k = max(1, int(round(frac * m)))
    out = np.empty(n_draws, dtype=np.float64)
    for t in range(n_draws):
        idx = rng.choice(m, size=k, replace=False)
        AS = A[idx]
        y = AS @ v
        F2 = float(np.sum(AS * AS))
        out[t] = float(np.dot(y, y)) / max(F2, EPS)
    return out


def _summary_stats(arr):
    """Return dict of mean/std/var/iqr/cv for a numeric array (NaN-safe)."""
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "var": float("nan"), "iqr": float("nan"),
                "cv": float("nan")}
    mu = float(a.mean())
    sd = float(a.std(ddof=0))
    var = float(a.var(ddof=0))
    q1, q3 = np.percentile(a, [25.0, 75.0])
    iqr = float(q3 - q1)
    cv = sd / abs(mu) if abs(mu) > EPS else float("nan")
    return {"mean": mu, "std": sd, "var": var, "iqr": iqr, "cv": cv}


# ---------------------------------------------------------------------------
# Outcome metrics: replication failure & carry-alignment decay
# ---------------------------------------------------------------------------


def _carry_V_next(snap_next):
    """Return V_state of the NEXT block's carry, shape (d, r), or None."""
    if snap_next is None:
        return None
    state = snap_next.get("state")
    if state is None:
        return None
    Vs = state.get("V")
    if Vs is None:
        return None
    Vs = np.asarray(Vs, dtype=np.float64)
    return Vs if Vs.size else None


def _carry_decay(v, V_state_next):
    """1 - max_j cos^2(v, V_state_next[:, j]). NaN if next state missing."""
    if v is None or V_state_next is None or V_state_next.size == 0:
        return float("nan")
    proj = V_state_next.T @ v
    return float(max(0.0, 1.0 - float(np.max(proj * proj))))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_matrix(args, matrix, anchor_blocks, rng_master):
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=matrix, n=args.n, preset=args.preset, seed=args.seed,
        r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
        shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, np.float64)
    V_exact = np.asarray(V_exact, np.float64)

    # Need anchor blocks plus their +1 neighbors for outcome metrics.
    blocks_needed = set(anchor_blocks) | {b + 1 for b in anchor_blocks}
    target = max(blocks_needed)
    snaps = stream_to_block(args, A, V_exact, work_dtype, int(args.rank), target,
                            blocks_needed)

    rows_out = []
    for b in anchor_blocks:
        if b not in snaps:
            continue
        snap = snaps[b]
        snap_next = snaps.get(b + 1)
        V_state_next = _carry_V_next(snap_next)

        consts = per_block_constants(A, b, args.half_win)
        c_sk = consts["c_sk"]
        c_g1 = consts["c_g1"]
        c_g2 = consts["c_g2"]
        cur_F2 = consts["cur_F2"]
        fut_F2 = consts["fut_F2"]
        A_cur = snap["A_cur"]
        A_fut = snap["A_fut"]

        # Per-slot panel
        for slot in SLOTS:
            panel = collect_panel(args, snap, V_exact, slot, c_sk, c_g1, c_g2)

            # Outcome at b: u_g2 = ||A_fut v||^2 / fut_F2 (PRED) and at next
            # block: u_g1_next = ||A_cur^{b+1} v||^2 / cur_F2^{b+1} (REAL).
            consts_next = (
                per_block_constants(A, b + 1, args.half_win) if snap_next else None
            )
            A_cur_next = consts_next["A_cur"] if consts_next else None
            cur_F2_next = consts_next["cur_F2"] if consts_next else float("nan")

            for label, v in panel.items():
                # Raw evidence at this block (full A_cur / A_fut).
                u_g1 = _u_full(A_cur, v)
                u_g2 = _u_full(A_fut, v)
                # relH1 of A_cur on v.
                rel, _ = entropy_relH1_value_grad(A_cur, v)

                # Next-window replication: predicted u_g2 vs realized u_g1_next.
                if A_cur_next is not None:
                    yn = A_cur_next @ v
                    u_g1_next = float(np.dot(yn, yn)) / max(cur_F2_next, EPS)
                    rep_fail = abs(u_g1_next - u_g2)
                    rep_fail_norm = rep_fail / max(u_g2, EPS)
                else:
                    u_g1_next = float("nan")
                    rep_fail = float("nan")
                    rep_fail_norm = float("nan")

                # Carry-alignment decay using NEXT block's V_state.
                decay = _carry_decay(v, V_state_next)

                # Subsample-stability statistics at p=0.5 and 0.75.
                stab = {}
                rng = np.random.default_rng(
                    int(rng_master.integers(0, 2**31 - 1))
                )
                for p in SUBSAMPLE_FRACTIONS:
                    g1_arr = _subsample_u(A_cur, v, p, args.n_subsamples, rng)
                    g2_arr = _subsample_u(A_fut, v, p, args.n_subsamples, rng)
                    diff_arr = g1_arr - g2_arr
                    s_g1 = _summary_stats(g1_arr)
                    s_g2 = _summary_stats(g2_arr)
                    s_d = _summary_stats(diff_arr)
                    pname = f"p{int(round(100*p)):02d}"
                    stab[pname] = {"g1": s_g1, "g2": s_g2, "diff": s_d}

                row = {
                    "matrix": matrix,
                    "block": int(b),
                    "slot": int(slot + 1),
                    "candidate": label,
                    "u_g1": u_g1,
                    "u_g2": u_g2,
                    "u_g1_next": u_g1_next,
                    "rel_h1": rel,
                    "rep_fail": rep_fail,
                    "rep_fail_norm": rep_fail_norm,
                    "carry_decay": decay,
                }
                for pname, dct in stab.items():
                    for which in ("g1", "g2", "diff"):
                        s = dct[which]
                        for stat_key, val in s.items():
                            row[f"{which}_{pname}_{stat_key}"] = val
                rows_out.append(row)
    return rows_out


# ---------------------------------------------------------------------------
# Synthesis (correlation analysis + verdict)
# ---------------------------------------------------------------------------


def _safe_pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan"), int(mask.sum())
    xs = x[mask] - x[mask].mean()
    ys = y[mask] - y[mask].mean()
    nx = float(np.linalg.norm(xs))
    ny = float(np.linalg.norm(ys))
    if nx <= EPS or ny <= EPS:
        return float("nan"), int(mask.sum())
    return float(np.dot(xs, ys) / (nx * ny)), int(mask.sum())


def _correlation_table(rows):
    """Return dict outcome -> dict predictor -> (r, n)."""
    outcomes = ["rep_fail_norm", "carry_decay"]
    predictors = [
        "u_g1", "rel_h1",
        "g1_p50_cv", "g1_p75_cv",
        "g2_p50_cv", "g2_p75_cv",
        "diff_p50_std", "diff_p75_std",
        "diff_p50_cv", "diff_p75_cv",
    ]
    out = {o: {} for o in outcomes}
    for o in outcomes:
        ys = [r.get(o, float("nan")) for r in rows]
        for p in predictors:
            xs = [r.get(p, float("nan")) for r in rows]
            r, n = _safe_pearson(xs, ys)
            out[o][p] = (r, n)
    return out


def _verdict(corr):
    """Decide SIGNAL / WEAK-SIGNAL / NO-SIGNAL per the rule in the docstring."""
    base_keys = ["u_g1", "rel_h1"]
    instab_keys = [
        "g1_p50_cv", "g1_p75_cv",
        "g2_p50_cv", "g2_p75_cv",
        "diff_p50_cv", "diff_p75_cv",
        "diff_p50_std", "diff_p75_std",
    ]

    summary = {}
    best_gap = -np.inf
    best_kind = "NO-SIGNAL"
    for outcome, preds in corr.items():
        base_max = 0.0
        for k in base_keys:
            r, _ = preds.get(k, (float("nan"), 0))
            if np.isfinite(r):
                base_max = max(base_max, abs(r))
        instab_max = 0.0
        instab_argmax = None
        for k in instab_keys:
            r, _ = preds.get(k, (float("nan"), 0))
            if np.isfinite(r) and abs(r) > instab_max:
                instab_max = abs(r)
                instab_argmax = (k, r)
        gap = instab_max - base_max
        summary[outcome] = {
            "base_max": base_max, "instab_max": instab_max,
            "gap": gap, "instab_argmax": instab_argmax,
        }
        if gap > best_gap:
            best_gap = gap
        if gap > 0.10:
            best_kind = "SIGNAL"
        elif gap > -0.05 and best_kind != "SIGNAL":
            if best_kind == "NO-SIGNAL":
                best_kind = "WEAK-SIGNAL"

    return best_kind, best_gap, summary


def write_synthesis(rows, corr, verdict, gap, vsummary, out_md, args):
    lines = []
    lines.append("# DIAG-03 subsample-stability diagnostic — synthesis")
    lines.append("")
    lines.append(f"Probe: `summary/diag03_subsample_stability/probe.py`  ")
    lines.append(f"Matrices: {', '.join(args.matrices)}  ")
    lines.append(f"Blocks (anchor): {ANCHOR_BLOCKS}; half_win={args.half_win}; "
                 f"slots: 1, 2  ")
    lines.append(f"Subsample draws per (frac, candidate): {args.n_subsamples}  ")
    lines.append(f"Subsample fractions: {list(SUBSAMPLE_FRACTIONS)}  ")
    lines.append(f"Total candidate-rows: {len(rows)}")
    lines.append("")
    lines.append("## Mean / variance / IQR by (matrix, block, slot, candidate)")
    lines.append("")
    lines.append("Full per-row dump in `raw.csv`. Below: summary aggregates of")
    lines.append("g1 (cur-window) sub-sample mean / CV at p=0.50 across all")
    lines.append("matrix-block-slot rows, broken down by candidate.")
    lines.append("")
    lines.append("| candidate | n | mean(g1_p50_mean) | median(g1_p50_cv) | "
                 "median(g2_p50_cv) | median(diff_p50_cv) |")
    lines.append("|---|---|---|---|---|---|")
    by_cand = {}
    for r in rows:
        by_cand.setdefault(r["candidate"], []).append(r)
    for cand, rs in sorted(by_cand.items()):
        def col(key):
            arr = np.asarray([x.get(key, np.nan) for x in rs])
            arr = arr[np.isfinite(arr)]
            return float(np.median(arr)) if arr.size else float("nan")
        m_mean = col("g1_p50_mean")
        m_cv1 = col("g1_p50_cv")
        m_cv2 = col("g2_p50_cv")
        m_cvd = col("diff_p50_cv")
        lines.append(f"| {cand} | {len(rs)} | {m_mean:.4g} | {m_cv1:.4g} | "
                     f"{m_cv2:.4g} | {m_cvd:.4g} |")
    lines.append("")
    lines.append("## Correlation: predictor -> outcome (Pearson r)")
    lines.append("")
    lines.append("Predictors: raw u_g1, rel_h1, and subsample-instability metrics.")
    lines.append("Outcomes: `rep_fail_norm` (next-window replication error,")
    lines.append("|u_g1_next - u_g2| / u_g2) and `carry_decay` (1 -")
    lines.append("max_j cos^2(v, V_state^{b+1}[:, j])).")
    lines.append("")
    for outcome, preds in corr.items():
        lines.append(f"### outcome = `{outcome}`")
        lines.append("")
        lines.append("| predictor | Pearson r | n |")
        lines.append("|---|---|---|")
        for k in [
            "u_g1", "rel_h1",
            "g1_p50_cv", "g1_p75_cv",
            "g2_p50_cv", "g2_p75_cv",
            "diff_p50_cv", "diff_p75_cv",
            "diff_p50_std", "diff_p75_std",
        ]:
            r, n = preds.get(k, (float("nan"), 0))
            lines.append(f"| `{k}` | {r:+.3f} | {n} |")
        lines.append("")

    lines.append("## Verdict-rule numbers")
    lines.append("")
    for outcome, s in vsummary.items():
        argmax = s["instab_argmax"]
        argmax_str = (
            f"{argmax[0]}={argmax[1]:+.3f}" if argmax is not None else "n/a"
        )
        lines.append(
            f"- `{outcome}`: base_max(|r|) over u_g1/rel_h1 = {s['base_max']:.3f}; "
            f"best instability |r| = {s['instab_max']:.3f} at {argmax_str}; "
            f"gap = {s['gap']:+.3f}."
        )
    lines.append("")
    lines.append(f"## VERDICT: **{verdict}**  (best gap = {gap:+.3f})")
    lines.append("")
    lines.append("Rule: SIGNAL if gap > +0.10 on at least one outcome; "
                 "WEAK-SIGNAL if -0.05 < gap <= +0.10; else NO-SIGNAL.")
    lines.append("")

    if verdict == "SIGNAL":
        lines.append("## FAM-09 unblock recommendation")
        lines.append("")
        lines.append("UNBLOCK FAM-09. Subsample instability adds predictive")
        lines.append("information beyond raw u_X / relH1 on at least one of")
        lines.append("{next-window replication, carry-alignment decay}.")
        lines.append("Suggested wiring: u_X -> u_X / (1 + lambda * CV(u_X|sub))")
        lines.append("with lambda calibrated on the high-entropy §6 matrices.")
        lines.append("Acceptance gate per the FAM-09 backlog still applies")
        lines.append("(gradient check or derivative-free; improve a")
        lines.append("high-entropy failure without §6 regression).")
    else:
        lines.append("## FAM-09 kill recommendation")
        lines.append("")
        if verdict == "WEAK-SIGNAL":
            lines.append("WEAK-SIGNAL. Subsample instability is correlated")
            lines.append("with the outcomes but does not exceed the raw")
            lines.append("u_X / rel_h1 baseline by a meaningful margin")
            lines.append("(gap is in (-0.05, +0.10]). Recommend KILL FAM-09")
            lines.append("under the workflow §1ter cost/value rule: a noisy")
            lines.append("subsample weighting introduces optimization")
            lines.append("complexity (variance estimates are not analytically")
            lines.append("differentiable; FAM-09 itself notes 'expensive and")
            lines.append("noisy') without a clear win over predictors that")
            lines.append("are already in the score.")
        else:
            lines.append("NO-SIGNAL. Subsample instability does not predict")
            lines.append("either next-window replication failure or carry-")
            lines.append("alignment decay better than raw u_X / rel_h1.")
            lines.append("Recommend KILL FAM-09 (workflow §5 acceptance:")
            lines.append("'must show signal on at least one high-entropy")
            lines.append("failure matrix' is not met).")
    lines.append("")
    lines.append("## Toolkit promotion note")
    lines.append("")
    lines.append("Per workflow §6 handoff checklist: any new infra built in a")
    lines.append("diagnostic should become a permanent toolkit entry if it is")
    lines.append("reusable. The subsample-stability sampler in `probe.py`")
    if verdict == "SIGNAL":
        lines.append("(`_subsample_u` + `_summary_stats`) IS reusable: it can")
        lines.append("be called from any candidate-vs-evidence audit. Promote")
        lines.append("`diagnostic_toolkit.txt §8(p)` from NOT BUILT -> SHIPPED")
        lines.append("with this probe as the canonical reference, and keep")
        lines.append("the helpers available for FAM-09 prototyping.")
    else:
        lines.append("(`_subsample_u` + `_summary_stats`) is reusable but the")
        lines.append("verdict above does not justify a permanent score-side")
        lines.append("integration. Recommend marking toolkit §8(p) as")
        lines.append("`SHIPPED-AS-DIAGNOSTIC, NEGATIVE-RESULT` rather than a")
        lines.append("permanent toolkit entry: keep this probe in the repo as")
        lines.append("the canonical proof-of-negative for Q15, but do not")
        lines.append("re-run by default in the score-family pipeline.")
    lines.append("")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()},
                    key=lambda x: (x not in ("matrix", "block", "slot",
                                             "candidate"), x))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrices", nargs="+", default=DEFAULT_MATRICES)
    p.add_argument("--anchor-blocks", nargs="+", type=int,
                   default=ANCHOR_BLOCKS)
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
    p.add_argument("--union-maxit", type=int, default=120)
    p.add_argument("--union-tol", type=float, default=1e-9)
    p.add_argument("--union-random-starts", type=int, default=24)
    p.add_argument("--r-sig", type=int, default=2)
    p.add_argument("--alpha-sig", type=float, default=0.003)
    p.add_argument("--alpha-tail", type=float, default=0.0145)
    p.add_argument("--tail-scale", type=float, default=0.99)
    p.add_argument("--sigma1", type=float, default=0.991)
    p.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    p.add_argument("--n-subsamples", type=int, default=N_SUBSAMPLES)
    p.add_argument("--out-dir",
                   default="summary/diag03_subsample_stability")
    p.add_argument("--rng-seed", type=int, default=20260429)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rng_master = np.random.default_rng(args.rng_seed)
    all_rows = []
    t0 = time.time()
    for matrix in args.matrices:
        print(f"[diag-03] running {matrix} ...", flush=True)
        t1 = time.time()
        rows = run_matrix(args, matrix, args.anchor_blocks, rng_master)
        all_rows.extend(rows)
        print(f"  {matrix}: {len(rows)} candidate-rows  "
              f"(elapsed={time.time() - t1:.2f}s)", flush=True)

    csv_path = os.path.join(args.out_dir, "raw.csv")
    write_csv(csv_path, all_rows)
    print(f"[diag-03] wrote {csv_path}", flush=True)

    corr = _correlation_table(all_rows)
    verdict, gap, vsummary = _verdict(corr)
    md_path = os.path.join(args.out_dir, "synthesis.md")
    write_synthesis(all_rows, corr, verdict, gap, vsummary, md_path, args)
    print(f"[diag-03] wrote {md_path}", flush=True)
    print(f"[diag-03] VERDICT = {verdict}  (best gap = {gap:+.3f})", flush=True)
    print(f"[diag-03] total elapsed {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
