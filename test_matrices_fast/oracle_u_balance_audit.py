"""Oracle u-balance audit (DIAG-04).

Per backlog item DIAG-04 in summary/overview/score_family_workflow.txt §5
(also closes diagnostic_toolkit.txt §8(q)), this audit walks blocks
1, 2, 6, 12, 31 for each matrix in the §6 suite and dumps the F-weighted
u-components for the projected oracle directions:

    u_sk(v) = ||A_sketch v||^2 / sk_F2_low      (rank-r CARRY F-norm²)
    u_g1(v) = ||A_cur   v||^2 / cur_F2
    u_g2(v) = ||A_fut   v||^2 / fut_F2

evaluated at v = oracle_v_k_proj for k ∈ {1, 2}.

Imbalance ratios reported per row:
    ratio_max  = max(u_sk, u_g1, u_g2) / min(u_sk, u_g1, u_g2)
    ratio_skg1 = u_sk / u_g1
    ratio_skg2 = u_sk / u_g2
    ratio_g1g2 = u_g1 / u_g2

Background: HM3's "smallest-link enforcer" property only fires correctly
if the oracle direction itself has roughly balanced u's. If oracle is
imbalanced under the chosen weights, HM3(oracle) is dominated by
oracle's smallest u, and any non-oracle direction with mediocre but
balanced u's can outscore oracle on HM3. This audit quantifies the
imbalance at scale across the §6 suite.

See score_design_overview.txt §1quater (M4 mechanism) and §2bis (b.iii)
for the framing; this script is the diagnostic that gates AB-03 phase 2
(value-only re-weighting search).

Output:
    summary/infra_oracle_u_balance/{matrix}_audit.csv
    summary/infra_oracle_u_balance/audit_summary.csv
    summary/infra_oracle_u_balance/synthesis.md

Run from test_matrices_fast/:
    python oracle_u_balance_audit.py
"""

import argparse
import csv
import math
import os
import time

import numpy as np

import cex_restricted_space_probe as probe
from frob_norm_diagnostic import collect_candidates as fnorm_collect_candidates
from hmean_evidence_score import per_block_constants, stream_to_block


DEFAULT_MATRICES = [
    "mixed-tail-sharp",
    "mixed-tail-balanced",
    "mixed-tail-soft",
    "static-cex",
    "diffuse-diffuse",
    "etf-basket-basis",
    "residual-spiky-shocks",
    "risk-residual-panel",
]

DEFAULT_BLOCKS = [1, 2, 6, 12, 31]

# We probe both slot-1 and slot-2 oracle projections — slot-2 is where
# S6 actually fails, but slot-1 imbalance is the single-clearest signal
# and lines up with the empirical sample we already have from INFRA-07.
ORACLE_SLOTS = [
    ("oracle_v1_proj", 1),
    ("oracle_v2_proj", 2),
]

# Approximate S6 cos[k]² from summary/bench_matrix_sweep_r_sk_g_S6/
# synthesis.txt at half_win=32 block 31 (used only in synthesis.md for
# correlation read; numerical truth still lives in the bench output).
S6_COS_BENCH = {
    "mixed-tail-sharp":      {"cos0": 0.733,  "cos1": 0.013},
    "mixed-tail-balanced":   {"cos0": 0.687,  "cos1": 0.0004},
    "mixed-tail-soft":       {"cos0": 0.847,  "cos1": 0.022},
    "static-cex":            {"cos0": 0.936,  "cos1": 0.025},
    "diffuse-diffuse":       {"cos0": 0.749,  "cos1": 0.005},
    "etf-basket-basis":      {"cos0": 1.000,  "cos1": 0.652},
    "residual-spiky-shocks": {"cos0": 0.333,  "cos1": 0.266},
    # risk-residual-panel: S6 not in the §6 table; left blank.
}


# --------------------------------------------------------------------------
# Per-block component computation
# --------------------------------------------------------------------------


def _ray_sq(A, v):
    if A is None or v is None or A.size == 0:
        return float("nan")
    y = A @ v
    return float(np.dot(y, y))


def _top_r_svd(A, r):
    """Return (V_top (n,k), s_top (k,)) where k=min(r, rank(A))."""
    if A is None or A.size == 0:
        return None, None
    Aw = np.asarray(A, dtype=np.float64)
    # Cheap path: full SVD then truncate (windows are small: half_win × n).
    _, s, Vt = np.linalg.svd(Aw, full_matrices=False)
    k = min(int(r), s.size)
    return Vt[:k].T, s[:k]


def _per_direction_w(V_top, s_top, v, fallback):
    """E2 weight: sigma[k]² where k=argmax_i (V_top[:,i]^T v)². Fallback if empty."""
    if V_top is None or s_top is None or s_top.size == 0:
        return fallback
    proj = V_top.T @ v
    k = int(np.argmax(proj * proj))
    return float(s_top[k] ** 2)


def _u_components(snap, v, sk_F2_low, cur_F2, fut_F2,
                  weight_scheme="current", rank=2,
                  cur_svd=None, fut_svd=None):
    """Return (u_sk, u_g1, u_g2, raw_sk, raw_g1, raw_g2) for unit-norm v.

    weight_scheme:
        "current" — F-weighting (||A||_F²) baseline.
        "E1"      — operator-norm cap: w = sigma_top(·)².
        "E2"      — alignment-weighted: w(v) = sigma[k(v)]² where k(v) =
                    argmax_i (V_top[:,i]^T v)².
    """
    A_cur = snap["A_cur"]
    A_fut = snap["A_fut"]
    A_sketch_arr = snap["A_sketch"]
    A_sketch = A_sketch_arr if A_sketch_arr.size else None
    state = snap.get("state")

    if v is None:
        return (float("nan"),) * 6

    eps = 1e-30
    raw_g1 = _ray_sq(A_cur, v)
    raw_g2 = _ray_sq(A_fut, v)

    if weight_scheme == "current":
        w_g1 = max(cur_F2, eps)
        w_g2 = max(fut_F2, eps)
    elif weight_scheme == "E1":
        # Top sigma squared per window.
        w_g1 = float(cur_svd[1][0] ** 2) if cur_svd and cur_svd[1] is not None and cur_svd[1].size else max(cur_F2, eps)
        w_g2 = float(fut_svd[1][0] ** 2) if fut_svd and fut_svd[1] is not None and fut_svd[1].size else max(fut_F2, eps)
    elif weight_scheme == "E2":
        V_cur, s_cur = (cur_svd if cur_svd is not None else (None, None))
        V_fut, s_fut = (fut_svd if fut_svd is not None else (None, None))
        w_g1 = _per_direction_w(V_cur, s_cur, v, max(cur_F2, eps))
        w_g2 = _per_direction_w(V_fut, s_fut, v, max(fut_F2, eps))
    else:
        raise ValueError(f"unknown weight_scheme {weight_scheme!r}")

    u_g1 = raw_g1 / max(w_g1, eps)
    u_g2 = raw_g2 / max(w_g2, eps)

    if A_sketch is None or sk_F2_low <= eps:
        u_sk = float("nan")
        raw_sk = float("nan")
    else:
        raw_sk = _ray_sq(A_sketch, v)
        if weight_scheme == "current":
            w_sk = sk_F2_low
        elif weight_scheme == "E1":
            if state is not None and "s" in state and np.asarray(state["s"]).size:
                w_sk = float(np.asarray(state["s"], dtype=np.float64)[0] ** 2)
            else:
                w_sk = sk_F2_low
        elif weight_scheme == "E2":
            if state is not None and "V" in state and "s" in state and np.asarray(state["s"]).size:
                V_st = np.asarray(state["V"], dtype=np.float64)
                s_st = np.asarray(state["s"], dtype=np.float64)
                w_sk = _per_direction_w(V_st, s_st, v, sk_F2_low)
            else:
                w_sk = sk_F2_low
        u_sk = raw_sk / max(w_sk, eps)
    return u_sk, u_g1, u_g2, raw_sk, raw_g1, raw_g2


def _imbalance_ratios(u_sk, u_g1, u_g2):
    """Compute the four imbalance ratios. NaN if any input is NaN/zero."""
    ratios = {
        "ratio_max": float("nan"),
        "ratio_skg1": float("nan"),
        "ratio_skg2": float("nan"),
        "ratio_g1g2": float("nan"),
    }
    if any(math.isnan(x) for x in (u_sk, u_g1, u_g2)):
        # Block-1 case: u_sk is NaN. Still report g1/g2 ratio.
        if not (math.isnan(u_g1) or math.isnan(u_g2)) and u_g2 > 0:
            ratios["ratio_g1g2"] = u_g1 / u_g2
        return ratios
    eps = 1e-30
    vals = [u_sk, u_g1, u_g2]
    if min(vals) > eps:
        ratios["ratio_max"] = max(vals) / min(vals)
        ratios["ratio_skg1"] = u_sk / u_g1
        ratios["ratio_skg2"] = u_sk / u_g2
        ratios["ratio_g1g2"] = u_g1 / u_g2
    return ratios


def block_components(args, A, V_exact, snap, block_id):
    """Build oracle slot-1 / slot-2 candidates and compute u-imbalance."""
    consts = per_block_constants(A, block_id, args.half_win)
    cur_F2 = float(consts["cur_F2"])
    fut_F2 = float(consts["fut_F2"])
    sk_F2_low = (
        float(np.sum(np.asarray(snap["A_sketch"], dtype=np.float64) ** 2))
        if snap["A_sketch"].size
        else 0.0
    )

    fnorm_cands = fnorm_collect_candidates(snap, V_exact)

    rank = int(args.rank)
    scheme = getattr(args, "weight_scheme", "current")
    cur_svd = _top_r_svd(snap["A_cur"], rank) if scheme in ("E1", "E2") else None
    fut_svd = _top_r_svd(snap["A_fut"], rank) if scheme in ("E1", "E2") else None

    rows = []
    for label, slot in ORACLE_SLOTS:
        v = fnorm_cands.get(label)
        u_sk, u_g1, u_g2, raw_sk, raw_g1, raw_g2 = _u_components(
            snap, v, sk_F2_low, cur_F2, fut_F2,
            weight_scheme=scheme, rank=rank,
            cur_svd=cur_svd, fut_svd=fut_svd,
        )
        ratios = _imbalance_ratios(u_sk, u_g1, u_g2)
        rows.append({
            "block": int(block_id),
            "slot": slot,
            "candidate": label,
            "u_sk": u_sk,
            "u_g1": u_g1,
            "u_g2": u_g2,
            "raw_sk": raw_sk,
            "raw_g1": raw_g1,
            "raw_g2": raw_g2,
            "sk_F2_low": sk_F2_low,
            "cur_F2": cur_F2,
            "fut_F2": fut_F2,
            "ratio_max": ratios["ratio_max"],
            "ratio_skg1": ratios["ratio_skg1"],
            "ratio_skg2": ratios["ratio_skg2"],
            "ratio_g1g2": ratios["ratio_g1g2"],
        })
    return rows


def run_matrix(args, matrix, blocks):
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    try:
        A, V_exact, _, _ = probe.generate_matrix_input(
            matrix=matrix, n=args.n, preset=args.preset, seed=args.seed,
            r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
            tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
            shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
        )
    except TypeError:
        A, V_exact, _, _ = probe.generate_matrix_input(
            matrix=matrix, n=args.n, preset=args.preset, seed=args.seed,
            shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
        )
    A = np.asarray(A, np.float64)
    V_exact = np.asarray(V_exact, np.float64)

    target = max(blocks)
    snaps = stream_to_block(args, A, V_exact, work_dtype, int(args.rank), target, set(blocks))

    all_rows = []
    for b in sorted(blocks):
        if b not in snaps:
            continue
        all_rows.extend(block_components(args, A, V_exact, snaps[b], b))
    return all_rows


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def write_per_matrix_csv(path, rows):
    if not rows:
        return
    fields = [
        "block", "slot", "candidate",
        "u_sk", "u_g1", "u_g2",
        "raw_sk", "raw_g1", "raw_g2",
        "sk_F2_low", "cur_F2", "fut_F2",
        "ratio_max", "ratio_skg1", "ratio_skg2", "ratio_g1g2",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(path, per_matrix_rows, blocks):
    """One row per (matrix, slot, block) with the four ratios. Wide form for grep."""
    fields = ["matrix", "slot", "block",
              "u_sk", "u_g1", "u_g2",
              "ratio_max", "ratio_skg1", "ratio_skg2", "ratio_g1g2"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for matrix, rows in per_matrix_rows.items():
            for row in rows:
                writer.writerow({
                    "matrix": matrix,
                    "slot": row["slot"],
                    "block": row["block"],
                    "u_sk": row["u_sk"],
                    "u_g1": row["u_g1"],
                    "u_g2": row["u_g2"],
                    "ratio_max": row["ratio_max"],
                    "ratio_skg1": row["ratio_skg1"],
                    "ratio_skg2": row["ratio_skg2"],
                    "ratio_g1g2": row["ratio_g1g2"],
                })


# --------------------------------------------------------------------------
# Synthesis
# --------------------------------------------------------------------------


def _row_at(rows, slot, block):
    for r in rows:
        if r["slot"] == slot and r["block"] == block:
            return r
    return None


def _fmt_ratio(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    if x >= 100:
        return f"{x:.0f}x"
    if x >= 10:
        return f"{x:.1f}x"
    return f"{x:.2f}x"


def write_synthesis(out_md, per_matrix_rows, blocks):
    lines = []
    lines.append("# DIAG-04 oracle u-balance audit — synthesis")
    lines.append("")
    lines.append("Probe: `oracle_u_balance_audit.py` (per-block, half_win=32, rank=2).")
    lines.append("Components: u_sk, u_g1, u_g2 from S6 (F-weighted HM3) at oracle_v_k_proj")
    lines.append("for slot k ∈ {1, 2}. Ratios: max(u)/min(u), sk/g1, sk/g2, g1/g2.")
    lines.append("")
    lines.append("Hypothesis (overview §1quater): HM3's \"smallest-link\" enforcer only")
    lines.append("rewards oracle if oracle is roughly balanced under the chosen weights.")
    lines.append("Frobenius weighting was inherited from §2(b)'s \"unit-fixer\" framing,")
    lines.append("not calibrated against an oracle-balance criterion. This audit measures")
    lines.append("the imbalance at scale and correlates it with S6 cos[k]² failure.")
    lines.append("")

    # ---- Slot-1 ratio_max table ----
    lines.append("## Slot-1 (oracle_v1_proj) ratio_max per block")
    lines.append("")
    lines.append("ratio_max = max(u_sk, u_g1, u_g2) / min(u_sk, u_g1, u_g2). 1.00x = perfect")
    lines.append("balance; HM3 reads oracle as the high-score point. Large values mean")
    lines.append("oracle's smallest u dominates HM3 and the score peak shifts off oracle.")
    lines.append("")
    cols = ["matrix"] + [f"b{b}" for b in blocks] + ["S6 cos1²"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for matrix, rows in per_matrix_rows.items():
        cells = [matrix]
        for b in blocks:
            r = _row_at(rows, 1, b)
            cells.append(_fmt_ratio(r["ratio_max"]) if r else "—")
        cos1 = S6_COS_BENCH.get(matrix, {}).get("cos1")
        cells.append(f"{cos1:.3f}" if cos1 is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # ---- Slot-2 ratio_max table ----
    lines.append("## Slot-2 (oracle_v2_proj) ratio_max per block")
    lines.append("")
    lines.append("Slot-2 is where S6 actually fails on T3. If slot-2 oracle is even more")
    lines.append("imbalanced than slot-1, the M4 hypothesis predicts S6 fails harder on")
    lines.append("slot-2 than slot-1 — which it empirically does.")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for matrix, rows in per_matrix_rows.items():
        cells = [matrix]
        for b in blocks:
            r = _row_at(rows, 2, b)
            cells.append(_fmt_ratio(r["ratio_max"]) if r else "—")
        cos1 = S6_COS_BENCH.get(matrix, {}).get("cos1")
        cells.append(f"{cos1:.3f}" if cos1 is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # ---- Slot-1 sk/g1 ratio (the \"sketch dominance\" axis) ----
    lines.append("## Slot-1 ratio_skg1 (u_sk / u_g1) per block")
    lines.append("")
    lines.append("ratio_skg1 > 1 means sketch is over-rewarded vs current half-window.")
    lines.append("ratio_skg1 < 1 means sketch is under-rewarded. Either direction breaks")
    lines.append("HM3's smallest-link reading of oracle. Note: at b1 sketch is empty so")
    lines.append("ratio_skg1 is NaN.")
    lines.append("")
    lines.append("| " + " | ".join(cols[:-1]) + " |")
    lines.append("| " + " | ".join("---" for _ in cols[:-1]) + " |")
    for matrix, rows in per_matrix_rows.items():
        cells = [matrix]
        for b in blocks:
            r = _row_at(rows, 1, b)
            cells.append(_fmt_ratio(r["ratio_skg1"]) if r else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # ---- u-component dump for slot-2 at b31 (the failure block) ----
    lines.append("## Slot-2 component breakdown at b31 (the streaming-bench terminal block)")
    lines.append("")
    lines.append("| matrix | u_sk | u_g1 | u_g2 | ratio_max | S6 cos1² |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for matrix, rows in per_matrix_rows.items():
        r = _row_at(rows, 2, 31)
        if r is None:
            continue
        cos1 = S6_COS_BENCH.get(matrix, {}).get("cos1")
        cos1_str = f"{cos1:.3f}" if cos1 is not None else "—"
        lines.append(
            f"| {matrix} | {r['u_sk']:.4f} | {r['u_g1']:.4f} | "
            f"{r['u_g2']:.4f} | {_fmt_ratio(r['ratio_max'])} | {cos1_str} |"
        )
    lines.append("")

    # ---- Reading / verdict template ----
    lines.append("## Reading")
    lines.append("")
    lines.append("Cross-matrix correlation between ratio_max and S6 cos1² failure.")
    lines.append("The hypothesis predicts: matrices with large ratio_max at b31")
    lines.append("(score peak shifted away from oracle) should have low S6 cos1².")
    lines.append("Etf-basket-basis is the calibration anchor — if the hypothesis is")
    lines.append("right, it should have the smallest ratio_max AND the highest S6 cos1².")
    lines.append("")
    lines.append("Possible verdicts:")
    lines.append("- VERIFIED if ratio_max ranks matrices in the same order as 1/cos1²")
    lines.append("  AND ratio_max ≥ 5x on the failure matrices and < 2x on")
    lines.append("  etf-basket-basis. AB-03 phase 2 is then strongly motivated.")
    lines.append("- PARTIALLY VERIFIED if the rank-correlation is positive but a")
    lines.append("  matrix has large ratio_max + decent cos1² (or vice versa). Mark")
    lines.append("  it as a boundary case and consider whether other mechanisms")
    lines.append("  (M2 carry pinning, §3 plateau drift) dominate there.")
    lines.append("- REFUTED if ratio_max is uniformly small (< 2x everywhere) or")
    lines.append("  rank-correlation with cos1² is near zero. AB-03 is killed at")
    lines.append("  the audit stage.")
    lines.append("")
    lines.append("Cross-references:")
    lines.append("- score_design_overview.txt §1quater (M4 mechanism)")
    lines.append("- score_design_overview.txt §2bis (b.iii) (calibration criterion)")
    lines.append("- score_family_workflow.txt [DIAG-04] / [AB-03]")
    lines.append("- diagnostic_toolkit.txt §6b (oracle u-imbalance signature)")
    lines.append("")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrices", nargs="+", default=DEFAULT_MATRICES)
    p.add_argument("--blocks", nargs="+", type=int, default=DEFAULT_BLOCKS)
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
    p.add_argument("--out-dir", default="summary/infra_oracle_u_balance")
    p.add_argument(
        "--weight-scheme",
        choices=("current", "E1", "E2"),
        default="current",
        help="Weight scheme for u-components. current=F-norm² (baseline); "
             "E1=op-norm cap (sigma_top²); E2=per-direction sigma[k(v)]².",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    blocks = sorted(set(int(b) for b in args.blocks))
    per_matrix_rows = {}
    t_global = time.time()
    for matrix in args.matrices:
        print(f"[oracle-u-balance] running {matrix} ...", flush=True)
        t0 = time.time()
        rows = run_matrix(args, matrix, blocks)
        per_matrix_rows[matrix] = rows

        csv_path = os.path.join(args.out_dir, f"{matrix}_audit.csv")
        write_per_matrix_csv(csv_path, rows)
        print(f"  wrote {csv_path}  (elapsed={time.time() - t0:.2f}s)", flush=True)

    summary_csv = os.path.join(args.out_dir, "audit_summary.csv")
    write_summary_csv(summary_csv, per_matrix_rows, blocks)
    print(f"[oracle-u-balance] wrote {summary_csv}", flush=True)

    out_md = os.path.join(args.out_dir, "synthesis.md")
    write_synthesis(out_md, per_matrix_rows, blocks)
    print(f"[oracle-u-balance] wrote {out_md}", flush=True)
    print(f"[oracle-u-balance] total elapsed {time.time() - t_global:.2f}s", flush=True)


if __name__ == "__main__":
    main()
