"""Aggregate v_cur_vs_v_fut probe cells into a per-matrix summary table and
the discrimination-test Spearman.

Reads:  summary/per_block_v_cur_vs_v_fut/cells.csv
Writes: summary/per_block_v_cur_vs_v_fut/summary.csv
        summary/per_block_v_cur_vs_v_fut/report.md
"""

import csv
import os
import statistics
import sys

import numpy as np


CELLS = "summary/per_block_v_cur_vs_v_fut/cells.csv"
SUMMARY = "summary/per_block_v_cur_vs_v_fut/summary.csv"
REPORT = "summary/per_block_v_cur_vs_v_fut/report.md"


def load(path):
    rows = []
    with open(path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            for k, v in r.items():
                if k in ("matrix",):
                    continue
                if k in ("seed", "block"):
                    r[k] = int(v)
                else:
                    r[k] = float(v)
            rows.append(r)
    return rows


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    nx, ny = float(np.linalg.norm(rx)), float(np.linalg.norm(ry))
    if nx <= 1e-30 or ny <= 1e-30:
        return float("nan")
    return float(np.dot(rx, ry) / (nx * ny))


def quantile_table(cells, group_key, target_metric, qs=(0.25, 0.5, 0.75, 1.0)):
    """Group cells of a matrix by quantile of target_metric, return median row_aligned_shift."""
    vals = sorted([c[target_metric] for c in cells])
    if not vals:
        return None
    cuts = []
    for q in qs:
        cuts.append(vals[min(int(q * len(vals)) - 1, len(vals) - 1)])
    buckets = [[] for _ in qs]
    for c in cells:
        v = c[target_metric]
        for i, cut in enumerate(cuts):
            if v <= cut:
                buckets[i].append(c)
                break
    return cuts, buckets


def median(vals):
    if not vals:
        return float("nan")
    return float(statistics.median(vals))


def main():
    cells = load(CELLS)
    matrices = sorted({c["matrix"] for c in cells})

    # ---- summary.csv: per-matrix aggregate ----
    summary_rows = []
    for m in matrices:
        sub = [c for c in cells if c["matrix"] == m]
        n = len(sub)
        cos_vv = [c["cos_vv"] for c in sub]
        ras = [c["row_aligned_shift"] for c in sub]
        ds = [c["direction_shift"] for c in sub]
        ms = [c["mass_shift"] for c in sub]
        es = [c["entropy_shift"] for c in sub]
        ees = [c["element_energy_shift"] for c in sub]
        rs = [c["rank_shift"] for c in sub]
        c2_oracle = [c["cos2_v_cur_oracle"] for c in sub]
        c2_reach = [c["cos2_v_cur_reach"] for c in sub]

        # Discrimination Spearman
        rho_oracle = spearman(c2_oracle, ras)
        rho_reach = spearman(c2_reach, ras)

        summary_rows.append({
            "matrix": m,
            "n_cells": n,
            "median_cos_vv": median(cos_vv),
            "median_dir_shift": median(ds),
            "median_row_aligned_shift": median(ras),
            "median_elem_energy_shift": median(ees),
            "median_rank_shift": median(rs),
            "median_mass_shift": median(ms),
            "median_entropy_shift": median(es),
            "max_row_aligned_shift": max(ras),
            "min_row_aligned_shift": min(ras),
            "median_cos2_oracle": median(c2_oracle),
            "median_cos2_reach": median(c2_reach),
            "rho_spearman_cos2oracle_vs_rowshift": rho_oracle,
            "rho_spearman_cos2reach_vs_rowshift": rho_reach,
        })

    keys = list(summary_rows[0].keys())
    with open(SUMMARY, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in summary_rows:
            cells_out = []
            for k in keys:
                v = r[k]
                if isinstance(v, str):
                    cells_out.append(v)
                elif isinstance(v, int):
                    cells_out.append(str(v))
                elif np.isnan(v):
                    cells_out.append("nan")
                else:
                    cells_out.append(f"{v:.4f}")
            f.write(",".join(cells_out) + "\n")

    # ---- report.md ----
    lines = []
    lines.append("# v_cur vs v_fut row-aligned shift probe — results")
    lines.append("")
    lines.append("Compares v_cur (combined-score lock on (S, A_cur)) against v_fut")
    lines.append("(combined-score lock on (S, A_cur, A_fut)), both read off on A_cur.")
    lines.append("")
    lines.append(f"Matrices: {', '.join(matrices)}")
    lines.append(f"Seeds: {sorted({c['seed'] for c in cells})}")
    lines.append(f"Blocks: 1..{max(c['block'] for c in cells)}  (half_win=32, n=1024, rank=2)")
    lines.append(f"Total cells: {len(cells)}")
    lines.append("")

    # Sanity table (predictions §6 stable: mass_shift ≈ 0, entropy_shift ≈ 0)
    lines.append("## §6 stable predictions (sanity)")
    lines.append("")
    lines.append("Both v_cur and v_fut are energy-maximizing locks on their data.")
    lines.append("If row-entropy saturates (relH≈1) on A_cur for both, mass_shift")
    lines.append("and entropy_shift should be small. The probe spec calls these the")
    lines.append("'inside-the-ambiguous-regime' sanity checks.")
    lines.append("")
    lines.append("| matrix | median(cos vv) | median(mass_shift) | median(entropy_shift) | median(relH_cur) | median(relH_fut) |")
    lines.append("|---|---|---|---|---|---|")
    for m in matrices:
        sub = [c for c in cells if c["matrix"] == m]
        lines.append(
            f"| {m} | {median([c['cos_vv'] for c in sub]):+.3f} | "
            f"{median([c['mass_shift'] for c in sub]):.3f} | "
            f"{median([c['entropy_shift'] for c in sub]):.3f} | "
            f"{median([c['relH_cur'] for c in sub]):.3f} | "
            f"{median([c['relH_fut'] for c in sub]):.3f} |"
        )
    lines.append("")

    # Headline: row-aligned shift across regimes
    lines.append("## Headline: row-aligned shift = 1 − |cos(e_cur, e_fut)|")
    lines.append("")
    lines.append("Larger = the two solutions disagree more on per-row energy distribution on A_cur.")
    lines.append("§6 substantive prediction: row-concentrated matrices (static-cex) → small (med < 0.1);")
    lines.append("diffuse / mixed-tail-soft → large (med > 0.2) with high block-to-block variance.")
    lines.append("")
    lines.append("| matrix | median | min | max | std | median direction_shift |")
    lines.append("|---|---|---|---|---|---|")
    for m in matrices:
        sub = [c for c in cells if c["matrix"] == m]
        ras = [c["row_aligned_shift"] for c in sub]
        ds = [c["direction_shift"] for c in sub]
        lines.append(
            f"| {m} | {median(ras):.3f} | {min(ras):.3f} | {max(ras):.3f} | "
            f"{statistics.stdev(ras) if len(ras) >= 2 else float('nan'):.3f} | "
            f"{median(ds):.3f} |"
        )
    lines.append("")

    # Discrimination test: Spearman
    lines.append("## Discrimination test (§6 substantive)")
    lines.append("")
    lines.append("Spearman ρ between cos²(v_cur, target) and row_aligned_shift, by matrix.")
    lines.append("Success line: ρ ≤ -0.3 (high oracle alignment ↔ low shift).")
    lines.append("")
    lines.append("- cos²(v_cur, V_exact[:,0]) — raw oracle alignment (visibility analysis §3 says this is ~0 across cells).")
    lines.append("- cos²(v_cur, P_search V_exact[:,0]) — alignment with the *reachable* oracle in rowspace([sketch; A_cur]).")
    lines.append("")
    lines.append("| matrix | n | ρ(cos²_oracle, row_shift) | ρ(cos²_reach, row_shift) | median cos²_oracle | median cos²_reach |")
    lines.append("|---|---|---|---|---|---|")
    for r in summary_rows:
        lines.append(
            f"| {r['matrix']} | {r['n_cells']} | "
            f"{r['rho_spearman_cos2oracle_vs_rowshift']:+.3f} | "
            f"{r['rho_spearman_cos2reach_vs_rowshift']:+.3f} | "
            f"{r['median_cos2_oracle']:.4f} | "
            f"{r['median_cos2_reach']:.3f} |"
        )
    lines.append("")

    # Quantile table per matrix using cos2_v_cur_reach
    lines.append("## Per-matrix quantile table (cos²_reach quartiles → median row_aligned_shift)")
    lines.append("")
    lines.append("If discrimination is real, the highest quartile of cos²_reach should have")
    lines.append("the lowest median row-aligned shift.")
    lines.append("")
    for m in matrices:
        sub = [c for c in cells if c["matrix"] == m]
        if not sub:
            continue
        # Sort by cos2_v_cur_reach and bucket into quartiles
        sub_sorted = sorted(sub, key=lambda c: c["cos2_v_cur_reach"])
        n = len(sub_sorted)
        q1 = sub_sorted[: n // 4]
        q2 = sub_sorted[n // 4: n // 2]
        q3 = sub_sorted[n // 2: 3 * n // 4]
        q4 = sub_sorted[3 * n // 4:]
        lines.append(f"### {m}")
        lines.append("")
        lines.append("| quartile of cos²_reach | n | median cos²_reach | median row_aligned_shift | median direction_shift |")
        lines.append("|---|---|---|---|---|")
        for label, bucket in [("Q1 (low)", q1), ("Q2", q2), ("Q3", q3), ("Q4 (high)", q4)]:
            if not bucket:
                continue
            lines.append(
                f"| {label} | {len(bucket)} | "
                f"{median([c['cos2_v_cur_reach'] for c in bucket]):.3f} | "
                f"{median([c['row_aligned_shift'] for c in bucket]):.3f} | "
                f"{median([c['direction_shift'] for c in bucket]):.3f} |"
            )
        lines.append("")

    with open(REPORT, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {SUMMARY} and {REPORT}")


if __name__ == "__main__":
    main()
