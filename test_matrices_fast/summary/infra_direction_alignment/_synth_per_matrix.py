"""Build per-matrix cross-block synthesis tables from the CSV outputs.

Reads:
  summary/infra_direction_alignment/<matrix>_b<block>_alignment.csv
  summary/infra_direction_alignment/<matrix>_b<block>_rowbias.csv

Writes:
  summary/infra_direction_alignment/synthesis_<matrix>.txt
  summary/infra_direction_alignment/synthesis_overview.txt

Tracks two things per block:
  FD-slot directions (sketch_v1 = V_state[:,0], sketch_v2 = V_state[:,1])
    - cos² vs oracle_v?_proj, vs Acur_topSV_v?, vs V_state_v? (sanity)
    - relH1 / eff_frac on visible window, top-1 row share
  outside direction (combined_v2 = V_default[:,1])
    - cos² vs V_state_frame (state_align), oracle_v2_proj, oracle_frame
    - relH1 / eff_frac visible
"""

from __future__ import annotations

import csv
import os

DIR = os.path.dirname(__file__)
MATRICES = ["static-cex", "mixed-tail-balanced", "diffuse-diffuse"]
BLOCKS = [1, 2, 6, 12, 31]


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_cos2(rows, input_label, ref_label):
    for r in rows:
        if r["input_label"] == input_label and r["reference_label"] == ref_label:
            try:
                return float(r["cos2_top"])
            except ValueError:
                return float("nan")
    return float("nan")


def get_rowbias(rows, input_label, window):
    for r in rows:
        if r["input_label"] == input_label and r["window"] == window:
            return r
    return None


def fmt(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return " nan "
    if x != x:
        return " nan "
    return f"{x:.3f}"


def per_matrix_table(matrix):
    lines = []
    lines.append(f"== {matrix} ==")
    lines.append("")
    lines.append("FD slot-1 (sketch_v1 = V_state[:,0]) and FD slot-2 (sketch_v2 = V_state[:,1])")
    lines.append("Outside-window direction (combined_v2 = V_default[:,1])")
    lines.append("")
    lines.append(
        "                                      "
        + "  ".join(f" b{b:>2}" for b in BLOCKS)
    )

    # Per-block alignment fetches.
    blockdata = {}
    for b in BLOCKS:
        align_path = os.path.join(DIR, f"{matrix}_b{b}_alignment.csv")
        rowb_path = os.path.join(DIR, f"{matrix}_b{b}_rowbias.csv")
        if not (os.path.exists(align_path) and os.path.exists(rowb_path)):
            blockdata[b] = (None, None)
            continue
        blockdata[b] = (load_csv(align_path), load_csv(rowb_path))

    # Helper that builds one row of the table.
    def row(label, fetcher):
        cells = []
        for b in BLOCKS:
            a, _ = blockdata[b]
            if a is None:
                cells.append(" -- ")
                continue
            cells.append(fmt(fetcher(a, b)))
        return f"  {label:<38}" + "  ".join(f"{c:>4}" for c in cells)

    def rowb_cell(b, input_label, key, window="visible"):
        _, rb = blockdata[b]
        if rb is None:
            return float("nan")
        r = get_rowbias(rb, input_label, window)
        if r is None:
            return float("nan")
        try:
            return float(r[key])
        except (ValueError, KeyError):
            return float("nan")

    lines.append("[FD slot-1 = sketch_v1]")
    lines.append(row("cos² vs oracle_v1_exact (population)",
                     lambda a, b: get_cos2(a, "sketch_v1", "oracle_v1_exact")))
    lines.append(row("cos² vs oracle_v1_proj  (in-window)",
                     lambda a, b: get_cos2(a, "sketch_v1", "oracle_v1_proj")))
    lines.append(row("cos² vs Acur_topSV_v1 (per-block)",
                     lambda a, b: get_cos2(a, "sketch_v1", "Acur_topSV_v1")))
    lines.append(row("cos² vs Afut_topSV_v1 (next-window)",
                     lambda a, b: get_cos2(a, "sketch_v1", "Afut_topSV_v1")))
    lines.append(row("cos² vs visible_topSV_v1 (cur+fut)",
                     lambda a, b: get_cos2(a, "sketch_v1", "visible_topSV_v1")))
    lines.append(row("cos² vs Acur_rowcheat_v1 (top row)",
                     lambda a, b: get_cos2(a, "sketch_v1", "Acur_rowcheat_v1")))
    lines.append(row("relH1(visible)",
                     lambda a, b: rowb_cell(b, "sketch_v1", "relH1")))
    lines.append(row("eff_frac(visible)",
                     lambda a, b: rowb_cell(b, "sketch_v1", "eff_frac")))
    lines.append(row("top1_share(visible)",
                     lambda a, b: rowb_cell(b, "sketch_v1", "top1_share")))
    lines.append("")

    lines.append("[FD slot-2 = sketch_v2]")
    lines.append(row("cos² vs oracle_v2_exact (population)",
                     lambda a, b: get_cos2(a, "sketch_v2", "oracle_v2_exact")))
    lines.append(row("cos² vs oracle_v2_proj  (in-window)",
                     lambda a, b: get_cos2(a, "sketch_v2", "oracle_v2_proj")))
    lines.append(row("cos² vs oracle_v1_proj (slot mix)",
                     lambda a, b: get_cos2(a, "sketch_v2", "oracle_v1_proj")))
    lines.append(row("cos² vs Acur_topSV_v2",
                     lambda a, b: get_cos2(a, "sketch_v2", "Acur_topSV_v2")))
    lines.append(row("cos² vs Afut_topSV_v2",
                     lambda a, b: get_cos2(a, "sketch_v2", "Afut_topSV_v2")))
    lines.append(row("cos² vs Acur_rowcheat_v2",
                     lambda a, b: get_cos2(a, "sketch_v2", "Acur_rowcheat_v2")))
    lines.append(row("relH1(visible)",
                     lambda a, b: rowb_cell(b, "sketch_v2", "relH1")))
    lines.append(row("eff_frac(visible)",
                     lambda a, b: rowb_cell(b, "sketch_v2", "eff_frac")))
    lines.append(row("top1_share(visible)",
                     lambda a, b: rowb_cell(b, "sketch_v2", "top1_share")))
    lines.append("")

    lines.append("[OUTSIDE direction = combined_v2 = V_default[:,1]]")
    lines.append(row("cos² vs V_state_frame (= state_align)",
                     lambda a, b: get_cos2(a, "combined_v2", "V_state_frame")))
    lines.append(row("cos² vs V_state_v1",
                     lambda a, b: get_cos2(a, "combined_v2", "V_state_v1")))
    lines.append(row("cos² vs V_state_v2",
                     lambda a, b: get_cos2(a, "combined_v2", "V_state_v2")))
    lines.append(row("cos² vs oracle_v2_exact (population)",
                     lambda a, b: get_cos2(a, "combined_v2", "oracle_v2_exact")))
    lines.append(row("cos² vs oracle_v2_proj  (in-window)",
                     lambda a, b: get_cos2(a, "combined_v2", "oracle_v2_proj")))
    lines.append(row("cos² vs oracle_v1_proj",
                     lambda a, b: get_cos2(a, "combined_v2", "oracle_v1_proj")))
    lines.append(row("cos² vs oracle_frame_proj",
                     lambda a, b: get_cos2(a, "combined_v2", "oracle_frame_proj")))
    lines.append(row("cos² vs Acur_topSV_v1",
                     lambda a, b: get_cos2(a, "combined_v2", "Acur_topSV_v1")))
    lines.append(row("cos² vs Afut_topSV_v1",
                     lambda a, b: get_cos2(a, "combined_v2", "Afut_topSV_v1")))
    lines.append(row("relH1(visible)",
                     lambda a, b: rowb_cell(b, "combined_v2", "relH1")))
    lines.append(row("eff_frac(visible)",
                     lambda a, b: rowb_cell(b, "combined_v2", "eff_frac")))
    lines.append(row("top1_share(visible)",
                     lambda a, b: rowb_cell(b, "combined_v2", "top1_share")))
    lines.append("")

    lines.append("[combined_v1 = V_default[:,0]  (M3: top SV of A_cur at b1; top SV of M_gain after)]")
    lines.append(row("cos² vs Acur_topSV_v1",
                     lambda a, b: get_cos2(a, "combined_v1", "Acur_topSV_v1")))
    lines.append(row("cos² vs oracle_v1_exact (population)",
                     lambda a, b: get_cos2(a, "combined_v1", "oracle_v1_exact")))
    lines.append(row("cos² vs oracle_v1_proj  (in-window)",
                     lambda a, b: get_cos2(a, "combined_v1", "oracle_v1_proj")))
    lines.append(row("cos² vs V_state_frame",
                     lambda a, b: get_cos2(a, "combined_v1", "V_state_frame")))
    lines.append("")

    # Oracle reachability per block. Pull from the txt header (first
    # numeric "slot 1" / "slot 2" line) — read once per block.
    lines.append("[oracle reachability ||P_B_union V_exact[:,k]||² per slot]")
    def reach_at(b, slot):
        path = os.path.join(DIR, f"{matrix}_b{b}.txt")
        if not os.path.exists(path):
            return float("nan")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"slot {slot}:"):
                    try:
                        return float(line.split(":", 1)[1].strip())
                    except ValueError:
                        return float("nan")
        return float("nan")
    cells_s1 = [fmt(reach_at(b, 1)) for b in BLOCKS]
    cells_s2 = [fmt(reach_at(b, 2)) for b in BLOCKS]
    lines.append(f"  {'oracle slot 1 reach':<38}" + "  ".join(f"{c:>4}" for c in cells_s1))
    lines.append(f"  {'oracle slot 2 reach':<38}" + "  ".join(f"{c:>4}" for c in cells_s2))
    lines.append("")
    return "\n".join(lines)


def main():
    overview = ["# direction_alignment_probe — FD vs outside-direction summary",
                "",
                "Generated from per-block CSVs in this directory by _synth_per_matrix.py.",
                "Blocks: 1, 2, 6, 12, 31. half_win=32, rank=2.",
                ""]
    for matrix in MATRICES:
        block = per_matrix_table(matrix)
        out_path = os.path.join(DIR, f"synthesis_{matrix}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(block + "\n")
        print(f"wrote {out_path}")
        overview.append(block)
        overview.append("")

    with open(os.path.join(DIR, "synthesis_overview.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(overview))
    print(f"wrote {os.path.join(DIR, 'synthesis_overview.txt')}")


if __name__ == "__main__":
    main()
