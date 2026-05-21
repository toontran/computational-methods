"""Aggregate landscape + force-oracle bench results into a single closure
table for AB-03 phase 1.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

OUT_DIR = "summary/score_family_aggregator_ablation"
MATRICES = (
    "diffuse-diffuse", "residual-spiky-shocks", "mixed-tail-soft",
    "mixed-tail-sharp", "static-cex", "mixed-tail-balanced", "etf-basket-basis",
)

SLIDING_RE = re.compile(
    r"^mode=sliding\s+policy=(?P<policy>\S+)\s+half_win=\d+\s+steps=\d+\s+"
    r".*?final_cos=\[(?P<c0>[0-9.eE+\-]+)\s+(?P<c1>[0-9.eE+\-]+)\].*?"
    r"final_exact_cos=\[(?P<e0>[0-9.eE+\-]+)\s+(?P<e1>[0-9.eE+\-]+)\].*?"
    r"final_oracle_proj_norm=\[(?P<o0>[0-9.eE+\-]+)\s+(?P<o1>[0-9.eE+\-]+)\].*?"
    r"final_tail_mass=(?P<tail>[0-9.eE+\-]+)"
)


def _grep_sliding(path, want_policy="future_hmean_r_sk_g"):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            m = SLIDING_RE.match(ln)
            if m and m.group("policy") == want_policy:
                d = m.groupdict()
                return {
                    "c0": float(d["c0"]), "c1": float(d["c1"]),
                    "e0": float(d["e0"]), "e1": float(d["e1"]),
                    "o0": float(d["o0"]), "o1": float(d["o1"]),
                    "tail": float(d["tail"]),
                    "policy": d["policy"],
                }
    return None


def main():
    rows = []
    for m in MATRICES:
        s6 = _grep_sliding(f"{OUT_DIR}/{m}_win64.txt")
        e2 = _grep_sliding(f"{OUT_DIR}/S6_E2_{m}_win64.txt")
        f2 = _grep_sliding(f"{OUT_DIR}/forceO2_{m}_win64.txt")
        ff = _grep_sliding(f"{OUT_DIR}/forceFrame_{m}_win64.txt")
        rows.append({"matrix": m, "S6_HM3": s6, "S6_E2": e2,
                     "force_O2": f2, "force_frame": ff})

    out_path = f"{OUT_DIR}/closure_table_b31.txt"
    with open(out_path, "w") as fh:
        fh.write("# AB-03 phase 1 closure — bench cos² at b31 across configurations\n")
        fh.write("# cos0/cos1 are principal cosines vs V_exact (sorted descending)\n")
        fh.write("# capture = c0² + c1²  (oracle-mass captured; max=2)\n")
        fh.write("\n")
        fh.write(
            f"{'matrix':<24} | "
            f"{'S6 c0/c1²':<16} | {'S6 cap':<7} | "
            f"{'E2 c0/c1²':<16} | {'E2 cap':<7} | "
            f"{'fO2 c0/c1²':<16} | {'fO2 cap':<7} | "
            f"{'fFr c0/c1²':<16} | {'fFr cap':<7}\n"
        )
        fh.write("-" * 175 + "\n")
        for r in rows:
            line = f"{r['matrix']:<24} | "
            for label, key in (("S6", "S6_HM3"), ("E2", "S6_E2"),
                               ("fO2", "force_O2"), ("fFr", "force_frame")):
                d = r[key]
                if d is None:
                    line += f"{'(missing)':<16} | {'-':<7} | "
                    continue
                c0sq = d["c0"] ** 2
                c1sq = d["c1"] ** 2
                cap = c0sq + c1sq
                line += f"{c0sq:6.3f} / {c1sq:6.3f} | {cap:6.3f} | "
            fh.write(line.rstrip(" |") + "\n")
        fh.write("\n# Key:\n")
        fh.write("#   S6   = baseline F-norm HM3 (current shipped variant)\n")
        fh.write("#   E2   = per-direction sigma² weighting (AB-03 / S6_E2)\n")
        fh.write("#   fO2  = chosen_v2 forced to oracle_v2_proj (V_default[:,0] from streaming)\n")
        fh.write("#   fFr  = entire V_selected forced to SVD-frame(oracle_v1_proj, oracle_v2_proj)\n")
    print(f"wrote {out_path}")
    with open(out_path) as fh:
        print(fh.read())


if __name__ == "__main__":
    main()
