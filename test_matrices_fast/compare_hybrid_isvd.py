#!/usr/bin/env python3
"""Compare entropyscore_hybrid vs isvd on kernel_stocks_1000_0.02.

Metric (per block):
    align_i = sqrt( max(i - ||C[:, :i]||_F^2, 0) )
for i = 1..6, where C = Vt @ Vt_exact[:k, :]^T stored in canonical_angles_data_*.txt.
This equals ||(I - V_r V_r^T) V_exact[:, :i]||_F (residual of the true top-i
singular vectors projected onto the complement of the estimated subspace).

LOWER is BETTER.  0 = perfect, sqrt(i) = orthogonal.

Report: per (k, win, sr) config, the mean-over-blocks-then-permutations of
align_i, side-by-side hybrid vs isvd, for i=1..6 (capped at i<=k).
"""
import os, json, glob, re
import numpy as np
from collections import defaultdict

OUT = "output"
PAT = re.compile(
    r"kernel_stocks_1000_0\.02_(?P<method>[a-z_]+?)_(?P<perm>random_uniform(?:_\d+)?|original)"
    r"_size_(?P<size>\d+)_ssize_(?P<ssize>\d+)_k_(?P<k>\d+)_sr_(?P<sr>\d+)_reservoir_greedy"
)

def align_i_per_block(folder, max_i=6):
    """Return mean over blocks of align_i for i=1..max_i (each a scalar)."""
    files = sorted(
        glob.glob(os.path.join(folder, "canonical_angles_data_*.txt")),
        key=lambda p: int(re.search(r"_(\d+)\.txt$", p).group(1)),
    )
    if not files:
        return None
    per_i_accum = {i: [] for i in range(1, max_i + 1)}
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        C = np.asarray(d.get("C", {}).get("value", []), dtype=float)
        if C.size == 0:
            continue
        # C has shape (k_eff, k_eff_exact). Vt was truncated to its own rank.
        k_rows, k_cols = C.shape
        for i in range(1, max_i + 1):
            if i > k_cols or i > k_rows:
                # ||C[:, :i]||_F^2 uses k_rows × min(i, k_cols). If i > k_cols,
                # C[:, :i] is just C[:, :k_cols]; residual against V_exact[:, :i]
                # has i orthogonal true directions of which only k_cols are
                # representable, so align = sqrt(i - ||C||_F^2) is valid as long
                # as i <= k_cols. If i > k_cols we skip.
                continue
            frob2 = float(np.sum(C[:, :i] ** 2))
            residual2 = max(i - frob2, 0.0)
            per_i_accum[i].append(np.sqrt(residual2))
    return {i: (sum(v) / len(v) if v else None) for i, v in per_i_accum.items()}


def load_all(max_i=6):
    results = defaultdict(dict)  # (k, ssize, sr, perm) -> {method: {i -> align}}
    for d in sorted(os.listdir(OUT)):
        m = PAT.match(d)
        if not m:
            continue
        folder = os.path.join(OUT, d)
        if not os.path.exists(os.path.join(folder, "other_info.txt")):
            continue
        meth = m.group("method")
        if meth not in ("entropyscore_hybrid", "isvd"):
            continue
        key = (int(m.group("k")), int(m.group("ssize")), int(m.group("sr")), m.group("perm"))
        mm = align_i_per_block(folder, max_i=max_i)
        if mm is not None:
            results[key][meth] = mm
    return results


MAX_I = 6
results = load_all(MAX_I)

by_config = defaultdict(lambda: {"hybrid": [], "isvd": []})
for (k, ssize, sr, perm), methods in results.items():
    for meth, mm in methods.items():
        short = "hybrid" if meth == "entropyscore_hybrid" else "isvd"
        by_config[(k, ssize, sr)][short].append(mm)

def col_avg(recs, i):
    vals = [r[i] for r in recs if r.get(i) is not None]
    return sum(vals) / len(vals) if vals else float("nan")

header_i = "  ".join(f"i={i:<12}" for i in range(1, MAX_I + 1))
print(f"{'k':>3} {'win':>3} {'sr':>2}  perms(h/i)   {header_i}")
print("-" * (22 + 15 * MAX_I))

hw = iw = tie = 0
for key in sorted(by_config):
    rec = by_config[key]
    if not rec["hybrid"] or not rec["isvd"]:
        continue
    k, ssize, sr = key
    hrec = rec["hybrid"]; irec = rec["isvd"]
    parts = []
    hybrid_better_count = 0
    isvd_better_count = 0
    for i in range(1, MAX_I + 1):
        h = col_avg(hrec, i)
        ii = col_avg(irec, i)
        if h != h or ii != ii:
            parts.append(f"  {'--':>6} {'--':>6}")
            continue
        if h < ii * 0.98:
            marker = "H"
            hybrid_better_count += 1
        elif ii < h * 0.98:
            marker = "I"
            isvd_better_count += 1
        else:
            marker = "-"
        parts.append(f"{h:>6.3f}/{ii:<6.3f}{marker}")
    if hybrid_better_count > isvd_better_count:
        hw += 1
    elif isvd_better_count > hybrid_better_count:
        iw += 1
    else:
        tie += 1
    print(f"{k:>3} {ssize:>3} {sr:>2}  {len(hrec):>2}/{len(irec):<2}        {'  '.join(parts)}")

print()
print(f"configs: hybrid-wins-overall={hw}, isvd-wins-overall={iw}, tie/mixed={tie}")
print("(H/I marker per i-slot: 'H' = hybrid >2% lower residual, 'I' = isvd >2% lower, '-' = within 2%)")
