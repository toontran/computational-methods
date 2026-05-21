"""Aggregate per-matrix JSON summaries for the 5-policy comparison.

Pulls combined / hybrid / isvd / future_hmean_online from the existing
bench_matrix_sweep run (those JSONs are *not* re-run) and merges
future_hmean_evidence from this directory's JSONs.
"""
import json
import os
import glob
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.normpath(os.path.join(HERE, "..", "bench_matrix_sweep"))
MTS_FILE = os.path.normpath(
    os.path.join(HERE, "..", "benchmark_online_vs_baselines_win64.json")
)

POLICIES = ["combined", "hybrid", "isvd", "future_hmean_online", "future_hmean_evidence"]
SHORT = {
    "combined": "comb",
    "hybrid": "hybr",
    "isvd": "isvd",
    "future_hmean_online": "online",
    "future_hmean_evidence": "evi",
}


def best_policy(rows, key, higher_is_better=True):
    vals = {p: rows[p][key] for p in rows if rows[p][key] == rows[p][key]}
    if not vals:
        return "-"
    return max(vals, key=vals.get) if higher_is_better else min(vals, key=vals.get)


def load_sliding(fp):
    with open(fp) as f:
        d = json.load(f)
    out = {}
    elapsed_by_policy = {}
    for r in d.get("results", []):
        if r.get("mode") != "sliding":
            continue
        elapsed_by_policy[r["policy"]] = r.get("elapsed")
    for s in d["summaries"]:
        if s["mode"] != "sliding":
            continue
        elapsed = s.get("elapsed") or elapsed_by_policy.get(s["policy"])
        sec_per_step = (elapsed / s["steps"]) if elapsed is not None and s["steps"] else None
        out[s["policy"]] = {
            "mean_align": s["mean_align"],
            "mean_cos1": s["mean_cos1"],
            "mean_cos2": s["mean_cos2"],
            "mean_exact_cos1": s["mean_exact_cos1"],
            "mean_exact_cos2": s["mean_exact_cos2"],
            "mean_relerr_sval": s["mean_relerr_sval"],
            "final_tail_mass": s["final_tail_mass"],
            "final_exact_cos1": s["final_exact_cos"][0],
            "final_exact_cos2": s["final_exact_cos"][1],
            "steps": s["steps"],
            "elapsed": elapsed,
            "sec_per_step": sec_per_step,
        }
    return out


def merge(base, other):
    """Merge `other` per-policy dict into `base` (in-place)."""
    for k, v in other.items():
        base[k] = v
    return base


def main():
    table = {}

    # Existing 4-policy data (do not re-run).
    for fp in sorted(glob.glob(os.path.join(SWEEP_DIR, "*_win64.json"))):
        matrix = os.path.basename(fp).replace("_win64.json", "")
        table[matrix] = load_sliding(fp)
    if os.path.exists(MTS_FILE):
        table["mixed-tail-sharp"] = load_sliding(MTS_FILE)

    # New 5th policy (this directory).
    for fp in sorted(glob.glob(os.path.join(HERE, "*_win64.json"))):
        matrix = os.path.basename(fp).replace("_win64.json", "")
        evi_data = load_sliding(fp)
        if matrix in table:
            merge(table[matrix], evi_data)
        else:
            table[matrix] = evi_data

    metrics = [
        "mean_exact_cos1", "mean_exact_cos2", "mean_relerr_sval",
        "final_exact_cos1", "final_exact_cos2", "final_tail_mass",
        "elapsed", "sec_per_step",
    ]

    header = (
        f"{'matrix':<30} "
        + " ".join(f"{SHORT[p]:<9}" for p in POLICIES)
        + "  winner"
    )
    for metric in metrics:
        higher = metric not in ("mean_relerr_sval", "final_tail_mass", "elapsed", "sec_per_step")
        print(f"\n=== {metric} ({'higher' if higher else 'lower'} is better) ===")
        print(header)
        print("-" * len(header))
        for matrix in sorted(table):
            rows = table[matrix]
            if not rows:
                continue
            vals = []
            for p in POLICIES:
                v = rows.get(p, {}).get(metric, float("nan"))
                if v is None or v != v:
                    vals.append(f"{'--':<9}")
                elif metric in ("elapsed", "sec_per_step"):
                    vals.append(f"{v:<9.3f}")
                else:
                    vals.append(f"{v:<9.4f}")
            winner = best_policy(rows, metric, higher_is_better=higher)
            print(f"{matrix:<30} {' '.join(vals)}  {winner}")

    print("\n=== Win counts across 17 matrices (sliding) ===")
    print(f"{'metric':<22} " + " ".join(f"{SHORT[p]:<9}" for p in POLICIES))
    for metric in metrics:
        higher = metric not in ("mean_relerr_sval", "final_tail_mass", "elapsed", "sec_per_step")
        wins = {p: 0 for p in POLICIES}
        for matrix, rows in table.items():
            if not rows:
                continue
            w = best_policy(rows, metric, higher_is_better=higher)
            if w in wins:
                wins[w] += 1
        print(f"{metric:<22} " + " ".join(f"{wins[p]:<9}" for p in POLICIES))

    print("\n=== Elapsed (s) summary across 17 matrices ===")
    print(f"{'policy':<26} {'min':<8} {'median':<8} {'mean':<8} {'max':<8} {'total':<8}")
    for p in POLICIES:
        vals = [
            rows[p]["elapsed"]
            for rows in table.values()
            if p in rows and rows[p].get("elapsed") is not None
        ]
        if not vals:
            continue
        print(
            f"{p:<26} {min(vals):<8.2f} {statistics.median(vals):<8.2f} "
            f"{sum(vals)/len(vals):<8.2f} {max(vals):<8.2f} {sum(vals):<8.1f}"
        )

    isvd_med = statistics.median([rows["isvd"]["elapsed"] for rows in table.values() if "isvd" in rows])
    print("\n=== Speedup of isvd vs each other policy (median across 17 matrices) ===")
    for p in POLICIES:
        if p == "isvd":
            continue
        vals = [rows[p]["elapsed"] for rows in table.values() if p in rows]
        if not vals:
            continue
        med = statistics.median(vals)
        print(f"  isvd is {med/isvd_med:>5.1f}x faster than {p}")

    print("\n=== future_hmean_evidence vs each baseline (W/T/L across 17 matrices) ===")
    for p in ["combined", "hybrid", "isvd", "future_hmean_online"]:
        for metric in metrics:
            higher = metric not in ("mean_relerr_sval", "final_tail_mass", "elapsed", "sec_per_step")
            w = l = t = 0
            for matrix, rows in table.items():
                if p not in rows or "future_hmean_evidence" not in rows:
                    continue
                vb = rows[p][metric]
                vo = rows["future_hmean_evidence"][metric]
                if vb != vb or vo != vo:
                    continue
                if abs(vo - vb) < 1e-6:
                    t += 1
                elif (vo > vb) == higher:
                    w += 1
                else:
                    l += 1
            print(f"  vs {p:<22} {metric:<22} evi W/T/L = {w}/{t}/{l}")


if __name__ == "__main__":
    main()
