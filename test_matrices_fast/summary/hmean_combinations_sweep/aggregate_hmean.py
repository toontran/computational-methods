import csv
import glob
import json
import os
import statistics


HERE = os.path.dirname(os.path.abspath(__file__))
POLICIES = [
    "future_hmean_online",
    "future_hmean_triplet_online",
    "future_hmean_nested_online",
    "future_hmean_pairwise_online",
    "future_hmean_weighted_online",
]
METRICS = [
    ("mean_exact_cos1", True),
    ("mean_exact_cos2", True),
    ("mean_relerr_sval", False),
    ("final_exact_cos1", True),
    ("final_exact_cos2", True),
    ("final_tail_mass", False),
    ("elapsed", False),
    ("sec_per_step", False),
]


def load_sliding(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    elapsed = {
        r["policy"]: r.get("elapsed")
        for r in data.get("results", [])
        if r.get("mode") == "sliding"
    }
    out = {}
    for summary in data["summaries"]:
        if summary["mode"] != "sliding":
            continue
        policy = summary["policy"]
        sec = elapsed.get(policy, summary.get("elapsed"))
        steps = summary["steps"]
        out[policy] = {
            "mean_exact_cos1": summary["mean_exact_cos1"],
            "mean_exact_cos2": summary["mean_exact_cos2"],
            "mean_relerr_sval": summary["mean_relerr_sval"],
            "final_exact_cos1": summary["final_exact_cos"][0],
            "final_exact_cos2": summary["final_exact_cos"][1],
            "final_tail_mass": summary["final_tail_mass"],
            "elapsed": sec,
            "sec_per_step": (sec / steps) if sec is not None and steps else None,
            "steps": steps,
            "labels": summary["selected_label_counts"],
        }
    return out


def valid_value(value):
    return value is not None and value == value


def winner(rows, metric, higher):
    vals = {
        policy: rows[policy][metric]
        for policy in POLICIES
        if policy in rows and valid_value(rows[policy].get(metric))
    }
    if not vals:
        return "-"
    return max(vals, key=vals.get) if higher else min(vals, key=vals.get)


def fmt(value, metric):
    if not valid_value(value):
        return "--"
    if metric in ("elapsed", "sec_per_step"):
        return f"{value:.3f}"
    return f"{value:.4f}"


def main():
    table = {}
    for path in sorted(glob.glob(os.path.join(HERE, "*_win64.json"))):
        matrix = os.path.basename(path).replace("_win64.json", "")
        table[matrix] = load_sliding(path)

    csv_path = os.path.join(HERE, "matrix_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = ["matrix", "policy"] + [m for m, _ in METRICS] + ["steps", "labels"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for matrix, rows in sorted(table.items()):
            for policy in POLICIES:
                rec = rows.get(policy)
                if rec is None:
                    continue
                writer.writerow(
                    {
                        "matrix": matrix,
                        "policy": policy,
                        **{metric: rec.get(metric) for metric, _ in METRICS},
                        "steps": rec.get("steps"),
                        "labels": json.dumps(rec.get("labels", {}), sort_keys=True),
                    }
                )

    lines = []
    lines.append("HM combinations sweep report")
    lines.append("============================")
    lines.append("")
    lines.append("Scope")
    lines.append("-----")
    lines.append("All 17 benchmark matrices, half_win=32, rank=2, seed=0, float32, sliding-mode metrics.")
    lines.append("Compared policies:")
    for policy in POLICIES:
        lines.append(f"  - {policy}")
    lines.append("")
    lines.append("Corrected parenthesized semantics")
    lines.append("---------------------------------")
    lines.append("  - future_hmean_nested_online uses HM(||[sketch; block1]v||^2, ||block2 v||^2) * relH1.")
    lines.append("  - future_hmean_pairwise_online uses HM(||[sketch; block1]v||^2, ||[sketch; block2]v||^2) * relH1.")
    lines.append("  - future_hmean_weighted_online uses weights (rows through N_{w-1}, |N_w|, |N_{w+1}|).")
    lines.append("")

    lines.append("Win counts across 17 matrices (sliding)")
    lines.append("---------------------------------------")
    header = f"{'metric':<22} " + " ".join(f"{p[:18]:<19}" for p in POLICIES)
    lines.append(header)
    win_counts = {}
    for metric, higher in METRICS:
        counts = {policy: 0 for policy in POLICIES}
        for rows in table.values():
            w = winner(rows, metric, higher)
            if w in counts:
                counts[w] += 1
        win_counts[metric] = counts
        lines.append(f"{metric:<22} " + " ".join(f"{counts[p]:<19}" for p in POLICIES))
    lines.append("")

    lines.append("Head-to-head vs future_hmean_online")
    lines.append("-----------------------------------")
    for policy in POLICIES[1:]:
        lines.append(f"vs {policy}")
        for metric, higher in METRICS:
            w = t = l = 0
            diffs = []
            for rows in table.values():
                if "future_hmean_online" not in rows or policy not in rows:
                    continue
                base = rows["future_hmean_online"].get(metric)
                val = rows[policy].get(metric)
                if not valid_value(base) or not valid_value(val):
                    continue
                if abs(val - base) < 1e-6:
                    t += 1
                elif (val > base) == higher:
                    w += 1
                else:
                    l += 1
                diffs.append(val - base)
            mean_delta = sum(diffs) / len(diffs) if diffs else float("nan")
            lines.append(f"  {metric:<22} W/T/L={w}/{t}/{l} mean_delta={fmt(mean_delta, metric)}")
        lines.append("")

    lines.append("Per-matrix winners")
    lines.append("------------------")
    for metric, higher in METRICS[:6]:
        lines.append(f"{metric} ({'higher' if higher else 'lower'} is better)")
        for matrix, rows in sorted(table.items()):
            w = winner(rows, metric, higher)
            vals = " ".join(f"{policy[:18]}={fmt(rows.get(policy, {}).get(metric), metric)}" for policy in POLICIES)
            lines.append(f"  {matrix:<30} winner={w:<31} {vals}")
        lines.append("")

    lines.append("Elapsed summary")
    lines.append("---------------")
    lines.append(f"{'policy':<32} {'median_s':<10} {'mean_s':<10} {'total_s':<10}")
    for policy in POLICIES:
        vals = [rows[policy]["elapsed"] for rows in table.values() if policy in rows and rows[policy].get("elapsed") is not None]
        lines.append(f"{policy:<32} {statistics.median(vals):<10.3f} {sum(vals)/len(vals):<10.3f} {sum(vals):<10.3f}")
    lines.append("")
    lines.append(f"Raw per-matrix metrics CSV: {csv_path}")

    report_path = os.path.join(HERE, "hmean_combinations_sweep_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {report_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
