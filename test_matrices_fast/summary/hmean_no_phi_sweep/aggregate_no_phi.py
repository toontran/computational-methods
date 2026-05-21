import csv
import glob
import json
import os
import statistics


HERE = os.path.dirname(os.path.abspath(__file__))
PHI_DIR = os.path.abspath(os.path.join(HERE, "..", "hmean_combinations_sweep"))
POLICIES = [
    "future_hmean_online",
    "future_hmean_online_no_phi",
    "future_hmean_triplet_online_no_phi",
    "future_hmean_nested_online_no_phi",
    "future_hmean_pairwise_online_no_phi",
    "future_hmean_weighted_online_no_phi",
]
PHI_POLICIES = [
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


def valid(value):
    return value is not None and value == value


def load_sliding(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    elapsed = {
        r["policy"]: r.get("elapsed")
        for r in data.get("results", [])
        if r.get("mode") == "sliding"
    }
    out = {}
    for s in data["summaries"]:
        if s["mode"] != "sliding":
            continue
        policy = s["policy"]
        sec = elapsed.get(policy, s.get("elapsed"))
        steps = s["steps"]
        out[policy] = {
            "mean_exact_cos1": s["mean_exact_cos1"],
            "mean_exact_cos2": s["mean_exact_cos2"],
            "mean_relerr_sval": s["mean_relerr_sval"],
            "final_exact_cos1": s["final_exact_cos"][0],
            "final_exact_cos2": s["final_exact_cos"][1],
            "final_tail_mass": s["final_tail_mass"],
            "elapsed": sec,
            "sec_per_step": (sec / steps) if sec is not None and steps else None,
            "steps": steps,
            "labels": s["selected_label_counts"],
        }
    return out


def load_table(directory):
    table = {}
    for path in sorted(glob.glob(os.path.join(directory, "*_win64.json"))):
        matrix = os.path.basename(path).replace("_win64.json", "")
        table[matrix] = load_sliding(path)
    return table


def winner(rows, policies, metric, higher):
    vals = {
        policy: rows[policy][metric]
        for policy in policies
        if policy in rows and valid(rows[policy].get(metric))
    }
    if not vals:
        return "-"
    return max(vals, key=vals.get) if higher else min(vals, key=vals.get)


def fmt(value, metric):
    if not valid(value):
        return "--"
    if metric in ("elapsed", "sec_per_step"):
        return f"{value:.3f}"
    return f"{value:.4f}"


def main():
    no_phi = load_table(HERE)
    phi = load_table(PHI_DIR) if os.path.isdir(PHI_DIR) else {}

    csv_path = os.path.join(HERE, "matrix_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = ["matrix", "policy"] + [m for m, _ in METRICS] + ["steps", "labels"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for matrix, rows in sorted(no_phi.items()):
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
    lines.append("No-phi HM combinations sweep report")
    lines.append("===================================")
    lines.append("")
    lines.append("Scope")
    lines.append("-----")
    lines.append("All 17 benchmark matrices, half_win=32, rank=2, seed=0, float32, sliding-mode metrics.")
    lines.append("No-phi policies remove the current-window entropy multiplier relH1 from the score.")
    lines.append("")

    lines.append("Win counts within no-phi sweep")
    lines.append("------------------------------")
    header = f"{'metric':<22} " + " ".join(f"{p[:18]:<19}" for p in POLICIES)
    lines.append(header)
    for metric, higher in METRICS:
        counts = {policy: 0 for policy in POLICIES}
        for rows in no_phi.values():
            w = winner(rows, POLICIES, metric, higher)
            if w in counts:
                counts[w] += 1
        lines.append(f"{metric:<22} " + " ".join(f"{counts[p]:<19}" for p in POLICIES))
    lines.append("")

    lines.append("Head-to-head vs phi future_hmean_online")
    lines.append("---------------------------------------")
    for policy in POLICIES[1:]:
        lines.append(f"vs {policy}")
        for metric, higher in METRICS[:6]:
            w = t = l = 0
            deltas = []
            for matrix, rows in no_phi.items():
                if policy not in rows or matrix not in phi or "future_hmean_online" not in phi[matrix]:
                    continue
                base = phi[matrix]["future_hmean_online"].get(metric)
                val = rows[policy].get(metric)
                if not valid(base) or not valid(val):
                    continue
                if abs(val - base) < 1e-6:
                    t += 1
                elif (val > base) == higher:
                    w += 1
                else:
                    l += 1
                deltas.append(val - base)
            mean_delta = sum(deltas) / len(deltas) if deltas else float("nan")
            lines.append(f"  {metric:<22} W/T/L={w}/{t}/{l} mean_delta={fmt(mean_delta, metric)}")
        lines.append("")

    if phi:
        lines.append("Best phi/no-phi policy by metric")
        lines.append("--------------------------------")
        all_policies = PHI_POLICIES + POLICIES[1:]
        for metric, higher in METRICS[:6]:
            lines.append(f"{metric} ({'higher' if higher else 'lower'} is better)")
            for matrix in sorted(no_phi):
                rows = {}
                rows.update(phi.get(matrix, {}))
                rows.update(no_phi.get(matrix, {}))
                w = winner(rows, all_policies, metric, higher)
                lines.append(f"  {matrix:<30} winner={w:<39} value={fmt(rows.get(w, {}).get(metric), metric)}")
            lines.append("")

    lines.append("Elapsed summary")
    lines.append("---------------")
    lines.append(f"{'policy':<40} {'median_s':<10} {'mean_s':<10} {'total_s':<10}")
    for policy in POLICIES:
        vals = [rows[policy]["elapsed"] for rows in no_phi.values() if policy in rows and rows[policy].get("elapsed") is not None]
        lines.append(f"{policy:<40} {statistics.median(vals):<10.3f} {sum(vals)/len(vals):<10.3f} {sum(vals):<10.3f}")
    lines.append("")
    lines.append(f"Raw per-matrix metrics CSV: {csv_path}")

    report_path = os.path.join(HERE, "hmean_no_phi_sweep_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {report_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
