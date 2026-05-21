# save as: run_cex_ablation.py
#
# Rewritten from scratch for the actual stdout format of cex_restricted_space_probe.py.
#
# What it does
# ------------
# Runs 4 ablations:
#   1) baseline
#   2) warm_start_only
#   3) continuation_only
#   4) warm_start_plus_continuation
#
# and parses metrics from stdout like:
#   rows 1:32
#   s:  0.9529  0.9505
#   H:  3.4541  3.4374
#   scores:  0.9039  0.9029
#   subspace_dims: [20, 32]
#   grad_perp_ratio:  0.999995  0.000389
#   sigma1    mean_align    mean_relerr_sval    elapsed
#   0.991      0.075217           0.00025843          2.178
#
# It writes:
#   ablation_runs/<timestamp>/summary.csv
#   ablation_runs/<timestamp>/summary.json
#   ablation_runs/<timestamp>/block_summary.csv
#   ablation_runs/<timestamp>/logs/*.stdout.txt
#   ablation_runs/<timestamp>/logs/*.stderr.txt
#
# Example:
#   python run_cex_ablation_probe.py --repeats 5 --verbose
#
# Optional:
#   python run_cex_ablation_probe.py --extra "--matrix static-cex --mode restricted --dtype float64"

import argparse
import csv
import datetime as dt
import json
import math
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path


def maybe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def min_or_none(xs):
    xs = [x for x in xs if x is not None]
    return min(xs) if xs else None


def max_or_none(xs):
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def stdev_or_none(xs):
    xs = [x for x in xs if x is not None]
    return statistics.pstdev(xs) if len(xs) >= 2 else (0.0 if len(xs) == 1 else None)


def fmt(x):
    if x is None:
        return "NA"
    return f"{x:.6g}"


def parse_two_floats_from_line(prefix, text):
    m = re.search(
        rf"(?m)^{re.escape(prefix)}\s*:\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)\s*$",
        text,
    )
    if not m:
        return None, None
    return maybe_float(m.group(1)), maybe_float(m.group(2))


def parse_final_table(text):
    """
    Parse the final 2-line table:
      sigma1    mean_align    mean_relerr_sval    elapsed
      0.991      0.075217           0.00025843          2.178
    """
    lines = text.splitlines()
    for i in range(len(lines) - 1):
        if re.search(r"\bsigma1\b", lines[i]) and re.search(r"\bmean_align\b", lines[i]):
            nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", lines[i + 1])
            if len(nums) >= 4:
                return {
                    "sigma1": maybe_float(nums[0]),
                    "mean_align": maybe_float(nums[1]),
                    "mean_relerr_sval": maybe_float(nums[2]),
                    "elapsed": maybe_float(nums[3]),
                }
    return {
        "sigma1": None,
        "mean_align": None,
        "mean_relerr_sval": None,
        "elapsed": None,
    }


def parse_block_sections(text):
    """
    Parse repeated sections of the form:
      rows 1:32
      s:  0.9529  0.9505
      H:  3.4541  3.4374
      scores:  0.9039  0.9029
      subspace_dims: [20, 32]
      grad_perp_ratio:  0.999995  0.000389
    """
    lines = text.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r"^rows\s+(\d+):(\d+)\s*$", line)
        if not m:
            i += 1
            continue

        row_lo = int(m.group(1))
        row_hi = int(m.group(2))
        block = {
            "rows_lo": row_lo,
            "rows_hi": row_hi,
            "s1": None,
            "s2": None,
            "H1": None,
            "H2": None,
            "score1": None,
            "score2": None,
            "subspace_dim1": None,
            "subspace_dim2": None,
            "grad_perp_ratio1": None,
            "grad_perp_ratio2": None,
        }

        j = i + 1
        while j < len(lines):
            cur = lines[j].strip()
            if re.match(r"^rows\s+\d+:\d+\s*$", cur):
                break

            if cur.startswith("s:"):
                nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", cur)
                if len(nums) >= 2:
                    block["s1"] = maybe_float(nums[0])
                    block["s2"] = maybe_float(nums[1])

            elif cur.startswith("H:"):
                nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", cur)
                if len(nums) >= 2:
                    block["H1"] = maybe_float(nums[0])
                    block["H2"] = maybe_float(nums[1])

            elif cur.startswith("scores:"):
                nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", cur)
                if len(nums) >= 2:
                    block["score1"] = maybe_float(nums[0])
                    block["score2"] = maybe_float(nums[1])

            elif cur.startswith("subspace_dims:"):
                nums = re.findall(r"\d+", cur)
                if len(nums) >= 2:
                    block["subspace_dim1"] = int(nums[0])
                    block["subspace_dim2"] = int(nums[1])

            elif cur.startswith("grad_perp_ratio:"):
                nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", cur)
                if len(nums) >= 2:
                    block["grad_perp_ratio1"] = maybe_float(nums[0])
                    block["grad_perp_ratio2"] = maybe_float(nums[1])

            j += 1

        blocks.append(block)
        i = j

    return blocks


def parse_run(stdout, stderr):
    text = stdout + "\n" + stderr
    final_tbl = parse_final_table(text)
    blocks = parse_block_sections(text)

    result = dict(final_tbl)
    result["num_blocks"] = len(blocks)

    # Aggregate block metrics
    for key in [
        "s1", "s2", "H1", "H2", "score1", "score2",
        "subspace_dim1", "subspace_dim2",
        "grad_perp_ratio1", "grad_perp_ratio2"
    ]:
        vals = [b[key] for b in blocks]
        result[f"{key}_mean"] = mean_or_none(vals)
        result[f"{key}_min"] = min_or_none(vals)
        result[f"{key}_max"] = max_or_none(vals)

    # Last block metrics are often useful
    if blocks:
        last = blocks[-1]
        for key, val in last.items():
            result[f"last_{key}"] = val

    return result, blocks


def build_configs(args):
    common = [
        "--matrix", args.matrix,
        "--mode", args.mode,
        "--dtype", args.dtype,
        "--carry", args.carry,
        "--reduced-optimizer", args.reduced_optimizer,
        "--expansion-direction", args.expansion_direction,
        "--n", str(args.n),
        "--win", str(args.win),
        "--rank", str(args.rank),
        "--preset", args.preset,
        "--q0", str(args.q0),
        "--qmax", str(args.qmax),
        "--krylov-depth", str(args.krylov_depth),
        "--residual-tol", str(args.residual_tol),
        "--expansion-maxit", str(args.expansion_maxit),
        "--num-restarts", str(args.num_restarts),
        "--maxit", str(args.maxit),
        "--tol", str(args.tol),
        "--post-expansion-maxit", str(args.post_expansion_maxit),
    ]

    if args.normalize_by_sigma:
        common.append("--normalize-by-sigma")
    if args.reuse_line_search_grad:
        common.append("--reuse-line-search-grad")
    else:
        common.append("--no-reuse-line-search-grad")
    if args.expansion_warm_start:
        common.append("--expansion-warm-start")
    else:
        common.append("--no-expansion-warm-start")
    if args.verbose:
        common.append("--verbose")
    if args.mat_input:
        common += ["--mat-input", args.mat_input]
    if args.extra:
        common += shlex.split(args.extra)

    configs = [
        {
            "name": "baseline",
            "flags": [
                "--no-warm-start-greedy",
                "--warm-start-perturbations", "0",
                "--no-continuation",
            ],
        },
        {
            "name": "warm_start_only",
            "flags": [
                "--warm-start-greedy",
                "--warm-start-perturbations", str(args.warm_start_perturbations),
                "--warm-start-perturb-scale", str(args.warm_start_perturb_scale),
                "--no-continuation",
            ],
        },
        {
            "name": "continuation_only",
            "flags": [
                "--no-warm-start-greedy",
                "--warm-start-perturbations", "0",
                "--continuation",
                "--continuation-schedule", args.continuation_schedule,
            ],
        },
        {
            "name": "warm_start_plus_continuation",
            "flags": [
                "--warm-start-greedy",
                "--warm-start-perturbations", str(args.warm_start_perturbations),
                "--warm-start-perturb-scale", str(args.warm_start_perturb_scale),
                "--continuation",
                "--continuation-schedule", args.continuation_schedule,
            ],
        },
    ]

    for cfg in configs:
        cfg["cmd"] = [sys.executable, args.script] + common + cfg["flags"]
    return configs


def summarize_group(rows):
    keys = [
        "sigma1", "mean_align", "mean_relerr_sval", "elapsed",
        "num_blocks",
        "s1_mean", "s2_mean", "H1_mean", "H2_mean",
        "score1_mean", "score2_mean",
        "subspace_dim1_mean", "subspace_dim2_mean",
        "grad_perp_ratio1_mean", "grad_perp_ratio2_mean",
        "last_score1", "last_score2",
        "last_subspace_dim1", "last_subspace_dim2",
        "last_grad_perp_ratio1", "last_grad_perp_ratio2",
    ]

    out = {
        "num_runs": len(rows),
        "num_success": sum(int(r["returncode"] == 0) for r in rows),
    }
    for k in keys:
        vals = [r.get(k) for r in rows]
        out[f"{k}_mean"] = mean_or_none(vals)
        out[f"{k}_std"] = stdev_or_none(vals)
        out[f"{k}_min"] = min_or_none(vals)
        out[f"{k}_max"] = max_or_none(vals)
    return out


def print_summary_table(summary):
    print("\n=== Aggregate summary ===")
    header = (
        f"{'ablation':34s} "
        f"{'succ':>5s} "
        f"{'align':>10s} "
        f"{'relerr':>10s} "
        f"{'elapsed':>10s} "
        f"{'score1':>10s} "
        f"{'score2':>10s}"
    )
    print(header)
    print("-" * len(header))
    for name, s in summary.items():
        succ = f"{s['num_success']}/{s['num_runs']}"
        print(
            f"{name:34s} "
            f"{succ:>5s} "
            f"{fmt(s['mean_align_mean']):>10s} "
            f"{fmt(s['mean_relerr_sval_mean']):>10s} "
            f"{fmt(s['elapsed_mean']):>10s} "
            f"{fmt(s['score1_mean_mean']):>10s} "
            f"{fmt(s['score2_mean_mean']):>10s}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--script", default="cex_restricted_space_probe.py")
    parser.add_argument("--matrix", default="static-cex")
    parser.add_argument("--mode", default="restricted")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--carry", default="left")
    parser.add_argument("--reduced-optimizer", default="cex")
    parser.add_argument("--expansion-direction", default="residual")

    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--win", type=int, default=32)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--preset", default="cex-replicate")

    parser.add_argument("--q0", type=int, default=4)
    parser.add_argument("--qmax", type=int, default=32)
    parser.add_argument("--krylov-depth", type=int, default=2)
    parser.add_argument("--residual-tol", type=float, default=1e-4)
    parser.add_argument("--expansion-maxit", type=int, default=8)
    parser.add_argument("--num-restarts", type=int, default=8)
    parser.add_argument("--maxit", type=int, default=250)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--post-expansion-maxit", type=int, default=120)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=1800)

    parser.add_argument("--warm-start-perturbations", type=int, default=4)
    parser.add_argument("--warm-start-perturb-scale", type=float, default=1e-2)
    parser.add_argument("--continuation-schedule", default="0.0,0.25,0.5,0.75,1.0")

    parser.add_argument("--mat-input", default=None)
    parser.add_argument("--extra", default="")

    parser.add_argument("--normalize-by-sigma", action="store_true")
    parser.add_argument("--reuse-line-search-grad", action="store_true", default=True)
    parser.add_argument("--expansion-warm_start_alias", dest="expansion_warm_start", action="store_true", default=True)
    parser.add_argument("--no-expansion-warm-start", dest="expansion_warm_start", action="store_false")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("ablation_runs") / timestamp
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    configs = build_configs(args)

    run_rows = []
    block_rows = []

    for cfg in configs:
        for rep in range(args.repeats):
            cmd = list(cfg["cmd"])
            if args.seed is None:
                rep_seed = 1000 + rep
                cmd += ["--seed", str(rep_seed)]
            else:
                rep_seed = args.seed
                cmd += ["--seed", str(rep_seed)]

            run_name = f"{cfg['name']}__rep{rep:02d}"
            stdout_path = log_dir / f"{run_name}.stdout.txt"
            stderr_path = log_dir / f"{run_name}.stderr.txt"

            print(f"\n=== Running {run_name} ===")
            print(" ".join(shlex.quote(x) for x in cmd))

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                    check=False,
                )
                stdout = proc.stdout
                stderr = proc.stderr
                rc = proc.returncode
            except subprocess.TimeoutExpired as e:
                stdout = e.stdout or ""
                stderr = (e.stderr or "") + "\n[TIMEOUT]"
                rc = -999

            stdout_path.write_text(stdout, encoding="utf-8", errors="ignore")
            stderr_path.write_text(stderr, encoding="utf-8", errors="ignore")

            parsed, blocks = parse_run(stdout, stderr)

            row = {
                "ablation": cfg["name"],
                "repeat": rep,
                "seed": rep_seed,
                "returncode": rc,
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "cmd": " ".join(shlex.quote(x) for x in cmd),
            }
            row.update(parsed)
            run_rows.append(row)

            for bi, b in enumerate(blocks):
                brow = {
                    "ablation": cfg["name"],
                    "repeat": rep,
                    "seed": rep_seed,
                    "block_index": bi + 1,
                }
                brow.update(b)
                block_rows.append(brow)

    # Raw run CSV
    run_csv = out_dir / "summary.csv"
    run_fields = sorted({k for r in run_rows for k in r.keys()})
    with run_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(run_rows)

    # Block CSV
    block_csv = out_dir / "block_summary.csv"
    if block_rows:
        block_fields = sorted({k for r in block_rows for k in r.keys()})
        with block_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=block_fields)
            writer.writeheader()
            writer.writerows(block_rows)

    # Aggregate JSON
    grouped = {}
    for r in run_rows:
        grouped.setdefault(r["ablation"], []).append(r)
    summary = {name: summarize_group(rows) for name, rows in grouped.items()}

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_summary_table(summary)
    print(f"\nRaw per-run CSV:   {run_csv}")
    print(f"Per-block CSV:     {block_csv}")
    print(f"Aggregate JSON:    {summary_json}")
    print(f"Logs directory:    {log_dir}")

    print(
        "\nRecommended comparison:\n"
        "  baseline vs warm_start_only\n"
        "  baseline vs continuation_only\n"
        "  warm_start_plus_continuation vs all others\n"
        "Focus on mean_align, mean_relerr_sval, elapsed, and score1/score2 block means.\n"
    )


if __name__ == "__main__":
    main()
    