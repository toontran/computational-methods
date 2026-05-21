"""FAM-09 A0 stability-weighted evidence prototype.

This is a diagnostic-only prototype. It reuses DIAG-03 candidate collection and
row-subsample helpers, then computes

    u_X_stab = u_X / (1 + lambda * CV_X)

on a small block set. It does not optimize a new score and does not run T3.

Run from repo root:
    python summary/score_family_stability_weighted_evidence/prototype_A0.py
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import cex_restricted_space_probe as matrix_probe
from hmean_evidence_score import per_block_constants, stream_to_block
from summary.diag03_subsample_stability import probe as diag03


EPS = 1e-30


def _u_full(A, v):
    if A is None or np.asarray(A).size == 0 or v is None:
        return float("nan")
    A = np.asarray(A, dtype=np.float64)
    y = A @ v
    return float(np.dot(y, y)) / max(float(np.sum(A * A)), EPS)


def _hm(values):
    vals = [float(x) for x in values if np.isfinite(x)]
    if not vals or any(x <= EPS for x in vals):
        return 0.0
    return float(len(vals) / sum(1.0 / x for x in vals))


def _cv(A, v, frac, n_draws, rng):
    if A is None or np.asarray(A).size == 0:
        return float("nan")
    arr = diag03._subsample_u(A, v, frac, n_draws, rng)
    stats = diag03._summary_stats(arr)
    return float(stats["cv"])


def _weighted(u, cv, lam):
    if not np.isfinite(u):
        return float("nan")
    if not np.isfinite(cv):
        return float(u)
    return float(u) / (1.0 + float(lam) * max(float(cv), 0.0))


def _args_for_stream(args):
    """Use DIAG-03 defaults for the streaming scaffold, with local overrides."""
    argv = sys.argv
    try:
        sys.argv = [argv[0]]
        ns = diag03.parse_args()
    finally:
        sys.argv = argv
    ns.n = args.n
    ns.half_win = args.half_win
    ns.rank = args.rank
    ns.preset = args.preset
    ns.seed = args.seed
    ns.n_subsamples = args.n_subsamples
    return ns


def run_matrix(args, matrix, rng_master):
    stream_args = _args_for_stream(args)
    work_dtype = np.float32 if stream_args.dtype == "float32" else np.float64
    A, V_exact, _, _ = matrix_probe.generate_matrix_input(
        matrix=matrix,
        n=stream_args.n,
        preset=stream_args.preset,
        seed=stream_args.seed,
        r_sig=stream_args.r_sig,
        alpha_sig=stream_args.alpha_sig,
        alpha_tail=stream_args.alpha_tail,
        tail_scale=stream_args.tail_scale,
        sigma1=stream_args.sigma1,
        v_type=stream_args.v_type,
        shuffle_rows=stream_args.shuffle_rows,
        row_shuffle_seed=stream_args.row_shuffle_seed,
    )
    A = np.asarray(A, dtype=np.float64)
    V_exact = np.asarray(V_exact, dtype=np.float64)

    blocks = sorted(set(args.blocks))
    snaps = stream_to_block(
        stream_args,
        A,
        V_exact,
        work_dtype,
        int(stream_args.rank),
        max(blocks),
        set(blocks),
    )

    rows = []
    for block in blocks:
        snap = snaps.get(block)
        if snap is None:
            continue
        consts = per_block_constants(A, block, args.half_win)
        A_sketch = snap["A_sketch"] if snap["A_sketch"].size else None
        A_cur = snap["A_cur"]
        A_fut = snap["A_fut"]
        c_sk, c_g1, c_g2 = consts["c_sk"], consts["c_g1"], consts["c_g2"]

        for slot in (0, 1):
            panel = diag03.collect_panel(stream_args, snap, V_exact, slot, c_sk, c_g1, c_g2)
            for candidate, v in sorted(panel.items()):
                rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
                u_sk = _u_full(A_sketch, v)
                u_g1 = _u_full(A_cur, v)
                u_g2 = _u_full(A_fut, v)
                cv_sk = _cv(A_sketch, v, args.frac, args.n_subsamples, rng)
                cv_g1 = _cv(A_cur, v, args.frac, args.n_subsamples, rng)
                cv_g2 = _cv(A_fut, v, args.frac, args.n_subsamples, rng)

                wu_sk = _weighted(u_sk, cv_sk, args.lam)
                wu_g1 = _weighted(u_g1, cv_g1, args.lam)
                wu_g2 = _weighted(u_g2, cv_g2, args.lam)
                hm_raw = _hm([u_g1, u_g2] if A_sketch is None else [u_sk, u_g1, u_g2])
                hm_stab = _hm(
                    [wu_g1, wu_g2] if A_sketch is None else [wu_sk, wu_g1, wu_g2]
                )
                oracle = V_exact[:, slot]
                oracle = oracle / max(float(np.linalg.norm(oracle)), EPS)
                align_oracle = float(np.dot(v, oracle) ** 2)
                rows.append({
                    "matrix": matrix,
                    "block": block,
                    "slot": slot + 1,
                    "candidate": candidate,
                    "lambda": args.lam,
                    "frac": args.frac,
                    "n_subsamples": args.n_subsamples,
                    "u_sk": u_sk,
                    "u_g1": u_g1,
                    "u_g2": u_g2,
                    "cv_sk": cv_sk,
                    "cv_g1": cv_g1,
                    "cv_g2": cv_g2,
                    "u_sk_stab": wu_sk,
                    "u_g1_stab": wu_g1,
                    "u_g2_stab": wu_g2,
                    "hm_raw": hm_raw,
                    "hm_stab": hm_stab,
                    "stab_over_raw": hm_stab / max(hm_raw, EPS),
                    "align_oracle_slot": align_oracle,
                })
    return rows


def add_ranks(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["matrix"], row["block"], row["slot"]), []).append(row)
    for group in grouped.values():
        for score_key, rank_key in (("hm_raw", "rank_raw"), ("hm_stab", "rank_stab")):
            ordered = sorted(group, key=lambda r: r[score_key], reverse=True)
            for rank, row in enumerate(ordered, start=1):
                row[rank_key] = rank


def write_csv(path, rows):
    fields = [
        "matrix", "block", "slot", "candidate", "lambda", "frac", "n_subsamples",
        "rank_raw", "rank_stab", "hm_raw", "hm_stab", "stab_over_raw",
        "align_oracle_slot", "u_sk", "u_g1", "u_g2", "cv_sk", "cv_g1", "cv_g2",
        "u_sk_stab", "u_g1_stab", "u_g2_stab",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(path, rows, args, elapsed):
    groups = {}
    for row in rows:
        groups.setdefault((row["matrix"], row["block"], row["slot"]), []).append(row)

    lines = [
        "# FAM-09 A0 prototype summary",
        "",
        f"Command defaults: matrices={args.matrices}, blocks={args.blocks}, "
        f"lambda={args.lam}, frac={args.frac}, n_subsamples={args.n_subsamples}, "
        f"n={args.n}, half_win={args.half_win}.",
        f"Elapsed seconds: {elapsed:.2f}",
        "",
        "This is a panel reranking diagnostic, not an optimizer or T3 bench.",
        "",
        "| matrix | block | slot | raw winner | stab winner | oracle raw rank | oracle stab rank |",
        "|---|---:|---:|---|---|---:|---:|",
    ]
    oracle_deltas = []
    for key in sorted(groups):
        group = groups[key]
        raw_w = min(group, key=lambda r: r["rank_raw"])
        stab_w = min(group, key=lambda r: r["rank_stab"])
        oracle_rows = [r for r in group if r["candidate"] == "oracle"]
        if oracle_rows:
            oracle_raw = oracle_rows[0]["rank_raw"]
            oracle_stab = oracle_rows[0]["rank_stab"]
            oracle_deltas.append(int(oracle_raw) - int(oracle_stab))
        else:
            oracle_raw = oracle_stab = ""
        matrix, block, slot = key
        lines.append(
            f"| {matrix} | {block} | {slot} | {raw_w['candidate']} | "
            f"{stab_w['candidate']} | {oracle_raw} | {oracle_stab} |"
        )

    if oracle_deltas:
        improved = sum(1 for x in oracle_deltas if x > 0)
        worsened = sum(1 for x in oracle_deltas if x < 0)
        same = sum(1 for x in oracle_deltas if x == 0)
        mean_delta = float(np.mean(oracle_deltas))
    else:
        improved = worsened = same = 0
        mean_delta = float("nan")

    lines.extend([
        "",
        "## Aggregate oracle-rank movement",
        "",
        f"- improved: {improved}",
        f"- unchanged: {same}",
        f"- worsened: {worsened}",
        f"- mean(raw_rank - stab_rank): {mean_delta:.3f}",
        "",
        "Positive mean means the stability penalty improved the oracle panel rank.",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrices", nargs="+", default=["static-cex", "diffuse-diffuse"])
    p.add_argument("--blocks", nargs="+", type=int, default=[1, 6])
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--half-win", type=int, default=32)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--preset", default="fast")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rng-seed", type=int, default=20260429)
    p.add_argument("--frac", type=float, default=0.50)
    p.add_argument("--n-subsamples", type=int, default=12)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument(
        "--out-dir",
        default="summary/score_family_stability_weighted_evidence/variants/A0/diagnostics",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng_master = np.random.default_rng(args.rng_seed)
    t0 = time.time()
    rows = []
    for matrix in args.matrices:
        print(f"[FAM-09 A0] running {matrix} ...", flush=True)
        rows.extend(run_matrix(args, matrix, rng_master))
    add_ranks(rows)
    elapsed = time.time() - t0
    csv_path = os.path.join(args.out_dir, "prototype_A0.csv")
    md_path = os.path.join(args.out_dir, "prototype_A0_summary.md")
    write_csv(csv_path, rows)
    write_summary(md_path, rows, args, elapsed)
    print(f"[FAM-09 A0] wrote {csv_path}")
    print(f"[FAM-09 A0] wrote {md_path}")


if __name__ == "__main__":
    main()
