"""Oracle entropy / regime-label audit (DIAG-01).

Reports per matrix/block/slot response entropy for the required candidate
families:

  - oracle_exact: V_exact[:, k]
  - S6 opt: sequential S6 optimizer slot k
  - mgain_svd: iSVD/M_gain candidate slot k
  - combined: combined-score carried candidate slot k
  - rowcheat: top rows of A_fut frame slot k

For each candidate we compute Shannon entropy of row-response energy
``p_i = (A_window v)_i^2 / sum_j (A_window v)_j^2`` on current, future,
visible=[current;future], and full-A windows. The normalized entropy is
``relH1 = H / log(m)`` and the effective support is ``exp(H)`` rows
(``eff_frac = exp(H) / m``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from statistics import median

import numpy as np

import cex_restricted_space_probe as probe
from future_hmean_optimizer_diagnostic import orth_basis_against, rowspace_basis
from hmean_evidence_score import per_block_constants, stream_to_block
from r_sk_g_score import _state_V, optimize_r_sk_g_in_basis
from row_cheat_baseline import top_r_rows_frame


DEFAULT_MATRICES = [
    "static-cex",
    "mixed-tail-sharp",
    "mixed-tail-balanced",
    "mixed-tail-soft",
    "diffuse-diffuse",
    "etf-basket-basis",
    "residual-spiky-shocks",
    "risk-residual-panel",
]

STORY_LABELS = {
    "static-cex": "HIGH",
    "mixed-tail-sharp": "HIGH",
    "mixed-tail-balanced": "HIGH",
    "mixed-tail-soft": "HIGH",
    "diffuse-diffuse": "HIGH",
    "etf-basket-basis": "HIGH",
    "residual-spiky-shocks": "LOW",
    "risk-residual-panel": "LOW",
}


def entropy_stats(A_window, v):
    A = np.asarray(A_window, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    m = int(A.shape[0]) if A.ndim == 2 else 0
    if m <= 0:
        return {
            "rows": m,
            "relH1": float("nan"),
            "eff_support": float("nan"),
            "eff_frac": float("nan"),
            "top1_share": float("nan"),
            "energy": 0.0,
        }
    y = A @ v
    e = y * y
    total = float(np.sum(e))
    if total <= 1e-300:
        return {
            "rows": m,
            "relH1": 0.0,
            "eff_support": 0.0,
            "eff_frac": 0.0,
            "top1_share": 0.0,
            "energy": 0.0,
        }
    p = e / total
    p_pos = np.maximum(p, 1e-300)
    H = -float(np.sum(p * np.log(p_pos)))
    eff = float(math.exp(H))
    return {
        "rows": m,
        "relH1": float(H / math.log(max(m, 2))),
        "eff_support": eff,
        "eff_frac": float(eff / m),
        "top1_share": float(np.max(p)),
        "energy": total,
    }


def project_unit(vec, B):
    if vec is None or B is None or B.size == 0:
        return None
    p = B @ (B.T @ vec)
    nrm = float(np.linalg.norm(p))
    return None if nrm <= 1e-30 else p / nrm


def candidate_vectors(args, A, V_exact, snap, block_id):
    rank = int(args.rank)
    A_cur = snap["A_cur"]
    A_fut = snap["A_fut"]
    A_sketch = snap["A_sketch"]
    state = snap["state"]
    V_default = snap["V_default"]

    consts = per_block_constants(A, block_id, int(args.half_win))
    c_sk = float(consts["c_sk"])
    cur_F2 = float(consts["cur_F2"])
    fut_F2 = float(consts["fut_F2"])
    A_sketch_for = A_sketch if A_sketch.size else None
    V_state = _state_V(state)
    sk_F2_low = float(np.sum(np.asarray(A_sketch_for, dtype=np.float64) ** 2)) if A_sketch_for is not None else 0.0

    if A_sketch_for is not None:
        union_stack = np.vstack([A_sketch, A_cur, A_fut])
    else:
        union_stack = np.vstack([A_cur, A_fut])
    B_union = rowspace_basis(union_stack)

    starts = [V_default[:, j] for j in range(min(rank, V_default.shape[1]))]
    if V_state is not None:
        starts.extend(V_state[:, j] for j in range(min(rank, V_state.shape[1])))
    if B_union is not None and B_union.size:
        for j in range(min(rank, V_exact.shape[1])):
            p = project_unit(V_exact[:, j], B_union)
            if p is not None:
                starts.append(p)

    s6_slots = []
    B_slot = B_union
    for slot in range(rank):
        result = optimize_r_sk_g_in_basis(
            A_cur,
            A_fut,
            A_sketch_for,
            c_sk,
            B_slot,
            starts,
            np.random.default_rng(args.seed + 70000 + 97 * block_id + slot),
            args.union_maxit,
            args.union_tol,
            args.union_random_starts,
            variant="S6",
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            V_state=V_state,
            cur_F2=cur_F2,
            fut_F2=fut_F2,
            sk_F2_low=sk_F2_low,
        )
        v = None if result is None else result["vec"]
        s6_slots.append(v)
        if v is None:
            break
        B_slot = orth_basis_against(B_slot, v)

    M_gain = np.asarray(snap["M_gain"], dtype=np.float64)
    if M_gain.size:
        _, _, Vt_mgain = np.linalg.svd(M_gain, full_matrices=False)
    else:
        Vt_mgain = np.zeros((0, A.shape[1]), dtype=np.float64)

    V_rowcheat = top_r_rows_frame(A_fut, rank)

    out = {}
    for slot in range(rank):
        out[("oracle_exact", slot + 1)] = V_exact[:, slot] if slot < V_exact.shape[1] else None
        out[("S6_opt", slot + 1)] = s6_slots[slot] if slot < len(s6_slots) else None
        out[("mgain_svd", slot + 1)] = Vt_mgain[slot] if slot < Vt_mgain.shape[0] else None
        out[("combined", slot + 1)] = V_default[:, slot] if slot < V_default.shape[1] else None
        out[("rowcheat", slot + 1)] = V_rowcheat[:, slot] if V_rowcheat is not None and slot < V_rowcheat.shape[1] else None
    return out


def audit_matrix(args, matrix):
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=matrix,
        n=args.n,
        preset=args.preset,
        seed=args.seed,
        r_sig=args.r_sig,
        alpha_sig=args.alpha_sig,
        alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale,
        sigma1=args.sigma1,
        v_type=args.v_type,
        shuffle_rows=args.shuffle_rows,
        row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, dtype=np.float64)
    V_exact = np.asarray(V_exact, dtype=np.float64)
    blocks = [b for b in args.blocks if (b + 1) * args.half_win <= A.shape[0]]
    if not blocks:
        return []
    snapshots = stream_to_block(args, A, V_exact, np.float32 if args.dtype == "float32" else np.float64, int(args.rank), max(blocks), set(blocks))

    rows = []
    for block_id in blocks:
        snap = snapshots[block_id]
        A_cur = snap["A_cur"]
        A_fut = snap["A_fut"]
        windows = {
            "cur": A_cur,
            "fut": A_fut,
            "visible": np.vstack([A_cur, A_fut]),
            "full": A,
        }
        candidates = candidate_vectors(args, A, V_exact, snap, block_id)
        for (candidate, slot), v in candidates.items():
            if v is None:
                continue
            nrm = float(np.linalg.norm(v))
            if nrm <= 1e-30:
                continue
            v = v / nrm
            for window, Aw in windows.items():
                s = entropy_stats(Aw, v)
                rows.append(
                    {
                        "matrix": matrix,
                        "story_label": STORY_LABELS.get(matrix, "UNKNOWN"),
                        "block": block_id,
                        "slot": slot,
                        "candidate": candidate,
                        "window": window,
                        **s,
                    }
                )
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for r in rows:
        if r["candidate"] == "oracle_exact" and r["window"] == "visible":
            grouped[(r["matrix"], r["slot"])].append(float(r["eff_frac"]))

    matrix_rows = []
    by_matrix = defaultdict(dict)
    for (matrix, slot), vals in grouped.items():
        if vals:
            by_matrix[matrix][slot] = {
                "median": median(vals),
                "min": min(vals),
                "max": max(vals),
            }
    for matrix in sorted(by_matrix):
        s1 = by_matrix[matrix].get(1, {})
        s2 = by_matrix[matrix].get(2, {})
        m1 = s1.get("median", float("nan"))
        m2 = s2.get("median", float("nan"))
        min2 = s2.get("min", float("nan"))
        measured = classify_matrix(m1, m2, min2)
        matrix_rows.append(
            {
                "matrix": matrix,
                "story_label": STORY_LABELS.get(matrix, "UNKNOWN"),
                "oracle_slot1_visible_eff_frac_median": m1,
                "oracle_slot2_visible_eff_frac_median": m2,
                "oracle_slot2_visible_eff_frac_min": min2,
                "measured_regime": measured,
            }
        )
    return matrix_rows


def classify_matrix(slot1_median, slot2_median, slot2_min):
    vals = [x for x in (slot1_median, slot2_median) if np.isfinite(x)]
    if not vals:
        return "UNKNOWN"
    # These thresholds are intentionally descriptive, not theorem status.
    # HIGH means both top oracle slots spread over at least half of visible rows
    # at the median probed block. LOW means slot-2's median support is under a
    # quarter of visible rows. Everything in between is boundary/mixed.
    if slot2_median < 0.25:
        return "LOW"
    if min(vals) >= 0.50 and (not np.isfinite(slot2_min) or slot2_min >= 0.25):
        return "HIGH"
    return "BOUNDARY"


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def fmt(x):
    if not np.isfinite(float(x)):
        return "nan"
    return f"{float(x):.3f}"


def write_synthesis(path, rows, matrix_summary):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    boundary = [r for r in matrix_summary if r["measured_regime"] != r["story_label"]]

    # Candidate contrast: median visible effective support by candidate/slot.
    cand = defaultdict(list)
    for r in rows:
        if r["window"] == "visible":
            cand[(r["matrix"], r["slot"], r["candidate"])].append(float(r["eff_frac"]))

    with open(path, "w") as f:
        f.write("# DIAG-01 Oracle Entropy and Regime-Label Audit\n\n")
        f.write("Date: 2026-04-28\n\n")
        f.write("Verdict: **ship**. The diagnostic is implemented and the regime labels are now measured. The revised labels are heuristic measurements, not theorem status for S6/HM3/relH1.\n\n")
        f.write("## Formula\n\n")
        f.write("For a window `W` and unit vector `v`, row-response energy is `e_i = (Wv)_i^2`, `p_i = e_i / sum_j e_j`, `H = -sum_i p_i log(p_i)`, `relH1 = H / log(m)`, and `effective_support = exp(H)` rows. The table below uses `eff_frac = effective_support / m` on `visible = [A_cur; A_fut]` for `V_exact[:,k]`.\n\n")
        f.write("## Revised Regime Table\n\n")
        f.write("| matrix | previous | measured | oracle slot1 median eff_frac | oracle slot2 median eff_frac | oracle slot2 min eff_frac | note |\n")
        f.write("| --- | --- | --- | ---: | ---: | ---: | --- |\n")
        for r in matrix_summary:
            note = ""
            if r["measured_regime"] != r["story_label"]:
                note = "boundary/correction: previous label not supported by oracle slot entropy"
            f.write(
                f"| {r['matrix']} | {r['story_label']} | {r['measured_regime']} | "
                f"{fmt(r['oracle_slot1_visible_eff_frac_median'])} | "
                f"{fmt(r['oracle_slot2_visible_eff_frac_median'])} | "
                f"{fmt(r['oracle_slot2_visible_eff_frac_min'])} | {note} |\n"
            )
        f.write("\nClassification rule used for this audit: HIGH if both oracle slots have median visible `eff_frac >= 0.50` and slot-2 never drops below `0.25` on the probed blocks; LOW if oracle slot-2 median visible `eff_frac < 0.25`; otherwise BOUNDARY. This is a measurement convention for regime labels, not a model theorem.\n\n")

        f.write("## Boundary Cases\n\n")
        if boundary:
            for r in boundary:
                f.write(
                    f"- {r['matrix']}: previous {r['story_label']} -> measured {r['measured_regime']}; "
                    f"oracle slot-2 median visible `eff_frac={fmt(r['oracle_slot2_visible_eff_frac_median'])}` "
                    f"(min {fmt(r['oracle_slot2_visible_eff_frac_min'])}).\n"
                )
        else:
            f.write("- None under the audit thresholds.\n")
        f.write("\n## Candidate Evidence\n\n")
        f.write("Median visible `eff_frac` by matrix, slot, and candidate across probed blocks:\n\n")
        f.write("| matrix | slot | oracle | S6 opt | mgain/iSVD | combined | rowcheat |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for matrix in sorted({r["matrix"] for r in rows}):
            for slot in (1, 2):
                vals = {}
                for candidate in ("oracle_exact", "S6_opt", "mgain_svd", "combined", "rowcheat"):
                    xs = cand.get((matrix, slot, candidate), [])
                    vals[candidate] = median(xs) if xs else float("nan")
                f.write(
                    f"| {matrix} | {slot} | {fmt(vals['oracle_exact'])} | "
                    f"{fmt(vals['S6_opt'])} | {fmt(vals['mgain_svd'])} | "
                    f"{fmt(vals['combined'])} | {fmt(vals['rowcheat'])} |\n"
                )
        f.write("\n## Outputs\n\n")
        f.write("- `summary/infra_oracle_entropy_audit/audit.csv`: long-form per matrix/block/slot/candidate/window table.\n")
        f.write("- `summary/infra_oracle_entropy_audit/audit.json`: same records as JSON.\n")
        f.write("- `summary/infra_oracle_entropy_audit/regime_summary.csv`: revised regime table inputs.\n\n")
        f.write("## Assumptions\n\n")
        f.write("- Blocks are `1,2,6,12,31` where feasible with `half_win=32`, `rank=2`, `n=1024`, preset `fast`, seed `0`.\n")
        f.write("- Snapshot/optimizer settings are the diagnostic defaults in `oracle_entropy_audit.py`: `q0=4`, `qmax=16`, `num_restarts=1`, `maxit=40`, `post_expansion_maxit=30`, `union_maxit=40`, and `union_random_starts=4`.\n")
        f.write("- Candidate entropy is measured on row responses, not on vector coefficients.\n")
        f.write("- The diagnostic includes exact oracle vectors for classification; this is an audit, not an operational score.\n")
        f.write("- S6/HM3/relH1 remain heuristic until TH-01/TH-02/TH-03 establish stronger status.\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrices", nargs="*", default=DEFAULT_MATRICES)
    p.add_argument("--blocks", nargs="+", type=int, default=[1, 2, 6, 12, 31])
    p.add_argument("--out-dir", default="summary/infra_oracle_entropy_audit")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--half-win", type=int, default=32)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--preset", default="fast")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-rows", action="store_true", default=True)
    p.add_argument("--row-shuffle-seed", type=int, default=0)
    p.add_argument("--old-memory-size", type=int, default=32)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--q0", type=int, default=4)
    p.add_argument("--qmax", type=int, default=16)
    p.add_argument("--krylov-depth", type=int, default=2)
    p.add_argument("--residual-tol", type=float, default=0.01)
    p.add_argument("--expansion-maxit", type=int, default=4)
    p.add_argument("--num-restarts", type=int, default=1)
    p.add_argument("--maxit", type=int, default=40)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--post-expansion-maxit", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--patience-rel-tol", type=float, default=1e-5)
    p.add_argument("--union-maxit", type=int, default=40)
    p.add_argument("--union-tol", type=float, default=1e-9)
    p.add_argument("--union-random-starts", type=int, default=4)
    p.add_argument("--r-sig", type=int, default=2)
    p.add_argument("--alpha-sig", type=float, default=0.003)
    p.add_argument("--alpha-tail", type=float, default=0.0145)
    p.add_argument("--tail-scale", type=float, default=0.99)
    p.add_argument("--sigma1", type=float, default=0.991)
    p.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    for matrix in args.matrices:
        print(f"matrix={matrix}", flush=True)
        rows.extend(audit_matrix(args, matrix))

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "audit.csv")
    json_path = os.path.join(args.out_dir, "audit.json")
    summary_csv_path = os.path.join(args.out_dir, "regime_summary.csv")
    synthesis_path = os.path.join(args.out_dir, "synthesis.md")

    write_csv(csv_path, rows)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    matrix_summary = summarize(rows)
    write_csv(summary_csv_path, matrix_summary)
    write_synthesis(synthesis_path, rows, matrix_summary)
    print(f"wrote {csv_path} {json_path} {summary_csv_path} {synthesis_path}")


if __name__ == "__main__":
    main()
