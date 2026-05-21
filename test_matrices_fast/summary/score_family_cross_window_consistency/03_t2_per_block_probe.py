"""FAM-07 T2 per-block cross-window consistency diagnostic.

Runs the T2 grid from 00_spec.md:

  matrices = mixed-tail-sharp, static-cex, diffuse-diffuse, etf-basket-basis
  blocks   = 1, 2, 12, 31

For each block it dumps vector and rank-r frame candidates with HM3 evidence,
rho_F / rho_frame, and the multiplicative FAM-07 score.

Outputs:
  04_t2_per_block.csv
  04_t2_per_block.json
  04_t2_synthesis.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TMF_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _TMF_DIR not in sys.path:
    sys.path.insert(0, _TMF_DIR)

import cex_restricted_space_probe as probe  # noqa: E402
from future_hmean_optimizer_diagnostic import orth_basis_against, rowspace_basis  # noqa: E402
from hmean_combinations_optimizer_diagnostic import candidate_denoms, optimize_combination_in_basis  # noqa: E402
from hmean_evidence_score import (  # noqa: E402
    hm_evi_value_grad,
    per_block_constants,
    stream_to_block,
)
import half_window_sliding_hmean_experiment as hm  # noqa: E402
from r_sk_g_score import _state_V, optimize_r_sk_g_in_basis  # noqa: E402
from row_cheat_baseline import (  # noqa: E402
    frame_score_S6,
    oracle_frame_proj,
    top_r_rows_frame,
)
from second_slot_tail_bias_diagnostic import raw_oracle_columns  # noqa: E402


EPS = 1e-30


def default_stream_args(matrix: str) -> SimpleNamespace:
    return SimpleNamespace(
        matrix=matrix,
        n=1024,
        half_win=32,
        rank=2,
        preset="fast",
        seed=0,
        shuffle_rows=True,
        row_shuffle_seed=0,
        old_memory_size=32,
        dtype="float32",
        q0=8,
        qmax=48,
        krylov_depth=2,
        residual_tol=0.01,
        expansion_maxit=8,
        num_restarts=3,
        maxit=120,
        tol=1e-8,
        post_expansion_maxit=80,
        patience=5,
        patience_rel_tol=1e-5,
        union_maxit=120,
        union_tol=1e-9,
        union_random_starts=24,
        r_sig=2,
        alpha_sig=0.003,
        alpha_tail=0.0145,
        tail_scale=0.99,
        sigma1=0.991,
        v_type="rand",
    )


def unit(v):
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    nv = float(np.linalg.norm(v))
    return None if nv <= EPS else v / nv


def project_unit(v, B):
    if v is None or B is None or B.size == 0:
        return None
    p = B @ (B.T @ v)
    return unit(p)


def orth_frame(cols):
    cols = [unit(c) for c in cols if c is not None]
    cols = [c for c in cols if c is not None]
    if not cols:
        return None
    Q, R = np.linalg.qr(np.column_stack(cols))
    d = np.abs(np.diag(R))
    if d.size == 0:
        return None
    keep = np.where(d > max(float(d.max()) * 1e-12, EPS))[0]
    return None if keep.size == 0 else np.ascontiguousarray(Q[:, keep])


def rho_F(A_cur, A_fut, v):
    v = unit(v)
    if v is None:
        return float("nan")
    p = A_cur.T @ (A_cur @ v)
    q = A_fut.T @ (A_fut @ v)
    np_ = float(np.linalg.norm(p))
    nq_ = float(np.linalg.norm(q))
    if np_ <= EPS or nq_ <= EPS:
        return float("nan")
    return float(np.dot(p, q) / (np_ * nq_))


def rho_frame(A_cur, A_fut, V):
    if V is None:
        return float("nan")
    V = np.asarray(V, dtype=np.float64)
    Gc = A_cur.T @ (A_cur @ V)
    Gf = A_fut.T @ (A_fut @ V)
    K = Gc.T @ Gf
    n = float(np.sum(K * K))
    dc = float(np.sum(Gc * Gc))
    df = float(np.sum(Gf * Gf))
    if dc <= EPS or df <= EPS:
        return float("nan")
    return float(n / (dc * df))


def s6_vector_score(A_sketch, A_cur, A_fut, sk_F2_low, cur_F2, fut_F2, v):
    v = unit(v)
    if v is None:
        return None
    raw_g1 = float(np.dot(A_cur @ v, A_cur @ v))
    raw_g2 = float(np.dot(A_fut @ v, A_fut @ v))
    u_g1 = raw_g1 / max(float(cur_F2), EPS)
    u_g2 = raw_g2 / max(float(fut_F2), EPS)
    if A_sketch is not None and float(sk_F2_low) > EPS:
        raw_sk = float(np.dot(A_sketch @ v, A_sketch @ v))
        u_sk = raw_sk / float(sk_F2_low)
        score = 0.0 if min(u_sk, u_g1, u_g2) <= EPS else 3.0 / (1.0 / u_sk + 1.0 / u_g1 + 1.0 / u_g2)
    else:
        u_sk = float("nan")
        score = 0.0 if min(u_g1, u_g2) <= EPS else 2.0 / (1.0 / u_g1 + 1.0 / u_g2)
    return {"u_sk": float(u_sk), "u_g1": float(u_g1), "u_g2": float(u_g2), "score": float(score)}


def collect_vector_candidates(args, snap, A, V_exact, B_union, A_sketch_for, sk_F2_low, cur_F2, fut_F2, c_sk):
    A_cur = snap["A_cur"]
    A_fut = snap["A_fut"]
    V_default = snap["V_default"]
    state = snap["state"]
    V_state = _state_V(state)

    oracle_v1 = unit(V_exact[:, 0])
    oracle_v2 = unit(V_exact[:, 1])
    oracle_v1_proj = project_unit(oracle_v1, B_union)
    oracle_v2_proj = project_unit(oracle_v2, B_union)

    Q_oracle, raw_oracle = raw_oracle_columns(snap["M_gain"], V_exact, int(args.rank), np.float64)
    pool = hm.build_candidates(V_default, Q_oracle, raw_oracle, snap["M_gain"], A_cur, A_fut)
    pool = {k: pool.get(k) for k in hm.ONLINE_POOL}
    denoms, _ = candidate_denoms(pool, A_cur, A_fut, A_sketch_for)
    B_search = orth_basis_against(B_union, V_default[:, 0])
    starts = [V_default[:, 1], oracle_v1_proj, oracle_v2_proj]
    starts.extend([v for v in pool.values() if v is not None])
    Vbasis = snap["diag"].get("Vbasis_final")
    if Vbasis is not None:
        Vb = np.asarray(Vbasis, dtype=np.float64)
        for j in range(min(Vb.shape[1], 8)):
            starts.append(Vb[:, j])

    weights_existing = (state["rows_seen"] if state is not None else 0, A_cur.shape[0], A_fut.shape[0])
    hm_triplet_evidence = optimize_combination_in_basis(
        "future_hmean_triplet_online",
        A_cur,
        A_fut,
        A_sketch_for,
        denoms,
        weights_existing,
        B_search,
        starts,
        np.random.default_rng(args.seed + 31337 + snap["block_id"]),
        args.union_maxit,
        args.union_tol,
        args.union_random_starts,
    )

    rsk_S6_v1 = optimize_r_sk_g_in_basis(
        A_cur,
        A_fut,
        A_sketch_for,
        c_sk,
        B_union,
        starts,
        np.random.default_rng(args.seed + 57000 + snap["block_id"]),
        args.union_maxit,
        args.union_tol,
        args.union_random_starts,
        variant="S6",
        V_state=V_state,
        cur_F2=cur_F2,
        fut_F2=fut_F2,
        sk_F2_low=sk_F2_low,
    )
    v1_S6 = None if rsk_S6_v1 is None else rsk_S6_v1["vec"]
    if v1_S6 is not None:
        B_def = orth_basis_against(B_union, v1_S6)
        rsk_S6_v2 = optimize_r_sk_g_in_basis(
            A_cur,
            A_fut,
            A_sketch_for,
            c_sk,
            B_def,
            starts,
            np.random.default_rng(args.seed + 58000 + snap["block_id"]),
            args.union_maxit,
            args.union_tol,
            args.union_random_starts,
            variant="S6",
            V_state=V_state,
            cur_F2=cur_F2,
            fut_F2=fut_F2,
            sk_F2_low=sk_F2_low,
        )
        v2_S6 = None if rsk_S6_v2 is None else rsk_S6_v2["vec"]
    else:
        v2_S6 = None

    return {
        "combined_optimizer_v2": (V_default[:, 1], "v2"),
        "hm_triplet_evidence_best": (None if hm_triplet_evidence is None else hm_triplet_evidence["vec"], "v2"),
        "oracle_v1_proj_S+G1+G2": (oracle_v1_proj, "v1"),
        "oracle_v2_proj_S+G1+G2": (oracle_v2_proj, "v2"),
        "frame_S6_greedy_v1": (v1_S6, "v1"),
        "frame_S6_greedy_v2": (v2_S6, "v2"),
    }, oracle_v1, oracle_v2, oracle_v1_proj, oracle_v2_proj, v1_S6, v2_S6


def analyze_block(args, matrix, A, V_exact, snap):
    block_id = snap["block_id"]
    A_cur = snap["A_cur"]
    A_fut = snap["A_fut"]
    A_sketch = snap["A_sketch"]
    A_sketch_for = A_sketch if A_sketch.size else None
    consts = per_block_constants(A, block_id, args.half_win)
    c_sk = float(consts["c_sk"])
    c_g1 = float(consts["c_g1"])
    c_g2 = float(consts["c_g2"])
    cur_F2 = float(consts["cur_F2"])
    fut_F2 = float(consts["fut_F2"])
    sk_F2_low = float(np.sum(A_sketch * A_sketch)) if A_sketch_for is not None else 0.0

    union_stack = np.vstack([A_sketch, A_cur, A_fut]) if A_sketch_for is not None else np.vstack([A_cur, A_fut])
    B_union = rowspace_basis(union_stack)
    candidates, oracle_v1, oracle_v2, oracle_v1_proj, oracle_v2_proj, v1_S6, v2_S6 = collect_vector_candidates(
        args, snap, A, V_exact, B_union, A_sketch_for, sk_F2_low, cur_F2, fut_F2, c_sk
    )

    # Fixed-weight HM_evi for the requested HM_evi diagnostic column.
    w_sk = float(args.rank)
    w_g1 = float(args.half_win)
    w_g2 = float(args.half_win)

    rows = []
    oracle_metrics = {}
    for label, (v, slot) in candidates.items():
        v = unit(v)
        if v is None:
            continue
        s6 = s6_vector_score(A_sketch_for, A_cur, A_fut, sk_F2_low, cur_F2, fut_F2, v)
        hm_score, _, _, _, _, hm_evi, relH1 = hm_evi_value_grad(
            A_sketch_for, A_cur, A_fut, c_sk, c_g1, c_g2, w_sk, w_g1, w_g2, v
        )
        rho = rho_F(A_cur, A_fut, v)
        row = {
            "matrix": matrix,
            "block": block_id,
            "candidate_type": "vector",
            "label": label,
            "slot_ref": slot,
            "u_sk": s6["u_sk"],
            "u_g1": s6["u_g1"],
            "u_g2": s6["u_g2"],
            "HM_evi": float(hm_evi),
            "relH1": float(relH1),
            "rho": rho,
            "score_HM3": s6["score"],
            "score_FAM07": s6["score"] * (rho * rho if rho == rho else float("nan")),
            "hm_evi_score_with_relH1": float(hm_score),
            "align_v1": float(np.dot(v, oracle_v1) ** 2),
            "align_v2": float(np.dot(v, oracle_v2) ** 2),
        }
        rows.append(row)
        if label == "oracle_v1_proj_S+G1+G2":
            oracle_metrics["v1"] = row
        elif label == "oracle_v2_proj_S+G1+G2":
            oracle_metrics["v2"] = row

    for row in rows:
        ref = oracle_metrics.get(row["slot_ref"])
        if ref is None or row["label"].startswith("oracle_"):
            row["t2_signature"] = False
            continue
        high = row["u_g1"] >= 0.5 * ref["u_g1"] and row["u_g2"] >= 0.5 * ref["u_g2"]
        low_rho = row["rho"] == row["rho"] and row["rho"] < 0.5
        row["t2_signature"] = bool(high and low_rho)

    frame_oracle = oracle_frame_proj(V_exact, B_union, int(args.rank))
    frame_s6 = orth_frame([v1_S6, v2_S6])
    frame_rowcheat = top_r_rows_frame(A_fut, int(args.rank))
    frame_candidates = {
        "frame_oracle_proj": frame_oracle,
        "frame_S6_greedy": frame_s6,
        "frame_rowcheat": frame_rowcheat,
    }
    frame_ref = frame_score_S6(A_sketch_for, A_cur, A_fut, sk_F2_low, cur_F2, fut_F2, frame_oracle) if frame_oracle is not None else None
    for label, V in frame_candidates.items():
        if V is None:
            continue
        s6 = frame_score_S6(A_sketch_for, A_cur, A_fut, sk_F2_low, cur_F2, fut_F2, V)
        rho = rho_frame(A_cur, A_fut, V)
        row = {
            "matrix": matrix,
            "block": block_id,
            "candidate_type": "frame",
            "label": label,
            "slot_ref": "frame",
            "u_sk": s6["u_sk"],
            "u_g1": s6["u_g1"],
            "u_g2": s6["u_g2"],
            "HM_evi": s6["score"],
            "relH1": float("nan"),
            "rho": rho,
            "score_HM3": s6["score"],
            "score_FAM07": s6["score"] * (rho if rho == rho else float("nan")),
            "hm_evi_score_with_relH1": float("nan"),
            "align_v1": float(np.sum((V.T @ oracle_v1) ** 2)),
            "align_v2": float(np.sum((V.T @ oracle_v2) ** 2)),
        }
        if label == "frame_oracle_proj" or frame_ref is None:
            row["t2_signature"] = False
        else:
            high = row["u_g1"] >= 0.5 * frame_ref["u_g1"] and row["u_g2"] >= 0.5 * frame_ref["u_g2"]
            low_rho = row["rho"] == row["rho"] and row["rho"] < 0.5
            row["t2_signature"] = bool(high and low_rho)
        rows.append(row)
    return rows


def run(args):
    all_rows = []
    for matrix in args.matrices:
        stream_args = default_stream_args(matrix)
        for name in vars(args):
            if hasattr(stream_args, name):
                setattr(stream_args, name, getattr(args, name))
        A, V_exact, _, _ = probe.generate_matrix_input(
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
        work_dtype = np.float32 if stream_args.dtype == "float32" else np.float64
        snaps = stream_to_block(
            stream_args,
            A,
            V_exact,
            work_dtype,
            int(stream_args.rank),
            max(args.blocks),
            set(args.blocks),
        )
        for block in args.blocks:
            snap = snaps[block]
            snap["block_id"] = block
            all_rows.extend(analyze_block(stream_args, matrix, A, V_exact, snap))
    return all_rows


def finite_for_json(x):
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if math.isnan(x) or math.isinf(x) else x
    return x


def write_outputs(rows, out_dir):
    csv_path = os.path.join(out_dir, "04_t2_per_block.csv")
    json_path = os.path.join(out_dir, "04_t2_per_block.json")
    md_path = os.path.join(out_dir, "04_t2_synthesis.md")
    fields = [
        "matrix", "block", "candidate_type", "label", "slot_ref",
        "u_sk", "u_g1", "u_g2", "HM_evi", "relH1", "rho",
        "score_HM3", "score_FAM07", "hm_evi_score_with_relH1",
        "align_v1", "align_v2", "t2_signature",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([{k: finite_for_json(v) for k, v in row.items()} for row in rows], f, indent=2, sort_keys=True)

    signature_rows = [r for r in rows if r.get("t2_signature")]
    vector_sig = [r for r in signature_rows if r["candidate_type"] == "vector"]
    frame_sig = [r for r in signature_rows if r["candidate_type"] == "frame"]
    cells = sorted({(r["matrix"], r["block"]) for r in signature_rows})
    by_matrix = {}
    for r in signature_rows:
        by_matrix.setdefault(r["matrix"], 0)
        by_matrix[r["matrix"]] += 1

    lines = [
        "# FAM-07 T2 per-block synthesis",
        "",
        "Command:",
        "",
        "```",
        "python summary/score_family_cross_window_consistency/03_t2_per_block_probe.py",
        "```",
        "",
        "Scope: matrices `mixed-tail-sharp`, `static-cex`, `diffuse-diffuse`, `etf-basket-basis`; blocks `1, 2, 12, 31`; half_win=32; rank=2; seed=0.",
        "",
        "Artifacts:",
        "",
        "- `04_t2_per_block.csv`",
        "- `04_t2_per_block.json`",
        "",
        f"Rows: {len(rows)} total ({sum(1 for r in rows if r['candidate_type'] == 'vector')} vector, {sum(1 for r in rows if r['candidate_type'] == 'frame')} frame).",
        f"T2 signature rows: {len(signature_rows)} total ({len(vector_sig)} vector, {len(frame_sig)} frame).",
        f"T2 signature cells: {len(cells)} of 16 matrix/block cells.",
        "",
        "Acceptance read:",
        "",
    ]
    if signature_rows:
        lines.append("PASS for T2/K2: cross-window consistency adds diagnostic signal beyond HM3 on at least one real high-entropy cell.")
    else:
        lines.append("FAIL for T2/K2: no high-HM, low-rho signature fired on the probe grid.")
    lines.extend(["", "Signature counts by matrix:", ""])
    if by_matrix:
        for matrix, count in sorted(by_matrix.items()):
            lines.append(f"- `{matrix}`: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "Strongest signature rows (lowest rho first):", ""])
    for r in sorted(signature_rows, key=lambda x: x["rho"])[:12]:
        lines.append(
            f"- `{r['matrix']}` b{r['block']} `{r['label']}` ({r['candidate_type']}): "
            f"rho={r['rho']:.4f}, u_g1={r['u_g1']:.4e}, u_g2={r['u_g2']:.4e}, "
            f"score_HM3={r['score_HM3']:.4e}, score_FAM07={r['score_FAM07']:.4e}"
        )
    lines.extend(["", "Notes:", ""])
    lines.append("- Vector `score_FAM07` is `score_HM3 * rho_F^2`; frame `score_FAM07` is `score_HM3 * rho_frame`.")
    lines.append("- The signature test excludes oracle rows and compares v1/v2 candidates against the matching projected oracle vector; frame rows compare against `frame_oracle_proj`.")
    lines.append("- This is T2 diagnostic evidence only; it does not wire FAM-07 into the streaming score path.")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return csv_path, json_path, md_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrices", nargs="+", default=["mixed-tail-sharp", "static-cex", "diffuse-diffuse", "etf-basket-basis"])
    p.add_argument("--blocks", nargs="+", type=int, default=[1, 2, 12, 31])
    p.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--half-win", type=int, default=32)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--preset", default="fast")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-rows", action="store_true", default=True)
    p.add_argument("--row-shuffle-seed", type=int, default=0)
    p.add_argument("--old-memory-size", type=int, default=32)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--q0", type=int, default=8)
    p.add_argument("--qmax", type=int, default=48)
    p.add_argument("--krylov-depth", type=int, default=2)
    p.add_argument("--residual-tol", type=float, default=0.01)
    p.add_argument("--expansion-maxit", type=int, default=8)
    p.add_argument("--num-restarts", type=int, default=3)
    p.add_argument("--maxit", type=int, default=120)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--post-expansion-maxit", type=int, default=80)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--patience-rel-tol", type=float, default=1e-5)
    p.add_argument("--union-maxit", type=int, default=120)
    p.add_argument("--union-tol", type=float, default=1e-9)
    p.add_argument("--union-random-starts", type=int, default=24)
    p.add_argument("--r-sig", type=int, default=2)
    p.add_argument("--alpha-sig", type=float, default=0.003)
    p.add_argument("--alpha-tail", type=float, default=0.0145)
    p.add_argument("--tail-scale", type=float, default=0.99)
    p.add_argument("--sigma1", type=float, default=0.991)
    p.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = run(args)
    paths = write_outputs(rows, args.out_dir)
    sig = sum(1 for r in rows if r.get("t2_signature"))
    print(f"FAM-07 T2 rows={len(rows)} signature_rows={sig}")
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
