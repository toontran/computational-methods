"""F-weighted HM3 score: ranking and component diagnostic.

Defines a new HM-style score:

  u_sk(v) = raw_sk(v) / ||A_sketch||_F^2          (rank-r carried sketch)
  u_g1(v) = raw_g1(v) / ||A_cur||_F^2
  u_g2(v) = raw_g2(v) / ||A_fut||_F^2

  score(v) = HM3(u_sk, u_g1, u_g2)        when sketch exists (block ≥ 2)
  score(v) = HM2(u_g1, u_g2)              when no sketch yet (block 1)

For each (matrix, block), this prints — sorted by new score, descending — a
table of candidates with the components shown in the form the user asked for:
  F2/raw_sk  (= ||A_sketch||_F^2 / raw_sk)
  F2/raw_g1  (= ||A_cur||_F^2    / raw_g1)
  F2/raw_g2  (= ||A_fut||_F^2    / raw_g2)
plus the new HM-score itself and a rank.

Candidates per block:
  combined_v1, combined_v2   (V_default[:, 0..1] from the carry-step optimizer)
  c_evi_v1, c_evi_v2          (c-weighted HM-evi optimizer in B_union; v2 deflated)
  oracle_v1_proj, oracle_v2_proj
"""

import argparse

import numpy as np

import cex_restricted_space_probe as probe
from future_hmean_optimizer_diagnostic import orth_basis_against, rowspace_basis
from hmean_evidence_score import (
    optimize_hm_evi_in_basis,
    per_block_constants,
    stream_to_block,
)
from row_cheat_baseline import (
    frame_score_S6 as _frame_score_S6,
    oracle_frame_proj as _oracle_frame_proj,
    top_r_rows_frame as _top_r_rows_frame,
)


def _unit(v):
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    nv = float(np.linalg.norm(v))
    if nv <= 1e-30:
        return None
    return v / nv


def _project(vec, B):
    if vec is None or B is None or B.size == 0:
        return None
    p = B @ (B.T @ vec)
    nv = float(np.linalg.norm(p))
    return None if nv <= 1e-30 else p / nv


def _ray_sq(A, v):
    if A is None or v is None or A.size == 0:
        return float("nan")
    y = A @ v
    return float(np.dot(y, y))


def f_hm_score(A_sketch, A_cur, A_fut, sk_F2_low, cur_F2, fut_F2, v):
    """Return (score, u_sk, u_g1, u_g2). Score = HM3 if sketch present else HM2."""
    eps = 1e-30
    u_g1 = _ray_sq(A_cur, v) / max(cur_F2, eps)
    u_g2 = _ray_sq(A_fut, v) / max(fut_F2, eps)
    if A_sketch is None or sk_F2_low <= eps:
        u_sk = float("nan")
        if u_g1 <= eps or u_g2 <= eps:
            return 0.0, u_sk, u_g1, u_g2
        score = 2.0 / (1.0 / u_g1 + 1.0 / u_g2)
    else:
        u_sk = _ray_sq(A_sketch, v) / sk_F2_low
        if u_sk <= eps or u_g1 <= eps or u_g2 <= eps:
            return 0.0, u_sk, u_g1, u_g2
        score = 3.0 / (1.0 / u_sk + 1.0 / u_g1 + 1.0 / u_g2)
    return score, u_sk, u_g1, u_g2


def collect_candidates(args, snap, V_exact, c_sk, c_g1, c_g2):
    A_cur = snap["A_cur"]
    A_fut = snap["A_fut"]
    A_sketch = snap["A_sketch"] if snap["A_sketch"].size else None
    V_default = snap["V_default"]

    if A_sketch is not None:
        union_stack = np.vstack([A_sketch, A_cur, A_fut])
    else:
        union_stack = np.vstack([A_cur, A_fut])
    B_union = rowspace_basis(union_stack)

    oracle_v1 = V_exact[:, 0]
    oracle_v2 = V_exact[:, 1]
    oracle_v1_proj = _project(oracle_v1, B_union)
    oracle_v2_proj = _project(oracle_v2, B_union)

    starts = []
    if V_default.shape[1] >= 2:
        starts.extend([V_default[:, 0], V_default[:, 1]])
    if oracle_v1_proj is not None:
        starts.extend([oracle_v1_proj, oracle_v2_proj])

    # c-weighted HM-evi: w_k = c_k^2 (so weighted HM3 = (Σc) / Σ(c_k/raw_k)).
    w_sk = float(c_sk * c_sk)
    w_g1 = float(c_g1 * c_g1)
    w_g2 = float(c_g2 * c_g2)

    c_v1 = optimize_hm_evi_in_basis(
        A_cur, A_fut, A_sketch, c_sk, c_g1, c_g2, w_sk, w_g1, w_g2,
        B_union, starts,
        np.random.default_rng(args.seed + 71001),
        args.union_maxit, args.union_tol, args.union_random_starts,
    )
    c_v1_vec = None if c_v1 is None else _unit(c_v1["vec"])
    if c_v1_vec is not None:
        B_def = orth_basis_against(B_union, c_v1_vec)
        c_v2 = optimize_hm_evi_in_basis(
            A_cur, A_fut, A_sketch, c_sk, c_g1, c_g2, w_sk, w_g1, w_g2,
            B_def, starts,
            np.random.default_rng(args.seed + 72002),
            args.union_maxit, args.union_tol, args.union_random_starts,
        )
        c_v2_vec = None if c_v2 is None else _unit(c_v2["vec"])
    else:
        c_v2_vec = None

    # Rank-r row-cheat baseline (INFRA-03): top-r rows of A_fut by squared
    # norm, orthonormalized via QR. Per-vector columns appear in the panel,
    # frame-level score is reported below in block_report.
    rank = int(args.rank)
    V_rowcheat = _top_r_rows_frame(A_fut, rank)
    rowcheat_v1 = V_rowcheat[:, 0] if V_rowcheat is not None and V_rowcheat.shape[1] >= 1 else None
    rowcheat_v2 = V_rowcheat[:, 1] if V_rowcheat is not None and V_rowcheat.shape[1] >= 2 else None

    cands = {
        "combined_v1": _unit(V_default[:, 0]),
        "combined_v2": _unit(V_default[:, 1]) if V_default.shape[1] >= 2 else None,
        "c_evi_v1": c_v1_vec,
        "c_evi_v2": c_v2_vec,
        "oracle_v1_proj": oracle_v1_proj,
        "oracle_v2_proj": oracle_v2_proj,
        "rowcheat_v1": rowcheat_v1,
        "rowcheat_v2": rowcheat_v2,
    }
    return cands, B_union, V_rowcheat


def block_report(args, A, V_exact, snap, block_id, out_lines):
    A_cur = snap["A_cur"]
    A_fut = snap["A_fut"]
    A_sketch = snap["A_sketch"] if snap["A_sketch"].size else None

    consts = per_block_constants(A, block_id, args.half_win)
    c_sk = consts["c_sk"]
    c_g1 = consts["c_g1"]
    c_g2 = consts["c_g2"]
    cur_F2 = consts["cur_F2"]
    fut_F2 = consts["fut_F2"]

    sk_F2_low = float(np.sum(np.asarray(snap["A_sketch"], dtype=np.float64) ** 2)) if A_sketch is not None else 0.0

    out_lines.append(
        f"== block {block_id}  half_win={args.half_win}  "
        f"sk_F2_low(rank-r)={sk_F2_low:.4e}  cur_F2={cur_F2:.4e}  fut_F2={fut_F2:.4e}"
    )

    cands, B_union, V_rowcheat = collect_candidates(args, snap, V_exact, c_sk, c_g1, c_g2)

    rows = []
    for label, v in cands.items():
        if v is None:
            continue
        score, u_sk, u_g1, u_g2 = f_hm_score(A_sketch, A_cur, A_fut, sk_F2_low, cur_F2, fut_F2, v)
        # User's display form: F2 / raw = 1 / u (with NaN if no sketch).
        rat_sk = (1.0 / u_sk) if (u_sk == u_sk and u_sk > 0.0) else float("nan")
        rat_g1 = (1.0 / u_g1) if u_g1 > 0.0 else float("nan")
        rat_g2 = (1.0 / u_g2) if u_g2 > 0.0 else float("nan")
        rows.append({
            "label": label,
            "score": score,
            "u_sk": u_sk, "u_g1": u_g1, "u_g2": u_g2,
            "rat_sk": rat_sk, "rat_g1": rat_g1, "rat_g2": rat_g2,
        })

    rows.sort(key=lambda r: r["score"], reverse=True)

    out_lines.append(
        f"   {'rank':>4} {'cand':<16} {'F2/raw_sk':>11} {'F2/raw_g1':>11} {'F2/raw_g2':>11} "
        f"{'u_sk':>10} {'u_g1':>10} {'u_g2':>10} {'F-HM-score':>12}"
    )
    for i, r in enumerate(rows, start=1):
        out_lines.append(
            f"   {i:>4} {r['label']:<16} "
            f"{r['rat_sk']:>11.4e} {r['rat_g1']:>11.4e} {r['rat_g2']:>11.4e} "
            f"{r['u_sk']:>10.4e} {r['u_g1']:>10.4e} {r['u_g2']:>10.4e} "
            f"{r['score']:>12.4e}"
        )

    # Rank-r exploitability check (INFRA-03). Print the F-HM3 frame score for
    # the rank-r row-cheat frame and the projected oracle frame, so the
    # exploitability invariant survives the rank-r lift. STOP rule:
    # score(V_oracle_frame) >= score(V_rowcheat_frame). When this fails on
    # any block, the F-HM3 score is row-exploitable at rank r.
    rank = int(args.rank)
    if rank > 1 and V_rowcheat is not None:
        V_oracle_frame = _oracle_frame_proj(V_exact, B_union, rank)
        rc = _frame_score_S6(A_sketch, A_cur, A_fut, sk_F2_low, cur_F2, fut_F2, V_rowcheat)
        if V_oracle_frame is not None:
            ora = _frame_score_S6(A_sketch, A_cur, A_fut, sk_F2_low, cur_F2, fut_F2, V_oracle_frame)
            d = ora["score"] - rc["score"]
            out_lines.append(
                f"   rank-r row-cheat (rank_r={V_rowcheat.shape[1]}, oracle_r="
                f"{V_oracle_frame.shape[1]}):"
            )
            out_lines.append(
                f"     rowcheat: F-HM3={rc['score']:.4e}  u_sk={rc['u_sk']:.4e}  "
                f"u_g1={rc['u_g1']:.4e}  u_g2={rc['u_g2']:.4e}"
            )
            out_lines.append(
                f"     oracle:   F-HM3={ora['score']:.4e}  u_sk={ora['u_sk']:.4e}  "
                f"u_g1={ora['u_g1']:.4e}  u_g2={ora['u_g2']:.4e}"
            )
            out_lines.append(
                f"     Δ(oracle_frame − rowcheat_frame) F-HM3 = {d:+.4e}"
            )
        else:
            out_lines.append(
                f"   rank-r row-cheat (rank_r={V_rowcheat.shape[1]}): "
                f"F-HM3={rc['score']:.4e}  (oracle frame unavailable)"
            )
    out_lines.append("")


def run_matrix(args, matrix, blocks, out_lines):
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=matrix, n=args.n, preset=args.preset, seed=args.seed,
        r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
        shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, np.float64)
    V_exact = np.asarray(V_exact, np.float64)
    target = max(blocks)
    snaps = stream_to_block(args, A, V_exact, work_dtype, int(args.rank), target, set(blocks))
    out_lines.append(f"### matrix={matrix}")
    for b in sorted(blocks):
        if b not in snaps:
            continue
        block_report(args, A, V_exact, snaps[b], b, out_lines)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrices", nargs="+", default=["mixed-tail-sharp"])
    p.add_argument("--blocks", nargs="+", type=int, default=[1, 2, 6, 12, 31])
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
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_lines = []
    for matrix in args.matrices:
        run_matrix(args, matrix, args.blocks, out_lines)
    text = "\n".join(out_lines)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
