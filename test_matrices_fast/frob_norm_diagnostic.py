"""Frobenius-norm diagnostic for HM-score reweighting.

For each (matrix, block), compute:
  - ||A_sketch||_F^2, ||A_cur||_F^2, ||A_fut||_F^2
  - For each candidate v (unit-normalised): raw_sk, raw_g1, raw_g2
  - Ratios r_sk_F = ||A_sketch||_F^2 / raw_sk(v),
           r_g1_F = ||A_cur||_F^2    / raw_g1(v),
           r_g2_F = ||A_fut||_F^2    / raw_g2(v)
  - For reference also print c_sk * raw_sk, c_g1 * raw_g1, c_g2 * raw_g2
    (the row-count-normalised versions used by the existing scores).

The intent is to inspect typical magnitudes of these quantities so that we
can pick sensible Frobenius-based weights for a new HM3 variant.

Note: A_sketch here is the carried rank-r low-rank summary
(state["s"] · state["V"]^T) — NOT the full A_sk_full used to define c_sk.
So sk_F2_low = sum(s_i^2) (rank-r) while sk_F2_full = ||A[:sk_end]||_F^2
(used by per_block_constants).  Both are reported.
"""

import argparse

import numpy as np

import cex_restricted_space_probe as probe
from future_hmean_optimizer_diagnostic import orth_basis_against, rowspace_basis
from hmean_evidence_score import per_block_constants, stream_to_block


CANDIDATE_LABELS = (
    "combined_v1",
    "combined_v2",
    "sketch_v1",
    "sketch_v2",
    "mgain_svd_v1",
    "mgain_svd_v2",
    "oracle_v1_proj",
    "oracle_v2_proj",
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


def _rayleigh_sq(A, v):
    if A is None or v is None or A.size == 0:
        return float("nan")
    y = A @ v
    return float(np.dot(y, y))


def collect_candidates(snap, V_exact):
    A_cur = snap["A_cur"]
    A_fut = snap["A_fut"]
    A_sketch = snap["A_sketch"] if snap["A_sketch"].size else None
    M_gain = snap["M_gain"]
    state = snap["state"]
    V_default = snap["V_default"]

    # Build basis for projecting oracle.
    if A_sketch is not None:
        union_stack = np.vstack([A_sketch, A_cur, A_fut])
    else:
        union_stack = np.vstack([A_cur, A_fut])
    B_union = rowspace_basis(union_stack)

    # Sketch right singular vectors (state.V).
    sketch_v1 = sketch_v2 = None
    if state is not None and state.get("V") is not None:
        Vs = np.asarray(state["V"], dtype=np.float64)
        if Vs.size:
            if Vs.shape[1] >= 1:
                sketch_v1 = Vs[:, 0]
            if Vs.shape[1] >= 2:
                sketch_v2 = Vs[:, 1]

    # Top-2 right singular vectors of M_gain.
    mgain_svd_v1 = mgain_svd_v2 = None
    M = np.asarray(M_gain, dtype=np.float64)
    if M.size:
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        if Vt.shape[0] >= 1:
            mgain_svd_v1 = Vt[0]
        if Vt.shape[0] >= 2:
            mgain_svd_v2 = Vt[1]

    o1 = V_exact[:, 0]
    o2 = V_exact[:, 1]
    oracle_v1_proj = _project(o1, B_union)
    oracle_v2_proj = _project(o2, B_union)

    return {
        "combined_v1": _unit(V_default[:, 0]),
        "combined_v2": _unit(V_default[:, 1]) if V_default.shape[1] >= 2 else None,
        "sketch_v1": _unit(sketch_v1),
        "sketch_v2": _unit(sketch_v2),
        "mgain_svd_v1": _unit(mgain_svd_v1),
        "mgain_svd_v2": _unit(mgain_svd_v2),
        "oracle_v1_proj": oracle_v1_proj,
        "oracle_v2_proj": oracle_v2_proj,
    }


def block_report(args, A, V_exact, snap, block_id, out_lines):
    A_cur = snap["A_cur"]
    A_fut = snap["A_fut"]
    A_sketch = snap["A_sketch"] if snap["A_sketch"].size else None

    consts = per_block_constants(A, block_id, args.half_win)
    c_sk = consts["c_sk"]
    c_g1 = consts["c_g1"]
    c_g2 = consts["c_g2"]
    sk_F2_full = consts["sk_F2"]
    cur_F2 = consts["cur_F2"]
    fut_F2 = consts["fut_F2"]
    N_sk = consts["N_sk"]

    # Frobenius-squared of the carried sketch (rank-r) — what the score sees.
    sk_F2_low = float(np.sum(np.asarray(snap["A_sketch"], dtype=np.float64) ** 2)) if A_sketch is not None else 0.0
    # Top sketch singular value (helpful sanity check).
    s_top = 0.0
    state = snap["state"]
    if state is not None and state.get("s") is not None:
        s_arr = np.asarray(state["s"], dtype=np.float64).reshape(-1)
        if s_arr.size:
            s_top = float(s_arr[0])

    out_lines.append(
        f"== block {block_id}  half_win={args.half_win}  N_sk={N_sk}  "
        f"sk_F2_full={sk_F2_full:.4e}  sk_F2_low(rank-r)={sk_F2_low:.4e}  "
        f"cur_F2={cur_F2:.4e}  fut_F2={fut_F2:.4e}\n"
        f"   c_sk={c_sk:.4e}  c_g1={c_g1:.4e}  c_g2={c_g2:.4e}  s_top(sketch)={s_top:.4e}"
    )
    out_lines.append(
        f"   {'cand':<16} {'raw_sk':>10} {'raw_g1':>10} {'raw_g2':>10} "
        f"{'F2/raw_sk':>11} {'F2/raw_g1':>11} {'F2/raw_g2':>11} "
        f"{'c_sk*raw_sk':>12} {'c_g1*raw_g1':>12} {'c_g2*raw_g2':>12}"
    )

    cands = collect_candidates(snap, V_exact)
    for label in CANDIDATE_LABELS:
        v = cands.get(label)
        if v is None:
            continue
        raw_sk = _rayleigh_sq(A_sketch, v) if A_sketch is not None else 0.0
        raw_g1 = _rayleigh_sq(A_cur, v)
        raw_g2 = _rayleigh_sq(A_fut, v)
        rat_sk = (sk_F2_low / raw_sk) if (A_sketch is not None and raw_sk > 1e-30) else float("nan")
        rat_g1 = (cur_F2 / raw_g1) if raw_g1 > 1e-30 else float("nan")
        rat_g2 = (fut_F2 / raw_g2) if raw_g2 > 1e-30 else float("nan")
        c_sk_raw = c_sk * raw_sk
        c_g1_raw = c_g1 * raw_g1
        c_g2_raw = c_g2 * raw_g2
        out_lines.append(
            f"   {label:<16} {raw_sk:>10.4e} {raw_g1:>10.4e} {raw_g2:>10.4e} "
            f"{rat_sk:>11.4e} {rat_g1:>11.4e} {rat_g2:>11.4e} "
            f"{c_sk_raw:>12.4e} {c_g1_raw:>12.4e} {c_g2_raw:>12.4e}"
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
    p.add_argument("--matrices", nargs="+", default=["static-cex", "mixed-tail-sharp", "diffuse-diffuse"])
    p.add_argument("--blocks", nargs="+", type=int, default=[2, 6, 12, 31])
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
    p.add_argument("--r-sig", type=int, default=2)
    p.add_argument("--alpha-sig", type=float, default=0.003)
    p.add_argument("--alpha-tail", type=float, default=0.0145)
    p.add_argument("--tail-scale", type=float, default=0.99)
    p.add_argument("--sigma1", type=float, default=0.991)
    p.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    p.add_argument("--out", default=None, help="Optional file to also write the report to.")
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
