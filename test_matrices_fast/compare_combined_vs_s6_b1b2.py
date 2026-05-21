"""Block 1 / block 2 comparison for combined vs S6 candidates.

For each candidate, print:
  F2_sk / raw_sk   = ||A_sketch||_F^2 / raw_sk     ( = 1 / u_sk )
  F2_cur / raw_g1  = ||A_cur||_F^2    / raw_g1     ( = 1 / u_g1 )
  F2_fut / raw_g2  = ||A_fut||_F^2    / raw_g2     ( = 1 / u_g2 )
  relH1            = entropy(A_cur v) / log(window)

Candidates: combined_v1/v2, oracle_v1/v2_proj, S6_v1_full / S6_v2_deflate.
"""
import argparse
import numpy as np

import cex_restricted_space_probe as probe
from future_hmean_optimizer_diagnostic import orth_basis_against, rowspace_basis
from hmean_evidence_score import (
    entropy_relH1_value_grad,
    per_block_constants,
    stream_to_block,
)
from r_sk_g_score import optimize_r_sk_g_in_basis


def _unit(v):
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    nv = float(np.linalg.norm(v))
    return None if nv <= 1e-30 else v / nv


def _proj(v, B):
    if v is None or B is None or B.size == 0:
        return None
    p = B @ (B.T @ v)
    return _unit(p)


def block_report(args, A, V_exact, snap, block_id):
    A_cur = np.asarray(snap["A_cur"], dtype=np.float64)
    A_fut = np.asarray(snap["A_fut"], dtype=np.float64)
    A_sketch = (
        np.asarray(snap["A_sketch"], dtype=np.float64)
        if snap["A_sketch"].size else None
    )
    V_default = np.asarray(snap["V_default"], dtype=np.float64)

    consts = per_block_constants(A, block_id, args.half_win)
    cur_F2 = float(consts["cur_F2"])
    fut_F2 = float(consts["fut_F2"])
    sk_F2_low = float(np.sum(A_sketch * A_sketch)) if A_sketch is not None else 0.0
    c_sk = float(consts["c_sk"])

    if A_sketch is not None:
        union_stack = np.vstack([A_sketch, A_cur, A_fut])
    else:
        union_stack = np.vstack([A_cur, A_fut])
    B_union = rowspace_basis(union_stack)

    oracle_v1 = _unit(V_exact[:, 0])
    oracle_v2 = _unit(V_exact[:, 1])
    oracle_v1_proj = _proj(oracle_v1, B_union)
    oracle_v2_proj = _proj(oracle_v2, B_union)

    # S6 sequential rank-2 over B_union.
    starts = [V_default[:, 0], V_default[:, 1], oracle_v1_proj, oracle_v2_proj]
    starts = [s for s in starts if s is not None]
    rng = np.random.default_rng(args.seed + 91000 + block_id)
    s6_v1 = optimize_r_sk_g_in_basis(
        A_cur, A_fut, A_sketch, c_sk,
        B_union, starts, rng,
        args.maxit, args.tol, args.random_starts,
        variant="S6",
        cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
    )
    s6_v1_vec = None if s6_v1 is None else _unit(s6_v1["vec"])
    if s6_v1_vec is not None:
        B_def = orth_basis_against(B_union, s6_v1_vec)
        s6_v2 = optimize_r_sk_g_in_basis(
            A_cur, A_fut, A_sketch, c_sk,
            B_def, starts, np.random.default_rng(args.seed + 92000 + block_id),
            args.maxit, args.tol, args.random_starts,
            variant="S6",
            cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
        )
        s6_v2_vec = None if s6_v2 is None else _unit(s6_v2["vec"])
    else:
        s6_v2_vec = None

    cands = {
        "combined_v1":   _unit(V_default[:, 0]),
        "combined_v2":   _unit(V_default[:, 1]) if V_default.shape[1] >= 2 else None,
        "oracle_v1_proj": oracle_v1_proj,
        "oracle_v2_proj": oracle_v2_proj,
        "S6_v1_full":    s6_v1_vec,
        "S6_v2_deflate": s6_v2_vec,
    }

    print(f"== block {block_id}  N_sk(prefix)={consts['N_sk']}  "
          f"sk_F2_low(rank-r carry)={sk_F2_low:.4e}  "
          f"cur_F2={cur_F2:.4e}  fut_F2={fut_F2:.4e}")
    print(f"   {'cand':<16} {'F2_sk/raw_sk':>14} {'F2_cur/raw_g1':>14} "
          f"{'F2_fut/raw_g2':>14} {'relH1':>10} "
          f"{'u_sk':>10} {'u_g1':>10} {'u_g2':>10} {'F-HM-score':>12}")

    for label, v in cands.items():
        if v is None:
            continue
        if A_sketch is not None and sk_F2_low > 0:
            raw_sk = float((A_sketch @ v) @ (A_sketch @ v))
            u_sk = raw_sk / sk_F2_low
            rat_sk = sk_F2_low / max(raw_sk, 1e-30)
            rat_sk_str = f"{rat_sk:>14.4e}"
            u_sk_str = f"{u_sk:>10.4e}"
        else:
            u_sk = float("nan")
            rat_sk_str = f"{'—':>14}"
            u_sk_str = f"{'—':>10}"

        raw_g1 = float((A_cur @ v) @ (A_cur @ v))
        raw_g2 = float((A_fut @ v) @ (A_fut @ v))
        u_g1 = raw_g1 / max(cur_F2, 1e-30)
        u_g2 = raw_g2 / max(fut_F2, 1e-30)
        rat_g1 = cur_F2 / max(raw_g1, 1e-30)
        rat_g2 = fut_F2 / max(raw_g2, 1e-30)

        relH1, _ = entropy_relH1_value_grad(A_cur, v)
        relH1 = float(relH1)

        eps = 1e-30
        if A_sketch is not None and sk_F2_low > 0 and u_sk > eps and u_g1 > eps and u_g2 > eps:
            score = 3.0 / (1.0 / u_sk + 1.0 / u_g1 + 1.0 / u_g2)
        elif u_g1 > eps and u_g2 > eps:
            score = 2.0 / (1.0 / u_g1 + 1.0 / u_g2)
        else:
            score = 0.0

        print(f"   {label:<16} {rat_sk_str} {rat_g1:>14.4e} {rat_g2:>14.4e} "
              f"{relH1:>10.4f} {u_sk_str} {u_g1:>10.4e} {u_g2:>10.4e} "
              f"{score:>12.4e}")
    print()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="mixed-tail-sharp")
    p.add_argument("--blocks", nargs="+", type=int, default=[1, 2])
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
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--post-expansion-maxit", type=int, default=80)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--patience-rel-tol", type=float, default=1e-5)
    p.add_argument("--random-starts", type=int, default=24)
    p.add_argument("--r-sig", type=int, default=2)
    p.add_argument("--alpha-sig", type=float, default=0.003)
    p.add_argument("--alpha-tail", type=float, default=0.0145)
    p.add_argument("--tail-scale", type=float, default=0.99)
    p.add_argument("--sigma1", type=float, default=0.991)
    p.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    return p.parse_args()


def main():
    args = parse_args()
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=args.matrix, n=args.n, preset=args.preset, seed=args.seed,
        r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
        shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, np.float64)
    V_exact = np.asarray(V_exact, np.float64)
    target = max(args.blocks)
    snaps = stream_to_block(
        args, A, V_exact, work_dtype, int(args.rank), target, set(args.blocks)
    )
    print(f"matrix={args.matrix}  half_win={args.half_win}\n")
    for b in sorted(args.blocks):
        if b in snaps:
            block_report(args, A, V_exact, snaps[b], b)


if __name__ == "__main__":
    main()
