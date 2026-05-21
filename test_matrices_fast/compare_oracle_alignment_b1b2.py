"""Cosine²-alignment of (combined v1, v2) and (S6 v1_full, v2_deflate)
against the oracle projections (oracle_v1_proj, oracle_v2_proj).

For each matrix and each of blocks 1, 2: prints a 2×2 cos² matrix between
{combined_v1, combined_v2} and {oracle_v1_proj, oracle_v2_proj}, and the
same for {S6_v1_full, S6_v2_deflate}.
"""
import argparse
import numpy as np

import cex_restricted_space_probe as probe
from future_hmean_optimizer_diagnostic import orth_basis_against, rowspace_basis
from hmean_evidence_score import per_block_constants, stream_to_block
from r_sk_g_score import optimize_r_sk_g_in_basis


def _unit(v):
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    nv = float(np.linalg.norm(v))
    return None if nv <= 1e-30 else v / nv


def _proj_unit(v, B):
    if v is None or B is None or B.size == 0:
        return None
    return _unit(B @ (B.T @ v))


def cos2(u, v):
    if u is None or v is None:
        return float("nan")
    return float(np.dot(u, v) ** 2)


def block_alignment(args, A, V_exact, snap, block_id):
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

    o1 = _proj_unit(V_exact[:, 0], B_union)
    o2 = _proj_unit(V_exact[:, 1], B_union)

    starts = [V_default[:, 0], V_default[:, 1], o1, o2]
    starts = [s for s in starts if s is not None]
    s6_v1 = optimize_r_sk_g_in_basis(
        A_cur, A_fut, A_sketch, c_sk,
        B_union, starts, np.random.default_rng(args.seed + 91000 + block_id),
        args.maxit, args.tol, args.random_starts,
        variant="S6", cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
    )
    s6_v1_vec = None if s6_v1 is None else _unit(s6_v1["vec"])
    s6_v2_vec = None
    if s6_v1_vec is not None:
        B_def = orth_basis_against(B_union, s6_v1_vec)
        s6_v2 = optimize_r_sk_g_in_basis(
            A_cur, A_fut, A_sketch, c_sk,
            B_def, starts, np.random.default_rng(args.seed + 92000 + block_id),
            args.maxit, args.tol, args.random_starts,
            variant="S6", cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
        )
        s6_v2_vec = None if s6_v2 is None else _unit(s6_v2["vec"])

    c1 = _unit(V_default[:, 0])
    c2 = _unit(V_default[:, 1]) if V_default.shape[1] >= 2 else None

    def fmt(x):
        return "    nan" if (x != x) else f"{x:7.4f}"

    print(f"  block {block_id} (sketch_present={A_sketch is not None})")
    print(f"    {'cos² vs:':<22} {'oracle_v1_proj':>16} {'oracle_v2_proj':>16}")
    print(f"    {'combined_v1':<22} {fmt(cos2(c1, o1)):>16} {fmt(cos2(c1, o2)):>16}")
    print(f"    {'combined_v2':<22} {fmt(cos2(c2, o1)):>16} {fmt(cos2(c2, o2)):>16}")
    print(f"    {'S6_v1_full':<22} {fmt(cos2(s6_v1_vec, o1)):>16} {fmt(cos2(s6_v1_vec, o2)):>16}")
    print(f"    {'S6_v2_deflate':<22} {fmt(cos2(s6_v2_vec, o1)):>16} {fmt(cos2(s6_v2_vec, o2)):>16}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrices", nargs="+",
                   default=["mixed-tail-sharp", "static-cex", "diffuse-diffuse"])
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
    for matrix in args.matrices:
        A, V_exact, _, _ = probe.generate_matrix_input(
            matrix=matrix, n=args.n, preset=args.preset, seed=args.seed,
            r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
            tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
            shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
        )
        A = np.asarray(A, np.float64)
        V_exact = np.asarray(V_exact, np.float64)
        target = max(args.blocks)
        snaps = stream_to_block(args, A, V_exact, work_dtype, int(args.rank), target, set(args.blocks))
        print(f"=== matrix={matrix} ===")
        for b in sorted(args.blocks):
            if b in snaps:
                block_alignment(args, A, V_exact, snaps[b], b)
        print()


if __name__ == "__main__":
    main()
