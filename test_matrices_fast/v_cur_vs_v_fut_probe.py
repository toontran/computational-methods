"""v_cur vs v_fut row-aligned shift probe.

For each (matrix, seed, block) cell:
  v_cur = combined-score optimum on rowsets {S, A_cur}    (the bench's actual lock)
  v_fut = combined-score optimum on rowsets {S, A_cur, A_fut}

Both share the same state S (sketch from prior blocks); v_fut differs only in
that its current-block is the 64-row stack [A_cur; A_fut] instead of A_cur.

The hypothesis (per_block_visibility_analysis.md → next-step probe):
  cos²(e_cur, e_fut) reads as an uncertainty signal where e* = A_cur · v_*.

Design notes:
- v_cur is the snapshot's V_default[:,0] from stream_to_block (no extra work).
- v_fut is computed by re-running entropy_iter_basis_forget with
  M_gain = [B_top; A_cur; A_fut], A_block = [A_cur; A_fut], same state_prev.
- Sign-align v_fut so v_fut · v_cur > 0 before reading e-vectors.
- Slot-1 only (rank=2 carry, but we look at V[:,0]).

Output:
  summary/per_block_v_cur_vs_v_fut/cells.csv   — one row per (matrix, seed, block)
  summary/per_block_v_cur_vs_v_fut/report.md   — aggregated table + discrimination stats
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

import cex_restricted_space_probe as probe
from hmean_evidence_score import stream_to_block

DEFAULT_MATRICES = ["diffuse-diffuse", "mixed-tail-sharp", "mixed-tail-soft", "static-cex"]
DEFAULT_SEEDS = [0, 1, 2]
N_BLOCKS = 16
HALF_WIN = 32
N = 1024
RANK = 2
PRESET = "fast"


def make_args(seed: int) -> SimpleNamespace:
    """Mirrors the bench's default flags so stream_to_block runs identically."""
    return SimpleNamespace(
        matrix="placeholder",
        half_win=HALF_WIN,
        n=N,
        rank=RANK,
        preset=PRESET,
        seed=seed,
        shuffle_rows=True,
        row_shuffle_seed=seed,
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
        r_sig=2,
        alpha_sig=0.003,
        alpha_tail=0.0145,
        tail_scale=0.99,
        sigma1=0.991,
        v_type="rand",
    )


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-30:
        return v
    return v / n


def compute_v_fut(A_cur, A_fut, A_sketch, state, seed, rng_seed) -> np.ndarray:
    """Run the same combined-score optimizer with A_block extended to [A_cur; A_fut]."""
    work_dtype = np.float32
    A_cur64 = np.vstack([A_cur, A_fut]).astype(work_dtype, copy=False)
    if A_sketch is not None and A_sketch.size:
        B_top = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
        M_gain = np.vstack([B_top, A_cur64]).astype(work_dtype, copy=False)
        rows_seen = state["rows_seen"] + A_cur64.shape[0]
    else:
        M_gain = A_cur64
        rows_seen = A_cur64.shape[0]

    V_init = probe.row_norm_seed(A_cur64, RANK)
    V_score, _, _, _, _ = probe.entropy_iter_basis_forget(
        M_gain=M_gain,
        active_r=RANK,
        rows_ref=N,  # rows_ref is the global matrix-row count, kept identical to bench
        V_init=np.asarray(V_init, dtype=work_dtype),
        q0=8, qmax=48, krylov_depth=2,
        residual_tol=0.01, expansion_maxit=8,
        num_restarts=3, maxit=120, tol=1e-8,
        rng=np.random.default_rng(rng_seed),
        verbose=False, state_prev=state, A_block=A_cur64, rows_total=rows_seen,
        reduced_optimizer="cex", basis_selection="greedy",
        work_dtype=work_dtype, expansion_direction="residual",
        reuse_line_search_grad=True, expansion_warm_start=True,
        post_expansion_maxit=80,
        score_variant="combined", old_row_memory=None,
        combined_rank=None, patience=5, patience_rel_tol=1e-5,
    )
    return np.ascontiguousarray(np.asarray(V_score[:, 0], dtype=np.float64))


def per_row_stats(A_cur, v_cur, v_fut):
    """Compute the drift stats described in the probe spec."""
    e_cur = A_cur @ v_cur
    e_fut = A_cur @ v_fut

    # Vector-space direction shift
    cos_vv = float(np.dot(v_cur, v_fut))
    direction_shift = 1.0 - abs(cos_vv)

    # Row-aligned shift on A_cur
    nc = float(np.linalg.norm(e_cur))
    nf = float(np.linalg.norm(e_fut))
    if nc > 1e-30 and nf > 1e-30:
        cos_ee = float(np.dot(e_cur, e_fut) / (nc * nf))
    else:
        cos_ee = 0.0
    row_aligned_shift = 1.0 - abs(cos_ee)

    # Element-energy shift (sign-free, in [0,1])
    e_cur_sq = e_cur * e_cur
    e_fut_sq = e_fut * e_fut
    sum_cur = float(np.sum(e_cur_sq))
    sum_fut = float(np.sum(e_fut_sq))
    denom = sum_cur + sum_fut
    if denom > 1e-30:
        element_energy_shift = float(np.sum(np.abs(e_cur_sq - e_fut_sq))) / denom
    else:
        element_energy_shift = 0.0

    # Rank shift: Spearman distance between |e_cur| and |e_fut| orderings
    # Normalized to [0, 1] via 1 - |spearman corr|
    abs_cur = np.abs(e_cur)
    abs_fut = np.abs(e_fut)
    rank_cur = abs_cur.argsort().argsort().astype(np.float64)
    rank_fut = abs_fut.argsort().argsort().astype(np.float64)
    rank_cur -= rank_cur.mean()
    rank_fut -= rank_fut.mean()
    rc_n = float(np.linalg.norm(rank_cur))
    rf_n = float(np.linalg.norm(rank_fut))
    if rc_n > 1e-30 and rf_n > 1e-30:
        spearman = float(np.dot(rank_cur, rank_fut) / (rc_n * rf_n))
    else:
        spearman = 0.0
    rank_shift = 1.0 - abs(spearman)

    # Mass shift
    mass_cur = sum_cur
    mass_fut = sum_fut
    mass_max = max(mass_cur, mass_fut, 1e-30)
    mass_shift = abs(mass_cur - mass_fut) / mass_max

    # Row-entropy shift (normalized)
    half_win = e_cur.shape[0]
    log_n = np.log(max(half_win, 2))
    def rel_H(e_sq):
        s = max(float(np.sum(e_sq)), 1e-30)
        p = e_sq / s
        p_pos = np.maximum(p, 1e-300)
        H = -float(np.sum(p * np.log(p_pos)))
        return H / log_n
    H_cur = rel_H(e_cur_sq)
    H_fut = rel_H(e_fut_sq)
    entropy_shift = abs(H_cur - H_fut)

    return {
        "direction_shift": direction_shift,
        "row_aligned_shift": row_aligned_shift,
        "element_energy_shift": element_energy_shift,
        "rank_shift": rank_shift,
        "mass_shift": mass_shift,
        "entropy_shift": entropy_shift,
        "cos_vv": cos_vv,
        "cos_ee": cos_ee,
        "mass_cur": mass_cur,
        "mass_fut": mass_fut,
        "relH_cur": H_cur,
        "relH_fut": H_fut,
    }


def cos2_to_oracle(v_cur, V_exact_col0):
    o = V_exact_col0 / max(np.linalg.norm(V_exact_col0), 1e-30)
    return float(np.dot(v_cur, o)) ** 2


def rowspace_basis_local(M):
    if M is None or M.size == 0:
        return None
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    tol = max(M.shape) * np.finfo(np.float64).eps * (s[0] if s.size else 0.0)
    keep = s > tol
    return np.ascontiguousarray(Vt[keep].T)


def cos2_in_basis(v, B, target):
    """cos²(v, P_B target) where P_B is the orthogonal projector onto col(B).
    Returns 0 if target has no mass in B.
    """
    if B is None or B.size == 0 or target is None:
        return 0.0
    p = B @ (B.T @ target)
    n = float(np.linalg.norm(p))
    if n <= 1e-30:
        return 0.0
    p = p / n
    return float(np.dot(v, p)) ** 2


def run_one_seed(matrix, seed, sanity_only=False):
    args = make_args(seed)
    args.matrix = matrix
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=matrix, n=args.n, preset=args.preset, seed=seed,
        r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
        shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, dtype=np.float64)
    V_exact = np.asarray(V_exact, dtype=np.float64)
    blocks = list(range(1, N_BLOCKS + 1))
    snapshots = stream_to_block(args, A, V_exact, np.float32, RANK, max(blocks), set(blocks))

    rows = []
    for b in blocks:
        if b not in snapshots:
            continue
        snap = snapshots[b]
        A_cur = snap["A_cur"]
        A_fut = snap["A_fut"]
        A_sketch = snap["A_sketch"]
        state = snap["state"]
        V_default = snap["V_default"]
        v_cur = normalize(np.asarray(V_default[:, 0], dtype=np.float64))

        # rng seed for v_fut: same family as bench's seed, perturbed by block id
        v_fut_raw = compute_v_fut(A_cur, A_fut, A_sketch, state, seed,
                                  rng_seed=seed + 7919 * b)
        v_fut = normalize(v_fut_raw)
        # sign-align on the vectors
        if float(np.dot(v_fut, v_cur)) < 0:
            v_fut = -v_fut

        stats = per_row_stats(A_cur, v_cur, v_fut)
        c2 = cos2_to_oracle(v_cur, V_exact[:, 0])
        c2_fut = cos2_to_oracle(v_fut, V_exact[:, 0])

        # Reachable-oracle cos²: V_exact[:,0] projected onto rowspace([sketch; A_cur])
        # then sign-normalized. This is a finer "is v_cur a good lock" signal
        # than raw V_exact alignment, and is what visibility-analysis §3 uses.
        if A_sketch.size:
            search_stack = np.vstack([A_sketch, A_cur])
        else:
            search_stack = A_cur
        B_search = rowspace_basis_local(search_stack.astype(np.float64))
        c2_v_cur_reach = cos2_in_basis(v_cur, B_search, V_exact[:, 0])

        row = {
            "matrix": matrix,
            "seed": seed,
            "block": b,
            "cos2_v_cur_oracle": c2,
            "cos2_v_fut_oracle": c2_fut,
            "cos2_v_cur_reach": c2_v_cur_reach,
            **stats,
        }
        rows.append(row)
        if sanity_only:
            print(f"  matrix={matrix} seed={seed} block={b}  "
                  f"cos²(v_cur,V0)={c2:.4f}  cos(v_cur,v_fut)={stats['cos_vv']:+.4f}  "
                  f"row_aligned_shift={stats['row_aligned_shift']:.4f}  "
                  f"mass_cur={stats['mass_cur']:.3e}  mass_fut={stats['mass_fut']:.3e}  "
                  f"relH(c)={stats['relH_cur']:.3f}  relH(f)={stats['relH_fut']:.3f}")

    return rows


def write_cells_csv(path, rows):
    if not rows:
        return
    keys = ["matrix", "seed", "block",
            "cos2_v_cur_oracle", "cos2_v_fut_oracle", "cos2_v_cur_reach",
            "direction_shift", "row_aligned_shift", "element_energy_shift",
            "rank_shift", "mass_shift", "entropy_shift",
            "cos_vv", "cos_ee",
            "mass_cur", "mass_fut", "relH_cur", "relH_fut"]
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(_fmt(r[k]) for k in keys) + "\n")


def _fmt(x):
    if isinstance(x, str):
        return x
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.6e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="summary/per_block_v_cur_vs_v_fut")
    p.add_argument("--matrices", nargs="+", default=DEFAULT_MATRICES)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--sanity", action="store_true",
                   help="Run only one (matrix, seed, blocks 1..4) to sanity-check.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.sanity:
        rows = run_one_seed(args.matrices[0], args.seeds[0], sanity_only=True)
        # Just print, no file write
        return

    all_rows = []
    for m in args.matrices:
        for s in args.seeds:
            print(f"=== {m}  seed={s} ===")
            sub = run_one_seed(m, s, sanity_only=False)
            for r in sub:
                print(f"  block={r['block']:2d}  "
                      f"cos²(v_cur,V0)={r['cos2_v_cur_oracle']:.4f}  "
                      f"cos(v_cur,v_fut)={r['cos_vv']:+.4f}  "
                      f"row_aligned_shift={r['row_aligned_shift']:.4f}  "
                      f"mass_cur={r['mass_cur']:.2e}  mass_fut={r['mass_fut']:.2e}  "
                      f"relH(c)={r['relH_cur']:.3f}  relH(f)={r['relH_fut']:.3f}")
            all_rows.extend(sub)

    cells_csv = os.path.join(args.out_dir, "cells.csv")
    write_cells_csv(cells_csv, all_rows)
    print(f"\nWrote {len(all_rows)} cells to {cells_csv}")

    # Drop a JSON snapshot too (lossless)
    cells_json = os.path.join(args.out_dir, "cells.json")
    with open(cells_json, "w") as f:
        json.dump(all_rows, f, indent=2, default=float)


if __name__ == "__main__":
    main()
