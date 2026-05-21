"""Three-way mass-shift probe — read each candidate's instability under peek
along the bench's combined trajectory.

For each (matrix, seed, block) along the pure-combined stream, compute:

  ms_combined = |‖A_cur·v_combined‖² − ‖A_cur·v_combined_fut‖²| / max(...)
  ms_isvd     = |‖A_cur·v_isvd‖²    − ‖A_cur·v_isvd_fut‖²|     / max(...)
  ms_oracle   = |‖A_cur·v_oracle‖²  − ‖A_cur·v_oracle_fut‖²|   / max(...)

where each candidate's "fut" version is computed on the extended search basis
[B_top; A_cur; A_fut] (sketch state held fixed at the bench's combined-trajectory
state at this block; only the search basis grows).

  v_combined: combined optimizer, A_block = A_cur          (already cached)
  v_combined_fut: combined optimizer, A_block = [A_cur; A_fut]   (already cached)
  v_isvd: top right SV of M_gain = [B_top; A_cur]
  v_isvd_fut: top right SV of M_gain_fut = [B_top; A_cur; A_fut]
  v_oracle: V_exact[:,0] projected onto rowspace(M_gain), renormalized
  v_oracle_fut: V_exact[:,0] projected onto rowspace(M_gain_fut), renormalized

Sign-align each *_fut against its current solution before computing mass_shift.

The diagnostic question: at each block, which candidate has the smallest ms?
Does argmin(ms_*) match the matrix's regime (combined for row-concentrated,
iSVD for diffuse, oracle as upper bound)?

Output: summary/three_way_mass_shift/cells.csv  +  report.md
"""

import argparse
import csv
import os
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
    return SimpleNamespace(
        matrix="placeholder",
        half_win=HALF_WIN, n=N, rank=RANK, preset=PRESET,
        seed=seed, shuffle_rows=True, row_shuffle_seed=seed,
        old_memory_size=32, dtype="float32",
        q0=8, qmax=48, krylov_depth=2, residual_tol=0.01,
        expansion_maxit=8, num_restarts=3, maxit=120, tol=1e-8,
        post_expansion_maxit=80, patience=5, patience_rel_tol=1e-5,
        r_sig=2, alpha_sig=0.003, alpha_tail=0.0145,
        tail_scale=0.99, sigma1=0.991, v_type="rand",
    )


def normed(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-30 else v


def run_combined_optimizer(M_gain, A_block, state_prev, rank, seed_offset):
    work_dtype = np.float32
    V_init = probe.row_norm_seed(A_block, rank)
    V, _, _, _, _ = probe.entropy_iter_basis_forget(
        M_gain=np.asarray(M_gain, dtype=work_dtype),
        active_r=rank, rows_ref=N,
        V_init=np.asarray(V_init, dtype=work_dtype),
        q0=8, qmax=48, krylov_depth=2,
        residual_tol=0.01, expansion_maxit=8,
        num_restarts=3, maxit=120, tol=1e-8,
        rng=np.random.default_rng(seed_offset),
        verbose=False, state_prev=state_prev,
        A_block=np.asarray(A_block, dtype=work_dtype),
        rows_total=int(state_prev["rows_seen"] + A_block.shape[0]) if state_prev is not None else int(A_block.shape[0]),
        reduced_optimizer="cex", basis_selection="greedy",
        work_dtype=work_dtype, expansion_direction="residual",
        reuse_line_search_grad=True, expansion_warm_start=True,
        post_expansion_maxit=80,
        score_variant="combined", old_row_memory=None,
        combined_rank=None, patience=5, patience_rel_tol=1e-5,
    )
    return np.ascontiguousarray(np.asarray(V[:, :rank], dtype=np.float64))


def rowspace_basis(M):
    if M is None or M.size == 0:
        return None
    _, s, Vt = np.linalg.svd(np.asarray(M, dtype=np.float64), full_matrices=False)
    tol = max(M.shape) * np.finfo(np.float64).eps * (s[0] if s.size else 0.0)
    keep = s > tol
    return np.ascontiguousarray(Vt[keep].T) if keep.any() else None


def project_unit(target, B):
    if B is None or B.size == 0:
        return None
    p = B @ (B.T @ target)
    n = float(np.linalg.norm(p))
    return p / n if n > 1e-30 else None


def mass_shift(A_cur, v, v_fut):
    """Sign-align v_fut against v, then compute mass_shift on A_cur."""
    if v is None or v_fut is None:
        return None, None, None
    if float(np.dot(v, v_fut)) < 0:
        v_fut = -v_fut
    e = A_cur @ v
    e_fut = A_cur @ v_fut
    mc = float(np.dot(e, e))
    mf = float(np.dot(e_fut, e_fut))
    denom = max(mc, mf, 1e-30)
    return abs(mc - mf) / denom, mc, mf


def cos2(a, b):
    if a is None or b is None:
        return None
    return float(np.dot(a, b)) ** 2


def run_one_seed(matrix, seed):
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
    v_oracle_global = V_exact[:, 0] / max(np.linalg.norm(V_exact[:, 0]), 1e-30)

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
        v_combined = normed(np.asarray(V_default[:, 0], dtype=np.float64))

        # M_gain (from bench's pure-combined trajectory) and M_gain_fut
        if A_sketch.size:
            M_gain = np.vstack([A_sketch, A_cur])
            M_gain_fut = np.vstack([A_sketch, A_cur, A_fut])
        else:
            M_gain = A_cur
            M_gain_fut = np.vstack([A_cur, A_fut])
        M_gain = np.asarray(M_gain, dtype=np.float64)
        M_gain_fut = np.asarray(M_gain_fut, dtype=np.float64)

        # 1. v_combined_fut: rerun combined optimizer on extended block
        A_cur64 = np.vstack([A_cur, A_fut]).astype(np.float32)
        V_combined_fut = run_combined_optimizer(M_gain_fut.astype(np.float32),
                                                A_cur64, state, RANK, seed + 7919 * b)
        v_combined_fut = normed(np.asarray(V_combined_fut[:, 0], dtype=np.float64))

        # 2. v_isvd: top right SV of M_gain
        _, _, Vt = np.linalg.svd(M_gain, full_matrices=False)
        v_isvd = normed(np.asarray(Vt[0], dtype=np.float64))
        _, _, Vt_fut = np.linalg.svd(M_gain_fut, full_matrices=False)
        v_isvd_fut = normed(np.asarray(Vt_fut[0], dtype=np.float64))

        # 3. v_oracle: V_exact[:,0] projected onto rowspace(M_gain)
        B = rowspace_basis(M_gain)
        v_oracle = project_unit(v_oracle_global, B)
        B_fut = rowspace_basis(M_gain_fut)
        v_oracle_fut = project_unit(v_oracle_global, B_fut)

        # mass_shifts on A_cur
        ms_c, mc_c, mf_c = mass_shift(A_cur.astype(np.float64), v_combined, v_combined_fut)
        ms_i, mc_i, mf_i = mass_shift(A_cur.astype(np.float64), v_isvd, v_isvd_fut)
        ms_o, mc_o, mf_o = mass_shift(A_cur.astype(np.float64), v_oracle, v_oracle_fut) \
            if v_oracle is not None and v_oracle_fut is not None else (None, None, None)

        # cos² to V_exact[:,0] for each candidate (signal-quality reference)
        c2_c = cos2(v_combined, v_oracle_global)
        c2_i = cos2(v_isvd, v_oracle_global)
        c2_o = cos2(v_oracle, v_oracle_global) if v_oracle is not None else None

        # argmin(ms): which candidate is most stable?
        candidates = [("combined", ms_c), ("isvd", ms_i)]
        if ms_o is not None:
            candidates.append(("oracle", ms_o))
        argmin_2way = min([c for c in candidates if c[0] in ("combined", "isvd")],
                          key=lambda kv: kv[1])[0]
        argmin_3way = min(candidates, key=lambda kv: kv[1])[0]

        rows.append({
            "matrix": matrix, "seed": seed, "block": b,
            "ms_combined": ms_c, "ms_isvd": ms_i,
            "ms_oracle": ms_o if ms_o is not None else float("nan"),
            "argmin_2way": argmin_2way, "argmin_3way": argmin_3way,
            "cos2_combined": c2_c, "cos2_isvd": c2_i,
            "cos2_oracle": c2_o if c2_o is not None else float("nan"),
            "mc_combined": mc_c, "mf_combined": mf_c,
            "mc_isvd": mc_i, "mf_isvd": mf_i,
            "mc_oracle": mc_o if mc_o is not None else float("nan"),
            "mf_oracle": mf_o if mf_o is not None else float("nan"),
        })
    return rows


def _fmt(x):
    if isinstance(x, str):
        return x
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    f = float(x)
    if np.isnan(f):
        return "nan"
    return f"{f:.6e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="summary/three_way_mass_shift")
    p.add_argument("--matrices", nargs="+", default=DEFAULT_MATRICES)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_rows = []
    for m in args.matrices:
        for s in args.seeds:
            print(f"=== {m}  seed={s} ===")
            sub = run_one_seed(m, s)
            for r in sub:
                print(f"  block={r['block']:2d}  "
                      f"ms_c={r['ms_combined']:.3f}  ms_i={r['ms_isvd']:.3f}  ms_o={r['ms_oracle']:.3f}  "
                      f"argmin2={r['argmin_2way']:8s}  argmin3={r['argmin_3way']:8s}  "
                      f"cos²(c,V0)={r['cos2_combined']:.3f}  cos²(i,V0)={r['cos2_isvd']:.3f}")
            all_rows.extend(sub)

    keys = ["matrix", "seed", "block",
            "ms_combined", "ms_isvd", "ms_oracle",
            "argmin_2way", "argmin_3way",
            "cos2_combined", "cos2_isvd", "cos2_oracle",
            "mc_combined", "mf_combined",
            "mc_isvd", "mf_isvd",
            "mc_oracle", "mf_oracle"]
    cells_csv = os.path.join(args.out_dir, "cells.csv")
    with open(cells_csv, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in all_rows:
            f.write(",".join(_fmt(r[k]) for k in keys) + "\n")
    print(f"\nWrote {len(all_rows)} cells to {cells_csv}")


if __name__ == "__main__":
    main()
