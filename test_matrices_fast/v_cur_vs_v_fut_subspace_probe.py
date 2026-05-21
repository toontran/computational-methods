"""Subspace-level peek-perturbation probe.

Vector-level probe (v_cur_vs_v_fut_probe.py) compares slot-1 only.
This extends to the full rank-r frame.

For each block:
  V_cur ∈ ℝ^{n×r}  — combined optimizer's full rank-r frame, A_block = A_cur
  V_fut ∈ ℝ^{n×r}  — same optimizer, A_block = [A_cur; A_fut]

Read on A_cur:
  E_cur = A_cur · V_cur ∈ ℝ^{half_win × r}
  E_fut = A_cur · V_fut

Subspace-level statistics (rotation/sign invariant where applicable):

  mass_F_cur, mass_F_fut    — ‖E_*‖_F² (Frobenius energy on A_cur)
  mass_shift_F              — |mass_F_cur − mass_F_fut| / max(...)
  principal_cos²            — top-r singular values squared of Q_cur^T Q_fut
                              where Q_* = orth(E_*) (col-span basis)
  subspace_angle_drift      — 1 − (1/r) Σ principal_cos²  (avg sin²θ_k)
                              ∈ [0, 1]; 0 = same span, 1 = orthogonal spans
  subspace_drift_F          — ‖E_cur − E_fut · R*‖_F² / max(‖E_cur‖_F², ‖E_fut‖_F²)
                              where R* aligns E_fut to E_cur via Procrustes
                              (rotation-aligned Euclidean drift on A_cur)
  vector_subspace_angle     — top-r principal cos² between span(V_cur) and span(V_fut)
                              (full-vector analog, n-dimensional)

Output: summary/per_block_v_cur_vs_v_fut_subspace/cells.csv + report.md
"""

import argparse
import csv
import json
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


def compute_v_fut_frame(A_cur, A_fut, A_sketch, state, rng_seed):
    """Re-run combined optimizer on extended block; return full rank-r frame."""
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
        M_gain=M_gain, active_r=RANK, rows_ref=N,
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
    return np.ascontiguousarray(np.asarray(V_score[:, :RANK], dtype=np.float64))


def orth(M, tol_factor=None):
    """Orthonormal column basis of M, dropping small singulars."""
    if M is None or M.size == 0:
        return np.zeros((M.shape[0] if M is not None else 0, 0))
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    if s.size == 0:
        return np.zeros((M.shape[0], 0))
    tol = (tol_factor or max(M.shape)) * np.finfo(np.float64).eps * s[0]
    keep = s > tol
    return np.ascontiguousarray(U[:, keep])


def principal_cos2(A, B):
    """Top-min(rA,rB) squared cosines of principal angles between col-spans."""
    if A.shape[1] == 0 or B.shape[1] == 0:
        return np.zeros(0)
    QA, QB = orth(A), orth(B)
    if QA.shape[1] == 0 or QB.shape[1] == 0:
        return np.zeros(0)
    sigmas = np.linalg.svd(QA.T @ QB, compute_uv=False)
    return np.clip(sigmas ** 2, 0.0, 1.0)


def procrustes_aligned_drift(E_cur, E_fut):
    """Optimal R minimizing ‖E_cur − E_fut R‖_F (orthogonal Procrustes).
    Returns (drift_norm_sq, mass_F_cur, mass_F_fut)
    where drift_norm_sq = ‖E_cur − E_fut R*‖_F² / max(‖E_cur‖_F², ‖E_fut‖_F²).
    """
    mass_cur = float(np.sum(E_cur * E_cur))
    mass_fut = float(np.sum(E_fut * E_fut))
    if mass_cur < 1e-30 and mass_fut < 1e-30:
        return 0.0, mass_cur, mass_fut
    # SVD of E_fut^T E_cur, R* = U V^T
    M = E_fut.T @ E_cur
    Up, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = Up @ Vt
    diff = E_cur - E_fut @ R
    drift_sq = float(np.sum(diff * diff))
    return drift_sq / max(mass_cur, mass_fut, 1e-30), mass_cur, mass_fut


def cos2_oracle_subspace(V, V_exact, r):
    """avg cos²(principal angle) between span(V) and span(V_exact[:, :r])."""
    cos2 = principal_cos2(V, V_exact[:, :r])
    if cos2.size == 0:
        return float("nan")
    return float(np.mean(cos2))


def normed(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-30 else v


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
    blocks = list(range(1, N_BLOCKS + 1))
    snapshots = stream_to_block(args, A, V_exact, np.float32, RANK, max(blocks), set(blocks))

    rows = []
    for b in blocks:
        if b not in snapshots:
            continue
        snap = snapshots[b]
        A_cur = snap["A_cur"].astype(np.float64)
        A_fut = snap["A_fut"]
        A_sketch = snap["A_sketch"]
        state = snap["state"]
        V_default = np.asarray(snap["V_default"], dtype=np.float64)  # (n, RANK)
        V_cur = V_default

        V_fut_raw = compute_v_fut_frame(A_cur.astype(np.float32), A_fut, A_sketch, state,
                                        rng_seed=seed + 7919 * b)
        V_fut = V_fut_raw

        # Vector-level slot-1 stats (for cross-check with previous probe)
        v_cur_1 = normed(V_cur[:, 0])
        v_fut_1 = normed(V_fut[:, 0])
        if float(np.dot(v_fut_1, v_cur_1)) < 0:
            v_fut_1 = -v_fut_1
        e_cur_1 = A_cur @ v_cur_1
        e_fut_1 = A_cur @ v_fut_1
        m_cur_1 = float(np.dot(e_cur_1, e_cur_1))
        m_fut_1 = float(np.dot(e_fut_1, e_fut_1))
        denom1 = max(m_cur_1, m_fut_1, 1e-30)
        cos_ee_1 = float(np.dot(e_cur_1, e_fut_1) / max(np.linalg.norm(e_cur_1) * np.linalg.norm(e_fut_1), 1e-30))
        drift_v1 = (m_cur_1 + m_fut_1 - 2 * cos_ee_1 * np.sqrt(max(m_cur_1 * m_fut_1, 0.0))) / denom1

        # Subspace-level on A_cur
        E_cur = A_cur @ V_cur     # (half_win, RANK)
        E_fut = A_cur @ V_fut

        drift_F, mass_F_cur, mass_F_fut = procrustes_aligned_drift(E_cur, E_fut)

        # principal cos² between row-projected col-spans on A_cur
        pc2_row = principal_cos2(E_cur, E_fut)
        # subspace angle drift: 1 - mean(cos²) = mean(sin²θ_k)
        if pc2_row.size:
            subspace_angle_drift_row = 1.0 - float(np.mean(pc2_row))
        else:
            subspace_angle_drift_row = float("nan")

        # mass shift (Frobenius)
        mass_shift_F = abs(mass_F_cur - mass_F_fut) / max(mass_F_cur, mass_F_fut, 1e-30)

        # full-vector subspace angle: principal cos² between span(V_cur) and span(V_fut)
        pc2_v = principal_cos2(V_cur, V_fut)
        subspace_angle_drift_vec = 1.0 - float(np.mean(pc2_v)) if pc2_v.size else float("nan")

        # cos² to V_exact subspace (signal-quality reference)
        sub_cos2_oracle = cos2_oracle_subspace(V_cur, V_exact, RANK)
        sub_cos2_oracle_fut = cos2_oracle_subspace(V_fut, V_exact, RANK)

        rows.append({
            "matrix": matrix, "seed": seed, "block": b,
            # subspace-level on A_cur
            "drift_F_subspace": drift_F,
            "mass_shift_F": mass_shift_F,
            "subspace_angle_drift_row": subspace_angle_drift_row,
            "principal_cos2_row_top1": float(pc2_row[0]) if pc2_row.size > 0 else float("nan"),
            "principal_cos2_row_top2": float(pc2_row[1]) if pc2_row.size > 1 else float("nan"),
            "mass_F_cur": mass_F_cur,
            "mass_F_fut": mass_F_fut,
            # subspace-level in ℝⁿ
            "subspace_angle_drift_vec": subspace_angle_drift_vec,
            "principal_cos2_vec_top1": float(pc2_v[0]) if pc2_v.size > 0 else float("nan"),
            "principal_cos2_vec_top2": float(pc2_v[1]) if pc2_v.size > 1 else float("nan"),
            # signal-quality
            "sub_cos2_oracle_cur": sub_cos2_oracle,
            "sub_cos2_oracle_fut": sub_cos2_oracle_fut,
            # vector-level slot-1 (for cross-check)
            "v1_drift": drift_v1,
            "v1_mass_cur": m_cur_1, "v1_mass_fut": m_fut_1,
            "v1_cos_ee": cos_ee_1,
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
    p.add_argument("--out-dir", default="summary/per_block_v_cur_vs_v_fut_subspace")
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
                print(f"  b={r['block']:2d}  "
                      f"drift_F={r['drift_F_subspace']:.3f}  "
                      f"angle_drift_row={r['subspace_angle_drift_row']:.3f}  "
                      f"mass_shift_F={r['mass_shift_F']:.3f}  "
                      f"angle_drift_vec={r['subspace_angle_drift_vec']:.3f}  "
                      f"v1_drift={r['v1_drift']:.3f}  "
                      f"sub_cos²_or={r['sub_cos2_oracle_cur']:.3f}")
            all_rows.extend(sub)

    keys = ["matrix", "seed", "block",
            "drift_F_subspace", "mass_shift_F", "subspace_angle_drift_row",
            "principal_cos2_row_top1", "principal_cos2_row_top2",
            "mass_F_cur", "mass_F_fut",
            "subspace_angle_drift_vec",
            "principal_cos2_vec_top1", "principal_cos2_vec_top2",
            "sub_cos2_oracle_cur", "sub_cos2_oracle_fut",
            "v1_drift", "v1_mass_cur", "v1_mass_fut", "v1_cos_ee"]
    cells_csv = os.path.join(args.out_dir, "cells.csv")
    with open(cells_csv, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in all_rows:
            f.write(",".join(_fmt(r[k]) for k in keys) + "\n")
    print(f"\nWrote {len(all_rows)} cells to {cells_csv}")


if __name__ == "__main__":
    main()
