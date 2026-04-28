"""Subspace alignment metrics (principal angles).

Reusable measurement utilities for the score-design line. The primary
entry point is `principal_angles(V_opt, V_oracle)` which returns the
principal angles between span(V_opt) and span(V_oracle) as both
cos² and radians (in non-decreasing angle order, i.e. cos² is
non-increasing).

Computed via SVD of V_opt^T V_oracle: the singular values of that
product are exactly cos(θ_i) for the r principal angles (Björck &
Golub, 1973). Inputs are assumed orthonormal Stiefel frames; we
re-orthonormalize via QR with a one-line guard so callers may pass
non-orthonormal matrices safely (e.g. raw V_exact[:, :r] columns).

This module is the home for INFRA-01 in the score-family workflow
backlog (see summary/overview/score_family_workflow.txt §5) and is
the dependency for INFRA-02 (Stiefel FD), INFRA-04 (trajectory
tracker), and the rank-r families FAM-01 / FAM-03.

Wired into r_sk_g_score.py and frob_hm3_score_diagnostic.py per-block
T2 tables alongside the existing pairwise `align_v?_proj` columns.
"""

from __future__ import annotations

import numpy as np


__all__ = [
    "principal_angles",
    "principal_angles_pair",
]


def _as_2d_orthonormal(V):
    """Coerce a vector or matrix to an orthonormal n×r Stiefel frame.

    - 1-D inputs become n×1.
    - Re-orthonormalize via QR; this is a no-op (up to sign) if V is
      already orthonormal.
    - Returns None if V is None or has no columns.
    """
    if V is None:
        return None
    A = np.asarray(V, dtype=np.float64)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.size == 0 or A.shape[1] == 0:
        return None
    Q, _ = np.linalg.qr(A)
    return Q[:, : A.shape[1]]


def principal_angles(V_opt, V_oracle):
    """Principal angles between span(V_opt) and span(V_oracle).

    Parameters
    ----------
    V_opt, V_oracle : array-like (n,) or (n, r)
        Two frames (assumed orthonormal; re-orthonormalized via QR if
        not). Both must have the same row dimension n. The number of
        columns may differ; the result has length min(r_opt, r_oracle).

    Returns
    -------
    cos2 : np.ndarray, shape (k,)
        cos² of each principal angle, sorted DESCENDING (so cos2[0] is
        the smallest angle, i.e. the most-aligned direction). Values
        clipped to [0, 1] before squaring.
    radians : np.ndarray, shape (k,)
        Principal angles in radians, sorted ASCENDING (radians[0] is
        the smallest angle).

    Notes
    -----
    For r = 1 this reduces to cos²(v_opt, v_oracle), matching the
    pairwise `align_v?_proj` invariant in §6 of diagnostic_toolkit.txt.
    """
    Q_opt = _as_2d_orthonormal(V_opt)
    Q_ora = _as_2d_orthonormal(V_oracle)
    if Q_opt is None or Q_ora is None:
        return np.array([]), np.array([])
    if Q_opt.shape[0] != Q_ora.shape[0]:
        raise ValueError(
            f"row-dim mismatch: V_opt has {Q_opt.shape[0]} rows, "
            f"V_oracle has {Q_ora.shape[0]} rows"
        )
    M = Q_opt.T @ Q_ora                         # shape (r_opt, r_oracle)
    s = np.linalg.svd(M, compute_uv=False)      # length min(r_opt, r_oracle)
    s = np.clip(s, 0.0, 1.0)                    # safety: numerical drift
    cos2 = s * s
    # SVD returns singular values in DESCENDING order, so cos2 is also
    # descending; the matching angles are ASCENDING.
    radians = np.arccos(s)
    return cos2, radians


def principal_angles_pair(v_opt, v_oracle):
    """Convenience for r=1: returns (cos2, radians) as scalars.

    Equivalent to principal_angles(v_opt, v_oracle) when both inputs
    are vectors, but always returns Python floats (one per scalar)
    so it slots cleanly into the existing per-row diagnostic table
    next to align_v?_proj.
    """
    cos2, rad = principal_angles(v_opt, v_oracle)
    if cos2.size == 0:
        return float("nan"), float("nan")
    return float(cos2[0]), float(rad[0])


# --------------------------------------------------------------------------
# Self-test (run as: python -m subspace_metrics  OR  python subspace_metrics.py)
# --------------------------------------------------------------------------


def _self_test():
    """Sanity tests:
       (1) V vs V·R for random orthogonal R → all cos² ≈ 1
       (2) r=1 case matches scalar cos²
       (3) two random orthonormal frames → cos² ∈ [0, 1]
    """
    rng = np.random.default_rng(0)
    n, r = 64, 4

    # (1) orthogonal-rotation invariance
    V, _ = np.linalg.qr(rng.standard_normal((n, r)))
    R, _ = np.linalg.qr(rng.standard_normal((r, r)))
    cos2, rad = principal_angles(V, V @ R)
    assert cos2.shape == (r,)
    assert np.allclose(cos2, 1.0, atol=1e-10), f"rotation invariance failed: cos2={cos2}"
    assert np.allclose(rad, 0.0, atol=1e-5), f"rotation invariance failed: rad={rad}"

    # (2) r=1 matches scalar cos²
    v = rng.standard_normal(n); v /= np.linalg.norm(v)
    w = rng.standard_normal(n); w /= np.linalg.norm(w)
    cos2, rad = principal_angles(v, w)
    expected = float(np.dot(v, w) ** 2)
    assert cos2.shape == (1,)
    assert abs(cos2[0] - expected) < 1e-12, f"r=1 mismatch: got {cos2[0]}, expected {expected}"

    # (3) two random orthonormal frames behave
    V1, _ = np.linalg.qr(rng.standard_normal((n, r)))
    V2, _ = np.linalg.qr(rng.standard_normal((n, r)))
    cos2, rad = principal_angles(V1, V2)
    assert cos2.shape == (r,)
    assert np.all(cos2 >= 0.0) and np.all(cos2 <= 1.0 + 1e-12)
    # descending-cos / ascending-radian property
    assert np.all(np.diff(cos2) <= 1e-12)
    assert np.all(np.diff(rad) >= -1e-12)

    # (4) different number of columns
    V1, _ = np.linalg.qr(rng.standard_normal((n, 3)))
    V2, _ = np.linalg.qr(rng.standard_normal((n, 5)))
    cos2, rad = principal_angles(V1, V2)
    assert cos2.shape == (3,), f"expected len 3, got {cos2.shape}"

    # (5) non-orthonormal input → QR guard rescues
    A = rng.standard_normal((n, r)) * 5.0       # arbitrary scale
    cos2, _ = principal_angles(A, A)
    assert np.allclose(cos2, 1.0, atol=1e-10)

    # (6) scalar pair convenience
    c2, r0 = principal_angles_pair(v, w)
    assert abs(c2 - expected) < 1e-12
    assert abs(np.cos(r0) ** 2 - expected) < 1e-10

    print("subspace_metrics self-test: OK")


if __name__ == "__main__":
    _self_test()
