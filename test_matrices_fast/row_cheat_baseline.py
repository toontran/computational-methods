"""Rank-r row-cheat baseline (INFRA-03).

Generalizes the slot-2 ``hm_triplet_raw_best`` candidate (the row-of-A_fut
"cheat") to a rank-r frame baseline so the exploitability invariant
(``score(V_oracle) >= score(V_rowcheat)``) survives the rank-r lift.

The single-vector row-cheat at slot 2 is the v that maximizes the raw
HM3(||A_sketch v||^2, ||A_cur v||^2, ||A_fut v||^2): when A_fut has a single
dominant row, the row direction wins. Its rank-r natural extension is to take
the r rows of A_fut with the largest squared norm, stack them as the columns
of an n x r candidate frame, and orthonormalize via QR. At r=1 this reduces
to the direction of the row of A_fut with the largest norm.

The companion frame-level scorers (``frame_score_S6`` / ``frame_score_S6_GM``)
implement the rank-r lift of S6 / S6_GM from
``score_design_overview.txt`` Section 5:

    u_X(V) = ||A_X V||_F^2 / ||A_X||_F^2     for V in R^{n x r}
    score(V) = HM3(u_sk, u_g1, u_g2)         (sketch present)
    score(V) = HM2(u_g1, u_g2)               (block-1 fall-through)

These scorers are SUBSPACE functions (invariant under right multiplication by
O in O(r)) so they evaluate any orthonormal frame consistently.

Acceptance check at rank r (T2 STOP rule):
    score(V_oracle_frame) >= score(V_rowcheat_frame)  on every probed block

If a value-only score puts the row-cheat frame above the oracle frame on any
block, the score is row-exploitable and must be killed before T3.
"""

import numpy as np


__all__ = [
    "top_r_rows_frame",
    "frame_score_S6",
    "frame_score_S6_GM",
    "oracle_frame_proj",
]


def top_r_rows_frame(A_fut, r):
    """Top-r rows of ``A_fut`` by squared norm, stacked as columns and
    orthonormalized via QR. Returns ``None`` when the frame collapses
    (no non-zero rows or rank deficient after QR).
    """
    if A_fut is None:
        return None
    A = np.asarray(A_fut, dtype=np.float64)
    if A.size == 0:
        return None
    r = int(r)
    if r <= 0:
        return None
    row_sq = np.sum(A * A, axis=1)
    if row_sq.size == 0 or not np.any(row_sq > 0.0):
        return None
    # Stable sort for reproducibility.
    order = np.argsort(-row_sq, kind="stable")
    pick = order[: min(r, order.size)]
    M = A[pick, :].T  # (n, r_eff)
    keep = [j for j in range(M.shape[1]) if np.linalg.norm(M[:, j]) > 1e-30]
    if not keep:
        return None
    M = M[:, keep]
    Q, R = np.linalg.qr(M)
    diagR = np.abs(np.diag(R))
    if diagR.size == 0:
        return None
    tol = max(diagR.max() * 1e-12, 1e-30)
    keep_cols = np.where(diagR > tol)[0]
    if keep_cols.size == 0:
        return None
    return np.ascontiguousarray(Q[:, keep_cols])


def _frame_unit_share(A_X, V, X_F2):
    """``u_X(V) = ||A_X V||_F^2 / ||A_X||_F^2`` for V in R^{n x r}."""
    eps = 1e-30
    if A_X is None or V is None:
        return float("nan")
    A = np.asarray(A_X, dtype=np.float64)
    if A.size == 0 or float(X_F2) <= eps:
        return float("nan")
    Y = A @ V
    return float(np.sum(Y * Y) / float(X_F2))


def frame_score_S6(A_sketch, A_cur, A_fut, sk_F2_low, cur_F2, fut_F2, V):
    """Rank-r generalization of the S6 score (HM3 aggregator).

        score(V) = HM3(u_sk, u_g1, u_g2)        sketch present
        score(V) = HM2(u_g1, u_g2)              block-1 fall-through

    Returns ``{"score", "u_sk", "u_g1", "u_g2"}``. Score is 0.0 when any
    required ``u`` is non-positive; sketch is ignored when ``A_sketch`` is
    ``None`` or ``sk_F2_low`` <= 0.
    """
    eps = 1e-30
    u_g1 = _frame_unit_share(A_cur, V, cur_F2)
    u_g2 = _frame_unit_share(A_fut, V, fut_F2)
    have_sketch = (
        A_sketch is not None
        and sk_F2_low is not None
        and float(sk_F2_low) > eps
    )
    u_sk = _frame_unit_share(A_sketch, V, sk_F2_low) if have_sketch else float("nan")
    if u_g1 != u_g1 or u_g2 != u_g2 or u_g1 <= eps or u_g2 <= eps:
        return {"score": 0.0, "u_sk": u_sk, "u_g1": u_g1, "u_g2": u_g2}
    if have_sketch:
        if u_sk != u_sk or u_sk <= eps:
            return {"score": 0.0, "u_sk": u_sk, "u_g1": u_g1, "u_g2": u_g2}
        score = 3.0 / (1.0 / u_sk + 1.0 / u_g1 + 1.0 / u_g2)
    else:
        score = 2.0 / (1.0 / u_g1 + 1.0 / u_g2)
    return {"score": float(score), "u_sk": u_sk, "u_g1": u_g1, "u_g2": u_g2}


def frame_score_S6_GM(A_sketch, A_cur, A_fut, sk_F2_low, cur_F2, fut_F2, V):
    """Rank-r generalization of the S6_GM aggregator (AB-01).

        score(V) = (u_sk * u_g1 * u_g2)^(1/3)   sketch present
        score(V) = (u_g1 * u_g2)^(1/2)          block-1 fall-through
    """
    eps = 1e-30
    u_g1 = _frame_unit_share(A_cur, V, cur_F2)
    u_g2 = _frame_unit_share(A_fut, V, fut_F2)
    have_sketch = (
        A_sketch is not None
        and sk_F2_low is not None
        and float(sk_F2_low) > eps
    )
    u_sk = _frame_unit_share(A_sketch, V, sk_F2_low) if have_sketch else float("nan")
    if u_g1 != u_g1 or u_g2 != u_g2 or u_g1 <= eps or u_g2 <= eps:
        return {"score": 0.0, "u_sk": u_sk, "u_g1": u_g1, "u_g2": u_g2}
    if have_sketch:
        if u_sk != u_sk or u_sk <= eps:
            return {"score": 0.0, "u_sk": u_sk, "u_g1": u_g1, "u_g2": u_g2}
        score = float((u_sk * u_g1 * u_g2) ** (1.0 / 3.0))
    else:
        score = float((u_g1 * u_g2) ** 0.5)
    return {"score": float(score), "u_sk": u_sk, "u_g1": u_g1, "u_g2": u_g2}


def oracle_frame_proj(V_exact, B_union, r):
    """Project the top-r oracle right singular vectors into ``B_union``
    column-by-column, then orthonormalize via QR. Mirrors how
    ``oracle_v?_proj`` is built in ``r_sk_g_score.analyze_block`` but stacks
    columns into a frame. Used as the rank-r oracle reference for the
    exploitability check (T2 STOP rule).
    """
    if V_exact is None or B_union is None:
        return None
    Ve = np.asarray(V_exact, dtype=np.float64)
    if Ve.size == 0 or B_union.size == 0:
        return None
    r = int(r)
    if r <= 0 or Ve.shape[1] < 1:
        return None
    cols = []
    for j in range(min(r, Ve.shape[1])):
        v = Ve[:, j]
        nv = float(np.linalg.norm(v))
        if nv <= 1e-30:
            continue
        v = v / nv
        p = B_union @ (B_union.T @ v)
        np_norm = float(np.linalg.norm(p))
        if np_norm <= 1e-30:
            continue
        cols.append(p / np_norm)
    if not cols:
        return None
    M = np.stack(cols, axis=1)
    Q, R = np.linalg.qr(M)
    diagR = np.abs(np.diag(R))
    if diagR.size == 0:
        return None
    tol = max(diagR.max() * 1e-12, 1e-30)
    keep_cols = np.where(diagR > tol)[0]
    if keep_cols.size == 0:
        return None
    return np.ascontiguousarray(Q[:, keep_cols])
