"""Stiefel finite-difference vs analytic gradient check (INFRA-02).

Column-wise FD with TANGENT-SPACE PROJECTION for the rank-r (Stiefel) lift of
S6 / S6_GM (and any other Stiefel-valued frame score). The single-vector
``--gradient-check`` in ``r_sk_g_score.py`` does not generalize at rank r; this
module fills that gap (toolkit §8 (d) / workflow §5 [INFRA-02]).

Stiefel manifold and tangent space
----------------------------------

    St(n, r) = {V in R^{n x r} : V^T V = I_r}.
    T_V St   = {Z in R^{n x r} : V^T Z + Z^T V = 0}      (i.e. V^T Z is skew).

Tangent projection (orthogonal in the canonical / Frobenius inner product)::

    P_V(G) = G - V * sym(V^T G),      sym(X) = (X + X^T)/2.

Equivalently ``G - V (V^T G + G^T V)/2``.  This is the projection into T_V St
that the Stiefel-ascent step uses; the FD check must compare ``trace(G_T^T Z)``
against the symmetric-difference quotient along directions Z that already lie
in T_V St.

Retraction (used for the FD probe)::

    R_V(W) = qr(V + W).Q[:, :r]          (QR retraction, first-order valid)

For an FD direction Z in T_V St,

    fd  = (f(R_V(eps Z_j)) - f(R_V(-eps Z_j))) / (2 * eps)
    ana = trace(G_T^T Z)
    rel = |fd - ana| / max(|fd|, |ana|, eps_safety)

where Z_j = P_V(E_j), and E_j has non-zero entries only in column j. The
projection can spill into other columns, which is expected: the perturbation
seed is column-wise, while the tested direction is a valid Stiefel tangent.

Subspace-function note: S6 / S6_GM are O(r)-invariant, so the Euclidean
gradient G already has its vertical (V * skew) component zero, and ``P_V(G) = G``
modulo numerical drift. We still apply the projection: it is the contract
expected by the ascent algorithm and the test must not be fragile to the
implementation choosing a non-tangent gradient.

Acceptance bar
--------------

(1) Trace-form sanity: f(V) = trace(V^T M V) with M = A_cur^T A_cur. Analytic
    grad = 2 M V. FD must agree to rel < 1e-7. If this fails, the FD-with-
    tangent-projection harness is broken, NOT the score; halt before testing
    HM3/GM3.

(2) Score gauntlet: rel < 1e-7 on every (matrix, block, r, variant) cell at
    float64. Matrices: mixed-tail-sharp / static-cex / diffuse-diffuse;
    blocks: 1, 2, 12, 31; ranks r in {2, 3, 4}; variants S6 and S6_GM.

CLI
---

    python stiefel_grad_check.py             # full gauntlet + sanity row
    python stiefel_grad_check.py --quick     # 1 matrix x 2 blocks (smoke)

Outputs (CLI mode):

    summary/infra_stiefel_fd_gradient_check/gauntlet.txt
    summary/infra_stiefel_fd_gradient_check/synthesis.md

Public API (module mode)
------------------------

    stiefel_fd_check(score_value_grad_fn, V, n_directions=8, eps=1e-5, rng=None)
        -> {"per_dir": [(ana, fd, rel, column), ...], "max_rel": float, ...}

    frame_S6_value_grad(A_sketch, A_cur, A_fut, sk_F2_low, cur_F2, fut_F2, V)
        -> (score: float, grad: ndarray of shape (n, r))
    frame_S6_GM_value_grad(...)  -> (score, grad)

These value-and-grad fns are reusable by FAM-01 / FAM-03 once the grad is
trusted.

Backlog: summary/overview/score_family_workflow.txt §5 [INFRA-02]
Toolkit gap closed: summary/overview/diagnostic_toolkit.txt §8 (d)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable

import numpy as np

# --------------------------------------------------------------------------
# Stiefel utilities
# --------------------------------------------------------------------------


def sym(X: np.ndarray) -> np.ndarray:
    """Symmetric part: (X + X^T) / 2."""
    return 0.5 * (X + X.T)


def stiefel_tangent_project(V: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Project G in R^{n x r} into the Stiefel tangent space at V.

    P_V(G) = G - V sym(V^T G).

    Equivalent to G - 0.5 * V (V^T G + G^T V).
    """
    VtG = V.T @ G
    return G - V @ sym(VtG)


def qr_retract(V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """QR retraction at V along W: R_V(W) = Q where (Q, R) = qr(V + W).

    Sign-fix: enforce R_diag >= 0 so the retraction is unique mod Stiefel
    representation; this avoids occasional sign flips that would produce
    spurious O(1) FD errors at otherwise-tiny eps.

    NOTE: QR retraction has O(eps^2) curvature that does NOT cancel under
    central differences (the retraction is not symmetric in W). For the FD
    check use ``polar_retract`` (symmetric, central-FD friendly) — the QR
    retraction is kept here because it is what the actual Stiefel-ascent
    step uses, and tests exist that pin its first-order behavior.
    """
    Q, R = np.linalg.qr(V + W)
    diag = np.diag(R)
    s = np.where(diag >= 0.0, 1.0, -1.0)
    return Q * s  # broadcast over rows


def polar_retract(V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Polar retraction at V along W: R_V(W) = (V + W) ((V+W)^T (V+W))^{-1/2}.

    Symmetric in W up to higher-order curvature, so central FD against the
    polar-retracted score has truncation O(eps^2) — the right choice for the
    FD harness. Computed via SVD: write Y = V + W = U S X^T, then
    Y (Y^T Y)^{-1/2} = U X^T.
    """
    Y = V + W
    U, _, Xt = np.linalg.svd(Y, full_matrices=False)
    return U @ Xt


def random_tangent(V: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample a unit-Frobenius tangent direction Z in T_V St.

    Strategy: draw G ~ N(0, I), project into T_V St, normalize. Uniform on the
    tangent unit-sphere in the canonical inner product.
    """
    n, r = V.shape
    G = rng.standard_normal((n, r))
    Z = stiefel_tangent_project(V, G)
    nrm = float(np.linalg.norm(Z))
    if nrm <= 1e-30:
        # Re-draw if degenerate (essentially impossible for n > r).
        return random_tangent(V, rng)
    return Z / nrm


def random_column_tangent(
    V: np.ndarray, column: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample a unit tangent direction from a single-column ambient seed.

    Draw E with non-zero entries only in ``column``, project E into T_V St,
    then normalize. This is the column-wise FD required by INFRA-02.
    """
    n, r = V.shape
    if column < 0 or column >= r:
        raise ValueError(f"column {column} out of range for r={r}")
    E = np.zeros((n, r), dtype=np.float64)
    E[:, column] = rng.standard_normal(n)
    Z = stiefel_tangent_project(V, E)
    nrm = float(np.linalg.norm(Z))
    if nrm <= 1e-30:
        return random_column_tangent(V, column, rng)
    return Z / nrm


def tangent_residual(V: np.ndarray, Z: np.ndarray) -> float:
    """Frobenius-norm of (V^T Z + Z^T V), should be ~ 0 for tangent Z."""
    M = V.T @ Z + Z.T @ V
    return float(np.linalg.norm(M))


# --------------------------------------------------------------------------
# FD harness
# --------------------------------------------------------------------------


def stiefel_fd_check(
    score_value_grad_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
    V: np.ndarray,
    n_directions: int = 8,
    eps: float = 1e-5,
    rng: np.random.Generator | None = None,
    retraction: str = "polar",
    direction_mode: str = "column",
) -> dict:
    """Compare analytic gradient vs FD along tangent directions in T_V St.

    Parameters
    ----------
    score_value_grad_fn : V -> (score: float, grad: ndarray (n, r))
        Returns the score value and its UNCONSTRAINED Euclidean gradient
        w.r.t. the entries of V (the function's natural grad). The harness
        applies the Stiefel tangent projection internally before the FD compare.
    V : ndarray of shape (n, r)
        A point on St(n, r). Will be QR-orthonormalized as a guard.
    n_directions : int
        Number of random tangent directions to test.
    eps : float
        FD step size. With ``polar`` retraction, central-difference
        truncation error is O(eps^2); ``eps=1e-5`` gives ~1e-10 relative
        accuracy, well below the 1e-7 acceptance bar.
    rng : np.random.Generator
        For reproducibility. Defaults to np.random.default_rng(0).
    retraction : {"polar", "qr", "none"}
        ``polar`` (default): symmetric in W to higher order; central FD has
            truncation O(eps^2). The right choice for the FD harness.
        ``qr``: matches the in-algorithm retraction; central FD has only
            O(eps) truncation due to QR's directional asymmetry — useful
            only as a sanity that the algorithm's actual ascent step
            agrees to first order.
        ``none``: Euclidean FD with V + eps*Z (no projection back to St).
            Score function must be defined off-Stiefel (S6 / S6_GM are);
            central-FD truncation is O(eps^2).
    direction_mode : {"column", "random"}
        ``column`` seeds each FD direction in one ambient column before tangent
        projection. ``random`` uses full-frame ambient random seeds.

    Returns
    -------
    dict with keys:
        "per_dir": list of (ana, fd, rel) tuples, length n_directions
        "max_rel": float, the largest relative error across directions
        "score0":  float, score at V (sanity)
        "grad_tan_resid": float, residual of (G - P_V(G)) (~0 for O(r)-invariant)
        "tangent_check": float, max |V^T Z + Z^T V| across tested Z (~0)
        "retraction": str, the retraction used.
        "direction_mode": str, the tangent seed mode.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    V = np.asarray(V, dtype=np.float64)
    if V.ndim != 2:
        raise ValueError(f"V must be 2-D (n, r); got shape {V.shape}")
    # Re-orthonormalize: harness must not be sensitive to caller's input precision.
    Q, _ = np.linalg.qr(V)
    V = Q[:, : V.shape[1]]

    score0, G = score_value_grad_fn(V)
    G = np.asarray(G, dtype=np.float64)
    if G.shape != V.shape:
        raise ValueError(f"grad shape {G.shape} != V shape {V.shape}")

    G_T = stiefel_tangent_project(V, G)
    # For O(r)-invariant scores, G itself is already tangent up to vertical
    # zero; record the residual ‖G - G_T‖_F as a sanity number.
    G_norm = float(np.linalg.norm(G))
    grad_tan_resid = float(np.linalg.norm(G - G_T) / max(G_norm, 1e-30))

    if retraction == "polar":
        retract_fn = polar_retract
    elif retraction == "qr":
        retract_fn = qr_retract
    elif retraction == "none":
        retract_fn = lambda V_, W_: V_ + W_
    else:
        raise ValueError(f"unknown retraction {retraction!r}; expected polar/qr/none")

    per_dir = []
    tan_resid_max = 0.0
    for i_dir in range(n_directions):
        if direction_mode == "column":
            column = i_dir % V.shape[1]
            Z = random_column_tangent(V, column, rng)
        elif direction_mode == "random":
            column = None
            Z = random_tangent(V, rng)
        else:
            raise ValueError(
                f"unknown direction_mode {direction_mode!r}; expected column/random"
            )
        tan_resid_max = max(tan_resid_max, tangent_residual(V, Z))
        Vp = retract_fn(V, eps * Z)
        Vm = retract_fn(V, -eps * Z)
        sp, _ = score_value_grad_fn(Vp)
        sm, _ = score_value_grad_fn(Vm)
        fd = (sp - sm) / (2.0 * eps)
        ana = float(np.sum(G_T * Z))           # trace(G_T^T Z)
        denom = max(abs(fd), abs(ana), 1e-30)
        rel = abs(fd - ana) / denom
        per_dir.append((float(ana), float(fd), float(rel), column))

    max_rel = max((r for _, _, r, _ in per_dir), default=float("nan"))
    return {
        "per_dir": per_dir,
        "max_rel": float(max_rel),
        "score0": float(score0),
        "grad_tan_resid": grad_tan_resid,
        "tangent_check": float(tan_resid_max),
        "retraction": retraction,
        "direction_mode": direction_mode,
    }


# --------------------------------------------------------------------------
# Frame-level value-and-grad for the rank-r lifts
# --------------------------------------------------------------------------

_EPS = 1e-30


def _u_value_grad(A_X: np.ndarray | None, V: np.ndarray, W_X: float):
    """u_X(V) = ||A_X V||_F^2 / W_X and its Euclidean gradient w.r.t. V.

    grad_V u_X = 2 (A_X^T A_X) V / W_X = 2 A_X^T (A_X V) / W_X.

    Returns (u, grad) with u = NaN and grad = zeros if A_X is unavailable
    (signals "skip this term" to the caller).
    """
    if A_X is None or A_X.size == 0 or W_X is None or float(W_X) <= _EPS:
        return float("nan"), np.zeros_like(V)
    A = np.asarray(A_X, dtype=np.float64)
    Y = A @ V                                      # (m, r)
    u = float(np.sum(Y * Y) / float(W_X))
    grad = (2.0 / float(W_X)) * (A.T @ Y)          # (n, r)
    return u, grad


def frame_S6_value_grad(A_sketch, A_cur, A_fut, sk_F2_low, cur_F2, fut_F2, V):
    """Rank-r S6 score and its Euclidean gradient ∂S/∂V.

    Aggregator: HM3 (or HM2 in block-1 fall-through). Score is 0 with
    grad 0 when any required u <= 0.

        S = 3 / (1/u_sk + 1/u_g1 + 1/u_g2)              (sketch present)
        S = 2 / (1/u_g1 + 1/u_g2)                       (block 1)
        ∂S/∂V = (S^2 / k) Σ (1/u_X^2) ∂u_X/∂V           (k = 3 or 2)
    """
    V = np.asarray(V, dtype=np.float64)

    have_sketch = (
        A_sketch is not None
        and sk_F2_low is not None
        and float(sk_F2_low) > _EPS
        and np.asarray(A_sketch).size > 0
    )

    u_g1, du_g1 = _u_value_grad(A_cur, V, cur_F2)
    u_g2, du_g2 = _u_value_grad(A_fut, V, fut_F2)
    if u_g1 != u_g1 or u_g2 != u_g2 or u_g1 <= _EPS or u_g2 <= _EPS:
        return 0.0, np.zeros_like(V)

    if have_sketch:
        u_sk, du_sk = _u_value_grad(A_sketch, V, sk_F2_low)
        if u_sk != u_sk or u_sk <= _EPS:
            return 0.0, np.zeros_like(V)
        D = 1.0 / u_sk + 1.0 / u_g1 + 1.0 / u_g2
        S = 3.0 / D
        coef = (S * S) / 3.0
        grad = coef * (
            (1.0 / (u_sk * u_sk)) * du_sk
            + (1.0 / (u_g1 * u_g1)) * du_g1
            + (1.0 / (u_g2 * u_g2)) * du_g2
        )
        return float(S), grad

    D = 1.0 / u_g1 + 1.0 / u_g2
    S = 2.0 / D
    coef = (S * S) / 2.0
    grad = coef * (
        (1.0 / (u_g1 * u_g1)) * du_g1
        + (1.0 / (u_g2 * u_g2)) * du_g2
    )
    return float(S), grad


def frame_S6_GM_value_grad(A_sketch, A_cur, A_fut, sk_F2_low, cur_F2, fut_F2, V):
    """Rank-r S6_GM score and its Euclidean gradient ∂S/∂V.

    Aggregator: GM3 (or GM2 in block-1 fall-through).

        S = (u_sk * u_g1 * u_g2)^(1/3)                  (sketch present)
        S = (u_g1 * u_g2)^(1/2)                         (block 1)
        ∂S/∂V = (S / k) Σ (1/u_X) ∂u_X/∂V               (k = 3 or 2)

    Derivation: log S = (1/k) Σ log u_X, so
        (1/S) ∂S = (1/k) Σ (1/u_X) ∂u_X
        ∂S      = (S/k) Σ (1/u_X) ∂u_X.
    """
    V = np.asarray(V, dtype=np.float64)

    have_sketch = (
        A_sketch is not None
        and sk_F2_low is not None
        and float(sk_F2_low) > _EPS
        and np.asarray(A_sketch).size > 0
    )

    u_g1, du_g1 = _u_value_grad(A_cur, V, cur_F2)
    u_g2, du_g2 = _u_value_grad(A_fut, V, fut_F2)
    if u_g1 != u_g1 or u_g2 != u_g2 or u_g1 <= _EPS or u_g2 <= _EPS:
        return 0.0, np.zeros_like(V)

    if have_sketch:
        u_sk, du_sk = _u_value_grad(A_sketch, V, sk_F2_low)
        if u_sk != u_sk or u_sk <= _EPS:
            return 0.0, np.zeros_like(V)
        S = (u_sk * u_g1 * u_g2) ** (1.0 / 3.0)
        coef = S / 3.0
        grad = coef * (
            (1.0 / u_sk) * du_sk
            + (1.0 / u_g1) * du_g1
            + (1.0 / u_g2) * du_g2
        )
        return float(S), grad

    S = (u_g1 * u_g2) ** 0.5
    coef = S / 2.0
    grad = coef * (
        (1.0 / u_g1) * du_g1
        + (1.0 / u_g2) * du_g2
    )
    return float(S), grad


# --------------------------------------------------------------------------
# Sanity trace-form (for testing the harness itself before HM3/GM3)
# --------------------------------------------------------------------------


def trace_form_value_grad(M_sym: np.ndarray, V: np.ndarray):
    """f(V) = trace(V^T M V), grad = 2 M V. M MUST be symmetric.

    A known-correct closed-form check: if FD vs ana fails this on a Stiefel
    point (rel >= 1e-9), the FD harness is broken — NOT the HM3/GM3 grads.
    """
    V = np.asarray(V, dtype=np.float64)
    MV = M_sym @ V
    f = float(np.sum(V * MV))                     # trace(V^T M V)
    grad = 2.0 * MV
    return f, grad


# --------------------------------------------------------------------------
# Gauntlet driver: matrices x blocks x ranks x variants
# --------------------------------------------------------------------------


def _build_args_namespace(half_win=32, n=1024, rank=2, seed=0,
                          dtype="float64", preset="fast",
                          v_type="rand", shuffle_rows=True, row_shuffle_seed=0):
    """SimpleNamespace mimicking the args object stream_to_block expects.

    Mirrors the defaults in r_sk_g_score.py:parse_args (the bench code path).
    All hyperparams that influence the matrix realization or streaming carry
    are pinned to their published defaults; only the per-test knobs (rank,
    half_win) are exposed.
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        # Streaming / state.
        n=int(n), half_win=int(half_win), rank=int(rank), seed=int(seed),
        dtype=dtype, preset=preset, shuffle_rows=shuffle_rows,
        row_shuffle_seed=int(row_shuffle_seed), old_memory_size=int(half_win),
        # Matrix-input knobs (defaults from r_sk_g_score.py / plateau_width_probe).
        r_sig=2, alpha_sig=0.003, alpha_tail=0.0145,
        tail_scale=0.99, sigma1=0.991, v_type=v_type,
        # Optimizer knobs (used only by stream_to_block; do not affect the
        # FD check itself, only the per-block (A_sketch, A_cur, A_fut) trio).
        q0=8, qmax=48, krylov_depth=2, residual_tol=0.01,
        expansion_maxit=8, num_restarts=3, maxit=120, tol=1e-8,
        post_expansion_maxit=80, patience=5, patience_rel_tol=1e-5,
        union_maxit=120, union_tol=1e-9, union_random_starts=24,
    )


def _build_per_block_trio(A, V_exact, args, block_id):
    """Run the streaming carry forward to ``block_id`` and return the
    (A_sketch, A_cur, A_fut, sk_F2_low, cur_F2, fut_F2, B_union) trio used
    by the rank-r score.

    Mirrors r_sk_g_score.gradient_check's setup.
    """
    import cex_restricted_space_probe as probe   # local import (slow module)
    from hmean_evidence_score import per_block_constants, stream_to_block

    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    snaps = stream_to_block(
        args, A, V_exact, work_dtype, int(args.rank), block_id, {block_id}
    )
    snap = snaps[block_id]

    A_sketch = snap["A_sketch"] if snap["A_sketch"].size else None
    A_cur = snap["A_cur"]
    A_fut = snap["A_fut"]

    consts = per_block_constants(A, block_id, args.half_win)
    cur_F2 = float(consts["cur_F2"])
    fut_F2 = float(consts["fut_F2"])
    sk_F2_low = (
        float(np.sum(np.asarray(A_sketch, dtype=np.float64) ** 2))
        if A_sketch is not None else 0.0
    )

    return {
        "A_sketch": A_sketch,
        "A_cur": np.asarray(A_cur, dtype=np.float64),
        "A_fut": np.asarray(A_fut, dtype=np.float64),
        "sk_F2_low": sk_F2_low,
        "cur_F2": cur_F2,
        "fut_F2": fut_F2,
    }


def _seed_stiefel_frame(n: int, r: int, rng: np.random.Generator) -> np.ndarray:
    """Random orthonormal V in St(n, r) for the FD probe."""
    G = rng.standard_normal((n, r))
    Q, _ = np.linalg.qr(G)
    return Q[:, :r]


def run_gauntlet(matrices, blocks, ranks, variants, n=1024, half_win=32,
                 n_directions=8, eps=1e-5, sanity_eps=1e-5,
                 rng_seed=0, log=print):
    """Run the gauntlet across (matrix x block x rank x variant) cells.

    Returns a list of result dicts and prints per-cell rel_err.
    """
    import cex_restricted_space_probe as probe   # heavy import

    rng = np.random.default_rng(rng_seed)
    results = []

    for matrix in matrices:
        log(f"\n== matrix = {matrix} ==")
        # Build A once per matrix; reuse across blocks/ranks.
        # rank arg only affects the streaming carry width; we use max(ranks)
        # so the carry can produce A_sketch that supports the largest probe.
        target_rank = max(ranks)
        target_block = max(blocks)
        args = _build_args_namespace(
            half_win=half_win, n=n, rank=target_rank, seed=0,
        )
        A, V_exact, _, _ = probe.generate_matrix_input(
            matrix=matrix, n=args.n, preset=args.preset, seed=args.seed,
            r_sig=args.r_sig, alpha_sig=args.alpha_sig,
            alpha_tail=args.alpha_tail, tail_scale=args.tail_scale,
            sigma1=args.sigma1, v_type=args.v_type,
            shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
        )
        A = np.asarray(A, dtype=np.float64)
        V_exact = np.asarray(V_exact, dtype=np.float64)

        # Stream once to the largest block; pull all needed blocks' snapshots
        # in a single forward pass for efficiency.
        from hmean_evidence_score import per_block_constants, stream_to_block
        work_dtype = np.float32 if args.dtype == "float32" else np.float64
        snaps = stream_to_block(
            args, A, V_exact, work_dtype, int(args.rank), target_block,
            set(int(b) for b in blocks),
        )

        for block_id in blocks:
            if block_id not in snaps:
                log(f"  block {block_id}: no snapshot (skipping)")
                continue
            snap = snaps[block_id]
            A_sketch = snap["A_sketch"] if snap["A_sketch"].size else None
            A_cur = np.asarray(snap["A_cur"], dtype=np.float64)
            A_fut = np.asarray(snap["A_fut"], dtype=np.float64)
            consts = per_block_constants(A, block_id, args.half_win)
            cur_F2 = float(consts["cur_F2"])
            fut_F2 = float(consts["fut_F2"])
            sk_F2_low = (
                float(np.sum(np.asarray(A_sketch, dtype=np.float64) ** 2))
                if A_sketch is not None else 0.0
            )

            # Build symmetric M for the trace-form sanity (uses A_cur for the
            # block; well-conditioned, non-degenerate).
            M_sym = A_cur.T @ A_cur

            for r in ranks:
                V_seed = _seed_stiefel_frame(args.n, r, rng)

                # (1) Sanity trace-form first; halt the cell if it fails.
                sanity = stiefel_fd_check(
                    lambda V_: trace_form_value_grad(M_sym, V_),
                    V_seed, n_directions=n_directions, eps=sanity_eps,
                    rng=np.random.default_rng(rng_seed + 1),
                )

                cell_rows = []
                for variant in variants:
                    if variant == "S6":
                        score_fn = (lambda V_: frame_S6_value_grad(
                            A_sketch, A_cur, A_fut,
                            sk_F2_low, cur_F2, fut_F2, V_,
                        ))
                    elif variant == "S6_GM":
                        score_fn = (lambda V_: frame_S6_GM_value_grad(
                            A_sketch, A_cur, A_fut,
                            sk_F2_low, cur_F2, fut_F2, V_,
                        ))
                    else:
                        raise ValueError(f"unknown variant {variant!r}")

                    # Reset RNG per (variant) so directions are reproducible
                    # AND comparable across variants for the same V_seed.
                    res = stiefel_fd_check(
                        score_fn, V_seed,
                        n_directions=n_directions, eps=eps,
                        rng=np.random.default_rng(rng_seed + 100),
                    )
                    cell_rows.append({
                        "matrix": matrix, "block": int(block_id), "rank": int(r),
                        "variant": variant,
                        "score": res["score0"],
                        "max_rel": res["max_rel"],
                        "tan_resid": res["tangent_check"],
                        "grad_tan_resid": res["grad_tan_resid"],
                        "sanity_max_rel": sanity["max_rel"],
                    })
                    log(
                        f"  [{matrix} b{block_id:>2} r={r} {variant:<6}] "
                        f"score={res['score0']: .4e}  "
                        f"max_rel={res['max_rel']:.2e}  "
                        f"sanity_rel={sanity['max_rel']:.2e}  "
                        f"grad_tan_resid={res['grad_tan_resid']:.2e}"
                    )
                results.extend(cell_rows)
    return results


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def write_gauntlet_txt(path: str, results: list[dict]):
    """Per-cell rel_err table sorted by (matrix, block, rank, variant)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = sorted(
        results,
        key=lambda r: (r["matrix"], r["block"], r["rank"], r["variant"]),
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("INFRA-02 Stiefel FD vs analytic gradient — gauntlet\n")
        f.write("=" * 72 + "\n")
        f.write(
            "Acceptance: max_rel < 1e-7 (HM3/GM3 score) and sanity_rel < 1e-7\n"
            "(trace-form, harness self-test). Both at float64.\n\n"
        )
        f.write(
            f"{'matrix':<22} {'block':>5} {'r':>3} {'variant':<8} "
            f"{'score':>13} {'max_rel':>11} {'sanity_rel':>11} "
            f"{'grad_tan_resid':>14}\n"
        )
        f.write("-" * 96 + "\n")
        for r in rows:
            mark = "" if r["max_rel"] < 1e-7 else "  <-- FAIL"
            sanity_mark = "" if r["sanity_max_rel"] < 1e-7 else "  (HARNESS BROKEN)"
            f.write(
                f"{r['matrix']:<22} {r['block']:>5d} {r['rank']:>3d} "
                f"{r['variant']:<8} {r['score']:>13.4e} {r['max_rel']:>11.2e} "
                f"{r['sanity_max_rel']:>11.2e} {r['grad_tan_resid']:>14.2e}"
                f"{mark}{sanity_mark}\n"
            )
        f.write("\n")
        n_fail = sum(1 for r in rows if r["max_rel"] >= 1e-7)
        n_sanity_fail = sum(1 for r in rows if r["sanity_max_rel"] >= 1e-7)
        f.write(f"TOTAL CELLS: {len(rows)}\n")
        f.write(f"FAILED (score rel >= 1e-7): {n_fail}\n")
        f.write(f"FAILED (harness sanity >= 1e-7): {n_sanity_fail}\n")
        if n_fail == 0 and n_sanity_fail == 0:
            f.write("STATUS: PASS\n")
        else:
            f.write("STATUS: FAIL\n")


def write_synthesis_md(path: str, results: list[dict], elapsed_s: float):
    """Pass/fail synthesis with diagnosis stub for any failing cells."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(results)
    n_fail = sum(1 for r in results if r["max_rel"] >= 1e-7)
    n_sanity_fail = sum(1 for r in results if r["sanity_max_rel"] >= 1e-7)
    max_rel = max((r["max_rel"] for r in results), default=float("nan"))
    max_sanity = max((r["sanity_max_rel"] for r in results), default=float("nan"))

    lines = []
    lines.append("# Stiefel FD gradient check (INFRA-02) — synthesis")
    lines.append("")
    lines.append("Date: 2026-04-28")
    lines.append("Backlog: `summary/overview/score_family_workflow.txt` §5 [INFRA-02]")
    lines.append("Toolkit gap closed: `summary/overview/diagnostic_toolkit.txt` §8 (d)")
    lines.append("Module: `test_matrices_fast/stiefel_grad_check.py`")
    lines.append("")
    lines.append("## What this probe does")
    lines.append("")
    lines.append(
        "For the rank-r (Stiefel) lift of S6 / S6_GM, compares the analytic "
        "Euclidean gradient ∂S/∂V against a symmetric-difference quotient "
        "along column-seeded, polar-retracted tangent directions in T_V St. The test passes "
        "iff `rel < 1e-7` on every (matrix, block, r, variant) cell at "
        "float64."
    )
    lines.append("")
    lines.append(
        "A trace-form sanity check `f(V) = trace(V^T M V)` (analytic grad "
        "`2 M V`) is run alongside on every cell — if it fails (`rel ≥ 1e-7`), "
        "the FD-with-tangent-projection harness is broken, NOT the score "
        "implementation."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Cells run: **{n}** (matrices × blocks × ranks × variants)")
    lines.append(f"- Score gradient cells passing (rel < 1e-7): **{n - n_fail}/{n}**")
    lines.append(f"- Trace-form sanity cells passing (rel < 1e-7): **{n - n_sanity_fail}/{n}**")
    lines.append(f"- Worst score rel_err: **{max_rel:.2e}**")
    lines.append(f"- Worst sanity rel_err: **{max_sanity:.2e}**")
    lines.append(f"- Wall time: **{elapsed_s:.1f} s**")
    lines.append(
        "- Trace epsilon sweep lives in "
        "`summary/infra_stiefel_fd_gradient_check/epsilon_sweep.txt`; the "
        "2026-04-28 smoke sweep stayed below `1e-7` for eps from `1e-4` "
        "through `1e-7`."
    )
    lines.append("")
    if n_fail == 0 and n_sanity_fail == 0:
        lines.append("## Verdict: ship")
        lines.append("")
        lines.append(
            "All cells pass the acceptance bar at float64. INFRA-02 can be "
            "closed, and FAM-01 is no longer blocked on the Stiefel FD "
            "gradient-check infrastructure."
        )
    else:
        lines.append("## Verdict: iterate")
        lines.append("")
        if n_sanity_fail > 0:
            lines.append(
                "Trace-form sanity FAILED on some cells. The FD harness "
                "itself is suspect; do NOT trust the HM3/GM3 numbers below "
                "until the harness is fixed."
            )
        if n_fail > 0:
            lines.append("Failing cells:")
            lines.append("")
            lines.append(
                "| matrix | block | r | variant | score | max_rel | sanity_rel |"
            )
            lines.append("|---|---|---|---|---|---|---|")
            for r in sorted(results, key=lambda x: (-x["max_rel"], x["matrix"])):
                if r["max_rel"] >= 1e-7:
                    lines.append(
                        f"| {r['matrix']} | {r['block']} | {r['rank']} | "
                        f"{r['variant']} | {r['score']:.4e} | "
                        f"{r['max_rel']:.2e} | {r['sanity_max_rel']:.2e} |"
                    )
    lines.append("")
    lines.append("## Per-cell table")
    lines.append("")
    lines.append("| matrix | block | r | variant | score | max_rel | sanity_rel | grad_tan_resid |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: (x["matrix"], x["block"], x["rank"], x["variant"])):
        lines.append(
            f"| {r['matrix']} | {r['block']} | {r['rank']} | {r['variant']} | "
            f"{r['score']:.4e} | {r['max_rel']:.2e} | "
            f"{r['sanity_max_rel']:.2e} | {r['grad_tan_resid']:.2e} |"
        )
    lines.append("")
    lines.append("## Formulas")
    lines.append("")
    lines.append("- Tangent space: `T_V St(n,r) = {Z : V^T Z + Z^T V = 0}`.")
    lines.append("- Tangent projection: `P_V(G) = G - V sym(V^T G)`.")
    lines.append(
        "- Column-wise FD direction: draw `E_j` with non-zero entries only in "
        "column `j`, set `Z_j = P_V(E_j) / ||P_V(E_j)||_F`, then compare "
        "`(f(R_V(eps Z_j)) - f(R_V(-eps Z_j))) / (2 eps)` against "
        "`trace(P_V(G)^T Z_j)`."
    )
    lines.append(
        "- Trace sanity: `f(V) = trace(V^T M V)`, `G = 2 M V`, with `M` "
        "symmetric."
    )
    lines.append(
        "- Rank-r S6 lift: `u_X(V) = ||A_X V||_F^2 / ||A_X||_F^2`; aggregate "
        "with HM3 when sketch is present and HM2 at block 1."
    )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- Module: `test_matrices_fast/stiefel_grad_check.py`")
    lines.append("- Gauntlet table: `summary/infra_stiefel_fd_gradient_check/gauntlet.txt`")
    lines.append("- Epsilon sweep: `summary/infra_stiefel_fd_gradient_check/epsilon_sweep.txt`")
    lines.append("- This synthesis: `summary/infra_stiefel_fd_gradient_check/synthesis.md`")
    lines.append("")
    lines.append("## Propagation")
    lines.append("")
    lines.append(
        "Diagnostic toolkit §2 / §8(d) and workflow INFRA-02 should be updated "
        "in the closing patch. No `score_design_overview.txt` propagation is "
        "needed because this ships infrastructure only; it does not change a "
        "fundamental score-design Q or the heuristic status of S6/HM3/relH1."
    )
    lines.append("")
    lines.append("## Notes for downstream consumers")
    lines.append("")
    lines.append(
        "- `stiefel_fd_check(score_value_grad_fn, V, ...)` is the public API. "
        "It accepts ANY value-and-grad function returning `(score, grad)` and "
        "runs the tangent-projection FD compare; reuse it for FAM-01 / FAM-03 "
        "as new variants land."
    )
    lines.append(
        "- `stiefel_tangent_project(V, G) = G - V sym(V^T G)` is exposed as a "
        "library function; `qr_retract(V, W)` is the QR retraction with a "
        "diag-sign fix to avoid spurious sign flips at small step sizes."
    )
    lines.append(
        "- For O(r)-invariant scores (S6 / S6_GM are subspace functions), the "
        "Euclidean gradient has no vertical `V * skew` component, but it can "
        "and usually does have a normal component removed by `P_V`. The "
        "reported `grad_tan_resid` is therefore diagnostic only, not an "
        "acceptance criterion."
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------
# Self-test (cheap; no streaming dependencies)
# --------------------------------------------------------------------------


def _self_test():
    """Cheap self-test: run sanity trace-form + a synthetic HM3 cell.

    Builds toy A_sketch / A_cur / A_fut from random Gaussians (no streaming
    carry), runs S6 and S6_GM grad checks, and asserts rel < 1e-7. Catches
    sign / scale errors in the closed-form grad even if the streaming setup
    is broken.
    """
    rng = np.random.default_rng(0)
    n = 64
    r = 3
    m_sk, m_c, m_f = 80, 32, 32

    # Random matrices.
    A_sk = rng.standard_normal((m_sk, n))
    A_c = rng.standard_normal((m_c, n))
    A_f = rng.standard_normal((m_f, n))
    sk_F2 = float(np.sum(A_sk * A_sk))
    cur_F2 = float(np.sum(A_c * A_c))
    fut_F2 = float(np.sum(A_f * A_f))

    # Random Stiefel point.
    V0 = _seed_stiefel_frame(n, r, rng)

    # (1) Trace-form sanity.
    M_sym = A_c.T @ A_c
    res = stiefel_fd_check(
        lambda V_: trace_form_value_grad(M_sym, V_), V0,
        n_directions=6, eps=1e-5, rng=np.random.default_rng(1),
    )
    assert res["max_rel"] < 1e-9, (
        f"Trace-form sanity failed: max_rel={res['max_rel']:.3e}"
    )

    # (2) S6 (sketch present).
    res = stiefel_fd_check(
        lambda V_: frame_S6_value_grad(A_sk, A_c, A_f, sk_F2, cur_F2, fut_F2, V_),
        V0, n_directions=6, eps=1e-5, rng=np.random.default_rng(2),
    )
    assert res["max_rel"] < 1e-7, (
        f"S6 grad check failed: max_rel={res['max_rel']:.3e}"
    )

    # (3) S6_GM.
    res = stiefel_fd_check(
        lambda V_: frame_S6_GM_value_grad(A_sk, A_c, A_f, sk_F2, cur_F2, fut_F2, V_),
        V0, n_directions=6, eps=1e-5, rng=np.random.default_rng(3),
    )
    assert res["max_rel"] < 1e-7, (
        f"S6_GM grad check failed: max_rel={res['max_rel']:.3e}"
    )

    # (4) S6 block-1 fall-through (no sketch).
    res = stiefel_fd_check(
        lambda V_: frame_S6_value_grad(None, A_c, A_f, 0.0, cur_F2, fut_F2, V_),
        V0, n_directions=6, eps=1e-5, rng=np.random.default_rng(4),
    )
    assert res["max_rel"] < 1e-7, (
        f"S6 block-1 fall-through grad check failed: max_rel={res['max_rel']:.3e}"
    )

    # (5) Tangent projection idempotent.
    G = rng.standard_normal((n, r))
    G_T = stiefel_tangent_project(V0, G)
    G_TT = stiefel_tangent_project(V0, G_T)
    assert np.linalg.norm(G_T - G_TT) / max(np.linalg.norm(G_T), 1e-30) < 1e-12, \
        "tangent projection not idempotent"
    # Skew check: V^T G_T should be skew.
    M = V0.T @ G_T
    skew_resid = float(np.linalg.norm(M + M.T) / max(np.linalg.norm(M), 1e-30))
    assert skew_resid < 1e-12, f"V^T G_T not skew: resid={skew_resid:.3e}"

    print("stiefel_grad_check self-test: OK")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="INFRA-02 Stiefel FD vs analytic gradient gauntlet."
    )
    p.add_argument(
        "--matrices", nargs="+",
        default=["mixed-tail-sharp", "static-cex", "diffuse-diffuse"],
    )
    p.add_argument("--blocks", nargs="+", type=int, default=[1, 2, 12, 31])
    p.add_argument("--ranks", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--variants", nargs="+", default=["S6", "S6_GM"])
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--half-win", type=int, default=32)
    p.add_argument("--n-directions", type=int, default=8)
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-dir", default="summary/infra_stiefel_fd_gradient_check",
        help="Where to write gauntlet.txt + synthesis.md.",
    )
    p.add_argument("--quick", action="store_true",
                   help="Smoke gauntlet: 1 matrix x 2 blocks x 1 rank x 2 variants.")
    p.add_argument("--self-test", action="store_true",
                   help="Run only the cheap synthetic self-test (no streaming).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.self_test:
        _self_test()
        return

    if args.quick:
        matrices = ["mixed-tail-sharp"]
        blocks = [2, 12]
        ranks = [2]
        variants = ["S6", "S6_GM"]
    else:
        matrices = args.matrices
        blocks = args.blocks
        ranks = args.ranks
        variants = args.variants

    # Always run the cheap self-test first (catches harness regressions).
    _self_test()

    t0 = time.time()
    results = run_gauntlet(
        matrices=matrices, blocks=blocks, ranks=ranks, variants=variants,
        n=args.n, half_win=args.half_win,
        n_directions=args.n_directions, eps=args.eps, rng_seed=args.seed,
    )
    elapsed = time.time() - t0

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    gauntlet_path = os.path.join(out_dir, "gauntlet.txt")
    synthesis_path = os.path.join(out_dir, "synthesis.md")
    write_gauntlet_txt(gauntlet_path, results)
    write_synthesis_md(synthesis_path, results, elapsed)

    n_fail = sum(1 for r in results if r["max_rel"] >= 1e-7)
    n_sanity_fail = sum(1 for r in results if r["sanity_max_rel"] >= 1e-7)
    print(f"\nDone. {len(results)} cells in {elapsed:.1f}s.")
    print(f"  score-grad fails (rel >= 1e-7): {n_fail}")
    print(f"  sanity fails (rel >= 1e-7):     {n_sanity_fail}")
    print(f"  wrote {gauntlet_path}")
    print(f"  wrote {synthesis_path}")
    if n_fail or n_sanity_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
