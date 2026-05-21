"""FAM-01-DIAG — frame-level oracle-vs-winner gap (Tier-A screen S-2).

Per score_design_overview.txt §1quinquies (controlling principle): a
score is "oracle-identifying" iff its argmax on the relevant manifold
coincides with the oracle. The vector-level gap test (S-1) is necessary
but not definitive at rank-r — a frame-level extension is required.

Frame extension (Grassmann-invariant; ships with FAM-01 B0):
  u_X(Z) = ||A_X Z||_F² / ||A_X||_F²  for Z ∈ Stiefel(d, 2)
         = (||A_X v_1||² + ||A_X v_2||²) / ||A_X||_F²    (Z = [v_1, v_2])
  Score(Z) = HM3(u_sk(Z), u_cur(Z), u_fut(Z))

Probe:
  Z_oracle = orthonormalized [oracle_v1_proj, oracle_v2_proj] in B_union.
  Z_winner = argmax of Score over Stiefel(d, 2) restricted to B_union,
             via analytic Stiefel ascent with random restarts for the
             canonical HM score.
  Reports Score(Z_oracle), Score(Z_winner), Δ = winner − oracle, and
  cos²-principal angles between the frames.

If Δ > 0 (winner > oracle) on a §6 regression matrix even at high
budget, the evidence model is INSUFFICIENT — joint optimization alone
cannot fix slot-2; evidence augmentation is required (cross-window
alignment, frame interactions, etc.).

Optional ablations (per user's 2026-04-28 update, point 5):
  --ablation hm                 base HM3 frame score (default)
  --ablation hm_x_energy        Score(Z) = HM3 · ||A_total Z||_F²
                                (tests whether absolute energy is the
                                missing identifying signal)
  --ablation hm_x_crosscorr     Score(Z) = HM3 ·
                                 (<A_cur Z, A_fut Z>_F /
                                  ||A_cur Z||_F ||A_fut Z||_F)
                                (tests cross-window response alignment)

Anchor-sensitivity probes (S-5 frame variant): when --anchor=oracle1,
fix v_1 = oracle_v1_proj and optimize v_2 only; --anchor=oracle2 fixes
v_2 = oracle_v2_proj and optimizes v_1.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import cex_restricted_space_probe as probe
import half_window_sliding_hmean_experiment  # noqa: F401
from future_hmean_optimizer_diagnostic import orth_basis_against, rowspace_basis
from hmean_evidence_score import per_block_constants, stream_to_block
from r_sk_g_score import _state_V
from stiefel_grad_check import (
    frame_S6_value_grad,
    polar_retract,
    run_gauntlet as run_stiefel_grad_gauntlet,
    stiefel_tangent_project,
)

from probe_e2_landscape import make_default_args


def _project_unit(v, B):
    if v is None or B is None or B.size == 0:
        return None
    p = B @ (B.T @ v)
    n = float(np.linalg.norm(p))
    return None if n <= 1e-30 else p / n


def _u_X(A_X, v, F2):
    """u_X(v) = ||A_X v||² / ||A_X||_F², the per-window F-norm-normalized
    response."""
    if A_X is None or A_X.size == 0 or F2 <= 0.0:
        return 0.0
    y = A_X @ v
    return float(np.dot(y, y) / F2)


def frame_score(Z, *, A_sk, A_cur, A_fut, sk_F2, cur_F2, fut_F2,
                ablation="hm"):
    """Score(Z) on the rank-2 frame Z. Frame-level Grassmann-invariant
    extension of S6's HM3.
    """
    eps = 1e-30
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    # Compute per-window u_X(Z) = ||A_X Z||_F² / ||A_X||_F²
    # = (||A_X v_1||² + ||A_X v_2||²) / ||A_X||_F²  for orthonormal Z.
    def u(A, F2):
        if A is None or A.size == 0 or F2 <= 0.0:
            return 0.0
        Y = A @ Z
        return float(np.sum(Y * Y) / F2)
    u_sk = u(A_sk, sk_F2) if A_sk is not None else 0.0
    u_g1 = u(A_cur, cur_F2)
    u_g2 = u(A_fut, fut_F2)

    have_sketch = A_sk is not None and sk_F2 > 0.0
    if have_sketch:
        if u_sk <= eps or u_g1 <= eps or u_g2 <= eps:
            base = 0.0
        else:
            base = 3.0 / (1.0 / u_sk + 1.0 / u_g1 + 1.0 / u_g2)
    else:
        if u_g1 <= eps or u_g2 <= eps:
            base = 0.0
        else:
            base = 2.0 / (1.0 / u_g1 + 1.0 / u_g2)

    if ablation == "hm":
        return base
    if ablation == "hm_x_energy":
        # Total energy = ||A_total Z||_F² where A_total = [A_sk; A_cur; A_fut]
        e_sk = (np.sum((A_sk @ Z) ** 2) if (A_sk is not None and A_sk.size) else 0.0)
        e_c = np.sum((A_cur @ Z) ** 2)
        e_f = np.sum((A_fut @ Z) ** 2)
        return base * (e_sk + e_c + e_f)
    if ablation == "hm_x_crosscorr":
        # Cross-window right-space agreement. Current and future rows are
        # independent/unpaired, so do not compare row coordinates directly.
        # Instead compare the response Grams in the frame coordinates:
        #   G_X = (A_X Z)^T (A_X Z).
        # The normalized Frobenius inner product trace(G_cur G_fut) /
        # (||G_cur||_F ||G_fut||_F) is invariant to Z -> ZR.
        Yc = A_cur @ Z
        Yf = A_fut @ Z
        Gc = Yc.T @ Yc
        Gf = Yf.T @ Yf
        nc = float(np.linalg.norm(Gc))
        nf = float(np.linalg.norm(Gf))
        if nc <= eps or nf <= eps:
            return 0.0
        cc = float(np.sum(Gc * Gf) / (nc * nf))
        return base * max(cc, 0.0)
    raise ValueError(f"unknown ablation {ablation!r}")


def _energy_value_grad(A_sk, A_cur, A_fut, V):
    """E(V)=||[A_sk; A_cur; A_fut] V||_F^2 and Euclidean gradient."""
    V = np.asarray(V, dtype=np.float64)
    val = 0.0
    grad = np.zeros_like(V)
    for A in (A_sk, A_cur, A_fut):
        if A is None or np.asarray(A).size == 0:
            continue
        Y = A @ V
        val += float(np.sum(Y * Y))
        grad += 2.0 * (A.T @ Y)
    return val, grad


def _gram_agreement_value_grad(A_cur, A_fut, V):
    """Right-space response agreement and Euclidean gradient.

    Current and future rows are not paired, so compare response Grams
    G_X=(A_X V)^T(A_X V), not raw row coordinates. The normalized Frobenius
    inner product is invariant to V -> V R.
    """
    eps = 1e-30
    V = np.asarray(V, dtype=np.float64)
    Yc = A_cur @ V
    Yf = A_fut @ V
    Gc = Yc.T @ Yc
    Gf = Yf.T @ Yf
    nc = float(np.linalg.norm(Gc))
    nf = float(np.linalg.norm(Gf))
    if nc <= eps or nf <= eps:
        return 0.0, np.zeros_like(V)
    q = float(np.sum(Gc * Gf) / (nc * nf))
    Cc = Gf / (nc * nf) - q * Gc / (nc * nc)
    Cf = Gc / (nc * nf) - q * Gf / (nf * nf)
    # Cc/Cf are symmetric here, but use symmetrized form for numerical safety.
    grad = (
        A_cur.T @ (Yc @ (Cc + Cc.T))
        + A_fut.T @ (Yf @ (Cf + Cf.T))
    )
    return q, grad


def frame_value_grad(A_sk, A_cur, A_fut, sk_F2, cur_F2, fut_F2, V,
                     ablation="hm"):
    """Value-and-gradient for canonical and ablation frame scores."""
    base, grad_base = frame_S6_value_grad(
        A_sk, A_cur, A_fut, sk_F2, cur_F2, fut_F2, V
    )
    if ablation == "hm":
        return base, grad_base
    if ablation == "hm_x_energy":
        e, grad_e = _energy_value_grad(A_sk, A_cur, A_fut, V)
        return float(base * e), e * grad_base + base * grad_e
    if ablation == "hm_x_crosscorr":
        q, grad_q = _gram_agreement_value_grad(A_cur, A_fut, V)
        if q <= 0.0:
            return 0.0, np.zeros_like(V)
        return float(base * q), q * grad_base + base * grad_q
    raise ValueError(f"unknown ablation {ablation!r}")


def _normalize_frame_in_basis(Z, B_union):
    """Project Z columns into B_union, then QR-orthonormalize."""
    Z = np.asarray(Z, dtype=np.float64)
    Z_proj = B_union @ (B_union.T @ Z)
    Q, R = np.linalg.qr(Z_proj)
    # Discard columns whose QR diagonal is tiny (linearly dependent).
    keep = np.abs(np.diag(R)) > 1e-12
    return Q[:, keep]


def _stiefel_frame_ascent(Z_init, *, A_sk, A_cur, A_fut, sk_F2, cur_F2,
                          fut_F2, B_union, ablation="hm", max_iter=200,
                          step0=1.0, tol=1e-11):
    """Joint Stiefel ascent for the frame score.

    This is the default FAM-01-DIAG optimizer. It uses analytic rank-r
    gradients instead of finite-differencing every coordinate.
    """
    Z = _normalize_frame_in_basis(Z_init, B_union)
    if Z.shape[1] < 2:
        return None
    Z = Z[:, :2]

    def value_grad(V):
        return frame_value_grad(
            A_sk, A_cur, A_fut, sk_F2, cur_F2, fut_F2, V,
            ablation=ablation,
        )

    score, _ = value_grad(Z)
    score = float(score)
    for _ in range(max_iter):
        _, G = value_grad(Z)
        # Keep the Euclidean gradient in the searched subspace before
        # projecting onto T_Z St(d,2).
        G = B_union @ (B_union.T @ G)
        Gt = stiefel_tangent_project(Z, G)
        gnorm = float(np.linalg.norm(Gt))
        if gnorm <= tol:
            break

        accepted = False
        for step in (step0, step0 / 2.0, step0 / 4.0, step0 / 8.0, step0 / 16.0):
            Z_try = polar_retract(Z, step * Gt)
            score_try, _ = value_grad(Z_try)
            score_try = float(score_try)
            if score_try > score + tol:
                Z = Z_try
                score = score_try
                accepted = True
                break
        if not accepted:
            break
    return Z, score


def _anchored_hm_ascent(fixed_v, free_init, free_col, *, A_sk, A_cur,
                        A_fut, sk_F2, cur_F2, fut_F2, B_union,
                        max_iter=240, step0=1.0, tol=1e-11):
    """Optimize one frame column while keeping the oracle anchor fixed."""
    fixed_v = _project_unit(fixed_v, B_union)
    if fixed_v is None:
        return None
    B_def = orth_basis_against(B_union, fixed_v)
    if B_def is None or B_def.size == 0:
        return None

    v = B_def @ (B_def.T @ free_init)
    nv = float(np.linalg.norm(v))
    if nv <= 1e-30:
        return None
    v /= nv

    def make_Z(vec):
        return (
            np.column_stack([fixed_v, vec])
            if free_col == 1
            else np.column_stack([vec, fixed_v])
        )

    Z = make_Z(v)
    score, _ = frame_S6_value_grad(A_sk, A_cur, A_fut, sk_F2, cur_F2, fut_F2, Z)
    score = float(score)
    for _ in range(max_iter):
        _, G = frame_S6_value_grad(A_sk, A_cur, A_fut, sk_F2, cur_F2, fut_F2, Z)
        gv = B_def @ (B_def.T @ G[:, free_col])
        gv = gv - float(v @ gv) * v
        gnorm = float(np.linalg.norm(gv))
        if gnorm <= tol:
            break
        accepted = False
        for step in (step0, step0 / 2.0, step0 / 4.0, step0 / 8.0, step0 / 16.0):
            v_try = v + step * gv
            nv = float(np.linalg.norm(v_try))
            if nv <= 1e-30:
                continue
            v_try /= nv
            Z_try = make_Z(v_try)
            score_try, _ = frame_S6_value_grad(
                A_sk, A_cur, A_fut, sk_F2, cur_F2, fut_F2, Z_try
            )
            score_try = float(score_try)
            if score_try > score + tol:
                v = v_try
                Z = Z_try
                score = score_try
                accepted = True
                break
        if not accepted:
            break
    return Z, score


def _alternating_optimize(Z_init, *, A_sk, A_cur, A_fut, sk_F2, cur_F2,
                           fut_F2, B_union, ablation, n_steps=80,
                           rng=None, search_starts=8):
    """Alternating block-coordinate ascent: fix one column, optimize the
    other in B_union ⊥ fixed-column. Use a small grid + sphere ascent."""
    if rng is None:
        rng = np.random.default_rng(0)
    Z = _normalize_frame_in_basis(Z_init, B_union)
    if Z.shape[1] < 2:
        return None
    best_score = frame_score(Z, A_sk=A_sk, A_cur=A_cur, A_fut=A_fut,
                             sk_F2=sk_F2, cur_F2=cur_F2, fut_F2=fut_F2,
                             ablation=ablation)
    for it in range(n_steps):
        improved = False
        for j in (0, 1):
            other = Z[:, 1 - j]
            B_def = orth_basis_against(B_union, other)
            if B_def is None or B_def.size == 0:
                continue
            # Sphere-ascent in B_def via projected gradient with a small
            # set of starts: include current Z[:,j] and a few random.
            cands = [B_def @ (B_def.T @ Z[:, j])]
            for _ in range(search_starts):
                z = B_def @ rng.standard_normal(B_def.shape[1])
                cands.append(z)
            best_v = None
            best_v_score = -np.inf
            for c in cands:
                n = float(np.linalg.norm(c))
                if n <= 1e-30:
                    continue
                v = c / n
                # Local ascent: 30 iterations of projected gradient.
                v = _local_sphere_ascent(
                    v, j, Z, B_def,
                    A_sk=A_sk, A_cur=A_cur, A_fut=A_fut,
                    sk_F2=sk_F2, cur_F2=cur_F2, fut_F2=fut_F2,
                    ablation=ablation, n_iter=40,
                )
                Z_try = Z.copy()
                Z_try[:, j] = v
                s = frame_score(Z_try, A_sk=A_sk, A_cur=A_cur, A_fut=A_fut,
                                sk_F2=sk_F2, cur_F2=cur_F2, fut_F2=fut_F2,
                                ablation=ablation)
                if s > best_v_score:
                    best_v_score = s
                    best_v = v
            if best_v is not None and best_v_score > best_score + 1e-12:
                Z[:, j] = best_v
                # Re-orthonormalize (paranoia).
                Z, _ = np.linalg.qr(Z)
                best_score = best_v_score
                improved = True
        if not improved:
            break
    return Z, best_score


def _local_sphere_ascent(v, j, Z, B_def, *, A_sk, A_cur, A_fut,
                          sk_F2, cur_F2, fut_F2, ablation, n_iter=40,
                          step=0.1):
    """Simple projected-gradient ascent of frame_score in v while keeping
    Z[:,1-j] fixed and v constrained to B_def. Uses finite-difference
    gradient (not analytic — keeps the probe lean)."""
    eps = 1e-6
    for _ in range(n_iter):
        # Compute FD gradient on B_def coordinates by finite differences
        # in B_def basis (cheap because B_def is small).
        d = B_def.shape[1]
        coords = B_def.T @ v
        Z_v = Z.copy(); Z_v[:, j] = v
        f0 = frame_score(Z_v, A_sk=A_sk, A_cur=A_cur, A_fut=A_fut,
                         sk_F2=sk_F2, cur_F2=cur_F2, fut_F2=fut_F2,
                         ablation=ablation)
        grad = np.zeros(d)
        for i in range(d):
            ci = coords.copy(); ci[i] += eps
            v_p = B_def @ ci
            v_p = v_p / max(np.linalg.norm(v_p), 1e-30)
            Z_p = Z.copy(); Z_p[:, j] = v_p
            f_p = frame_score(Z_p, A_sk=A_sk, A_cur=A_cur, A_fut=A_fut,
                              sk_F2=sk_F2, cur_F2=cur_F2, fut_F2=fut_F2,
                              ablation=ablation)
            grad[i] = (f_p - f0) / eps
        # Project gradient to tangent: in B_def basis, the tangent at
        # coords is (I - coords coords^T)·grad.
        coords_n = coords / max(np.linalg.norm(coords), 1e-30)
        grad_t = grad - float(coords_n @ grad) * coords_n
        gnorm = float(np.linalg.norm(grad_t))
        if gnorm < 1e-10:
            break
        # Line search: try a few step sizes.
        best_v = v
        best_f = f0
        for s in (step, step / 2, step / 4):
            new_coords = coords + s * grad_t
            new_v = B_def @ new_coords
            n = float(np.linalg.norm(new_v))
            if n <= 1e-30:
                continue
            new_v /= n
            Z_n = Z.copy(); Z_n[:, j] = new_v
            f_n = frame_score(Z_n, A_sk=A_sk, A_cur=A_cur, A_fut=A_fut,
                              sk_F2=sk_F2, cur_F2=cur_F2, fut_F2=fut_F2,
                              ablation=ablation)
            if f_n > best_f:
                best_f = f_n
                best_v = new_v
                break
        if best_f <= f0 + 1e-12:
            break
        v = best_v
    return v


def run_for_matrix(matrix, *, block=31, ablation="hm", n_starts=100,
                   anchor="free", max_iter=200):
    args = make_default_args(matrix, block=block)
    # Some legacy snapshot paths still draw from NumPy's module-level RNG.
    # Re-seed per run so anchor/ablation comparisons share the same snapshot.
    np.random.seed(args.seed)
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=matrix, n=args.n, preset=args.preset, seed=args.seed,
        r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
        shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, np.float64)
    V_exact = np.asarray(V_exact, np.float64)
    snapshots = stream_to_block(args, A, V_exact, work_dtype, int(args.rank), block, {block})
    snap = snapshots[block]
    consts = per_block_constants(A, block, args.half_win)

    A_cur = np.asarray(snap["A_cur"], dtype=np.float64)
    A_fut = np.asarray(snap["A_fut"], dtype=np.float64)
    A_sketch = np.asarray(snap["A_sketch"], dtype=np.float64)
    A_sk = A_sketch if A_sketch.size else None
    state = snap["state"]
    V_state = _state_V(state)

    cur_F2 = float(consts["cur_F2"])
    fut_F2 = float(consts["fut_F2"])
    sk_F2_low = float(np.sum(A_sketch ** 2)) if A_sk is not None else 0.0

    if A_sk is not None:
        union_stack = np.vstack([A_sketch, A_cur, A_fut])
    else:
        union_stack = np.vstack([A_cur, A_fut])
    B_union = rowspace_basis(union_stack)

    oracle_v1_proj = _project_unit(V_exact[:, 0], B_union)
    oracle_v2_proj = _project_unit(V_exact[:, 1], B_union)
    if oracle_v1_proj is None or oracle_v2_proj is None:
        return None

    Z_oracle, _ = np.linalg.qr(np.column_stack([oracle_v1_proj, oracle_v2_proj]))
    score_kw = dict(A_sk=A_sk, A_cur=A_cur, A_fut=A_fut,
                    sk_F2=sk_F2_low, cur_F2=cur_F2, fut_F2=fut_F2,
                    ablation=ablation)
    score_oracle = frame_score(Z_oracle, **score_kw)

    rng = np.random.default_rng(args.seed + 444_777 + block)

    # Build initial frame candidates.
    init_frames = []
    if anchor == "oracle1":
        # Fix v_1 = oracle_v1_proj. Random v_2 ⊥ oracle_v1_proj in B_union.
        B_def = orth_basis_against(B_union, oracle_v1_proj)
        for _ in range(n_starts):
            z = B_def @ rng.standard_normal(B_def.shape[1])
            n = float(np.linalg.norm(z))
            if n > 1e-30:
                v2 = z / n
                init_frames.append(np.column_stack([oracle_v1_proj, v2]))
        # Also include oracle_v2_proj as a start.
        init_frames.append(np.column_stack([oracle_v1_proj, oracle_v2_proj]))
    elif anchor == "oracle2":
        B_def = orth_basis_against(B_union, oracle_v2_proj)
        for _ in range(n_starts):
            z = B_def @ rng.standard_normal(B_def.shape[1])
            n = float(np.linalg.norm(z))
            if n > 1e-30:
                v1 = z / n
                init_frames.append(np.column_stack([v1, oracle_v2_proj]))
        init_frames.append(np.column_stack([oracle_v1_proj, oracle_v2_proj]))
    else:
        # Free: oracle warm-start + sketch warm-start + random.
        init_frames.append(Z_oracle)
        if V_state is not None and V_state.shape[1] >= 2:
            init_frames.append(V_state[:, :2])
        for _ in range(n_starts):
            M = B_union @ rng.standard_normal((B_union.shape[1], 2))
            init_frames.append(M)

    best_Z = None
    best_score = -np.inf
    if ablation == "hm" and anchor in ("oracle1", "oracle2"):
        fixed = oracle_v1_proj if anchor == "oracle1" else oracle_v2_proj
        free_col = 1 if anchor == "oracle1" else 0
        for Z0 in init_frames:
            Z0 = _normalize_frame_in_basis(Z0, B_union)
            if Z0.shape[1] < 2:
                continue
            res = _anchored_hm_ascent(
                fixed, Z0[:, free_col], free_col,
                A_sk=A_sk, A_cur=A_cur, A_fut=A_fut,
                sk_F2=sk_F2_low, cur_F2=cur_F2, fut_F2=fut_F2,
                B_union=B_union, max_iter=max_iter,
            )
            if res is None:
                continue
            Z_try, s = res
            if s > best_score:
                best_score = s
                best_Z = Z_try
    elif anchor in ("oracle1", "oracle2"):
        # Anchored: only optimize the free column. We can use the same
        # alternating optimizer; the fixed column stays the same column
        # because alternating ascent will leave it unchanged if the
        # free column constraint always orthogonalizes against it. Easy
        # path: just run rank-1 sphere ascent on the free column.
        for Z0 in init_frames:
            Z0 = _normalize_frame_in_basis(Z0, B_union)
            if Z0.shape[1] < 2:
                continue
            free = 1 if anchor == "oracle1" else 0
            other = Z0[:, 1 - free]
            B_def = orth_basis_against(B_union, other)
            if B_def is None or B_def.size == 0:
                continue
            v = Z0[:, free]
            v = _local_sphere_ascent(
                v, free, Z0, B_def,
                A_sk=A_sk, A_cur=A_cur, A_fut=A_fut,
                sk_F2=sk_F2_low, cur_F2=cur_F2, fut_F2=fut_F2,
                ablation=ablation, n_iter=120,
            )
            Z_try = Z0.copy(); Z_try[:, free] = v
            Z_try, _ = np.linalg.qr(Z_try)
            s = frame_score(Z_try, **score_kw)
            if s > best_score:
                best_score = s
                best_Z = Z_try
    else:
        for Z0 in init_frames:
            res = _stiefel_frame_ascent(
                Z0, A_sk=A_sk, A_cur=A_cur, A_fut=A_fut,
                sk_F2=sk_F2_low, cur_F2=cur_F2, fut_F2=fut_F2,
                B_union=B_union, ablation=ablation, max_iter=max_iter,
            )
            if res is None:
                continue
            Z, s = res
            if s > best_score:
                best_score = s
                best_Z = Z

    # Principal cosines between best_Z and Z_oracle.
    pa_cos = None
    if best_Z is not None:
        M = best_Z.T @ Z_oracle
        sv = np.linalg.svd(M, compute_uv=False)
        pa_cos = sv ** 2  # cos² principal angles

    return {
        "matrix": matrix, "block": block, "ablation": ablation, "anchor": anchor,
        "score_oracle": float(score_oracle),
        "score_winner": float(best_score) if best_score > -np.inf else float("nan"),
        "delta": float(best_score - score_oracle) if best_score > -np.inf else float("nan"),
        "pa_cos2": pa_cos.tolist() if pa_cos is not None else None,
        "n_starts": int(n_starts),
        "max_iter": int(max_iter),
        "optimizer": "analytic_stiefel_frame",
        "status": (
            "evidence_insufficient" if best_score > score_oracle + 1e-10
            else "oracle_identifying"
        ) if best_score > -np.inf else "no_winner",
    }


def write_synthesis(rows, path, block, elapsed):
    free_rows = [r for r in rows if r["anchor"] == "free"]
    canonical = [r for r in free_rows if r["ablation"] == "hm"]
    fail = [r["matrix"] for r in canonical if r["delta"] > 1e-10]
    passed = [r["matrix"] for r in canonical if r["delta"] <= 1e-10]
    with open(path, "w") as fh:
        fh.write("# FAM-01-DIAG synthesis\n\n")
        fh.write(f"Run date: 2026-04-28\n")
        fh.write(f"Block: b{block}\n")
        fh.write(f"Elapsed seconds: {elapsed:.3f}\n\n")
        fh.write("Canonical screen: `hm` with `anchor=free`.\n")
        fh.write("Optimizer: analytic retraction-based Stiefel gradient ascent.\n\n")
        if canonical:
            fh.write("Canonical verdict by matrix:\n")
            for r in canonical:
                verdict = "FAIL evidence model insufficient" if r["delta"] > 1e-10 else "PASS oracle identifying"
                fh.write(f"- {r['matrix']}: {verdict}; delta={r['delta']:+.6g}; pa_cos2={r['pa_cos2']}\n")
            fh.write("\nCanonical summary:\n")
            fh.write(f"- delta > 0: {', '.join(fail) if fail else 'none'}\n")
            fh.write(f"- delta <= 0: {', '.join(passed) if passed else 'none'}\n\n")
        if free_rows:
            fh.write("Free-frame rows:\n")
            for r in free_rows:
                verdict = "FAIL" if r["delta"] > 1e-10 else "PASS"
                fh.write(
                    f"- {r['matrix']} / {r['ablation']}: {verdict}; "
                    f"delta={r['delta']:+.6g}; pa_cos2={r['pa_cos2']}\n"
                )


def write_t1_gradient_check(rows, out_dir, elapsed):
    """Write the FAM-01-DIAG T1 gradient-check artifact.

    This is intentionally a thin FAM-01-DIAG wrapper around the INFRA-02
    Stiefel FD harness. The checked score is exactly the canonical frame
    score used by this diagnostic:

        Score(Z) = HM(u_sk(Z), u_cur(Z), u_fut(Z)),
        u_X(Z) = ||A_X Z||_F^2 / ||A_X||_F^2.
    """
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, "T1_gradient_check.txt")
    md_path = os.path.join(out_dir, "T1_gradient_check.md")
    n = len(rows)
    n_score_fail = sum(1 for r in rows if r["max_rel"] >= 1e-7)
    n_sanity_fail = sum(1 for r in rows if r["sanity_max_rel"] >= 1e-7)
    worst_score = max((r["max_rel"] for r in rows), default=float("nan"))
    worst_sanity = max((r["sanity_max_rel"] for r in rows), default=float("nan"))
    status = "PASS" if n_score_fail == 0 and n_sanity_fail == 0 else "FAIL"

    rows_sorted = sorted(
        rows, key=lambda r: (r["matrix"], r["block"], r["rank"], r["variant"])
    )
    with open(txt_path, "w") as fh:
        fh.write("FAM-01-DIAG T1 gradient check\n")
        fh.write("=" * 72 + "\n")
        fh.write("Score: HM(u_sk(Z), u_cur(Z), u_fut(Z)); ")
        fh.write("u_X(Z)=||A_X Z||_F^2/||A_X||_F^2\n")
        fh.write("Acceptance: score max_rel < 1e-7 and trace sanity < 1e-7\n\n")
        fh.write(
            f"{'matrix':<22} {'block':>5} {'r':>3} {'score':>13} "
            f"{'max_rel':>11} {'sanity_rel':>11} {'grad_tan_resid':>14}\n"
        )
        fh.write("-" * 86 + "\n")
        for r in rows_sorted:
            mark = "" if r["max_rel"] < 1e-7 else "  <-- FAIL"
            sanity_mark = "" if r["sanity_max_rel"] < 1e-7 else "  (SANITY FAIL)"
            fh.write(
                f"{r['matrix']:<22} {r['block']:>5d} {r['rank']:>3d} "
                f"{r['score']:>13.4e} {r['max_rel']:>11.2e} "
                f"{r['sanity_max_rel']:>11.2e} {r['grad_tan_resid']:>14.2e}"
                f"{mark}{sanity_mark}\n"
            )
        fh.write("\n")
        fh.write(f"TOTAL CELLS: {n}\n")
        fh.write(f"FAILED (score rel >= 1e-7): {n_score_fail}\n")
        fh.write(f"FAILED (trace sanity >= 1e-7): {n_sanity_fail}\n")
        fh.write(f"WORST SCORE REL: {worst_score:.2e}\n")
        fh.write(f"WORST SANITY REL: {worst_sanity:.2e}\n")
        fh.write(f"ELAPSED_SECONDS: {elapsed:.3f}\n")
        fh.write(f"STATUS: {status}\n")

    with open(md_path, "w") as fh:
        fh.write("# FAM-01-DIAG T1 gradient check\n\n")
        fh.write("Run date: 2026-04-28\n\n")
        fh.write(
            "This checks the analytic gradient for the canonical frame score "
            "`Score(Z)=HM(u_sk(Z), u_cur(Z), u_fut(Z))`, with "
            "`u_X(Z)=||A_X Z||_F^2/||A_X||_F^2`, using the INFRA-02 "
            "Stiefel finite-difference harness.\n\n"
        )
        fh.write(f"- Status: **{status}**\n")
        fh.write(f"- Cells run: **{n}**\n")
        fh.write(f"- Score-gradient failures: **{n_score_fail}**\n")
        fh.write(f"- Trace-sanity failures: **{n_sanity_fail}**\n")
        fh.write(f"- Worst score rel_err: **{worst_score:.2e}**\n")
        fh.write(f"- Worst trace sanity rel_err: **{worst_sanity:.2e}**\n")
        fh.write(f"- Wall time: **{elapsed:.1f} s**\n\n")
        fh.write("| matrix | block | r | score | max_rel | sanity_rel |\n")
        fh.write("|---|---:|---:|---:|---:|---:|\n")
        for r in rows_sorted:
            fh.write(
                f"| {r['matrix']} | {r['block']} | {r['rank']} | "
                f"{r['score']:.4e} | {r['max_rel']:.2e} | "
                f"{r['sanity_max_rel']:.2e} |\n"
            )
    return txt_path, md_path


def run_t1_gradient_check(args):
    matrices = args.matrices
    blocks = args.t1_blocks
    ranks = args.t1_ranks
    if args.quick:
        matrices = ["mixed-tail-sharp"]
        blocks = [2, 12]
        ranks = [2]

    t0 = time.time()
    rows = run_stiefel_grad_gauntlet(
        matrices=matrices,
        blocks=blocks,
        ranks=ranks,
        variants=["S6"],
        n=args.n,
        half_win=args.half_win,
        n_directions=args.n_directions,
        eps=args.eps,
        rng_seed=args.seed,
    )
    elapsed = time.time() - t0
    txt_path, md_path = write_t1_gradient_check(rows, args.out_dir, elapsed)
    n_score_fail = sum(1 for r in rows if r["max_rel"] >= 1e-7)
    n_sanity_fail = sum(1 for r in rows if r["sanity_max_rel"] >= 1e-7)
    print(f"wrote {txt_path}")
    print(f"wrote {md_path}")
    if n_score_fail or n_sanity_fail:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", nargs="+", default=[
        "diffuse-diffuse", "residual-spiky-shocks", "mixed-tail-soft",
        "mixed-tail-sharp", "static-cex", "mixed-tail-balanced",
        "etf-basket-basis",
    ])
    ap.add_argument("--block", type=int, default=31)
    ap.add_argument("--ablations", nargs="+", default=["hm"])
    ap.add_argument("--anchors", nargs="+",
                    default=["free", "oracle1", "oracle2"])
    ap.add_argument("--n-starts", type=int, default=100)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--out-dir", default="summary/infra_frame_oracle_gap")
    ap.add_argument("--gradient-check", action="store_true",
                    help="Run T1 gradient check for the canonical HM frame score.")
    ap.add_argument("--quick", action="store_true",
                    help="Small smoke run for --gradient-check or S-2.")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--half-win", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-directions", type=int, default=8)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--t1-blocks", nargs="+", type=int, default=[1, 2, 12, 31])
    ap.add_argument("--t1-ranks", nargs="+", type=int, default=[2])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.gradient_check:
        run_t1_gradient_check(args)
        return

    rows = []
    t0 = time.time()
    for m in args.matrices:
        print(f"== {m} ==", flush=True)
        for ab in args.ablations:
            for an in args.anchors:
                r = run_for_matrix(m, block=args.block, ablation=ab,
                                   n_starts=args.n_starts, anchor=an,
                                   max_iter=args.max_iter)
                if r is None:
                    continue
                rows.append(r)
                print(f"  {ab:<16} {an:<8} oracle={r['score_oracle']:.4f}  "
                      f"winner={r['score_winner']:.4f}  Δ={r['delta']:+.4f}  "
                      f"pa²={r['pa_cos2']}", flush=True)

    json_path = os.path.join(args.out_dir, f"frame_gap_b{args.block}.json")
    with open(json_path, "w") as fh:
        json.dump(rows, fh, indent=2, default=float)

    txt_path = os.path.join(args.out_dir, f"frame_gap_b{args.block}.txt")
    with open(txt_path, "w") as fh:
        fh.write("# FAM-01-DIAG: frame-level oracle-vs-winner gap (S-2 screen)\n")
        fh.write(f"# block={args.block}\n")
        fh.write("# Δ > 0 (winner > oracle) on a regression matrix → evidence model insufficient at rank-2\n\n")
        fh.write(f"{'matrix':<24} {'ablation':<16} {'anchor':<8} "
                 f"{'oracle':>10} {'winner':>10} {'Δ':>10} {'pa²[0]':>8} {'pa²[1]':>8}\n")
        for r in rows:
            pa = r["pa_cos2"] or [float("nan"), float("nan")]
            pa = (pa + [float("nan"), float("nan")])[:2]
            fh.write(f"{r['matrix']:<24} {r['ablation']:<16} {r['anchor']:<8} "
                     f"{r['score_oracle']:>10.4f} {r['score_winner']:>10.4f} "
                     f"{r['delta']:>+10.4f} {pa[0]:>8.3f} {pa[1]:>8.3f}\n")
    synthesis_path = os.path.join(args.out_dir, "synthesis.md")
    write_synthesis(rows, synthesis_path, args.block, time.time() - t0)
    print(f"wrote {json_path}\nwrote {txt_path}\nwrote {synthesis_path}")


if __name__ == "__main__":
    main()
