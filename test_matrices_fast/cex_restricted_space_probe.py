import argparse
import os
import sys
import time

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.linalg as la


def kahan_sum(x):
    x = np.asarray(x).reshape(-1)
    s = 0.0
    c = 0.0
    for xi in x:
        y = float(xi) - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def project_feasible(x, Q):
    x = np.asarray(x).reshape(-1)
    if Q is not None and Q.size:
        x = x - Q @ (Q.T @ x)
    return np.ascontiguousarray(x)


def retract_feasible(x, Q):
    x = project_feasible(x, Q)
    nx = np.sqrt(kahan_sum(np.abs(x) ** 2))
    if nx <= 1e-14:
        return None
    return x / nx


def make_basic_restart_seeds(M, Q, k_idx, V_init, num_restarts):
    work_dtype = np.asarray(M).dtype
    d = M.shape[1]
    _, _, Vh = np.linalg.svd(np.asarray(M), full_matrices=False)
    Vsvd = np.ascontiguousarray(Vh.T, dtype=work_dtype)
    V_init_work = None if V_init is None else np.ascontiguousarray(np.asarray(V_init, dtype=work_dtype))
    num_top = min(4, Vsvd.shape[1])
    alpha_grid = [0.98, 0.9, 0.75, 0.5, 0.25, 0.0]
    starts = []

    for restart in range(num_restarts):
        v_prev = (
            V_init_work[:, k_idx]
            if V_init_work is not None and V_init_work.size and V_init_work.shape[1] > k_idx
            else None
        )

        restart_type = (restart % 5) + 1
        restart_block = restart // 5

        if restart_type == 1:
            if v_prev is not None:
                xi = np.random.standard_normal(d).astype(work_dtype, copy=False)
                xi = project_feasible(xi, Q)
                nxi = np.sqrt(kahan_sum(np.abs(xi) ** 2))
                if nxi > 1e-14:
                    xi = np.ascontiguousarray(xi / nxi, dtype=work_dtype)
                alpha = alpha_grid[restart_block % len(alpha_grid)]
                v0 = np.ascontiguousarray(
                    alpha * v_prev + np.sqrt(max(0.0, 1.0 - alpha ** 2)) * xi,
                    dtype=work_dtype,
                )
            else:
                v0 = Vsvd[:, 0]
        elif restart_type == 2:
            j = restart_block % num_top
            v0 = Vsvd[:, j]
        elif restart_type == 3:
            j1 = restart_block % num_top
            j2 = (restart_block + 1) % num_top
            alpha = alpha_grid[restart_block % len(alpha_grid)]
            v0 = np.ascontiguousarray(
                alpha * Vsvd[:, j1] + np.sqrt(max(0.0, 1.0 - alpha ** 2)) * Vsvd[:, j2],
                dtype=work_dtype,
            )
        elif restart_type == 4:
            j = restart_block % num_top
            v0 = np.ascontiguousarray(
                Vsvd[:, j] + np.asarray(1e-2 * np.random.standard_normal(d), dtype=work_dtype),
                dtype=work_dtype,
            )
        else:
            v0 = np.random.standard_normal(d).astype(work_dtype, copy=False)

        v = retract_feasible(v0, Q)
        if v is None:
            v = retract_feasible(np.random.standard_normal(d).astype(work_dtype, copy=False), Q)
        if v is None:
            raise RuntimeError("Could not generate feasible restart seed.")
        starts.append(np.ascontiguousarray(v, dtype=work_dtype))

    return starts


def project_reduced(x, Qz):
    x = np.asarray(x).reshape(-1)
    if Qz is None or Qz.size == 0:
        return np.ascontiguousarray(x)
    return np.ascontiguousarray(x - Qz @ (Qz.T @ x))


def retract_reduced(z, Qz, eps=1e-14):
    y = project_reduced(z, Qz)
    ny = np.linalg.norm(y)
    if ny <= eps:
        return None
    return np.ascontiguousarray(y / ny, dtype=y.dtype)


def append_unique_reduced_seed(starts, z, cos_tol=1e-10):
    if z is None:
        return False
    for prev in starts:
        if abs(float(prev @ z)) > 1.0 - cos_tol:
            return False
    starts.append(z)
    return True


def sym_part(A):
    A_arr = np.asarray(A)
    return 0.5 * (A_arr + A_arr.T)


def project_reduced_columns(X, Qz):
    X_arr = np.asarray(X)
    if Qz is None or Qz.size == 0:
        return np.ascontiguousarray(X_arr)
    return np.ascontiguousarray(X_arr - Qz @ (Qz.T @ X_arr))


def retract_stiefel_reduced(Y, Qz=None, eps=1e-12):
    Yp = project_reduced_columns(Y, Qz)
    if Yp.ndim != 2:
        raise ValueError("Stiefel retraction expects a matrix.")
    if Yp.shape[1] == 0:
        return np.zeros((Yp.shape[0], 0), dtype=Yp.dtype)
    if Yp.shape[0] < Yp.shape[1]:
        return None
    U, s, Vt = np.linalg.svd(np.asarray(Yp, dtype=np.float64), full_matrices=False)
    if s.size < Yp.shape[1] or float(np.min(s[: Yp.shape[1]])) <= eps:
        return None
    Z = U[:, : Yp.shape[1]] @ Vt[: Yp.shape[1], :]
    return np.ascontiguousarray(Z, dtype=Yp.dtype)


def stiefel_tangent_gradient(Z, G, Qz=None):
    Gp = project_reduced_columns(G, Qz)
    Xi = Gp - Z @ sym_part(Z.T @ Gp)
    return np.ascontiguousarray(project_reduced_columns(Xi, Qz), dtype=Z.dtype)


def append_unique_stiefel_seed(starts, Z, cos_tol=1e-10):
    if Z is None:
        return False
    for prev in starts:
        if prev.shape != Z.shape:
            continue
        same = np.linalg.norm(prev - Z, ord="fro")
        signed = np.linalg.norm(prev + Z, ord="fro")
        if min(same, signed) <= cos_tol:
            return False
    starts.append(Z)
    return True


def make_oracle_stiefel_warm_start(M_gain, Vbasis, V_exact, joint_rank, active_r, Qz=None, row_samples=None):
    if V_exact is None:
        return None
    target = np.asarray(V_exact)
    if target.ndim != 2 or target.size == 0:
        return None
    oracle_cols = min(int(active_r), int(joint_rank), target.shape[1])
    if oracle_cols <= 0:
        return None

    Q_oracle, _ = projected_true_span_oracle(
        M_gain, target, oracle_cols, dtype=np.asarray(Vbasis).dtype, row_samples=row_samples
    )
    if Q_oracle.shape[1] < oracle_cols:
        return None

    Vbasis_arr = np.asarray(Vbasis)
    Z0 = np.zeros((Vbasis_arr.shape[1], int(joint_rank)), dtype=Vbasis_arr.dtype)
    Z0[:, :oracle_cols] = np.ascontiguousarray(Vbasis_arr.T @ Q_oracle[:, :oracle_cols], dtype=Vbasis_arr.dtype)
    for fill_idx in range(oracle_cols, int(joint_rank)):
        if fill_idx < Vbasis_arr.shape[1]:
            Z0[fill_idx, fill_idx] = 1.0
        else:
            Z0[:, fill_idx] = np.asarray(np.random.default_rng(fill_idx).standard_normal(Vbasis_arr.shape[1]), dtype=Vbasis_arr.dtype)
    return retract_stiefel_reduced(Z0, Qz)


def make_oracle_reduced_warm_start(M_gain, Vbasis, V_exact, k_idx, active_r, Qz=None, row_samples=None):
    if V_exact is None:
        return None
    target = np.asarray(V_exact)
    if target.ndim != 2 or target.size == 0:
        return None
    oracle_cols = min(int(active_r), target.shape[1])
    if int(k_idx) < 0 or int(k_idx) >= oracle_cols:
        return None

    Q_oracle, _ = projected_true_span_oracle(
        M_gain, target, oracle_cols, dtype=np.asarray(Vbasis).dtype, row_samples=row_samples
    )
    if Q_oracle.shape[1] <= int(k_idx):
        return None

    Vbasis_arr = np.asarray(Vbasis)
    z0 = np.ascontiguousarray(Vbasis_arr.T @ Q_oracle[:, int(k_idx)], dtype=Vbasis_arr.dtype)
    return retract_reduced(z0, Qz)


def orthonormalize_columns(X, dtype=None, eps=1e-12):
    X_arr = np.asarray(X if X is not None else np.zeros((0, 0)))
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
    if X_arr.size == 0:
        rows = X_arr.shape[0] if X_arr.ndim == 2 else 0
        out_dtype = np.float32 if dtype is None else dtype
        return np.zeros((rows, 0), dtype=out_dtype)
    if dtype is None:
        dtype = X_arr.dtype
    Q, R = np.linalg.qr(np.asarray(X_arr, dtype=np.float64), mode="reduced")
    keep = np.abs(np.diag(R)) > eps
    if not np.any(keep):
        return np.zeros((X_arr.shape[0], 0), dtype=dtype)
    return np.ascontiguousarray(Q[:, keep], dtype=dtype)


def append_basis_columns(Vbasis, X_new, max_cols=None, eps=1e-12):
    X_arr = np.asarray(X_new if X_new is not None else np.zeros((Vbasis.shape[0], 0), dtype=Vbasis.dtype))
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
    if X_arr.size == 0:
        return Vbasis
    if max_cols is not None:
        remaining = max(0, int(max_cols) - Vbasis.shape[1])
        if remaining == 0:
            return np.ascontiguousarray(Vbasis, dtype=Vbasis.dtype)
        X_arr = X_arr[:, :remaining]
    combined = np.column_stack([Vbasis, X_arr]) if Vbasis.size else X_arr
    Q = orthonormalize_columns(combined, dtype=Vbasis.dtype, eps=eps)
    if max_cols is not None and Q.shape[1] > max_cols:
        Q = Q[:, :max_cols]
    return np.ascontiguousarray(Q, dtype=Vbasis.dtype)


def build_entropy_fast_subspace(M_gain, active_r, q_subspace=None, method="setup_svd", q_oversample=16, dtype=np.float32):
    M_arr = np.asarray(M_gain, dtype=dtype)
    min_dim = min(M_arr.shape)
    if min_dim <= 1:
        raise ValueError("M_gain too small for reduced entropy subspace.")

    if q_subspace is None:
        q_subspace = min(max(2 * active_r, active_r + q_oversample, 64), min_dim - 1)
    q_subspace = max(active_r, min(int(q_subspace), min_dim - 1))

    if method == "setup_svd":
        _, s, vh = np.linalg.svd(M_arr, full_matrices=False)
        Vred = np.ascontiguousarray(vh[:q_subspace, :].T, dtype=dtype)
        sred = np.asarray(s[:q_subspace], dtype=float)
    elif method == "lanczos":
        _, s, vh = sp.sparse.linalg.svds(M_arr, k=q_subspace, which="LM")
        order = np.argsort(s)[::-1]
        vh = vh[order, :]
        s = s[order]
        Vred = np.ascontiguousarray(vh.T, dtype=dtype)
        sred = np.asarray(s, dtype=float)
    else:
        raise ValueError(f"Unknown subspace builder: {method}")

    Bred = np.ascontiguousarray(M_arr @ Vred, dtype=dtype)
    return Vred, Bred, sred


def entropyscore_forget_logscore_grad_rows(A_block, v, rows_ref):
    A_arr = np.asarray(A_block)
    work_dtype = A_arr.dtype
    v_work = np.ascontiguousarray(np.asarray(v, dtype=work_dtype).reshape(-1))
    y = np.ascontiguousarray(A_arr @ v_work, dtype=work_dtype)
    y2_sq = max(float(np.dot(y, y)), 1e-30)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 1e-30)
    rows_block = max(int(A_arr.shape[0]), 2)
    rows_ref = max(int(rows_ref), rows_block)
    c = np.log(rows_block / rows_ref) / np.log(rows_block)
    alpha = (rows_block / rows_ref) ** 0.25
    logpsi = np.log(alpha) + (1.0 - 0.5 * c) * np.log(y2_sq) + 0.25 * c * np.log(y4_4)
    y3 = np.ascontiguousarray(y * y * y, dtype=work_dtype)
    aty = np.ascontiguousarray(A_arr.T @ y, dtype=work_dtype)
    aty3 = np.ascontiguousarray(A_arr.T @ y3, dtype=work_dtype)
    grad = (2.0 - c) * (aty / y2_sq) + c * (aty3 / y4_4)
    H = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    s = float(np.sqrt(y2_sq))
    return logpsi, grad, s, H


def entropyscore_forget_streaming_logscore_grad(M_gain, A_block, V_old, s2_old, v, rows_ref):
    M_gain_arr = np.asarray(M_gain)
    A_block_arr = np.asarray(A_block)
    work_dtype = M_gain_arr.dtype
    v_work = np.ascontiguousarray(np.asarray(v, dtype=work_dtype).reshape(-1))
    V_old_work = np.ascontiguousarray(np.asarray(V_old, dtype=work_dtype))
    s2_old_work = np.asarray(s2_old, dtype=work_dtype)

    gain_vec = np.ascontiguousarray(M_gain_arr @ v_work, dtype=work_dtype)
    gain2 = max(float(np.dot(gain_vec, gain_vec)), 1e-30)
    a = np.ascontiguousarray(V_old_work.T @ v_work, dtype=work_dtype)
    y = np.ascontiguousarray(A_block_arr @ v_work, dtype=work_dtype)
    y2_sq = max(float(np.dot(y, y)), 1e-30)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 1e-30)
    rows_block = max(int(A_block_arr.shape[0]), 2)
    rows_ref = max(int(rows_ref), rows_block)
    c = np.log(rows_block / rows_ref) / np.log(rows_block)
    alpha = (rows_block / rows_ref) ** 0.25

    E_old = float(np.sum((a ** 2) * s2_old_work))
    logpsi = np.log(alpha) + (1.0 - 0.5 * c) * np.log(y2_sq) + 0.25 * c * np.log(y4_4)
    psi = float(np.exp(logpsi))
    score = max(E_old + psi, 1e-30)
    logf = np.log(score)

    y3 = np.ascontiguousarray(y * y * y, dtype=work_dtype)
    a_old = np.ascontiguousarray(V_old_work @ (s2_old_work * a), dtype=work_dtype)
    aty = np.ascontiguousarray(A_block_arr.T @ y, dtype=work_dtype)
    aty3 = np.ascontiguousarray(A_block_arr.T @ y3, dtype=work_dtype)
    gpsi = psi * ((2.0 - c) * (aty / y2_sq) + c * (aty3 / y4_4))
    grad_score = 2.0 * a_old + gpsi
    grad = grad_score / score
    H = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    s = float(np.sqrt(gain2))
    return logf, grad, s, H


def row_concentration_penalty_grad(B, z):
    B_arr = np.asarray(B)
    work_dtype = B_arr.dtype
    z_work = np.ascontiguousarray(np.asarray(z, dtype=work_dtype).reshape(-1))
    y = np.ascontiguousarray(B_arr @ z_work, dtype=work_dtype)
    y2 = y * y
    y2_sum = max(float(np.sum(y2)), 1e-30)
    y4_sum = max(float(np.sum(y2 * y2)), 1e-30)
    penalty = y4_sum / (y2_sum * y2_sum)
    dy = np.ascontiguousarray(4.0 * y * (y2 * y2_sum - y4_sum) / (y2_sum ** 3), dtype=work_dtype)
    grad = np.ascontiguousarray(B_arr.T @ dy, dtype=work_dtype)
    return float(penalty), grad


def row_leverage_penalty_grad(B, z, row_leverage_weights):
    B_arr = np.asarray(B)
    work_dtype = B_arr.dtype
    z_work = np.ascontiguousarray(np.asarray(z, dtype=work_dtype).reshape(-1))
    weights = np.asarray(row_leverage_weights, dtype=work_dtype).reshape(-1)
    if weights.shape[0] != B_arr.shape[0]:
        raise ValueError("row_leverage_weights length must match B_block rows.")
    y = np.ascontiguousarray(B_arr @ z_work, dtype=work_dtype)
    y2 = y * y
    y2_sum = max(float(np.sum(y2)), 1e-30)
    penalty = float(np.dot(weights, y2) / y2_sum)
    dy = np.ascontiguousarray(2.0 * y * (weights - penalty) / y2_sum, dtype=work_dtype)
    grad = np.ascontiguousarray(B_arr.T @ dy, dtype=work_dtype)
    return penalty, grad


def apply_row_regularizers(
    score, grad, B_block, z, row_concentration_lambda=0.0,
    row_leverage_lambda=0.0, row_leverage_weights=None
):
    reg_score = float(score)
    reg_grad = np.ascontiguousarray(grad)
    lam = float(row_concentration_lambda)
    if lam != 0.0:
        penalty, grad_penalty = row_concentration_penalty_grad(B_block, z)
        reg_score -= lam * penalty
        reg_grad = np.ascontiguousarray(reg_grad - lam * grad_penalty, dtype=reg_grad.dtype)
    lev_lam = float(row_leverage_lambda)
    if lev_lam != 0.0:
        if row_leverage_weights is None:
            raise ValueError("row_leverage_weights is required when row_leverage_lambda != 0.")
        penalty, grad_penalty = row_leverage_penalty_grad(B_block, z, row_leverage_weights)
        reg_score -= lev_lam * penalty
        reg_grad = np.ascontiguousarray(reg_grad - lev_lam * grad_penalty, dtype=reg_grad.dtype)
    return reg_score, reg_grad


def apply_row_concentration_regularizer(score, grad, B_block, z, row_concentration_lambda=0.0):
    return apply_row_regularizers(
        score, grad, B_block, z, row_concentration_lambda=row_concentration_lambda
    )


def row_leverage_weights_from_block(B_block, mode="none", rank=2):
    mode = str(mode)
    B_arr = np.asarray(B_block, dtype=np.float64)
    if mode == "none" or B_arr.size == 0:
        return None
    if mode == "row-norm":
        weights = np.sum(B_arr * B_arr, axis=1)
    elif mode == "top-svd":
        k = max(1, min(int(rank), min(B_arr.shape)))
        if k >= min(B_arr.shape):
            U, _, _ = np.linalg.svd(B_arr, full_matrices=False)
        else:
            U, _, _ = np.linalg.svd(B_arr, full_matrices=False)
        weights = np.sum(U[:, :k] * U[:, :k], axis=1)
    else:
        raise ValueError(f"Unknown row_leverage_mode: {mode}")
    mean = max(float(np.mean(weights)), 1e-30)
    return np.ascontiguousarray(weights / mean, dtype=B_arr.dtype)


def entropyscore_forget_logscore_grad_reduced(B, z, rows_block, rows_ref):
    B_arr = np.asarray(B)
    work_dtype = B_arr.dtype
    z = np.ascontiguousarray(np.asarray(z, dtype=work_dtype).reshape(-1))
    y = np.ascontiguousarray(B_arr @ z, dtype=work_dtype)
    y2_sq = max(float(np.dot(y, y)), 1e-30)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 1e-30)
    rows_block = max(int(rows_block), 2)
    rows_ref = max(int(rows_ref), rows_block)
    c = np.log(rows_block / rows_ref) / np.log(rows_block)
    alpha = (rows_block / rows_ref) ** 0.25
    logf = np.log(alpha) + (1.0 - 0.5 * c) * np.log(y2_sq) + 0.25 * c * np.log(y4_4)

    y3 = np.ascontiguousarray(y * y * y, dtype=work_dtype)
    g2 = np.ascontiguousarray(B_arr.T @ y, dtype=work_dtype) / y2_sq
    g4 = np.ascontiguousarray(B_arr.T @ y3, dtype=work_dtype) / y4_4
    grad = (2.0 - c) * g2 + c * g4
    H = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    s = float(np.sqrt(y2_sq))
    return logf, grad, s, H


def entropyscore_forget_streaming_logscore_grad_reduced(B_gain, B_block, C_prev, s2_old, z, rows_block, rows_ref):
    B_gain_arr = np.asarray(B_gain)
    B_block_arr = np.asarray(B_block)
    work_dtype = B_gain_arr.dtype
    z = np.ascontiguousarray(np.asarray(z, dtype=work_dtype).reshape(-1))
    C_prev_arr = np.ascontiguousarray(np.asarray(C_prev, dtype=work_dtype))
    s2_old_arr = np.asarray(s2_old, dtype=work_dtype)

    gain_vec = np.ascontiguousarray(B_gain_arr @ z, dtype=work_dtype)
    gain2 = max(float(np.dot(gain_vec, gain_vec)), 1e-30)
    a = np.ascontiguousarray(C_prev_arr @ z, dtype=work_dtype)
    y = np.ascontiguousarray(B_block_arr @ z, dtype=work_dtype)
    y2_sq = max(float(np.dot(y, y)), 1e-30)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 1e-30)
    rows_block = max(int(rows_block), 2)
    rows_ref = max(int(rows_ref), rows_block)
    c = np.log(rows_block / rows_ref) / np.log(rows_block)
    alpha = (rows_block / rows_ref) ** 0.25

    E_old = float(np.sum((a ** 2) * s2_old_arr))
    logpsi = np.log(alpha) + (1.0 - 0.5 * c) * np.log(y2_sq) + 0.25 * c * np.log(y4_4)
    psi = float(np.exp(logpsi))
    score = max(E_old + psi, 1e-30)
    logf = np.log(score)

    y3 = np.ascontiguousarray(y * y * y, dtype=work_dtype)
    aty = np.ascontiguousarray(B_block_arr.T @ y, dtype=work_dtype)
    aty3 = np.ascontiguousarray(B_block_arr.T @ y3, dtype=work_dtype)
    g_old = np.ascontiguousarray(C_prev_arr.T @ (s2_old_arr * a), dtype=work_dtype)
    grad_score = 2.0 * g_old + psi * ((2.0 - c) * (aty / y2_sq) + c * (aty3 / y4_4))
    grad = grad_score / score
    H = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    s = float(np.sqrt(gain2))
    return logf, grad, s, H


def entropyscore_forget_score_grad_reduced(B, z, rows_block, rows_ref, row_concentration_lambda=0.0):
    logf, grad_log, s, H = entropyscore_forget_logscore_grad_reduced(B, z, rows_block, rows_ref)
    score = float(np.exp(logf))
    grad = np.ascontiguousarray(score * grad_log, dtype=np.asarray(grad_log).dtype)
    score, grad = apply_row_concentration_regularizer(score, grad, B, z, row_concentration_lambda)
    return score, grad, s, H


def entropyscore_forget_streaming_score_grad_reduced(
    B_gain, B_block, C_prev, s2_old, z, rows_block, rows_ref, row_concentration_lambda=0.0
):
    logf, grad_log, s, H = entropyscore_forget_streaming_logscore_grad_reduced(
        B_gain, B_block, C_prev, s2_old, z, rows_block, rows_ref
    )
    score = float(np.exp(logf))
    grad = np.ascontiguousarray(score * grad_log, dtype=np.asarray(grad_log).dtype)
    score, grad = apply_row_concentration_regularizer(score, grad, B_block, z, row_concentration_lambda)
    return score, grad, s, H


def oldcorrected_score_grad_reduced(B, z, rows_block, rows_ref, row_concentration_lambda=0.0):
    B_arr = np.asarray(B)
    work_dtype = B_arr.dtype
    z = np.ascontiguousarray(np.asarray(z, dtype=work_dtype).reshape(-1))
    y = np.ascontiguousarray(B_arr @ z, dtype=work_dtype)
    y2_sq = max(float(np.dot(y, y)), 1e-30)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 1e-30)
    rows_block = max(int(rows_block), 2)
    rows_ref = max(int(rows_ref), rows_block)
    c = np.log(rows_block / rows_ref) / np.log(rows_block)

    score = float(np.exp((1.0 - 0.5 * c) * np.log(y2_sq) + 0.25 * c * np.log(y4_4)))
    y3 = np.ascontiguousarray(y * y * y, dtype=work_dtype)
    g2 = np.ascontiguousarray(B_arr.T @ y, dtype=work_dtype) / y2_sq
    g4 = np.ascontiguousarray(B_arr.T @ y3, dtype=work_dtype) / y4_4
    grad = np.ascontiguousarray(score * ((2.0 - c) * g2 + c * g4), dtype=work_dtype)
    H = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    s = float(np.sqrt(y2_sq))
    score, grad = apply_row_concentration_regularizer(score, grad, B, z, row_concentration_lambda)
    return score, grad, s, H


def oldcorrected_streaming_score_grad_reduced(
    B_gain, B_block, C_prev, s2_old, R_old_block, z, rows_block, rows_ref,
    n_old, k_old=None, row_concentration_lambda=0.0
):
    del k_old
    B_gain_arr = np.asarray(B_gain)
    B_block_arr = np.asarray(B_block)
    work_dtype = B_gain_arr.dtype
    z = np.ascontiguousarray(np.asarray(z, dtype=work_dtype).reshape(-1))
    C_prev_arr = np.ascontiguousarray(np.asarray(C_prev, dtype=work_dtype))
    s2_old_arr = np.asarray(s2_old, dtype=work_dtype)

    gain_vec = np.ascontiguousarray(B_gain_arr @ z, dtype=work_dtype)
    gain2 = max(float(np.dot(gain_vec, gain_vec)), 1e-30)

    a = np.ascontiguousarray(C_prev_arr @ z, dtype=work_dtype)
    y = np.ascontiguousarray(B_block_arr @ z, dtype=work_dtype)
    y2_sq = max(float(np.dot(y, y)), 1e-30)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 1e-30)
    rows_block = max(int(rows_block), 2)
    rows_ref = max(int(rows_ref), rows_block)
    c_new = np.log(rows_block / rows_ref) / np.log(rows_block)

    psi_new = float(np.exp((1.0 - 0.5 * c_new) * np.log(y2_sq) + 0.25 * c_new * np.log(y4_4)))
    y3 = np.ascontiguousarray(y * y * y, dtype=work_dtype)
    aty = np.ascontiguousarray(B_block_arr.T @ y, dtype=work_dtype)
    aty3 = np.ascontiguousarray(B_block_arr.T @ y3, dtype=work_dtype)
    grad_new = psi_new * ((2.0 - c_new) * (aty / y2_sq) + c_new * (aty3 / y4_4))

    E_old = float(np.sum((a ** 2) * s2_old_arr))
    g_old_energy = np.ascontiguousarray(C_prev_arr.T @ (s2_old_arr * a), dtype=work_dtype)
    R_old_arr = None if R_old_block is None else np.asarray(R_old_block, dtype=work_dtype)
    if R_old_arr is None or R_old_arr.size == 0 or R_old_arr.shape[0] <= 1:
        phi_old = 1.0
        psi_old = E_old
        grad_old = 2.0 * g_old_energy
        H_old_proxy = np.nan
    else:
        s_sample = max(int(R_old_arr.shape[0]), 2)
        n_old_eff = max(int(n_old), s_sample, 2)
        rows_ref_eff = max(int(rows_ref), n_old_eff)
        c_old = np.log(n_old_eff / rows_ref_eff) / np.log(s_sample)
        r = np.ascontiguousarray(R_old_arr @ z, dtype=work_dtype)
        r2_sq = max(float(np.dot(r, r)), 1e-30)
        r4_4 = max(float(np.sum((r * r) * (r * r))), 1e-30)
        phi_old = float(np.exp(-0.5 * c_old * np.log(r2_sq) + 0.25 * c_old * np.log(r4_4)))
        r3 = np.ascontiguousarray(r * r * r, dtype=work_dtype)
        rt_r = np.ascontiguousarray(R_old_arr.T @ r, dtype=work_dtype)
        rt_r3 = np.ascontiguousarray(R_old_arr.T @ r3, dtype=work_dtype)
        grad_phi_log = (-c_old) * (rt_r / r2_sq) + c_old * (rt_r3 / r4_4)
        psi_old = E_old * phi_old
        grad_old = 2.0 * phi_old * g_old_energy + psi_old * grad_phi_log
        H_old_proxy = -(np.log(r4_4) - 2.0 * np.log(r2_sq))

    score = max(float(psi_old + psi_new), 1e-30)
    grad = np.ascontiguousarray(grad_old + grad_new, dtype=work_dtype)
    H = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    s = float(np.sqrt(gain2))
    score, grad = apply_row_concentration_regularizer(score, grad, B_block_arr, z, row_concentration_lambda)
    return score, grad, s, H


def combined_score_grad_reduced(
    B, z, rows_block, rows_ref, row_concentration_lambda=0.0,
    row_leverage_lambda=0.0, row_leverage_weights=None
):
    B_arr = np.asarray(B)
    work_dtype = B_arr.dtype
    z = np.ascontiguousarray(np.asarray(z, dtype=work_dtype).reshape(-1))
    y = np.ascontiguousarray(B_arr @ z, dtype=work_dtype)
    y2_sq = max(float(np.dot(y, y)), 1e-30)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 1e-30)
    rows_block = max(int(rows_block), 2)
    rows_ref = max(int(rows_ref), rows_block)
    c = np.log(rows_block / rows_ref) / (2.0 * np.log(rows_block))

    log_phi = c * (np.log(y4_4) - 2.0 * np.log(y2_sq))
    score = float(np.exp(np.log(y2_sq) + log_phi))
    y3 = np.ascontiguousarray(y * y * y, dtype=work_dtype)
    g2 = np.ascontiguousarray(B_arr.T @ y, dtype=work_dtype) / y2_sq
    g4 = np.ascontiguousarray(B_arr.T @ y3, dtype=work_dtype) / y4_4
    grad = np.ascontiguousarray(score * ((2.0 - 4.0 * c) * g2 + 4.0 * c * g4), dtype=work_dtype)
    H = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    s = float(np.sqrt(y2_sq))
    score, grad = apply_row_regularizers(
        score, grad, B, z,
        row_concentration_lambda=row_concentration_lambda,
        row_leverage_lambda=row_leverage_lambda,
        row_leverage_weights=row_leverage_weights,
    )
    return score, grad, s, H


def combined_streaming_score_grad_reduced(
    B_gain, B_block, C_prev, s2_old, R_old_block, z, rows_block, rows_ref,
    n_old, k_old=None, row_concentration_lambda=0.0,
    row_leverage_lambda=0.0, row_leverage_weights=None
):
    del C_prev, s2_old, k_old
    B_gain_arr = np.asarray(B_gain)
    B_block_arr = np.asarray(B_block)
    work_dtype = B_gain_arr.dtype
    z = np.ascontiguousarray(np.asarray(z, dtype=work_dtype).reshape(-1))

    gain_vec = np.ascontiguousarray(B_gain_arr @ z, dtype=work_dtype)
    gain2 = max(float(np.dot(gain_vec, gain_vec)), 1e-30)
    grad_energy = np.ascontiguousarray(2.0 * (B_gain_arr.T @ gain_vec), dtype=work_dtype)

    y = np.ascontiguousarray(B_block_arr @ z, dtype=work_dtype)
    y2_sq = max(float(np.dot(y, y)), 0.0)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 0.0)
    cy = np.ascontiguousarray(B_block_arr.T @ y, dtype=work_dtype)
    y3 = np.ascontiguousarray(y * y * y, dtype=work_dtype)
    cy3 = np.ascontiguousarray(B_block_arr.T @ y3, dtype=work_dtype)
    rows_entropy = max(int(rows_block), 0)

    R_old_arr = None if R_old_block is None else np.asarray(R_old_block, dtype=work_dtype)
    if R_old_arr is not None and R_old_arr.size and R_old_arr.shape[0] > 0:
        r = np.ascontiguousarray(R_old_arr @ z, dtype=work_dtype)
        y2_sq += max(float(np.dot(r, r)), 0.0)
        y4_4 += max(float(np.sum((r * r) * (r * r))), 0.0)
        cy = np.ascontiguousarray(cy + R_old_arr.T @ r, dtype=work_dtype)
        r3 = np.ascontiguousarray(r * r * r, dtype=work_dtype)
        cy3 = np.ascontiguousarray(cy3 + R_old_arr.T @ r3, dtype=work_dtype)
        rows_entropy += int(R_old_arr.shape[0])

    y2_sq = max(y2_sq, 1e-30)
    y4_4 = max(y4_4, 1e-30)
    rows_entropy = max(rows_entropy, 2)
    rows_ref = max(int(rows_ref), rows_entropy)
    rows_seen = min(max(int(n_old) + int(rows_block), 1), rows_ref)
    c = np.log(rows_seen / rows_ref) / (2.0 * np.log(rows_entropy))

    log_phi = c * (np.log(y4_4) - 2.0 * np.log(y2_sq))
    phi = float(np.exp(log_phi))
    score = max(float(gain2 * phi), 1e-30)
    grad_log_phi = np.ascontiguousarray(4.0 * c * (cy3 / y4_4 - cy / y2_sq), dtype=work_dtype)
    grad = np.ascontiguousarray(phi * grad_energy + score * grad_log_phi, dtype=work_dtype)
    H = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    s = float(np.sqrt(gain2))
    score, grad = apply_row_regularizers(
        score, grad, B_block_arr, z,
        row_concentration_lambda=row_concentration_lambda,
        row_leverage_lambda=row_leverage_lambda,
        row_leverage_weights=row_leverage_weights,
    )
    return score, grad, s, H


def combined_streaming_score_grad_reduced_with_aux(
    B_gain, B_block, C_prev, s2_old, R_old_block, z, rows_block, rows_ref,
    n_old, k_old=None, row_concentration_lambda=0.0,
    row_leverage_lambda=0.0, row_leverage_weights=None,
    A_aux=None,
):
    # S7-by-spec extension: pool A_aux (peek/future rows) into the energy term
    # and the entropy pool, and add row(A_aux) to rows_seen. When A_aux is None
    # or empty, defers to the unmodified combined kernel byte-for-byte.
    A_aux_arr = None if A_aux is None else np.asarray(A_aux)
    if A_aux_arr is None or A_aux_arr.size == 0 or A_aux_arr.shape[0] == 0:
        return combined_streaming_score_grad_reduced(
            B_gain, B_block, C_prev, s2_old, R_old_block, z, rows_block, rows_ref,
            n_old, k_old=k_old,
            row_concentration_lambda=row_concentration_lambda,
            row_leverage_lambda=row_leverage_lambda,
            row_leverage_weights=row_leverage_weights,
        )
    B_gain_arr = np.asarray(B_gain)
    B_block_arr = np.asarray(B_block)
    work_dtype = np.result_type(B_gain_arr.dtype, A_aux_arr.dtype)
    A_aux_cast = np.ascontiguousarray(A_aux_arr.astype(work_dtype, copy=False))
    B_gain_aug = np.vstack([B_gain_arr.astype(work_dtype, copy=False), A_aux_cast])
    B_block_aug = np.vstack([B_block_arr.astype(work_dtype, copy=False), A_aux_cast])
    rows_block_aug = int(rows_block) + int(A_aux_cast.shape[0])
    return combined_streaming_score_grad_reduced(
        B_gain_aug, B_block_aug, C_prev, s2_old, R_old_block, z, rows_block_aug, rows_ref,
        n_old, k_old=k_old,
        row_concentration_lambda=row_concentration_lambda,
        row_leverage_lambda=row_leverage_lambda,
        row_leverage_weights=row_leverage_weights,
    )


def l4l2_score_grad_reduced(
    B, z, rows_block, rows_ref, row_concentration_lambda=0.0
):
    del rows_block, rows_ref
    B_arr = np.asarray(B)
    work_dtype = B_arr.dtype
    z = np.ascontiguousarray(np.asarray(z, dtype=work_dtype).reshape(-1))
    y = np.ascontiguousarray(B_arr @ z, dtype=work_dtype)
    y2_sq = max(float(np.dot(y, y)), 1e-30)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 1e-30)
    gain2 = y2_sq

    theta = float(np.exp(np.log(y2_sq) - 0.5 * np.log(y4_4)))
    score = max(float(gain2 * theta), 1e-30)
    y3 = np.ascontiguousarray(y * y * y, dtype=work_dtype)
    cy = np.ascontiguousarray(B_arr.T @ y, dtype=work_dtype)
    cy3 = np.ascontiguousarray(B_arr.T @ y3, dtype=work_dtype)
    grad_energy = np.ascontiguousarray(2.0 * cy, dtype=work_dtype)
    grad_log_theta = np.ascontiguousarray(2.0 * (cy / y2_sq - cy3 / y4_4), dtype=work_dtype)
    grad = np.ascontiguousarray(theta * grad_energy + score * grad_log_theta, dtype=work_dtype)
    H = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    s = float(np.sqrt(gain2))
    score, grad = apply_row_regularizers(
        score, grad, B, z,
        row_concentration_lambda=row_concentration_lambda,
    )
    return score, grad, s, H


def l4l2_streaming_score_grad_reduced(
    B_gain, B_block, C_prev, s2_old, R_old_block, z, rows_block, rows_ref,
    n_old, k_old=None, row_concentration_lambda=0.0
):
    del C_prev, s2_old, rows_block, rows_ref, n_old, k_old
    B_gain_arr = np.asarray(B_gain)
    B_block_arr = np.asarray(B_block)
    work_dtype = B_gain_arr.dtype
    z = np.ascontiguousarray(np.asarray(z, dtype=work_dtype).reshape(-1))

    gain_vec = np.ascontiguousarray(B_gain_arr @ z, dtype=work_dtype)
    gain2 = max(float(np.dot(gain_vec, gain_vec)), 1e-30)
    grad_energy = np.ascontiguousarray(2.0 * (B_gain_arr.T @ gain_vec), dtype=work_dtype)

    y = np.ascontiguousarray(B_block_arr @ z, dtype=work_dtype)
    y2_sq = max(float(np.dot(y, y)), 0.0)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 0.0)
    cy = np.ascontiguousarray(B_block_arr.T @ y, dtype=work_dtype)
    y3 = np.ascontiguousarray(y * y * y, dtype=work_dtype)
    cy3 = np.ascontiguousarray(B_block_arr.T @ y3, dtype=work_dtype)

    R_old_arr = None if R_old_block is None else np.asarray(R_old_block, dtype=work_dtype)
    if R_old_arr is not None and R_old_arr.size and R_old_arr.shape[0] > 0:
        r = np.ascontiguousarray(R_old_arr @ z, dtype=work_dtype)
        y2_sq += max(float(np.dot(r, r)), 0.0)
        y4_4 += max(float(np.sum((r * r) * (r * r))), 0.0)
        cy = np.ascontiguousarray(cy + R_old_arr.T @ r, dtype=work_dtype)
        r3 = np.ascontiguousarray(r * r * r, dtype=work_dtype)
        cy3 = np.ascontiguousarray(cy3 + R_old_arr.T @ r3, dtype=work_dtype)

    y2_sq = max(y2_sq, 1e-30)
    y4_4 = max(y4_4, 1e-30)
    theta = float(np.exp(np.log(y2_sq) - 0.5 * np.log(y4_4)))
    score = max(float(gain2 * theta), 1e-30)
    grad_log_theta = np.ascontiguousarray(2.0 * (cy / y2_sq - cy3 / y4_4), dtype=work_dtype)
    grad = np.ascontiguousarray(theta * grad_energy + score * grad_log_theta, dtype=work_dtype)
    H = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    s = float(np.sqrt(gain2))
    score, grad = apply_row_regularizers(
        score, grad, B_block_arr, z,
        row_concentration_lambda=row_concentration_lambda,
    )
    return score, grad, s, H


SUBSETMASS_DENOM_EPS = 1e-6


def subsetmass_score_grad_reduced(
    B, z, rows_block, rows_ref, row_concentration_lambda=0.0
):
    B_arr = np.asarray(B)
    work_dtype = B_arr.dtype
    z = np.ascontiguousarray(np.asarray(z, dtype=work_dtype).reshape(-1))
    y = np.ascontiguousarray(B_arr @ z, dtype=work_dtype)
    y2_sq = max(float(np.dot(y, y)), 1e-30)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 1e-30)
    rows_block = max(int(rows_block), 2)
    rows_ref = max(int(rows_ref), rows_block)
    q_w = max(rows_ref - rows_block, 0)
    tau_w = np.log(rows_ref) / max(np.log(rows_block), 1e-30)

    # psi = sqrt(q_w) * (y4_4 / y2_sq^2) ** (tau_w / 2)
    if q_w == 0:
        psi = 0.0
    else:
        log_psi = 0.5 * np.log(q_w) + (tau_w / 2.0) * (np.log(y4_4) - 2.0 * np.log(y2_sq))
        psi = float(np.exp(log_psi))
    denom = max(1.0 - psi, SUBSETMASS_DENOM_EPS)
    score = float(y2_sq / denom)

    y3 = np.ascontiguousarray(y * y * y, dtype=work_dtype)
    g2 = np.ascontiguousarray(B_arr.T @ y, dtype=work_dtype) / y2_sq
    g4 = np.ascontiguousarray(B_arr.T @ y3, dtype=work_dtype) / y4_4

    if q_w == 0:
        # ψ ≡ 0: score reduces to y2_sq; gradient is 2 B^T B v.
        grad = np.ascontiguousarray(2.0 * (g2 * y2_sq), dtype=work_dtype)
    else:
        kappa = tau_w * psi / denom
        grad_log = (2.0 - 2.0 * kappa) * g2 + 2.0 * kappa * g4
        grad = np.ascontiguousarray(score * grad_log, dtype=work_dtype)

    H = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    s = float(np.sqrt(y2_sq))
    score, grad = apply_row_regularizers(
        score, grad, B, z,
        row_concentration_lambda=row_concentration_lambda,
    )
    return score, grad, s, H


def subsetmass_streaming_score_grad_reduced(
    B_gain, B_block, C_prev, s2_old, R_old_block, z, rows_block, rows_ref,
    n_old, k_old=None, row_concentration_lambda=0.0
):
    del C_prev, s2_old, k_old
    B_gain_arr = np.asarray(B_gain)
    B_block_arr = np.asarray(B_block)
    work_dtype = B_gain_arr.dtype
    z = np.ascontiguousarray(np.asarray(z, dtype=work_dtype).reshape(-1))

    gain_vec = np.ascontiguousarray(B_gain_arr @ z, dtype=work_dtype)
    gain2 = max(float(np.dot(gain_vec, gain_vec)), 1e-30)
    grad_energy = np.ascontiguousarray(2.0 * (B_gain_arr.T @ gain_vec), dtype=work_dtype)

    y = np.ascontiguousarray(B_block_arr @ z, dtype=work_dtype)
    y2_sq = max(float(np.dot(y, y)), 0.0)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 0.0)
    cy = np.ascontiguousarray(B_block_arr.T @ y, dtype=work_dtype)
    y3 = np.ascontiguousarray(y * y * y, dtype=work_dtype)
    cy3 = np.ascontiguousarray(B_block_arr.T @ y3, dtype=work_dtype)
    rows_entropy = max(int(rows_block), 0)

    R_old_arr = None if R_old_block is None else np.asarray(R_old_block, dtype=work_dtype)
    if R_old_arr is not None and R_old_arr.size and R_old_arr.shape[0] > 0:
        r = np.ascontiguousarray(R_old_arr @ z, dtype=work_dtype)
        y2_sq += max(float(np.dot(r, r)), 0.0)
        y4_4 += max(float(np.sum((r * r) * (r * r))), 0.0)
        cy = np.ascontiguousarray(cy + R_old_arr.T @ r, dtype=work_dtype)
        r3 = np.ascontiguousarray(r * r * r, dtype=work_dtype)
        cy3 = np.ascontiguousarray(cy3 + R_old_arr.T @ r3, dtype=work_dtype)
        rows_entropy += int(R_old_arr.shape[0])

    y2_sq = max(y2_sq, 1e-30)
    y4_4 = max(y4_4, 1e-30)
    rows_entropy = max(rows_entropy, 2)
    rows_ref = max(int(rows_ref), rows_entropy)
    rows_seen = min(max(int(n_old) + int(rows_block), 1), rows_ref)
    q_w = max(rows_ref - rows_seen, 0)
    tau_w = np.log(rows_ref) / max(np.log(rows_entropy), 1e-30)

    if q_w == 0:
        psi = 0.0
    else:
        log_psi = 0.5 * np.log(q_w) + (tau_w / 2.0) * (np.log(y4_4) - 2.0 * np.log(y2_sq))
        psi = float(np.exp(log_psi))
    denom = max(1.0 - psi, SUBSETMASS_DENOM_EPS)
    score = float(gain2 / denom)

    if q_w == 0:
        grad = np.ascontiguousarray(grad_energy, dtype=work_dtype)
    else:
        kappa = tau_w * psi / denom
        grad_h2 = np.ascontiguousarray(4.0 * (cy / y2_sq - cy3 / y4_4), dtype=work_dtype)
        # grad log score = grad_energy / gain2 - kappa/2 * grad_h2
        grad_log = grad_energy / gain2 - 0.5 * kappa * grad_h2
        grad = np.ascontiguousarray(score * grad_log, dtype=work_dtype)

    H = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    s = float(np.sqrt(gain2))
    score, grad = apply_row_regularizers(
        score, grad, B_block_arr, z,
        row_concentration_lambda=row_concentration_lambda,
    )
    return score, grad, s, H


def score_grad_reduced_by_variant(
    score_variant, B, z, rows_block, rows_ref, row_concentration_lambda=0.0,
    row_leverage_lambda=0.0, row_leverage_weights=None
):
    if float(row_leverage_lambda) != 0.0 and score_variant != "combined":
        raise ValueError("row_leverage_lambda is currently supported only for score_variant='combined'.")
    if score_variant == "oldcorrected":
        return oldcorrected_score_grad_reduced(
            B, z, rows_block, rows_ref, row_concentration_lambda=row_concentration_lambda
        )
    if score_variant == "combined":
        return combined_score_grad_reduced(
            B, z, rows_block, rows_ref,
            row_concentration_lambda=row_concentration_lambda,
            row_leverage_lambda=row_leverage_lambda,
            row_leverage_weights=row_leverage_weights,
        )
    if score_variant == "l4l2":
        return l4l2_score_grad_reduced(
            B, z, rows_block, rows_ref,
            row_concentration_lambda=row_concentration_lambda,
        )
    if score_variant == "subsetmass":
        return subsetmass_score_grad_reduced(
            B, z, rows_block, rows_ref,
            row_concentration_lambda=row_concentration_lambda,
        )
    return entropyscore_forget_score_grad_reduced(
        B, z, rows_block, rows_ref, row_concentration_lambda=row_concentration_lambda
    )


def streaming_score_grad_reduced_by_variant(
    score_variant, B_gain, B_block, C_prev, s2_old, R_old_block, z, rows_block, rows_ref,
    n_old, k_old=None, row_concentration_lambda=0.0,
    row_leverage_lambda=0.0, row_leverage_weights=None
):
    if float(row_leverage_lambda) != 0.0 and score_variant != "combined":
        raise ValueError("row_leverage_lambda is currently supported only for score_variant='combined'.")
    if score_variant == "oldcorrected":
        return oldcorrected_streaming_score_grad_reduced(
            B_gain, B_block, C_prev, s2_old, R_old_block, z, rows_block, rows_ref,
            n_old=n_old, k_old=k_old, row_concentration_lambda=row_concentration_lambda
        )
    if score_variant == "combined":
        return combined_streaming_score_grad_reduced(
            B_gain, B_block, C_prev, s2_old, R_old_block, z, rows_block, rows_ref,
            n_old=n_old, k_old=k_old,
            row_concentration_lambda=row_concentration_lambda,
            row_leverage_lambda=row_leverage_lambda,
            row_leverage_weights=row_leverage_weights,
        )
    if score_variant == "l4l2":
        return l4l2_streaming_score_grad_reduced(
            B_gain, B_block, C_prev, s2_old, R_old_block, z, rows_block, rows_ref,
            n_old=n_old, k_old=k_old,
            row_concentration_lambda=row_concentration_lambda,
        )
    if score_variant == "subsetmass":
        return subsetmass_streaming_score_grad_reduced(
            B_gain, B_block, C_prev, s2_old, R_old_block, z, rows_block, rows_ref,
            n_old=n_old, k_old=k_old,
            row_concentration_lambda=row_concentration_lambda,
        )
    return entropyscore_forget_streaming_score_grad_reduced(
        B_gain, B_block, C_prev, s2_old, z, rows_block, rows_ref,
        row_concentration_lambda=row_concentration_lambda
    )


def basic_projected_ascent_single_reduced_forget_cex(
    B, z0, Qz, rows_block, rows_ref, maxit=80, tol=1e-8, reuse_line_search_grad=True,
    row_concentration_lambda=0.0, score_variant="legacy",
    row_leverage_lambda=0.0, row_leverage_weights=None,
    patience=0, patience_rel_tol=1e-5,
):
    z = retract_reduced(z0, Qz)
    if z is None:
        raise RuntimeError("Initial reduced seed is infeasible.")

    score, grad, s, H = score_grad_reduced_by_variant(
        score_variant, B, z, rows_block, rows_ref,
        row_concentration_lambda=row_concentration_lambda,
        row_leverage_lambda=row_leverage_lambda,
        row_leverage_weights=row_leverage_weights,
    )
    stop = {"reason": "maxit", "iters": maxit, "grad_norm": np.nan}
    progress_f_tol = 1e-12
    progress_step_tol = 1e-10
    plateau_count = 0

    for it in range(maxit):
        gtan = project_reduced(grad - z * float(z @ grad), Qz)
        gnorm = float(np.linalg.norm(gtan))
        if gnorm <= tol:
            stop = {"reason": "grad_tol", "iters": it, "grad_norm": gnorm}
            break

        accepted = False
        accepted_eval = None
        alpha = 1.0
        score_old = score
        z_old = z

        for ls_iter in range(20):
            zt = retract_reduced(z + alpha * gtan, Qz)
            if zt is not None:
                score_t, grad_t, s_t, H_t = score_grad_reduced_by_variant(
                    score_variant, B, zt, rows_block, rows_ref,
                    row_concentration_lambda=row_concentration_lambda,
                    row_leverage_lambda=row_leverage_lambda,
                    row_leverage_weights=row_leverage_weights,
                )
                rhs = score_old + 1e-4 * alpha * float(gtan @ gtan)
                if score_t >= rhs:
                    z = zt
                    accepted_eval = (score_t, grad_t, s_t, H_t)
                    accepted = True
                    break
            alpha *= 0.5

        if not accepted:
            stop = {"reason": "line_search_fail", "iters": it + 1, "grad_norm": gnorm, "line_search_steps": 20}
            z = z_old
            break

        if reuse_line_search_grad and accepted_eval is not None:
            score, grad, s, H = accepted_eval
        else:
            score, grad, s, H = score_grad_reduced_by_variant(
                score_variant, B, z, rows_block, rows_ref,
                row_concentration_lambda=row_concentration_lambda,
                row_leverage_lambda=row_leverage_lambda,
                row_leverage_weights=row_leverage_weights,
            )
        step_norm = float(np.linalg.norm(z - z_old))
        f_change = abs(score - score_old)
        f_threshold = progress_f_tol * max(1.0, abs(score_old))
        stop = {
            "reason": "progress",
            "iters": it + 1,
            "grad_norm": gnorm,
            "step_norm": step_norm,
            "f_change": f_change,
            "f_threshold": f_threshold,
            "line_search_alpha": alpha,
            "line_search_steps": ls_iter + 1,
        }
        if f_change <= f_threshold:
            stop["reason"] = "f_change_tol"
            break
        if step_norm <= progress_step_tol:
            stop["reason"] = "step_tol"
            break
        if patience > 0:
            rel_change = f_change / max(abs(score_old), 1e-30)
            if rel_change <= patience_rel_tol:
                plateau_count += 1
                if plateau_count >= patience:
                    stop["reason"] = "patience"
                    stop["plateau_count"] = plateau_count
                    break
            else:
                plateau_count = 0
    else:
        gtan = project_reduced(grad - z * float(z @ grad), Qz)
        stop = {"reason": "maxit", "iters": maxit, "grad_norm": float(np.linalg.norm(gtan))}

    return z, score, s, H, stop


def basic_projected_ascent_single_reduced_streaming_forget_cex(
    B_gain, B_block, C_prev, s2_old, z0, Qz, rows_block, rows_ref,
    maxit=80, tol=1e-8, reuse_line_search_grad=True, row_concentration_lambda=0.0,
    score_variant="legacy", R_old_block=None, n_old=0, k_old=None,
    row_leverage_lambda=0.0, row_leverage_weights=None,
    patience=0, patience_rel_tol=1e-5,
):
    z = retract_reduced(z0, Qz)
    if z is None:
        raise RuntimeError("Initial reduced seed is infeasible.")

    score, grad, s, H = streaming_score_grad_reduced_by_variant(
        score_variant, B_gain, B_block, C_prev, s2_old, R_old_block, z, rows_block, rows_ref,
        n_old=n_old, k_old=k_old, row_concentration_lambda=row_concentration_lambda,
        row_leverage_lambda=row_leverage_lambda, row_leverage_weights=row_leverage_weights,
    )
    stop = {"reason": "maxit", "iters": maxit, "grad_norm": np.nan}
    progress_f_tol = 1e-12
    progress_step_tol = 1e-10
    plateau_count = 0

    for it in range(maxit):
        gtan = project_reduced(grad - z * float(z @ grad), Qz)
        gnorm = float(np.linalg.norm(gtan))
        if gnorm <= tol:
            stop = {"reason": "grad_tol", "iters": it, "grad_norm": gnorm}
            break

        accepted = False
        accepted_eval = None
        alpha = 1.0
        score_old = score
        z_old = z

        for ls_iter in range(20):
            zt = retract_reduced(z + alpha * gtan, Qz)
            if zt is not None:
                score_t, grad_t, s_t, H_t = streaming_score_grad_reduced_by_variant(
                    score_variant, B_gain, B_block, C_prev, s2_old, R_old_block, zt,
                    rows_block, rows_ref, n_old=n_old, k_old=k_old,
                    row_concentration_lambda=row_concentration_lambda,
                    row_leverage_lambda=row_leverage_lambda,
                    row_leverage_weights=row_leverage_weights,
                )
                rhs = score_old + 1e-4 * alpha * float(gtan @ gtan)
                if score_t >= rhs:
                    z = zt
                    accepted_eval = (score_t, grad_t, s_t, H_t)
                    accepted = True
                    break
            alpha *= 0.5

        if not accepted:
            stop = {"reason": "line_search_fail", "iters": it + 1, "grad_norm": gnorm, "line_search_steps": 20}
            z = z_old
            break

        if reuse_line_search_grad and accepted_eval is not None:
            score, grad, s, H = accepted_eval
        else:
            score, grad, s, H = streaming_score_grad_reduced_by_variant(
                score_variant, B_gain, B_block, C_prev, s2_old, R_old_block, z,
                rows_block, rows_ref, n_old=n_old, k_old=k_old,
                row_concentration_lambda=row_concentration_lambda,
                row_leverage_lambda=row_leverage_lambda,
                row_leverage_weights=row_leverage_weights,
            )
        step_norm = float(np.linalg.norm(z - z_old))
        f_change = abs(score - score_old)
        f_threshold = progress_f_tol * max(1.0, abs(score_old))
        stop = {
            "reason": "progress",
            "iters": it + 1,
            "grad_norm": gnorm,
            "step_norm": step_norm,
            "f_change": f_change,
            "f_threshold": f_threshold,
            "line_search_alpha": alpha,
            "line_search_steps": ls_iter + 1,
        }
        if f_change <= f_threshold:
            stop["reason"] = "f_change_tol"
            break
        if step_norm <= progress_step_tol:
            stop["reason"] = "step_tol"
            break
        if patience > 0:
            rel_change = f_change / max(abs(score_old), 1e-30)
            if rel_change <= patience_rel_tol:
                plateau_count += 1
                if plateau_count >= patience:
                    stop["reason"] = "patience"
                    stop["plateau_count"] = plateau_count
                    break
            else:
                plateau_count = 0
    else:
        gtan = project_reduced(grad - z * float(z @ grad), Qz)
        stop = {"reason": "maxit", "iters": maxit, "grad_norm": float(np.linalg.norm(gtan))}

    return z, score, s, H, stop


def entropyscore_forget_joint_reduced_eval(
    B, Z, rows_block, rows_ref, optimizer="cex", row_concentration_lambda=0.0,
    score_variant="legacy", row_leverage_lambda=0.0, row_leverage_weights=None
):
    r = Z.shape[1]
    vals = np.zeros(r, dtype=float)
    G = np.zeros_like(Z)
    s = np.zeros(r, dtype=float)
    H = np.zeros(r, dtype=float)
    for j in range(r):
        if optimizer == "cex":
            vals[j], G[:, j], s[j], H[j] = score_grad_reduced_by_variant(
                score_variant, B, Z[:, j], rows_block, rows_ref,
                row_concentration_lambda=row_concentration_lambda,
                row_leverage_lambda=row_leverage_lambda,
                row_leverage_weights=row_leverage_weights,
            )
        elif optimizer == "legacy":
            if float(row_concentration_lambda) != 0.0:
                raise ValueError("row_concentration_lambda is only supported with reduced_optimizer='cex'.")
            vals[j], G[:, j], s[j], H[j] = entropyscore_forget_logscore_grad_reduced(
                B, Z[:, j], rows_block, rows_ref
            )
        else:
            raise ValueError(f"Unknown reduced_optimizer: {optimizer}")
    return float(np.sum(vals)), vals, G, s, H


def entropyscore_forget_joint_streaming_reduced_eval(
    B_gain, B_block, C_prev, s2_old, Z, rows_block, rows_ref, optimizer="cex",
    row_concentration_lambda=0.0, score_variant="legacy", R_old_block=None,
    row_leverage_lambda=0.0, row_leverage_weights=None,
    n_old=0, k_old=None
):
    r = Z.shape[1]
    vals = np.zeros(r, dtype=float)
    G = np.zeros_like(Z)
    s = np.zeros(r, dtype=float)
    H = np.zeros(r, dtype=float)
    for j in range(r):
        if optimizer == "cex":
            vals[j], G[:, j], s[j], H[j] = streaming_score_grad_reduced_by_variant(
                score_variant, B_gain, B_block, C_prev, s2_old, R_old_block, Z[:, j],
                rows_block, rows_ref, n_old=n_old, k_old=k_old,
                row_concentration_lambda=row_concentration_lambda,
                row_leverage_lambda=row_leverage_lambda,
                row_leverage_weights=row_leverage_weights,
            )
        elif optimizer == "legacy":
            if float(row_concentration_lambda) != 0.0:
                raise ValueError("row_concentration_lambda is only supported with reduced_optimizer='cex'.")
            vals[j], G[:, j], s[j], H[j] = entropyscore_forget_streaming_logscore_grad_reduced(
                B_gain, B_block, C_prev, s2_old, Z[:, j], rows_block, rows_ref
            )
        else:
            raise ValueError(f"Unknown reduced_optimizer: {optimizer}")
    return float(np.sum(vals)), vals, G, s, H


def basic_projected_ascent_joint_reduced_forget(
    B, Z0, Qz, rows_block, rows_ref, maxit=80, tol=1e-8, optimizer="cex",
    reuse_line_search_grad=True, row_concentration_lambda=0.0, score_variant="legacy",
    row_leverage_lambda=0.0, row_leverage_weights=None
):
    Z = retract_stiefel_reduced(Z0, Qz)
    if Z is None:
        raise RuntimeError("Initial reduced Stiefel seed is infeasible.")

    total, vals, G, s, H = entropyscore_forget_joint_reduced_eval(
        B, Z, rows_block, rows_ref, optimizer=optimizer,
        row_concentration_lambda=row_concentration_lambda, score_variant=score_variant,
        row_leverage_lambda=row_leverage_lambda, row_leverage_weights=row_leverage_weights,
    )
    stop = {"reason": "maxit", "iters": maxit, "grad_norm": np.nan}
    for it in range(maxit):
        Xi = stiefel_tangent_gradient(Z, G, Qz)
        gnorm = float(np.linalg.norm(Xi, ord="fro"))
        if gnorm <= tol:
            stop = {"reason": "grad_tol", "iters": it, "grad_norm": gnorm}
            break

        accepted = False
        accepted_eval = None
        alpha = 1.0
        total_old = total
        Z_old = Z

        for ls_iter in range(20):
            Zt = retract_stiefel_reduced(Z + alpha * Xi, Qz)
            if Zt is not None:
                total_t, vals_t, G_t, s_t, H_t = entropyscore_forget_joint_reduced_eval(
                    B, Zt, rows_block, rows_ref, optimizer=optimizer,
                    row_concentration_lambda=row_concentration_lambda, score_variant=score_variant,
                    row_leverage_lambda=row_leverage_lambda, row_leverage_weights=row_leverage_weights,
                )
                rhs = total_old + 1e-4 * alpha * float(np.sum(Xi * Xi))
                if total_t >= rhs:
                    Z = Zt
                    accepted_eval = (total_t, vals_t, G_t, s_t, H_t)
                    accepted = True
                    break
            alpha *= 0.5

        if not accepted:
            stop = {"reason": "line_search_fail", "iters": it + 1, "grad_norm": gnorm, "line_search_steps": 20}
            Z = Z_old
            break

        if reuse_line_search_grad and accepted_eval is not None:
            total, vals, G, s, H = accepted_eval
        else:
            total, vals, G, s, H = entropyscore_forget_joint_reduced_eval(
                B, Z, rows_block, rows_ref, optimizer=optimizer,
                row_concentration_lambda=row_concentration_lambda, score_variant=score_variant,
                row_leverage_lambda=row_leverage_lambda, row_leverage_weights=row_leverage_weights,
            )
        step_norm = float(np.linalg.norm(Z - Z_old, ord="fro"))
        f_change = abs(total - total_old)
        stop = {
            "reason": "progress",
            "iters": it + 1,
            "grad_norm": gnorm,
            "step_norm": step_norm,
            "f_change": f_change,
            "line_search_alpha": alpha,
            "line_search_steps": ls_iter + 1,
        }
    else:
        Xi = stiefel_tangent_gradient(Z, G, Qz)
        stop = {"reason": "maxit", "iters": maxit, "grad_norm": float(np.linalg.norm(Xi, ord="fro"))}

    return Z, total, vals, s, H, stop


def basic_projected_ascent_joint_reduced_streaming_forget(
    B_gain, B_block, C_prev, s2_old, Z0, Qz, rows_block, rows_ref,
    maxit=80, tol=1e-8, optimizer="cex", reuse_line_search_grad=True,
    row_concentration_lambda=0.0, score_variant="legacy", R_old_block=None,
    n_old=0, k_old=None, row_leverage_lambda=0.0, row_leverage_weights=None
):
    Z = retract_stiefel_reduced(Z0, Qz)
    if Z is None:
        raise RuntimeError("Initial reduced Stiefel seed is infeasible.")

    total, vals, G, s, H = entropyscore_forget_joint_streaming_reduced_eval(
        B_gain, B_block, C_prev, s2_old, Z, rows_block, rows_ref, optimizer=optimizer,
        row_concentration_lambda=row_concentration_lambda, score_variant=score_variant,
        R_old_block=R_old_block, n_old=n_old, k_old=k_old,
        row_leverage_lambda=row_leverage_lambda, row_leverage_weights=row_leverage_weights,
    )
    stop = {"reason": "maxit", "iters": maxit, "grad_norm": np.nan}
    for it in range(maxit):
        Xi = stiefel_tangent_gradient(Z, G, Qz)
        gnorm = float(np.linalg.norm(Xi, ord="fro"))
        if gnorm <= tol:
            stop = {"reason": "grad_tol", "iters": it, "grad_norm": gnorm}
            break

        accepted = False
        accepted_eval = None
        alpha = 1.0
        total_old = total
        Z_old = Z

        for ls_iter in range(20):
            Zt = retract_stiefel_reduced(Z + alpha * Xi, Qz)
            if Zt is not None:
                total_t, vals_t, G_t, s_t, H_t = entropyscore_forget_joint_streaming_reduced_eval(
                    B_gain, B_block, C_prev, s2_old, Zt, rows_block, rows_ref, optimizer=optimizer,
                    row_concentration_lambda=row_concentration_lambda, score_variant=score_variant,
                    R_old_block=R_old_block, n_old=n_old, k_old=k_old,
                    row_leverage_lambda=row_leverage_lambda, row_leverage_weights=row_leverage_weights,
                )
                rhs = total_old + 1e-4 * alpha * float(np.sum(Xi * Xi))
                if total_t >= rhs:
                    Z = Zt
                    accepted_eval = (total_t, vals_t, G_t, s_t, H_t)
                    accepted = True
                    break
            alpha *= 0.5

        if not accepted:
            stop = {"reason": "line_search_fail", "iters": it + 1, "grad_norm": gnorm, "line_search_steps": 20}
            Z = Z_old
            break

        if reuse_line_search_grad and accepted_eval is not None:
            total, vals, G, s, H = accepted_eval
        else:
            total, vals, G, s, H = entropyscore_forget_joint_streaming_reduced_eval(
                B_gain, B_block, C_prev, s2_old, Z, rows_block, rows_ref, optimizer=optimizer,
                row_concentration_lambda=row_concentration_lambda, score_variant=score_variant,
                R_old_block=R_old_block, n_old=n_old, k_old=k_old,
                row_leverage_lambda=row_leverage_lambda, row_leverage_weights=row_leverage_weights,
            )
        step_norm = float(np.linalg.norm(Z - Z_old, ord="fro"))
        f_change = abs(total - total_old)
        stop = {
            "reason": "progress",
            "iters": it + 1,
            "grad_norm": gnorm,
            "step_norm": step_norm,
            "f_change": f_change,
            "line_search_alpha": alpha,
            "line_search_steps": ls_iter + 1,
        }
    else:
        Xi = stiefel_tangent_gradient(Z, G, Qz)
        stop = {"reason": "maxit", "iters": maxit, "grad_norm": float(np.linalg.norm(Xi, ord="fro"))}

    return Z, total, vals, s, H, stop


def _stiefel_constraint_values(x, q, r):
    Z = x.reshape(q, r)
    C = Z.T @ Z - np.eye(r)
    vals = []
    for i in range(r):
        for j in range(i, r):
            vals.append(C[i, j])
    return np.asarray(vals, dtype=float)


def _stiefel_constraint_jacobian(x, q, r):
    Z = x.reshape(q, r)
    rows = []
    for i in range(r):
        for j in range(i, r):
            J = np.zeros((q, r), dtype=float)
            if i == j:
                J[:, i] = 2.0 * Z[:, i]
            else:
                J[:, i] = Z[:, j]
                J[:, j] = Z[:, i]
            rows.append(J.reshape(-1))
    return np.asarray(rows, dtype=float)


def basic_slsqp_joint_reduced_forget(
    B, Z0, Qz, rows_block, rows_ref, maxit=200, tol=1e-10, optimizer="cex",
    row_concentration_lambda=0.0, score_variant="legacy"
):
    if Qz is not None and np.asarray(Qz).size:
        raise NotImplementedError("SLSQP joint solver currently supports unconstrained reduced Stiefel frames only.")
    Z_init = retract_stiefel_reduced(Z0, Qz)
    if Z_init is None:
        raise RuntimeError("Initial reduced Stiefel seed is infeasible.")
    q, r = Z_init.shape

    def fun(x):
        Z = x.reshape(q, r)
        total, _, _, _, _ = entropyscore_forget_joint_reduced_eval(
            B, Z, rows_block, rows_ref, optimizer=optimizer, row_concentration_lambda=row_concentration_lambda,
            score_variant=score_variant
        )
        return -total

    def jac(x):
        Z = x.reshape(q, r)
        _, _, G, _, _ = entropyscore_forget_joint_reduced_eval(
            B, Z, rows_block, rows_ref, optimizer=optimizer, row_concentration_lambda=row_concentration_lambda,
            score_variant=score_variant
        )
        return -G.reshape(-1)

    constraints = [{
        "type": "eq",
        "fun": lambda x: _stiefel_constraint_values(x, q, r),
        "jac": lambda x: _stiefel_constraint_jacobian(x, q, r),
    }]
    res = sp.optimize.minimize(
        fun,
        Z_init.reshape(-1),
        method="SLSQP",
        jac=jac,
        constraints=constraints,
        options={"maxiter": int(maxit), "ftol": float(tol), "disp": False},
    )
    Z = retract_stiefel_reduced(res.x.reshape(q, r), Qz)
    total, vals, G, s, H = entropyscore_forget_joint_reduced_eval(
        B, Z, rows_block, rows_ref, optimizer=optimizer, row_concentration_lambda=row_concentration_lambda,
        score_variant=score_variant
    )
    Xi = stiefel_tangent_gradient(Z, G, Qz)
    stop = {
        "reason": "slsqp_success" if res.success else "slsqp_fail",
        "iters": int(getattr(res, "nit", 0)),
        "grad_norm": float(np.linalg.norm(Xi, ord="fro")),
        "message": str(res.message),
    }
    return Z, total, vals, s, H, stop


def basic_slsqp_joint_reduced_streaming_forget(
    B_gain, B_block, C_prev, s2_old, Z0, Qz, rows_block, rows_ref,
    maxit=200, tol=1e-10, optimizer="cex", row_concentration_lambda=0.0,
    score_variant="legacy", R_old_block=None, n_old=0, k_old=None
):
    if Qz is not None and np.asarray(Qz).size:
        raise NotImplementedError("SLSQP joint solver currently supports unconstrained reduced Stiefel frames only.")
    Z_init = retract_stiefel_reduced(Z0, Qz)
    if Z_init is None:
        raise RuntimeError("Initial reduced Stiefel seed is infeasible.")
    q, r = Z_init.shape

    def fun(x):
        Z = x.reshape(q, r)
        total, _, _, _, _ = entropyscore_forget_joint_streaming_reduced_eval(
            B_gain, B_block, C_prev, s2_old, Z, rows_block, rows_ref, optimizer=optimizer,
            row_concentration_lambda=row_concentration_lambda, score_variant=score_variant,
            R_old_block=R_old_block, n_old=n_old, k_old=k_old
        )
        return -total

    def jac(x):
        Z = x.reshape(q, r)
        _, _, G, _, _ = entropyscore_forget_joint_streaming_reduced_eval(
            B_gain, B_block, C_prev, s2_old, Z, rows_block, rows_ref, optimizer=optimizer,
            row_concentration_lambda=row_concentration_lambda, score_variant=score_variant,
            R_old_block=R_old_block, n_old=n_old, k_old=k_old
        )
        return -G.reshape(-1)

    constraints = [{
        "type": "eq",
        "fun": lambda x: _stiefel_constraint_values(x, q, r),
        "jac": lambda x: _stiefel_constraint_jacobian(x, q, r),
    }]
    res = sp.optimize.minimize(
        fun,
        Z_init.reshape(-1),
        method="SLSQP",
        jac=jac,
        constraints=constraints,
        options={"maxiter": int(maxit), "ftol": float(tol), "disp": False},
    )
    Z = retract_stiefel_reduced(res.x.reshape(q, r), Qz)
    total, vals, G, s, H = entropyscore_forget_joint_streaming_reduced_eval(
        B_gain, B_block, C_prev, s2_old, Z, rows_block, rows_ref, optimizer=optimizer,
        row_concentration_lambda=row_concentration_lambda, score_variant=score_variant,
        R_old_block=R_old_block, n_old=n_old, k_old=k_old
    )
    Xi = stiefel_tangent_gradient(Z, G, Qz)
    stop = {
        "reason": "slsqp_success" if res.success else "slsqp_fail",
        "iters": int(getattr(res, "nit", 0)),
        "grad_norm": float(np.linalg.norm(Xi, ord="fro")),
        "message": str(res.message),
    }
    return Z, total, vals, s, H, stop


def make_greedy_stiefel_warm_start(
    B_gain, B_block, C_prev, s2_old, prior_Z, active_r, rows_block, rows_ref, num_restarts,
    maxit, tol, rng, is_initial_block, reduced_optimizer="cex", reuse_line_search_grad=True,
    row_concentration_lambda=0.0, score_variant="legacy", R_old_block=None,
    n_old=0, k_old=None, row_leverage_lambda=0.0, row_leverage_weights=None
):
    q = B_block.shape[1]
    if active_r <= 0 or q < active_r:
        return None

    cols = []
    for k_idx in range(active_r):
        Qz = np.column_stack(cols) if cols else np.zeros((q, 0), dtype=B_block.dtype)
        if Qz.size:
            Qz = orthonormalize_columns(Qz, dtype=B_block.dtype)

        starts = []
        if prior_Z is not None and prior_Z.shape[1] > k_idx:
            append_unique_reduced_seed(starts, retract_reduced(prior_Z[:, k_idx], Qz))
        if k_idx < q:
            e = np.zeros(q, dtype=B_block.dtype)
            e[k_idx] = 1.0
            append_unique_reduced_seed(starts, retract_reduced(e, Qz))
        while len(starts) < max(1, num_restarts):
            zrand = np.ascontiguousarray(rng.standard_normal(q), dtype=B_block.dtype)
            append_unique_reduced_seed(starts, retract_reduced(zrand, Qz))

        best = None
        for z0 in starts:
            if is_initial_block:
                if reduced_optimizer == "cex":
                    cand = basic_projected_ascent_single_reduced_forget_cex(
                        B_block, z0, Qz, rows_block, rows_ref, maxit=maxit, tol=tol,
                        reuse_line_search_grad=reuse_line_search_grad,
                        row_concentration_lambda=row_concentration_lambda,
                        score_variant=score_variant,
                        row_leverage_lambda=row_leverage_lambda,
                        row_leverage_weights=row_leverage_weights,
                    )
                    cand_key = float(cand[1])
                else:
                    cand = basic_projected_ascent_single_reduced_forget(
                        B_block, z0, Qz, rows_block, rows_ref, maxit=maxit, tol=tol
                    )
                    cand_key = float(cand[1])
            else:
                if reduced_optimizer == "cex":
                    cand = basic_projected_ascent_single_reduced_streaming_forget_cex(
                        B_gain, B_block, C_prev, s2_old, z0, Qz, rows_block, rows_ref,
                        maxit=maxit, tol=tol, reuse_line_search_grad=reuse_line_search_grad,
                        row_concentration_lambda=row_concentration_lambda,
                        score_variant=score_variant,
                        R_old_block=R_old_block,
                        n_old=n_old,
                        k_old=k_old,
                        row_leverage_lambda=row_leverage_lambda,
                        row_leverage_weights=row_leverage_weights,
                    )
                    cand_key = float(cand[1])
                else:
                    cand = basic_projected_ascent_single_reduced_streaming_forget(
                        B_gain, B_block, C_prev, s2_old, z0, Qz, rows_block, rows_ref,
                        maxit=maxit, tol=tol,
                    )
                    cand_key = float(cand[1])
            if best is None or cand_key > best[0]:
                best = (cand_key, cand[0])
        if best is None:
            return None
        cols.append(best[1])

    return retract_stiefel_reduced(np.column_stack(cols), None)


def make_rotated_stiefel_seeds(Z, num_rotations, rng, max_angle=np.pi / 4):
    if Z is None or num_rotations <= 0:
        return []
    r = Z.shape[1]
    if r <= 1:
        return []

    seeds = []
    if r == 2:
        count = int(num_rotations)
        angles = np.linspace(-float(max_angle), float(max_angle), count + 2)[1:-1]
        for theta in angles:
            c = np.cos(theta)
            s = np.sin(theta)
            R = np.array([[c, -s], [s, c]], dtype=Z.dtype)
            seeds.append(np.ascontiguousarray(Z @ R, dtype=Z.dtype))
        return seeds

    for _ in range(int(num_rotations)):
        A = rng.standard_normal((r, r))
        K = A - A.T
        norm_K = np.linalg.norm(K, ord="fro")
        if norm_K <= 1e-14:
            continue
        K *= float(max_angle) / norm_K
        R = la.expm(K)
        seeds.append(np.ascontiguousarray(Z @ R.astype(Z.dtype, copy=False), dtype=Z.dtype))
    return seeds


def make_tangent_perturbed_stiefel_seeds(Z, num_perturbations, perturb_scale, rng):
    if Z is None or num_perturbations <= 0 or perturb_scale <= 0.0:
        return []
    seeds = []
    for _ in range(int(num_perturbations)):
        W = np.ascontiguousarray(rng.standard_normal(Z.shape), dtype=Z.dtype)
        Xi = stiefel_tangent_gradient(Z, W)
        nXi = np.linalg.norm(Xi, ord="fro")
        if nXi <= 1e-14:
            continue
        Zp = retract_stiefel_reduced(Z + float(perturb_scale) * Xi / nXi)
        if Zp is not None:
            seeds.append(Zp)
    return seeds


def basic_projected_ascent_single_reduced_forget(B, z0, Qz, rows_block, rows_ref, maxit=80, tol=1e-8):
    z = retract_reduced(z0, Qz)
    if z is None:
        raise RuntimeError("Initial reduced seed is infeasible.")

    logf, grad, s, H = entropyscore_forget_logscore_grad_reduced(B, z, rows_block, rows_ref)
    stop = {"reason": "maxit", "iters": maxit, "grad_norm": np.nan}

    for it in range(maxit):
        gtan = project_reduced(grad - z * float(z @ grad), Qz)
        gnorm = float(np.linalg.norm(gtan))
        if gnorm <= tol:
            stop = {"reason": "grad_tol", "iters": it, "grad_norm": gnorm}
            break

        step = gtan / max(gnorm, 1e-30)
        alpha = 1.0
        improved = False
        for _ in range(20):
            zt = retract_reduced(z + alpha * step, Qz)
            if zt is None:
                alpha *= 0.5
                continue
            logf_t, grad_t, s_t, H_t = entropyscore_forget_logscore_grad_reduced(B, zt, rows_block, rows_ref)
            if logf_t > logf:
                z, logf, grad, s, H = zt, logf_t, grad_t, s_t, H_t
                improved = True
                break
            alpha *= 0.5

        if not improved:
            stop = {"reason": "line_search_fail", "iters": it + 1, "grad_norm": gnorm}
            break
    else:
        gtan = project_reduced(grad - z * float(z @ grad), Qz)
        stop = {"reason": "maxit", "iters": maxit, "grad_norm": float(np.linalg.norm(gtan))}

    return z, logf, s, H, stop


def basic_projected_ascent_single_reduced_streaming_forget(
    B_gain, B_block, C_prev, s2_old, z0, Qz, rows_block, rows_ref, maxit=80, tol=1e-8
):
    z = retract_reduced(z0, Qz)
    if z is None:
        raise RuntimeError("Initial reduced seed is infeasible.")

    logf, grad, s, H = entropyscore_forget_streaming_logscore_grad_reduced(
        B_gain, B_block, C_prev, s2_old, z, rows_block, rows_ref
    )
    stop = {"reason": "maxit", "iters": maxit, "grad_norm": np.nan}

    for it in range(maxit):
        gtan = project_reduced(grad - z * float(z @ grad), Qz)
        gnorm = float(np.linalg.norm(gtan))
        if gnorm <= tol:
            stop = {"reason": "grad_tol", "iters": it, "grad_norm": gnorm}
            break

        step = gtan / max(gnorm, 1e-30)
        alpha = 1.0
        improved = False
        for _ in range(20):
            zt = retract_reduced(z + alpha * step, Qz)
            if zt is None:
                alpha *= 0.5
                continue
            logf_t, grad_t, s_t, H_t = entropyscore_forget_streaming_logscore_grad_reduced(
                B_gain, B_block, C_prev, s2_old, zt, rows_block, rows_ref
            )
            if logf_t > logf:
                z, logf, grad, s, H = zt, logf_t, grad_t, s_t, H_t
                improved = True
                break
            alpha *= 0.5

        if not improved:
            stop = {"reason": "line_search_fail", "iters": it + 1, "grad_norm": gnorm}
            break
    else:
        gtan = project_reduced(grad - z * float(z @ grad), Qz)
        stop = {"reason": "maxit", "iters": maxit, "grad_norm": float(np.linalg.norm(gtan))}

    return z, logf, s, H, stop


def entropyscore_forget_full_gradient_residual(
    M_gain, A_block, v, Vred, state_prev, rows_ref, Q_prev=None, return_vector=False,
    row_concentration_lambda=0.0, score_variant="legacy", old_row_memory=None,
    row_leverage_lambda=0.0, row_leverage_weights=None
):
    if state_prev is None:
        if score_variant == "legacy":
            logf, grad_log, _, _ = entropyscore_forget_logscore_grad_rows(A_block, v, rows_ref)
            score = float(np.exp(logf))
            grad = np.ascontiguousarray(score * grad_log, dtype=np.asarray(grad_log).dtype)
            _, grad = apply_row_concentration_regularizer(
                score, grad, A_block, v, row_concentration_lambda
            )
        else:
            _, grad, _, _ = score_grad_reduced_by_variant(
                score_variant,
                A_block,
                v,
                np.asarray(A_block).shape[0],
                rows_ref,
                row_concentration_lambda=row_concentration_lambda,
                row_leverage_lambda=row_leverage_lambda,
                row_leverage_weights=row_leverage_weights,
            )
    else:
        if score_variant in {"oldcorrected", "combined", "l4l2", "subsetmass"}:
            _, grad, _, _ = streaming_score_grad_reduced_by_variant(
                score_variant,
                M_gain,
                A_block,
                np.asarray(state_prev["V"]).T,
                state_prev["s2"],
                old_row_memory,
                v,
                np.asarray(A_block).shape[0],
                rows_ref,
                n_old=int(state_prev.get("rows_seen", 0)),
                k_old=len(state_prev["s2"]),
                row_concentration_lambda=row_concentration_lambda,
                row_leverage_lambda=row_leverage_lambda,
                row_leverage_weights=row_leverage_weights,
            )
        else:
            logf, grad_log, _, _ = entropyscore_forget_streaming_logscore_grad(
                M_gain, A_block, state_prev["V"], state_prev["s2"], v, rows_ref
            )
            score = float(np.exp(logf))
            grad = np.ascontiguousarray(score * grad_log, dtype=np.asarray(grad_log).dtype)
            _, grad = apply_row_concentration_regularizer(
                score, grad, A_block, v, row_concentration_lambda
            )
    dtype = np.result_type(np.asarray(grad).dtype, np.asarray(v).dtype)
    grad = np.ascontiguousarray(np.asarray(grad, dtype=dtype).reshape(-1))
    v = np.ascontiguousarray(np.asarray(v, dtype=dtype).reshape(-1))

    g_tan = grad - v * float(v @ grad)
    if Q_prev is not None and np.asarray(Q_prev).size:
        Q_prev_arr = np.ascontiguousarray(np.asarray(Q_prev, dtype=dtype))
        g_tan = g_tan - Q_prev_arr @ (Q_prev_arr.T @ g_tan)

    Vred_arr = np.ascontiguousarray(np.asarray(Vred, dtype=dtype))
    r = g_tan - Vred_arr @ (Vred_arr.T @ g_tan)
    r = np.ascontiguousarray(r, dtype=dtype)
    r_norm = float(np.linalg.norm(r))
    g_norm = float(np.linalg.norm(g_tan))
    if return_vector:
        return r_norm, g_norm, r
    return r_norm, g_norm


def project_onto_span(X, Q):
    X_arr = np.asarray(X)
    if Q is None or not np.asarray(Q).size:
        return np.zeros_like(X_arr)
    Q_arr = np.ascontiguousarray(np.asarray(Q, dtype=np.result_type(X_arr.dtype, np.asarray(Q).dtype)))
    return np.ascontiguousarray(Q_arr @ (Q_arr.T @ X_arr), dtype=np.result_type(X_arr.dtype, Q_arr.dtype))


def right_singular_row_basis(M_gain, dtype=None, eps=1e-12):
    M_arr = np.asarray(M_gain)
    if dtype is None:
        dtype = M_arr.dtype
        if not np.issubdtype(dtype, np.floating):
            dtype = np.float64
    if M_arr.size == 0:
        n = M_arr.shape[1] if M_arr.ndim == 2 else 0
        return np.zeros((n, 0), dtype=dtype)

    _, s, Vh = np.linalg.svd(np.asarray(M_arr, dtype=np.float64), full_matrices=False)
    if s.size == 0:
        return np.zeros((M_arr.shape[1], 0), dtype=dtype)
    keep = s > eps
    if not np.any(keep):
        return np.zeros((M_arr.shape[1], 0), dtype=dtype)
    return np.ascontiguousarray(Vh[keep, :].T, dtype=dtype)


def subspace_residual(X, Q):
    X_arr = np.asarray(X)
    return np.ascontiguousarray(X_arr - project_onto_span(X_arr, Q), dtype=X_arr.dtype)


def subspace_principal_cosines(Q1, Q2):
    Q1_arr = orthonormalize_columns(Q1)
    Q2_arr = orthonormalize_columns(Q2)
    if Q1_arr.size == 0 or Q2_arr.size == 0:
        return np.zeros(0, dtype=float)
    s = la.svd(Q1_arr.T @ Q2_arr, compute_uv=False, lapack_driver="gesdd")
    return np.clip(np.asarray(s, dtype=float), 0.0, 1.0)


def projected_true_span_oracle(M_gain, V_exact, r, dtype=None, eps=1e-12, row_samples=None):
    if V_exact is None:
        n = int(np.asarray(M_gain).shape[1])
        out_dtype = np.float64 if dtype is None else dtype
        return np.zeros((n, 0), dtype=out_dtype), np.zeros((n, 0), dtype=out_dtype)

    if dtype is None:
        dtype = np.result_type(np.asarray(M_gain).dtype, np.asarray(V_exact).dtype)
        if row_samples is not None:
            dtype = np.result_type(dtype, np.asarray(row_samples).dtype)
        if not np.issubdtype(dtype, np.floating):
            dtype = np.float64

    M_arr = np.ascontiguousarray(np.asarray(M_gain, dtype=dtype))
    V_target = orthonormalize_columns(np.asarray(V_exact, dtype=dtype)[:, : int(r)], dtype=dtype, eps=eps)
    row_basis_source = M_arr
    if row_samples is not None and np.asarray(row_samples).size:
        R_arr = np.asarray(row_samples, dtype=dtype)
        if R_arr.ndim == 1:
            R_arr = R_arr.reshape(1, -1)
        if R_arr.shape[1] != M_arr.shape[1]:
            raise ValueError("row_samples must have the same column count as M_gain.")
        row_basis_source = np.ascontiguousarray(np.vstack([M_arr, R_arr]), dtype=dtype)
    Q_row = right_singular_row_basis(row_basis_source, dtype=dtype, eps=eps)
    if Q_row.size == 0 or V_target.size == 0:
        return np.zeros((M_arr.shape[1], 0), dtype=dtype), Q_row

    V_proj = project_onto_span(V_target, Q_row)
    Q_oracle = orthonormalize_columns(V_proj, dtype=dtype, eps=eps)
    return Q_oracle, Q_row


def score_full_vector_forget(
    M_gain, A_block, v, rows_ref, state_prev=None, score_variant="legacy",
    old_row_memory=None, row_concentration_lambda=0.0
):
    score, _, _ = score_full_vector_details_forget(
        M_gain, A_block, v, rows_ref,
        state_prev=state_prev,
        score_variant=score_variant,
        old_row_memory=old_row_memory,
        row_concentration_lambda=row_concentration_lambda,
    )
    return float(score)


def score_full_vector_details_forget(
    M_gain, A_block, v, rows_ref, state_prev=None, score_variant="legacy",
    old_row_memory=None, row_concentration_lambda=0.0
):
    A_arr = np.asarray(A_block)
    work_dtype = A_arr.dtype
    v_work = np.ascontiguousarray(np.asarray(v, dtype=work_dtype).reshape(-1))
    if state_prev is None:
        score, _, s, H = score_grad_reduced_by_variant(
            score_variant,
            A_arr,
            v_work,
            A_arr.shape[0],
            rows_ref,
            row_concentration_lambda=row_concentration_lambda,
        )
        return float(score), float(s), float(H)

    score, _, s, H = streaming_score_grad_reduced_by_variant(
        score_variant,
        np.asarray(M_gain, dtype=work_dtype),
        A_arr,
        np.asarray(state_prev["V"], dtype=work_dtype).T,
        np.asarray(state_prev["s2"], dtype=work_dtype),
        None if old_row_memory is None else np.asarray(old_row_memory, dtype=work_dtype),
        v_work,
        A_arr.shape[0],
        rows_ref,
        n_old=int(state_prev.get("rows_seen", 0)),
        k_old=len(state_prev["s2"]),
        row_concentration_lambda=row_concentration_lambda,
    )
    return float(score), float(s), float(H)


def oldcorrected_score_component_details(
    M_gain, A_block, v, rows_ref, state_prev=None, old_row_memory=None
):
    A_arr = np.asarray(A_block)
    work_dtype = np.result_type(A_arr.dtype, np.asarray(v).dtype)
    v_work = np.ascontiguousarray(np.asarray(v, dtype=work_dtype).reshape(-1))
    y = np.ascontiguousarray(np.asarray(A_block, dtype=work_dtype) @ v_work, dtype=work_dtype)
    y2_sq = max(float(np.dot(y, y)), 1e-30)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 1e-30)
    rows_block = max(int(A_arr.shape[0]), 2)
    rows_ref_eff = max(int(rows_ref), rows_block)
    c_new = np.log(rows_block / rows_ref_eff) / np.log(rows_block)
    H_new = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    psi_new = float(np.exp((1.0 - 0.5 * c_new) * np.log(y2_sq) + 0.25 * c_new * np.log(y4_4)))

    gain_vec = np.ascontiguousarray(np.asarray(M_gain, dtype=work_dtype) @ v_work, dtype=work_dtype)
    gain2 = max(float(np.dot(gain_vec, gain_vec)), 1e-30)
    out = {
        "gain2": gain2,
        "score_total": psi_new,
        "new_y2_sq": y2_sq,
        "new_y4_4": y4_4,
        "new_H": H_new,
        "new_c": float(c_new),
        "new_psi": psi_new,
        "old_E": 0.0,
        "old_sample_r2_sq": np.nan,
        "old_sample_r4_4": np.nan,
        "old_H": np.nan,
        "old_c": np.nan,
        "old_phi": 1.0,
        "old_psi": 0.0,
        "old_rows": 0,
    }
    out["new_rel_H"] = H_new / np.log(rows_block) if rows_block > 1 else np.nan
    out["old_rel_H"] = np.nan
    if state_prev is None:
        return out

    V_prev = np.asarray(state_prev["V"], dtype=work_dtype)
    s2_old_arr = np.asarray(state_prev["s2"], dtype=work_dtype)
    a = np.ascontiguousarray(V_prev.T @ v_work, dtype=work_dtype)
    E_old = float(np.sum((a ** 2) * s2_old_arr))
    R_old_arr = None if old_row_memory is None else np.asarray(old_row_memory, dtype=work_dtype)
    if R_old_arr is None or R_old_arr.size == 0 or R_old_arr.shape[0] <= 1:
        psi_old = E_old
        out.update({"old_E": E_old, "old_psi": psi_old, "score_total": psi_old + psi_new})
        return out

    if R_old_arr.ndim == 1:
        R_old_arr = R_old_arr.reshape(1, -1)
    s_sample = max(int(R_old_arr.shape[0]), 2)
    n_old_eff = max(int(state_prev.get("rows_seen", 0)), s_sample, 2)
    rows_ref_old = max(int(rows_ref), n_old_eff)
    c_old = np.log(n_old_eff / rows_ref_old) / np.log(s_sample)
    r = np.ascontiguousarray(R_old_arr @ v_work, dtype=work_dtype)
    r2_sq = max(float(np.dot(r, r)), 1e-30)
    r4_4 = max(float(np.sum((r * r) * (r * r))), 1e-30)
    H_old = -(np.log(r4_4) - 2.0 * np.log(r2_sq))
    phi_old = float(np.exp(-0.5 * c_old * np.log(r2_sq) + 0.25 * c_old * np.log(r4_4)))
    psi_old = E_old * phi_old
    out.update({
        "score_total": psi_old + psi_new,
        "old_E": E_old,
        "old_sample_r2_sq": r2_sq,
        "old_sample_r4_4": r4_4,
        "old_H": H_old,
        "old_c": float(c_old),
        "old_phi": phi_old,
        "old_psi": psi_old,
        "old_rows": int(R_old_arr.shape[0]),
    })
    out["old_rel_H"] = H_old / np.log(s_sample) if s_sample > 1 else np.nan
    return out


def combined_score_component_details(
    M_gain, A_block, v, rows_ref, state_prev=None, old_row_memory=None
):
    A_arr = np.asarray(A_block)
    work_dtype = np.result_type(A_arr.dtype, np.asarray(v).dtype)
    v_work = np.ascontiguousarray(np.asarray(v, dtype=work_dtype).reshape(-1))

    gain_vec = np.ascontiguousarray(np.asarray(M_gain, dtype=work_dtype) @ v_work, dtype=work_dtype)
    gain2 = max(float(np.dot(gain_vec, gain_vec)), 1e-30)

    y = np.ascontiguousarray(np.asarray(A_block, dtype=work_dtype) @ v_work, dtype=work_dtype)
    new_y2_sq = max(float(np.dot(y, y)), 0.0)
    new_y4_4 = max(float(np.sum((y * y) * (y * y))), 0.0)
    rows_block = max(int(A_arr.shape[0]), 0)
    rows_entropy = rows_block

    pooled_y2_sq = new_y2_sq
    pooled_y4_4 = new_y4_4
    old_y2_sq = np.nan
    old_y4_4 = np.nan
    old_H = np.nan
    old_rel_H = np.nan
    old_rows = 0

    R_old_arr = None if old_row_memory is None else np.asarray(old_row_memory, dtype=work_dtype)
    if state_prev is not None and R_old_arr is not None and R_old_arr.size:
        if R_old_arr.ndim == 1:
            R_old_arr = R_old_arr.reshape(1, -1)
        r = np.ascontiguousarray(R_old_arr @ v_work, dtype=work_dtype)
        old_y2_sq = max(float(np.dot(r, r)), 0.0)
        old_y4_4 = max(float(np.sum((r * r) * (r * r))), 0.0)
        pooled_y2_sq += old_y2_sq
        pooled_y4_4 += old_y4_4
        old_rows = int(R_old_arr.shape[0])
        rows_entropy += old_rows
        if old_y2_sq > 0.0 and old_y4_4 > 0.0:
            old_H = -(np.log(old_y4_4) - 2.0 * np.log(old_y2_sq))
            old_rel_H = old_H / np.log(max(old_rows, 2))

    pooled_y2_sq = max(pooled_y2_sq, 1e-30)
    pooled_y4_4 = max(pooled_y4_4, 1e-30)
    rows_entropy = max(rows_entropy, 2)
    rows_ref_eff = max(int(rows_ref), rows_entropy)
    n_old = 0 if state_prev is None else int(state_prev.get("rows_seen", 0))
    rows_seen = min(max(n_old + rows_block, 1), rows_ref_eff)
    c = np.log(rows_seen / rows_ref_eff) / (2.0 * np.log(rows_entropy))
    H_pooled = -(np.log(pooled_y4_4) - 2.0 * np.log(pooled_y2_sq))
    phi = float(np.exp(c * (np.log(pooled_y4_4) - 2.0 * np.log(pooled_y2_sq))))
    score_total = gain2 * phi
    new_H = np.nan
    new_rel_H = np.nan
    if new_y2_sq > 0.0 and new_y4_4 > 0.0:
        new_H = -(np.log(new_y4_4) - 2.0 * np.log(new_y2_sq))
        new_rel_H = new_H / np.log(max(rows_block, 2))

    return {
        "gain2": gain2,
        "score_total": score_total,
        "phi": phi,
        "pooled_y2_sq": pooled_y2_sq,
        "pooled_y4_4": pooled_y4_4,
        "pooled_H": H_pooled,
        "pooled_rel_H": H_pooled / np.log(rows_entropy) if rows_entropy > 1 else np.nan,
        "combined_c": float(c),
        "rows_entropy": int(rows_entropy),
        "rows_seen": int(rows_seen),
        "new_y2_sq": new_y2_sq,
        "new_y4_4": new_y4_4,
        "new_H": new_H,
        "new_rel_H": new_rel_H,
        "old_y2_sq": old_y2_sq,
        "old_y4_4": old_y4_4,
        "old_H": old_H,
        "old_rel_H": old_rel_H,
        "old_rows": old_rows,
    }


def l4l2_score_component_details(
    M_gain, A_block, v, rows_ref, state_prev=None, old_row_memory=None
):
    del rows_ref
    A_arr = np.asarray(A_block)
    work_dtype = np.result_type(A_arr.dtype, np.asarray(v).dtype)
    v_work = np.ascontiguousarray(np.asarray(v, dtype=work_dtype).reshape(-1))

    gain_vec = np.ascontiguousarray(np.asarray(M_gain, dtype=work_dtype) @ v_work, dtype=work_dtype)
    gain2 = max(float(np.dot(gain_vec, gain_vec)), 1e-30)

    y = np.ascontiguousarray(np.asarray(A_block, dtype=work_dtype) @ v_work, dtype=work_dtype)
    new_y2_sq = max(float(np.dot(y, y)), 0.0)
    new_y4_4 = max(float(np.sum((y * y) * (y * y))), 0.0)
    rows_block = max(int(A_arr.shape[0]), 0)
    rows_ratio = rows_block

    pooled_y2_sq = new_y2_sq
    pooled_y4_4 = new_y4_4
    old_y2_sq = np.nan
    old_y4_4 = np.nan
    old_H = np.nan
    old_rel_H = np.nan
    old_rows = 0

    R_old_arr = None if old_row_memory is None else np.asarray(old_row_memory, dtype=work_dtype)
    if state_prev is not None and R_old_arr is not None and R_old_arr.size:
        if R_old_arr.ndim == 1:
            R_old_arr = R_old_arr.reshape(1, -1)
        r = np.ascontiguousarray(R_old_arr @ v_work, dtype=work_dtype)
        old_y2_sq = max(float(np.dot(r, r)), 0.0)
        old_y4_4 = max(float(np.sum((r * r) * (r * r))), 0.0)
        pooled_y2_sq += old_y2_sq
        pooled_y4_4 += old_y4_4
        old_rows = int(R_old_arr.shape[0])
        rows_ratio += old_rows
        if old_y2_sq > 0.0 and old_y4_4 > 0.0:
            old_H = -(np.log(old_y4_4) - 2.0 * np.log(old_y2_sq))
            old_rel_H = old_H / np.log(max(old_rows, 2))

    pooled_y2_sq = max(pooled_y2_sq, 1e-30)
    pooled_y4_4 = max(pooled_y4_4, 1e-30)
    rows_ratio = max(rows_ratio, 2)
    pooled_H = -(np.log(pooled_y4_4) - 2.0 * np.log(pooled_y2_sq))
    pooled_l4l2_sq = float(np.sqrt(pooled_y4_4) / pooled_y2_sq)
    theta = float(pooled_y2_sq / np.sqrt(pooled_y4_4))
    score_total = gain2 * theta
    new_H = np.nan
    new_rel_H = np.nan
    if new_y2_sq > 0.0 and new_y4_4 > 0.0:
        new_H = -(np.log(new_y4_4) - 2.0 * np.log(new_y2_sq))
        new_rel_H = new_H / np.log(max(rows_block, 2))

    return {
        "gain2": gain2,
        "score_total": score_total,
        "theta": theta,
        "pooled_l4l2_sq": pooled_l4l2_sq,
        "pooled_y2_sq": pooled_y2_sq,
        "pooled_y4_4": pooled_y4_4,
        "pooled_H": pooled_H,
        "pooled_rel_H": pooled_H / np.log(rows_ratio) if rows_ratio > 1 else np.nan,
        "rows_ratio": int(rows_ratio),
        "new_y2_sq": new_y2_sq,
        "new_y4_4": new_y4_4,
        "new_H": new_H,
        "new_rel_H": new_rel_H,
        "old_y2_sq": old_y2_sq,
        "old_y4_4": old_y4_4,
        "old_H": old_H,
        "old_rel_H": old_rel_H,
        "old_rows": old_rows,
    }


def print_score_component_dump(label, score_variant, vectors, M_gain, A_block, rows_ref, state_prev, old_row_memory):
    print(f"{label}:")
    for name, vec in vectors:
        if score_variant == "oldcorrected":
            comp = oldcorrected_score_component_details(
                M_gain, A_block, vec, rows_ref, state_prev=state_prev, old_row_memory=old_row_memory
            )
            print(
                f"  {name}: total={comp['score_total']:.12f} gain2={comp['gain2']:.12f} "
                f"new_psi={comp['new_psi']:.12f} new_y2={comp['new_y2_sq']:.12f} "
                f"new_y4={comp['new_y4_4']:.12f} new_H={comp['new_H']:.12f} "
                f"new_rel_H={comp['new_rel_H']:.12f} new_c={comp['new_c']:.12f} "
                f"old_psi={comp['old_psi']:.12f} old_E={comp['old_E']:.12f} old_phi={comp['old_phi']:.12f} "
                f"old_r2={comp['old_sample_r2_sq']:.12f} old_r4={comp['old_sample_r4_4']:.12f} "
                f"old_H={comp['old_H']:.12f} old_rel_H={comp['old_rel_H']:.12f} "
                f"old_c={comp['old_c']:.12f} old_rows={comp['old_rows']}"
            )
        elif score_variant == "combined":
            comp = combined_score_component_details(
                M_gain, A_block, vec, rows_ref, state_prev=state_prev, old_row_memory=old_row_memory
            )
            print(
                f"  {name}: total={comp['score_total']:.12f} gain2={comp['gain2']:.12f} "
                f"phi={comp['phi']:.12f} pooled_y2={comp['pooled_y2_sq']:.12f} "
                f"pooled_y4={comp['pooled_y4_4']:.12f} pooled_H={comp['pooled_H']:.12f} "
                f"pooled_rel_H={comp['pooled_rel_H']:.12f} combined_c={comp['combined_c']:.12f} "
                f"rows_entropy={comp['rows_entropy']} rows_seen={comp['rows_seen']} "
                f"new_y2={comp['new_y2_sq']:.12f} new_y4={comp['new_y4_4']:.12f} "
                f"new_H={comp['new_H']:.12f} new_rel_H={comp['new_rel_H']:.12f} "
                f"old_y2={comp['old_y2_sq']:.12f} old_y4={comp['old_y4_4']:.12f} "
                f"old_H={comp['old_H']:.12f} old_rel_H={comp['old_rel_H']:.12f} "
                f"old_rows={comp['old_rows']}"
            )
        elif score_variant == "l4l2":
            comp = l4l2_score_component_details(
                M_gain, A_block, vec, rows_ref, state_prev=state_prev, old_row_memory=old_row_memory
            )
            print(
                f"  {name}: total={comp['score_total']:.12f} gain2={comp['gain2']:.12f} "
                f"theta={comp['theta']:.12f} pooled_l4l2_sq={comp['pooled_l4l2_sq']:.12f} "
                f"pooled_y2={comp['pooled_y2_sq']:.12f} pooled_y4={comp['pooled_y4_4']:.12f} "
                f"pooled_H={comp['pooled_H']:.12f} pooled_rel_H={comp['pooled_rel_H']:.12f} "
                f"rows_ratio={comp['rows_ratio']} "
                f"new_y2={comp['new_y2_sq']:.12f} new_y4={comp['new_y4_4']:.12f} "
                f"new_H={comp['new_H']:.12f} new_rel_H={comp['new_rel_H']:.12f} "
                f"old_y2={comp['old_y2_sq']:.12f} old_y4={comp['old_y4_4']:.12f} "
                f"old_H={comp['old_H']:.12f} old_rel_H={comp['old_rel_H']:.12f} "
                f"old_rows={comp['old_rows']}"
            )


def oracle_projection_diagnostics(
    M_gain, A_block, V_exact, V_opt, rank, rows_ref, state_prev=None,
    score_variant="legacy", old_row_memory=None, row_concentration_lambda=0.0,
    oracle_projection_row_samples=None,
):
    if V_exact is None or np.asarray(V_exact).size == 0:
        return None

    diag_dtype = np.result_type(np.asarray(M_gain).dtype, np.asarray(V_exact).dtype, np.float64)
    Q_oracle, Q_row = projected_true_span_oracle(
        np.asarray(M_gain, dtype=diag_dtype),
        np.asarray(V_exact, dtype=diag_dtype)[:, : int(rank)],
        int(rank),
        dtype=diag_dtype,
        row_samples=oracle_projection_row_samples,
    )
    raw_cols = []
    for j in range(min(int(rank), np.asarray(V_exact).shape[1])):
        v_proj = project_onto_span(np.asarray(V_exact, dtype=diag_dtype)[:, j], Q_row).reshape(-1)
        v_norm = float(np.linalg.norm(v_proj))
        if v_norm > 1e-30:
            raw_cols.append(np.ascontiguousarray(v_proj / v_norm, dtype=diag_dtype))

    if not raw_cols:
        return None

    raw_proj = np.column_stack(raw_cols)
    raw_scores = np.asarray([
        score_full_vector_forget(
            M_gain, A_block, raw_proj[:, j], rows_ref,
            state_prev=state_prev,
            score_variant=score_variant,
            old_row_memory=old_row_memory,
            row_concentration_lambda=row_concentration_lambda,
        )
        for j in range(raw_proj.shape[1])
    ], dtype=float)
    qr_scores = np.asarray([
        score_full_vector_forget(
            M_gain, A_block, Q_oracle[:, j], rows_ref,
            state_prev=state_prev,
            score_variant=score_variant,
            old_row_memory=old_row_memory,
            row_concentration_lambda=row_concentration_lambda,
        )
        for j in range(Q_oracle.shape[1])
    ], dtype=float)

    V_opt_arr = orthonormalize_columns(np.asarray(V_opt, dtype=diag_dtype), dtype=diag_dtype)
    projected_into_opt = V_opt_arr @ (V_opt_arr.T @ raw_proj) if V_opt_arr.size else np.zeros_like(raw_proj)
    opt_proj_norms = np.linalg.norm(projected_into_opt, axis=0)
    principal_cosines = subspace_principal_cosines(V_opt_arr, Q_oracle)
    raw_overlap = np.nan
    if raw_proj.shape[1] >= 2:
        raw_overlap = abs(float(raw_proj[:, 0] @ raw_proj[:, 1]))
    # breakpoint()

    return {
        "raw_oracle_scores": raw_scores,
        "raw_oracle_score_sum": float(np.sum(raw_scores)),
        "qr_oracle_scores": qr_scores,
        "qr_oracle_score_sum": float(np.sum(qr_scores)),
        "opt_proj_norms": np.asarray(opt_proj_norms, dtype=float),
        "opt_vs_qoracle_cosines": np.asarray(principal_cosines, dtype=float),
        "raw_oracle_overlap": float(raw_overlap),
    }


def oracle_projection_candidate(
    M_gain, A_block, V_exact, rank, rows_ref, state_prev=None,
    score_variant="legacy", old_row_memory=None, row_concentration_lambda=0.0,
    oracle_projection_row_samples=None,
):
    if V_exact is None or np.asarray(V_exact).size == 0:
        return None

    cand_dtype = np.result_type(np.asarray(M_gain).dtype, np.asarray(V_exact).dtype, np.float64)
    Q_oracle, _ = projected_true_span_oracle(
        np.asarray(M_gain, dtype=cand_dtype),
        np.asarray(V_exact, dtype=cand_dtype)[:, : int(rank)],
        int(rank),
        dtype=cand_dtype,
        row_samples=oracle_projection_row_samples,
    )
    if Q_oracle.shape[1] < int(rank):
        return None

    scores = np.zeros(int(rank), dtype=float)
    s_vals = np.zeros(int(rank), dtype=float)
    H_vals = np.zeros(int(rank), dtype=float)
    for j in range(int(rank)):
        scores[j], s_vals[j], H_vals[j] = score_full_vector_details_forget(
            M_gain, A_block, Q_oracle[:, j], rows_ref,
            state_prev=state_prev,
            score_variant=score_variant,
            old_row_memory=old_row_memory,
            row_concentration_lambda=row_concentration_lambda,
        )

    return {
        "V": np.ascontiguousarray(Q_oracle[:, : int(rank)], dtype=np.asarray(M_gain).dtype),
        "score": scores,
        "s": s_vals,
        "H": H_vals,
        "score_sum": float(np.sum(scores)),
    }


def entropyscore_forget_joint_full_eval(
    M_gain, A_block, Z, rows_ref, state_prev=None, optimizer="cex",
    row_concentration_lambda=0.0
):
    Z_arr = np.ascontiguousarray(np.asarray(Z))
    r = Z_arr.shape[1]
    vals = np.zeros(r, dtype=float)
    G = np.zeros_like(Z_arr)
    s = np.zeros(r, dtype=float)
    H = np.zeros(r, dtype=float)

    for j in range(r):
        z = Z_arr[:, j]
        if state_prev is None:
            if optimizer == "cex":
                logf, grad_log, s[j], H[j] = entropyscore_forget_logscore_grad_rows(A_block, z, rows_ref)
            elif optimizer == "legacy":
                logf, grad_log, s[j], H[j] = entropyscore_forget_logscore_grad_rows(A_block, z, rows_ref)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")
        else:
            if optimizer == "cex":
                logf, grad_log, s[j], H[j] = entropyscore_forget_streaming_logscore_grad(
                    M_gain, A_block, state_prev["V"], state_prev["s2"], z, rows_ref
                )
            elif optimizer == "legacy":
                logf, grad_log, s[j], H[j] = entropyscore_forget_streaming_logscore_grad(
                    M_gain, A_block, state_prev["V"], state_prev["s2"], z, rows_ref
                )
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")

        vals[j] = float(np.exp(logf))
        G[:, j] = np.ascontiguousarray(vals[j] * grad_log, dtype=Z_arr.dtype)
        vals[j], G[:, j] = apply_row_concentration_regularizer(
            vals[j], G[:, j], A_block, z, row_concentration_lambda
        )

    return float(np.sum(vals)), vals, G, s, H


def entropyscore_forget_joint_full_score_tangent(
    M_gain, A_block, Z, rows_ref, state_prev=None, optimizer="cex", Qz=None,
    row_concentration_lambda=0.0
):
    total, vals, G, s, H = entropyscore_forget_joint_full_eval(
        M_gain, A_block, Z, rows_ref, state_prev=state_prev, optimizer=optimizer,
        row_concentration_lambda=row_concentration_lambda
    )
    Xi = stiefel_tangent_gradient(Z, G, Qz)
    return total, vals, G, Xi, s, H


def stiefel_tangent_rotation_split(Z, Xi, Qz=None):
    Z_arr = np.ascontiguousarray(np.asarray(Z))
    Xi_arr = np.ascontiguousarray(np.asarray(Xi, dtype=Z_arr.dtype))
    Xi_tan = stiefel_tangent_gradient(Z_arr, Xi_arr, Qz)
    rot = Z_arr @ (Z_arr.T @ Xi_tan)
    outside = Xi_tan - rot
    return {
        "tangent": Xi_tan,
        "inside_rotation": rot,
        "outside_complement": outside,
        "inside_rotation_norm": float(np.linalg.norm(rot, ord="fro")),
        "outside_complement_norm": float(np.linalg.norm(outside, ord="fro")),
        "tangent_norm": float(np.linalg.norm(Xi_tan, ord="fro")),
    }


def projected_subspace_svd(M_gain, V_basis):
    if V_basis is None or not V_basis.size:
        return V_basis, np.zeros(0)
    Q, _ = np.linalg.qr(V_basis, mode="reduced")
    B_proj = M_gain @ Q
    _, s_proj, R_proj_h = la.svd(B_proj, full_matrices=False, lapack_driver="gesdd")
    R_proj = R_proj_h.T
    V_proj = Q @ R_proj
    return V_proj, s_proj


def left_projected_operator_svd_factors(Vt, combined):
    dtype = np.result_type(np.asarray(Vt).dtype, np.asarray(combined).dtype)
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float64
    V_basis = orthonormalize_columns(np.asarray(Vt, dtype=dtype).T, dtype=dtype)
    if V_basis.size == 0:
        return (
            np.zeros((0, combined.shape[0]), dtype=dtype),
            np.zeros(0, dtype=dtype),
            np.zeros((0, combined.shape[1]), dtype=dtype),
            np.zeros((combined.shape[0], 0), dtype=dtype),
        )

    combined_arr = np.asarray(combined, dtype=dtype)
    Y = combined_arr @ V_basis
    if Y.size == 0:
        U_hat = np.zeros((combined.shape[0], 0), dtype=dtype)
    else:
        U_hat, _ = np.linalg.qr(np.asarray(Y, dtype=np.float64), mode="reduced")
        U_hat = np.ascontiguousarray(U_hat.astype(dtype, copy=False))
    if U_hat.size == 0:
        return (
            np.zeros((0, combined.shape[0]), dtype=dtype),
            np.zeros(0, dtype=dtype),
            np.zeros((0, combined.shape[1]), dtype=dtype),
            U_hat,
        )

    compressed = np.ascontiguousarray(U_hat.T @ combined_arr, dtype=dtype)
    U_small, s_proj, Vt_proj = np.linalg.svd(compressed, full_matrices=False)
    U_proj = np.ascontiguousarray(U_hat @ U_small, dtype=dtype)
    return (
        np.ascontiguousarray(U_proj.T, dtype=dtype),
        np.asarray(s_proj, dtype=dtype),
        np.ascontiguousarray(Vt_proj, dtype=dtype),
        U_hat,
    )


def load_matlab_cex_input(mat_input):
    if not os.path.exists(mat_input):
        raise FileNotFoundError(f"{mat_input} not found. Run export_cex1_input.m in activated MATLAB first.")

    data = sio.loadmat(mat_input)
    required = ("A", "V", "svec")
    missing = [name for name in required if name not in data]
    if missing:
        raise KeyError(f"{mat_input} is missing required variable(s): {', '.join(missing)}")

    A = np.asarray(data["A"], dtype=float)
    V = np.asarray(data["V"], dtype=float)
    svec = np.asarray(data["svec"], dtype=float).reshape(-1)
    if "sigma1" in data:
        sigma1 = float(np.asarray(data["sigma1"]).reshape(-1)[0])
    else:
        sigma1 = float(svec[0])

    if A.ndim != 2 or V.ndim != 2:
        raise ValueError("Loaded A and V must be matrices.")
    if A.shape[1] != V.shape[0]:
        raise ValueError(f"Incompatible loaded shapes: A={A.shape}, V={V.shape}.")
    if svec.size < 2:
        raise ValueError("Loaded svec must contain at least two singular values.")

    return A, V, svec, sigma1


def generate_structured_cex_input(
    n=1024,
    r_sig=2,
    alpha_sig=0.003,
    alpha_tail=0.0145,
    tail_scale=0.99,
    sigma1=0.991,
    v_type="id",
    u_sig_basis="hadamard",   # "hadamard", "gaussian", "hadamard+noise"
    u_sig_noise=0.0,           # std for "hadamard+noise"
    tail_decay="power",        # "power" or "exponential"
):
    """Structured cex matrix.

    u_sig_basis controls the first r_sig columns of U (the signal directions):
      "hadamard"        — original behavior: first r_sig hadamard cols / sqrt(n).
      "gaussian"        — i.i.d. Gaussian columns (orthogonalized by QR below).
      "hadamard+noise"  — hadamard cols + u_sig_noise * randn (then QR).

    tail_decay controls how σ decays:
      "power"       — original: σ_j = tail_scale · j^(-alpha_tail).
      "exponential" — log-linear interp (= exponential decay) between the
                      same start/end as "power", so endpoints are preserved
                      but the middle is fuller.
    """
    if n <= r_sig:
        raise ValueError("n must be larger than r_sig.")
    needs_hadamard = u_sig_basis in ("hadamard", "hadamard+noise")
    if needs_hadamard and (n & (n - 1)):
        raise ValueError("n must be a power of two for hadamard-based u_sig_basis.")

    k = n
    U0 = np.zeros((n, n), dtype=float)
    if u_sig_basis == "hadamard":
        H = la.hadamard(n).astype(float)
        U0[:, :r_sig] = H[:, :r_sig] / np.sqrt(n)
    elif u_sig_basis == "gaussian":
        # Raw Gaussian columns; QR below will normalize and orthogonalize.
        U0[:, :r_sig] = np.random.randn(n, r_sig)
    elif u_sig_basis == "hadamard+noise":
        H = la.hadamard(n).astype(float)
        U0[:, :r_sig] = H[:, :r_sig] / np.sqrt(n) + u_sig_noise * np.random.randn(n, r_sig)
    else:
        raise ValueError(
            f"u_sig_basis must be one of: 'hadamard', 'gaussian', 'hadamard+noise' (got {u_sig_basis!r})."
        )

    a_tail = np.sqrt(1.0 - r_sig / n)
    b_tail = 1.0 / np.sqrt(n)
    for j in range(r_sig, n):
        col = np.zeros(n, dtype=float)
        idx_large = j - r_sig
        if idx_large > n - r_sig - 1:
            raise RuntimeError("Tail index out of range; reduce r_sig or adjust construction.")
        col[idx_large] = a_tail
        col[n - r_sig:n] = b_tail
        U0[:, j] = col

    U, _ = np.linalg.qr(U0, mode="reduced")
    for j in range(r_sig):
        if float(U[:, j].T @ U0[:, j]) < 0:
            U[:, j] = -U[:, j]

    if v_type == "id":
        V = np.eye(n, k)
    elif v_type == "U":
        V = U
    elif v_type == "rand":
        V, _ = np.linalg.qr(np.random.randn(n, k), mode="reduced")
    else:
        raise ValueError("v_type must be one of: id, U, rand.")

    if tail_decay == "power":
        sig_block = sigma1 * np.arange(1, r_sig + 1, dtype=float) ** (-alpha_sig)
        tail_block = tail_scale * np.arange(1, k - r_sig + 1, dtype=float) ** (-alpha_tail)
    elif tail_decay == "exponential":
        # Preserve start/end of the power-law tail; replace the middle
        # by a log-linear interpolation (= exponential decay in σ).
        if r_sig == 1:
            sig_block = np.array([sigma1])
        else:
            sig_start = sigma1
            sig_end = sigma1 * (float(r_sig) ** (-alpha_sig))
            sig_block = np.exp(np.linspace(np.log(sig_start), np.log(sig_end), r_sig))
        tail_len = k - r_sig
        if tail_len == 1:
            tail_block = np.array([tail_scale])
        else:
            tail_start = tail_scale
            tail_end = tail_scale * (float(tail_len) ** (-alpha_tail))
            tail_block = np.exp(np.linspace(np.log(tail_start), np.log(tail_end), tail_len))
    else:
        raise ValueError(f"tail_decay must be 'power' or 'exponential' (got {tail_decay!r}).")

    svec = np.concatenate([sig_block, tail_block])
    svec[0] = sigma1

    A_unpermuted = (U * svec[None, :]) @ V.T
    p = np.random.permutation(n)
    A = A_unpermuted[p, :]
    return A, V, svec, sigma1


def generate_futures_term_structure_input(n=1024, preset="fast", seed=0):
    """Synthetic futures-curve return panel with level/slope leaders and roll shocks."""
    if n < 8:
        raise ValueError("n must be at least 8 for futures-term-structure.")

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    tau = np.linspace(0.0, 1.0, n)

    level = np.ones(n)
    slope = tau - tau.mean()
    curvature = (tau - 0.5) ** 2
    curvature -= curvature.mean()
    carry = np.exp(-3.5 * tau)
    carry -= carry.mean()
    basis = np.column_stack([level, slope, curvature, carry])
    Qv, _ = np.linalg.qr(basis, mode="reduced")

    level_factor = 1.2 * np.sin(2.0 * np.pi * 3.0 * t + 0.25) + 0.35 * rng.standard_normal(n)
    slope_factor = 0.9 * np.sign(np.sin(2.0 * np.pi * 5.0 * t)) + 0.25 * rng.standard_normal(n)
    curve_factor = 0.35 * np.sin(2.0 * np.pi * 11.0 * t + 0.7) + 0.20 * rng.standard_normal(n)
    carry_factor = 0.22 * rng.standard_t(df=5, size=n)
    factors = np.column_stack([level_factor, slope_factor, curve_factor, carry_factor])
    factors -= factors.mean(axis=0, keepdims=True)

    if preset == "small":
        strengths = np.array([1.00, 0.96, 0.34, 0.22])
        roll_amp = 0.135
        noise_scale = 0.018
    else:
        strengths = np.array([1.00, 0.94, 0.38, 0.24])
        roll_amp = 0.16
        noise_scale = 0.02

    A = factors @ (Qv * strengths[None, :]).T

    front = np.exp(-18.0 * tau)
    front /= max(np.linalg.norm(front), 1e-30)
    back = np.exp(-18.0 * (1.0 - tau))
    back /= max(np.linalg.norm(back), 1e-30)
    roll_phase = np.sin(2.0 * np.pi * 8.0 * t)
    roll_sign = np.where(roll_phase >= 0.0, 1.0, -1.0)
    A += roll_amp * roll_sign[:, None] * (front - 0.55 * back)[None, :]

    jump_rows = np.arange(0, n, max(4, n // 16))
    for row in jump_rows:
        maturity = (3 * row + 5) % n
        width = max(2.0 / n, 0.018)
        shock = np.exp(-0.5 * ((tau - tau[maturity]) / width) ** 2)
        shock /= max(np.linalg.norm(shock), 1e-30)
        A[row, :] += 0.11 * ((-1.0) ** row) * shock

    A += noise_scale * rng.standard_normal((n, n))
    A -= A.mean(axis=0, keepdims=True)

    _, svec, Vh = np.linalg.svd(A, full_matrices=False)
    V_exact = Vh.T
    sigma1 = float(svec[0])
    return A, V_exact, svec, sigma1


def generate_crowded_strategy_input(n=1024, preset="fast", seed=0):
    """Synthetic crowded long/short strategy panel with staggered unwind episodes."""
    if n < 8:
        raise ValueError("n must be at least 8 for crowded-strategy.")

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, endpoint=False)

    crowd = np.ones(n)
    crowd[n // 2:] = -1.0
    crowd += 0.08 * rng.standard_normal(n)

    sectors = min(8, max(2, n // 16))
    sector = np.zeros(n)
    edges = np.linspace(0, n, sectors + 1, dtype=int)
    for j in range(sectors):
        sector[edges[j]:edges[j + 1]] = 1.0 if j % 2 == 0 else -1.0

    liquidity = np.linspace(-1.0, 1.0, n)
    crowded_names = rng.choice(n, size=max(1, n // 12), replace=False)
    liquidity[crowded_names] += 2.5 * np.sign(crowd[crowded_names])

    value = np.sin(2.0 * np.pi * np.arange(n) / max(2, n))
    basis = np.column_stack([crowd, sector, liquidity, value])
    Qv, _ = np.linalg.qr(basis, mode="reduced")

    centers = np.array([0.18, 0.52, 0.83])
    widths = np.array([0.035, 0.06, 0.04])
    unwind = np.zeros(n)
    for center, width in zip(centers, widths):
        unwind += np.exp(-0.5 * ((t - center) / width) ** 2)
    unwind += 0.18 + 0.04 * rng.standard_normal(n)

    staggered = np.roll(unwind, max(1, n // 10))
    staggered *= 1.0 + 0.25 * np.sin(2.0 * np.pi * np.arange(n) / max(2, n))
    rebalance = np.sign(np.sin(2.0 * np.pi * 9.0 * t + 0.2))
    liquidity_wave = np.sin(2.0 * np.pi * 5.0 * t + 0.7) + 0.35 * rng.standard_t(df=5, size=n)
    factors = np.column_stack([unwind, staggered, rebalance, liquidity_wave])
    factors -= factors.mean(axis=0, keepdims=True)

    if preset == "small":
        strengths = np.array([1.00, 0.985, 0.42, 0.30])
        idio_scale = 0.010
        burst_amp = 0.090
    else:
        strengths = np.array([1.00, 0.990, 0.46, 0.32])
        idio_scale = 0.012
        burst_amp = 0.110

    A = factors @ (Qv * strengths[None, :]).T

    for row in np.arange(0, n, max(4, n // 12)):
        names = rng.choice(n, size=max(2, n // 24), replace=False)
        A[row, names] += burst_amp * np.sign(crowd[names])

    A += idio_scale * rng.standard_normal((n, n))
    A -= A.mean(axis=0, keepdims=True)

    _, svec, Vh = np.linalg.svd(A, full_matrices=False)
    V_exact = Vh.T
    sigma1 = float(svec[0])
    return A, V_exact, svec, sigma1


def generate_stat_arb_spreads_input(n=1024, preset="fast", seed=0):
    """Synthetic statistical-arbitrage spread panel with near-tied residual modes."""
    if n < 16:
        raise ValueError("n must be at least 16 for stat-arb-spreads.")

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    x = np.linspace(0.0, 1.0, n)
    sectors = min(10, max(4, n // 16))
    sector_id = (np.arange(n) * sectors) // n

    pair_spread = np.where((np.arange(n) % 2) == 0, 1.0, -1.0)
    pair_spread += 0.28 * np.sin(2.0 * np.pi * x)
    sector_spread = np.zeros(n)
    for g in range(sectors):
        mask = sector_id == g
        local = np.linspace(-1.0, 1.0, int(np.sum(mask)))
        sector_spread[mask] = ((-1.0) ** g) * (1.0 - 0.35 * local ** 2)
    borrow_liquidity = 0.45 * np.cos(4.0 * np.pi * x) + np.where(x > 0.72, 1.0, -0.35)
    beta_residual = x - x.mean()
    reversal_bucket = np.sin(8.0 * np.pi * x + 0.4) + 0.25 * np.sign(np.sin(22.0 * np.pi * x))
    exposures = np.column_stack([pair_spread, sector_spread, borrow_liquidity, beta_residual, reversal_bucket])
    Qv = orthonormalize_columns(exposures, dtype=float)

    mean_reversion = 0.92 * np.sin(2.0 * np.pi * 4.0 * t + 0.25)
    mean_reversion += 0.30 * np.sin(2.0 * np.pi * 13.0 * t)
    mean_reversion += 0.12 * rng.standard_normal(n)
    crowded_residual = np.roll(mean_reversion, max(1, n // 14))
    crowded_residual *= 0.88 + 0.24 * np.exp(-0.5 * ((t - 0.63) / 0.11) ** 2)
    liquidity_factor = np.sign(np.sin(2.0 * np.pi * 7.0 * t + 0.15))
    liquidity_factor += 0.55 * np.exp(-0.5 * ((t - 0.36) / 0.055) ** 2)
    beta_factor = 0.52 * np.cos(2.0 * np.pi * 3.0 * t + 0.7) + 0.18 * rng.standard_t(df=5, size=n)
    reversal_factor = 0.30 * np.sin(2.0 * np.pi * 17.0 * t + 0.9)
    factors = np.column_stack([mean_reversion, crowded_residual, liquidity_factor, beta_factor, reversal_factor])
    factors -= factors.mean(axis=0, keepdims=True)

    if preset == "small":
        strengths = np.array([1.00, 0.982, 0.44, 0.30, 0.22])
        break_amp = 0.090
        noise_scale = 0.011
    else:
        strengths = np.array([1.00, 0.988, 0.48, 0.32, 0.24])
        break_amp = 0.110
        noise_scale = 0.013

    A = factors @ (Qv * strengths[None, :]).T

    width = max(2.0 / n, 0.018)
    stride = max(4, n // 14)
    for event, row in enumerate(range(0, n, stride)):
        center = ((7 * row + 3) % n) / max(n - 1, 1)
        local_break = np.exp(-0.5 * ((x - center) / width) ** 2)
        local_break -= local_break.mean()
        local_break /= max(np.linalg.norm(local_break), 1e-30)
        pulse_rows = slice(row, min(row + 3, n))
        sign = -1.0 if event % 2 else 1.0
        A[pulse_rows, :] += sign * break_amp * local_break[None, :]

    hetero = 0.65 + 0.35 * (np.abs(pair_spread) / max(np.max(np.abs(pair_spread)), 1e-30))
    A += noise_scale * rng.standard_normal((n, n)) * hetero[None, :]
    A -= A.mean(axis=0, keepdims=True)

    _, svec, Vh = np.linalg.svd(A, full_matrices=False)
    V_exact = Vh.T
    sigma1 = float(svec[0])
    return A, V_exact, svec, sigma1


def generate_rates_cross_currency_input(n=1024, preset="fast", seed=0):
    """Synthetic cross-currency rates panel with funding, carry, and basis shocks."""
    if n < 16:
        raise ValueError("n must be at least 16 for rates-cross-currency.")

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    blocs = (np.arange(n) * 8) // n
    tenors = ((np.arange(n) * 5) // n) % 5

    usd_funding = np.where(blocs % 2 == 0, 1.0, -1.0)
    usd_funding += 0.22 * np.cos(2.0 * np.pi * x)
    carry_curve = np.choose(tenors, [-1.0, -0.35, 0.12, 0.58, 1.0])
    carry_curve += 0.42 * np.where(np.isin(blocs, [2, 5, 7]), 1.0, -0.5)
    basis_smile = (x - 0.5) ** 2
    basis_smile -= basis_smile.mean()
    regional_basis = np.sin(6.0 * np.pi * x) + 0.35 * (blocs == 3)
    exposures = np.column_stack([usd_funding, carry_curve, basis_smile, regional_basis])
    Qv, _ = np.linalg.qr(exposures, mode="reduced")

    liquidity = 1.0 + 0.25 * np.sin(2.0 * np.pi * t)
    crisis = np.exp(-0.5 * ((t - 0.68) / 0.07) ** 2)
    fixing = np.exp(-0.5 * ((t - 0.18) / 0.04) ** 2)
    funding_factor = liquidity + 1.15 * crisis + 0.08 * rng.standard_normal(n)
    carry_factor = np.sign(np.sin(5.0 * np.pi * t)) * (0.48 + fixing)
    carry_factor += 0.08 * rng.standard_t(df=5, size=n)
    basis_factor = np.cos(3.0 * np.pi * t) + 0.35 * crisis + 0.06 * rng.standard_normal(n)
    regional_factor = np.where((t > 0.42) & (t < 0.55), 1.0, -0.25)
    regional_factor += 0.20 * np.sin(17.0 * np.pi * t)
    factors = np.column_stack([funding_factor, carry_factor, basis_factor, regional_factor])
    factors -= factors.mean(axis=0, keepdims=True)

    if preset == "small":
        strengths = np.array([1.00, 0.985, 0.36, 0.24])
        basis_jump_amp = 0.075
        noise_scale = 0.012
    else:
        strengths = np.array([1.00, 0.990, 0.40, 0.27])
        basis_jump_amp = 0.095
        noise_scale = 0.014

    A = factors @ (Qv * strengths[None, :]).T

    jump_cols = np.arange(0, n, max(4, n // 16))
    for row, col in enumerate(jump_cols):
        width = max(2.0 / n, 0.022)
        local_basis = np.exp(-0.5 * ((x - x[col]) / width) ** 2)
        local_basis /= max(np.linalg.norm(local_basis), 1e-30)
        A[(3 * row + n // 7) % n, :] += basis_jump_amp * ((-1.0) ** row) * local_basis

    A += noise_scale * rng.standard_normal((n, n))
    A -= A.mean(axis=0, keepdims=True)

    _, svec, Vh = np.linalg.svd(A, full_matrices=False)
    V_exact = Vh.T
    sigma1 = float(svec[0])
    return A, V_exact, svec, sigma1


def generate_risk_residual_panel_input(n=1024, preset="fast", seed=0):
    """Synthetic quant residual-return panel with crowded residual modes."""
    if n <= 8:
        raise ValueError("n must be larger than 8 for risk-residual-panel.")

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    groups = min(8, max(4, n // 16))
    asset_groups = np.array_split(np.arange(n), groups)

    V0 = np.zeros((n, n), dtype=float)
    crowding = np.zeros(n, dtype=float)
    liquidity = np.zeros(n, dtype=float)
    for g, idx in enumerate(asset_groups):
        local = np.linspace(-1.0, 1.0, idx.size)
        side = 1.0 if g % 2 == 0 else -1.0
        crowding[idx] = side * (1.0 + 0.25 * np.sin(np.pi * local))
        liquidity[idx] = (local - local.mean()) * (1.0 + 0.12 * g)
    V0[:, 0] = crowding - crowding.mean()
    V0[:, 1] = liquidity - liquidity.mean()

    for j in range(2, n):
        idx = j - 2
        group = asset_groups[idx % groups]
        V0[idx, j] = 1.0
        V0[:, j] += 0.015 * rng.standard_normal(n)
        V0[group, j] += 0.08 * rng.standard_normal(group.size)
        V0[:, j] -= V0[:, j].mean()
    V, _ = np.linalg.qr(V0, mode="reduced")
    for j, ref in enumerate((V0[:, 0], V0[:, 1])):
        if float(V[:, j] @ ref) < 0.0:
            V[:, j] = -V[:, j]

    U0 = np.zeros((n, n), dtype=float)
    crowding_event = np.exp(-0.5 * ((t - 0.64) / 0.105) ** 2)
    crowding_event += 0.42 * np.exp(-0.5 * ((t - 0.22) / 0.055) ** 2)
    crowding_event *= 1.0 + 0.2 * np.sin(2.0 * np.pi * 5.0 * t)
    residual_turnover = np.sin(2.0 * np.pi * 3.0 * t + 0.35)
    residual_turnover += 0.45 * np.sin(2.0 * np.pi * 11.0 * t)
    residual_turnover *= 0.8 + 0.5 * np.exp(-0.5 * ((t - 0.78) / 0.09) ** 2)
    U0[:, 0] = crowding_event - crowding_event.mean()
    U0[:, 1] = residual_turnover - residual_turnover.mean()
    for j in range(2, n):
        center = (j - 2 + 0.5) / max(1, n - 2)
        width = 0.012 + 0.018 * ((j % 5) / 4.0)
        pulse = np.exp(-0.5 * (((t - center + 0.5) % 1.0 - 0.5) / width) ** 2)
        pulse += 0.03 * rng.standard_normal(n)
        U0[:, j] = pulse - pulse.mean()
    U, _ = np.linalg.qr(U0, mode="reduced")
    for j in range(2):
        if float(U[:, j] @ U0[:, j]) < 0.0:
            U[:, j] = -U[:, j]

    if preset == "small":
        signal_gap = 0.012
        tail_scale = 0.975
        idio_scale = 0.006
    else:
        signal_gap = 0.010
        tail_scale = 0.982
        idio_scale = 0.007

    s_signal = np.array([1.0, 1.0 - signal_gap])
    tail_len = n - s_signal.size
    tail = tail_scale * np.arange(1, tail_len + 1, dtype=float) ** (-0.035)
    tail *= 1.0 + 0.018 * np.sin(np.arange(tail_len, dtype=float) * 0.71)
    svec = np.concatenate([s_signal, tail])

    A = (U * svec[None, :]) @ V.T
    vol = 0.82 + 0.5 * np.exp(-0.5 * ((t - 0.66) / 0.12) ** 2)
    vol += 0.08 * np.sin(2.0 * np.pi * 13.0 * t)
    A = vol[:, None] * A
    A += idio_scale * rng.standard_t(df=5, size=A.shape) / np.sqrt(5.0 / 3.0)
    A -= A.mean(axis=0, keepdims=True)

    _, svec, Vh = np.linalg.svd(A, full_matrices=False)
    V_exact = Vh.T
    sigma1 = float(svec[0])
    return A, V_exact, svec, sigma1


def generate_macro_factor_panel_input(n=1024, preset="fast", seed=0):
    """Synthetic macro factor panel with near-tied growth/inflation leaders."""
    if n < 16:
        raise ValueError("n must be at least 16 for macro-factor-panel.")

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    regions = (np.arange(n) * 6) // n
    sectors = ((np.arange(n) * 9) // n) % 9

    inflation_beta = 0.8 * np.cos(2.0 * np.pi * x) + np.where(np.isin(regions, [1, 4]), 0.55, -0.20)
    growth_beta = np.sin(2.0 * np.pi * x + 0.35) + np.where(sectors < 3, 0.45, -0.25)
    rates_beta = x - x.mean()
    dollar_beta = np.where(np.isin(regions, [0, 3, 5]), 1.0, -0.55)
    commodity_beta = np.sin(8.0 * np.pi * x) + 0.35 * (sectors == 7)
    liquidity_beta = np.exp(-3.2 * x) - np.exp(-3.2 * (1.0 - x))
    exposures = np.column_stack(
        [inflation_beta, growth_beta, rates_beta, dollar_beta, commodity_beta, liquidity_beta]
    )
    Qv = orthonormalize_columns(exposures, dtype=float)

    inflation_cycle = 0.95 * np.sin(2.0 * np.pi * 2.0 * t + 0.2)
    inflation_cycle += 0.55 * np.exp(-0.5 * ((t - 0.68) / 0.10) ** 2)
    inflation_cycle += 0.10 * rng.standard_normal(n)
    growth_cycle = 0.90 * np.cos(2.0 * np.pi * 2.0 * t - 0.4)
    growth_cycle -= 0.50 * np.exp(-0.5 * ((t - 0.38) / 0.08) ** 2)
    growth_cycle += 0.10 * rng.standard_normal(n)
    rates_cycle = 0.52 * np.sin(2.0 * np.pi * 5.0 * t + 0.6) + 0.12 * rng.standard_t(df=5, size=n)
    dollar_cycle = np.sign(np.sin(2.0 * np.pi * 4.0 * t + 0.1)) * (0.34 + 0.24 * (t > 0.58))
    commodity_cycle = 0.26 * np.sin(2.0 * np.pi * 11.0 * t + 1.2)
    liquidity_cycle = 0.22 * rng.standard_t(df=6, size=n)
    factors = np.column_stack(
        [inflation_cycle, growth_cycle, rates_cycle, dollar_cycle, commodity_cycle, liquidity_cycle]
    )
    factors -= factors.mean(axis=0, keepdims=True)

    if preset == "small":
        strengths = np.array([1.00, 0.985, 0.46, 0.34, 0.24, 0.18])
        event_amp = 0.105
        noise_scale = 0.011
    else:
        strengths = np.array([1.00, 0.990, 0.50, 0.36, 0.26, 0.20])
        event_amp = 0.125
        noise_scale = 0.013

    A = factors @ (Qv * strengths[None, :]).T

    stride = max(4, n // 14)
    width = max(2.0 / n, 0.020)
    for j, row in enumerate(range(0, n, stride)):
        center = ((7 * row + 3) % n) / max(n - 1, 1)
        local = np.exp(-0.5 * ((x - center) / width) ** 2)
        local -= local.mean()
        local /= max(np.linalg.norm(local), 1e-30)
        shock_sign = -1.0 if j % 2 else 1.0
        A[row:min(row + 2, n), :] += shock_sign * event_amp * local[None, :]

    release_cols = np.zeros(n)
    release_cols[:: max(1, n // 24)] = 1.0
    release_cols -= release_cols.mean()
    release_cols /= max(np.linalg.norm(release_cols), 1e-30)
    release_factor = 0.08 * np.sign(np.sin(2.0 * np.pi * 12.0 * t + 0.5))
    A += release_factor[:, None] * release_cols[None, :]

    A += noise_scale * rng.standard_normal((n, n))
    A -= A.mean(axis=0, keepdims=True)

    _, svec, Vh = np.linalg.svd(A, full_matrices=False)
    V_exact = Vh.T
    sigma1 = float(svec[0])
    return A, V_exact, svec, sigma1


def generate_options_vol_surface_input(n=1024, preset="fast", seed=0):
    """Synthetic option implied-vol surface panel with level/skew/smile leaders."""
    if n < 8:
        raise ValueError("n must be at least 8 for options-vol-surface.")

    rng = np.random.default_rng(seed)
    n_maturities = max(3, int(np.floor(np.sqrt(n))))
    n_strikes = int(np.ceil(n / n_maturities))
    maturities = np.exp(np.linspace(np.log(7.0 / 365.0), np.log(2.0), n_maturities))
    log_moneyness = np.linspace(-0.35, 0.35, n_strikes)
    T_grid, x_grid = np.meshgrid(maturities, log_moneyness, indexing="ij")
    T = T_grid.reshape(-1)[:n]
    x = x_grid.reshape(-1)[:n]

    tau = (np.log(T) - np.mean(np.log(T))) / max(np.std(np.log(T)), 1e-12)
    x_std = x / max(np.std(x), 1e-12)
    smile = x_std ** 2
    smile -= np.mean(smile)
    right_wing = np.exp(-0.5 * ((x - 0.14) / 0.055) ** 2)
    left_wing = np.exp(-0.5 * ((x + 0.16) / 0.075) ** 2)
    surface_modes = np.column_stack([np.ones(n), tau, x_std, smile, tau * x_std, right_wing - left_wing])
    Qv = orthonormalize_columns(surface_modes, dtype=float)

    t = np.linspace(0.0, 1.0, n, endpoint=False)
    level_factor = 1.15 * np.sin(2.0 * np.pi * 2.0 * t + 0.35) + 0.22 * rng.standard_normal(n)
    term_factor = 0.62 * np.cos(2.0 * np.pi * 3.0 * t + 0.6) + 0.18 * rng.standard_normal(n)
    skew_factor = -0.88 * np.sin(2.0 * np.pi * 5.0 * t + 0.1) + 0.20 * rng.standard_normal(n)
    smile_factor = 0.42 * np.sign(np.sin(2.0 * np.pi * 7.0 * t)) + 0.15 * rng.standard_normal(n)
    cross_factor = 0.26 * np.sin(2.0 * np.pi * 11.0 * t + 1.2)
    wing_factor = np.zeros(n)
    for loc, amp in zip([n // 5, n // 2, (4 * n) // 5], [0.36, -0.31, 0.29]):
        width = max(2.0, n / 30.0)
        wing_factor += amp * np.exp(-0.5 * ((np.arange(n) - loc) / width) ** 2)

    factors = np.column_stack([level_factor, term_factor, skew_factor, smile_factor, cross_factor, wing_factor])
    factors -= factors.mean(axis=0, keepdims=True)

    if preset == "small":
        strengths = np.array([1.00, 0.82, 0.64, 0.40, 0.24, 0.18])
        noise_scale = 0.010
        quote_band = 0.010
    else:
        strengths = np.array([1.00, 0.80, 0.66, 0.42, 0.26, 0.20])
        noise_scale = 0.012
        quote_band = 0.012

    A = factors @ (Qv * strengths[None, :]).T
    row_phase = np.sin(2.0 * np.pi * np.arange(n)[:, None] / max(n_strikes, 2))
    col_phase = np.cos(2.0 * np.pi * np.arange(n)[None, :] / max(n_maturities, 2))
    liquidity = np.exp(-0.5 * (x / 0.30) ** 2)
    A += quote_band * row_phase * col_phase * liquidity[None, :]
    A += noise_scale * rng.standard_normal((n, n)) * (0.55 + 0.45 * liquidity[None, :])
    A -= A.mean(axis=0, keepdims=True)

    _, svec, Vh = np.linalg.svd(A, full_matrices=False)
    V_exact = Vh.T
    sigma1 = float(svec[0])
    return A, V_exact, svec, sigma1


def generate_etf_basket_basis_input(n=1024, preset="fast", seed=0):
    """Synthetic ETF/basket basis panel with dense basket modes and local dislocations."""
    if n < 8:
        raise ValueError("n must be at least 8 for etf-basket-basis.")

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    x = np.linspace(0.0, 1.0, n)

    index_mode = np.ones(n)
    sector_a = np.where(x < 0.33, 1.0, np.where(x < 0.66, -0.45, -0.15))
    sector_b = np.where((x >= 0.33) & (x < 0.66), 1.0, -0.35)
    basis_slope = x - x.mean()
    hedge_tilt = np.sin(2.0 * np.pi * x) + 0.35 * np.cos(6.0 * np.pi * x)
    dense_basis = np.column_stack([index_mode, sector_a, sector_b, basis_slope, hedge_tilt])
    Qv, _ = np.linalg.qr(dense_basis, mode="reduced")

    common_factor = 1.15 * np.sin(2.0 * np.pi * 2.0 * t + 0.2) + 0.35 * rng.standard_normal(n)
    sector_factor_a = 0.68 * np.sin(2.0 * np.pi * 5.0 * t + 0.8) + 0.22 * rng.standard_normal(n)
    sector_factor_b = 0.55 * np.sign(np.sin(2.0 * np.pi * 7.0 * t + 0.1)) + 0.18 * rng.standard_normal(n)
    basis_factor = 0.36 * rng.standard_t(df=5, size=n)
    hedge_factor = 0.24 * np.sin(2.0 * np.pi * 13.0 * t + 0.5) + 0.12 * rng.standard_normal(n)
    factors = np.column_stack([common_factor, sector_factor_a, sector_factor_b, basis_factor, hedge_factor])
    factors -= factors.mean(axis=0, keepdims=True)

    if preset == "small":
        strengths = np.array([1.00, 0.82, 0.76, 0.38, 0.28])
        dislocation_amp = 0.15
        noise_scale = 0.016
    else:
        strengths = np.array([1.00, 0.84, 0.78, 0.40, 0.30])
        dislocation_amp = 0.17
        noise_scale = 0.018

    A = factors @ (Qv * strengths[None, :]).T

    basket_width = max(2.0 / n, 0.022)
    stride = max(4, n // 18)
    for row in range(0, n, stride):
        center = ((5 * row + 11) % n) / max(n - 1, 1)
        local = np.exp(-0.5 * ((x - center) / basket_width) ** 2)
        local -= local.mean()
        local /= max(np.linalg.norm(local), 1e-30)
        sign = -1.0 if (row // stride) % 2 else 1.0
        A[row:min(row + 2, n), :] += sign * dislocation_amp * local[None, :]

    etf_col = np.zeros(n)
    etf_col[:: max(1, n // 32)] = 1.0
    etf_col -= etf_col.mean()
    etf_col /= max(np.linalg.norm(etf_col), 1e-30)
    stale_factor = 0.11 * np.sign(np.sin(2.0 * np.pi * 9.0 * t + 0.3))
    A += stale_factor[:, None] * etf_col[None, :]

    A += noise_scale * rng.standard_normal((n, n))
    A -= A.mean(axis=0, keepdims=True)

    _, svec, Vh = np.linalg.svd(A, full_matrices=False)
    V_exact = Vh.T
    sigma1 = float(svec[0])
    return A, V_exact, svec, sigma1


def generate_realized_vol_corr_input(n=1024, preset="fast", seed=0):
    """Synthetic realized-vol/correlation panel with market-vol and corr-cluster modes."""
    if n < 16:
        raise ValueError("n must be at least 16 for realized-vol-corr.")

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    x = np.linspace(0.0, 1.0, n)
    sectors = min(8, max(4, n // 16))
    sector_id = (np.arange(n) * sectors) // n

    market_vol = np.ones(n)
    corr_cluster = np.where(sector_id % 2 == 0, 1.0, -0.8)
    corr_cluster += 0.22 * np.cos(2.0 * np.pi * x)
    term_decay = np.exp(-3.0 * x)
    term_decay -= term_decay.mean()
    dispersion = np.sin(6.0 * np.pi * x) + 0.28 * rng.standard_normal(n)
    exposures = np.column_stack([market_vol, corr_cluster, term_decay, dispersion])
    Qv = orthonormalize_columns(exposures, dtype=float)

    vol_spike = np.exp(-0.5 * ((t - 0.30) / 0.045) ** 2)
    vol_spike += 0.85 * np.exp(-0.5 * ((t - 0.73) / 0.070) ** 2)
    market_factor = 0.72 + 1.1 * vol_spike + 0.18 * rng.standard_t(df=5, size=n)
    corr_factor = np.sin(2.0 * np.pi * 4.0 * t + 0.4)
    corr_factor += 0.95 * np.exp(-0.5 * ((t - 0.63) / 0.085) ** 2)
    corr_factor += 0.12 * rng.standard_normal(n)
    term_factor = 0.55 * np.sign(np.sin(2.0 * np.pi * 7.0 * t)) + 0.10 * rng.standard_normal(n)
    dispersion_factor = 0.38 * rng.standard_t(df=6, size=n)
    factors = np.column_stack([market_factor, corr_factor, term_factor, dispersion_factor])
    factors -= factors.mean(axis=0, keepdims=True)

    if preset == "small":
        strengths = np.array([1.00, 0.975, 0.42, 0.30])
        burst_amp = 0.090
        noise_scale = 0.012
    else:
        strengths = np.array([1.00, 0.982, 0.46, 0.32])
        burst_amp = 0.110
        noise_scale = 0.014

    A = factors @ (Qv * strengths[None, :]).T

    stride = max(4, n // 14)
    width = max(2.0 / n, 0.018)
    for j, row in enumerate(range(0, n, stride)):
        center = ((7 * row + 3) % n) / max(n - 1, 1)
        local_corr = np.exp(-0.5 * ((x - center) / width) ** 2)
        local_corr -= local_corr.mean()
        local_corr /= max(np.linalg.norm(local_corr), 1e-30)
        A[row:min(row + 2, n), :] += burst_amp * ((-1.0) ** j) * local_corr[None, :]

    leverage = 0.06 * np.sin(2.0 * np.pi * 11.0 * t + 0.2)
    A += leverage[:, None] * (corr_cluster / max(np.linalg.norm(corr_cluster), 1e-30))[None, :]
    A += noise_scale * rng.standard_normal((n, n))
    A -= A.mean(axis=0, keepdims=True)

    _, svec, Vh = np.linalg.svd(A, full_matrices=False)
    V_exact = Vh.T
    sigma1 = float(svec[0])
    return A, V_exact, svec, sigma1


def generate_intraday_liquidity_shape_input(n=1024, preset="fast", seed=0):
    """Synthetic intraday liquidity panel with U-shape and auction-pressure modes."""
    if n < 16:
        raise ValueError("n must be at least 16 for intraday-liquidity-shape.")

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    x = np.linspace(0.0, 1.0, n)
    buckets = min(8, max(4, n // 16))
    bucket_id = (np.arange(n) * buckets) // n

    u_shape = np.exp(-6.0 * x) + np.exp(-6.0 * (1.0 - x))
    u_shape -= u_shape.mean()
    close_auction = np.exp(-0.5 * ((x - 0.92) / 0.075) ** 2)
    close_auction -= close_auction.mean()
    venue_split = np.where(bucket_id % 2 == 0, 1.0, -0.85)
    venue_split += 0.18 * np.sin(4.0 * np.pi * x)
    midday_drought = -np.exp(-0.5 * ((x - 0.52) / 0.16) ** 2)
    midday_drought -= midday_drought.mean()
    spread_slope = x - x.mean()
    exposures = np.column_stack([u_shape, close_auction, venue_split, midday_drought, spread_slope])
    Qv = orthonormalize_columns(exposures, dtype=float)

    open_close_wave = 1.05 * np.exp(-0.5 * ((t - 0.09) / 0.055) ** 2)
    open_close_wave += 0.92 * np.exp(-0.5 * ((t - 0.86) / 0.075) ** 2)
    open_close_wave += 0.12 * rng.standard_normal(n)
    auction_pressure = 0.86 * np.sin(2.0 * np.pi * 3.0 * t + 0.2)
    auction_pressure += 0.82 * np.exp(-0.5 * ((t - 0.78) / 0.08) ** 2)
    auction_pressure += 0.10 * rng.standard_normal(n)
    venue_factor = 0.50 * np.sign(np.sin(2.0 * np.pi * 7.0 * t + 0.35))
    venue_factor += 0.14 * rng.standard_t(df=5, size=n)
    drought_factor = 0.38 * np.cos(2.0 * np.pi * 5.0 * t + 0.55)
    drought_factor -= 0.40 * np.exp(-0.5 * ((t - 0.48) / 0.10) ** 2)
    slope_factor = 0.24 * np.sin(2.0 * np.pi * 13.0 * t) + 0.10 * rng.standard_normal(n)
    factors = np.column_stack([open_close_wave, auction_pressure, venue_factor, drought_factor, slope_factor])
    factors -= factors.mean(axis=0, keepdims=True)

    if preset == "small":
        strengths = np.array([1.00, 0.982, 0.43, 0.32, 0.22])
        burst_amp = 0.085
        noise_scale = 0.010
    else:
        strengths = np.array([1.00, 0.988, 0.46, 0.34, 0.24])
        burst_amp = 0.105
        noise_scale = 0.012

    A = factors @ (Qv * strengths[None, :]).T

    stride = max(4, n // 16)
    width = max(2.0 / n, 0.017)
    for j, row in enumerate(range(0, n, stride)):
        center = ((11 * row + 5) % n) / max(n - 1, 1)
        local_liquidity = np.exp(-0.5 * ((x - center) / width) ** 2)
        local_liquidity -= local_liquidity.mean()
        local_liquidity /= max(np.linalg.norm(local_liquidity), 1e-30)
        A[row:min(row + 2, n), :] += burst_amp * ((-1.0) ** j) * local_liquidity[None, :]

    halt_cols = np.zeros(n)
    halt_cols[:: max(1, n // 28)] = 1.0
    halt_cols -= halt_cols.mean()
    halt_cols /= max(np.linalg.norm(halt_cols), 1e-30)
    halt_factor = 0.065 * np.sign(np.sin(2.0 * np.pi * 10.0 * t + 0.15))
    A += halt_factor[:, None] * halt_cols[None, :]

    A += noise_scale * rng.standard_normal((n, n)) * (0.70 + 0.30 * np.abs(u_shape)[None, :])
    A -= A.mean(axis=0, keepdims=True)

    _, svec, Vh = np.linalg.svd(A, full_matrices=False)
    V_exact = Vh.T
    sigma1 = float(svec[0])
    return A, V_exact, svec, sigma1


def generate_execution_cost_slippage_input(n=1024, preset="fast", seed=0):
    """Synthetic execution-cost/slippage panel with liquidity and impact leaders."""
    if n < 16:
        raise ValueError("n must be at least 16 for execution-cost-slippage.")

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    x = np.linspace(0.0, 1.0, n)
    venues = min(8, max(4, n // 16))
    venue_id = (np.arange(n) * venues) // n

    spread_liquidity = np.exp(-2.4 * x) + 0.30 * np.where(venue_id % 2 == 0, 1.0, -0.55)
    impact_curve = np.sqrt(x + 0.03)
    impact_curve -= impact_curve.mean()
    venue_pressure = np.where(venue_id % 3 == 0, 1.0, np.where(venue_id % 3 == 1, -0.65, 0.20))
    queue_imbalance = np.sin(8.0 * np.pi * x + 0.25) + 0.24 * rng.standard_normal(n)
    close_bucket = np.exp(-0.5 * ((x - 0.82) / 0.075) ** 2)
    close_bucket -= close_bucket.mean()
    exposures = np.column_stack([spread_liquidity, impact_curve, venue_pressure, queue_imbalance, close_bucket])
    Qv = orthonormalize_columns(exposures, dtype=float)

    open_auction = np.exp(-0.5 * ((t - 0.08) / 0.045) ** 2)
    close_auction = np.exp(-0.5 * ((t - 0.86) / 0.060) ** 2)
    liquidity_factor = 0.78 + 0.92 * open_auction + 0.70 * close_auction
    liquidity_factor += 0.12 * rng.standard_t(df=5, size=n)
    impact_factor = 0.90 * np.sin(2.0 * np.pi * 3.0 * t + 0.35)
    impact_factor += 0.62 * np.exp(-0.5 * ((t - 0.58) / 0.09) ** 2)
    impact_factor += 0.10 * rng.standard_normal(n)
    venue_factor = 0.52 * np.sign(np.sin(2.0 * np.pi * 9.0 * t + 0.15))
    venue_factor += 0.16 * rng.standard_t(df=6, size=n)
    queue_factor = 0.36 * np.sin(2.0 * np.pi * 17.0 * t + 0.8)
    queue_factor += 0.12 * rng.standard_normal(n)
    close_factor = 0.30 * np.exp(-0.5 * ((t - 0.78) / 0.08) ** 2)
    close_factor += 0.12 * np.sign(np.sin(2.0 * np.pi * 5.0 * t))
    factors = np.column_stack([liquidity_factor, impact_factor, venue_factor, queue_factor, close_factor])
    factors -= factors.mean(axis=0, keepdims=True)

    if preset == "small":
        strengths = np.array([1.00, 0.980, 0.44, 0.30, 0.22])
        slippage_amp = 0.095
        noise_scale = 0.012
    else:
        strengths = np.array([1.00, 0.987, 0.48, 0.32, 0.24])
        slippage_amp = 0.115
        noise_scale = 0.014

    A = factors @ (Qv * strengths[None, :]).T

    stride = max(4, n // 15)
    width = max(2.0 / n, 0.018)
    for j, row in enumerate(range(0, n, stride)):
        center = ((11 * row + 5) % n) / max(n - 1, 1)
        liquidity_pocket = np.exp(-0.5 * ((x - center) / width) ** 2)
        liquidity_pocket -= liquidity_pocket.mean()
        liquidity_pocket /= max(np.linalg.norm(liquidity_pocket), 1e-30)
        sign = -1.0 if j % 2 else 1.0
        A[row:min(row + 2, n), :] += sign * slippage_amp * liquidity_pocket[None, :]

    venue_cols = np.zeros(n)
    venue_cols[:: max(1, n // 20)] = 1.0
    venue_cols -= venue_cols.mean()
    venue_cols /= max(np.linalg.norm(venue_cols), 1e-30)
    routing_factor = 0.08 * np.sign(np.sin(2.0 * np.pi * 13.0 * t + 0.45))
    A += routing_factor[:, None] * venue_cols[None, :]
    A += noise_scale * rng.standard_normal((n, n)) * (0.65 + 0.35 * np.abs(spread_liquidity)[None, :])
    A -= A.mean(axis=0, keepdims=True)

    _, svec, Vh = np.linalg.svd(A, full_matrices=False)
    V_exact = Vh.T
    sigma1 = float(svec[0])
    return A, V_exact, svec, sigma1


def generate_alternative_data_signals_input(n=1024, preset="fast", seed=0):
    """Synthetic alternative-data signal panel with delayed source revisions."""
    if n < 16:
        raise ValueError("n must be at least 16 for alternative-data-signals.")

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    x = np.linspace(0.0, 1.0, n)
    sectors = min(10, max(4, n // 16))
    sources = ((np.arange(n) * 5) // n) % 5
    sector_id = (np.arange(n) * sectors) // n

    web_growth = np.sin(2.0 * np.pi * x) + 0.35 * np.where(sector_id % 2 == 0, 1.0, -0.8)
    card_spend = np.cos(2.0 * np.pi * x + 0.25) + 0.45 * np.where(np.isin(sources, [1, 3]), 1.0, -0.45)
    app_engagement = np.where(sources == 2, 1.0, -0.25) + 0.30 * np.sin(6.0 * np.pi * x)
    sentiment = np.where(sector_id < sectors // 2, 1.0, -0.65) + 0.22 * rng.standard_normal(n)
    satellite = np.exp(-3.0 * x) - np.exp(-3.0 * (1.0 - x))
    coverage = 0.55 + 0.45 * ((np.arange(n) % 7) == 0)
    coverage -= coverage.mean()
    exposures = np.column_stack([web_growth, card_spend, app_engagement, sentiment, satellite, coverage])
    Qv = orthonormalize_columns(exposures, dtype=float)

    search_factor = 0.95 * np.sin(2.0 * np.pi * 3.0 * t + 0.25)
    search_factor += 0.42 * np.exp(-0.5 * ((t - 0.58) / 0.085) ** 2)
    search_factor += 0.10 * rng.standard_normal(n)
    spend_factor = np.roll(search_factor, max(1, n // 20))
    spend_factor *= 0.92 + 0.18 * np.sin(2.0 * np.pi * 2.0 * t)
    app_factor = 0.52 * np.sign(np.sin(2.0 * np.pi * 7.0 * t + 0.1))
    app_factor += 0.18 * rng.standard_t(df=5, size=n)
    sentiment_factor = 0.40 * np.cos(2.0 * np.pi * 5.0 * t + 0.7)
    sentiment_factor += 0.34 * np.exp(-0.5 * ((t - 0.31) / 0.045) ** 2)
    satellite_factor = 0.28 * np.sin(2.0 * np.pi * 11.0 * t + 1.1)
    coverage_factor = 0.22 * rng.standard_t(df=6, size=n)
    factors = np.column_stack(
        [search_factor, spend_factor, app_factor, sentiment_factor, satellite_factor, coverage_factor]
    )
    factors -= factors.mean(axis=0, keepdims=True)

    if preset == "small":
        strengths = np.array([1.00, 0.982, 0.44, 0.32, 0.23, 0.16])
        revision_amp = 0.090
        noise_scale = 0.012
    else:
        strengths = np.array([1.00, 0.988, 0.48, 0.34, 0.25, 0.18])
        revision_amp = 0.110
        noise_scale = 0.014

    A = factors @ (Qv * strengths[None, :]).T

    stride = max(4, n // 15)
    width = max(2.0 / n, 0.020)
    for j, row in enumerate(range(0, n, stride)):
        center = ((11 * row + 5) % n) / max(n - 1, 1)
        cluster = np.exp(-0.5 * ((x - center) / width) ** 2)
        cluster -= cluster.mean()
        cluster /= max(np.linalg.norm(cluster), 1e-30)
        sign = -1.0 if j % 2 else 1.0
        A[row:min(row + 2, n), :] += sign * revision_amp * cluster[None, :]

    delayed_cols = np.zeros(n)
    delayed_cols[:: max(1, n // 20)] = 1.0
    delayed_cols -= delayed_cols.mean()
    delayed_cols /= max(np.linalg.norm(delayed_cols), 1e-30)
    delay_factor = 0.08 * np.sign(np.sin(2.0 * np.pi * 10.0 * t + 0.4))
    A += delay_factor[:, None] * delayed_cols[None, :]

    coverage_weight = 0.55 + 0.45 * (sources == 0) + 0.25 * ((np.arange(n) % 11) == 0)
    A += noise_scale * rng.standard_normal((n, n)) * coverage_weight[None, :]
    A -= A.mean(axis=0, keepdims=True)

    _, svec, Vh = np.linalg.svd(A, full_matrices=False)
    V_exact = Vh.T
    sigma1 = float(svec[0])
    return A, V_exact, svec, sigma1


def generate_diffuse_diffuse_input(
    n=1024,
    r_sig=2,
    alpha_sig=0.003,
    alpha_tail=0.0145,
    tail_scale=0.99,
    sigma1=0.991,
    v_type="rand",
    seed=0,
):
    if n <= r_sig:
        raise ValueError("n must be larger than r_sig.")
    if n & (n - 1):
        raise ValueError("n must be a power of two for scipy.linalg.hadamard.")

    rng = np.random.default_rng(seed)
    U0 = np.zeros((n, n), dtype=float)
    H = la.hadamard(n).astype(float)
    U0[:, :r_sig] = H[:, :r_sig] / np.sqrt(n)

    G = rng.standard_normal((n, n - r_sig))
    G -= U0[:, :r_sig] @ (U0[:, :r_sig].T @ G)
    Qtail, _ = np.linalg.qr(G, mode="reduced")
    U0[:, r_sig:] = Qtail

    U, _ = np.linalg.qr(U0, mode="reduced")
    for j in range(r_sig):
        if float(U[:, j].T @ U0[:, j]) < 0:
            U[:, j] = -U[:, j]

    if v_type == "id":
        V = np.eye(n, n)
    elif v_type == "U":
        V = U
    elif v_type == "rand":
        V, _ = np.linalg.qr(rng.standard_normal((n, n)), mode="reduced")
    else:
        raise ValueError("v_type must be one of: id, U, rand.")

    sig_block = sigma1 * np.arange(1, r_sig + 1, dtype=float) ** (-alpha_sig)
    tail_block = tail_scale * np.arange(1, n - r_sig + 1, dtype=float) ** (-alpha_tail)
    svec = np.concatenate([sig_block, tail_block])
    svec[0] = sigma1

    A = (U * svec[None, :]) @ V.T
    return A, V, svec, sigma1


def generate_mixed_tail_input(
    n=1024,
    r_sig=2,
    alpha_sig=0.003,
    alpha_tail=0.0145,
    tail_scale=0.99,
    sigma1=0.991,
    v_type="rand",
    seed=0,
    tail_spikiness=0.5,
):
    """Hadamard signal with a tail interpolating between diffuse and localized."""
    if n <= r_sig:
        raise ValueError("n must be larger than r_sig.")
    if n & (n - 1):
        raise ValueError("n must be a power of two for scipy.linalg.hadamard.")
    tail_spikiness = float(tail_spikiness)
    if not (0.0 <= tail_spikiness <= 1.0):
        raise ValueError("tail_spikiness must be in [0, 1].")

    rng = np.random.default_rng(seed)
    H = la.hadamard(n).astype(float)
    U_sig = H[:, :r_sig] / np.sqrt(n)

    G = rng.standard_normal((n, n - r_sig))
    G -= U_sig @ (U_sig.T @ G)
    U_diffuse_tail, _ = np.linalg.qr(G, mode="reduced")

    U_spiky_raw = np.zeros((n, n - r_sig), dtype=float)
    a_tail = np.sqrt(1.0 - r_sig / n)
    b_tail = 1.0 / np.sqrt(n)
    for j in range(n - r_sig):
        col = np.zeros(n, dtype=float)
        col[j] = a_tail
        col[n - r_sig:n] = b_tail
        U_spiky_raw[:, j] = col
    U_spiky_raw -= U_sig @ (U_sig.T @ U_spiky_raw)
    U_spiky_tail, _ = np.linalg.qr(U_spiky_raw, mode="reduced")

    tail_raw = (
        np.sqrt(1.0 - tail_spikiness) * U_diffuse_tail
        + np.sqrt(tail_spikiness) * U_spiky_tail
    )
    tail_raw -= U_sig @ (U_sig.T @ tail_raw)
    U_tail, _ = np.linalg.qr(tail_raw, mode="reduced")

    U0 = np.column_stack([U_sig, U_tail])
    U, _ = np.linalg.qr(U0, mode="reduced")
    for j in range(r_sig):
        if float(U[:, j].T @ U_sig[:, j]) < 0:
            U[:, j] = -U[:, j]

    if v_type == "id":
        V = np.eye(n, n)
    elif v_type == "U":
        V = U
    elif v_type == "rand":
        V, _ = np.linalg.qr(rng.standard_normal((n, n)), mode="reduced")
    else:
        raise ValueError("v_type must be one of: id, U, rand.")

    sig_block = sigma1 * np.arange(1, r_sig + 1, dtype=float) ** (-alpha_sig)
    tail_block = tail_scale * np.arange(1, n - r_sig + 1, dtype=float) ** (-alpha_tail)
    svec = np.concatenate([sig_block, tail_block])
    svec[0] = sigma1

    A = (U * svec[None, :]) @ V.T
    return A, V, svec, sigma1


def generate_residual_spiky_shocks_input(n=1024, preset="fast", seed=0):
    """Residual-panel geometry of risk-residual-panel, but with sparse
    row-localized idiosyncratic shocks instead of dense heavy-tailed
    heteroskedastic noise. Intended as a factorial counterfactual against
    risk-residual-panel to isolate whether failure is driven by noise
    diffuseness or by residualization."""
    if n <= 8:
        raise ValueError("n must be larger than 8 for residual-spiky-shocks.")

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    groups = min(8, max(4, n // 16))
    asset_groups = np.array_split(np.arange(n), groups)

    V0 = np.zeros((n, n), dtype=float)
    crowding = np.zeros(n, dtype=float)
    liquidity = np.zeros(n, dtype=float)
    for g, idx in enumerate(asset_groups):
        local = np.linspace(-1.0, 1.0, idx.size)
        side = 1.0 if g % 2 == 0 else -1.0
        crowding[idx] = side * (1.0 + 0.25 * np.sin(np.pi * local))
        liquidity[idx] = (local - local.mean()) * (1.0 + 0.12 * g)
    V0[:, 0] = crowding - crowding.mean()
    V0[:, 1] = liquidity - liquidity.mean()
    for j in range(2, n):
        idx = j - 2
        group = asset_groups[idx % groups]
        V0[idx, j] = 1.0
        V0[:, j] += 0.015 * rng.standard_normal(n)
        V0[group, j] += 0.08 * rng.standard_normal(group.size)
        V0[:, j] -= V0[:, j].mean()
    V, _ = np.linalg.qr(V0, mode="reduced")
    for j, ref in enumerate((V0[:, 0], V0[:, 1])):
        if float(V[:, j] @ ref) < 0.0:
            V[:, j] = -V[:, j]

    U0 = np.zeros((n, n), dtype=float)
    crowding_event = np.exp(-0.5 * ((t - 0.64) / 0.105) ** 2)
    crowding_event += 0.42 * np.exp(-0.5 * ((t - 0.22) / 0.055) ** 2)
    crowding_event *= 1.0 + 0.2 * np.sin(2.0 * np.pi * 5.0 * t)
    residual_turnover = np.sin(2.0 * np.pi * 3.0 * t + 0.35)
    residual_turnover += 0.45 * np.sin(2.0 * np.pi * 11.0 * t)
    residual_turnover *= 0.8 + 0.5 * np.exp(-0.5 * ((t - 0.78) / 0.09) ** 2)
    U0[:, 0] = crowding_event - crowding_event.mean()
    U0[:, 1] = residual_turnover - residual_turnover.mean()
    for j in range(2, n):
        center = (j - 2 + 0.5) / max(1, n - 2)
        width = 0.012 + 0.018 * ((j % 5) / 4.0)
        pulse = np.exp(-0.5 * (((t - center + 0.5) % 1.0 - 0.5) / width) ** 2)
        pulse += 0.03 * rng.standard_normal(n)
        U0[:, j] = pulse - pulse.mean()
    U, _ = np.linalg.qr(U0, mode="reduced")
    for j in range(2):
        if float(U[:, j] @ U0[:, j]) < 0.0:
            U[:, j] = -U[:, j]

    if preset == "small":
        signal_gap = 0.012
        tail_scale = 0.975
        spike_rate = 0.015
        spike_amp = 0.10
        noise_floor = 0.001
    else:
        signal_gap = 0.010
        tail_scale = 0.982
        spike_rate = 0.02
        spike_amp = 0.10
        noise_floor = 0.0012

    s_signal = np.array([1.0, 1.0 - signal_gap])
    tail_len = n - s_signal.size
    tail = tail_scale * np.arange(1, tail_len + 1, dtype=float) ** (-0.035)
    tail *= 1.0 + 0.018 * np.sin(np.arange(tail_len, dtype=float) * 0.71)
    svec = np.concatenate([s_signal, tail])

    A = (U * svec[None, :]) @ V.T

    n_spike_rows = max(2, int(round(spike_rate * n)))
    spike_rows = rng.choice(n, size=n_spike_rows, replace=False)
    support_size = max(2, n // 64)
    spike_support = rng.choice(n, size=support_size, replace=False)
    spike_direction = rng.standard_normal(support_size)
    spike_direction /= max(np.linalg.norm(spike_direction), 1e-30)
    spike_matrix = spike_amp * spike_direction[None, :]
    for row in spike_rows:
        A[row, spike_support] += spike_matrix[0]

    A += noise_floor * rng.standard_normal((n, n))
    A -= A.mean(axis=0, keepdims=True)

    _, svec_exact, Vh = np.linalg.svd(A, full_matrices=False)
    V_exact = Vh.T
    sigma1 = float(svec_exact[0])
    return A, V_exact, svec_exact, sigma1


def apply_row_shuffle(A, seed):
    rng = np.random.default_rng(seed)
    return np.asarray(A)[rng.permutation(A.shape[0]), :]


def generate_kernel_stocks_input(n=1024, ls=0.2236, **kwargs):
    """RBF kernel on first n rows of ~/data/data_2m.mtx, lengthscale ls."""
    import os, scipy.io as _spio
    home = os.path.expanduser("~")
    points = np.asarray(_spio.mmread(os.path.join(home, "data/data_2m.mtx")))[:int(n), :]
    sq = np.sum(points**2, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * points @ points.T, 0.0)
    A = np.exp(-d2 / (2.0 * float(ls)**2)).astype(np.float64)
    _, svec, Vh = np.linalg.svd(A, full_matrices=False)
    V_exact = Vh.T
    sigma1 = float(svec[0])
    return A, V_exact, svec, sigma1


def generate_matrix_input(matrix, n=1024, preset="fast", seed=0, shuffle_rows=False, row_shuffle_seed=None, **kwargs):
    if matrix.startswith("kernel_stocks_"):
        # name format: kernel_stocks_<N>_<ls>  e.g. kernel_stocks_1024_0.2236
        parts = matrix.split("_")
        try:
            n_kernel = int(parts[2]); ls = float(parts[3])
        except (IndexError, ValueError):
            raise ValueError(f"Bad kernel_stocks name: {matrix!r}; expected kernel_stocks_<N>_<ls>")
        A, V_exact, svec, sigma1 = generate_kernel_stocks_input(n=n_kernel, ls=ls)
    elif matrix == "static-cex":
        A, V_exact, svec, sigma1 = generate_structured_cex_input(n=n, **kwargs)
    elif matrix == "static-cex-gauss":
        A, V_exact, svec, sigma1 = generate_structured_cex_input(
            n=n, u_sig_basis="gaussian", **kwargs)
    elif matrix == "static-cex-exptail":
        A, V_exact, svec, sigma1 = generate_structured_cex_input(
            n=n, tail_decay="exponential", **kwargs)
    elif matrix == "static-cex-gauss-exptail":
        A, V_exact, svec, sigma1 = generate_structured_cex_input(
            n=n, u_sig_basis="gaussian", tail_decay="exponential", **kwargs)
    elif matrix.startswith("static-cex-noisy"):
        # Name format: static-cex-noisy[_<eps>][-exptail]
        # default eps = 0.05 if no suffix.
        rest = matrix[len("static-cex-noisy"):]
        exptail = rest.endswith("-exptail")
        if exptail:
            rest = rest[: -len("-exptail")]
        eps = 0.05
        if rest.startswith("_"):
            try:
                eps = float(rest[1:])
            except ValueError:
                raise ValueError(f"Bad static-cex-noisy name: {matrix!r}; expected static-cex-noisy[_<eps>][-exptail]")
        elif rest:
            raise ValueError(f"Bad static-cex-noisy name: {matrix!r}; expected static-cex-noisy[_<eps>][-exptail]")
        A, V_exact, svec, sigma1 = generate_structured_cex_input(
            n=n,
            u_sig_basis="hadamard+noise",
            u_sig_noise=eps,
            tail_decay="exponential" if exptail else "power",
            **kwargs,
        )
    elif matrix == "diffuse-diffuse":
        A, V_exact, svec, sigma1 = generate_diffuse_diffuse_input(n=n, seed=seed, **kwargs)
    elif matrix == "mixed-tail-soft":
        A, V_exact, svec, sigma1 = generate_mixed_tail_input(n=n, seed=seed, tail_spikiness=0.25, **kwargs)
    elif matrix == "mixed-tail-balanced":
        A, V_exact, svec, sigma1 = generate_mixed_tail_input(n=n, seed=seed, tail_spikiness=0.50, **kwargs)
    elif matrix == "mixed-tail-sharp":
        A, V_exact, svec, sigma1 = generate_mixed_tail_input(n=n, seed=seed, tail_spikiness=0.75, **kwargs)
    elif matrix == "residual-spiky-shocks":
        A, V_exact, svec, sigma1 = generate_residual_spiky_shocks_input(n=n, preset=preset, seed=seed)
    elif matrix == "alternative-data-signals":
        A, V_exact, svec, sigma1 = generate_alternative_data_signals_input(n=n, preset=preset, seed=seed)
    elif matrix == "crowded-strategy":
        A, V_exact, svec, sigma1 = generate_crowded_strategy_input(n=n, preset=preset, seed=seed)
    elif matrix == "execution-cost-slippage":
        A, V_exact, svec, sigma1 = generate_execution_cost_slippage_input(n=n, preset=preset, seed=seed)
    elif matrix == "etf-basket-basis":
        A, V_exact, svec, sigma1 = generate_etf_basket_basis_input(n=n, preset=preset, seed=seed)
    elif matrix == "futures-term-structure":
        A, V_exact, svec, sigma1 = generate_futures_term_structure_input(n=n, preset=preset, seed=seed)
    elif matrix == "intraday-liquidity-shape":
        A, V_exact, svec, sigma1 = generate_intraday_liquidity_shape_input(n=n, preset=preset, seed=seed)
    elif matrix == "macro-factor-panel":
        A, V_exact, svec, sigma1 = generate_macro_factor_panel_input(n=n, preset=preset, seed=seed)
    elif matrix == "rates-cross-currency":
        A, V_exact, svec, sigma1 = generate_rates_cross_currency_input(n=n, preset=preset, seed=seed)
    elif matrix == "stat-arb-spreads":
        A, V_exact, svec, sigma1 = generate_stat_arb_spreads_input(n=n, preset=preset, seed=seed)
    elif matrix == "options-vol-surface":
        A, V_exact, svec, sigma1 = generate_options_vol_surface_input(n=n, preset=preset, seed=seed)
    elif matrix == "risk-residual-panel":
        A, V_exact, svec, sigma1 = generate_risk_residual_panel_input(n=n, preset=preset, seed=seed)
    elif matrix == "realized-vol-corr":
        A, V_exact, svec, sigma1 = generate_realized_vol_corr_input(n=n, preset=preset, seed=seed)
    else:
        raise ValueError(f"Unknown matrix family: {matrix}")

    if shuffle_rows:
        A = apply_row_shuffle(A, seed if row_shuffle_seed is None else row_shuffle_seed)
    return A, V_exact, svec, sigma1


def entropy_iter_basis_forget(M_gain, active_r, rows_ref, V_init=None, q0=5, qmax=None,
                              krylov_depth=2, residual_tol=1e-2, expansion_maxit=8,
                              num_restarts=3, maxit=40, tol=1e-8, rng=None, verbose=True,
                              state_prev=None, A_block=None, rows_total=None,
                              reduced_optimizer="legacy", work_dtype=np.float32,
                              expansion_direction="krylov_v",
                              reuse_line_search_grad=True,
                              expansion_warm_start=False,
                              post_expansion_maxit=None,
                              basis_selection="greedy",
                              joint_warm_start_greedy=False,
                              joint_warm_start_oracle=False,
                              oracle_warm_start_target=None,
                              joint_warm_start_rotations=0,
                              joint_warm_start_rotation_angle=np.pi / 4,
                              joint_warm_start_perturbations=0,
                              joint_warm_start_perturb_scale=1e-2,
                              joint_default_svd_start=True,
                              joint_oversample=0,
                              joint_oversample_rotate="svd",
                              joint_solver="riemannian",
                              row_concentration_lambda=0.0,
                              row_leverage_lambda=0.0,
                              row_leverage_mode="none",
                              row_leverage_rank=2,
                              score_variant="legacy",
                              old_row_memory=None,
                              oracle_projection_row_samples=None,
                              combined_rank=None,
                              patience=0,
                              patience_rel_tol=1e-5):
    del rows_total
    if rng is None:
        rng = np.random.default_rng(0)
    if reduced_optimizer not in {"legacy", "cex"}:
        raise ValueError(f"Unknown reduced_optimizer: {reduced_optimizer}")
    if score_variant not in {"legacy", "oldcorrected", "combined", "l4l2", "subsetmass"}:
        raise ValueError(f"Unknown score_variant: {score_variant}")
    if score_variant in {"oldcorrected", "combined", "l4l2", "subsetmass"} and reduced_optimizer != "cex":
        raise ValueError(f"score_variant='{score_variant}' requires reduced_optimizer='cex'.")
    row_concentration_lambda = float(row_concentration_lambda)
    row_leverage_lambda = float(row_leverage_lambda)
    if row_concentration_lambda != 0.0 and reduced_optimizer != "cex":
        raise ValueError("row_concentration_lambda is only supported with reduced_optimizer='cex'.")
    if row_leverage_lambda != 0.0 and (reduced_optimizer != "cex" or score_variant != "combined"):
        raise ValueError("row_leverage_lambda requires reduced_optimizer='cex' and score_variant='combined'.")
    if row_leverage_lambda != 0.0 and str(row_leverage_mode) == "none":
        raise ValueError("row_leverage_mode must be row-norm or top-svd when row_leverage_lambda != 0.")
    if basis_selection not in {"greedy", "joint"}:
        raise ValueError(f"Unknown basis_selection: {basis_selection}")
    if combined_rank is None:
        combined_rank_eff = int(active_r)
    else:
        combined_rank_eff = max(0, min(int(combined_rank), int(active_r)))
    if basis_selection == "joint" and combined_rank_eff != int(active_r):
        raise ValueError("combined_rank override is only supported with basis_selection='greedy'.")
    if joint_oversample_rotate not in {"none", "svd"}:
        raise ValueError(f"Unknown joint_oversample_rotate: {joint_oversample_rotate}")
    if joint_solver not in {"riemannian", "slsqp"}:
        raise ValueError(f"Unknown joint_solver: {joint_solver}")
    if expansion_direction not in {"krylov_v", "residual"}:
        raise ValueError(f"Unknown expansion_direction: {expansion_direction}")
    if A_block is None:
        raise ValueError("A_block is required for entropyscore_forget.")

    work_dtype = np.dtype(work_dtype)
    M_arr = np.asarray(M_gain, dtype=work_dtype)
    A_block_arr = np.asarray(A_block, dtype=M_arr.dtype)
    is_initial_block = state_prev is None
    joint_rank = active_r + max(0, int(joint_oversample)) if basis_selection == "joint" else active_r

    if active_r <= 0:
        empty = np.zeros((M_arr.shape[1], 0), dtype=M_arr.dtype)
        return empty, np.zeros(0), np.zeros(0), np.zeros(0), {
            "seed_rank": 0,
            "max_rank": 0,
            "krylov_depth": int(krylov_depth),
            "residual_tol": float(residual_tol),
            "reduced_optimizer": reduced_optimizer,
            "basis_selection": basis_selection,
            "joint_warm_start_greedy": bool(joint_warm_start_greedy),
            "joint_warm_start_oracle": bool(joint_warm_start_oracle),
            "joint_warm_start_rotations": int(joint_warm_start_rotations),
            "joint_warm_start_rotation_angle": float(joint_warm_start_rotation_angle),
            "joint_warm_start_perturbations": int(joint_warm_start_perturbations),
            "joint_warm_start_perturb_scale": float(joint_warm_start_perturb_scale),
            "joint_default_svd_start": bool(joint_default_svd_start),
            "joint_oversample": int(joint_oversample),
            "joint_oversample_rotate": joint_oversample_rotate,
            "joint_solver": joint_solver,
            "joint_rank": int(joint_rank),
            "row_concentration_lambda": row_concentration_lambda,
            "row_leverage_lambda": row_leverage_lambda,
            "row_leverage_mode": str(row_leverage_mode),
            "row_leverage_rank": int(row_leverage_rank),
            "work_dtype": str(M_arr.dtype),
            "expansion_direction": expansion_direction,
            "reuse_line_search_grad": bool(reuse_line_search_grad),
            "expansion_warm_start": bool(expansion_warm_start),
            "post_expansion_maxit": None if post_expansion_maxit is None else int(post_expansion_maxit),
            "subspace_dims": [],
            "expansion_iters": [],
            "grad_perp_ratio": np.zeros(0),
            "regularized_score": np.zeros(0),
            "regularized_score_sum": 0.0,
        }

    min_dim = min(M_arr.shape)
    min_search_rank = joint_rank
    q0_eff = max(1, min(int(q0), M_arr.shape[1], min_dim))
    q0_eff = max(min_search_rank, q0_eff)
    q0_eff = min(q0_eff, M_arr.shape[1], min_dim)
    if qmax is None:
        qmax = min(M_arr.shape[1], max(4 * q0_eff, q0_eff + 4 * max(1, krylov_depth) * active_r, 32))
    qmax = max(q0_eff, min(int(qmax), M_arr.shape[1]))

    t0 = time.time()
    if q0_eff >= min_dim:
        _, _, vh = np.linalg.svd(M_arr, full_matrices=False)
        Vbasis = np.ascontiguousarray(vh[:q0_eff, :].T, dtype=M_arr.dtype)
    else:
        Vbasis, _, _ = build_entropy_fast_subspace(
            M_arr, active_r=min(active_r, max(1, q0_eff)), q_subspace=q0_eff, method="lanczos", dtype=M_arr.dtype
        )
    subspace_build_time = time.time() - t0

    prev_basis = None
    prev_s2 = None
    if not is_initial_block:
        prev_basis = np.ascontiguousarray(np.asarray(state_prev["V"], dtype=M_arr.dtype))
        prev_s2 = np.asarray(state_prev["s2"], dtype=M_arr.dtype)
    old_row_memory_arr = None
    if old_row_memory is not None and np.asarray(old_row_memory).size:
        old_row_memory_arr = np.asarray(old_row_memory, dtype=M_arr.dtype)
    row_leverage_weights_arr = None
    if row_leverage_lambda != 0.0:
        row_leverage_weights_arr = row_leverage_weights_from_block(
            A_block_arr, mode=row_leverage_mode, rank=row_leverage_rank
        )
        row_leverage_weights_arr = np.asarray(row_leverage_weights_arr, dtype=M_arr.dtype)
    oracle_projection_row_samples_arr = None
    if oracle_projection_row_samples is not None and np.asarray(oracle_projection_row_samples).size:
        oracle_projection_row_samples_arr = np.asarray(oracle_projection_row_samples, dtype=M_arr.dtype)

    V_out = np.zeros((M_arr.shape[1], active_r), dtype=M_arr.dtype)
    s_out = np.zeros(active_r, dtype=float)
    H_out = np.zeros(active_r, dtype=float)
    score_out = np.zeros(active_r, dtype=float)
    regularized_score_out = np.zeros(active_r, dtype=float)
    grad_perp_ratio = np.zeros(active_r, dtype=float)
    subspace_dims = []
    expansion_iters = []
    timing_totals = {"reduced_setup": 0.0, "reduced_opt": 0.0, "full_gradient": 0.0, "expansion_matvec": 0.0, "expansion_append": 0.0}
    timing_counts = {"basis_solves": 0, "restart_solves": 0, "full_gradient_evals": 0, "expansion_steps": 0}

    if verbose:
        print({
            "EntropyScoreForget_setup": {
                "M_gain_shape": M_arr.shape,
                "A_block_shape": A_block_arr.shape,
                "active_rank": active_r,
                "seed_rank": q0_eff,
                "max_rank": qmax,
                "krylov_depth": int(krylov_depth),
                "residual_tol": float(residual_tol),
                "reduced_optimizer": reduced_optimizer,
                "score_variant": score_variant,
                "basis_selection": basis_selection,
                "joint_warm_start_greedy": bool(joint_warm_start_greedy),
                "joint_warm_start_oracle": bool(joint_warm_start_oracle),
                "joint_warm_start_rotations": int(joint_warm_start_rotations),
                "joint_warm_start_rotation_angle": float(joint_warm_start_rotation_angle),
                "joint_warm_start_perturbations": int(joint_warm_start_perturbations),
                "joint_warm_start_perturb_scale": float(joint_warm_start_perturb_scale),
                "joint_default_svd_start": bool(joint_default_svd_start),
                "joint_oversample": int(joint_oversample),
                "joint_oversample_rotate": joint_oversample_rotate,
                "joint_solver": joint_solver,
                "joint_rank": int(joint_rank),
                "row_concentration_lambda": row_concentration_lambda,
                "row_leverage_lambda": row_leverage_lambda,
                "row_leverage_mode": str(row_leverage_mode),
                "row_leverage_rank": int(row_leverage_rank),
                "work_dtype": str(M_arr.dtype),
                "expansion_direction": expansion_direction,
                "reuse_line_search_grad": bool(reuse_line_search_grad),
                "expansion_warm_start": bool(expansion_warm_start),
                "post_expansion_maxit": None if post_expansion_maxit is None else int(post_expansion_maxit),
                "subspace_build_time": subspace_build_time,
            }
        })

    solve_t0 = time.time()
    V_init_work = None if V_init is None else np.ascontiguousarray(np.asarray(V_init, dtype=M_arr.dtype))
    prior_basis = orthonormalize_columns(V_init_work, dtype=M_arr.dtype) if V_init_work is not None and V_init_work.size else None

    if basis_selection == "joint":
        Z_warm = None
        best_restart = None
        best_stop = None
        best_seed_label = None
        best_seed_full = None
        joint_seed_history = []
        stop_reason = "max_expansion"
        expansion_count = 0
        V_best = None
        score_vals_best = np.zeros(joint_rank, dtype=float)
        grad_ratio = np.nan

        while True:
            t_stage = time.perf_counter()
            B_gain = np.ascontiguousarray(M_arr @ Vbasis, dtype=Vbasis.dtype)
            B_block = np.ascontiguousarray(A_block_arr @ Vbasis, dtype=Vbasis.dtype)
            C_prev = None if is_initial_block else np.ascontiguousarray(prev_basis.T @ Vbasis, dtype=Vbasis.dtype)
            R_old_block = None if is_initial_block or old_row_memory_arr is None else np.ascontiguousarray(old_row_memory_arr @ Vbasis, dtype=Vbasis.dtype)
            q = Vbasis.shape[1]
            Qz = np.zeros((q, 0), dtype=Vbasis.dtype)

            starts = []
            start_labels = []

            def add_stiefel_start(label, Z):
                if append_unique_stiefel_seed(starts, Z):
                    start_labels.append(label)
                    return True
                return False

            if expansion_warm_start and Z_warm is not None:
                add_stiefel_start("expansion-warm", retract_stiefel_reduced(Z_warm, Qz))
            if joint_warm_start_oracle and oracle_warm_start_target is not None:
                Z_oracle = make_oracle_stiefel_warm_start(
                    M_arr, Vbasis, oracle_warm_start_target, joint_rank, active_r,
                    Qz=Qz, row_samples=oracle_projection_row_samples_arr
                )
                add_stiefel_start("oracle-projected", Z_oracle)
            prior_Z = None
            if prior_basis is not None and prior_basis.shape[1] >= min(active_r, joint_rank):
                prior_Z = np.zeros((q, joint_rank), dtype=Vbasis.dtype)
                prior_cols = min(prior_basis.shape[1], joint_rank)
                prior_Z[:, :prior_cols] = np.ascontiguousarray(
                    Vbasis.T @ prior_basis[:, :prior_cols], dtype=Vbasis.dtype
                )
                for fill_idx in range(prior_cols, joint_rank):
                    if fill_idx < q:
                        prior_Z[fill_idx, fill_idx] = 1.0
                add_stiefel_start("prior-carried", retract_stiefel_reduced(prior_Z, Qz))
            if joint_warm_start_greedy:
                Z_greedy = make_greedy_stiefel_warm_start(
                    B_gain, B_block, C_prev, prev_s2, prior_Z, joint_rank, A_block_arr.shape[0], rows_ref,
                    num_restarts=max(1, num_restarts), maxit=maxit, tol=tol, rng=rng,
                    is_initial_block=is_initial_block, reduced_optimizer=reduced_optimizer,
                    reuse_line_search_grad=reuse_line_search_grad,
                    row_concentration_lambda=row_concentration_lambda,
                    score_variant=score_variant,
                    row_leverage_lambda=row_leverage_lambda,
                    row_leverage_weights=row_leverage_weights_arr,
                    R_old_block=R_old_block,
                    n_old=0 if state_prev is None else int(state_prev.get("rows_seen", 0)),
                    k_old=None if state_prev is None else len(prev_s2),
                )
                add_stiefel_start("greedy-warm", Z_greedy)
                for rot_idx, Z_rot in enumerate(make_rotated_stiefel_seeds(
                    Z_greedy, joint_warm_start_rotations, rng, max_angle=joint_warm_start_rotation_angle
                ), start=1):
                    add_stiefel_start(f"greedy-rotation-{rot_idx}", Z_rot)
                for pert_idx, Z_pert in enumerate(make_tangent_perturbed_stiefel_seeds(
                    Z_greedy, joint_warm_start_perturbations, joint_warm_start_perturb_scale, rng
                ), start=1):
                    add_stiefel_start(f"greedy-perturb-{pert_idx}", Z_pert)
            if joint_default_svd_start and q >= joint_rank:
                Z_eye = np.zeros((q, joint_rank), dtype=Vbasis.dtype)
                Z_eye[:joint_rank, :joint_rank] = np.eye(joint_rank, dtype=Vbasis.dtype)
                add_stiefel_start("subspace-svd-eye", retract_stiefel_reduced(Z_eye, Qz))

            random_start_idx = 1
            while len(starts) < max(1, num_restarts):
                Zrand = np.ascontiguousarray(rng.standard_normal((q, joint_rank)), dtype=Vbasis.dtype)
                if add_stiefel_start(f"random-{random_start_idx}", retract_stiefel_reduced(Zrand, Qz)):
                    random_start_idx += 1
                if len(starts) == 0 and q < joint_rank:
                    raise RuntimeError("Reduced basis is too small for joint Stiefel optimization.")
            timing_totals["reduced_setup"] += time.perf_counter() - t_stage

            t_stage = time.perf_counter()
            cand_results = []
            iter_budget = maxit
            if Z_warm is not None and post_expansion_maxit is not None:
                iter_budget = max(1, min(int(maxit), int(post_expansion_maxit)))
            for Z0 in starts:
                if is_initial_block:
                    if joint_solver == "slsqp":
                        cand = basic_slsqp_joint_reduced_forget(
                            B_block, Z0, Qz, A_block_arr.shape[0], rows_ref,
                            maxit=iter_budget, tol=tol, optimizer=reduced_optimizer,
                            row_concentration_lambda=row_concentration_lambda,
                            score_variant=score_variant,
                        )
                    else:
                        cand = basic_projected_ascent_joint_reduced_forget(
                            B_block, Z0, Qz, A_block_arr.shape[0], rows_ref,
                            maxit=iter_budget, tol=tol, optimizer=reduced_optimizer,
                            reuse_line_search_grad=reuse_line_search_grad,
                            row_concentration_lambda=row_concentration_lambda,
                            score_variant=score_variant,
                            row_leverage_lambda=row_leverage_lambda,
                            row_leverage_weights=row_leverage_weights_arr,
                        )
                else:
                    if joint_solver == "slsqp":
                        cand = basic_slsqp_joint_reduced_streaming_forget(
                            B_gain, B_block, C_prev, prev_s2, Z0, Qz, A_block_arr.shape[0], rows_ref,
                            maxit=iter_budget, tol=tol, optimizer=reduced_optimizer,
                            row_concentration_lambda=row_concentration_lambda,
                            score_variant=score_variant,
                            R_old_block=R_old_block,
                            n_old=0 if state_prev is None else int(state_prev.get("rows_seen", 0)),
                            k_old=None if state_prev is None else len(prev_s2),
                        )
                    else:
                        cand = basic_projected_ascent_joint_reduced_streaming_forget(
                            B_gain, B_block, C_prev, prev_s2, Z0, Qz, A_block_arr.shape[0], rows_ref,
                            maxit=iter_budget, tol=tol, optimizer=reduced_optimizer,
                            reuse_line_search_grad=reuse_line_search_grad,
                            row_concentration_lambda=row_concentration_lambda,
                            score_variant=score_variant,
                            row_leverage_lambda=row_leverage_lambda,
                            row_leverage_weights=row_leverage_weights_arr,
                            R_old_block=R_old_block,
                            n_old=0 if state_prev is None else int(state_prev.get("rows_seen", 0)),
                            k_old=None if state_prev is None else len(prev_s2),
                        )
                cand_results.append(cand)
            timing_totals["reduced_opt"] += time.perf_counter() - t_stage
            timing_counts["restart_solves"] += len(starts)

            best = None
            best_start = None
            for restart_idx, cand in enumerate(cand_results):
                if best is None or cand[1] > best[1]:
                    best = cand
                    best_start = starts[restart_idx]
                    best_restart = restart_idx + 1
                    best_seed_label = start_labels[restart_idx]

            Z_best, _, score_vals_best, s_best_all, H_best_all, best_stop = best
            best_seed_full = np.ascontiguousarray(Vbasis @ best_start, dtype=Vbasis.dtype)
            V_best = np.ascontiguousarray(Vbasis @ Z_best, dtype=Vbasis.dtype)
            gram_err = float(np.linalg.norm(V_best.T @ V_best - np.eye(joint_rank), ord="fro"))
            if V_best.shape[1] < joint_rank or gram_err > 1e-4:
                raise RuntimeError("Joint Stiefel optimizer produced an invalid frame.")
            joint_seed_history.append({
                "expansion": int(expansion_count),
                "subspace_dim": int(Vbasis.shape[1]),
                "best_restart": int(best_restart),
                "best_seed_label": best_seed_label,
                "best_score_sum": float(best[1]),
                "start_labels": list(start_labels),
                "best_seed_full": best_seed_full,
                "best_solution_full": V_best,
            })

            t_stage = time.perf_counter()
            residual_cols = []
            ratios = np.zeros(joint_rank, dtype=float)
            for j in range(joint_rank):
                r_norm, g_full_norm, r_dir = entropyscore_forget_full_gradient_residual(
                    M_arr,
                    A_block_arr,
                    V_best[:, j],
                    Vbasis,
                    state_prev,
                    rows_ref,
                    Q_prev=V_best[:, :j] if j > 0 else None,
                    return_vector=True,
                    row_concentration_lambda=row_concentration_lambda,
                    row_leverage_lambda=row_leverage_lambda,
                    row_leverage_weights=row_leverage_weights_arr,
                    score_variant=score_variant,
                    old_row_memory=old_row_memory_arr,
                )
                ratios[j] = r_norm / max(g_full_norm, 1e-30)
                residual_cols.append(r_dir)
                timing_counts["full_gradient_evals"] += 1
            grad_ratio = float(np.max(ratios)) if ratios.size else 0.0
            timing_totals["full_gradient"] += time.perf_counter() - t_stage

            if grad_ratio <= residual_tol:
                stop_reason = "subspace_grad_tol"
                break
            if Vbasis.shape[1] >= qmax:
                stop_reason = "subspace_rank_cap"
                break
            if expansion_count >= expansion_maxit:
                stop_reason = "expansion_maxit"
                break

            if expansion_direction == "residual":
                seed_cols = residual_cols
            else:
                seed_cols = [V_best[:, j] for j in range(joint_rank)]

            new_cols = []
            t_stage = time.perf_counter()
            for seed_col in seed_cols:
                g_dir = np.ascontiguousarray(seed_col, dtype=Vbasis.dtype)
                new_cols.append(g_dir)
                for _ in range(max(0, int(krylov_depth) - 1)):
                    g_dir = np.ascontiguousarray(M_arr.T @ (M_arr @ g_dir), dtype=Vbasis.dtype)
                    new_cols.append(g_dir)
            timing_totals["expansion_matvec"] += time.perf_counter() - t_stage

            prev_qdim = Vbasis.shape[1]
            t_stage = time.perf_counter()
            Vbasis = append_basis_columns(Vbasis, np.column_stack(new_cols), max_cols=qmax)
            timing_totals["expansion_append"] += time.perf_counter() - t_stage
            if Vbasis.shape[1] == prev_qdim:
                stop_reason = "no_expandable_direction"
                break

            Z_warm = np.zeros((Vbasis.shape[1], joint_rank), dtype=Vbasis.dtype)
            Z_warm[:Z_best.shape[0], :] = Z_best
            expansion_count += 1
            timing_counts["expansion_steps"] += 1

        if joint_rank > active_r and joint_oversample_rotate == "svd":
            V_rot, s_rot = projected_subspace_svd(M_arr.astype(np.float64), V_best.astype(np.float64))
            V_final = np.ascontiguousarray(V_rot[:, :active_r], dtype=M_arr.dtype)
            s_final = np.asarray(s_rot[:active_r], dtype=float)
        else:
            V_final = np.ascontiguousarray(V_best[:, :active_r], dtype=M_arr.dtype)
            s_final = np.asarray(s_best_all[:active_r], dtype=float)

        V_out[:, :active_r] = V_final[:, :active_r]
        s_out[:active_r] = s_final[:active_r]
        for j in range(active_r):
            if is_initial_block:
                score_j, _, s_j, H_j = score_grad_reduced_by_variant(
                    score_variant, A_block_arr @ Vbasis, Vbasis.T @ V_out[:, j],
                    A_block_arr.shape[0], rows_ref
                )
                reg_score_j, _, _, _ = score_grad_reduced_by_variant(
                    score_variant, A_block_arr @ Vbasis, Vbasis.T @ V_out[:, j], A_block_arr.shape[0], rows_ref,
                    row_concentration_lambda=row_concentration_lambda,
                    row_leverage_lambda=row_leverage_lambda,
                    row_leverage_weights=row_leverage_weights_arr,
                )
            else:
                score_j, _, s_j, H_j = streaming_score_grad_reduced_by_variant(
                    score_variant, M_arr @ Vbasis, A_block_arr @ Vbasis, prev_basis.T @ Vbasis,
                    prev_s2, R_old_block, Vbasis.T @ V_out[:, j], A_block_arr.shape[0], rows_ref,
                    n_old=int(state_prev.get("rows_seen", 0)), k_old=len(prev_s2)
                )
                reg_score_j, _, _, _ = streaming_score_grad_reduced_by_variant(
                    score_variant, M_arr @ Vbasis, A_block_arr @ Vbasis, prev_basis.T @ Vbasis,
                    prev_s2, R_old_block, Vbasis.T @ V_out[:, j], A_block_arr.shape[0], rows_ref,
                    n_old=int(state_prev.get("rows_seen", 0)), k_old=len(prev_s2),
                    row_concentration_lambda=row_concentration_lambda,
                    row_leverage_lambda=row_leverage_lambda,
                    row_leverage_weights=row_leverage_weights_arr,
                )
            score_out[j] = score_j
            regularized_score_out[j] = reg_score_j
            H_out[j] = H_j
        grad_perp_ratio[:active_r] = grad_ratio
        subspace_dims = [int(Vbasis.shape[1])] * active_r
        expansion_iters = [int(expansion_count)] * active_r
        timing_counts["basis_solves"] += 1

        if verbose:
            print({
                "basis": "joint",
                "best_restart": best_restart,
                "best_seed_label": best_seed_label,
                "stop_reason": stop_reason,
                "solver_stop_reason": None if best_stop is None else best_stop["reason"],
                "iters": None if best_stop is None else best_stop["iters"],
                "grad_norm": None if best_stop is None else best_stop["grad_norm"],
                "subspace_dim": int(Vbasis.shape[1]),
                "expansions": int(expansion_count),
                "joint_warm_start_greedy": bool(joint_warm_start_greedy),
                "joint_warm_start_oracle": bool(joint_warm_start_oracle),
                "joint_warm_start_rotations": int(joint_warm_start_rotations),
                "joint_warm_start_perturbations": int(joint_warm_start_perturbations),
                "joint_default_svd_start": bool(joint_default_svd_start),
                "joint_solver": joint_solver,
                "score_sum": float(np.sum(score_out[:active_r])),
                "regularized_score_sum": float(np.sum(regularized_score_out[:active_r])),
                "time": time.time() - solve_t0,
                "grad_perp_ratio": float(grad_ratio),
            })

        solve_time = time.time() - solve_t0
        diag = {
            "seed_rank": q0_eff,
            "max_rank": qmax,
            "krylov_depth": int(krylov_depth),
            "residual_tol": float(residual_tol),
            "reduced_optimizer": reduced_optimizer,
            "score_variant": score_variant,
            "basis_selection": basis_selection,
            "joint_warm_start_greedy": bool(joint_warm_start_greedy),
            "joint_warm_start_oracle": bool(joint_warm_start_oracle),
            "joint_warm_start_rotations": int(joint_warm_start_rotations),
            "joint_warm_start_rotation_angle": float(joint_warm_start_rotation_angle),
            "joint_warm_start_perturbations": int(joint_warm_start_perturbations),
            "joint_warm_start_perturb_scale": float(joint_warm_start_perturb_scale),
            "joint_default_svd_start": bool(joint_default_svd_start),
            "joint_oversample": int(joint_oversample),
            "joint_oversample_rotate": joint_oversample_rotate,
            "joint_solver": joint_solver,
            "joint_rank": int(joint_rank),
            "row_concentration_lambda": row_concentration_lambda,
            "row_leverage_lambda": row_leverage_lambda,
            "row_leverage_mode": str(row_leverage_mode),
            "row_leverage_rank": int(row_leverage_rank),
            "work_dtype": str(M_arr.dtype),
            "expansion_direction": expansion_direction,
            "reuse_line_search_grad": bool(reuse_line_search_grad),
            "expansion_warm_start": bool(expansion_warm_start),
            "post_expansion_maxit": None if post_expansion_maxit is None else int(post_expansion_maxit),
            "subspace_build_time": subspace_build_time,
            "reduced_solve_time": solve_time,
            "grad_perp_ratio": grad_perp_ratio,
            "regularized_score": regularized_score_out,
            "regularized_score_sum": float(np.sum(regularized_score_out[:active_r])),
            "subspace_dims": np.asarray(subspace_dims, dtype=int),
            "expansion_iters": np.asarray(expansion_iters, dtype=int),
            "timing_totals": dict(timing_totals),
            "timing_counts": dict(timing_counts),
            "Vbasis_final": Vbasis,
            "joint_best_restart": best_restart,
            "joint_best_seed_label": best_seed_label,
            "joint_best_seed_full": best_seed_full,
            "joint_best_solution_full": V_best,
            "joint_seed_history": joint_seed_history,
        }
        return V_out, s_out, H_out, score_out, diag

    for k_idx in range(combined_rank_eff):
        basis_t0 = time.time()
        z_warm = None
        stop_reason = "max_expansion"
        best_restart = None
        best_stop = None
        expansion_count = 0
        v_best = None
        s_best = 0.0
        H_best = np.inf
        logf_best = -np.inf
        grad_ratio = np.nan

        while True:
            t_stage = time.perf_counter()
            B_gain = np.ascontiguousarray(M_arr @ Vbasis, dtype=Vbasis.dtype)
            B_block = np.ascontiguousarray(A_block_arr @ Vbasis, dtype=Vbasis.dtype)
            C_prev = None if is_initial_block else np.ascontiguousarray(prev_basis.T @ Vbasis, dtype=Vbasis.dtype)
            R_old_block = None if is_initial_block or old_row_memory_arr is None else np.ascontiguousarray(old_row_memory_arr @ Vbasis, dtype=Vbasis.dtype)
            q = Vbasis.shape[1]
            Qz = np.ascontiguousarray(Vbasis.T @ V_out[:, :k_idx], dtype=Vbasis.dtype) if k_idx > 0 else np.zeros((q, 0), dtype=Vbasis.dtype)
            if k_idx > 0:
                Qz = orthonormalize_columns(Qz, dtype=Vbasis.dtype)

            starts = []
            if reduced_optimizer == "cex":
                if expansion_warm_start and z_warm is not None:
                    append_unique_reduced_seed(starts, retract_reduced(z_warm, Qz))
                if joint_warm_start_oracle and oracle_warm_start_target is not None:
                    z_oracle = make_oracle_reduced_warm_start(
                        M_arr, Vbasis, oracle_warm_start_target, k_idx, active_r,
                        Qz=Qz, row_samples=oracle_projection_row_samples_arr
                    )
                    append_unique_reduced_seed(starts, z_oracle)
                if V_init_work is not None and V_init_work.size and V_init_work.shape[1] > k_idx:
                    v_init_col = np.asarray(V_init_work[:, k_idx], dtype=Vbasis.dtype)
                    z_init = np.ascontiguousarray(Vbasis.T @ v_init_col, dtype=Vbasis.dtype)
                    append_unique_reduced_seed(starts, retract_reduced(z_init, Qz))
                if prev_basis is not None and prev_basis.shape[1] > k_idx:
                    v_prev_col = np.asarray(prev_basis[:, k_idx], dtype=Vbasis.dtype)
                    z_prev = np.ascontiguousarray(Vbasis.T @ v_prev_col, dtype=Vbasis.dtype)
                    append_unique_reduced_seed(starts, retract_reduced(z_prev, Qz))

                cex_restart_budget = max(0, max(1, num_restarts) - len(starts))
                if cex_restart_budget:
                    Q_full = np.ascontiguousarray(V_out[:, :k_idx], dtype=M_arr.dtype) if k_idx > 0 else np.zeros((M_arr.shape[1], 0), dtype=M_arr.dtype)
                    full_starts = make_basic_restart_seeds(M_arr, Q_full, k_idx, V_init_work, cex_restart_budget)
                    for v0 in full_starts:
                        z0 = np.ascontiguousarray(Vbasis.T @ np.asarray(v0, dtype=Vbasis.dtype), dtype=Vbasis.dtype)
                        append_unique_reduced_seed(starts, retract_reduced(z0, Qz))
            else:
                if z_warm is not None:
                    append_unique_reduced_seed(starts, retract_reduced(z_warm, Qz))
                if joint_warm_start_oracle and oracle_warm_start_target is not None:
                    z_oracle = make_oracle_reduced_warm_start(
                        M_arr, Vbasis, oracle_warm_start_target, k_idx, active_r,
                        Qz=Qz, row_samples=oracle_projection_row_samples_arr
                    )
                    append_unique_reduced_seed(starts, z_oracle)
                if prior_basis is not None and prior_basis.shape[1] > k_idx:
                    z_prior = np.ascontiguousarray(Vbasis.T @ prior_basis[:, k_idx], dtype=Vbasis.dtype)
                    append_unique_reduced_seed(starts, retract_reduced(z_prior, Qz))
                if k_idx < q:
                    e = np.zeros(q, dtype=Vbasis.dtype)
                    e[k_idx] = 1.0
                    append_unique_reduced_seed(starts, retract_reduced(e, Qz))

            while len(starts) < max(1, num_restarts):
                zrand = np.ascontiguousarray(rng.standard_normal(q), dtype=Vbasis.dtype)
                append_unique_reduced_seed(starts, retract_reduced(zrand, Qz))
                if len(starts) == 0 and q == 0:
                    raise RuntimeError("Forget basis became empty.")
            timing_totals["reduced_setup"] += time.perf_counter() - t_stage

            t_stage = time.perf_counter()
            cand_results = []
            iter_budget = maxit
            if z_warm is not None and post_expansion_maxit is not None:
                iter_budget = max(1, min(int(maxit), int(post_expansion_maxit)))
            for z0 in starts:
                if is_initial_block:
                    if reduced_optimizer == "cex":
                        cand = basic_projected_ascent_single_reduced_forget_cex(
                            B_block, z0, Qz, A_block_arr.shape[0], rows_ref,
                            maxit=iter_budget, tol=tol, reuse_line_search_grad=reuse_line_search_grad,
                            row_concentration_lambda=row_concentration_lambda,
                            score_variant=score_variant,
                            row_leverage_lambda=row_leverage_lambda,
                            row_leverage_weights=row_leverage_weights_arr,
                            patience=patience,
                            patience_rel_tol=patience_rel_tol,
                        )
                    else:
                        cand = basic_projected_ascent_single_reduced_forget(
                            B_block, z0, Qz, A_block_arr.shape[0], rows_ref, maxit=iter_budget, tol=tol
                        )
                else:
                    if reduced_optimizer == "cex":
                        cand = basic_projected_ascent_single_reduced_streaming_forget_cex(
                            B_gain, B_block, C_prev, prev_s2, z0, Qz, A_block_arr.shape[0], rows_ref,
                            maxit=iter_budget, tol=tol, reuse_line_search_grad=reuse_line_search_grad,
                            row_concentration_lambda=row_concentration_lambda,
                            score_variant=score_variant,
                            row_leverage_lambda=row_leverage_lambda,
                            row_leverage_weights=row_leverage_weights_arr,
                            R_old_block=R_old_block,
                            n_old=0 if state_prev is None else int(state_prev.get("rows_seen", 0)),
                            k_old=None if state_prev is None else len(prev_s2),
                            patience=patience,
                            patience_rel_tol=patience_rel_tol,
                        )
                    else:
                        cand = basic_projected_ascent_single_reduced_streaming_forget(
                            B_gain, B_block, C_prev, prev_s2, z0, Qz, A_block_arr.shape[0], rows_ref,
                            maxit=iter_budget, tol=tol
                        )
                cand_results.append(cand)
            timing_totals["reduced_opt"] += time.perf_counter() - t_stage
            timing_counts["restart_solves"] += len(starts)

            best = None
            for restart_idx, cand in enumerate(cand_results):
                if best is None or cand[1] > best[1]:
                    best = cand
                    best_restart = restart_idx + 1

            z_best, best_objective, s_best, H_best, best_stop = best
            v_best = np.ascontiguousarray(Vbasis @ z_best, dtype=Vbasis.dtype)
            v_best = np.ascontiguousarray(v_best / max(np.linalg.norm(v_best), 1e-30), dtype=Vbasis.dtype)

            t_stage = time.perf_counter()
            r_norm, g_full_norm, r_dir = entropyscore_forget_full_gradient_residual(
                M_arr,
                A_block_arr,
                v_best,
                Vbasis,
                state_prev,
                rows_ref,
                Q_prev=V_out[:, :k_idx] if k_idx > 0 else None,
                return_vector=True,
                row_concentration_lambda=row_concentration_lambda,
                row_leverage_lambda=row_leverage_lambda,
                row_leverage_weights=row_leverage_weights_arr,
                score_variant=score_variant,
                old_row_memory=old_row_memory_arr,
            )
            grad_ratio = r_norm / max(g_full_norm, 1e-30)
            timing_totals["full_gradient"] += time.perf_counter() - t_stage
            timing_counts["full_gradient_evals"] += 1

            if grad_ratio <= residual_tol:
                stop_reason = "subspace_grad_tol"
                break
            if Vbasis.shape[1] >= qmax:
                stop_reason = "subspace_rank_cap"
                break
            if expansion_count >= expansion_maxit:
                stop_reason = "expansion_maxit"
                break

            if expansion_direction == "residual":
                new_cols = [r_dir]
                g_dir = np.ascontiguousarray(r_dir, dtype=Vbasis.dtype)
            else:
                new_cols = [v_best]
                g_dir = np.ascontiguousarray(v_best, dtype=Vbasis.dtype)
            t_stage = time.perf_counter()
            for _ in range(max(0, int(krylov_depth) - 1)):
                g_dir = np.ascontiguousarray(M_arr.T @ (M_arr @ g_dir), dtype=Vbasis.dtype)
                new_cols.append(g_dir)
            timing_totals["expansion_matvec"] += time.perf_counter() - t_stage

            prev_qdim = Vbasis.shape[1]
            t_stage = time.perf_counter()
            Vbasis = append_basis_columns(Vbasis, np.column_stack(new_cols), max_cols=qmax)
            timing_totals["expansion_append"] += time.perf_counter() - t_stage
            if Vbasis.shape[1] == prev_qdim:
                stop_reason = "no_expandable_direction"
                break

            z_warm = np.zeros(Vbasis.shape[1], dtype=Vbasis.dtype)
            z_warm[:z_best.shape[0]] = z_best
            expansion_count += 1
            timing_counts["expansion_steps"] += 1

        V_out[:, k_idx] = v_best
        s_out[k_idx] = s_best
        H_out[k_idx] = H_best
        z_final = np.ascontiguousarray(Vbasis.T @ V_out[:, k_idx], dtype=Vbasis.dtype)
        if is_initial_block:
            raw_score, _, _, _ = score_grad_reduced_by_variant(
                score_variant, A_block_arr @ Vbasis, z_final, A_block_arr.shape[0], rows_ref
            )
            reg_score, _, _, _ = score_grad_reduced_by_variant(
                score_variant, A_block_arr @ Vbasis, z_final, A_block_arr.shape[0], rows_ref,
                row_concentration_lambda=row_concentration_lambda,
                row_leverage_lambda=row_leverage_lambda,
                row_leverage_weights=row_leverage_weights_arr,
            )
        else:
            raw_score, _, _, _ = streaming_score_grad_reduced_by_variant(
                score_variant, M_arr @ Vbasis, A_block_arr @ Vbasis, prev_basis.T @ Vbasis,
                prev_s2, R_old_block, z_final, A_block_arr.shape[0], rows_ref,
                n_old=int(state_prev.get("rows_seen", 0)), k_old=len(prev_s2)
            )
            reg_score, _, _, _ = streaming_score_grad_reduced_by_variant(
                score_variant, M_arr @ Vbasis, A_block_arr @ Vbasis, prev_basis.T @ Vbasis,
                prev_s2, R_old_block, z_final, A_block_arr.shape[0], rows_ref,
                n_old=int(state_prev.get("rows_seen", 0)), k_old=len(prev_s2),
                row_concentration_lambda=row_concentration_lambda,
                row_leverage_lambda=row_leverage_lambda,
                row_leverage_weights=row_leverage_weights_arr,
            )
        score_out[k_idx] = raw_score
        regularized_score_out[k_idx] = reg_score
        grad_perp_ratio[k_idx] = grad_ratio
        subspace_dims.append(int(Vbasis.shape[1]))
        expansion_iters.append(int(expansion_count))
        timing_counts["basis_solves"] += 1

        if verbose and ((k_idx < 10) or ((k_idx + 1) % 25 == 0) or (k_idx + 1 == active_r)):
            print({
                "basis": k_idx + 1,
                "best_restart": best_restart,
                "stop_reason": stop_reason,
                "solver_stop_reason": None if best_stop is None else best_stop["reason"],
                "iters": None if best_stop is None else best_stop["iters"],
                "grad_norm": None if best_stop is None else best_stop["grad_norm"],
                "subspace_dim": int(Vbasis.shape[1]),
                "expansions": int(expansion_count),
                "s": float(s_best),
                "H": float(H_best),
                "time": time.time() - basis_t0,
                "grad_perp_ratio": float(grad_perp_ratio[k_idx]),
            })

    if combined_rank_eff < active_r:
        r2 = int(active_r) - int(combined_rank_eff)
        V1 = V_out[:, :combined_rank_eff]
        if combined_rank_eff > 0:
            MV1 = M_arr @ V1
            M_proj = M_arr - MV1 @ V1.T
        else:
            M_proj = M_arr
        _, svd_s, svd_vh = np.linalg.svd(M_proj, full_matrices=False)
        take = min(r2, svd_vh.shape[0])
        V_svd = np.ascontiguousarray(svd_vh[:take, :].T, dtype=M_arr.dtype)
        if combined_rank_eff > 0 and take > 0:
            V_svd = V_svd - V1 @ (V1.T @ V_svd)
            V_svd, _ = np.linalg.qr(V_svd, mode="reduced")
        if take > 0:
            V_out[:, combined_rank_eff:combined_rank_eff + take] = V_svd[:, :take]
            s_out[combined_rank_eff:combined_rank_eff + take] = svd_s[:take]
        for j in range(combined_rank_eff, active_r):
            subspace_dims.append(int(Vbasis.shape[1]))
            expansion_iters.append(0)

    solve_time = time.time() - solve_t0
    diag = {
        "seed_rank": q0_eff,
        "max_rank": qmax,
        "krylov_depth": int(krylov_depth),
        "residual_tol": float(residual_tol),
        "reduced_optimizer": reduced_optimizer,
        "score_variant": score_variant,
        "basis_selection": basis_selection,
        "joint_warm_start_greedy": bool(joint_warm_start_greedy),
        "joint_warm_start_oracle": bool(joint_warm_start_oracle),
        "joint_warm_start_rotations": int(joint_warm_start_rotations),
        "joint_warm_start_rotation_angle": float(joint_warm_start_rotation_angle),
        "joint_warm_start_perturbations": int(joint_warm_start_perturbations),
        "joint_warm_start_perturb_scale": float(joint_warm_start_perturb_scale),
        "joint_default_svd_start": bool(joint_default_svd_start),
        "joint_oversample": int(joint_oversample),
        "joint_oversample_rotate": joint_oversample_rotate,
        "joint_solver": joint_solver,
        "joint_rank": int(joint_rank),
        "row_concentration_lambda": row_concentration_lambda,
        "row_leverage_lambda": row_leverage_lambda,
        "row_leverage_mode": str(row_leverage_mode),
        "row_leverage_rank": int(row_leverage_rank),
        "work_dtype": str(M_arr.dtype),
        "expansion_direction": expansion_direction,
        "reuse_line_search_grad": bool(reuse_line_search_grad),
        "expansion_warm_start": bool(expansion_warm_start),
        "post_expansion_maxit": None if post_expansion_maxit is None else int(post_expansion_maxit),
        "subspace_build_time": subspace_build_time,
        "reduced_solve_time": solve_time,
        "grad_perp_ratio": grad_perp_ratio,
        "regularized_score": regularized_score_out,
        "regularized_score_sum": float(np.sum(regularized_score_out[:active_r])),
        "subspace_dims": np.asarray(subspace_dims, dtype=int),
        "expansion_iters": np.asarray(expansion_iters, dtype=int),
        "timing_totals": dict(timing_totals),
        "timing_counts": dict(timing_counts),
        "Vbasis_final": Vbasis,
    }
    return V_out, s_out, H_out, score_out, diag


FAST_BALANCED_PRESET = {
    "q0": 5,
    "qmax": 200,
    "krylov_depth": 2,
    "residual_tol": 1e-2,
    "expansion_maxit": 64,
    "num_restarts": 2,
    "maxit": 120,
    "tol": 1e-8,
    "carry": "left",
    "reduced_optimizer": "cex",
    "basis_selection": "greedy",
    "joint_warm_start_greedy": False,
    "joint_warm_start_rotations": 0,
    "joint_warm_start_rotation_angle": np.pi / 4,
    "joint_warm_start_perturbations": 0,
    "joint_warm_start_perturb_scale": 1e-2,
    "joint_oversample": 0,
    "joint_oversample_rotate": "svd",
    "joint_solver": "riemannian",
    "dtype": "float32",
    "expansion_direction": "residual",
    "reuse_line_search_grad": True,
    "expansion_warm_start": True,
    "post_expansion_maxit": 60,
}

CEX_REPLICATE_PRESET = {
    "q0": 200,
    "qmax": 200,
    "krylov_depth": 2,
    "residual_tol": 1e-2,
    "expansion_maxit": 0,
    "num_restarts": 8,
    "maxit": 200,
    "tol": 1e-8,
    "carry": "left",
    "reduced_optimizer": "cex",
    "basis_selection": "greedy",
    "joint_warm_start_greedy": False,
    "joint_warm_start_rotations": 0,
    "joint_warm_start_rotation_angle": np.pi / 4,
    "joint_warm_start_perturbations": 0,
    "joint_warm_start_perturb_scale": 1e-2,
    "joint_oversample": 0,
    "joint_oversample_rotate": "svd",
    "joint_solver": "riemannian",
    "dtype": "float64",
    "expansion_direction": "krylov_v",
    "reuse_line_search_grad": True,
    "expansion_warm_start": False,
    "post_expansion_maxit": None,
}

SMALL_PROBE_PRESET = {
    "q0": 5,
    "qmax": 32,
    "krylov_depth": 2,
    "residual_tol": 1e-2,
    "expansion_maxit": 8,
    "num_restarts": 3,
    "maxit": 40,
    "tol": 1e-8,
    "carry": "left",
    "reduced_optimizer": "cex",
    "basis_selection": "greedy",
    "joint_warm_start_greedy": False,
    "joint_warm_start_rotations": 0,
    "joint_warm_start_rotation_angle": np.pi / 4,
    "joint_warm_start_perturbations": 0,
    "joint_warm_start_perturb_scale": 1e-2,
    "joint_oversample": 0,
    "joint_oversample_rotate": "svd",
    "joint_solver": "riemannian",
    "dtype": "float32",
    "expansion_direction": "krylov_v",
    "reuse_line_search_grad": True,
    "expansion_warm_start": False,
    "post_expansion_maxit": None,
}

PRESETS = {
    "fast": FAST_BALANCED_PRESET,
    "cex-replicate": CEX_REPLICATE_PRESET,
    "small": SMALL_PROBE_PRESET,
}


def fmt_row(x, precision=4):
    return " ".join(f"{float(v): .{precision}f}" for v in np.asarray(x).reshape(-1))


def response_entropy_stats(response):
    y = np.asarray(response, dtype=float).reshape(-1)
    y2_sq = max(float(np.dot(y, y)), 1e-30)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 1e-30)
    H = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    rel_H = H / np.log(max(y.size, 2))
    return H, rel_H, y2_sq, y4_4


def svd_sketch_update(A_block, V_r, S_r, rank, mode):
    if V_r is None or S_r is None:
        M = A_block
    else:
        M = np.vstack([S_r @ V_r.T, A_block])

    _, s, Vh = np.linalg.svd(M, full_matrices=False)
    rr = min(rank, s.size)
    if mode == "isvd":
        s_new = s[:rr]
    elif mode == "fd":
        delta = s[rr] ** 2 if s.size > rr else 0.0
        s_new = np.sqrt(np.maximum(s[:rr] ** 2 - delta, 0.0))
    else:
        raise ValueError(f"Unknown SVD sketch mode: {mode}")

    V_new = Vh[:rr, :].T
    S_new = np.diag(s_new)
    return V_new, S_new, s_new


def row_norm_seed(A_block, rank):
    """Top right singular vectors after normalizing each row to unit L2 norm."""
    A_arr = np.asarray(A_block)
    row_norms = np.linalg.norm(A_arr, axis=1, keepdims=True)
    safe = np.where(row_norms > 0, row_norms, 1.0)
    _, _, Vt = np.linalg.svd(A_arr / safe, full_matrices=False)
    return np.ascontiguousarray(Vt.T[:, : int(rank)])


def select_old_row_memory(seen_rows, V_r, max_rows, rng, return_indices=False):
    if max_rows is None or int(max_rows) <= 0:
        return (None, None) if return_indices else None
    A_seen = np.asarray(seen_rows)
    if A_seen.size == 0:
        return (None, None) if return_indices else None
    max_rows = int(max_rows)
    if A_seen.shape[0] <= max_rows:
        out = np.ascontiguousarray(A_seen.copy())
        idx = np.arange(A_seen.shape[0], dtype=int)
        return (out, idx) if return_indices else out

    uniform_count = max(1, max_rows // 2)
    high_count = max_rows - uniform_count
    uniform_idx = rng.choice(A_seen.shape[0], size=uniform_count, replace=False)

    if high_count > 0:
        if V_r is not None and np.asarray(V_r).size:
            responses = A_seen @ np.asarray(V_r)
            high_scores = np.sum(responses * responses, axis=1)
        else:
            high_scores = np.sum(A_seen * A_seen, axis=1)
        high_idx = np.argsort(high_scores)[-high_count:]
        idx = np.unique(np.concatenate([uniform_idx, high_idx]))
        if idx.size < max_rows:
            remaining = np.setdiff1d(np.arange(A_seen.shape[0]), idx, assume_unique=False)
            fill = rng.choice(remaining, size=max_rows - idx.size, replace=False)
            idx = np.concatenate([idx, fill])
    else:
        idx = uniform_idx
    idx = np.asarray(idx[:max_rows], dtype=int)
    out = np.ascontiguousarray(A_seen[idx, :].copy())
    return (out, idx) if return_indices else out


def select_old_row_holdout(seen_rows, max_rows, rng, exclude_indices=None, return_indices=False):
    if max_rows is None or int(max_rows) <= 0:
        return (None, None) if return_indices else None
    A_seen = np.asarray(seen_rows)
    if A_seen.size == 0:
        return (None, None) if return_indices else None
    all_idx = np.arange(A_seen.shape[0], dtype=int)
    if exclude_indices is not None:
        exclude = np.asarray(exclude_indices, dtype=int).reshape(-1)
        candidate_idx = np.setdiff1d(all_idx, exclude, assume_unique=False)
    else:
        candidate_idx = all_idx
    if candidate_idx.size == 0:
        return (None, None) if return_indices else None
    take = min(int(max_rows), int(candidate_idx.size))
    chosen = rng.choice(candidate_idx, size=take, replace=False)
    chosen = np.asarray(chosen, dtype=int)
    out = np.ascontiguousarray(A_seen[chosen, :].copy())
    return (out, chosen) if return_indices else out


def _fmt_scalar(x):
    if x is None:
        return "nan"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "nan"
    if np.isnan(xf):
        return "nan"
    return f"{xf:.12f}"


def _score_frame_components(M_gain, A_block, V, rows_ref, state_prev, score_variant, old_row_memory):
    V_arr = np.asarray(V)
    scores = []
    rel_h = []
    gain2 = []
    phi = []
    for j in range(V_arr.shape[1]):
        comp = combined_score_component_details(
            M_gain, A_block, V_arr[:, j], rows_ref,
            state_prev=state_prev,
            old_row_memory=old_row_memory,
        )
        scores.append(float(comp["score_total"]))
        rel_h.append(float(comp["pooled_rel_H"]))
        gain2.append(float(comp["gain2"]))
        phi.append(float(comp["phi"]))
    return {
        "score_sum": float(np.sum(scores)) if scores else np.nan,
        "score": np.asarray(scores, dtype=float),
        "pooled_rel_H": np.asarray(rel_h, dtype=float),
        "gain2": np.asarray(gain2, dtype=float),
        "phi": np.asarray(phi, dtype=float),
    }


def dump_consecutive_tail_diagnostics(
    block_idx, start0, end0, M_gain, A_block, V_exact, V_opt, Q_oracle, rank, rows_ref,
    state_prev, score_variant, old_row_memory, old_row_memory_idx, holdout_memory,
    prev_V_opt=None,
):
    if score_variant != "combined" or V_exact is None or np.asarray(V_exact).size == 0:
        print("consecutive_tail_diag: skipped requires combined score and V_exact")
        return

    diag_dtype = np.result_type(np.asarray(M_gain).dtype, np.asarray(V_exact).dtype, np.float64)
    V_opt_q = orthonormalize_columns(np.asarray(V_opt, dtype=diag_dtype)[:, : int(rank)], dtype=diag_dtype)
    Q_oracle_q = orthonormalize_columns(np.asarray(Q_oracle, dtype=diag_dtype)[:, : int(rank)], dtype=diag_dtype)
    V_sig = orthonormalize_columns(np.asarray(V_exact, dtype=diag_dtype)[:, : int(rank)], dtype=diag_dtype)

    sig_proj = V_sig @ (V_sig.T @ V_opt_q) if V_sig.size and V_opt_q.size else np.zeros_like(V_opt_q)
    sig_mass = float(np.linalg.norm(sig_proj, ord="fro") ** 2 / max(int(rank), 1)) if V_opt_q.size else np.nan
    tail_mass = float(max(0.0, 1.0 - sig_mass)) if not np.isnan(sig_mass) else np.nan
    opt_oracle_cos = subspace_principal_cosines(V_opt_q, Q_oracle_q)

    train = _score_frame_components(
        M_gain, A_block, V_opt_q, rows_ref, state_prev, score_variant, old_row_memory
    )
    oracle_train = (
        _score_frame_components(M_gain, A_block, Q_oracle_q, rows_ref, state_prev, score_variant, old_row_memory)
        if Q_oracle_q.size else None
    )
    hold = None
    oracle_hold = None
    if holdout_memory is not None and np.asarray(holdout_memory).size:
        hold = _score_frame_components(
            M_gain, A_block, V_opt_q, rows_ref, state_prev, score_variant, holdout_memory
        )
        oracle_hold = (
            _score_frame_components(M_gain, A_block, Q_oracle_q, rows_ref, state_prev, score_variant, holdout_memory)
            if Q_oracle_q.size else None
        )

    prev_cos = np.zeros(0, dtype=float)
    tail_prev_cos = np.zeros(0, dtype=float)
    prev_curr = None
    if prev_V_opt is not None and np.asarray(prev_V_opt).size:
        prev_q = orthonormalize_columns(np.asarray(prev_V_opt, dtype=diag_dtype)[:, : int(rank)], dtype=diag_dtype)
        prev_cos = subspace_principal_cosines(prev_q, V_opt_q)
        prev_curr = _score_frame_components(
            M_gain, A_block, prev_q, rows_ref, state_prev, score_variant, old_row_memory
        )
        if V_sig.size:
            prev_tail = prev_q - V_sig @ (V_sig.T @ prev_q)
            curr_tail = V_opt_q - V_sig @ (V_sig.T @ V_opt_q)
            prev_tail_q = orthonormalize_columns(prev_tail, dtype=diag_dtype)
            curr_tail_q = orthonormalize_columns(curr_tail, dtype=diag_dtype)
            tail_prev_cos = subspace_principal_cosines(prev_tail_q, curr_tail_q)

    train_rows = 0 if old_row_memory is None else int(np.asarray(old_row_memory).shape[0])
    hold_rows = 0 if holdout_memory is None else int(np.asarray(holdout_memory).shape[0])
    print(
        "consecutive_tail_diag: "
        f"block={block_idx} rows={start0 + 1}:{end0} "
        f"train_old_rows={train_rows} holdout_old_rows={hold_rows} "
        f"train_old_idx={np.array2string(np.asarray(old_row_memory_idx, dtype=int) + 1, max_line_width=120) if old_row_memory_idx is not None else '[]'}"
    )
    print(
        "  opt_vs_prev_cos: "
        f"{fmt_row(prev_cos, precision=6)} "
        f"tail_vs_prev_cos={fmt_row(tail_prev_cos, precision=6)}"
    )
    print(
        "  opt_vs_oracle_cos: "
        f"{fmt_row(opt_oracle_cos, precision=6)} "
        f"sig_mass={_fmt_scalar(sig_mass)} tail_mass={_fmt_scalar(tail_mass)}"
    )
    print(
        "  train_opt: "
        f"score_sum={_fmt_scalar(train['score_sum'])} "
        f"relH={fmt_row(train['pooled_rel_H'], precision=6)} "
        f"gain2={fmt_row(train['gain2'], precision=6)} phi={fmt_row(train['phi'], precision=6)}"
    )
    if oracle_train is not None:
        print(
            "  train_oracle_qr: "
            f"score_sum={_fmt_scalar(oracle_train['score_sum'])} "
            f"relH={fmt_row(oracle_train['pooled_rel_H'], precision=6)}"
        )
    if prev_curr is not None:
        print(
            "  train_prev_opt_on_curr: "
            f"score_sum={_fmt_scalar(prev_curr['score_sum'])} "
            f"relH={fmt_row(prev_curr['pooled_rel_H'], precision=6)}"
        )
    if hold is None:
        print("  holdout_opt: unavailable")
    else:
        print(
            "  holdout_opt: "
            f"score_sum={_fmt_scalar(hold['score_sum'])} "
            f"relH={fmt_row(hold['pooled_rel_H'], precision=6)} "
            f"gain2={fmt_row(hold['gain2'], precision=6)} phi={fmt_row(hold['phi'], precision=6)}"
        )
        if oracle_hold is not None:
            print(
                "  holdout_oracle_qr: "
                f"score_sum={_fmt_scalar(oracle_hold['score_sum'])} "
                f"relH={fmt_row(oracle_hold['pooled_rel_H'], precision=6)}"
            )


def dump_oracle_old_row_responses(
    A, M_gain, V_exact, rank, old_row_memory, old_row_memory_idx=None, label="",
    oracle_projection_row_samples=None,
):
    if V_exact is None or np.asarray(V_exact).size == 0:
        print("oracle_old_row_response_dump: skipped no V_exact")
        return
    if old_row_memory is None or np.asarray(old_row_memory).size == 0:
        print("oracle_old_row_response_dump: skipped no old_row_memory")
        return

    dump_dtype = np.result_type(np.asarray(M_gain).dtype, np.asarray(V_exact).dtype, np.float64)
    _, Q_row = projected_true_span_oracle(
        np.asarray(M_gain, dtype=dump_dtype),
        np.asarray(V_exact, dtype=dump_dtype)[:, : int(rank)],
        int(rank),
        dtype=dump_dtype,
        row_samples=oracle_projection_row_samples,
    )
    R_old = np.asarray(old_row_memory, dtype=dump_dtype)
    print(f"oracle_old_row_response_dump{label}: rows={R_old.shape[0]}")
    if old_row_memory_idx is not None:
        print(
            "old_row_memory_indices_1based: "
            f"{np.array2string(np.asarray(old_row_memory_idx, dtype=int) + 1, max_line_width=120)}"
        )

    for j in range(min(int(rank), np.asarray(V_exact).shape[1])):
        v_proj = project_onto_span(np.asarray(V_exact, dtype=dump_dtype)[:, j], Q_row).reshape(-1)
        v_norm = float(np.linalg.norm(v_proj))
        if v_norm <= 1e-30:
            print(f"v{j + 1}_proj_old_row_response: skipped zero projection")
            continue
        v_proj = np.ascontiguousarray(v_proj / v_norm, dtype=dump_dtype)
        response = np.ascontiguousarray(R_old @ v_proj, dtype=dump_dtype)
        H, rel_H, y2_sq, y4_4 = response_entropy_stats(response)
        print(f"v{j + 1}_proj_old_row_response:")
        print(np.array2string(response, precision=8, suppress_small=False, max_line_width=120))
        print(
            f"v{j + 1}_proj_old_row_response_stats: "
            f"h={H:.12f} rel_h={rel_H:.12f} norm2_sq={y2_sq:.12f} norm4_4={y4_4:.12f}"
        )
        # breakpoint()


def run(args):
    np.random.seed(args.seed)
    t0 = time.time()

    if args.mat_input:
        A, V_exact, svec, sigma1 = load_matlab_cex_input(args.mat_input)
        source_desc = args.mat_input
    else:
        A, V_exact, svec, sigma1 = generate_matrix_input(
            matrix=args.matrix,
            n=args.n,
            preset=args.preset,
            seed=args.seed,
            r_sig=args.r_sig,
            alpha_sig=args.alpha_sig,
            alpha_tail=args.alpha_tail,
            tail_scale=args.tail_scale,
            sigma1=args.sigma1,
            v_type=args.v_type,
            shuffle_rows=args.shuffle_rows,
            row_shuffle_seed=args.row_shuffle_seed,
        )
        if args.matrix == "static-cex":
            source_desc = (
                "generated structured cex "
                f"(n={args.n}, r_sig={args.r_sig}, sigma1={args.sigma1}, "
                f"alpha_sig={args.alpha_sig}, alpha_tail={args.alpha_tail}, "
                f"tail_scale={args.tail_scale}, v_type={args.v_type})"
            )
        else:
            source_desc = f"generated {args.matrix} (n={args.n}, preset={args.preset}, seed={args.seed})"
        if args.shuffle_rows:
            source_desc += f", row_shuffled(seed={args.seed if args.row_shuffle_seed is None else args.row_shuffle_seed})"
    A = np.asarray(A, dtype=np.float64)
    if args.normalize_by_sigma:
        A = A / sigma1
    n = A.shape[0]
    r = args.rank
    win = args.win

    state = None
    V_r = None
    S_r = None
    old_row_memory = None
    old_row_memory_idx = None
    prev_V_opt_diag = None

    print(f"Input: {source_desc}: A={A.shape}, sigma1={sigma1:.12g}, normalize_by_sigma={args.normalize_by_sigma}")
    work_dtype = np.float64 if args.dtype == "float64" else np.float32
    print(f"Mode: {args.mode}")
    if args.mode in {"restricted", "combined", "l4l2", "subsetmass"}:
        print(
            "Restricted optimizer params: "
            f"preset={args.preset}, q0={args.q0}, qmax={args.qmax}, krylov_depth={args.krylov_depth}, "
            f"residual_tol={args.residual_tol}, expansion_maxit={args.expansion_maxit}, "
            f"num_restarts={args.num_restarts}, maxit={args.maxit}, carry={args.carry}, "
            f"reduced_optimizer={args.reduced_optimizer}, dtype={np.dtype(work_dtype)}, "
            f"basis_selection={args.basis_selection}, "
            f"joint_warm_start_greedy={args.joint_warm_start_greedy}, "
            f"joint_warm_start_oracle={args.joint_warm_start_oracle}, "
            f"joint_warm_start_rotations={args.joint_warm_start_rotations}, "
            f"joint_warm_start_rotation_angle={args.joint_warm_start_rotation_angle}, "
            f"joint_warm_start_perturbations={args.joint_warm_start_perturbations}, "
            f"joint_warm_start_perturb_scale={args.joint_warm_start_perturb_scale}, "
            f"joint_default_svd_start={args.joint_default_svd_start}, "
            f"joint_oversample={args.joint_oversample}, "
            f"joint_oversample_rotate={args.joint_oversample_rotate}, "
            f"joint_solver={args.joint_solver}, "
            f"row_concentration_lambda={args.row_concentration_lambda}, "
            f"row_leverage_lambda={args.row_leverage_lambda}, "
            f"row_leverage_mode={args.row_leverage_mode}, "
            f"row_leverage_rank={args.row_leverage_rank}, "
            f"score_variant={args.score_variant}, "
            f"rownorm_seed_first_block={getattr(args, 'rownorm_seed_first_block', False)}, "
            f"rownorm_seed_all_blocks={getattr(args, 'rownorm_seed_all_blocks', False)}, "
            f"old_memory_size={args.old_memory_size}, "
            f"debug_mode={args.debug_mode}, "
            f"oracle_candidate_check={args.oracle_candidate_check}, "
            f"oracle_sketch_all_seen_rows={args.oracle_sketch_all_seen_rows}, "
            f"dump_score_components={args.dump_score_components}, "
            f"dump_consecutive_tail_diagnostics={args.dump_consecutive_tail_diagnostics}, "
            f"old_memory_holdout_size={args.old_memory_holdout_size}, "
            f"dump_oracle_old_row_responses={args.dump_oracle_old_row_responses}, "
            f"dump_oracle_old_row_response_block={args.dump_oracle_old_row_response_block}, "
            f"expansion_direction={args.expansion_direction}, "
            f"reuse_line_search_grad={args.reuse_line_search_grad}, "
            f"expansion_warm_start={args.expansion_warm_start}, "
            f"post_expansion_maxit={args.post_expansion_maxit}"
        )

    for block_idx, start0 in enumerate(range(0, n, win), start=1):
        end0 = min(start0 + win, n)
        A_block = A[start0:end0, :]
        A_block_work = A_block.astype(work_dtype, copy=False)

        if args.mode in {"isvd", "fd"}:
            print(f"\n===== block rows {start0 + 1}:{end0} ({args.mode}) =====")
            V_r, S_r, s_new = svd_sketch_update(A_block, V_r, S_r, r, args.mode)
            print(f"rows {start0 + 1}:{end0}")
            print(f"s: {fmt_row(s_new)}")
            continue

        if state is None:
            M_gain = A_block_work
            if getattr(args, "rownorm_seed_first_block", False) or getattr(args, "rownorm_seed_all_blocks", False):
                V_init = np.asarray(row_norm_seed(A_block_work, r), dtype=work_dtype)
            else:
                V_init = None
            rows_seen = A_block.shape[0]
            print(f"\n===== block rows {start0 + 1}:{end0} (initial restricted score) =====")
        else:
            B_top = (state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T)
            M_gain = np.vstack([B_top, A_block_work]).astype(work_dtype, copy=False)
            if getattr(args, "rownorm_seed_all_blocks", False):
                V_init = np.asarray(row_norm_seed(A_block_work, r), dtype=work_dtype)
            else:
                V_init = state["V"].astype(work_dtype, copy=False)
            rows_seen = state["rows_seen"] + A_block.shape[0]
            print(f"\n===== block rows {start0 + 1}:{end0} (streaming restricted score) =====")

        oracle_projection_row_samples = None
        score_component_old_rows = old_row_memory
        if args.oracle_sketch_all_seen_rows:
            oracle_projection_row_samples = A[:end0, :].astype(work_dtype, copy=False)
            score_component_old_rows = A[:start0, :].astype(work_dtype, copy=False) if start0 > 0 else None

        V_score, s_score, H_score, score_score, diag = entropy_iter_basis_forget(
            M_gain=M_gain,
            active_r=r,
            rows_ref=n,
            V_init=V_init,
            q0=args.q0,
            qmax=args.qmax,
            krylov_depth=args.krylov_depth,
            residual_tol=args.residual_tol,
            expansion_maxit=args.expansion_maxit,
            num_restarts=args.num_restarts,
            maxit=args.maxit,
            tol=args.tol,
            rng=np.random.default_rng(args.seed),
            verbose=args.verbose,
            state_prev=state,
            A_block=A_block_work,
            rows_total=rows_seen,
            reduced_optimizer=args.reduced_optimizer,
            basis_selection=args.basis_selection,
            joint_warm_start_greedy=args.joint_warm_start_greedy,
            joint_warm_start_oracle=args.joint_warm_start_oracle,
            oracle_warm_start_target=V_exact,
            joint_warm_start_rotations=args.joint_warm_start_rotations,
            joint_warm_start_rotation_angle=args.joint_warm_start_rotation_angle,
            joint_warm_start_perturbations=args.joint_warm_start_perturbations,
            joint_warm_start_perturb_scale=args.joint_warm_start_perturb_scale,
            joint_default_svd_start=args.joint_default_svd_start,
            joint_oversample=args.joint_oversample,
            joint_oversample_rotate=args.joint_oversample_rotate,
            joint_solver=args.joint_solver,
            work_dtype=work_dtype,
            expansion_direction=args.expansion_direction,
            reuse_line_search_grad=args.reuse_line_search_grad,
            expansion_warm_start=args.expansion_warm_start,
            post_expansion_maxit=args.post_expansion_maxit,
            row_concentration_lambda=args.row_concentration_lambda,
            row_leverage_lambda=args.row_leverage_lambda,
            row_leverage_mode=args.row_leverage_mode,
            row_leverage_rank=args.row_leverage_rank,
            score_variant=args.score_variant,
            old_row_memory=old_row_memory,
            oracle_projection_row_samples=oracle_projection_row_samples,
            combined_rank=getattr(args, "combined_rank", None),
        )

        oracle_candidate_status = None
        if (
            args.oracle_candidate_check
            and args.score_variant in {"oldcorrected", "combined"}
            and args.row_concentration_lambda == 0.0
            and V_exact is not None
        ):
            oracle_candidate = oracle_projection_candidate(
                M_gain=M_gain,
                A_block=A_block_work,
                V_exact=V_exact,
                rank=r,
                rows_ref=n,
                state_prev=state,
                score_variant=args.score_variant,
                old_row_memory=old_row_memory,
                row_concentration_lambda=0.0,
                oracle_projection_row_samples=oracle_projection_row_samples,
            )
            if oracle_candidate is not None:
                optimizer_sum = float(np.sum(score_score[:r]))
                candidate_sum = float(oracle_candidate["score_sum"])
                accepted = candidate_sum > optimizer_sum + 1e-10
                oracle_candidate_status = {
                    "accepted": bool(accepted),
                    "optimizer_sum": optimizer_sum,
                    "candidate_sum": candidate_sum,
                }
                if accepted:
                    V_score[:, :r] = oracle_candidate["V"][:, :r]
                    score_score[:r] = oracle_candidate["score"][:r]
                    H_score[:r] = oracle_candidate["H"][:r]
                    s_score[:r] = oracle_candidate["s"][:r]
                    diag["regularized_score"][:r] = oracle_candidate["score"][:r]
                    diag["regularized_score_sum"] = candidate_sum
            diag["oracle_candidate_check"] = oracle_candidate_status

        oracle_diag = oracle_projection_diagnostics(
            M_gain=M_gain,
            A_block=A_block_work,
            V_exact=V_exact,
            V_opt=V_score[:, :r],
            rank=r,
            rows_ref=n,
            state_prev=state,
            score_variant=args.score_variant,
            old_row_memory=old_row_memory,
            row_concentration_lambda=0.0,
            oracle_projection_row_samples=oracle_projection_row_samples,
        )

        if args.dump_consecutive_tail_diagnostics and args.score_variant == "combined" and V_exact is not None:
            holdout_size = args.old_memory_holdout_size
            if holdout_size is None:
                holdout_size = args.old_memory_size
            holdout_memory, holdout_idx = select_old_row_holdout(
                A[:start0, :].astype(work_dtype, copy=False),
                holdout_size,
                np.random.default_rng(args.seed + 104729 + start0),
                exclude_indices=old_row_memory_idx,
                return_indices=True,
            )
            diag_dtype = np.result_type(np.asarray(M_gain).dtype, np.asarray(V_exact).dtype, np.float64)
            Q_oracle_diag, _ = projected_true_span_oracle(
                np.asarray(M_gain, dtype=diag_dtype),
                np.asarray(V_exact, dtype=diag_dtype)[:, : int(r)],
                int(r),
                dtype=diag_dtype,
                row_samples=oracle_projection_row_samples,
            )
            dump_consecutive_tail_diagnostics(
                block_idx=block_idx,
                start0=start0,
                end0=end0,
                M_gain=np.asarray(M_gain, dtype=diag_dtype),
                A_block=np.asarray(A_block_work, dtype=diag_dtype),
                V_exact=np.asarray(V_exact, dtype=diag_dtype),
                V_opt=np.asarray(V_score[:, :r], dtype=diag_dtype),
                Q_oracle=Q_oracle_diag,
                rank=r,
                rows_ref=n,
                state_prev=state,
                score_variant=args.score_variant,
                old_row_memory=None if old_row_memory is None else np.asarray(old_row_memory, dtype=diag_dtype),
                old_row_memory_idx=old_row_memory_idx,
                holdout_memory=None if holdout_memory is None else np.asarray(holdout_memory, dtype=diag_dtype),
                prev_V_opt=prev_V_opt_diag,
            )

        if args.dump_score_components and args.score_variant in {"oldcorrected", "combined", "l4l2"} and V_exact is not None:
            comp_dtype = np.result_type(np.asarray(M_gain).dtype, np.asarray(V_exact).dtype, np.float64)
            _, Q_row_comp = projected_true_span_oracle(
                np.asarray(M_gain, dtype=comp_dtype),
                np.asarray(V_exact, dtype=comp_dtype)[:, : int(r)],
                int(r),
                dtype=comp_dtype,
                row_samples=oracle_projection_row_samples,
            )
            comp_vectors = []
            for j in range(min(int(r), np.asarray(V_exact).shape[1])):
                v_proj = project_onto_span(np.asarray(V_exact, dtype=comp_dtype)[:, j], Q_row_comp).reshape(-1)
                v_norm = float(np.linalg.norm(v_proj))
                if v_norm > 1e-30:
                    comp_vectors.append((f"oracle_raw_v{j + 1}", np.ascontiguousarray(v_proj / v_norm, dtype=comp_dtype)))
            for j in range(min(int(r), V_score.shape[1])):
                comp_vectors.append((f"optimizer_v{j + 1}", np.ascontiguousarray(V_score[:, j], dtype=comp_dtype)))
            print_score_component_dump(
                f"{args.score_variant}_score_components",
                args.score_variant,
                comp_vectors,
                np.asarray(M_gain, dtype=comp_dtype),
                np.asarray(A_block_work, dtype=comp_dtype),
                n,
                state,
                None if score_component_old_rows is None else np.asarray(score_component_old_rows, dtype=comp_dtype),
            )

        if (
            args.dump_oracle_old_row_responses
            and (args.dump_oracle_old_row_response_block == 0 or args.dump_oracle_old_row_response_block == block_idx)
        ):
            dump_oracle_old_row_responses(
                A=A,
                M_gain=M_gain,
                V_exact=V_exact,
                rank=r,
                old_row_memory=old_row_memory,
                old_row_memory_idx=old_row_memory_idx,
                label=f" block={block_idx} rows={start0 + 1}:{end0}",
                oracle_projection_row_samples=oracle_projection_row_samples,
            )
            # breakpoint()

        if args.carry == "left":
            _, s_new, Vt_new, _ = left_projected_operator_svd_factors(V_score.T, M_gain)
            V_r = Vt_new.T
        else:
            V_r, s_new = projected_subspace_svd(M_gain.astype(np.float64), V_score.astype(np.float64))
            s_new = s_new.astype(np.float32, copy=False)
            V_r = V_r.astype(np.float32, copy=False)
        S_r = np.diag(s_new)
        state = {
            "V": V_r,
            "s": s_new,
            "s2": s_new ** 2,
            "H": np.asarray(H_score[: len(s_new)], dtype=np.float32),
            "score": np.asarray(score_score[: len(s_new)], dtype=np.float32),
            "rows_seen": rows_seen,
            "diag": diag,
            "old_row_memory_rows": 0 if old_row_memory is None else int(old_row_memory.shape[0]),
        }
        prev_V_opt_diag = np.ascontiguousarray(np.asarray(V_score[:, :r], dtype=np.float64))

        old_row_memory, old_row_memory_idx = select_old_row_memory(
            A[:end0, :].astype(work_dtype, copy=False),
            V_r.astype(work_dtype, copy=False),
            args.old_memory_size,
            np.random.default_rng(args.seed + end0),
            return_indices=True,
        )
        state["old_row_memory_rows"] = 0 if old_row_memory is None else int(old_row_memory.shape[0])

        print(f"rows {start0 + 1}:{end0}")
        print(f"s: {fmt_row(s_new)}")
        print(f"H: {fmt_row(H_score)}")
        print(f"scores: {fmt_row(score_score)}")
        if oracle_candidate_status is not None:
            verdict = "accepted" if oracle_candidate_status["accepted"] else "rejected"
            print(
                "oracle_candidate_check: "
                f"{verdict} optimizer_sum={oracle_candidate_status['optimizer_sum']:.12f} "
                f"candidate_sum={oracle_candidate_status['candidate_sum']:.12f}"
            )
        if args.row_concentration_lambda != 0.0:
            print(f"regularized_scores: {fmt_row(diag['regularized_score'][:r])}")
        if oracle_diag is not None:
            print(
                "oracle_raw_projection_scores: "
                f"{fmt_row(oracle_diag['raw_oracle_scores'])} "
                f"sum={oracle_diag['raw_oracle_score_sum']:.12f}"
            )
            print(
                "oracle_qr_projection_scores: "
                f"{fmt_row(oracle_diag['qr_oracle_scores'])} "
                f"sum={oracle_diag['qr_oracle_score_sum']:.12f}"
            )
            print(f"optimizer_score_sum: {float(np.sum(score_score[:r])):.12f}")
            print(
                "vecnorm(V_score V_score' [v1_proj v2_proj], 2): "
                f"{fmt_row(oracle_diag['opt_proj_norms'], precision=6)}"
            )
            print(
                "principal_cosines(V_score, Q_oracle): "
                f"{fmt_row(oracle_diag['opt_vs_qoracle_cosines'], precision=6)}"
            )
            if not np.isnan(oracle_diag["raw_oracle_overlap"]):
                print(f"raw_oracle_projection_overlap: {oracle_diag['raw_oracle_overlap']:.12f}")
        print(f"subspace_dims: {diag['subspace_dims'][:r].tolist()}")
        print(f"grad_perp_ratio: {fmt_row(diag['grad_perp_ratio'][:r], precision=6)}")
        if args.score_variant in {"oldcorrected", "combined", "l4l2"}:
            print(f"old_row_memory_rows: {state['old_row_memory_rows']}")

    align = np.linalg.norm((V_r @ V_r.T) @ V_exact[:, :1], "fro")
    top_sval_est = S_r[0, 0]
    rel_err_sval = abs(top_sval_est - sigma1) / sigma1
    elapsed = time.time() - t0
    print("sigma1    mean_align    mean_relerr_sval    elapsed")
    print(f"{sigma1:.3f}      {align:.6f}           {rel_err_sval:.8f}          {elapsed:.3f}")

    method_label = args.mode
    if args.mode in {"restricted", "combined", "l4l2", "subsetmass"}:
        method_label = f"{args.mode}-{args.basis_selection}"
        if args.basis_selection == "joint" and args.joint_warm_start_greedy:
            method_label += "-greedywarm"
            if args.joint_warm_start_rotations:
                method_label += f"-rot{args.joint_warm_start_rotations}"
            if args.joint_warm_start_perturbations:
                method_label += f"-pert{args.joint_warm_start_perturbations}"
        if args.basis_selection == "joint" and args.joint_warm_start_oracle:
            method_label += "-oraclewarm"
        if args.basis_selection == "joint" and not args.joint_default_svd_start:
            method_label += "-no-svdstart"
        if args.basis_selection == "joint" and args.joint_oversample:
            method_label += f"-over{args.joint_oversample}-{args.joint_oversample_rotate}"
        if args.basis_selection == "joint" and args.joint_solver != "riemannian":
            method_label += f"-{args.joint_solver}"
        if args.row_concentration_lambda != 0.0:
            method_label += f"-rowconc{args.row_concentration_lambda:g}"
        if args.row_leverage_lambda != 0.0:
            method_label += f"-rowlev{args.row_leverage_mode}{args.row_leverage_lambda:g}"
        if args.score_variant != "legacy" and not (
            args.mode in {"combined", "l4l2", "subsetmass"} and args.score_variant == args.mode
        ):
            method_label += f"-{args.score_variant}"
        if args.oracle_candidate_check and args.score_variant in {"oldcorrected", "combined"}:
            method_label += "-oraclecheck"

    result = {
        "matrix": args.matrix if not args.mat_input else os.path.basename(args.mat_input),
        "method": method_label,
        "mean_align": float(align),
        "mean_relerr_sval": float(rel_err_sval),
        "elapsed": float(elapsed),
    }
    if args.benchmark_output:
        mode = "a" if args.benchmark_append else "w"
        with open(args.benchmark_output, mode, encoding="utf-8") as f:
            if not args.benchmark_append:
                f.write("matrix\tmethod\tmean_align\tmean_relerr_sval\telapsed\n")
            f.write(
                f"{result['matrix']}\t{result['method']}\t"
                f"{result['mean_align']:.6f}\t{result['mean_relerr_sval']:.8f}\t{result['elapsed']:.3f}\n"
            )
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the restricted-space entropy-forget optimizer on the cex MATLAB input.",
        epilog=(
            "Default preset is the tested fast/balanced setting condensed from main.py: "
            "q0=5, qmax=200, residual expansion, restarts=2, maxit=120, "
            "float32, Armijo gradient reuse, expansion warm-start, post-expansion maxit=60. "
            "Use --preset cex-replicate for the high-budget float64 reproduction setting."
        ),
    )
    parser.add_argument(
        "--mat-input",
        help="Optional MATLAB .mat input file. By default the script generates the structured cex matrix internally.",
    )
    parser.add_argument(
        "--matrix",
        choices=(
            "alternative-data-signals",
            "static-cex",
            "diffuse-diffuse",
            "mixed-tail-soft",
            "mixed-tail-balanced",
            "mixed-tail-sharp",
            "residual-spiky-shocks",
            "crowded-strategy",
            "execution-cost-slippage",
            "etf-basket-basis",
            "futures-term-structure",
            "intraday-liquidity-shape",
            "macro-factor-panel",
            "rates-cross-currency",
            "options-vol-surface",
            "realized-vol-corr",
            "risk-residual-panel",
            "stat-arb-spreads",
        ),
        default="static-cex",
        help="Generated matrix family to use when --mat-input is not provided.",
    )
    parser.add_argument("--n", type=int, default=1024, help="Generated matrix dimension; must be a power of two.")
    parser.add_argument("--r-sig", type=int, default=2, help="Generated signal-block rank.")
    parser.add_argument("--alpha-sig", type=float, default=0.003)
    parser.add_argument("--alpha-tail", type=float, default=0.0145)
    parser.add_argument("--tail-scale", type=float, default=0.99)
    parser.add_argument("--sigma1", type=float, default=0.991)
    parser.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    parser.add_argument(
        "--mode",
        choices=("restricted", "combined", "l4l2", "subsetmass", "isvd", "fd", "iSVD", "FD"),
        default="restricted",
        help="Streaming method to run: restricted optimizer, combined-score restricted shorthand, incremental SVD, or Frequent Directions.",
    )
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--win", type=int, default=100)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="fast")
    parser.add_argument(
        "--cex-replicate",
        action="store_true",
        help="Shortcut for --preset cex-replicate.",
    )
    parser.add_argument("--q0", type=int)
    parser.add_argument("--qmax", type=int)
    parser.add_argument("--krylov-depth", type=int)
    parser.add_argument("--residual-tol", type=float)
    parser.add_argument("--expansion-maxit", type=int)
    parser.add_argument("--num-restarts", type=int)
    parser.add_argument("--maxit", type=int)
    parser.add_argument("--tol", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shuffle-rows",
        action="store_true",
        help="Apply a deterministic full row permutation after matrix generation.",
    )
    parser.add_argument(
        "--row-shuffle-seed",
        type=int,
        help="Seed for --shuffle-rows. Defaults to --seed.",
    )
    parser.add_argument("--normalize-by-sigma", action="store_true")
    parser.add_argument("--carry", choices=("left", "right"))
    parser.add_argument("--reduced-optimizer", choices=("legacy", "cex"))
    parser.add_argument(
        "--basis-selection",
        choices=("greedy", "joint"),
        help="Rank-r direction selection: current sequential greedy solve or coupled joint Stiefel solve.",
    )
    parser.add_argument(
        "--combined-rank",
        type=int,
        default=None,
        help=(
            "Hybrid split: first combined_rank directions are chosen by combined-score greedy; "
            "the remaining rank-combined_rank directions are top right singular vectors of M_gain "
            "restricted to the orthogonal complement of the greedy part. Requires --basis-selection greedy."
        ),
    )
    parser.add_argument(
        "--joint-warm-start-greedy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="When --basis-selection joint is used, add the sequential greedy frame as one initial Stiefel seed.",
    )
    parser.add_argument(
        "--joint-warm-start-oracle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add orth(P_row(M_gain) V_exact[:, :rank]) as an oracle warm start. "
            "For --basis-selection joint this is a Stiefel seed; for greedy it adds per-direction restart seeds."
        ),
    )
    parser.add_argument(
        "--joint-warm-start-rotations",
        type=int,
        help="Number of right-rotated greedy warm-start Stiefel seeds to add.",
    )
    parser.add_argument(
        "--joint-warm-start-rotation-angle",
        type=float,
        help="Maximum rotation angle in radians for right-rotated greedy warm-start seeds.",
    )
    parser.add_argument(
        "--joint-warm-start-perturbations",
        type=int,
        help="Number of random tangent perturbations of the greedy warm-start Stiefel seed to add.",
    )
    parser.add_argument(
        "--joint-warm-start-perturb-scale",
        type=float,
        help="Frobenius-norm scale for random tangent perturbations of the greedy warm-start seed.",
    )
    parser.add_argument(
        "--joint-default-svd-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For joint Stiefel selection, include the leading reduced-basis/SVD-eye frame as a default start.",
    )
    parser.add_argument(
        "--joint-oversample",
        type=int,
        help="Optimize rank + this many joint Stiefel columns before compressing back to --rank.",
    )
    parser.add_argument(
        "--joint-oversample-rotate",
        choices=("none", "svd"),
        help="How to rotate/compress an oversampled joint frame back to --rank.",
    )
    parser.add_argument(
        "--joint-solver",
        choices=("riemannian", "slsqp"),
        help="Joint Stiefel solver: projected Riemannian ascent or SLSQP quasi-Newton refinement.",
    )
    parser.add_argument(
        "--row-concentration-lambda",
        type=float,
        default=0.0,
        help="Penalty weight for sum_i (A_block v)_i^4 / ||A_block v||_2^4. Default 0 preserves old behavior.",
    )
    parser.add_argument(
        "--row-leverage-lambda",
        type=float,
        default=0.0,
        help="Penalty weight for sum_i leverage_i p_i, where p_i is current-window response energy share.",
    )
    parser.add_argument(
        "--row-leverage-mode",
        choices=("none", "row-norm", "top-svd"),
        default="none",
        help="Row leverage weights for --row-leverage-lambda. Weights are normalized to mean 1.",
    )
    parser.add_argument(
        "--row-leverage-rank",
        type=int,
        default=2,
        help="Rank used by --row-leverage-mode top-svd.",
    )
    parser.add_argument(
        "--score-variant",
        choices=("legacy", "oldcorrected", "combined", "l4l2", "subsetmass"),
        default="legacy",
        help=(
            "Score formula for restricted mode. legacy preserves the current score; "
            "oldcorrected uses separate old/new entropy corrections; combined uses the pooled "
            "row-memory/current-window entropy correction from reports/approximation/new_approx_combined.txt; "
            "l4l2 uses the pooled raw reciprocal L4/L2 correction without row-entropy normalization; "
            "subsetmass uses the Cauchy-Schwarz subset-mass upper-bound score from "
            "reports/approximation/new_approx_subsetmass.txt."
        ),
    )
    parser.add_argument(
        "--rownorm-seed-first-block",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Initialize the first restricted block from the top right singular vectors of row-L2-normalized A_block.",
    )
    parser.add_argument(
        "--rownorm-seed-all-blocks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Initialize every restricted block from the top right singular vectors of row-L2-normalized current A_block.",
    )
    parser.add_argument(
        "--debug-mode",
        choices=("off", "combined", "summary"),
        default="off",
        help=(
            "Enable a bundled diagnostic workflow from summary/summary_debug.txt. "
            "'combined' selects combined mode unless explicitly overridden. "
            "'combined' and 'summary' both enable oracle candidate checks, score-component "
            "dumps, and oracle old-row response dumps for every block."
        ),
    )
    parser.add_argument(
        "--old-memory-size",
        type=int,
        default=None,
        help=(
            "Rows retained as old row memory for oldcorrected/combined score variants. "
            "Defaults to --win. Uses half uniform sample and half high-response rows."
        ),
    )
    parser.add_argument(
        "--oracle-candidate-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For diagnostic runs with known V_exact, compare orth(P_row(M_gain) V_exact[:, :rank]) against the optimizer before compression.",
    )
    parser.add_argument(
        "--oracle-sketch-all-seen-rows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use all rows seen through the current block as the optional oracle projection row sketch. "
            "By default oracle projection uses only span(M_gain)."
        ),
    )
    parser.add_argument(
        "--dump-oldcorrected-score-components",
        dest="dump_score_components",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dump-score-components",
        dest="dump_score_components",
        action="store_true",
        help=(
            "Print score-component diagnostics for oracle-projected top right singular vectors "
            "and optimizer vectors. Supports oldcorrected, combined, and l4l2 score variants."
        ),
    )
    parser.add_argument(
        "--dump-oracle-old-row-responses",
        action="store_true",
        help="Print old_row_memory @ normalized P_row(M_gain) V_exact[:, j] for oracle-projected directions.",
    )
    parser.add_argument(
        "--dump-oracle-old-row-response-block",
        type=int,
        default=3,
        help="1-based block index for --dump-oracle-old-row-responses; use 0 to dump every block.",
    )
    parser.add_argument(
        "--dump-consecutive-tail-diagnostics",
        action="store_true",
        help=(
            "Print consecutive selected-subspace, signal/tail-mass, cross-score, and "
            "train-vs-holdout row-entropy diagnostics for combined-score runs."
        ),
    )
    parser.add_argument(
        "--old-memory-holdout-size",
        type=int,
        default=None,
        help=(
            "Old raw rows reserved for holdout scoring in --dump-consecutive-tail-diagnostics. "
            "Defaults to --old-memory-size."
        ),
    )
    parser.add_argument("--dtype", choices=("float32", "float64"))
    parser.add_argument("--expansion-direction", choices=("krylov_v", "residual"))
    parser.add_argument(
        "--reuse-line-search-grad",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reuse the accepted Armijo trial score/gradient instead of recomputing it.",
    )
    parser.add_argument(
        "--expansion-warm-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the lifted previous reduced optimum as the first seed after subspace expansion.",
    )
    parser.add_argument(
        "--post-expansion-maxit",
        type=int,
        help="Iteration cap for solve batches after a subspace expansion.",
    )
    parser.add_argument("--benchmark-output", help="Optional path for a tab-separated benchmark row.")
    parser.add_argument(
        "--benchmark-append",
        action="store_true",
        help="Append to --benchmark-output instead of overwriting it.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.mode == "iSVD":
        args.mode = "isvd"
    elif args.mode == "FD":
        args.mode = "fd"
    elif args.mode == "combined" and "--score-variant" not in sys.argv:
        args.score_variant = "combined"
    elif args.mode == "l4l2" and "--score-variant" not in sys.argv:
        args.score_variant = "l4l2"
    elif args.mode == "subsetmass" and "--score-variant" not in sys.argv:
        args.score_variant = "subsetmass"

    if args.cex_replicate:
        args.preset = "cex-replicate"

    if args.debug_mode == "combined":
        if "--mode" not in sys.argv:
            args.mode = "combined"
        if "--score-variant" not in sys.argv:
            args.score_variant = "combined"

    if args.debug_mode in {"combined", "summary"}:
        if "--oracle-candidate-check" not in sys.argv and "--no-oracle-candidate-check" not in sys.argv:
            args.oracle_candidate_check = True
        args.dump_score_components = True
        args.dump_oracle_old_row_responses = True
        if "--dump-oracle-old-row-response-block" not in sys.argv:
            args.dump_oracle_old_row_response_block = 0

    preset_values = PRESETS[args.preset]
    for name, value in preset_values.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.old_memory_size is None:
        args.old_memory_size = args.win

    return args


if __name__ == "__main__":
    run(parse_args())
