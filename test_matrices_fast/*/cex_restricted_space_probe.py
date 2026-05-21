import argparse
import os
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


def parse_continuation_schedule(schedule, default=(0.0, 0.25, 0.5, 0.75, 1.0)):
    if schedule is None:
        vals = list(default)
    elif isinstance(schedule, str):
        vals = [float(tok.strip()) for tok in schedule.split(",") if tok.strip()]
    else:
        vals = [float(v) for v in schedule]
    if not vals:
        vals = [1.0]
    vals = [min(max(v, 0.0), 1.0) for v in vals]
    vals = sorted(vals)
    if vals[-1] < 1.0:
        vals.append(1.0)
    return tuple(vals)


def make_reduced_warm_start_seeds(Vbasis, Qz, k_idx, z_warm=None, prior_basis=None,
                                  num_perturb=0, perturb_scale=1e-2, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    q = Vbasis.shape[1]
    starts = []

    if z_warm is not None:
        append_unique_reduced_seed(starts, retract_reduced(z_warm, Qz))

    if prior_basis is not None and prior_basis.shape[1] > k_idx:
        z_prior = np.ascontiguousarray(Vbasis.T @ prior_basis[:, k_idx], dtype=Vbasis.dtype)
        z_prior = retract_reduced(z_prior, Qz)
        append_unique_reduced_seed(starts, z_prior)
        if z_prior is not None:
            for _ in range(max(0, int(num_perturb))):
                noise = np.ascontiguousarray(rng.standard_normal(q), dtype=Vbasis.dtype)
                append_unique_reduced_seed(
                    starts,
                    retract_reduced(z_prior + np.asarray(perturb_scale, dtype=Vbasis.dtype) * noise, Qz),
                )

    if k_idx < q:
        e = np.zeros(q, dtype=Vbasis.dtype)
        e[k_idx] = 1.0
        append_unique_reduced_seed(starts, retract_reduced(e, Qz))

    return starts


def continuation_single_vector(solver_fn, z0, schedule, solver_kwargs):
    schedule_vals = parse_continuation_schedule(schedule, default=(1.0,))
    z = z0
    last = None
    for c_scale in schedule_vals:
        last = solver_fn(z0=z, c_scale=c_scale, **solver_kwargs)
        z = last[0]
    return last


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


def entropyscore_forget_logscore_grad_reduced(B, z, rows_block, rows_ref, c_scale=1.0):
    B_arr = np.asarray(B)
    work_dtype = B_arr.dtype
    z = np.ascontiguousarray(np.asarray(z, dtype=work_dtype).reshape(-1))
    y = np.ascontiguousarray(B_arr @ z, dtype=work_dtype)
    y2_sq = max(float(np.dot(y, y)), 1e-30)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 1e-30)
    rows_block = max(int(rows_block), 2)
    rows_ref = max(int(rows_ref), rows_block)
    c_raw = np.log(rows_block / rows_ref) / np.log(rows_block)
    c = c_scale * c_raw
    alpha = (rows_block / rows_ref) ** 0.25
    logf = np.log(alpha) + (1.0 - 0.5 * c) * np.log(y2_sq) + 0.25 * c * np.log(y4_4)

    y3 = np.ascontiguousarray(y * y * y, dtype=work_dtype)
    g2 = np.ascontiguousarray(B_arr.T @ y, dtype=work_dtype) / y2_sq
    g4 = np.ascontiguousarray(B_arr.T @ y3, dtype=work_dtype) / y4_4
    grad = (2.0 - c) * g2 + c * g4
    H = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    s = float(np.sqrt(y2_sq))
    return logf, grad, s, H


def entropyscore_forget_streaming_logscore_grad_reduced(B_gain, B_block, C_prev, s2_old, z, rows_block, rows_ref, c_scale=1.0):
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
    c_raw = np.log(rows_block / rows_ref) / np.log(rows_block)
    c = c_scale * c_raw
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


def entropyscore_forget_score_grad_reduced(B, z, rows_block, rows_ref, c_scale=1.0):
    logf, grad_log, s, H = entropyscore_forget_logscore_grad_reduced(B, z, rows_block, rows_ref, c_scale=c_scale)
    score = float(np.exp(logf))
    grad = np.ascontiguousarray(score * grad_log, dtype=np.asarray(grad_log).dtype)
    return score, grad, s, H


def entropyscore_forget_streaming_score_grad_reduced(B_gain, B_block, C_prev, s2_old, z, rows_block, rows_ref, c_scale=1.0):
    logf, grad_log, s, H = entropyscore_forget_streaming_logscore_grad_reduced(
        B_gain, B_block, C_prev, s2_old, z, rows_block, rows_ref, c_scale=c_scale
    )
    score = float(np.exp(logf))
    grad = np.ascontiguousarray(score * grad_log, dtype=np.asarray(grad_log).dtype)
    return score, grad, s, H


def basic_projected_ascent_single_reduced_forget_cex(
    B, z0, Qz, rows_block, rows_ref, maxit=80, tol=1e-8, reuse_line_search_grad=True, c_scale=1.0
):
    z = retract_reduced(z0, Qz)
    if z is None:
        raise RuntimeError("Initial reduced seed is infeasible.")

    score, grad, s, H = entropyscore_forget_score_grad_reduced(B, z, rows_block, rows_ref, c_scale=c_scale)
    stop = {"reason": "maxit", "iters": maxit, "grad_norm": np.nan}
    progress_f_tol = 1e-12
    progress_step_tol = 1e-10

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
                score_t, grad_t, s_t, H_t = entropyscore_forget_score_grad_reduced(B, zt, rows_block, rows_ref, c_scale=c_scale)
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
            score, grad, s, H = entropyscore_forget_score_grad_reduced(B, z, rows_block, rows_ref, c_scale=c_scale)
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
    else:
        gtan = project_reduced(grad - z * float(z @ grad), Qz)
        stop = {"reason": "maxit", "iters": maxit, "grad_norm": float(np.linalg.norm(gtan))}

    return z, score, s, H, stop


def basic_projected_ascent_single_reduced_streaming_forget_cex(
    B_gain, B_block, C_prev, s2_old, z0, Qz, rows_block, rows_ref,
    maxit=80, tol=1e-8, reuse_line_search_grad=True, c_scale=1.0
):
    z = retract_reduced(z0, Qz)
    if z is None:
        raise RuntimeError("Initial reduced seed is infeasible.")

    score, grad, s, H = entropyscore_forget_streaming_score_grad_reduced(
        B_gain, B_block, C_prev, s2_old, z, rows_block, rows_ref, c_scale=c_scale
    )
    stop = {"reason": "maxit", "iters": maxit, "grad_norm": np.nan}
    progress_f_tol = 1e-12
    progress_step_tol = 1e-10

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
                score_t, grad_t, s_t, H_t = entropyscore_forget_streaming_score_grad_reduced(
                    B_gain, B_block, C_prev, s2_old, zt, rows_block, rows_ref, c_scale=c_scale
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
            score, grad, s, H = entropyscore_forget_streaming_score_grad_reduced(
                B_gain, B_block, C_prev, s2_old, z, rows_block, rows_ref, c_scale=c_scale
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
    else:
        gtan = project_reduced(grad - z * float(z @ grad), Qz)
        stop = {"reason": "maxit", "iters": maxit, "grad_norm": float(np.linalg.norm(gtan))}

    return z, score, s, H, stop


def basic_projected_ascent_single_reduced_forget(B, z0, Qz, rows_block, rows_ref, maxit=80, tol=1e-8, c_scale=1.0):
    z = retract_reduced(z0, Qz)
    if z is None:
        raise RuntimeError("Initial reduced seed is infeasible.")

    logf, grad, s, H = entropyscore_forget_logscore_grad_reduced(B, z, rows_block, rows_ref, c_scale=c_scale)
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
            logf_t, grad_t, s_t, H_t = entropyscore_forget_logscore_grad_reduced(B, zt, rows_block, rows_ref, c_scale=c_scale)
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
    B_gain, B_block, C_prev, s2_old, z0, Qz, rows_block, rows_ref, maxit=80, tol=1e-8, c_scale=1.0
):
    z = retract_reduced(z0, Qz)
    if z is None:
        raise RuntimeError("Initial reduced seed is infeasible.")

    logf, grad, s, H = entropyscore_forget_streaming_logscore_grad_reduced(
        B_gain, B_block, C_prev, s2_old, z, rows_block, rows_ref, c_scale=c_scale
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
                B_gain, B_block, C_prev, s2_old, zt, rows_block, rows_ref, c_scale=c_scale
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
    M_gain, A_block, v, Vred, state_prev, rows_ref, Q_prev=None, return_vector=False
):
    if state_prev is None:
        _, grad, _, _ = entropyscore_forget_logscore_grad_rows(A_block, v, rows_ref)
    else:
        _, grad, _, _ = entropyscore_forget_streaming_logscore_grad(
            M_gain, A_block, state_prev["V"], state_prev["s2"], v, rows_ref
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
    v_type="rand",
):
    if n <= r_sig:
        raise ValueError("n must be larger than r_sig.")
    if n & (n - 1):
        raise ValueError("n must be a power of two for scipy.linalg.hadamard.")

    k = n
    U0 = np.zeros((n, n), dtype=float)
    H = la.hadamard(n).astype(float)
    U0[:, :r_sig] = H[:, :r_sig] / np.sqrt(n)

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

    sig_block = sigma1 * np.arange(1, r_sig + 1, dtype=float) ** (-alpha_sig)
    tail_block = tail_scale * np.arange(1, k - r_sig + 1, dtype=float) ** (-alpha_tail)
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


def generate_matrix_input(matrix, n=1024, preset="fast", seed=0, **kwargs):
    if matrix == "static-cex":
        return generate_structured_cex_input(n=n, **kwargs)
    if matrix == "alternative-data-signals":
        return generate_alternative_data_signals_input(n=n, preset=preset, seed=seed)
    if matrix == "crowded-strategy":
        return generate_crowded_strategy_input(n=n, preset=preset, seed=seed)
    if matrix == "execution-cost-slippage":
        return generate_execution_cost_slippage_input(n=n, preset=preset, seed=seed)
    if matrix == "etf-basket-basis":
        return generate_etf_basket_basis_input(n=n, preset=preset, seed=seed)
    if matrix == "futures-term-structure":
        return generate_futures_term_structure_input(n=n, preset=preset, seed=seed)
    if matrix == "intraday-liquidity-shape":
        return generate_intraday_liquidity_shape_input(n=n, preset=preset, seed=seed)
    if matrix == "macro-factor-panel":
        return generate_macro_factor_panel_input(n=n, preset=preset, seed=seed)
    if matrix == "rates-cross-currency":
        return generate_rates_cross_currency_input(n=n, preset=preset, seed=seed)
    if matrix == "stat-arb-spreads":
        return generate_stat_arb_spreads_input(n=n, preset=preset, seed=seed)
    if matrix == "options-vol-surface":
        return generate_options_vol_surface_input(n=n, preset=preset, seed=seed)
    if matrix == "risk-residual-panel":
        return generate_risk_residual_panel_input(n=n, preset=preset, seed=seed)
    if matrix == "realized-vol-corr":
        return generate_realized_vol_corr_input(n=n, preset=preset, seed=seed)
    raise ValueError(f"Unknown matrix family: {matrix}")


def entropy_iter_basis_forget(M_gain, active_r, rows_ref, V_init=None, q0=5, qmax=None,
                              krylov_depth=2, residual_tol=1e-2, expansion_maxit=8,
                              num_restarts=3, maxit=40, tol=1e-8, rng=None, verbose=True,
                              state_prev=None, A_block=None, rows_total=None,
                              reduced_optimizer="legacy", work_dtype=np.float32,
                              expansion_direction="krylov_v",
                              reuse_line_search_grad=True,
                              expansion_warm_start=False,
                              post_expansion_maxit=None,
                              warm_start_greedy=True,
                              warm_start_perturbations=2,
                              warm_start_perturb_scale=1e-2,
                              continuation=True,
                              continuation_schedule=(0.0, 0.25, 0.5, 0.75, 1.0)):
    continuation_schedule = parse_continuation_schedule(continuation_schedule, default=(0.0, 0.25, 0.5, 0.75, 1.0))
    del rows_total
    if rng is None:
        rng = np.random.default_rng(0)
    if reduced_optimizer not in {"legacy", "cex"}:
        raise ValueError(f"Unknown reduced_optimizer: {reduced_optimizer}")
    if expansion_direction not in {"krylov_v", "residual"}:
        raise ValueError(f"Unknown expansion_direction: {expansion_direction}")
    if A_block is None:
        raise ValueError("A_block is required for entropyscore_forget.")

    work_dtype = np.dtype(work_dtype)
    M_arr = np.asarray(M_gain, dtype=work_dtype)
    A_block_arr = np.asarray(A_block, dtype=M_arr.dtype)
    is_initial_block = state_prev is None

    if active_r <= 0:
        empty = np.zeros((M_arr.shape[1], 0), dtype=M_arr.dtype)
        return empty, np.zeros(0), np.zeros(0), np.zeros(0), {
            "seed_rank": 0,
            "max_rank": 0,
            "krylov_depth": int(krylov_depth),
            "residual_tol": float(residual_tol),
            "reduced_optimizer": reduced_optimizer,
            "work_dtype": str(M_arr.dtype),
            "expansion_direction": expansion_direction,
            "reuse_line_search_grad": bool(reuse_line_search_grad),
            "expansion_warm_start": bool(expansion_warm_start),
            "post_expansion_maxit": None if post_expansion_maxit is None else int(post_expansion_maxit),
            "warm_start_greedy": bool(warm_start_greedy),
            "warm_start_perturbations": int(warm_start_perturbations),
            "warm_start_perturb_scale": float(warm_start_perturb_scale),
            "continuation": bool(continuation),
            "continuation_schedule": tuple(float(v) for v in continuation_schedule),
            "subspace_dims": [],
            "expansion_iters": [],
            "grad_perp_ratio": np.zeros(0),
        }

    min_dim = min(M_arr.shape)
    q0_eff = max(1, min(int(q0), M_arr.shape[1], min_dim))
    q0_eff = max(active_r, q0_eff)
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

    V_out = np.zeros((M_arr.shape[1], active_r), dtype=M_arr.dtype)
    s_out = np.zeros(active_r, dtype=float)
    H_out = np.zeros(active_r, dtype=float)
    score_out = np.zeros(active_r, dtype=float)
    grad_perp_ratio = np.zeros(active_r, dtype=float)
    warm_init_score_out = np.full(active_r, np.nan, dtype=float)
    warm_final_score_out = np.full(active_r, np.nan, dtype=float)
    warm_gain_out = np.full(active_r, np.nan, dtype=float)
    best_init_score_out = np.full(active_r, np.nan, dtype=float)
    best_gain_out = np.full(active_r, np.nan, dtype=float)
    best_restart_was_warm = np.zeros(active_r, dtype=bool)
    warm_restart_count = np.zeros(active_r, dtype=int)
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
                "work_dtype": str(M_arr.dtype),
                "expansion_direction": expansion_direction,
                "reuse_line_search_grad": bool(reuse_line_search_grad),
                "expansion_warm_start": bool(expansion_warm_start),
                "post_expansion_maxit": None if post_expansion_maxit is None else int(post_expansion_maxit),
                "warm_start_greedy": bool(warm_start_greedy),
            "warm_start_perturbations": int(warm_start_perturbations),
            "warm_start_perturb_scale": float(warm_start_perturb_scale),
            "continuation": bool(continuation),
            "continuation_schedule": tuple(float(v) for v in continuation_schedule),
            "subspace_build_time": subspace_build_time,
            }
        })

    solve_t0 = time.time()
    V_init_work = None if V_init is None else np.ascontiguousarray(np.asarray(V_init, dtype=M_arr.dtype))
    prior_basis = orthonormalize_columns(V_init_work, dtype=M_arr.dtype) if V_init_work is not None and V_init_work.size else None

    for k_idx in range(active_r):
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
            q = Vbasis.shape[1]
            Qz = np.ascontiguousarray(Vbasis.T @ V_out[:, :k_idx], dtype=Vbasis.dtype) if k_idx > 0 else np.zeros((q, 0), dtype=Vbasis.dtype)
            if k_idx > 0:
                Qz = orthonormalize_columns(Qz, dtype=Vbasis.dtype)

            starts = []
            if warm_start_greedy:
                starts.extend(
                    make_reduced_warm_start_seeds(
                        Vbasis,
                        Qz,
                        k_idx,
                        z_warm=z_warm if expansion_warm_start else None,
                        prior_basis=prior_basis,
                        num_perturb=warm_start_perturbations,
                        perturb_scale=warm_start_perturb_scale,
                        rng=rng,
                    )
                )
            warm_seed_count = len(starts)

            if reduced_optimizer == "cex":
                cex_restart_budget = max(0, max(1, num_restarts) - len(starts))
                if cex_restart_budget:
                    Q_full = np.ascontiguousarray(V_out[:, :k_idx], dtype=M_arr.dtype) if k_idx > 0 else np.zeros((M_arr.shape[1], 0), dtype=M_arr.dtype)
                    full_starts = make_basic_restart_seeds(M_arr, Q_full, k_idx, V_init_work, cex_restart_budget)
                    for v0 in full_starts:
                        z0 = np.ascontiguousarray(Vbasis.T @ np.asarray(v0, dtype=Vbasis.dtype), dtype=Vbasis.dtype)
                        append_unique_reduced_seed(starts, retract_reduced(z0, Qz))
            else:
                if z_warm is not None and not warm_start_greedy:
                    append_unique_reduced_seed(starts, retract_reduced(z_warm, Qz))
                if prior_basis is not None and prior_basis.shape[1] > k_idx and not warm_start_greedy:
                    z_prior = np.ascontiguousarray(Vbasis.T @ prior_basis[:, k_idx], dtype=Vbasis.dtype)
                    append_unique_reduced_seed(starts, retract_reduced(z_prior, Qz))
                if k_idx < q and not warm_start_greedy:
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
            init_scores = []
            iter_budget = maxit
            if z_warm is not None and post_expansion_maxit is not None:
                iter_budget = max(1, min(int(maxit), int(post_expansion_maxit)))
            for z0 in starts:
                if is_initial_block:
                    if reduced_optimizer == "cex":
                        solver_fn = basic_projected_ascent_single_reduced_forget_cex
                        solver_kwargs = dict(
                            B=B_block,
                            Qz=Qz,
                            rows_block=A_block_arr.shape[0],
                            rows_ref=rows_ref,
                            maxit=iter_budget,
                            tol=tol,
                            reuse_line_search_grad=reuse_line_search_grad,
                        )
                        init_score_fn = entropyscore_forget_score_grad_reduced
                        init_score_kwargs = dict(B=B_block, rows_block=A_block_arr.shape[0], rows_ref=rows_ref)
                    else:
                        solver_fn = basic_projected_ascent_single_reduced_forget
                        solver_kwargs = dict(
                            B=B_block,
                            Qz=Qz,
                            rows_block=A_block_arr.shape[0],
                            rows_ref=rows_ref,
                            maxit=iter_budget,
                            tol=tol,
                        )
                        init_score_fn = entropyscore_forget_score_grad_reduced
                        init_score_kwargs = dict(B=B_block, rows_block=A_block_arr.shape[0], rows_ref=rows_ref)
                else:
                    if reduced_optimizer == "cex":
                        solver_fn = basic_projected_ascent_single_reduced_streaming_forget_cex
                        solver_kwargs = dict(
                            B_gain=B_gain,
                            B_block=B_block,
                            C_prev=C_prev,
                            s2_old=prev_s2,
                            Qz=Qz,
                            rows_block=A_block_arr.shape[0],
                            rows_ref=rows_ref,
                            maxit=iter_budget,
                            tol=tol,
                            reuse_line_search_grad=reuse_line_search_grad,
                        )
                        init_score_fn = entropyscore_forget_streaming_score_grad_reduced
                        init_score_kwargs = dict(B_gain=B_gain, B_block=B_block, C_prev=C_prev, s2_old=prev_s2, rows_block=A_block_arr.shape[0], rows_ref=rows_ref)
                    else:
                        solver_fn = basic_projected_ascent_single_reduced_streaming_forget
                        solver_kwargs = dict(
                            B_gain=B_gain,
                            B_block=B_block,
                            C_prev=C_prev,
                            s2_old=prev_s2,
                            Qz=Qz,
                            rows_block=A_block_arr.shape[0],
                            rows_ref=rows_ref,
                            maxit=iter_budget,
                            tol=tol,
                        )
                        init_score_fn = entropyscore_forget_streaming_score_grad_reduced
                        init_score_kwargs = dict(B_gain=B_gain, B_block=B_block, C_prev=C_prev, s2_old=prev_s2, rows_block=A_block_arr.shape[0], rows_ref=rows_ref)

                init_score, _, _, _ = init_score_fn(z=z0, c_scale=1.0, **init_score_kwargs)
                init_scores.append(float(init_score))

                if continuation:
                    cand = continuation_single_vector(
                        solver_fn=solver_fn,
                        z0=z0,
                        schedule=continuation_schedule,
                        solver_kwargs=solver_kwargs,
                    )
                else:
                    cand = solver_fn(z0=z0, c_scale=1.0, **solver_kwargs)

                if reduced_optimizer == "cex":
                    cand = (cand[0], np.log(max(cand[1], 1e-300)), cand[2], cand[3], cand[4])
                cand_results.append(cand)
            timing_totals["reduced_opt"] += time.perf_counter() - t_stage
            timing_counts["restart_solves"] += len(starts)

            best = None
            for restart_idx, cand in enumerate(cand_results):
                if best is None or cand[1] > best[1]:
                    best = cand
                    best_restart = restart_idx + 1

            z_best, logf_best, s_best, H_best, best_stop = best
            warm_scores = init_scores[:warm_seed_count]
            warm_finals = [float(np.exp(c[1])) for c in cand_results[:warm_seed_count]]
            if warm_scores:
                warm_init_score_out[k_idx] = float(max(warm_scores))
                warm_final_score_out[k_idx] = float(max(warm_finals))
                warm_gain_out[k_idx] = float(warm_final_score_out[k_idx] - warm_init_score_out[k_idx])
            best_init_score_out[k_idx] = float(init_scores[best_restart - 1]) if best_restart is not None else np.nan
            best_gain_out[k_idx] = float(np.exp(logf_best) - best_init_score_out[k_idx]) if best_restart is not None else np.nan
            best_restart_was_warm[k_idx] = bool(best_restart is not None and best_restart <= warm_seed_count)
            warm_restart_count[k_idx] = int(warm_seed_count)
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
        score_out[k_idx] = float(np.exp(logf_best))
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
                "warm_restart_count": int(warm_restart_count[k_idx]),
                "warm_init_score": None if np.isnan(warm_init_score_out[k_idx]) else float(warm_init_score_out[k_idx]),
                "warm_final_score": None if np.isnan(warm_final_score_out[k_idx]) else float(warm_final_score_out[k_idx]),
                "warm_gain": None if np.isnan(warm_gain_out[k_idx]) else float(warm_gain_out[k_idx]),
                "best_init_score": None if np.isnan(best_init_score_out[k_idx]) else float(best_init_score_out[k_idx]),
                "best_gain": None if np.isnan(best_gain_out[k_idx]) else float(best_gain_out[k_idx]),
                "best_restart_was_warm": bool(best_restart_was_warm[k_idx]),
            })

    solve_time = time.time() - solve_t0
    diag = {
        "seed_rank": q0_eff,
        "max_rank": qmax,
        "krylov_depth": int(krylov_depth),
        "residual_tol": float(residual_tol),
        "reduced_optimizer": reduced_optimizer,
        "work_dtype": str(M_arr.dtype),
        "expansion_direction": expansion_direction,
        "reuse_line_search_grad": bool(reuse_line_search_grad),
        "expansion_warm_start": bool(expansion_warm_start),
        "post_expansion_maxit": None if post_expansion_maxit is None else int(post_expansion_maxit),
        "warm_start_greedy": bool(warm_start_greedy),
                "warm_start_perturbations": int(warm_start_perturbations),
                "warm_start_perturb_scale": float(warm_start_perturb_scale),
                "continuation": bool(continuation),
                "continuation_schedule": tuple(float(v) for v in continuation_schedule),
                "subspace_build_time": subspace_build_time,
        "reduced_solve_time": solve_time,
        "grad_perp_ratio": grad_perp_ratio,
        "warm_init_score": warm_init_score_out,
        "warm_final_score": warm_final_score_out,
        "warm_gain": warm_gain_out,
        "best_init_score": best_init_score_out,
        "best_gain": best_gain_out,
        "best_restart_was_warm": best_restart_was_warm,
        "warm_restart_count": warm_restart_count,
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
    "dtype": "float32",
    "expansion_direction": "residual",
    "reuse_line_search_grad": True,
    "expansion_warm_start": True,
    "post_expansion_maxit": 60,
    "warm_start_greedy": True,
    "warm_start_perturbations": 2,
    "warm_start_perturb_scale": 1e-2,
    "continuation": True,
    "continuation_schedule": (0.0, 0.25, 0.5, 0.75, 1.0),
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
    "dtype": "float64",
    "expansion_direction": "krylov_v",
    "reuse_line_search_grad": True,
    "expansion_warm_start": False,
    "post_expansion_maxit": None,
    "warm_start_greedy": True,
    "warm_start_perturbations": 2,
    "warm_start_perturb_scale": 1e-2,
    "continuation": True,
    "continuation_schedule": (0.0, 0.25, 0.5, 0.75, 1.0),
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
    "dtype": "float32",
    "expansion_direction": "krylov_v",
    "reuse_line_search_grad": True,
    "expansion_warm_start": False,
    "post_expansion_maxit": None,
    "warm_start_greedy": True,
    "warm_start_perturbations": 2,
    "warm_start_perturb_scale": 1e-2,
    "continuation": True,
    "continuation_schedule": (0.0, 0.25, 0.5, 0.75, 1.0),
}

PRESETS = {
    "fast": FAST_BALANCED_PRESET,
    "cex-replicate": CEX_REPLICATE_PRESET,
    "small": SMALL_PROBE_PRESET,
}


def fmt_row(x, precision=4):
    return " ".join(f"{float(v): .{precision}f}" for v in np.asarray(x).reshape(-1))


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
    A = np.asarray(A, dtype=np.float64)
    if args.normalize_by_sigma:
        A = A / sigma1
    n = A.shape[0]
    r = args.rank
    win = args.win

    state = None
    V_r = None
    S_r = None

    print(f"Input: {source_desc}: A={A.shape}, sigma1={sigma1:.12g}, normalize_by_sigma={args.normalize_by_sigma}")
    work_dtype = np.float64 if args.dtype == "float64" else np.float32
    print(f"Mode: {args.mode}")
    if args.mode == "restricted":
        print(
            "Restricted optimizer params: "
            f"preset={args.preset}, q0={args.q0}, qmax={args.qmax}, krylov_depth={args.krylov_depth}, "
            f"residual_tol={args.residual_tol}, expansion_maxit={args.expansion_maxit}, "
            f"num_restarts={args.num_restarts}, maxit={args.maxit}, carry={args.carry}, "
            f"reduced_optimizer={args.reduced_optimizer}, dtype={np.dtype(work_dtype)}, "
            f"expansion_direction={args.expansion_direction}, "
            f"reuse_line_search_grad={args.reuse_line_search_grad}, "
            f"expansion_warm_start={args.expansion_warm_start}, "
            f"post_expansion_maxit={args.post_expansion_maxit}, "
            f"warm_start_greedy={args.warm_start_greedy}, "
            f"warm_start_perturbations={args.warm_start_perturbations}, "
            f"warm_start_perturb_scale={args.warm_start_perturb_scale}, "
            f"continuation={args.continuation}, "
            f"continuation_schedule={args.continuation_schedule}"
        )

    for start0 in range(0, n, win):
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
            V_init = None
            rows_seen = A_block.shape[0]
            print(f"\n===== block rows {start0 + 1}:{end0} (initial restricted score) =====")
        else:
            B_top = (state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T)
            M_gain = np.vstack([B_top, A_block_work]).astype(work_dtype, copy=False)
            V_init = state["V"].astype(work_dtype, copy=False)
            rows_seen = state["rows_seen"] + A_block.shape[0]
            print(f"\n===== block rows {start0 + 1}:{end0} (streaming restricted score) =====")

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
            work_dtype=work_dtype,
            expansion_direction=args.expansion_direction,
            reuse_line_search_grad=args.reuse_line_search_grad,
            expansion_warm_start=args.expansion_warm_start,
            post_expansion_maxit=args.post_expansion_maxit,
            warm_start_greedy=args.warm_start_greedy,
            warm_start_perturbations=args.warm_start_perturbations,
            warm_start_perturb_scale=args.warm_start_perturb_scale,
            continuation=args.continuation,
            continuation_schedule=args.continuation_schedule,
        )

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
        }

        print(f"rows {start0 + 1}:{end0}")
        print(f"s: {fmt_row(s_new)}")
        print(f"H: {fmt_row(H_score)}")
        print(f"scores: {fmt_row(score_score)}")
        warm_init_block = diag['warm_init_score'][:r]
        warm_final_block = diag['warm_final_score'][:r]
        warm_gain_block = diag['warm_gain'][:r]
        best_init_block = diag['best_init_score'][:r]
        best_gain_block = diag['best_gain'][:r]
        best_warm_block = diag['best_restart_was_warm'][:r]
        print(f"warm_init_scores: {fmt_row(warm_init_block)}")
        print(f"warm_final_scores: {fmt_row(warm_final_block)}")
        print(f"warm_score_gain: {fmt_row(warm_gain_block)}")
        print(f"best_init_scores: {fmt_row(best_init_block)}")
        print(f"best_score_gain: {fmt_row(best_gain_block)}")
        print(f"best_restart_was_warm: {best_warm_block.tolist()}")
        print(f"sum_scores: {float(np.sum(score_score)):.6f}")
        print(f"sum_warm_init_scores: {float(np.nansum(warm_init_block)):.6f}")
        print(f"sum_warm_final_scores: {float(np.nansum(warm_final_block)):.6f}")
        print(f"sum_warm_score_gain: {float(np.nansum(warm_gain_block)):.6f}")
        print(f"sum_best_init_scores: {float(np.nansum(best_init_block)):.6f}")
        print(f"sum_best_score_gain: {float(np.nansum(best_gain_block)):.6f}")
        print(f"subspace_dims: {diag['subspace_dims'][:r].tolist()}")
        print(f"grad_perp_ratio: {fmt_row(diag['grad_perp_ratio'][:r], precision=6)}")

    align = np.linalg.norm((V_r @ V_r.T) @ V_exact[:, :1], "fro")
    top_sval_est = S_r[0, 0]
    rel_err_sval = abs(top_sval_est - sigma1) / sigma1
    elapsed = time.time() - t0
    print("sigma1    mean_align    mean_relerr_sval    elapsed")
    print(f"{sigma1:.3f}      {align:.6f}           {rel_err_sval:.8f}          {elapsed:.3f}")

    result = {
        "matrix": args.matrix if not args.mat_input else os.path.basename(args.mat_input),
        "method": args.mode,
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
        choices=("restricted", "isvd", "fd", "iSVD", "FD"),
        default="restricted",
        help="Streaming method to run: restricted optimizer, incremental SVD, or Frequent Directions.",
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
    parser.add_argument("--normalize-by-sigma", action="store_true")
    parser.add_argument("--carry", choices=("left", "right"))
    parser.add_argument("--reduced-optimizer", choices=("legacy", "cex"))
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
    parser.add_argument(
        "--warm-start-greedy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include greedy/deflation-style reduced warm starts before random restarts.",
    )
    parser.add_argument(
        "--warm-start-perturbations",
        type=int,
        help="Number of small perturbations of the prior warm start to include.",
    )
    parser.add_argument(
        "--warm-start-perturb-scale",
        type=float,
        help="Scale of small perturbations used for warm-start seeds.",
    )
    parser.add_argument(
        "--continuation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run continuation / homotopy in the entropy exponent c before the final objective.",
    )
    parser.add_argument(
        "--continuation-schedule",
        help="Comma-separated c-scale schedule, e.g. 0,0.25,0.5,0.75,1.0.",
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

    if args.cex_replicate:
        args.preset = "cex-replicate"

    preset_values = PRESETS[args.preset]
    for name, value in preset_values.items():
        if getattr(args, name) is None:
            setattr(args, name, value)

    args.continuation_schedule = parse_continuation_schedule(args.continuation_schedule)
    return args


if __name__ == "__main__":
    run(parse_args())
