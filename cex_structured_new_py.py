import argparse
import os
import time

import numpy as np
import scipy.linalg as la
import scipy.io as sio


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
    x = np.asarray(x, dtype=float).reshape(-1)
    if Q is not None and Q.size:
        x = x - Q @ (Q.T @ x)
    return x


def retract_feasible(x, Q):
    x = project_feasible(x, Q)
    nx = np.sqrt(kahan_sum(np.abs(x) ** 2))
    if nx <= 1e-14:
        return None
    return x / nx


def project_to_feasible_tangent(g, v, Q):
    g_feas = np.asarray(g, dtype=float).reshape(-1)
    if Q is not None and Q.size:
        g_feas = g_feas - Q @ (Q.T @ g_feas)
    g_feas = g_feas - v * float(v.T @ g_feas)
    return g_feas


def window_score_grad_rows(M, v, n):
    y = M @ v
    abs_y = np.abs(y)
    y2_sq = kahan_sum(abs_y ** 2)
    y4_4 = kahan_sum(abs_y ** 4)
    rows_new = M.shape[0]

    if rows_new <= 1 or y2_sq <= 1e-28 or y4_4 <= 1e-28 or np.any(~np.isfinite(y)):
        return 0.0, np.zeros_like(v), y2_sq, np.inf

    c_k = np.log(rows_new / n) / np.log(rows_new)
    scale = (rows_new / n) ** 0.25
    score = scale * np.exp((1 - 0.5 * c_k) * np.log(y2_sq) + 0.25 * c_k * np.log(y4_4))

    My = M.T @ y
    My3 = M.T @ (y ** 3)
    g = score * ((2 - c_k) * (My / y2_sq) + c_k * (My3 / y4_4))
    H = -(np.log(y4_4) - 2 * np.log(y2_sq))
    return score, g, y2_sq, H


def window_streaming_score_grad(A_block, V_old, s2_old, v, n):
    a = V_old.T @ v
    y = A_block @ v
    abs_y = np.abs(y)
    y2_sq = kahan_sum(abs_y ** 2)
    y4_4 = kahan_sum(abs_y ** 4)

    P_old = float(np.sum((a ** 2) * s2_old))
    g_old = 2 * (V_old @ (s2_old * a))
    rows_new = A_block.shape[0]

    if rows_new <= 1 or y2_sq <= 1e-28 or y4_4 <= 1e-28 or np.any(~np.isfinite(y)):
        return P_old, g_old, P_old, np.inf

    c_k = np.log(rows_new / n) / np.log(rows_new)
    scale = (rows_new / n) ** 0.25
    psi = scale * np.exp((1 - 0.5 * c_k) * np.log(y2_sq) + 0.25 * c_k * np.log(y4_4))

    Ay = A_block.T @ y
    Ay3 = A_block.T @ (y ** 3)
    g_psi = psi * ((2 - c_k) * (Ay / y2_sq) + c_k * (Ay3 / y4_4))

    score = P_old + psi
    g = g_old + g_psi
    P_total = P_old + y2_sq
    Hcurr = -(np.log(y4_4) - 2 * np.log(y2_sq))
    return score, g, P_total, Hcurr


def window_score_fast(M_or_Ablock, v, n, V_old=None, s2_old=None):
    if V_old is None and s2_old is None:
        score, _, _, _ = window_score_grad_rows(M_or_Ablock, v, n)
        return score
    if V_old is None or s2_old is None:
        raise ValueError("window_score_fast expects either exact or streaming inputs.")

    A_block = M_or_Ablock
    a = V_old.T @ v
    y = A_block @ v
    abs_y = np.abs(y)
    y2_sq = kahan_sum(abs_y ** 2)
    y4_4 = kahan_sum(abs_y ** 4)
    rows_new = A_block.shape[0]
    P_old = float(np.sum((a ** 2) * s2_old))

    if rows_new <= 1 or y2_sq <= 1e-28 or y4_4 <= 1e-28 or np.any(~np.isfinite(y)):
        return P_old

    c_k = np.log(rows_new / n) / np.log(rows_new)
    scale = (rows_new / n) ** 0.25
    psi = scale * np.exp((1 - 0.5 * c_k) * np.log(y2_sq) + 0.25 * c_k * np.log(y4_4))
    return P_old + psi


def basic_projected_ascent_single_exact(M, v0, Q, n, maxit, tol):
    v = retract_feasible(v0, Q)
    if v is None:
        raise RuntimeError("Initial vector infeasible in exact optimizer.")

    score, gradE, s2, H2 = window_score_grad_rows(M, v, n)
    progress_f_tol = 1e-12
    progress_step_tol = 1e-10

    for _ in range(maxit):
        g = project_to_feasible_tangent(gradE, v, Q)
        gnorm = np.sqrt(kahan_sum(np.abs(g) ** 2))
        if gnorm <= tol:
            return v, score, s2, H2

        accepted = False
        alpha = 1.0
        score_old = score
        v_old = v

        for _ in range(20):
            vt = retract_feasible(v + alpha * g, Q)
            if vt is not None:
                score_trial, _, _, _ = window_score_grad_rows(M, vt, n)
                rhs = score_old + 1e-4 * alpha * float(g.T @ g)
                if score_trial >= rhs:
                    accepted = True
                    v = vt
                    break
            alpha *= 0.5

        if not accepted:
            v = v_old
            return v, score, s2, H2

        score, gradE, s2, H2 = window_score_grad_rows(M, v, n)
        step_norm = np.sqrt(kahan_sum(np.abs(v - v_old) ** 2))
        f_change = abs(score - score_old)
        f_threshold = progress_f_tol * max(1.0, abs(score_old))

        if f_change <= f_threshold or step_norm <= progress_step_tol:
            return v, score, s2, H2

    return v, score, s2, H2


def basic_projected_ascent_single_streaming(A_block, V_old, s2_old, n, v0, Q, maxit, tol):
    v = retract_feasible(v0, Q)
    if v is None:
        raise RuntimeError("Initial vector infeasible in streaming optimizer.")

    score, gradE, s2_total, H_curr = window_streaming_score_grad(A_block, V_old, s2_old, v, n)
    progress_f_tol = 1e-12
    progress_step_tol = 1e-10

    for _ in range(maxit):
        g = project_to_feasible_tangent(gradE, v, Q)
        gnorm = np.sqrt(kahan_sum(np.abs(g) ** 2))
        if gnorm <= tol:
            return v, score, s2_total, H_curr

        accepted = False
        alpha = 1.0
        score_old = score
        v_old = v

        for _ in range(20):
            vt = retract_feasible(v + alpha * g, Q)
            if vt is not None:
                score_trial, _, _, _ = window_streaming_score_grad(A_block, V_old, s2_old, vt, n)
                rhs = score_old + 1e-4 * alpha * float(g.T @ g)
                if score_trial >= rhs:
                    accepted = True
                    v = vt
                    break
            alpha *= 0.5

        if not accepted:
            v = v_old
            return v, score, s2_total, H_curr

        score, gradE, s2_total, H_curr = window_streaming_score_grad(A_block, V_old, s2_old, v, n)
        step_norm = np.sqrt(kahan_sum(np.abs(v - v_old) ** 2))
        f_change = abs(score - score_old)
        f_threshold = progress_f_tol * max(1.0, abs(score_old))

        if f_change <= f_threshold or step_norm <= progress_step_tol:
            return v, score, s2_total, H_curr

    return v, score, s2_total, H_curr


def make_basic_restart_seeds(M, Q, k, V_init, num_restarts):
    d = M.shape[1]
    _, _, Vh = la.svd(M, full_matrices=False, lapack_driver="gesdd")
    Vsvd = Vh.T
    num_top = min(4, Vsvd.shape[1])
    alpha_grid = np.array([0.98, 0.9, 0.75, 0.5, 0.25, 0.0])
    starts = []

    for restart in range(1, num_restarts + 1):
        if V_init is not None and V_init.size and V_init.shape[1] >= k:
            v_prev = V_init[:, k - 1]
        else:
            v_prev = None

        restart_type = ((restart - 1) % 5) + 1
        restart_block = (restart - 1) // 5

        if restart_type == 1:
            if v_prev is not None:
                xi = np.random.randn(d)
                xi = project_feasible(xi, Q)
                nxi = np.sqrt(kahan_sum(np.abs(xi) ** 2))
                if nxi > 1e-14:
                    xi = xi / nxi
                alpha = alpha_grid[restart_block % len(alpha_grid)]
                v0 = alpha * v_prev + np.sqrt(max(0.0, 1 - alpha ** 2)) * xi
            else:
                v0 = Vsvd[:, 0]
        elif restart_type == 2:
            j = restart_block % num_top
            v0 = Vsvd[:, j]
        elif restart_type == 3:
            j1 = restart_block % num_top
            j2 = (restart_block + 1) % num_top
            alpha = alpha_grid[restart_block % len(alpha_grid)]
            v0 = alpha * Vsvd[:, j1] + np.sqrt(max(0.0, 1 - alpha ** 2)) * Vsvd[:, j2]
        elif restart_type == 4:
            j = restart_block % num_top
            v0 = Vsvd[:, j] + 1e-2 * np.random.randn(d)
        else:
            v0 = np.random.randn(d)

        v = retract_feasible(v0, Q)
        if v is None:
            v = retract_feasible(np.random.randn(d), Q)
        if v is None:
            raise RuntimeError("Could not generate feasible restart seed.")
        starts.append(v)

    return starts


def window_iter_basis_streaming(A_block, r, n, state_prev, V_init, num_restarts, maxit, tol):
    d = A_block.shape[1]
    rows_new = A_block.shape[0]
    V_out = np.zeros((d, r))
    s_out = np.zeros(r)
    H_out = np.full(r, -np.inf)
    score_out = np.full(r, -np.inf)
    Q = np.zeros((d, 0))

    is_initial_block = state_prev is None
    if is_initial_block:
        rows_total = rows_new
        M_gain = A_block
        prev_basis = None
        prev_s2 = None
    else:
        rows_total = state_prev["rows_seen"] + rows_new
        B_top = state_prev["s"][:, None] * state_prev["V"].T
        M_gain = np.vstack([B_top, A_block])
        prev_basis = state_prev["V"]
        prev_s2 = state_prev["s2"]

    for kk in range(1, r + 1):
        starts = make_basic_restart_seeds(M_gain, Q, kk, V_init, num_restarts)
        best_v = None
        best_score = -np.inf
        best_s2 = 0.0
        best_H = np.inf

        for v0 in starts:
            if is_initial_block:
                v_cand, score_cand, s2_cand, H_cand = basic_projected_ascent_single_exact(
                    A_block, v0, Q, n, maxit, tol
                )
            else:
                v_cand, score_cand, s2_cand, H_cand = basic_projected_ascent_single_streaming(
                    A_block, prev_basis, prev_s2, n, v0, Q, maxit, tol
                )
            if score_cand > best_score:
                best_score = score_cand
                best_v = v_cand
                best_s2 = s2_cand
                best_H = H_cand

        if best_v is None:
            raise RuntimeError(f"All restarts failed for k={kk}.")

        Q = np.column_stack([Q, best_v])
        V_out[:, kk - 1] = best_v
        s_out[kk - 1] = np.sqrt(max(best_s2, 0.0))
        H_out[kk - 1] = best_H
        score_out[kk - 1] = best_score

    state_out = {
        "V": V_out,
        "s": s_out,
        "s2": s_out ** 2,
        "H": H_out,
        "score": score_out,
        "rows_seen": rows_total,
        "prev_basis": prev_basis,
        "prev_s2": prev_s2,
        "prev_sketch": None,
    }
    return V_out, s_out, H_out, score_out, state_out


def projected_subspace_svd(M_gain, V_basis):
    if V_basis is None or not V_basis.size:
        return V_basis, np.zeros(0)
    Q, _ = np.linalg.qr(V_basis, mode="reduced")
    B_proj = M_gain @ Q
    _, s_proj, R_proj_h = la.svd(B_proj, full_matrices=False, lapack_driver="gesdd")
    R_proj = R_proj_h.T
    V_proj = Q @ R_proj
    return V_proj, s_proj


def fmt_row(x, precision=4):
    return " ".join(f"{float(v): .{precision}f}" for v in np.asarray(x).reshape(-1))


def load_matlab_cex_input(mat_input):
    if not os.path.exists(mat_input):
        raise FileNotFoundError(
            f"{mat_input} not found. Run export_cex1_input.m in activated MATLAB first."
        )

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


def run(mat_input=None, mode="Combined", deflate_window=True):
    """Streaming experiment driver.

    Parameters
    ----------
    mat_input : str or None
        Optional MATLAB-exported input.
    mode : {"Combined", "DefSVD"}
        Direction-policy. "Combined" runs the optimizer + projected re-SVD
        (original behavior). "DefSVD" picks V_hat from the SVD of a
        soft-thresholded stack: SVT_{Δ²}(S_r V_r') stacked with
        (optionally) SVT_{Δ²}(A_block), where Δ² is the cumulative
        σ_{r+1}^2 discarded by the deflated stack across prior blocks.
        Magnitudes are honest against the raw M_gain via
        projected_subspace_svd (C2 from the design discussion).
    deflate_window : bool
        Only meaningful for mode == "DefSVD". If True, soft-threshold
        the current window the same way as the carry; if False, leave
        the window pristine (carry-only deflation — closer in spirit to
        Liberty's FD's "fresh data is sacred" principle).
    """
    np.random.seed(0)
    t0 = time.time()

    n = 1024
    r = 2
    l = 1
    win = 100
    v_type = "rand"
    r_sig = 2
    alpha_sig = 0.003
    alpha_tail = 0.0145
    tail_scale = 0.99
    first_svals = np.array([0.991])
    num_exper = 1
    k = n

    if mat_input:
        A_loaded, V, svec_loaded, sigma1_loaded = load_matlab_cex_input(mat_input)
        n = A_loaded.shape[0]
        k = A_loaded.shape[1]
        first_svals = np.array([sigma1_loaded])
        print(f"Loaded MATLAB cex input from {mat_input}")
    else:
        A_loaded = None
        svec_loaded = None

        U0 = np.zeros((n, n))
        H = la.hadamard(n).astype(float)
        U0[:, :r_sig] = H[:, :r_sig] / np.sqrt(n)

        a_tail = np.sqrt(1 - r_sig / n)
        b_tail = 1 / np.sqrt(n)
        for j in range(r_sig, n):
            col = np.zeros(n)
            idx_large = j - r_sig
            if idx_large <= n - r_sig - 1:
                col[idx_large] = a_tail
            else:
                raise RuntimeError("Tail index out of range; reduce r_sig or adjust construction.")
            col[n - r_sig:n] = b_tail
            U0[:, j] = col

        Qtmp, _ = np.linalg.qr(U0, mode="reduced")
        for j in range(r_sig):
            if float(Qtmp[:, j].T @ U0[:, j]) < 0:
                Qtmp[:, j] = -Qtmp[:, j]
        U = Qtmp[:, :k]

        if v_type == "id":
            V = np.eye(n, k)
        elif v_type == "U":
            V = U
        elif v_type == "rand":
            V, _ = np.linalg.qr(np.random.randn(n, k), mode="reduced")
        else:
            raise RuntimeError("Unknown V_type. Use 'id', 'U', or 'rand'.")

    alignment_results = np.zeros((len(first_svals), num_exper))
    relerr_sval_results = np.zeros((len(first_svals), num_exper))
    low_sval_indicator = np.zeros((len(first_svals), num_exper))

    for i, sigma1 in enumerate(first_svals):
        if svec_loaded is None:
            sig_block = sigma1 * np.arange(1, r_sig + 1, dtype=float) ** (-alpha_sig)
            tail_block = tail_scale * np.arange(1, k - r_sig + 1, dtype=float) ** (-alpha_tail)
            svec = np.concatenate([sig_block, tail_block])
            svec[0] = sigma1
        else:
            svec = svec_loaded
        S = np.diag(svec)
        E_opt = np.sum(svec[r:] ** 2) if r < k else 0.0

        for e in range(num_exper):
            if A_loaded is None:
                p = np.random.permutation(n)
                A = U @ S @ V.T
                A = A[p, :]
            else:
                A = A_loaded
            mA = A.shape[0]

            state = None
            V_r = None
            S_r = None
            delta_sum_sq = 0.0   # DefSVD cumulative Δ² (σ² units); ignored by Combined.

            for start0 in range(0, mA, win):
                end0 = min(start0 + win, mA)
                start_row = start0 + 1
                end_row = end0
                A_block = A[start0:end0, :]

                if state is None:
                    print(f"\n===== block rows {start_row}:{end_row} (initial exact block score) =====")
                else:
                    print(f"\n===== block rows {start_row}:{end_row} (streaming additive block score) =====")

                if mode == "Combined":
                    V_hat, s_new, H_new, score_new, state_new = window_iter_basis_streaming(
                        A_block, r, n, state, V_r, 8, 200, 1e-8
                    )

                    if V_r is None or S_r is None:
                        state_new["prev_sketch"] = None
                    else:
                        state_new["prev_sketch"] = S_r @ V_r.T

                    if state_new["prev_sketch"] is None:
                        current_block = A_block
                    else:
                        current_block = np.vstack([state_new["prev_sketch"], A_block])

                    V_hat, s_new = projected_subspace_svd(current_block, V_hat)
                    state_new["V"] = V_hat
                    state_new["s"] = s_new
                    state_new["s2"] = s_new ** 2

                elif mode == "DefSVD":
                    # Raw augmented matrix — used for honest magnitudes (C2).
                    if V_r is None or S_r is None:
                        prev_sketch = None
                        M_gain = A_block
                    else:
                        prev_sketch = S_r @ V_r.T
                        M_gain = np.vstack([prev_sketch, A_block])

                    # Deflate carry by Δ² in σ² units: SVT_{√Δ²} on σ.
                    if V_r is None or S_r is None:
                        B_def = None
                    else:
                        s_sketch = np.diag(S_r)
                        s_sketch_def = np.sqrt(np.maximum(s_sketch ** 2 - delta_sum_sq, 0.0))
                        B_def = (V_r * s_sketch_def).T   # = diag(s_sketch_def) @ V_r.T

                    # Optionally deflate the current window the same way.
                    if deflate_window:
                        U_w, s_w, Vh_w = la.svd(A_block, full_matrices=False, lapack_driver="gesdd")
                        s_w_def = np.sqrt(np.maximum(s_w ** 2 - delta_sum_sq, 0.0))
                        A_w_def = (U_w * s_w_def) @ Vh_w
                    else:
                        A_w_def = A_block

                    M_def = A_w_def if B_def is None else np.vstack([B_def, A_w_def])

                    # Direction policy: top-r right singular vectors of deflated stack.
                    _, s_def, Vh_def = la.svd(M_def, full_matrices=False, lapack_driver="gesdd")
                    V_def_full = Vh_def.T
                    rr = min(r, V_def_full.shape[1])
                    V_hat_raw = V_def_full[:, :rr]

                    # Δ² accumulator: largest σ² discarded by the deflated stack.
                    delta_this_sq = float(s_def[rr] ** 2) if len(s_def) > rr else 0.0

                    # Magnitude policy: honest projection onto raw M_gain.
                    V_hat, s_new = projected_subspace_svd(M_gain, V_hat_raw)
                    s_new = np.asarray(s_new).reshape(-1)
                    H_new = np.full(s_new.shape, np.nan)
                    score_new = s_new ** 2

                    state_new = {
                        "V": V_hat,
                        "s": s_new,
                        "s2": s_new ** 2,
                        "H": H_new,
                        "score": score_new,
                        "rows_seen": end_row,
                        "prev_basis": V_r,
                        "prev_s2": (np.diag(S_r) ** 2) if S_r is not None else None,
                        "prev_sketch": prev_sketch,
                        "delta_sum_sq_in": delta_sum_sq,
                        "delta_this_sq": delta_this_sq,
                    }

                    print(
                        f"DefSVD (deflate_window={deflate_window})"
                        f"  Δ²_in={delta_sum_sq:.6g}  δ²_this={delta_this_sq:.6g}"
                    )

                    delta_sum_sq += delta_this_sq

                else:
                    raise ValueError(f"Unknown mode: {mode!r} (expected 'Combined' or 'DefSVD')")

                V_r = V_hat
                S_r = np.diag(s_new)
                state = state_new

                print(f"rows {start_row}:{end_row}")
                print(f"s: {fmt_row(s_new)}")
                print(f"H: {fmt_row(H_new)}")
                print(f"scores: {fmt_row(score_new)}")

                if mode != "Combined":
                    # Skip the Combined-specific oracle-comparison debug
                    # block; downstream metrics are still computed below.
                    continue

                if start_row == 1:
                    _, _, Vh_tmp = la.svd(A_block, full_matrices=False, lapack_driver="gesdd")
                    vtmp = Vh_tmp.T
                    e1_proj = vtmp @ (vtmp.T @ V[:, 0])
                    e2_proj = vtmp @ (vtmp.T @ V[:, 1])
                    e1_proj_norm = np.linalg.norm(e1_proj)
                    e2_proj_norm = np.linalg.norm(e2_proj)
                    if e1_proj_norm > 1e-14:
                        e1_proj = e1_proj / e1_proj_norm
                    if e2_proj_norm > 1e-14:
                        e2_proj = e2_proj / e2_proj_norm
                    if e1_proj_norm > 1e-14:
                        score_e1_proj = window_score_fast(A_block, e1_proj, n)
                        print(f"score of v1 projection onto window space: {score_e1_proj: .4f}")
                        print("actual score:", fmt_row(score_new))
                        print(f"V(1,1)={V_hat[0,0]:.5f}")
                        print(f"should be: {e1_proj[0]:.5f}")
                        if V_hat.shape[1] >= 2 and e2_proj_norm > 1e-14:
                            score_e2_proj = window_score_fast(A_block, e2_proj, n)
                            score_vhat2 = window_score_fast(A_block, V_hat[:, 1], n)
                            print(f"approx score of v2 projection onto sketch+window space: {score_e2_proj: .4f}")
                            print(f"approx score of V_hat(:,2): {score_vhat2: .4f}")
                            print(f"reported score_new(2): {score_new[1]: .4f}")
                            print("V_hat(2,:)")
                            print(fmt_row(V_hat[1, :]))
                            print(f"should be: {e2_proj[1]:.5f}")
                        ll = min(l, V_r.shape[1])
                        align = np.linalg.norm(V_r @ (V_r.T @ V[:, :ll]), "fro") / np.sqrt(ll)
                        print(f"{align:.4f}")
                        print("Projection norms:")
                        print(fmt_row(np.linalg.norm(V_r @ (V_r.T @ np.column_stack([e1_proj, e2_proj])), axis=0)))
                else:
                    if state["prev_basis"] is not None:
                        a_dbg = state["prev_basis"].T @ V_hat[:, 0]
                        y_dbg = A_block @ V_hat[:, 0]
                        E_old_dbg = float(np.sum((a_dbg ** 2) * state["prev_s2"]))
                        print(f"debug E_old(first vec)={E_old_dbg:.12e}")
                        print(f"debug ||A_w v||_2^2(first vec)={kahan_sum(np.abs(y_dbg) ** 2):.12e}")
                        print(f"debug ||A_w v||_4^4(first vec)={kahan_sum(np.abs(y_dbg) ** 4):.12e}")
                    if state["prev_sketch"] is None:
                        current_block = A_block
                    else:
                        current_block = np.vstack([state["prev_sketch"], A_block])
                    _, _, Vh_stream = la.svd(current_block, full_matrices=False, lapack_driver="gesdd")
                    vtmp_stream = Vh_stream.T
                    e1_proj = vtmp_stream @ (vtmp_stream.T @ V[:, 0])
                    e2_proj = vtmp_stream @ (vtmp_stream.T @ V[:, 1])
                    e1_proj_norm = np.linalg.norm(e1_proj)
                    e2_proj_norm = np.linalg.norm(e2_proj)
                    if e1_proj_norm > 1e-14:
                        e1_proj = e1_proj / e1_proj_norm
                    if e2_proj_norm > 1e-14:
                        e2_proj = e2_proj / e2_proj_norm
                    if e1_proj_norm > 1e-14:
                        n_ref = n - start_row - 1
                        score_e1_proj = window_score_fast(A_block, e1_proj, n_ref, state["prev_basis"], state["prev_s2"])
                        score_vhat1 = window_score_fast(A_block, V_hat[:, 0], n_ref, state["prev_basis"], state["prev_s2"])
                        print(f"approx score of v1 projection onto sketch+window space: {score_e1_proj: .4f}")
                        print(f"approx score of V_hat(:,1): {score_vhat1: .4f}")
                        print(f"reported score_new(1): {score_new[0]: .4f}")
                        print("V_hat(1,:)")
                        print(fmt_row(V_hat[0, :]))
                        print(f"should be: {e1_proj[0]:.5f}")
                        if V_hat.shape[1] >= 2 and e2_proj_norm > 1e-14:
                            score_e2_proj = window_score_fast(A_block, e2_proj, n_ref, state["prev_basis"], state["prev_s2"])
                            score_vhat2 = window_score_fast(A_block, V_hat[:, 1], n_ref, state["prev_basis"], state["prev_s2"])
                            print(f"approx score of v2 projection onto sketch+window space: {score_e2_proj: .4f}")
                            print(f"approx score of V_hat(:,2): {score_vhat2: .4f}")
                            print(f"reported score_new(2): {score_new[1]: .4f}")
                            print("V_hat(2,:)")
                            print(fmt_row(V_hat[1, :]))
                            print(f"should be: {e2_proj[1]:.5f}")
                        ll = min(l, V_r.shape[1])
                        align = np.linalg.norm(V_r @ (V_r.T @ V[:, :ll]), "fro") / np.sqrt(ll)
                        print(f"{align:.4f}")
                        print("Projection norms:")
                        print(fmt_row(np.linalg.norm(V_r @ (V_r.T @ np.column_stack([e1_proj, e2_proj])), axis=0)))
                        print(f"{window_score_fast(A_block, e1_proj, n_ref): .4f}")
                        print(f"{window_score_fast(A_block, e2_proj, n_ref): .4f}")
                        print(f"{window_score_fast(A_block, V_hat[:, 0], n_ref): .4f}")
                        print(f"{window_score_fast(A_block, V_hat[:, 1], n_ref): .4f}")

            ll = min(l, V_r.shape[1])
            align = np.linalg.norm(V_r @ (V_r.T @ V[:, :ll]), "fro") / np.sqrt(ll)
            top_sval_est = S_r[0, 0] if S_r is not None else 0.0
            rel_err_sval = abs(top_sval_est - sigma1) / sigma1
            if V_r is None:
                E_alg = np.linalg.norm(A, "fro") ** 2
            else:
                E_alg = np.linalg.norm(A - A @ V_r @ V_r.T, "fro") ** 2
            _Delta = E_alg - E_opt
            alignment_results[i, e] = align
            relerr_sval_results[i, e] = rel_err_sval
            low_sval_indicator[i, e] = float(top_sval_est <= 0.99)

    mean_align = np.mean(alignment_results, axis=1)
    std_align = np.std(alignment_results, axis=1, ddof=0)
    mean_relerr_sval = np.mean(relerr_sval_results, axis=1)
    std_relerr_sval = np.std(relerr_sval_results, axis=1, ddof=0)
    low_sval_count = np.sum(low_sval_indicator, axis=1)
    print("sigma1    mean_align    std_align    mean_relerr_sval    std_relerr_sval    count_sval_le_099_over_1")
    for vals in zip(first_svals, mean_align, std_align, mean_relerr_sval, std_relerr_sval, low_sval_count):
        print(f"{vals[0]:.3f}      {vals[1]:.5f}          {vals[2]:.5g}           {vals[3]:.7f}               {vals[4]:.5g}                      {int(vals[5])}")
    print(f"Elapsed time: {time.time() - t0:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Literal Python reproduction of cex_structured_new.m."
    )
    parser.add_argument(
        "--mat-input",
        default=None,
        help="Optional MATLAB-exported input file, e.g. matlab/cex1_input.mat.",
    )
    parser.add_argument(
        "--mode",
        default="Combined",
        choices=("Combined", "DefSVD"),
        help="Direction policy. Combined: optimizer-based (original). "
             "DefSVD: soft-thresholded augmented SVD with cumulative Δ² (C2).",
    )
    parser.add_argument(
        "--no-deflate-window",
        dest="deflate_window",
        action="store_false",
        help="DefSVD only. If set, leave the current window pristine and "
             "deflate only the carried sketch (FD-spirit, asymmetric).",
    )
    parser.set_defaults(deflate_window=True)
    args = parser.parse_args()
    run(mat_input=args.mat_input, mode=args.mode, deflate_window=args.deflate_window)
