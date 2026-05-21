import numpy as np

from cex_structured_new_py import (
    basic_projected_ascent_single_exact,
    load_matlab_cex_input,
    make_basic_restart_seeds,
    window_score_grad_rows,
)
from utils import (
    basic_projected_ascent_single_reduced_forget_cex,
    basic_projected_ascent_single_reduced_forget,
    entropyscore_forget_logscore_grad_reduced,
)


def run():
    np.random.seed(0)
    A, _, _, _ = load_matlab_cex_input("matlab/cex1_input.mat")
    A_block = A[:100, :]
    n = A.shape[0]
    r = 2
    num_restarts = 8
    maxit = 200
    tol = 1e-8

    _, _, Vh = np.linalg.svd(A_block.astype(float), full_matrices=False)
    Vbasis = Vh.T.astype(float, copy=False)
    B = A_block.astype(float, copy=False) @ Vbasis

    Q_full = np.zeros((A.shape[1], 0))
    Qz = np.zeros((Vbasis.shape[1], 0), dtype=float)
    V_full_out = []
    V_reduced_out = []
    V_cex_reduced_out = []

    print("First block optimizer comparison on same full row-space basis")
    print(f"A_block={A_block.shape}, Vbasis={Vbasis.shape}, restarts={num_restarts}, maxit={maxit}")

    for kk in range(1, r + 1):
        full_starts = make_basic_restart_seeds(A_block, Q_full, kk, None, num_restarts)

        best_full = None
        for v0 in full_starts:
            cand = basic_projected_ascent_single_exact(A_block, v0, Q_full, n, maxit, tol)
            if best_full is None or cand[1] > best_full[1]:
                best_full = cand
        v_full, score_full, s2_full, H_full = best_full

        reduced_starts = []
        for v0 in full_starts:
            z0 = Vbasis.T @ v0
            z0 = z0.astype(float, copy=False)
            nz = np.linalg.norm(z0)
            if nz > 1e-14:
                reduced_starts.append(z0 / nz)

        best_old_red = None
        best_cex_red = None
        for z0 in reduced_starts:
            cand_old = basic_projected_ascent_single_reduced_forget(
                B, z0, Qz, A_block.shape[0], n, maxit=maxit, tol=tol
            )
            if best_old_red is None or cand_old[1] > best_old_red[1]:
                best_old_red = cand_old

            cand_cex = basic_projected_ascent_single_reduced_forget_cex(
                B, z0, Qz, A_block.shape[0], n, maxit=maxit, tol=tol
            )
            if best_cex_red is None or cand_cex[1] > best_cex_red[1]:
                best_cex_red = cand_cex

        z_old, logf_old, s_old, H_old, stop_old = best_old_red
        v_old_red = Vbasis @ z_old
        score_old_red, _, _, _ = window_score_grad_rows(A_block, v_old_red, n)
        _, _, s_old_check, _ = entropyscore_forget_logscore_grad_reduced(
            B, z_old, A_block.shape[0], n
        )

        z_cex, score_cex_red, s_cex, H_cex, stop_cex = best_cex_red
        v_cex_red = Vbasis @ z_cex
        score_cex_check, _, _, _ = window_score_grad_rows(A_block, v_cex_red, n)

        print(f"\ncomponent {kk}")
        print(f"cex full:        score={score_full:.10f}, s={np.sqrt(s2_full):.10f}, H={H_full:.10f}")
        print(
            "old reduced:     "
            f"score={score_old_red:.10f}, exp(logf)={np.exp(logf_old):.10f}, "
            f"s={s_old:.10f}, s_check={s_old_check:.10f}, H={H_old:.10f}, stop={stop_old}"
        )
        print(
            "cex reduced:     "
            f"score={score_cex_check:.10f}, score_opt={score_cex_red:.10f}, "
            f"s={s_cex:.10f}, H={H_cex:.10f}, stop={stop_cex}"
        )
        print(f"alignment |v_full.T v_old_red|={abs(float(v_full @ v_old_red)):.10f}")
        print(f"alignment |v_full.T v_cex_red|={abs(float(v_full @ v_cex_red)):.10f}")

        Q_full = np.column_stack([Q_full, v_full])
        V_full_out.append(v_full)
        V_reduced_out.append(v_old_red)
        V_cex_reduced_out.append(v_cex_red)
        Qz = Vbasis.T @ np.column_stack(V_full_out)
        qz_q, _ = np.linalg.qr(Qz)
        Qz = qz_q.astype(float, copy=False)


if __name__ == "__main__":
    run()
