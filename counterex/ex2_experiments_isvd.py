#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================
# CONFIG
# ============================

# Cluster / scheduler config
NUM_MACHINES = 22    # total number of machine indices (0..21)

# Experiment parameters
n   = 10000          # matrix size (n x n)
r   = 1              # target rank
l   = 1              # number of true singular vectors to track in alignment
win = 100            # window / block size
num_exper = 100      # experiments per sigma

# Spectrum curvature parameters (same idea as earlier 1024→n scaling)
n_old     = 1024
alpha_old = 0.0145


# ============================
# Sigma construction (USE MATLAB LIST)
# ============================

def build_sigma_list():
    """
    Build sigma list exactly from the MATLAB explicit list, then unique+sorted.
    """
    first_svals = np.array([
        0.991, 0.992, 0.995,
        1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
        4, 8, 16, 32, 64,
        1.010, 1.015, 1.020, 1.025, 1.030, 1.035, 1.040,
        1.010, 1.015, 1.020, 1.025, 1.030, 1.035, 1.040,
        1.050, 1.056, 1.057, 1.058, 1.059, 1.060, 1.068500,
        1.06851955645752,
        1.06851955646610,
        1.069, 1.100, 1.130, 1.150
    ], dtype=float)

    # MATLAB: first_svals = unique(first_svals, 'sorted');
    sigma_all = np.unique(first_svals)
    return sigma_all.astype(float)


# ============================
# Spectrum + U construction
# ============================

def build_base_spectrum():
    """
    Curvature-preserved base spectrum of length n, using the same
    scaling logic as your 1024→n_new construction.
    """
    t_new = np.linspace(0.0, 1.0, n)  # n points between 0 and 1
    base_svec = (1.0 + t_new * (n_old - 1)) ** (-alpha_old)
    base_svec /= base_svec[0]        # normalize so first entry = 1
    return base_svec


def build_U_hadamard_block():
    """
    Build U with a 2x2 Hadamard block at the top-left, plus a random
    orthogonal block on R^{n-2}, then move column 2 to the end.

    Matches MATLAB:
        p = 2
        Q1 = hadamard(p)/sqrt(p)
        Q2 = qr(randn(n-p,n-p),0)
        U0 = blkdiag(Q1,Q2)
        k = p-r (=1)
        cols_to_move = 2:k+1 (=2)
        new_order = [remain_cols, cols_to_move]
    """
    p = 2
    if p != 2:
        raise ValueError("This construction assumes p = 2 for the 2x2 Hadamard block.")

    # Q1: 2x2 Hadamard (normalized)
    H2 = np.array([[1.0,  1.0],
                   [1.0, -1.0]], dtype=float)
    Q1 = H2 / np.sqrt(2.0)

    # Q2: random orthogonal on R^{n-p}
    Q2_rand = np.random.randn(n - p, n - p)
    Q2, _ = np.linalg.qr(Q2_rand, mode="reduced")

    # Block-diagonal U0 = blkdiag(Q1, Q2)
    U0 = np.zeros((n, n), dtype=float)
    U0[:p, :p] = Q1
    U0[p:, p:] = Q2

    # Move second column of U0 to the end
    # MATLAB 1-based: move col 2; Python 0-based: move col 1
    cols_to_move = [1]
    remain_cols = [j for j in range(n) if j not in cols_to_move]
    new_order = remain_cols + cols_to_move

    U = U0[:, new_order]
    return U


# ============================
# Main experiment logic
# ============================

def main():
    # -------------------------
    # Parse arguments
    # -------------------------
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python isvd_worker_hadamard.py <machine_index>")

    m = int(sys.argv[1])
    if not (0 <= m < NUM_MACHINES):
        raise SystemExit(
            f"machine_index must be in [0, {NUM_MACHINES - 1}], got {m}"
        )

    # -------------------------
    # Build global objects
    # -------------------------
    sigma_all = build_sigma_list()
    S = sigma_all.size

    base_svec = build_base_spectrum()
    U = build_U_hadamard_block()

    # V is identity; denom_align = ||V(:,1:l)' * V(:,1:l)||_F = sqrt(l)
    denom_align = np.sqrt(l)

    # -------------------------
    # Assign sigmas to this machine (strided)
    # -------------------------
    assigned_sigma_indices = list(range(m, S, NUM_MACHINES))

    # -------------------------
    # Print sigma config for debugging
    # -------------------------
    print("\n================ SIGMA CONFIGURATION ================\n")

    print("[INFO] Full sigma list (index → value):")
    for idx, s in enumerate(sigma_all):
        print(f"   {idx:3d}: {s:.15g}")
    print(f"\n[INFO] Total sigma count = {S}\n")

    print(f"[INFO] Machine index {m} handling {len(assigned_sigma_indices)} "
          f"sigmas out of {S} total.")
    print("[INFO] Assigned sigma indices:")
    print("   ", assigned_sigma_indices)

    print("\n[INFO] Assigned sigma values:")
    for k in assigned_sigma_indices:
        print(f"   idx {k:3d} → sigma = {sigma_all[k]:.15g}")

    print("\n=====================================================\n")

    if not assigned_sigma_indices:
        print("[WARN] No sigmas assigned to this machine, exiting.")
        return

    # -------------------------
    # Main loop over assigned sigmas
    # -------------------------
    rows_summary = []

    for i in tqdm(assigned_sigma_indices,
                  desc=f"Machine {m} outer loop (sigma1)", unit="σ"):

        sigma1 = float(sigma_all[i])

        # Build spectrum svec with this sigma1
        svec = base_svec.copy()
        svec[0] = sigma1
        S_diag = svec  # diagonal entries of S

        # Optimal tail error and comparator (as in MATLAB)
        E_opt = float(np.sum(S_diag[r:] ** 2))
        Delta_comp = float(np.sum(S_diag[:r] ** 2) - np.sum(S_diag[r:2 * r] ** 2))

        # Storage for experiments of this sigma
        alignment_results     = np.zeros(num_exper)
        top_sval_results      = np.zeros(num_exper)
        relerr_sval_results   = np.zeros(num_exper)   # NEW: plot 2 (relative error)
        Delta_results         = np.zeros(num_exper)
        low_sval_indicator    = np.zeros(num_exper, dtype=int)

        # A = U * diag(S_diag) (column-wise scaling)
        A_template = U * S_diag  # broadcasting: each column scaled

        for e in range(num_exper):
            # Random row permutation per experiment
            p_rows = np.random.permutation(n)
            A = A_template[p_rows, :]
            mA, nA = A.shape

            # Streaming SVD state
            S_r = None
            V_r = None

            # Stream over row blocks
            for start_row in range(0, mA, win):
                end_row = min(start_row + win, mA)
                A_block = A[start_row:end_row, :]

                if V_r is None:
                    # First block: normal SVD, truncate rank r
                    _, s_hat, Vt_hat = np.linalg.svd(
                        A_block, full_matrices=False
                    )
                    S_r = np.diag(s_hat[:r])
                    V_r = Vt_hat[:r, :].T   # nA x r
                else:
                    # B = [ S_r V_r^T ; A_block ]
                    B_top = S_r @ V_r.T
                    B = np.vstack([B_top, A_block])

                    _, s_hat, Vt_hat = np.linalg.svd(
                        B, full_matrices=False
                    )
                    S_r = np.diag(s_hat[:r])
                    V_r = Vt_hat[:r, :].T

            # ----- Metrics after full pass -----

            # Alignment: V = I so this is || first l rows of V_r ||_F / sqrt(l)
            Vr_first_rows = V_r[:l, :]  # shape (l x r)
            align = float(np.linalg.norm(Vr_first_rows, 'fro') / denom_align)

            # Final top singular value estimate (raw)
            top_sval_est = float(S_r[0, 0])

            # Relative error (matches MATLAB: abs(top_sval_est - sigma1)/sigma1)
            rel_err_sval = float(abs(top_sval_est - sigma1) / sigma1)

            # Error metric
            E_alg = float(np.linalg.norm(A - A @ V_r @ V_r.T, 'fro') ** 2)
            Delta = float(E_alg - E_opt)

            # Store
            alignment_results[e]   = align
            top_sval_results[e]    = top_sval_est
            relerr_sval_results[e] = rel_err_sval
            Delta_results[e]       = Delta
            low_sval_indicator[e]  = int(top_sval_est <= 0.99)

        # Summaries for this sigma (plot-ready stats)
        mean_align = float(alignment_results.mean())
        std_align  = float(alignment_results.std(ddof=0))

        mean_sval = float(top_sval_results.mean())
        std_sval  = float(top_sval_results.std(ddof=0))

        mean_relerr_sval = float(relerr_sval_results.mean())
        std_relerr_sval  = float(relerr_sval_results.std(ddof=0))

        mean_Delta = float(Delta_results.mean())
        std_Delta  = float(Delta_results.std(ddof=0))

        low_sval_count = int(low_sval_indicator.sum())

        rows_summary.append({
            "sigma1": sigma1,

            # Plot 1
            "mean_align": mean_align,
            "std_align": std_align,

            # Plot 2
            "mean_relerr_sval": mean_relerr_sval,
            "std_relerr_sval": std_relerr_sval,

            # Plot 3
            "low_sval_count": low_sval_count,
            "num_exper": int(num_exper),

            # Extra (often useful later)
            "mean_sval": mean_sval,
            "std_sval": std_sval,
            "E_opt": E_opt,
            "Delta_comp": Delta_comp,
            "mean_Delta": mean_Delta,
            "std_Delta": std_Delta,

            "sigma_index": int(i),
            "machine_index": int(m),
        })

    # -------------------------
    # Save per-machine summaries
    # -------------------------
    summary_df = pd.DataFrame(rows_summary).sort_values("sigma1")

    outfile = f"summary_machine_ex2_{m:02d}.csv"
    summary_df.to_csv(outfile, index=False)
    print(f"[INFO] Saved summary for machine {m} to {outfile}")

    plotfile = f"plotdata_machine_ex2_{m:02d}.csv"
    summary_df.to_csv(plotfile, index=False)
    print(f"[INFO] Saved plot data for machine {m} to {plotfile}")


if __name__ == "__main__":
    main()
