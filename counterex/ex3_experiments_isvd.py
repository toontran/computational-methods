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
n   = 10000          # matrix size (n x n) — scaled-up version
r   = 1              # target rank
l   = 1              # number of true singular vectors to track in alignment
win = 100            # window / block size
num_exper = 100      # experiments per sigma

# Spectrum curvature parameters (same idea as before)
n_old     = 1024
alpha_old = 0.0145


# ============================
# Sigma construction
# ============================

def build_sigma_list():
    """
    Build the full sigma array (coarse + refinement),
    deduped and sorted ascending, following the MATLAB logic.
    """

    # --- Coarse grid over sigma1 (extended range) ---
    # first_svals = [0.991, 0.992, 0.995, 1.0:0.2:2.0, 2 * 2.^(1:5)];
    sigma_coarse = np.concatenate([
        np.array([0.991, 0.992, 0.995]),
        np.arange(1.0, 2.0 + 0.2 / 2, 0.2),   # 1.0, 1.2, ..., 2.0
        2.0 * (2.0 ** np.arange(1, 6)),       # 4, 8, 16, 32, 64
    ])
    sigma_coarse = np.unique(sigma_coarse)     # sorted

    # --- Refinement sigma: 1.01–1.04 step 0.005, plus your extra points ---
    sigma_refine = np.concatenate([
        np.arange(1.05, 1.1 + 0.005 / 2, 0.005),
    ])

    # Combine & dedupe & sort
    sigma_all = np.unique(np.concatenate([sigma_coarse, sigma_refine]))
    return sigma_all.astype(float)


# ============================
# Spectrum + U construction
# ============================

def build_base_spectrum():
    """
    Curvature-preserved base spectrum of length n, using the same
    scaling logic as the 1024→n_new construction.
    """
    t_new = np.linspace(0.0, 1.0, n)  # n points between 0 and 1
    base_svec = (1.0 + t_new * (n_old - 1)) ** (-alpha_old)
    base_svec /= base_svec[0]        # normalize so first entry = 1
    return base_svec


def build_U_hadamard_eye():
    """
    Build U with a 2x2 Hadamard block at the top-left and an identity
    block on R^{n-2}, following the MATLAB structure:

        p = 2
        Q1 = hadamard(2)/sqrt(2)
        Q2 = eye(n-p)
        U0 = blkdiag(Q1, Q2)
        move second column of U0 to the end
    """
    p = 2
    if p != 2:
        raise ValueError("This construction assumes p = 2 for the 2x2 Hadamard block.")

    # Q1: 2x2 Hadamard (normalized)
    H2 = np.array([[1.0,  1.0],
                   [1.0, -1.0]])
    Q1 = H2 / np.sqrt(2.0)   # same as hadamard(2)/sqrt(2)

    # Q2: identity on R^{n-p}
    Q2 = np.eye(n - p)

    # Block-diagonal U0
    U0 = np.zeros((n, n))
    U0[:p, :p] = Q1
    U0[p:, p:] = Q2

    # Move second column of U0 to the end
    # MATLAB (1-based):
    #   cols_to_move = 2;
    #   remain_cols  = setdiff(1:n, cols_to_move);
    #   new_order    = [remain_cols, cols_to_move];
    # Python (0-based): move column index 1
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
        raise SystemExit("Usage: python isvd_worker_hadamard_eye.py <machine_index>")

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
    U = build_U_hadamard_eye()

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
        print(f"   {idx:3d}: {s:.6f}")
    print(f"\n[INFO] Total sigma count = {S}\n")

    print(f"[INFO] Machine index {m} handling {len(assigned_sigma_indices)} "
          f"sigmas out of {S} total.")
    print("[INFO] Assigned sigma indices:")
    print("   ", assigned_sigma_indices)

    print("\n[INFO] Assigned sigma values:")
    for k in assigned_sigma_indices:
        print(f"   idx {k:3d} → sigma = {sigma_all[k]:.6f}")

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

        sigma1 = sigma_all[i]

        # Build spectrum svec with this sigma1
        svec = base_svec.copy()
        svec[0] = sigma1
        S_diag = svec  # diagonal entries of S

        # Optimal tail error and comparator
        E_opt = np.sum(S_diag[r:] ** 2)
        Delta_comp = np.sum(S_diag[:r] ** 2) - np.sum(S_diag[r:2 * r] ** 2)

        # Storage for experiments of this sigma
        alignment_results = np.zeros(num_exper)
        top_sval_results  = np.zeros(num_exper)
        Delta_results     = np.zeros(num_exper)
        low_sval_indicator = np.zeros(num_exper, dtype=int)

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
                    U_hat, s_hat, Vt_hat = np.linalg.svd(
                        A_block, full_matrices=False
                    )
                    S_r = np.diag(s_hat[:r])
                    V_r = Vt_hat[:r, :].T   # nA x r
                else:
                    # B = [ S_r V_r^T ; A_block ]
                    B_top = S_r @ V_r.T
                    B = np.vstack([B_top, A_block])

                    U_hat, s_hat, Vt_hat = np.linalg.svd(
                        B, full_matrices=False
                    )
                    S_r = np.diag(s_hat[:r])
                    V_r = Vt_hat[:r, :].T

            # ----- Metrics after full pass -----

            # Alignment: V = I so this is || first l rows of V_r ||_F / sqrt(l)
            Vr_first_rows = V_r[:l, :]  # shape (l x r)
            align = np.linalg.norm(Vr_first_rows, 'fro') / denom_align

            # Top singular value estimate
            top_sval = float(S_r[0, 0])

            # Error metric
            E_alg = np.linalg.norm(A - A @ V_r @ V_r.T, 'fro') ** 2
            Delta = E_alg - E_opt

            alignment_results[e] = align
            top_sval_results[e]  = top_sval
            Delta_results[e]     = Delta
            low_sval_indicator[e] = int(top_sval <= 0.99)

        # Summaries for this sigma
        mean_align = alignment_results.mean()
        std_align  = alignment_results.std(ddof=0)
        mean_sval  = top_sval_results.mean()
        std_sval   = top_sval_results.std(ddof=0)
        low_sval_count = int(low_sval_indicator.sum())

        rows_summary.append({
            "sigma1": sigma1,
            "mean_align": mean_align,
            "std_align": std_align,
            "mean_sval": mean_sval,
            "std_sval": std_sval,
            f"count_sval_le_099 (over {num_exper} total)": low_sval_count,
            "sigma_index": i,
            "machine_index": m,
            "Delta_comp": Delta_comp,
        })

    # -------------------------
    # Save per-machine summary
    # -------------------------
    summary_df = pd.DataFrame(rows_summary)
    summary_df = summary_df.sort_values("sigma1")

    outfile = f"summary_machine_ex3_{m:02d}.csv"
    summary_df.to_csv(outfile, index=False)
    print(f"[INFO] Saved summary for machine {m} to {outfile}")


if __name__ == "__main__":
    main()
