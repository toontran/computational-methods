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

# Spectrum curvature parameters (same idea as MATLAB code)
n_old     = 1024
alpha_old = 0.0145


def build_sigma_list():
    """Build the full sigma array (coarse + refine, deduped, order preserved)."""
    sigma_coarse = np.concatenate([
        np.array([0.991, 0.992, 0.995]),
        np.arange(1.0, 2.0 + 0.2 / 2, 0.2),   # 1.0, 1.2, ..., 2.0
        2.0 * (2.0 ** np.arange(1, 5)),      # 4, 8, 16, 32
    ])

    # Refinement list(s)
    a = np.arange(1.0, 1.1 + 1e-12, 0.01)

    # NOTE: using YOUR list (unchanged). MATLAB will handle its own list anyway.
    b = np.array([
        1.04, 1.042, 44, 46, 48, 50, 52, 54, 56, 58, 1.06
    ], dtype=float)

    # Combine, remove duplicates, sort (fine; MATLAB will transform/plot)
    sigma_refine = np.unique(np.concatenate([a, b]))

    # Combine, preserve order, drop duplicates
    sigma_all_list = list(dict.fromkeys(
        list(sigma_coarse) + list(sigma_refine)
    ))
    sigma_all = np.array(sigma_all_list, dtype=float)
    return sigma_all


def build_base_spectrum():
    """Curvature-preserved base spectrum of length n."""
    t_new = np.linspace(0.0, 1.0, n)  # n points between 0 and 1
    base_svec = (1.0 + t_new * (n_old - 1)) ** (-alpha_old)
    base_svec /= base_svec[0]        # normalize so first entry = 1
    return base_svec


def build_U(n, r_struct):
    """
    Build U once (random unit vectors for first r_struct columns,
    flat columns for the rest), then orthogonalize (QR).
    """
    U0 = np.zeros((n, n), dtype=float)

    # First r_struct columns: random unit vectors
    G = np.random.randn(n, r_struct)
    G /= np.linalg.norm(G, axis=0, keepdims=True)
    U0[:, :r_struct] = G

    # Remaining columns: flat = ones/sqrt(n)
    flat_col = np.ones((n, 1)) / np.sqrt(n)
    U0[:, r_struct:] = flat_col

    # QR to orthonormalize
    U, _ = np.linalg.qr(U0, mode="reduced")  # U is n x n

    # Align signs of the first r_struct columns
    for j in range(r_struct):
        if np.dot(U[:, j], U0[:, j]) < 0:
            U[:, j] *= -1.0

    return U


def main():
    # -------------------------
    # Parse arguments
    # -------------------------
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python isvd_worker.py <machine_index>")

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
    U = build_U(n, r_struct=r)

    # denominator in alignment metric (V = I, so this is sqrt(l))
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
        sigma1 = float(sigma_all[i])

        # Build spectrum svec with this sigma1
        svec = base_svec.copy()
        svec[0] = sigma1
        S_diag = svec  # diagonal entries of S

        # Optimal tail error and comparator
        E_opt = float(np.sum(S_diag[r:] ** 2))
        Delta_comp = float(np.sum(S_diag[:r] ** 2) - np.sum(S_diag[r:2 * r] ** 2))

        # Storage for experiments of this sigma
        alignment_results     = np.zeros(num_exper)
        top_sval_results      = np.zeros(num_exper)
        relerr_sval_results   = np.zeros(num_exper)     # NEW (matches MATLAB plot 2)
        Delta_results         = np.zeros(num_exper)
        low_sval_indicator    = np.zeros(num_exper, dtype=int)

        # A = U * diag(S_diag) (column-wise scaling)
        A_template = U * S_diag  # broadcasting: each column scaled

        for e in range(num_exper):
            # Random row permutation per experiment
            p = np.random.permutation(n)
            A = A_template[p, :]     # permute rows
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

            # Alignment:
            # MATLAB: align = ||V_r(:,1:l)' * V(:,1:l)||_F / denom_align with V = I.
            # That product just picks the first l rows of V_r.
            Vr_first_rows = V_r[:l, :]  # shape (l x r)
            align = float(np.linalg.norm(Vr_first_rows, 'fro') / denom_align)

            # Top singular value estimate
            top_sval = float(S_r[0, 0])

            # Relative error of retrieved top singular value (matches MATLAB)
            rel_err_sval = float(abs(top_sval - sigma1) / sigma1)

            # Error metric (optional)
            E_alg = float(np.linalg.norm(A - A @ V_r @ V_r.T, 'fro') ** 2)
            Delta = float(E_alg - E_opt)

            alignment_results[e]   = align
            top_sval_results[e]    = top_sval
            relerr_sval_results[e] = rel_err_sval
            Delta_results[e]       = Delta
            low_sval_indicator[e]  = int(top_sval <= 0.99)

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

            # Extra (often useful in analysis)
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
    summary_df = pd.DataFrame(rows_summary)
    summary_df = summary_df.sort_values("sigma1")

    # Keep your original naming (if you want)
    outfile = f"summary_machine_{m:02d}.csv"
    summary_df.to_csv(outfile, index=False)
    print(f"[INFO] Saved summary for machine {m} to {outfile}")

    # Explicit plot-ready file (same content, just clearer intent)
    plotfile = f"plotdata_machine_{m:02d}.csv"
    summary_df.to_csv(plotfile, index=False)
    print(f"[INFO] Saved plot data for machine {m} to {plotfile}")


if __name__ == "__main__":
    main()
    