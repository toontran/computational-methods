import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================
# 1. Basic parameters
# ============================
n   = 10000   # scaled up
r   = 1       # target rank
l   = 1       # number of true singular vectors to track in alignment
win = 100     # window / block size

# Optional: fix seed if you want reproducibility
# np.random.seed(0)

# ============================
# 2. Curvature-preserved spectrum
#    n_new = 10000, based on old n_old = 1024
# ============================
n_old     = 1024
alpha_old = 0.0145

t_new = np.linspace(0.0, 1.0, n)          # n_new points
base_svec = (1.0 + t_new * (n_old - 1)) ** (-alpha_old)
base_svec = base_svec / base_svec[0]      # normalize so base_svec[0] = 1

# Values for the first singular value sigma1
# first_svals = np.concatenate([
#     np.array([0.991, 0.992, 0.995]),
#     np.arange(1.0, 2.0 + 0.2 / 2, 0.2),      # 1.0, 1.2, ..., 2.0
#     2.0 * (2.0 ** np.arange(1, 5))          # 2 * 2^(1:5)
# ])

# first_svals = np.concatenate([
#     np.array([0.991, 1.0, 1.2, 1.4]),
# ])
first_svals = np.arange(0.9900, 0.9910, 0.0001)[1:]
num_svals = first_svals.size
num_exper = 10
print("Spectrum built")

# ============================
# 3. Build U once (random + flat columns)
# ============================
r_struct = r

# Allocate
U0 = np.zeros((n, n))

# First r columns: random unit vectors
G = np.random.randn(n, r_struct)
G /= np.linalg.norm(G, axis=0, keepdims=True)
U0[:, :r_struct] = G

# Columns r+1 .. n: flat profile = ones/sqrt(n)
flat_col = np.ones((n, 1)) / np.sqrt(n)
U0[:, r_struct:] = flat_col

# Orthogonalize via QR to get an orthonormal basis U
# (equivalent to MATLAB [Q, ~] = qr(U0, 0) when U0 is square)
print("Before qr", U0.shape)
U, _ = np.linalg.qr(U0, mode='reduced')
print("After qr")

# Align signs of the first r columns
for j in range(r_struct):
    if np.dot(U[:, j], U0[:, j]) < 0:
        U[:, j] *= -1.0

print("U built")
# No need to store full V = I_n; we only need its effect
# but we keep the same alignment formula logically.

# ============================
# 4. Allocate storage for results
# ============================
alignment_results = np.zeros((num_svals, num_exper))
top_sval_results  = np.zeros((num_svals, num_exper))
Delta_results     = np.zeros((num_svals, num_exper))
DeltaComp_results = np.zeros((num_svals, num_exper))

# Indicator for S_r(1,1) <= 0.99
low_sval_indicator = np.zeros((num_svals, num_exper), dtype=int)

# Precompute denominator in alignment metric
# denom_align = ||V(:,1:l)' * V(:,1:l)||_F, with V = I_n  ->  ||I_l||_F = sqrt(l)
denom_align = np.sqrt(l)

print("Starting experiments...")

# ============================
# 5. Outer loop over sigma1 values
# ============================
for i, sigma1 in enumerate(tqdm(first_svals, desc="Outer loop (sigma1)", unit="σ")):
    # Build spectrum with this sigma1, using curvature-preserved base_svec
    svec = base_svec.copy()
    svec[0] = sigma1
    S_diag = svec  # store only diagonal

    # Optimal tail error and comparator Delta_comp for this spectrum
    E_opt = np.sum(S_diag[r:] ** 2)
    Delta_comp = np.sum(S_diag[:r] ** 2) - np.sum(S_diag[r:2 * r] ** 2)
    DeltaComp_results[i, :] = Delta_comp  # same across experiments

    # Inner loop: multiple experiments with different row permutations
    for e in range(num_exper):
        # Construct A and apply random row permutation
        # A = U * S * V' with V = I_n => A = U * diag(S_diag)
        A = U * S_diag  # column-wise scaling via broadcasting
        p = np.random.permutation(n)
        A = A[p, :]

        mA, nA = A.shape

        # Initialize streaming SVD state
        U_r = None
        S_r = None
        V_r = None
        rows_seen = 0

        # Streaming over row blocks
        for start_row in range(0, mA, win):
            end_row = min(start_row + win, mA)
            A_block = A[start_row:end_row, :]
            b_blk = A_block.shape[0]

            if V_r is None:
                # First block: standard SVD and truncate to rank r
                U_hat, s_hat, Vt_hat = np.linalg.svd(A_block, full_matrices=False)
                S_r = np.diag(s_hat[:r])
                V_r = Vt_hat[:r, :].T   # (nA x r)
                rows_seen = b_blk
            else:
                # Stack:
                # [  S_r V_r^T ]   (r x nA)
                # [   A_block  ]   (b_blk x nA)
                B_top = S_r @ V_r.T
                B = np.vstack([B_top, A_block])

                U_hat, s_hat, Vt_hat = np.linalg.svd(B, full_matrices=False)
                S_r = np.diag(s_hat[:r])
                V_r = Vt_hat[:r, :].T
                rows_seen += b_blk

        # ============================
        # 6. Metrics after full pass
        # ============================

        # Alignment with top-l true right singular vectors.
        # In MATLAB: align = ||V_r(:,1:l)' * V(:,1:l)||_F / denom_align, with V = I_n.
        # V(:,1:l) are standard basis vectors e1..el, so
        # V_r(:,1:l)' * V(:,1:l) just picks the first l rows of V_r.
        Vr_first_rows = V_r[:l, :]          # (l x r)
        align = np.linalg.norm(Vr_first_rows, 'fro') / denom_align

        # Final top singular value estimate
        top_sval = S_r[0, 0]

        # Error metrics
        E_alg = np.linalg.norm(A - A @ V_r @ V_r.T, 'fro') ** 2
        Delta = E_alg - E_opt

        # Store
        alignment_results[i, e] = align
        top_sval_results[i, e]  = top_sval
        Delta_results[i, e]     = Delta

        # Indicator for top singular value collapse
        low_sval_indicator[i, e] = int(top_sval <= 0.99)

# ============================
# 7. Summaries
# ============================
mean_align = alignment_results.mean(axis=1)
std_align  = alignment_results.std(axis=1, ddof=0)

mean_sval = top_sval_results.mean(axis=1)
std_sval  = top_sval_results.std(axis=1, ddof=0)

low_sval_count    = low_sval_indicator.sum(axis=1)
low_sval_fraction = low_sval_count / num_exper   # kept for completeness

col_name_count = f"count_sval_le_099 (over {num_exper} total)"

summary_df = pd.DataFrame({
    "sigma1": first_svals,
    "mean_align": mean_align,
    "std_align": std_align,
    "mean_sval": mean_sval,
    "std_sval": std_sval,
    col_name_count: low_sval_count
})

print(summary_df.to_string(index=False))
