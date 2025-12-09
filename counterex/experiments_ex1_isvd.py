import sys
import numpy as np
import pandas as pd

# ------------- 1. Define sigma_all -------------
sigma_coarse = np.concatenate([
    np.array([0.991, 0.992, 0.995]),
    np.arange(1.0, 2.0 + 0.2/2, 0.2),   # 1.0, 1.2, ..., 2.0
    2.0 * (2.0 ** np.arange(1, 11)),    # 4, 8, ..., 2048
])

sigma_refine = np.array([
    1.04, 1.05, 1.06,
    1.10, 1.13, 1.15,
    1.056, 1.057, 1.058, 1.059,
])

sigma_all_list = list(dict.fromkeys(
    list(sigma_coarse) + list(sigma_refine)
))
sigma_all = np.array(sigma_all_list)
S = sigma_all.size

# experiments per sigma
num_exper = 100

# ------------- 2. Get machine index -------------
if len(sys.argv) != 2:
    raise SystemExit("Usage: python <script_name>.py <index>")

M = 22  # number of machines
m = int(sys.argv[1])
if not (0 <= m < M):
    raise SystemExit(f"index must be in [0, {M-1}], got {m}")

# ------------- 3. Sigma indices assigned to this machine -------------
assigned_sigma_indices = list(range(m, S, M))

print(f"[INFO] Machine index {m} handling {len(assigned_sigma_indices)} sigmas "
      f"out of {S} total.")