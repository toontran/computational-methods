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

print("\n================ SIGMA CONFIGURATION ================\n")

# Print the full sigma list
print("[INFO] Full sigma list (sorted):")
for idx, s in enumerate(sigma_all):
    print(f"   {idx:3d}: {s:.6f}")
print(f"\n[INFO] Total sigma count = {S}\n")

# Print this machine’s assignment
print(f"[INFO] Machine index {m} handling {len(assigned_sigma_indices)} sigmas out of {S} total.")
print("[INFO] Assigned sigma indices:")
print("   ", assigned_sigma_indices)

print("\n[INFO] Assigned sigma values:")
for k in assigned_sigma_indices:
    print(f"   idx {k:3d} → sigma = {sigma_all[k]:.6f}")

print("\n=====================================================\n")