import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
FIG_DIR = "figures"

BASE_PREFIX = "kernel_stocks_5000_0.7071_isvd_random_uniform"
SIZE = 110
RESERVOIR_METHOD = "greedy"

def load_trace_error_curve(exp_dir):
    files = glob.glob(os.path.join(exp_dir, "spectrum_data_*.npz"))
    if not files:
        return None

    file_numbers = sorted(
        int(re.search(r"spectrum_data_(\d+)\.npz$", os.path.basename(f)).group(1))
        for f in files
    )

    # keep only consecutive files starting at 0, matching the original script logic
    last_consecutive = -1
    current = 0
    file_number_set = set(file_numbers)
    while current in file_number_set:
        last_consecutive = current
        current += 1

    if last_consecutive < 0:
        return None

    Ss = []
    S_exact = None

    for iteration in range(last_consecutive + 1):
        path = os.path.join(exp_dir, f"spectrum_data_{iteration}.npz")
        data = np.load(path, allow_pickle=True)
        Ss.append(data["S"].reshape(1, -1))
        S_exact = data["S_exact"]

    Ss = np.concatenate(Ss, axis=0)
    rank_limit = min(10, Ss.shape[1])
    tr_S = np.sum(Ss[:, :rank_limit], axis=1)
    denom = np.sum(S_exact[:rank_limit])

    e_i = np.abs(denom - tr_S) / denom
    e_i = np.clip(e_i, np.finfo(float).eps, None)
    return e_i

def parse_ssize_k(folder_name):
    m = re.search(r"_ssize_(\d+)_k_(\d+)_reservoir_", folder_name)
    if not m:
        return None
    ssize = int(m.group(1))
    k = int(m.group(2))
    return ssize, k

all_folders = [
    d for d in os.listdir(OUTPUT_DIR)
    if d.startswith(f"{BASE_PREFIX}_size_{SIZE}_ssize_")
    and d.endswith(f"_reservoir_{RESERVOIR_METHOD}")
]

selected = []
for folder in all_folders:
    parsed = parse_ssize_k(folder)
    if parsed is None:
        continue
    ssize, k = parsed
    if ssize + k == SIZE:
        selected.append((ssize, k, folder))

selected.sort(key=lambda x: x[0])  # sort by ssize

plt.figure(figsize=(12, 8))

for ssize, k, folder in selected:
    exp_dir = os.path.join(OUTPUT_DIR, folder)
    curve = load_trace_error_curve(exp_dir)
    if curve is None:
        print(f"Skipping {folder}: no usable spectrum_data_*.npz")
        continue

    x = np.arange(1, len(curve) + 1)   # avoid 0 for log scale

    plt.loglog(
        x,
        curve,
        marker="o",
        label=f"ssize={ssize}, k={k}"
    )

plt.xlabel("Window index")
plt.ylabel("Relative trace error")
plt.title(f"{BASE_PREFIX}, size={SIZE}, reservoir={RESERVOIR_METHOD}")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=9, ncol=2)
plt.tight_layout()

os.makedirs(FIG_DIR, exist_ok=True)
save_path = os.path.join(
    FIG_DIR,
    f"{BASE_PREFIX}_size_{SIZE}_all_ssize_k_sum_{SIZE}_error_over_time_log_compare.png"
)
plt.savefig(save_path, bbox_inches="tight", dpi=200)
plt.show()

print(f"Saved to: {save_path}")
print("Included folders:")
for ssize, k, folder in selected:
    print(f"  ssize={ssize:>3}, k={k:>3} -> {folder}")