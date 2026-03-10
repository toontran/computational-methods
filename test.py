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

# Choose which scalar whole-space curve to plot:
#   "ws_reg_2norm" -> whole_space_regular_residuals_2norm   (preferred if available)
#   "ws_reg_fro"   -> whole_space_regular_residuals_fro
#   "fallback_l2"  -> np.linalg.norm(approx_residuals, 2) from residuals_data_*.npz
#   "fallback_max" -> np.max(np.abs(approx_residuals)) from residuals_data_*.npz
MODE = "ws_reg_2norm"


def parse_ssize_k(folder_name):
    m = re.search(r"_ssize_(\d+)_k_(\d+)_reservoir_", folder_name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def list_consecutive_iterations(exp_dir, prefix):
    files = glob.glob(os.path.join(exp_dir, f"{prefix}_*.npz"))
    if not files:
        return []

    nums = []
    for f in files:
        m = re.search(rf"{re.escape(prefix)}_(\d+)\.npz$", os.path.basename(f))
        if m:
            nums.append(int(m.group(1)))
    nums = sorted(set(nums))

    consecutive = []
    j = 0
    numset = set(nums)
    while j in numset:
        consecutive.append(j)
        j += 1
    return consecutive


def load_wholespace_curve(exp_dir, mode="ws_reg_2norm"):
    # First try the exact whole-space residual files used by plot_wholespace_residual
    reservoir_iters = list_consecutive_iterations(exp_dir, "reservoir_residuals_data")

    if reservoir_iters and mode in {"ws_reg_2norm", "ws_reg_fro"}:
        curve = []
        field = (
            "whole_space_regular_residuals_2norm"
            if mode == "ws_reg_2norm"
            else "whole_space_regular_residuals_fro"
        )
        for j in reservoir_iters:
            data = np.load(os.path.join(exp_dir, f"reservoir_residuals_data_{j}.npz"), allow_pickle=True)
            curve.append(float(data[field]))
        return np.asarray(curve), f"exact:{field}"

    # Fallback: aggregate the saved per-vector residuals
    residual_iters = list_consecutive_iterations(exp_dir, "residuals_data")
    if not residual_iters:
        return None, None

    curve = []
    for j in residual_iters:
        data = np.load(os.path.join(exp_dir, f"residuals_data_{j}.npz"), allow_pickle=True)
        r = np.asarray(data["approx_residuals"]).reshape(-1)

        if mode == "fallback_max":
            value = np.max(np.abs(r))
        else:
            # default fallback
            value = np.linalg.norm(r, 2)

        curve.append(float(value))

    label = "fallback:max(approx_residuals)" if mode == "fallback_max" else "fallback:l2(approx_residuals)"
    return np.asarray(curve), label


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

selected.sort(key=lambda x: x[0])

plt.figure(figsize=(12, 8))
used_source_kind = None

for ssize, k, folder in selected:
    exp_dir = os.path.join(OUTPUT_DIR, folder)
    curve, source_kind = load_wholespace_curve(exp_dir, mode=MODE)
    if curve is None:
        print(f"Skipping {folder}: no usable residual files found")
        continue

    if used_source_kind is None:
        used_source_kind = source_kind

    curve = np.clip(curve, np.finfo(float).eps, None)

    plt.semilogy(
        np.arange(len(curve)),
        curve,
        marker="o",
        label=f"ssize={ssize}, k={k}"
    )

plt.xlabel("Window index")
plt.ylabel("Whole-space residual")
plt.title(f"{BASE_PREFIX}, size={SIZE}, reservoir={RESERVOIR_METHOD}\nsource={used_source_kind}")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()

os.makedirs(FIG_DIR, exist_ok=True)
save_path = os.path.join(
    FIG_DIR,
    f"{BASE_PREFIX}_size_{SIZE}_all_ssize_k_sum_{SIZE}_ws_residual_compare_{MODE}.png"
)
plt.savefig(save_path, bbox_inches="tight", dpi=200)
plt.show()

print(f"Saved to: {save_path}")
print("Included folders:")
for ssize, k, folder in selected:
    print(f"  ssize={ssize:>3}, k={k:>3} -> {folder}")