import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
FIG_DIR = "figures"

BASE_PREFIX = "kernel_stocks_1000_0.7071_isvd"
SIZE = 110
RESERVOIR_METHOD = "greedy"

# Preferred:
#   "ws_reg_2norm"  -> from reservoir_residuals_data_*.npz
# Fallbacks:
#   "fallback_l2"   -> l2 norm of approx_residuals from residuals_data_*.npz
#   "fallback_max"  -> max abs entry of approx_residuals from residuals_data_*.npz
MODE = "ws_reg_2norm"

# Whether to draw variability band across seeds
SHOW_BAND = True


def parse_folder(folder_name):
    """
    Parse folders like:
      kernel_stocks_1000_0.7071_isvd_random_uniform_size_110_ssize_109_k_1_reservoir_greedy
      kernel_stocks_1000_0.7071_isvd_random_uniform_4_size_110_ssize_109_k_1_reservoir_greedy
    """
    pattern = re.compile(
        r"^(?P<prefix>.+?)_random_uniform"
        r"(?:_(?P<seed>\d+))?"
        r"_size_(?P<size>\d+)"
        r"_ssize_(?P<ssize>\d+)"
        r"_k_(?P<k>\d+)"
        r"_reservoir_(?P<reservoir>.+)$"
    )
    m = pattern.match(folder_name)
    if not m:
        return None

    seed = int(m.group("seed")) if m.group("seed") is not None else 1
    return {
        "prefix": m.group("prefix"),
        "seed": seed,
        "size": int(m.group("size")),
        "ssize": int(m.group("ssize")),
        "k": int(m.group("k")),
        "reservoir": m.group("reservoir"),
    }


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
    out = []
    j = 0
    numset = set(nums)
    while j in numset:
        out.append(j)
        j += 1
    return out


def load_wholespace_curve(exp_dir, mode="ws_reg_2norm"):
    # Exact whole-space residual files, if present
    reservoir_iters = list_consecutive_iterations(exp_dir, "reservoir_residuals_data")
    if reservoir_iters and mode == "ws_reg_2norm":
        curve = []
        for j in reservoir_iters:
            data = np.load(
                os.path.join(exp_dir, f"reservoir_residuals_data_{j}.npz"),
                allow_pickle=True
            )
            curve.append(float(data["whole_space_regular_residuals_2norm"]))
        return np.asarray(curve), "exact:whole_space_regular_residuals_2norm"

    # Fallback to residuals_data_*.npz
    residual_iters = list_consecutive_iterations(exp_dir, "residuals_data")
    if not residual_iters:
        return None, None

    curve = []
    for j in residual_iters:
        data = np.load(
            os.path.join(exp_dir, f"residuals_data_{j}.npz"),
            allow_pickle=True
        )
        r = np.asarray(data["approx_residuals"]).reshape(-1)

        if mode == "fallback_max":
            value = np.max(np.abs(r))
        else:
            value = np.linalg.norm(r, 2)

        curve.append(float(value))

    source = "fallback:max(approx_residuals)" if mode == "fallback_max" else "fallback:l2(approx_residuals)"
    return np.asarray(curve), source


# Collect matching folders
groups = {}  # (ssize, k) -> list of (seed, folder_name)

for folder in os.listdir(OUTPUT_DIR):
    parsed = parse_folder(folder)
    if parsed is None:
        continue

    if parsed["prefix"] != BASE_PREFIX:
        continue
    if parsed["size"] != SIZE:
        continue
    if parsed["reservoir"] != RESERVOIR_METHOD:
        continue
    if parsed["ssize"] + parsed["k"] != SIZE:
        continue

    key = (parsed["ssize"], parsed["k"])
    groups.setdefault(key, []).append((parsed["seed"], folder))

# Sort each group by seed
for key in groups:
    groups[key].sort(key=lambda x: x[0])

# Plot one averaged curve per (ssize, k)
plt.figure(figsize=(13, 8))
source_kind_used = None

sorted_keys = sorted(groups.keys(), key=lambda x: x[0])  # sort by ssize

for ssize, k in sorted_keys:
    seed_folders = groups[(ssize, k)]

    curves = []
    seeds_found = []

    for seed, folder in seed_folders:
        exp_dir = os.path.join(OUTPUT_DIR, folder)
        curve, source_kind = load_wholespace_curve(exp_dir, mode=MODE)
        if curve is None:
            print(f"Skipping seed {seed} for {folder}: no usable residual files")
            continue

        curve = np.clip(curve, np.finfo(float).eps, None)
        curves.append(curve)
        seeds_found.append(seed)

        if source_kind_used is None:
            source_kind_used = source_kind

    if not curves:
        continue

    min_len = min(len(c) for c in curves)
    arr = np.stack([c[:min_len] for c in curves], axis=0)

    mean_curve = arr.mean(axis=0)
    lower_curve = arr.min(axis=0)
    upper_curve = arr.max(axis=0)

    x = np.arange(min_len)
    label = f"ssize={ssize}, k={k} (n={len(curves)})"

    line, = plt.semilogy(x, mean_curve, marker="o", linewidth=1.5, label=label)

    if SHOW_BAND and len(curves) > 1:
        plt.fill_between(x, lower_curve, upper_curve, alpha=0.18, color=line.get_color())

    print(f"(ssize={ssize}, k={k}) seeds used: {seeds_found}")

plt.xlabel("Window index")
plt.ylabel("Whole-space residual")
plt.title(
    f"{BASE_PREFIX}_random_uniform[*], size={SIZE}, reservoir={RESERVOIR_METHOD}\n"
    f"mean across seeds, source={source_kind_used}"
)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()

os.makedirs(FIG_DIR, exist_ok=True)
save_path = os.path.join(
    FIG_DIR,
    f"{BASE_PREFIX}_random_uniform_allseeds_size_{SIZE}_all_ssize_k_sum_{SIZE}_ws_residual_compare_{MODE}.png"
)
plt.savefig(save_path, bbox_inches="tight", dpi=200)
# plt.show()

print(f"Saved to: {save_path}")