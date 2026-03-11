import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
FIG_DIR = "figures"

MATRIX_NAMES = [
    "kernel_stocks_1000_10.0",
    "kernel_stocks_1000_2.2361",
    "kernel_stocks_1000_0.7071",
    "kernel_stocks_1000_0.2236",
]

METHODS = ["isvd", "isvdstG"]

SIZE = 100
SSIZE = 1
K = 10
RESERVOIR_METHOD = "greedy"

# Residual source:
#   "ws_reg_2norm" -> use whole_space_regular_residuals_2norm from reservoir_residuals_data_*.npz
#   "fallback_l2"  -> use ||approx_residuals||_2 from residuals_data_*.npz
#   "fallback_max" -> use max(abs(approx_residuals)) from residuals_data_*.npz
RESIDUAL_MODE = "ws_reg_2norm"

TRACE_RANK = 10
SHOW_BAND = True


def parse_folder(folder_name):
    """
    Examples:
      kernel_stocks_1000_0.2236_isvd_random_uniform_size_100_ssize_1_k_10_reservoir_greedy
      kernel_stocks_1000_0.2236_isvdstG_random_uniform_4_size_100_ssize_1_k_10_reservoir_greedy
    """
    pattern = re.compile(
        r"^(?P<matrix>kernel_stocks_\d+_[^_]+)_(?P<method>isvd(?:stG)?)_random_uniform"
        r"(?:_(?P<seed>\d+))?"
        r"_size_(?P<size>\d+)"
        r"_ssize_(?P<ssize>\d+)"
        r"_k_(?P<k>\d+)"
        r"_reservoir_(?P<reservoir>.+)$"
    )
    m = pattern.match(folder_name)
    if not m:
        return None

    return {
        "matrix": m.group("matrix"),
        "method": m.group("method"),
        "seed": int(m.group("seed")) if m.group("seed") is not None else 1,
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


def load_residual_curve(exp_dir, mode="ws_reg_2norm"):
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
            val = np.max(np.abs(r))
            source = "fallback:max(abs(approx_residuals))"
        else:
            val = np.linalg.norm(r, 2)
            source = "fallback:l2(approx_residuals)"

        curve.append(float(val))

    return np.asarray(curve), source


def load_trace_error_curve(exp_dir, rank_limit=10):
    iters = list_consecutive_iterations(exp_dir, "spectrum_data")
    if not iters:
        return None

    Ss = []
    S_exact = None

    for j in iters:
        data = np.load(
            os.path.join(exp_dir, f"spectrum_data_{j}.npz"),
            allow_pickle=True
        )
        S = np.asarray(data["S"]).reshape(-1)
        S_exact = np.asarray(data["S_exact"]).reshape(-1)
        Ss.append(S)

    min_rank = min(min(len(S) for S in Ss), len(S_exact), rank_limit)
    if min_rank <= 0:
        return None

    denom = np.sum(S_exact[:min_rank])
    if denom == 0:
        return None

    curve = []
    for S in Ss:
        tr_S = np.sum(S[:min_rank])
        e = np.abs(denom - tr_S) / np.abs(denom)
        curve.append(float(e))

    return np.asarray(curve)


def aggregate_seed_curves(curves):
    min_len = min(len(c) for c in curves)
    arr = np.stack([c[:min_len] for c in curves], axis=0)
    mean_curve = arr.mean(axis=0)
    low_curve = arr.min(axis=0)
    high_curve = arr.max(axis=0)
    mean_endpoint = arr[:, -1].mean()
    return mean_curve, low_curve, high_curve, mean_endpoint, min_len


# Collect matching folders by (matrix, method)
groups = {}
for folder in os.listdir(OUTPUT_DIR):
    parsed = parse_folder(folder)
    if parsed is None:
        continue
    if parsed["matrix"] not in MATRIX_NAMES:
        continue
    if parsed["method"] not in METHODS:
        continue
    if parsed["size"] != SIZE:
        continue
    if parsed["ssize"] != SSIZE:
        continue
    if parsed["k"] != K:
        continue
    if parsed["reservoir"] != RESERVOIR_METHOD:
        continue

    key = (parsed["matrix"], parsed["method"])
    groups.setdefault(key, []).append((parsed["seed"], folder))

for key in groups:
    groups[key].sort(key=lambda x: x[0])

fig, axes = plt.subplots(len(MATRIX_NAMES), 2, figsize=(16, 5 * len(MATRIX_NAMES)), squeeze=False)

residual_source_kind = None

for row, matrix_name in enumerate(MATRIX_NAMES):
    ax_res = axes[row, 0]
    ax_tr = axes[row, 1]

    for method in METHODS:
        key = (matrix_name, method)
        seed_folders = groups.get(key, [])

        residual_curves = []
        trace_curves = []
        residual_seeds = []
        trace_seeds = []

        for seed, folder in seed_folders:
            exp_dir = os.path.join(OUTPUT_DIR, folder)

            rcurve, source_kind = load_residual_curve(exp_dir, mode=RESIDUAL_MODE)
            if rcurve is not None:
                rcurve = np.clip(rcurve, np.finfo(float).eps, None)
                residual_curves.append(rcurve)
                residual_seeds.append(seed)
                if residual_source_kind is None:
                    residual_source_kind = source_kind

            tcurve = load_trace_error_curve(exp_dir, rank_limit=TRACE_RANK)
            if tcurve is not None:
                tcurve = np.clip(tcurve, np.finfo(float).eps, None)
                trace_curves.append(tcurve)
                trace_seeds.append(seed)

        # Residual panel
        if residual_curves:
            mean_curve, low_curve, high_curve, mean_endpoint, npts = aggregate_seed_curves(residual_curves)
            x = np.arange(1, npts + 1)
            label = f"{method}, end={mean_endpoint:.3e}, n={len(residual_curves)}"
            line, = ax_res.semilogy(x, mean_curve, marker="o", linewidth=1.8, label=label)
            if SHOW_BAND and len(residual_curves) > 1:
                ax_res.fill_between(x, low_curve, high_curve, alpha=0.18, color=line.get_color())

            print(f"{matrix_name} | residual | {method} | seeds={residual_seeds} | mean end={mean_endpoint:.6e}")

        # Trace panel
        if trace_curves:
            mean_curve, low_curve, high_curve, mean_endpoint, npts = aggregate_seed_curves(trace_curves)
            x = np.arange(1, npts + 1)
            label = f"{method}, end={mean_endpoint:.3e}, n={len(trace_curves)}"
            line, = ax_tr.semilogy(x, mean_curve, marker="o", linewidth=1.8, label=label)
            if SHOW_BAND and len(trace_curves) > 1:
                ax_tr.fill_between(x, low_curve, high_curve, alpha=0.18, color=line.get_color())

            print(f"{matrix_name} | trace    | {method} | seeds={trace_seeds} | mean end={mean_endpoint:.6e}")

    # format residual axis
    ax_res.set_xscale("log")
    ax_res.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_res.set_xlabel("Window index (log scale)")
    ax_res.set_ylabel("Whole-space residual")
    ax_res.set_title(f"{matrix_name}: residual\nsize={SIZE}, ssize={SSIZE}, k={K}, source={residual_source_kind}")
    ax_res.legend(fontsize=9)

    # format trace axis
    ax_tr.set_xscale("log")
    ax_tr.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_tr.set_xlabel("Window index (log scale)")
    ax_tr.set_ylabel("Relative trace error")
    ax_tr.set_title(f"{matrix_name}: trace error\nsize={SIZE}, ssize={SSIZE}, k={K}, top-{TRACE_RANK} trace")
    ax_tr.legend(fontsize=9)

plt.tight_layout()
os.makedirs(FIG_DIR, exist_ok=True)

save_path = os.path.join(
    FIG_DIR,
    f"kernel_stocks_1000_compare_isvd_vs_isvdstG_size_{SIZE}_ssize_{SSIZE}_k_{K}_residual_and_trace.png"
)
plt.savefig(save_path, bbox_inches="tight", dpi=220)
plt.show()

print(f"\nSaved to: {save_path}")