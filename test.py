import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
FIG_DIR = "figures"

METHOD_NAME = "isvd"
MEM_SIZE = 129
MATRICES = [
    "bad_case1_1000",
    "bad_case2_1000",
    "bad_case3_1000",
]
K_VALUES = [1, 8, 64, 128]
RESERVOIR_METHOD = "greedy"

TRACE_RANK = 10
SHOW_BAND = True

# Residual fallback mode from residuals_data_*.npz
# "l2" or "max"
RESIDUAL_MODE = "l2"


def parse_folder(folder_name):
    """
    Expected examples:
      bad_case1_1000_isvd_random_uniform_size_129_ssize_128_k_1_reservoir_greedy
      bad_case1_1000_isvd_random_uniform_4_size_129_ssize_128_k_1_reservoir_greedy
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

    return {
        "prefix": m.group("prefix"),
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


def load_trace_error_curve(exp_dir, rank_limit=10):
    iters = list_consecutive_iterations(exp_dir, "spectrum_data")
    if not iters:
        return None

    Ss = []
    S_exact = None

    for j in iters:
        data = np.load(os.path.join(exp_dir, f"spectrum_data_{j}.npz"), allow_pickle=True)
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


def load_residual_curve(exp_dir, mode="l2"):
    """
    Uses residuals_data_*.npz saved by save_residuals(...) in non-symmetric mode.
    """
    iters = list_consecutive_iterations(exp_dir, "residuals_data")
    if not iters:
        return None

    curve = []
    for j in iters:
        data = np.load(os.path.join(exp_dir, f"residuals_data_{j}.npz"), allow_pickle=True)
        r = np.asarray(data["approx_residuals"]).reshape(-1)

        if mode == "max":
            val = np.max(np.abs(r))
        else:
            val = np.linalg.norm(r, 2)

        curve.append(float(val))

    return np.asarray(curve)


def load_leftout(exp_dir):
    iters = list_consecutive_iterations(exp_dir, "leftout_data")
    if not iters:
        return None, None

    current_totals = []
    current_throws = []
    cumulative_throw = None

    for j in iters:
        data = np.load(os.path.join(exp_dir, f"leftout_data_{j}.npz"), allow_pickle=True)

        current_total = np.asarray(data["current_total"]).reshape(-1)
        throw = np.asarray(data["throw"]).reshape(-1)

        if cumulative_throw is None:
            cumulative_throw = np.zeros_like(throw, dtype=float)

        cumulative_throw = cumulative_throw + throw
        current_totals.append(current_total.astype(float))
        current_throws.append(cumulative_throw.copy())

    return np.asarray(current_totals), np.asarray(current_throws)


def aggregate_seed_curves(curves):
    min_len = min(len(c) for c in curves)
    arr = np.stack([c[:min_len] for c in curves], axis=0)
    mean_curve = arr.mean(axis=0)
    low_curve = arr.min(axis=0)
    high_curve = arr.max(axis=0)
    mean_endpoint = arr[:, -1].mean()
    return mean_curve, low_curve, high_curve, mean_endpoint, min_len


def sanitize(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def get_colors(num_sv):
    color_range = np.linspace(0, 1.0, num_sv)
    if num_sv > 2:
        color_range[2] = (color_range[-1] + color_range[-2]) / 2
    color_range = np.sort(color_range)
    return plt.cm.jet(color_range)


def plot_leftout_for_folder(folder, exp_dir):
    current_totals, current_throws = load_leftout(exp_dir)
    if current_totals is None:
        print(f"Skipping leftout for {folder}: no leftout_data_*.npz")
        return

    current_totals = np.clip(np.abs(current_totals), np.finfo(float).eps, None)
    current_throws = np.clip(np.abs(current_throws), np.finfo(float).eps, None)

    num_sv = current_totals.shape[1]
    windows = np.arange(1, current_totals.shape[0] + 1)
    colors = get_colors(num_sv)

    # current_total log
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(num_sv):
        ax.semilogy(
            windows,
            current_totals[:, i],
            color=colors[i],
            linestyle='-',
            marker='o',
            label=f"Total #{i}"
        )
    ax.set_xlabel("Window")
    ax.set_ylabel("Current total (log scale)")
    ax.set_title(f"{folder} current_total_log")
    ax.grid(True, which="both")
    ax.legend()
    ax.tick_params(axis='x', labelrotation=45)
    fig.tight_layout()

    save_total = os.path.join(FIG_DIR, sanitize(f"{folder}_current_total_log.png"))
    fig.savefig(save_total, bbox_inches="tight", dpi=200)
    plt.close(fig)

    # cumulative throw log
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(num_sv):
        ax.semilogy(
            windows,
            current_throws[:, i],
            color=colors[i],
            linestyle='--',
            marker='x',
            label=f"Throw #{i}"
        )
    ax.set_xlabel("Window")
    ax.set_ylabel("Cumulative throw (log scale)")
    ax.set_title(f"{folder} cumulative_throw_log")
    ax.grid(True, which="both")
    ax.legend()
    ax.tick_params(axis='x', labelrotation=45)
    fig.tight_layout()

    save_throw = os.path.join(FIG_DIR, sanitize(f"{folder}_cumulative_throw_log.png"))
    fig.savefig(save_throw, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"Saved leftout plots for {folder}")


os.makedirs(FIG_DIR, exist_ok=True)

for matrix_name in MATRICES:
    base_prefix = f"{matrix_name}_{METHOD_NAME}"

    # Collect folders for this matrix
    groups = {}  # (ssize, k) -> list of (seed, folder)

    for folder in os.listdir(OUTPUT_DIR):
        parsed = parse_folder(folder)
        if parsed is None:
            continue
        if parsed["prefix"] != base_prefix:
            continue
        if parsed["size"] != MEM_SIZE:
            continue
        if parsed["reservoir"] != RESERVOIR_METHOD:
            continue
        if parsed["k"] not in K_VALUES:
            continue
        if parsed["ssize"] != MEM_SIZE - parsed["k"]:
            continue

        key = (parsed["ssize"], parsed["k"])
        groups.setdefault(key, []).append((parsed["seed"], folder))

    for key in groups:
        groups[key].sort(key=lambda x: x[0])

    sorted_keys = sorted(groups.keys(), key=lambda x: x[1])  # sort by k

    # -------- residual + trace figure for this matrix --------
    fig, (ax_res, ax_tr) = plt.subplots(1, 2, figsize=(16, 7))
    used_any_res = False
    used_any_tr = False

    for ssize, k in sorted_keys:
        seed_folders = groups[(ssize, k)]

        residual_curves = []
        trace_curves = []
        used_folders = []

        for seed, folder in seed_folders:
            exp_dir = os.path.join(OUTPUT_DIR, folder)

            rcurve = load_residual_curve(exp_dir, mode=RESIDUAL_MODE)
            if rcurve is not None:
                residual_curves.append(np.clip(rcurve, np.finfo(float).eps, None))

            tcurve = load_trace_error_curve(exp_dir, rank_limit=TRACE_RANK)
            if tcurve is not None:
                trace_curves.append(np.clip(tcurve, np.finfo(float).eps, None))

            # also make per-experiment leftout plots
            plot_leftout_for_folder(folder, exp_dir)
            used_folders.append(folder)

        if residual_curves:
            mean_curve, low_curve, high_curve, mean_endpoint, npts = aggregate_seed_curves(residual_curves)
            x = np.arange(1, npts + 1)
            label = f"k={k}, ssize={ssize}, end={mean_endpoint:.3e}, n={len(residual_curves)}"
            line, = ax_res.semilogy(x, mean_curve, marker="o", linewidth=1.5, label=label)
            if SHOW_BAND and len(residual_curves) > 1:
                ax_res.fill_between(x, low_curve, high_curve, alpha=0.18, color=line.get_color())
            used_any_res = True

        if trace_curves:
            mean_curve, low_curve, high_curve, mean_endpoint, npts = aggregate_seed_curves(trace_curves)
            x = np.arange(1, npts + 1)
            label = f"k={k}, ssize={ssize}, end={mean_endpoint:.3e}, n={len(trace_curves)}"
            line, = ax_tr.semilogy(x, mean_curve, marker="o", linewidth=1.5, label=label)
            if SHOW_BAND and len(trace_curves) > 1:
                ax_tr.fill_between(x, low_curve, high_curve, alpha=0.18, color=line.get_color())
            used_any_tr = True

    for ax in (ax_res, ax_tr):
        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.set_xlabel("Window index (log scale)")

    ax_res.set_ylabel("Residual")
    ax_res.set_title(f"{matrix_name}: residual across k")

    ax_tr.set_ylabel("Relative trace error")
    ax_tr.set_title(f"{matrix_name}: trace error across k")

    if used_any_res:
        ax_res.legend(fontsize=8)
    if used_any_tr:
        ax_tr.legend(fontsize=8)

    plt.tight_layout()
    save_compare = os.path.join(
        FIG_DIR,
        f"{matrix_name}_{METHOD_NAME}_size_{MEM_SIZE}_trace_and_residual_compare.png"
    )
    plt.savefig(save_compare, bbox_inches="tight", dpi=220)
    plt.close(fig)

    print(f"Saved comparison figure: {save_compare}")