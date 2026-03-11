import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

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

# residual mode:
#   "ws_reg_fro"    -> use whole_space_regular_residuals_fro from reservoir_residuals_data_*.npz
#   "ws_reg_2norm"  -> use whole_space_regular_residuals_2norm
#   "fallback_l2"   -> use ||approx_residuals||_2 from residuals_data_*.npz
#   "fallback_max"  -> use max(abs(approx_residuals)) from residuals_data_*.npz
RESIDUAL_MODE = "ws_reg_fro"

TRACE_RANK = 10
LEFT_OUT_LEGEND_THRESHOLD = 20


def sanitize(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


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


def parse_experiment_folder(folder_name):
    """
    Expected examples:
      bad_case1_1000_isvd_random_uniform_size_129_k_1
      bad_case1_1000_isvd_random_uniform_4_size_129_k_1
      bad_case1_1000_isvd_random_uniform_size_129_ssize_128_k_1_reservoir_greedy
      bad_case1_1000_isvd_random_uniform_2_size_129_ssize_121_k_8_reservoir_greedy
    """
    pattern = re.compile(
        r"^(?P<matrix>.+?)_"
        r"(?P<method>[^_]+)_random_uniform"
        r"(?:_(?P<seed>\d+))?"
        r"_size_(?P<size>\d+)"
        r"(?:_ssize_(?P<ssize>\d+))?"
        r"_k_(?P<k>\d+)"
        r"(?:_reservoir_(?P<reservoir>.+))?$"
    )
    m = pattern.match(folder_name)
    if not m:
        return None

    return {
        "matrix": m.group("matrix"),
        "method": m.group("method"),
        "seed": int(m.group("seed")) if m.group("seed") is not None else 1,
        "size": int(m.group("size")),
        "ssize": int(m.group("ssize")) if m.group("ssize") is not None else None,
        "k": int(m.group("k")),
        "reservoir": m.group("reservoir"),
    }


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

    return np.clip(np.asarray(curve), np.finfo(float).eps, None)


def load_residual_curve(exp_dir, mode="ws_reg_fro"):
    reservoir_iters = list_consecutive_iterations(exp_dir, "reservoir_residuals_data")

    if reservoir_iters and mode in {"ws_reg_fro", "ws_reg_2norm"}:
        field = (
            "whole_space_regular_residuals_fro"
            if mode == "ws_reg_fro"
            else "whole_space_regular_residuals_2norm"
        )
        curve = []
        for j in reservoir_iters:
            data = np.load(os.path.join(exp_dir, f"reservoir_residuals_data_{j}.npz"), allow_pickle=True)
            curve.append(float(data[field]))
        return np.clip(np.asarray(curve), np.finfo(float).eps, None), f"exact:{field}"

    residual_iters = list_consecutive_iterations(exp_dir, "residuals_data")
    if not residual_iters:
        return None, None

    curve = []
    for j in residual_iters:
        data = np.load(os.path.join(exp_dir, f"residuals_data_{j}.npz"), allow_pickle=True)
        r = np.asarray(data["approx_residuals"]).reshape(-1)

        if mode == "fallback_max":
            value = np.max(np.abs(r))
            source = "fallback:max(abs(approx_residuals))"
        else:
            value = np.linalg.norm(r, 2)
            source = "fallback:l2(approx_residuals)"

        curve.append(float(value))

    return np.clip(np.asarray(curve), np.finfo(float).eps, None), source


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

    current_totals = np.asarray(current_totals)
    current_throws = np.asarray(current_throws)

    current_totals = np.clip(np.abs(current_totals), np.finfo(float).eps, None)
    current_throws = np.clip(np.abs(current_throws), np.finfo(float).eps, None)
    return current_totals, current_throws


def get_component_colors(num_sv, cmap_name="viridis"):
    cmap = cm.get_cmap(cmap_name)
    norm = colors.Normalize(vmin=0, vmax=max(num_sv - 1, 1))
    color_list = [cmap(norm(i)) for i in range(num_sv)]
    return cmap, norm, color_list


def finalize_component_annotation(fig, ax, num_sv, use_legend, cmap, norm, prefix):
    if use_legend:
        ax.legend(ncol=2 if num_sv > 10 else 1, fontsize=9)
    else:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(f"{prefix} component index")
        if num_sv <= 50:
            cbar.set_ticks(np.arange(num_sv))


def plot_component_matrix(values, windows, ylabel, title, save_path, prefix_for_colorbar,
                          linestyle="-", marker="o", cmap_name="viridis"):
    num_sv = values.shape[1]
    fig, ax = plt.subplots(figsize=(12, 8))

    cmap, norm, color_list = get_component_colors(num_sv, cmap_name=cmap_name)
    use_legend = num_sv <= LEFT_OUT_LEGEND_THRESHOLD

    for i in range(num_sv):
        ax.semilogy(
            windows,
            values[:, i],
            color=color_list[i],
            linestyle=linestyle,
            marker=marker,
            label=f"{prefix_for_colorbar} #{i}" if use_legend else None,
            linewidth=1.2,
            markersize=4,
        )

    ax.set_xlabel("Window")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.tick_params(axis="x", labelrotation=45)

    finalize_component_annotation(fig, ax, num_sv, use_legend, cmap, norm, prefix_for_colorbar)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def choose_one_folder_for_k(candidates):
    """
    Prefer the unseeded folder (seed=1 naming), otherwise lowest seed.
    """
    candidates = sorted(candidates, key=lambda x: (x["seed"], x["folder"]))
    for c in candidates:
        if c["seed"] == 1:
            return c
    return candidates[0]


os.makedirs(FIG_DIR, exist_ok=True)

# collect folders
by_matrix_k = {matrix: {k: [] for k in K_VALUES} for matrix in MATRICES}

for folder in os.listdir(OUTPUT_DIR):
    parsed = parse_experiment_folder(folder)
    if parsed is None:
        continue
    if parsed["matrix"] not in MATRICES:
        continue
    if parsed["method"] != METHOD_NAME:
        continue
    if parsed["size"] != MEM_SIZE:
        continue
    if parsed["k"] not in K_VALUES:
        continue

    parsed["folder"] = folder
    by_matrix_k[parsed["matrix"]][parsed["k"]].append(parsed)

# make plots
for matrix_name in MATRICES:
    selected = {}

    for k in K_VALUES:
        candidates = by_matrix_k[matrix_name][k]
        if not candidates:
            print(f"Missing folder for matrix={matrix_name}, k={k}")
            continue
        selected[k] = choose_one_folder_for_k(candidates)

    if not selected:
        print(f"No experiments found for {matrix_name}")
        continue

    # -------- trace error comparison across k --------
    plt.figure(figsize=(12, 6))
    any_trace = False
    for k in K_VALUES:
        if k not in selected:
            continue
        folder = selected[k]["folder"]
        exp_dir = os.path.join(OUTPUT_DIR, folder)
        curve = load_trace_error_curve(exp_dir, rank_limit=TRACE_RANK)
        if curve is None:
            print(f"No spectrum_data for {folder}")
            continue

        x = np.arange(1, len(curve) + 1)
        plt.semilogy(x, curve, marker="o", label=f"k={k}")
        any_trace = True

    if any_trace:
        plt.xscale("log")
        plt.xlabel("Window index (log scale)")
        plt.ylabel("Relative trace error")
        plt.title(f"{matrix_name}: trace error across k")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(FIG_DIR, f"{sanitize(matrix_name)}_size_{MEM_SIZE}_trace_error_compare.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=220)
        print(f"Saved {save_path}")
    plt.close()

    # -------- residual comparison across k --------
    plt.figure(figsize=(12, 6))
    any_res = False
    residual_source_kind = None
    for k in K_VALUES:
        if k not in selected:
            continue
        folder = selected[k]["folder"]
        exp_dir = os.path.join(OUTPUT_DIR, folder)
        curve, source_kind = load_residual_curve(exp_dir, mode=RESIDUAL_MODE)
        if curve is None:
            print(f"No residual data for {folder}")
            continue

        x = np.arange(1, len(curve) + 1)
        plt.semilogy(x, curve, marker="o", label=f"k={k}")
        any_res = True
        if residual_source_kind is None:
            residual_source_kind = source_kind

    if any_res:
        plt.xscale("log")
        plt.xlabel("Window index (log scale)")
        plt.ylabel("Whole-space residual")
        plt.title(f"{matrix_name}: residual across k ({residual_source_kind})")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(FIG_DIR, f"{sanitize(matrix_name)}_size_{MEM_SIZE}_residual_compare.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=220)
        print(f"Saved {save_path}")
    plt.close()

    # -------- leftout plots per selected experiment --------
    for k in K_VALUES:
        if k not in selected:
            continue

        folder = selected[k]["folder"]
        seed = selected[k]["seed"]
        ssize = selected[k]["ssize"]
        exp_dir = os.path.join(OUTPUT_DIR, folder)

        current_totals, current_throws = load_leftout(exp_dir)
        if current_totals is None:
            print(f"No leftout data for {folder}")
            continue

        windows = np.arange(current_totals.shape[0])

        label_base = f"{matrix_name}, k={k}, seed={seed}"
        if ssize is not None:
            label_base += f", ssize={ssize}"

        save_total = os.path.join(FIG_DIR, sanitize(f"{folder}_current_total_log.png"))
        plot_component_matrix(
            values=current_totals,
            windows=windows,
            ylabel="Current total (log scale)",
            title=f"{label_base} current_total_log",
            save_path=save_total,
            prefix_for_colorbar="Total",
            linestyle="-",
            marker="o",
            cmap_name="viridis",
        )
        print(f"Saved {save_total}")

        save_throw = os.path.join(FIG_DIR, sanitize(f"{folder}_cumulative_throw_log.png"))
        plot_component_matrix(
            values=current_throws,
            windows=windows,
            ylabel="Cumulative throw (log scale)",
            title=f"{label_base} cumulative_throw_log",
            save_path=save_throw,
            prefix_for_colorbar="Throw",
            linestyle="--",
            marker="x",
            cmap_name="plasma",
        )
        print(f"Saved {save_throw}")