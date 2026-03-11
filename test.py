import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

OUTPUT_DIR = "output"
FIG_DIR = "figures"

# Fixed experiment settings
WINDOW_SIZE = 1
K = 8

MATRICES = [
    # "bad_case1_1000",
    # "bad_case2_1000",
    # "bad_case3_1000",
    "kernel_stocks_1000_10.0"
    "kernel_stocks_1000_2.2361"
    "kernel_stocks_1000_0.7071"
    "kernel_stocks_1000_0.2236"
]

METHODS = [
    "isvd",
    "isvdstG",
]

# Trace error settings
TRACE_RANK = 10

# Residual mode:
#   "ws_reg_fro"    -> whole_space_regular_residuals_fro from reservoir_residuals_data_*.npz
#   "ws_reg_2norm"  -> whole_space_regular_residuals_2norm from reservoir_residuals_data_*.npz
#   "fallback_l2"   -> ||approx_residuals||_2 from residuals_data_*.npz
#   "fallback_max"  -> max(abs(approx_residuals)) from residuals_data_*.npz
RESIDUAL_MODE = "ws_reg_fro"

# Plotting behavior
SHOW_BAND = True
LEFTOUT_LEGEND_THRESHOLD = 20


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def list_consecutive_iterations(exp_dir: str, prefix: str):
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


def parse_experiment_folder(folder_name: str):
    """
    Expected examples:
      bad_case1_1000_isvd_random_uniform_size_1_k_8
      bad_case1_1000_isvd_random_uniform_4_size_1_k_8
      bad_case1_1000_isvd_size_1_k_8
      bad_case1_1000_isvdstG_random_uniform_size_1_ssize_121_k_8_reservoir_greedy
    """
    patterns = [
        re.compile(
            r"^(?P<matrix>.+?)_"
            r"(?P<method>[^_]+)_random_uniform"
            r"(?:_(?P<seed>\d+))?"
            r"_size_(?P<size>\d+)"
            r"(?:_ssize_(?P<ssize>\d+))?"
            r"_k_(?P<k>\d+)"
            r"(?:_reservoir_(?P<reservoir>.+))?$"
        ),
        re.compile(
            r"^(?P<matrix>.+?)_"
            r"(?P<method>[^_]+)"
            r"_size_(?P<size>\d+)"
            r"(?:_ssize_(?P<ssize>\d+))?"
            r"_k_(?P<k>\d+)"
            r"(?:_reservoir_(?P<reservoir>.+))?$"
        ),
    ]

    for pattern in patterns:
        m = pattern.match(folder_name)
        if m:
            seed = m.groupdict().get("seed")
            ssize = m.groupdict().get("ssize")
            reservoir = m.groupdict().get("reservoir")
            return {
                "matrix": m.group("matrix"),
                "method": m.group("method"),
                "seed": int(seed) if seed is not None else 1,
                "size": int(m.group("size")),
                "ssize": int(ssize) if ssize is not None else None,
                "k": int(m.group("k")),
                "reservoir": reservoir,
                "folder": folder_name,
                "is_random_uniform": "_random_uniform" in folder_name,
            }

    return None


def load_trace_error_curve(exp_dir: str, rank_limit: int = 10):
    iters = list_consecutive_iterations(exp_dir, "spectrum_data")
    if not iters:
        return None

    Ss = []
    S_exact = None

    for j in iters:
        path = os.path.join(exp_dir, f"spectrum_data_{j}.npz")
        data = np.load(path, allow_pickle=True)
        S = np.asarray(data["S"]).reshape(-1)
        S_exact = np.asarray(data["S_exact"]).reshape(-1)
        Ss.append(S)

    if S_exact is None:
        return None

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


def load_residual_curve(exp_dir: str, mode: str = "ws_reg_fro"):
    reservoir_iters = list_consecutive_iterations(exp_dir, "reservoir_residuals_data")

    if reservoir_iters and mode in {"ws_reg_fro", "ws_reg_2norm"}:
        field = (
            "whole_space_regular_residuals_fro"
            if mode == "ws_reg_fro"
            else "whole_space_regular_residuals_2norm"
        )
        curve = []
        for j in reservoir_iters:
            path = os.path.join(exp_dir, f"reservoir_residuals_data_{j}.npz")
            data = np.load(path, allow_pickle=True)
            curve.append(float(data[field]))
        return np.clip(np.asarray(curve), np.finfo(float).eps, None), f"exact:{field}"

    residual_iters = list_consecutive_iterations(exp_dir, "residuals_data")
    if not residual_iters:
        return None, None

    curve = []
    source = None
    for j in residual_iters:
        path = os.path.join(exp_dir, f"residuals_data_{j}.npz")
        data = np.load(path, allow_pickle=True)
        r = np.asarray(data["approx_residuals"]).reshape(-1)

        if mode == "fallback_max":
            value = np.max(np.abs(r))
            source = "fallback:max(abs(approx_residuals))"
        else:
            value = np.linalg.norm(r, 2)
            source = "fallback:l2(approx_residuals)"

        curve.append(float(value))

    return np.clip(np.asarray(curve), np.finfo(float).eps, None), source


def load_leftout(exp_dir: str):
    iters = list_consecutive_iterations(exp_dir, "leftout_data")
    if not iters:
        return None, None

    current_totals = []
    current_throws = []
    cumulative_throw = None

    for j in iters:
        path = os.path.join(exp_dir, f"leftout_data_{j}.npz")
        data = np.load(path, allow_pickle=True)

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


def aggregate_seed_curves(curves):
    min_len = min(len(c) for c in curves)
    arr = np.stack([c[:min_len] for c in curves], axis=0)
    mean_curve = arr.mean(axis=0)
    low_curve = arr.min(axis=0)
    high_curve = arr.max(axis=0)
    mean_endpoint = arr[:, -1].mean()
    return mean_curve, low_curve, high_curve, mean_endpoint, min_len


def get_component_colors(num_sv: int, cmap_name: str = "viridis"):
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


def plot_component_matrix(
    values,
    windows,
    ylabel,
    title,
    save_path,
    prefix_for_colorbar,
    linestyle="-",
    marker="o",
    cmap_name="viridis",
):
    num_sv = values.shape[1]
    fig, ax = plt.subplots(figsize=(12, 8))

    cmap, norm, color_list = get_component_colors(num_sv, cmap_name=cmap_name)
    use_legend = num_sv <= LEFTOUT_LEGEND_THRESHOLD

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
    fig.savefig(save_path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def collect_experiments():
    by_matrix_method = {
        matrix: {method: [] for method in METHODS}
        for matrix in MATRICES
    }

    for folder in os.listdir(OUTPUT_DIR):
        parsed = parse_experiment_folder(folder)
        if parsed is None:
            continue
        if parsed["matrix"] not in MATRICES:
            continue
        if parsed["method"] not in METHODS:
            continue
        if parsed["size"] != WINDOW_SIZE:
            continue
        if parsed["k"] != K:
            continue

        by_matrix_method[parsed["matrix"]][parsed["method"]].append(parsed)

    for matrix_name in MATRICES:
        for method in METHODS:
            by_matrix_method[matrix_name][method].sort(
                key=lambda x: (x["seed"], x["folder"])
            )

    return by_matrix_method


def plot_trace_error_for_matrix(matrix_name: str, groups):
    plt.figure(figsize=(12, 6))
    any_trace = False

    for method in METHODS:
        candidates = groups[matrix_name][method]
        if not candidates:
            print(f"Missing experiments for matrix={matrix_name}, method={method}")
            continue

        curves = []
        seeds_used = []

        for item in candidates:
            exp_dir = os.path.join(OUTPUT_DIR, item["folder"])
            curve = load_trace_error_curve(exp_dir, rank_limit=TRACE_RANK)
            if curve is None:
                print(f"No spectrum data for {item['folder']}")
                continue
            curves.append(curve)
            seeds_used.append(item["seed"])

        if not curves:
            continue

        mean_curve, low_curve, high_curve, mean_endpoint, npts = aggregate_seed_curves(curves)
        x = np.arange(1, npts + 1)

        label = f"{method}, end={mean_endpoint:.3e}, n={len(curves)}"
        line, = plt.semilogy(x, mean_curve, marker="o", linewidth=1.5, label=label)
        if SHOW_BAND and len(curves) > 1:
            plt.fill_between(x, low_curve, high_curve, alpha=0.18, color=line.get_color())

        print(f"Trace  {matrix_name}, method={method}, seeds={seeds_used}, mean endpoint={mean_endpoint:.6e}")
        any_trace = True

    if any_trace:
        plt.xscale("log")
        plt.xlabel("Window index (log scale)")
        plt.ylabel("Relative trace error")
        plt.title(f"{matrix_name}: trace error across methods (mean over seeds)\nwindow_size={WINDOW_SIZE}, k={K}")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(
            FIG_DIR,
            f"{sanitize(matrix_name)}_wsize_{WINDOW_SIZE}_k_{K}_trace_error_compare_methods_allseeds.png"
        )
        plt.savefig(save_path, bbox_inches="tight", dpi=220)
        print(f"Saved {save_path}")

    plt.close()


def plot_residual_for_matrix(matrix_name: str, groups):
    plt.figure(figsize=(12, 6))
    any_res = False
    residual_source_kind = None

    for method in METHODS:
        candidates = groups[matrix_name][method]
        if not candidates:
            print(f"Missing experiments for matrix={matrix_name}, method={method}")
            continue

        curves = []
        seeds_used = []

        for item in candidates:
            exp_dir = os.path.join(OUTPUT_DIR, item["folder"])
            curve, source_kind = load_residual_curve(exp_dir, mode=RESIDUAL_MODE)
            if curve is None:
                print(f"No residual data for {item['folder']}")
                continue
            curves.append(curve)
            seeds_used.append(item["seed"])
            if residual_source_kind is None:
                residual_source_kind = source_kind

        if not curves:
            continue

        mean_curve, low_curve, high_curve, mean_endpoint, npts = aggregate_seed_curves(curves)
        x = np.arange(1, npts + 1)

        label = f"{method}, end={mean_endpoint:.3e}, n={len(curves)}"
        line, = plt.semilogy(x, mean_curve, marker="o", linewidth=1.5, label=label)
        if SHOW_BAND and len(curves) > 1:
            plt.fill_between(x, low_curve, high_curve, alpha=0.18, color=line.get_color())

        print(f"Residual {matrix_name}, method={method}, seeds={seeds_used}, mean endpoint={mean_endpoint:.6e}")
        any_res = True

    if any_res:
        plt.xscale("log")
        plt.xlabel("Window index (log scale)")
        plt.ylabel("Whole-space residual")
        plt.title(
            f"{matrix_name}: residual across methods (mean over seeds, {residual_source_kind})\n"
            f"window_size={WINDOW_SIZE}, k={K}"
        )
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(
            FIG_DIR,
            f"{sanitize(matrix_name)}_wsize_{WINDOW_SIZE}_k_{K}_residual_compare_methods_allseeds.png"
        )
        plt.savefig(save_path, bbox_inches="tight", dpi=220)
        print(f"Saved {save_path}")

    plt.close()


def plot_leftout_for_all_experiments(groups):
    used_folders = set()

    for matrix_name in MATRICES:
        for method in METHODS:
            candidates = groups[matrix_name][method]
            if not candidates:
                continue

            for item in candidates:
                folder = item["folder"]
                if folder in used_folders:
                    continue
                used_folders.add(folder)

                exp_dir = os.path.join(OUTPUT_DIR, folder)
                current_totals, current_throws = load_leftout(exp_dir)
                if current_totals is None:
                    print(f"No leftout data for {folder}")
                    continue

                windows = np.arange(current_totals.shape[0])
                label_base = f"{matrix_name}, method={method}, seed={item['seed']}, window_size={WINDOW_SIZE}, k={K}"
                if item["ssize"] is not None:
                    label_base += f", ssize={item['ssize']}"

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


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    groups = collect_experiments()

    for matrix_name in MATRICES:
        plot_trace_error_for_matrix(matrix_name, groups)
        plot_residual_for_matrix(matrix_name, groups)

    plot_leftout_for_all_experiments(groups)


if __name__ == "__main__":
    main()
