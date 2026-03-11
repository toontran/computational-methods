import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

OUTPUT_DIR = "output"
FIG_DIR = "figures"

BASE_PREFIX = "kernel_stocks_1000_0.7071_isvd"
SIZE = 110
RESERVOIR_METHOD = "greedy"
LEGEND_THRESHOLD = 20


def parse_folder(folder_name):
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


def sanitize(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def get_component_colors(num_sv, cmap_name="viridis"):
    cmap = cm.get_cmap(cmap_name)
    norm = colors.Normalize(vmin=0, vmax=max(num_sv - 1, 1))
    color_list = [cmap(norm(i)) for i in range(num_sv)]
    return cmap, norm, color_list


def finalize_component_annotation(fig, ax, num_sv, color_mode, cmap, norm, prefix):
    if color_mode == "legend":
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
    values = np.clip(np.abs(values), np.finfo(float).eps, None)
    num_sv = values.shape[1]

    fig, ax = plt.subplots(figsize=(12, 8))

    cmap, norm, color_list = get_component_colors(num_sv, cmap_name=cmap_name)
    use_legend = num_sv <= LEGEND_THRESHOLD

    for i in range(num_sv):
        label = f"{prefix_for_colorbar} #{i}" if use_legend else None
        ax.semilogy(
            windows,
            values[:, i],
            color=color_list[i],
            linestyle=linestyle,
            marker=marker,
            label=label,
            linewidth=1.2,
            markersize=4,
        )

    ax.set_xlabel("Window")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.tick_params(axis="x", labelrotation=45)

    finalize_component_annotation(
        fig,
        ax,
        num_sv,
        "legend" if use_legend else "colorbar",
        cmap,
        norm,
        prefix_for_colorbar,
    )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


os.makedirs(FIG_DIR, exist_ok=True)

folders = []
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
    folders.append((parsed["ssize"], parsed["k"], parsed["seed"], folder))

folders.sort(key=lambda x: (x[0], x[1], x[2]))

for ssize, k, seed, folder in folders:
    exp_dir = os.path.join(OUTPUT_DIR, folder)
    current_totals, current_throws = load_leftout(exp_dir)

    if current_totals is None:
        print(f"Skipping {folder}: no leftout_data_*.npz found")
        continue

    windows = np.arange(current_totals.shape[0])

    label_base = f"ssize={ssize}, k={k}, seed={seed}"

    save_total = os.path.join(
        FIG_DIR,
        sanitize(f"{folder}_current_total_log.png")
    )
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

    save_throw = os.path.join(
        FIG_DIR,
        sanitize(f"{folder}_cumulative_throw_log.png")
    )
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

    print(f"Saved:")
    print(f"  {save_total}")
    print(f"  {save_throw}")