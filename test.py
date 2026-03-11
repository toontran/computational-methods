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
        data = np.load(
            os.path.join(exp_dir, f"leftout_data_{j}.npz"),
            allow_pickle=True
        )

        current_total = np.asarray(data["current_total"]).reshape(-1)
        throw = np.asarray(data["throw"]).reshape(-1)

        if cumulative_throw is None:
            cumulative_throw = np.zeros_like(throw, dtype=float)

        cumulative_throw = cumulative_throw + throw
        current_totals.append(current_total.astype(float))
        current_throws.append(cumulative_throw.copy())

    current_totals = np.asarray(current_totals)
    current_throws = np.asarray(current_throws)
    return current_totals, current_throws


def sanitize(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


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

    current_totals = np.clip(np.abs(current_totals), np.finfo(float).eps, None)
    current_throws = np.clip(np.abs(current_throws), np.finfo(float).eps, None)

    num_sv = current_totals.shape[1]
    windows = np.arange(1, current_totals.shape[0] + 1)

    color_range = np.linspace(0, 1.0, num_sv)
    if num_sv > 2:
        color_range[2] = (color_range[-1] + color_range[-2]) / 2
    color_range = np.sort(color_range)
    colors = plt.cm.jet(color_range)

    label_base = f"ssize={ssize}, k={k}, seed={seed}"

    # ---------- current_total (log) ----------
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
    ax.set_title(f"{label_base} current_total_log")
    ax.grid(True, which="both")
    ax.legend()
    ax.tick_params(axis='x', labelrotation=45)
    fig.tight_layout()

    save_total = os.path.join(
        FIG_DIR,
        sanitize(f"{folder}_current_total_log.png")
    )
    fig.savefig(save_total, bbox_inches="tight", dpi=200)
    plt.close(fig)

    # ---------- cumulative throw (log) ----------
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
    ax.set_title(f"{label_base} cumulative_throw_log")
    ax.grid(True, which="both")
    ax.legend()
    ax.tick_params(axis='x', labelrotation=45)
    fig.tight_layout()

    save_throw = os.path.join(
        FIG_DIR,
        sanitize(f"{folder}_cumulative_throw_log.png")
    )
    fig.savefig(save_throw, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"Saved:")
    print(f"  {save_total}")
    print(f"  {save_throw}")
    