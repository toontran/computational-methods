import os
import re
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
FIG_DIR = "figures"

MATRICES = [
    "bad_case1_1000",
    "bad_case2_1000",
    "bad_case3_1000",
]

# Optional filters. Set to None if you do not want to restrict.
METHOD_PREFERENCE = ["isvd", "isvdstG"]
WINDOW_SIZE = None   # e.g. 1
K = None             # e.g. 8

ZOOM_RANK = 50


def parse_experiment_folder(folder_name: str):
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


def choose_one_experiment(matrix_name: str):
    candidates = []

    for folder in os.listdir(OUTPUT_DIR):
        parsed = parse_experiment_folder(folder)
        if parsed is None:
            continue
        if parsed["matrix"] != matrix_name:
            continue
        if WINDOW_SIZE is not None and parsed["size"] != WINDOW_SIZE:
            continue
        if K is not None and parsed["k"] != K:
            continue
        if METHOD_PREFERENCE is not None and parsed["method"] not in METHOD_PREFERENCE:
            continue
        candidates.append(parsed)

    if not candidates:
        return None

    method_rank = {m: i for i, m in enumerate(METHOD_PREFERENCE)} if METHOD_PREFERENCE else {}

    # Prefer:
    # 1) earlier method in METHOD_PREFERENCE
    # 2) unseeded / seed 1
    # 3) non-random_uniform before random_uniform
    # 4) lexicographically smaller folder name
    candidates.sort(
        key=lambda x: (
            method_rank.get(x["method"], 10**9),
            x["seed"],
            1 if x["is_random_uniform"] else 0,
            x["folder"],
        )
    )
    return candidates[0]


def save_spectrum_plots(matrix_name: str, exp_dir: str):
    spectrum_path = os.path.join(exp_dir, "spectrum_data_0.npz")
    if not os.path.exists(spectrum_path):
        print(f"Missing {spectrum_path}")
        return

    data = np.load(spectrum_path, allow_pickle=True)
    S_exact = np.asarray(data["S_exact"]).reshape(-1)

    # Full spectrum
    plt.figure(figsize=(12, 6))
    plt.semilogy(np.arange(len(S_exact)), S_exact, label=matrix_name)
    plt.ylabel("Eigenvalue", fontsize=16)
    plt.xlabel("Index", fontsize=16)
    plt.title("Spectrum")
    plt.legend()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_full = os.path.join(FIG_DIR, f"{matrix_name}_spectrum.png")
    plt.savefig(save_full, bbox_inches="tight", dpi=220)
    plt.close()

    # Zoomed spectrum
    viz_rank = min(ZOOM_RANK, len(S_exact))
    plt.figure(figsize=(12, 6))
    plt.semilogy(np.arange(viz_rank), S_exact[:viz_rank], label=matrix_name)
    plt.ylabel("Eigenvalue")
    plt.xlabel("Index")
    plt.title("Spectrum")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    save_zoom = os.path.join(FIG_DIR, f"{matrix_name}_spectrum_zoomed.png")
    plt.savefig(save_zoom, bbox_inches="tight", dpi=220)
    plt.close()

    print(f"Saved {save_full}")
    print(f"Saved {save_zoom}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    for matrix_name in MATRICES:
        chosen = choose_one_experiment(matrix_name)
        if chosen is None:
            print(f"No matching experiment found for {matrix_name}")
            continue

        exp_dir = os.path.join(OUTPUT_DIR, chosen["folder"])
        print(
            f"Using {chosen['folder']} "
            f"(method={chosen['method']}, seed={chosen['seed']}, "
            f"size={chosen['size']}, k={chosen['k']})"
        )
        save_spectrum_plots(matrix_name, exp_dir)


if __name__ == "__main__":
    main()