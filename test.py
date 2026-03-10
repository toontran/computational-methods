import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = "figures"

# Fixed setting requested
PREFIX = "log_stocks_5000_0.7071_isvd_random_uniform"
SIZE = 110

# Match folders like:
# figures/log_stocks_5000_0.7071_isvd_random_uniform_size_110_ssize_46_k_64_reservoir_greedy/
folder_re = re.compile(
    rf"^{re.escape(PREFIX)}_size_{SIZE}_ssize_(\d+)_k_(\d+)_reservoir_greedy$"
)

def load_relative_trace_error(folder):
    files = sorted(
        glob.glob(os.path.join(folder, "spectrum_data_*.npz")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1])
    )
    if not files:
        return None

    e = []
    for f in files:
        data = np.load(f)
        S = data["S"]
        S_exact = data["S_exact"]

        limit_S = min(10, len(S))
        denom = np.sum(S_exact[:limit_S])
        if denom == 0:
            e.append(np.nan)
        else:
            tr_S = np.sum(S[:limit_S])
            e.append(np.abs(np.sum(S_exact[:limit_S]) - tr_S) / denom)
    return np.asarray(e)

curves = []

for path in glob.glob(os.path.join(BASE_DIR, f"{PREFIX}_size_{SIZE}_ssize_*_k_*_reservoir_greedy")):
    name = os.path.basename(path.rstrip("/\\"))
    m = folder_re.match(name)
    if not m:
        continue

    ssize = int(m.group(1))
    k = int(m.group(2))

    if ssize + k != SIZE:
        continue

    e = load_relative_trace_error(path)
    if e is None:
        continue

    curves.append((ssize, k, e))

curves.sort(key=lambda t: t[0])  # sort by ssize

plt.figure(figsize=(12, 7))
for ssize, k, e in curves:
    plt.semilogy(np.arange(len(e)), e, marker="o", label=f"ssize={ssize}, k={k}")

plt.xlabel("Window index")
plt.ylabel("Relative trace error")
plt.title(f"Relative trace error over windows\n{PREFIX}, size={SIZE}, ssize+k={SIZE}")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig(f"{PREFIX}_size_{SIZE}_all_pairs_trace_error_compare.png", bbox_inches="tight")
plt.show()