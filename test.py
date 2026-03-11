import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
FIG_DIR = "figures"

# BASE_PREFIX = "bad_case1_1000_isvd"
# BASE_PREFIX = "bad_case2_1000_isvd"
# BASE_PREFIX = "bad_case3_1000_isvd"
# BASE_PREFIX = "kernel_stocks_1000_0.7071_isvd"
BASE_PREFIX = "kernel_stocks_1000_2.2361_isvd"

SIZE = 129
RESERVOIR_METHOD = "greedy"

# Residual mode:
#   "ws_reg_2norm"  -> use reservoir_residuals_data_* if available
#   "fallback_l2"   -> use ||approx_residuals||_2 from residuals_data_*
#   "fallback_max"  -> use max(abs(approx_residuals)) from residuals_data_*
RESIDUAL_MODE = "ws_reg_2norm"

SHOW_BAND = True   # min-max band across seeds
TRACE_RANK = 10    # same top-r trace comparison as before


def parse_folder(folder_name):
    """
    Examples:
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


def load_residual_curve(exp_dir, mode="ws_reg_2norm"):
    # Preferred exact whole-space residual files
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

    # Fallback to per-vector residuals
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

    if "k_64" in exp_dir:
        import pdb;pdb.set_trace()
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
    """
    Align curves to shortest length and return mean/min/max and mean endpoint.
    """
    min_len = min(len(c) for c in curves)
    arr = np.log10(np.stack([c[:min_len] for c in curves], axis=0))
    mean_curve = 10**arr.mean(axis=0)
    low_curve = 10**arr.min(axis=0)
    high_curve = 10**arr.max(axis=0)
    mean_endpoint = 10**arr[:, -1].mean()
    return mean_curve, low_curve, high_curve, mean_endpoint, min_len


# Collect matching folders grouped by (ssize, k)
groups = {}
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

for key in groups:
    groups[key].sort(key=lambda x: x[0])

sorted_keys = sorted(groups.keys(), key=lambda x: x[0])

fig, (ax_res, ax_tr) = plt.subplots(1, 2, figsize=(16, 7))

residual_source_kind = None
used_any_residual = False
used_any_trace = False

# import pdb;pdb.set_trace()
sorted_keys = [(ssize, k) for ssize, k in sorted_keys if k > 10]
for ssize, k in sorted_keys:
    seed_folders = groups[(ssize, k)]

    residual_curves = []
    trace_curves = []
    seeds_used_residual = []
    seeds_used_trace = []

    for seed, folder in seed_folders:
        exp_dir = os.path.join(OUTPUT_DIR, folder)

        # Residual
        rcurve, source_kind = load_residual_curve(exp_dir, mode=RESIDUAL_MODE)
        if rcurve is not None:
            rcurve = np.clip(rcurve, np.finfo(float).eps, None)
            residual_curves.append(rcurve)
            seeds_used_residual.append(seed)
            if residual_source_kind is None:
                residual_source_kind = source_kind

        # Trace error
        tcurve = load_trace_error_curve(exp_dir, rank_limit=TRACE_RANK)
        if tcurve is not None:
            tcurve = np.clip(tcurve, np.finfo(float).eps, None)
            trace_curves.append(tcurve)
            seeds_used_trace.append(seed)

    # Residual plot
    if residual_curves:
        # import pdb;pdb.set_trace()
        mean_curve, low_curve, high_curve, mean_endpoint, npts = aggregate_seed_curves(residual_curves)
        x = SIZE + np.arange(0, npts)*ssize
        x[-1] = 1000
        # x = np.arange(1, npts + 1)  # start at 1 so log x-scale works
        label = f"ssize={ssize}, k={k}, end={mean_endpoint:.3e}, n={len(residual_curves)}"
        line, = ax_res.semilogy(x, mean_curve, marker="o", linewidth=1.5, label=label)
        if SHOW_BAND and len(residual_curves) > 1:
            ax_res.fill_between(x, low_curve, high_curve, alpha=0.18, color=line.get_color())
        used_any_residual = True
        print(f"Residual  (ssize={ssize}, k={k}) seeds used: {seeds_used_residual}, mean endpoint={mean_endpoint:.6e}")

    # Trace plot
    if trace_curves:
        mean_curve, low_curve, high_curve, mean_endpoint, npts = aggregate_seed_curves(trace_curves)
        x = SIZE + np.arange(0, npts)*ssize  # start at 1 so log x-scale works
        x[-1] = 1000
        label = f"ssize={ssize}, k={k}, end={mean_endpoint:.3e}, n={len(trace_curves)}"
        line, = ax_tr.semilogy(x, mean_curve, marker="o", linewidth=1.5, label=label)
        if SHOW_BAND and len(trace_curves) > 1:
            ax_tr.fill_between(x, low_curve, high_curve, alpha=0.18, color=line.get_color())
        used_any_trace = True
        print(f"TraceErr  (ssize={ssize}, k={k}) seeds used: {seeds_used_trace}, mean endpoint={mean_endpoint:.6e}")

# Set x-axis to log scale on both plots
for ax in (ax_res, ax_tr):
    # ax.set_xscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.set_xlabel("Window size (log scale)")
    ax.set_xlim(left=750, right=1000)

ax_res.set_ylabel("Whole-space residual")
ax_res.set_title(
    f"Residual: {BASE_PREFIX}_random_uniform[*]\n"
    f"size={SIZE}, reservoir={RESERVOIR_METHOD}, source={residual_source_kind}"
)

ax_tr.set_ylabel("Relative trace error")
ax_tr.set_title(
    f"Trace error: {BASE_PREFIX}_random_uniform[*]\n"
    f"size={SIZE}, reservoir={RESERVOIR_METHOD}, top-{TRACE_RANK} trace"
)

if used_any_residual:
    ax_res.legend(fontsize=8, ncol=1)
if used_any_trace:
    ax_tr.legend(fontsize=8, ncol=1)

plt.tight_layout()
os.makedirs(FIG_DIR, exist_ok=True)

save_path = os.path.join(
    FIG_DIR,
    f"{BASE_PREFIX}_random_uniform_allseeds_size_{SIZE}_all_ssize_k_sum_{SIZE}_"
    f"residual_and_trace_compare.png"
)
plt.savefig(save_path, bbox_inches="tight", dpi=220)
plt.show()

print(f"\nSaved to: {save_path}")