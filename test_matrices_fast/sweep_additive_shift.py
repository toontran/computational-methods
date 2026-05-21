"""Additive σ² shift sweep.

For each block:
    s_new = sqrt(max(s^2 + ε, 0))

ε > 0  → inflate (add to σ²)
ε = 0  → iSVD baseline
ε < 0  → deflate (subtract; clamp to zero)

Three locations to apply the shift:
    'carry'  : shift only carry's σ² (window pristine). ε<0 matches the
               defsvd-carryonly family.
    'window' : shift only the window's σ² (carry pristine). The
               "outside-only" variant the user asked about.
    'both'   : shift both. Note that on the un-projected window (no V_r
               orthogonalization), this is NOT identity-equivalent — V_r
               and the window's right SVs can overlap, so adding εI to
               their combined Gram doesn't just rescale eigenvalues.

Magnitudes are always computed honestly via
projected_subspace_svd(M_gain, V_hat_raw) against the un-modified M_gain.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from cex_structured_new_py import projected_subspace_svd  # noqa: E402
from cex_restricted_space_probe import generate_matrix_input  # noqa: E402


HARD_MATRICES = [
    "static-cex",
    "diffuse-diffuse",
    "mixed-tail-soft",
    "mixed-tail-balanced",
    "mixed-tail-sharp",
    "residual-spiky-shocks",
    "risk-residual-panel",
]
EASY_REFS = [
    "alternative-data-signals",
    "futures-term-structure",
]
DEFAULT_EPS = [-1.0, -0.5, -0.25, -0.1, -0.01, 0.0, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
LOCATIONS = ("carry", "window", "both")


def run_eps(A, V_exact, sigma1, r, win, eps, location, l=1):
    n = A.shape[1]
    mA = A.shape[0]

    V_r = None
    S_r = None

    for start0 in range(0, mA, win):
        end0 = min(start0 + win, mA)
        A_block = A[start0:end0, :]

        prev_sketch = None if (V_r is None or S_r is None) else S_r @ V_r.T
        M_gain = A_block if prev_sketch is None else np.vstack([prev_sketch, A_block])

        # Carry: pristine or shifted.
        if V_r is None or S_r is None:
            B_def = None
        else:
            s_carry = np.diag(S_r)
            if location in ("carry", "both"):
                s_carry_new = np.sqrt(np.maximum(s_carry ** 2 + eps, 0.0))
            else:
                s_carry_new = s_carry
            B_def = (V_r * s_carry_new).T

        # Window: pristine or shifted (via SVD then σ² shift).
        if location in ("window", "both"):
            U_w, s_w, Vh_w = la.svd(A_block, full_matrices=False, lapack_driver="gesdd")
            s_w_new = np.sqrt(np.maximum(s_w ** 2 + eps, 0.0))
            A_w_def = (U_w * s_w_new) @ Vh_w
        else:
            A_w_def = A_block

        M_def = A_w_def if B_def is None else np.vstack([B_def, A_w_def])
        _, s_def, Vh_def = la.svd(M_def, full_matrices=False, lapack_driver="gesdd")
        V_def_full = Vh_def.T
        rr = min(r, V_def_full.shape[1])
        V_hat_raw = V_def_full[:, :rr]

        V_hat, s_new = projected_subspace_svd(M_gain, V_hat_raw)
        s_new = np.asarray(s_new).reshape(-1)

        V_r = V_hat
        S_r = np.diag(s_new)

    ll = min(l, V_r.shape[1])
    align = float(np.linalg.norm(V_r @ (V_r.T @ V_exact[:, :ll]), "fro") / np.sqrt(ll))
    top_sval_est = float(S_r[0, 0]) if S_r is not None and S_r.size else 0.0
    rel_err = abs(top_sval_est - sigma1) / sigma1 if sigma1 != 0 else 0.0
    return align, rel_err


def print_table(title, matrices, eps_list, results):
    print(f"\n=== {title} ===")
    header = ["matrix"] + [f"ε={e:g}" for e in eps_list] + ["argmax_ε"]
    print("\t".join(header))
    for matrix in matrices:
        if matrix not in results:
            continue
        row = [matrix]
        best_e, best_a = None, -1.0
        for e in eps_list:
            a = results[matrix][e]
            row.append(f"{a:.4f}")
            if a > best_a:
                best_a, best_e = a, e
        row.append(f"{best_e:g}")
        print("\t".join(row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--win", type=int, default=32)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--matrices", nargs="+", default=HARD_MATRICES + EASY_REFS)
    parser.add_argument("--eps", nargs="+", type=float, default=DEFAULT_EPS)
    parser.add_argument("--save-plot", default="sweep_additive_shift.png")
    args = parser.parse_args()

    eps_list = sorted(set(args.eps))
    aligns = {loc: {} for loc in LOCATIONS}

    for matrix in args.matrices:
        try:
            np.random.seed(args.seed)
            A, V_exact, _, sigma1 = generate_matrix_input(
                matrix, n=args.n, preset=args.preset, seed=args.seed,
            )
        except (ValueError, RuntimeError, TypeError) as e:
            print(f"# SKIP {matrix}: {e}", file=sys.stderr, flush=True)
            continue
        for loc in LOCATIONS:
            aligns[loc][matrix] = {}
            for eps in eps_list:
                np.random.seed(args.seed)
                a, _ = run_eps(A, V_exact, sigma1, args.rank, args.win, eps, loc)
                aligns[loc][matrix][eps] = a

    for loc in LOCATIONS:
        print_table(f"alignment ‖V_r V_rᵀ V_exact[:,0]‖  |  shift on {loc.upper()}",
                    args.matrices, eps_list, aligns[loc])

    # Multi-panel plot: one subplot per matrix, three lines (carry/window/both).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matrices_with_data = [m for m in args.matrices if m in aligns["carry"]]
        ncols = 3
        nrows = int(np.ceil(len(matrices_with_data) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=False)
        axes = np.atleast_2d(axes).reshape(nrows, ncols)
        for idx, matrix in enumerate(matrices_with_data):
            ax = axes[idx // ncols, idx % ncols]
            for loc, marker in zip(LOCATIONS, ("o", "s", "^")):
                ys = [aligns[loc][matrix][e] for e in eps_list]
                ax.plot(eps_list, ys, marker=marker, label=loc, alpha=0.85)
            ax.axvline(0.0, color="gray", linestyle="--", alpha=0.5)
            ax.set_title(matrix, fontsize=9)
            ax.set_xlabel("ε (σ² shift)")
            ax.set_ylabel("align")
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(loc="best", fontsize=7)
        for idx in range(len(matrices_with_data), nrows * ncols):
            axes[idx // ncols, idx % ncols].axis("off")
        fig.suptitle(f"Additive σ² shift sweep (n={args.n}, win={args.win}, r={args.rank})")
        out = HERE / args.save_plot
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        print(f"\n# Plot saved to {out}", file=sys.stderr)
    except ImportError:
        print("# matplotlib not available; skipping plot.", file=sys.stderr)


if __name__ == "__main__":
    main()
