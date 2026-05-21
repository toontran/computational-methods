"""Sweep the carry-vs-window ratio knob θ across the benchmark matrices.

At each block we form the deflated stack as
    B_scaled = (θ * diag(S_r)) @ V_r.T
    M_def    = [B_scaled ; A_block]            (window pristine)
    V_hat    = top-r right SV of svd(M_def)
    V_hat, s = projected_subspace_svd(M_gain, V_hat)   # honest magnitudes

θ controls how strongly the carry votes in basis selection:
    θ = 0    → carry zeroed out; basis = top-r right SV of A_block alone (no memory)
    θ ∈(0,1) → carry weakened relative to window (FD-spirit, ratio decreased)
    θ = 1    → exactly iSVD (carry at full strength)
    θ > 1    → carry amplified beyond iSVD (basis pinned to carry)
    θ → ∞    → basis = V_r forever, no refinement.

The user's claim being tested: iSVD (θ=1) is just one point on this spectrum.
There is no a-priori reason θ=1 must win on a given matrix. Different
matrices should peak at different θ.
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
DEFAULT_THETAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]


def run_theta(A, V_exact, sigma1, r, win, theta, l=1):
    n = A.shape[1]
    mA = A.shape[0]
    t0 = time.time()

    V_r = None
    S_r = None

    for start0 in range(0, mA, win):
        end0 = min(start0 + win, mA)
        A_block = A[start0:end0, :]

        prev_sketch = None if (V_r is None or S_r is None) else S_r @ V_r.T
        M_gain = A_block if prev_sketch is None else np.vstack([prev_sketch, A_block])

        # Direction policy: scaled carry stacked with pristine window.
        if V_r is None or S_r is None:
            B_scaled = None
        else:
            s_carry = np.diag(S_r)
            B_scaled = (V_r * (theta * s_carry)).T

        M_def = A_block if B_scaled is None else np.vstack([B_scaled, A_block])
        _, s_def, Vh_def = la.svd(M_def, full_matrices=False, lapack_driver="gesdd")
        V_def_full = Vh_def.T
        rr = min(r, V_def_full.shape[1])
        V_hat_raw = V_def_full[:, :rr]

        # Honest magnitudes from un-modified M_gain.
        V_hat, s_new = projected_subspace_svd(M_gain, V_hat_raw)
        s_new = np.asarray(s_new).reshape(-1)

        V_r = V_hat
        S_r = np.diag(s_new)

    ll = min(l, V_r.shape[1])
    align = float(np.linalg.norm(V_r @ (V_r.T @ V_exact[:, :ll]), "fro") / np.sqrt(ll))
    top_sval_est = float(S_r[0, 0]) if S_r is not None and S_r.size else 0.0
    rel_err = abs(top_sval_est - sigma1) / sigma1 if sigma1 != 0 else 0.0
    elapsed = time.time() - t0
    return align, rel_err, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--win", type=int, default=32)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset", default="fast")
    parser.add_argument(
        "--matrices", nargs="+",
        default=HARD_MATRICES + EASY_REFS,
    )
    parser.add_argument(
        "--thetas", nargs="+", type=float, default=DEFAULT_THETAS,
    )
    parser.add_argument("--save-plot", default="sweep_carry_scale_align.png")
    args = parser.parse_args()

    thetas = sorted(set(args.thetas))

    # Collect results: results[matrix][theta] = align
    results = {}
    rel_errs = {}
    for matrix in args.matrices:
        try:
            np.random.seed(args.seed)
            A, V_exact, _, sigma1 = generate_matrix_input(
                matrix, n=args.n, preset=args.preset, seed=args.seed,
            )
        except (ValueError, RuntimeError, TypeError) as e:
            print(f"# SKIP {matrix}: {e}", file=sys.stderr, flush=True)
            continue
        results[matrix] = {}
        rel_errs[matrix] = {}
        for theta in thetas:
            np.random.seed(args.seed)
            align, rel_err, _ = run_theta(
                A, V_exact, sigma1, args.rank, args.win, theta,
            )
            results[matrix][theta] = align
            rel_errs[matrix][theta] = rel_err

    # Print align table.
    print("\n=== alignment ||V_r V_r^T V_exact[:,0]||  (higher = better) ===")
    header = ["matrix"] + [f"θ={t:g}" for t in thetas] + ["argmax_θ"]
    print("\t".join(header))
    for matrix in results:
        row = [matrix]
        best_t = None
        best_a = -1.0
        for t in thetas:
            a = results[matrix][t]
            row.append(f"{a:.4f}")
            if a > best_a:
                best_a = a
                best_t = t
        row.append(f"{best_t:g}")
        print("\t".join(row))

    # Print rel_err table.
    print("\n=== rel_err σ_1  (lower = better) ===")
    print("\t".join(header[:-1]))
    for matrix in rel_errs:
        row = [matrix]
        for t in thetas:
            row.append(f"{rel_errs[matrix][t]:.4f}")
        print("\t".join(row))

    # Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        for matrix in results:
            ys = [results[matrix][t] for t in thetas]
            ax.plot(thetas, ys, marker="o", label=matrix)
        ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5, label="iSVD (θ=1)")
        ax.set_xlabel("θ (carry scale)")
        ax.set_ylabel("alignment ||V_r V_r^T V_exact[:,0]||")
        ax.set_title(f"Carry-scale sweep (n={args.n}, win={args.win}, r={args.rank})")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        out_path = HERE / args.save_plot
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"\n# Plot saved to {out_path}", file=sys.stderr)
    except ImportError:
        print("# matplotlib not available; skipping plot.", file=sys.stderr)


if __name__ == "__main__":
    main()
