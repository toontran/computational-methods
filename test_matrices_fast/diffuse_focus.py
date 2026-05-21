"""Focused diagnostics for diffuse-diffuse — characterize the iSVD spike.

Three probes:
  (a) Fine ε sweep around 0 on carry and window (additive σ² shift),
      to see the shape of the spike at ε=0.
  (b) Per-block trajectory of alignment for iSVD vs small perturbations.
  (c) Per-block decomposition: how much of V_r lives in span(V_exact[:,:r])
      vs the orthogonal complement.

Diffuse-diffuse properties (relevant context):
  σ_signal ≈ 1.0, σ_tail ≈ 0.99 → signal/tail σ ratio ≈ 1.01.
  V_exact is some fixed orthonormal frame (the matrix's true right SVs).
  iSVD's SVD of [B; A_block] just barely picks signal over tail; the
  perturbation tolerance is on the order of (σ_sig² - σ_tail²) ≈ 0.02.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as la

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from cex_structured_new_py import projected_subspace_svd  # noqa: E402
from cex_restricted_space_probe import generate_matrix_input  # noqa: E402


def streaming_with_trajectory(A, V_exact, sigma1, r, win, eps, location):
    n = A.shape[1]
    mA = A.shape[0]
    V_r = None
    S_r = None

    traj = []  # (block_id, align_slot0, align_slot1, frac_in_Vexact_span)
    for block_id, start0 in enumerate(range(0, mA, win)):
        end0 = min(start0 + win, mA)
        A_block = A[start0:end0, :]

        prev_sketch = None if (V_r is None or S_r is None) else S_r @ V_r.T
        M_gain = A_block if prev_sketch is None else np.vstack([prev_sketch, A_block])

        # Carry possibly shifted.
        if V_r is None or S_r is None:
            B_def = None
        else:
            s_carry = np.diag(S_r)
            if location in ("carry", "both"):
                s_carry_new = np.sqrt(np.maximum(s_carry ** 2 + eps, 0.0))
            else:
                s_carry_new = s_carry
            B_def = (V_r * s_carry_new).T

        # Window possibly shifted.
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

        # Diagnostics — match the benchmark metric: projection norm of
        # V_exact[:, 0] onto span(V_r). This is invariant to internal
        # rotation of V_r (unlike slot-0 dot product).
        align_v0 = float(np.linalg.norm(V_r.T @ V_exact[:, 0]))  # = ||V_r V_r^T V_exact[:,0]||
        align_v1 = float(np.linalg.norm(V_r.T @ V_exact[:, 1])) if V_exact.shape[1] >= 2 else 0.0
        # span fraction: how much of span(V_exact[:, :r]) is captured
        span_frac = float(np.linalg.norm(V_exact[:, :r].T @ V_r, "fro") ** 2 / r)
        traj.append((block_id, align_v0, align_v1, span_frac))

    return traj


def fine_eps_sweep(A, V_exact, sigma1, r, win, location, eps_list):
    """Returns {eps: align_slot0}."""
    out = {}
    for eps in eps_list:
        np.random.seed(0)
        traj = streaming_with_trajectory(A, V_exact, sigma1, r, win, eps, location)
        # Use final block's slot0 alignment.
        out[eps] = traj[-1][1]   # a0
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--win", type=int, default=32)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-plot", default="diffuse_focus.png")
    args = parser.parse_args()

    np.random.seed(args.seed)
    A, V_exact, _, sigma1 = generate_matrix_input(
        "diffuse-diffuse", n=args.n, preset="fast", seed=args.seed,
    )
    print(f"# diffuse-diffuse: n={A.shape[0]}, win={args.win}, r={args.rank}")

    # Look at the full σ spectrum so we know the gap.
    _, s_full, _ = la.svd(A, full_matrices=False, lapack_driver="gesdd")
    print(f"# top 5 σ of full A: {s_full[:5]}")
    print(f"# σ_3 - σ_2: {s_full[2] - s_full[1]:.4g}")
    print(f"# σ_2² - σ_3²: {s_full[1]**2 - s_full[2]**2:.4g}  (perturbation tolerance scale)")

    # === (a) Fine ε sweep around 0 ===
    fine_eps = [-0.5, -0.2, -0.1, -0.05, -0.02, -0.01, -0.005, -0.002,
                0.0,
                0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    sweep_carry = fine_eps_sweep(A, V_exact, sigma1, args.rank, args.win, "carry", fine_eps)
    sweep_window = fine_eps_sweep(A, V_exact, sigma1, args.rank, args.win, "window", fine_eps)

    print("\n=== Fine ε sweep around 0 — alignment ||V_r V_r^T V_exact[:,0]|| (projection norm) ===")
    print("ε\tcarry\twindow")
    for e in fine_eps:
        print(f"{e:+.3f}\t{sweep_carry[e]:.4f}\t{sweep_window[e]:.4f}")

    # === (b) Per-block trajectories ===
    print("\n=== Per-block trajectory (align V_e[:,0], align V_e[:,1], span fraction) ===")
    trajectories = {
        "iSVD (ε=0)": streaming_with_trajectory(A, V_exact, sigma1, args.rank, args.win, 0.0, "carry"),
        "carry ε=-0.1": streaming_with_trajectory(A, V_exact, sigma1, args.rank, args.win, -0.1, "carry"),
        "carry ε=-0.01": streaming_with_trajectory(A, V_exact, sigma1, args.rank, args.win, -0.01, "carry"),
        "window ε=+0.01": streaming_with_trajectory(A, V_exact, sigma1, args.rank, args.win, 0.01, "window"),
        "window ε=+0.1": streaming_with_trajectory(A, V_exact, sigma1, args.rank, args.win, 0.1, "window"),
    }
    print("block\t" + "\t".join(f"{lbl}_align0/1/span" for lbl in trajectories))
    nblocks = max(len(t) for t in trajectories.values())
    for b in range(nblocks):
        row = [str(b)]
        for lbl, traj in trajectories.items():
            if b < len(traj):
                _, a0, a1, span = traj[b]
                row.append(f"{a0:.2f}/{a1:.2f}/{span:.2f}")
            else:
                row.append("--")
        print("\t".join(row))

    # === Plot ===
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: fine ε sweep.
        ax = axes[0]
        eps_arr = np.array(fine_eps)
        ax.plot(eps_arr, [sweep_carry[e] for e in fine_eps], marker="o", label="carry shift")
        ax.plot(eps_arr, [sweep_window[e] for e in fine_eps], marker="s", label="window shift")
        ax.axvline(0.0, color="gray", linestyle="--", alpha=0.5, label="ε=0 (iSVD)")
        ax.set_xlabel("ε (σ² shift)")
        ax.set_ylabel("|V_r[:,0] · V_exact[:,0]|")
        ax.set_title("diffuse-diffuse: fine ε sweep around 0")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: per-block trajectory.
        ax = axes[1]
        for lbl, traj in trajectories.items():
            blocks = [t[0] for t in traj]
            a0s = [t[1] for t in traj]
            ax.plot(blocks, a0s, marker="o", label=lbl, alpha=0.85)
        ax.set_xlabel("block index")
        ax.set_ylabel("|V_r[:,0] · V_exact[:,0]|")
        ax.set_title("diffuse-diffuse: per-block alignment trajectory")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        out = HERE / args.save_plot
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        print(f"\n# Plot saved to {out}", file=sys.stderr)
    except ImportError:
        print("# matplotlib not available; skipping plot.", file=sys.stderr)


if __name__ == "__main__":
    main()
