"""Focused diagnostic for diffuse-diffuse: why does θ=1 win so sharply?

Three probes:
  1. Singular-value spectrum of A (is θ=1 just balancing a degenerate spectrum?)
  2. Fine θ sweep around 1.0 (how narrow is the peak?)
  3. Per-block alignment trace for several θ (where does each θ diverge?)
  4. Robustness across seeds (is the θ=1 spike a generic property or a fluke?)
"""

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


def run_theta_trace(A, V_exact, sigma1, r, win, theta):
    """Same as sweep_carry_scale.run_theta but records per-block alignment."""
    n = A.shape[1]
    mA = A.shape[0]
    V_r = None
    S_r = None
    aligns = []
    for start0 in range(0, mA, win):
        end0 = min(start0 + win, mA)
        A_block = A[start0:end0, :]

        prev_sketch = None if (V_r is None or S_r is None) else S_r @ V_r.T
        M_gain = A_block if prev_sketch is None else np.vstack([prev_sketch, A_block])

        if V_r is None or S_r is None:
            B_scaled = None
        else:
            s_carry = np.diag(S_r)
            B_scaled = (V_r * (theta * s_carry)).T

        M_def = A_block if B_scaled is None else np.vstack([B_scaled, A_block])
        _, _, Vh_def = la.svd(M_def, full_matrices=False, lapack_driver="gesdd")
        V_def_full = Vh_def.T
        rr = min(r, V_def_full.shape[1])
        V_hat_raw = V_def_full[:, :rr]

        V_hat, s_new = projected_subspace_svd(M_gain, V_hat_raw)
        s_new = np.asarray(s_new).reshape(-1)

        V_r = V_hat
        S_r = np.diag(s_new)

        ll = 1
        align = float(np.linalg.norm(V_r @ (V_r.T @ V_exact[:, :ll]), "fro") / np.sqrt(ll))
        aligns.append(align)

    return aligns


def main():
    n = 128
    r = 2
    win = 32
    seed = 0

    np.random.seed(seed)
    A, V_exact, svec, sigma1 = generate_matrix_input(
        "diffuse-diffuse", n=n, preset="fast", seed=seed,
    )

    print(f"=== diffuse-diffuse spectrum (n={n}) ===")
    print(f"svec top-6: {svec[:6]}")
    print(f"σ_1 = {svec[0]:.6f}, σ_2 = {svec[1]:.6f}, σ_3 = {svec[2]:.6f}")
    print(f"signal-vs-tail gap σ_2 - σ_3: {svec[1] - svec[2]:+.6f}")
    # The actual signal/tail boundary is between index r_sig=2 and r_sig+1
    print(f"first 5 tail σ: {svec[2:7]}")
    print()

    # Per-block A_block spectrum (what each window's SVD sees).
    print("=== per-block A_block spectrum (top-4 σ) ===")
    for start0 in range(0, A.shape[0], win):
        end0 = min(start0 + win, A.shape[0])
        s_block = la.svd(A[start0:end0, :], compute_uv=False, lapack_driver="gesdd")
        print(f"block rows {start0+1}:{end0}: {s_block[:4]}")
    print()

    # Fine θ sweep around 1.0.
    print("=== fine θ sweep around 1.0 ===")
    fine_thetas = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0,
                   1.005, 1.01, 1.02, 1.05, 1.1, 1.2, 1.5]
    print(f"{'θ':>8s}  {'final align':>12s}")
    for theta in fine_thetas:
        np.random.seed(seed)
        aligns = run_theta_trace(A, V_exact, sigma1, r, win, theta)
        print(f"{theta:>8.4f}  {aligns[-1]:>12.6f}")
    print()

    # Per-block alignment trace for a few representative θ.
    print("=== per-block alignment trace ===")
    trace_thetas = [0.0, 0.5, 0.75, 0.99, 1.0, 1.01, 1.5]
    traces = {}
    for theta in trace_thetas:
        np.random.seed(seed)
        traces[theta] = run_theta_trace(A, V_exact, sigma1, r, win, theta)
    n_blocks = len(traces[1.0])
    header = ["block"] + [f"θ={t:g}" for t in trace_thetas]
    print("\t".join(header))
    for b in range(n_blocks):
        row = [f"{b+1}"]
        for t in trace_thetas:
            row.append(f"{traces[t][b]:.4f}")
        print("\t".join(row))
    print()

    # Robustness across seeds.
    print("=== θ=1 sharpness across seeds ===")
    print(f"{'seed':>5s}  {'θ=0.9':>8s}  {'θ=1.0':>8s}  {'θ=1.1':>8s}  {'argmax_θ':>8s}")
    test_thetas_robust = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5, 2.0]
    for s_seed in range(8):
        np.random.seed(s_seed)
        A_s, V_s, _, sigma1_s = generate_matrix_input(
            "diffuse-diffuse", n=n, preset="fast", seed=s_seed,
        )
        per_theta_align = {}
        for t in test_thetas_robust:
            np.random.seed(s_seed)
            aligns = run_theta_trace(A_s, V_s, sigma1_s, r, win, t)
            per_theta_align[t] = aligns[-1]
        best_t = max(per_theta_align, key=per_theta_align.get)
        print(
            f"{s_seed:>5d}  {per_theta_align[0.9]:>8.4f}  "
            f"{per_theta_align[1.0]:>8.4f}  {per_theta_align[1.1]:>8.4f}  "
            f"{best_t:>8.3f}"
        )


if __name__ == "__main__":
    main()
