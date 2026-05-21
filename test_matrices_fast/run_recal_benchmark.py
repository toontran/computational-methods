"""Standalone runner for defsvd-recal-carry.

(1) Graceful-degradation verification (uniform Delta -> defl_i == delta_sum_sq).
(2) Benchmark at l=1 and l=r across the discriminator + financial matrices,
    against isvd-ref, fd-ref, defsvd-carryonly, defsvd-symm, defsvd-recal-carry.

Does NOT mutate the benchmark methods list; calls run_streaming directly.
"""

import sys
import numpy as np

from benchmark_defsvd import run_streaming
from cex_restricted_space_probe import generate_matrix_input


def graceful_degradation_check():
    """All Delta_i equal => w_i = 1/r => defl_i = (r*delta_sum_sq)*(1/r) = delta_sum_sq.
    This is EXACTLY the per-direction deflation of defsvd-carryonly."""
    print("=== Graceful-degradation check ===")
    failures = 0
    for r in (2, 4, 8):
        for delta_sum_sq in (0.0, 1.0, 7.3, 123.456):
            delta_i = np.ones(r)                    # all equal
            mean_delta = float(np.mean(delta_i))
            delta_tilde = (np.zeros_like(delta_i) if mean_delta == 0.0
                           else delta_i / mean_delta)
            w = 1.0 / (1.0 + delta_tilde)
            w = w / np.sum(w)
            defl = (r * delta_sum_sq) * w
            ok = np.allclose(defl, delta_sum_sq) and np.allclose(w, 1.0 / r)
            print(f"  r={r:>2} delta_sum_sq={delta_sum_sq:>8.3f}: "
                  f"w={w[0]:.6f} (=1/r={1.0/r:.6f}), defl_i={defl[0]:.6f} "
                  f"(==delta_sum_sq={delta_sum_sq:.6f}) -> {'PASS' if ok else 'FAIL'}")
            if not ok:
                failures += 1
    print(f"  RESULT: {'ALL PASS' if failures == 0 else f'{failures} FAILURES'}")
    print()
    return failures == 0


METHODS = [
    ("isvd-ref", "iSVD", True),
    ("fd-ref", "FD", True),
    ("defsvd-symm", "DefSVD", True),
    ("defsvd-carryonly", "DefSVD", False),
    ("defsvd-recal-carry", "DefSVD-RecalCarry", False),
]

MATRICES = [
    "static-cex-noisy",
    "static-cex-gauss",
    "static-cex-exptail",
    "static-cex",
    "mixed-tail-sharp",
    "crowded-strategy",
    "options-vol-surface",
]


def run_benchmark(n=128, win=32, rank=2, seed=0, preset="fast"):
    print(f"=== Benchmark (n={n}, win={win}, rank={rank}, seed={seed}, preset={preset}) ===")
    print(f"{'matrix':<20}{'method':<22}{'align_l1':>10}{'align_lr':>10}"
          f"{'relerr_s1':>12}{'elapsed':>9}")
    results = {}
    for matrix in MATRICES:
        try:
            np.random.seed(seed)
            A, V_exact, _, sigma1 = generate_matrix_input(
                matrix, n=n, preset=preset, seed=seed)
        except Exception as e:
            print(f"# SKIP {matrix}: {e}", file=sys.stderr)
            continue
        results[matrix] = {}
        for label, mode, deflate_window in METHODS:
            np.random.seed(seed)
            align_l1, rel_err, elapsed = run_streaming(
                A, V_exact, sigma1, rank, win, mode, deflate_window, l=1)
            np.random.seed(seed)
            align_lr, _, _ = run_streaming(
                A, V_exact, sigma1, rank, win, mode, deflate_window, l=rank)
            results[matrix][label] = (align_l1, align_lr, rel_err, elapsed)
            print(f"{matrix:<20}{label:<22}{align_l1:>10.6f}{align_lr:>10.6f}"
                  f"{rel_err:>12.6e}{elapsed:>9.3f}")
        print()
    return results


if __name__ == "__main__":
    ok = graceful_degradation_check()
    run_benchmark()
    if not ok:
        sys.exit(1)
