"""Combined_s7 with future-block weight sweep.

For each matrix and each peek ratio h2/(h1+h2), run combined_s7 with future
weight w ∈ {1, 10, 100, 1000}. Weight w scales A_half2 by sqrt(w) so the
2-norm energy contribution ||A_fut v||^2 (in both gain and entropy pool) is
multiplied by w.

Combined baseline (no peek, block size T = h1+h2) is shown once per row.
"""

from __future__ import annotations

import sys
import time
import numpy as np

import cex_restricted_space_probe as probe
import half_window_sliding_hmean_experiment as exp


SEED = 0
ROW_SHUFFLE_SEED = 0
N = 1024

MATRICES = (
    "static-cex",
    "diffuse-diffuse",
    "mixed-tail-sharp",
    "mixed-tail-balanced",
    "mixed-tail-soft",
)

H1_S7 = 32
H2_LIST = (32, 64)            # peek ratios 0.500, 0.667
WEIGHTS = (1, 10, 100, 1000)


def run_combined_at_T(args, A, V_exact, sigma1, T):
    args.h1_mult = 1.0
    args.h2_mult = 0.0
    args.combined_s7_future_weight = 1.0
    np.random.seed(args.seed)
    r = exp.run_pair_stream(A, V_exact, sigma1, args, "combined",
                            half_win=T, sliding=True)
    last = r["rows"][-1]
    return last["cos1"] ** 2, last["cos2"] ** 2, len(r["rows"])


def run_s7(args, A, V_exact, sigma1, h1, h2, future_weight):
    args.h1_mult = 1.0
    args.h2_mult = h2 / h1
    args.combined_s7_future_weight = float(future_weight)
    np.random.seed(args.seed)
    r = exp.run_pair_stream(A, V_exact, sigma1, args, "combined_s7",
                            half_win=h1, sliding=True)
    last = r["rows"][-1]
    return last["cos1"] ** 2, last["cos2"] ** 2, len(r["rows"])


def main():
    sys.argv = ["x", "--seed", str(SEED)]
    args = exp.parse_args()
    args.row_shuffle_seed = ROW_SHUFFLE_SEED

    print(f"seed={SEED}  row_shuffle_seed={ROW_SHUFFLE_SEED}  n={N}  s7_h1={H1_S7}")
    print()

    for matrix in MATRICES:
        A, V_exact, _, sigma1 = probe.generate_matrix_input(
            matrix, n=N, preset="fast", seed=0,
            shuffle_rows=True, row_shuffle_seed=ROW_SHUFFLE_SEED,
        )
        print(f"=== {matrix} ===")
        print(f"{'h2':>3} {'T':>3} {'h2/T':>5}  "
              f"{'comb R':>6} {'comb c1²':>9} {'comb c2²':>9}  "
              f"{'w':>5} {'s7 R':>4} {'s7 c1²':>7} {'s7 c2²':>7}  "
              f"{'Δc1²':>7} {'Δc2²':>7}")
        print("-" * 110)
        for h2 in H2_LIST:
            T = H1_S7 + h2
            ratio = h2 / T
            t0 = time.time()
            comb_c1, comb_c2, comb_R = run_combined_at_T(args, A, V_exact, sigma1, T)
            for w in WEIGHTS:
                s7_c1, s7_c2, s7_R = run_s7(args, A, V_exact, sigma1, H1_S7, h2, w)
                d1 = s7_c1 - comb_c1
                d2 = s7_c2 - comb_c2
                print(f"{h2:>3} {T:>3} {ratio:>5.3f}  "
                      f"{comb_R:>6} {comb_c1:>9.4f} {comb_c2:>9.4f}  "
                      f"{w:>5} {s7_R:>4} {s7_c1:>7.4f} {s7_c2:>7.4f}  "
                      f"{d1:>+7.4f} {d2:>+7.4f}")
        print()


if __name__ == "__main__":
    main()
