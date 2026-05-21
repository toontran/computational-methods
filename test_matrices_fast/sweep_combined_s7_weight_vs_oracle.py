"""Vary block size T = h1+h2 with h1=32 fixed; compare s7 with weight w in
{1,10,100,1000} vs combined (block size T, no peek). Report achieved cos1²/cos2²
and oracle_proj_norm at the FINAL round so we can see the achievable upper bound
(rowspace(M_gain) ∩ V_exact[:,j] mass) vs what each policy attained.
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
T_LIST = (32, 48, 64, 96, 128, 192)
WEIGHTS = (1, 10, 100, 1000)


def run_combined_at_T(args, A, V_exact, sigma1, T):
    args.h1_mult = 1.0
    args.h2_mult = 0.0
    args.combined_s7_future_weight = 1.0
    np.random.seed(args.seed)
    r = exp.run_pair_stream(A, V_exact, sigma1, args, "combined",
                            half_win=T, sliding=True)
    last = r["rows"][-1]
    return last["cos1"] ** 2, last["cos2"] ** 2, last["oracle_proj_norm1"], last["oracle_proj_norm2"], len(r["rows"])


def run_s7(args, A, V_exact, sigma1, h1, h2, future_weight):
    args.h1_mult = 1.0
    args.h2_mult = h2 / h1 if h1 > 0 else 0.0
    args.combined_s7_future_weight = float(future_weight)
    np.random.seed(args.seed)
    r = exp.run_pair_stream(A, V_exact, sigma1, args, "combined_s7",
                            half_win=h1, sliding=True)
    last = r["rows"][-1]
    return last["cos1"] ** 2, last["cos2"] ** 2, last["oracle_proj_norm1"], last["oracle_proj_norm2"], len(r["rows"])


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
        # cos² shown alongside the per-round-final oracle bound (best achievable
        # cos² for any direction in rowspace(M_gain) at that round).
        print(f"{'T':>4} {'h2/T':>5}  "
              f"{'cmb R':>5} {'cmb c1²':>7} {'cmb c2²':>7} {'cmb oR1':>7} {'cmb oR2':>7}  "
              f"{'w':>5} {'s7 R':>4} {'s7 c1²':>7} {'s7 c2²':>7} {'s7 oR1':>7} {'s7 oR2':>7}  "
              f"{'Δc1²':>7} {'Δc2²':>7}")
        print("-" * 140)
        for T in T_LIST:
            h2 = T - H1_S7
            ratio = h2 / T if T > 0 else 0.0
            cmb_c1, cmb_c2, cmb_o1, cmb_o2, cmb_R = run_combined_at_T(args, A, V_exact, sigma1, T)
            for w in WEIGHTS:
                s7_c1, s7_c2, s7_o1, s7_o2, s7_R = run_s7(args, A, V_exact, sigma1, H1_S7, h2, w)
                d1 = s7_c1 - cmb_c1
                d2 = s7_c2 - cmb_c2
                print(f"{T:>4} {ratio:>5.3f}  "
                      f"{cmb_R:>5} {cmb_c1:>7.4f} {cmb_c2:>7.4f} {cmb_o1:>7.4f} {cmb_o2:>7.4f}  "
                      f"{w:>5} {s7_R:>4} {s7_c1:>7.4f} {s7_c2:>7.4f} {s7_o1:>7.4f} {s7_o2:>7.4f}  "
                      f"{d1:>+7.4f} {d2:>+7.4f}")
                if h2 == 0:
                    break  # weight irrelevant at h2=0
        print()


if __name__ == "__main__":
    main()
