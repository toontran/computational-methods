"""Fair comparison: combined (block size = h1+h2) vs combined_s7 (h1 commit + h2 peek).
Both see the same total row budget T = h1+h2 per round in their score, but
combined commits all T rows while combined_s7 commits h1 and peeks h2.
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

MATRICES = ("mixed-tail-sharp", "mixed-tail-balanced", "mixed-tail-soft")

# Fix the s7 commit half_win = 32. Vary h2 to span peek ratio 0..2/3.
H1_S7 = 32
H2_LIST = (0, 8, 16, 32, 48, 64)


def run_combined_at_T(args, A, V_exact, sigma1, T):
    # combined with block size = T (no peek): half_win=T, h1_mult=1.0, h2_mult=0.
    args.h1_mult = 1.0
    args.h2_mult = 0.0
    np.random.seed(args.seed)
    r = exp.run_pair_stream(A, V_exact, sigma1, args, "combined",
                            half_win=T, sliding=True)
    last = r["rows"][-1]
    return {
        "cos1_sq": last["cos1"] ** 2,
        "cos2_sq": last["cos2"] ** 2,
        "n_rounds": len(r["rows"]),
    }


def run_s7(args, A, V_exact, sigma1, h1, h2):
    # combined_s7 with commit h1 and peek h2: half_win=h1, h1_mult=1.0,
    # h2_mult=h2/h1.
    args.h1_mult = 1.0
    args.h2_mult = h2 / h1 if h1 > 0 else 0.0
    np.random.seed(args.seed)
    r = exp.run_pair_stream(A, V_exact, sigma1, args, "combined_s7",
                            half_win=h1, sliding=True)
    last = r["rows"][-1]
    return {
        "cos1_sq": last["cos1"] ** 2,
        "cos2_sq": last["cos2"] ** 2,
        "n_rounds": len(r["rows"]),
    }


def main():
    sys.argv = ["x", "--seed", str(SEED)]
    args = exp.parse_args()
    args.row_shuffle_seed = ROW_SHUFFLE_SEED

    print(f"seed={SEED}  row_shuffle_seed={ROW_SHUFFLE_SEED}  n={N}  s7_h1={H1_S7}")
    print()

    rows = []
    for matrix in MATRICES:
        A, V_exact, _, sigma1 = probe.generate_matrix_input(
            matrix, n=N, preset="fast", seed=0,
            shuffle_rows=True, row_shuffle_seed=ROW_SHUFFLE_SEED,
        )
        for h2 in H2_LIST:
            T = H1_S7 + h2
            ratio = h2 / T
            t0 = time.time()
            rec_c = run_combined_at_T(args, A, V_exact, sigma1, T)
            rec_s = run_s7(args, A, V_exact, sigma1, H1_S7, h2)
            rows.append({
                "matrix": matrix, "h1": H1_S7, "h2": h2, "T": T, "ratio": ratio,
                "comb_c1": rec_c["cos1_sq"], "comb_c2": rec_c["cos2_sq"],
                "comb_R": rec_c["n_rounds"],
                "s7_c1": rec_s["cos1_sq"], "s7_c2": rec_s["cos2_sq"],
                "s7_R": rec_s["n_rounds"],
                "elapsed": time.time() - t0,
            })

    print(f"{'matrix':<20} {'h1':>3} {'h2':>3} {'T':>3} {'h2/T':>5}  "
          f"{'comb R':>6} {'comb c1²':>9} {'comb c2²':>9}  "
          f"{'s7 R':>4} {'s7 c1²':>7} {'s7 c2²':>7}  "
          f"{'Δc1²':>7} {'Δc2²':>7}")
    print("-" * 116)
    for r in rows:
        d1 = r["s7_c1"] - r["comb_c1"]
        d2 = r["s7_c2"] - r["comb_c2"]
        print(f"{r['matrix']:<20} {r['h1']:>3} {r['h2']:>3} {r['T']:>3} {r['ratio']:>5.3f}  "
              f"{r['comb_R']:>6} {r['comb_c1']:>9.4f} {r['comb_c2']:>9.4f}  "
              f"{r['s7_R']:>4} {r['s7_c1']:>7.4f} {r['s7_c2']:>7.4f}  "
              f"{d1:>+7.4f} {d2:>+7.4f}")


if __name__ == "__main__":
    main()
