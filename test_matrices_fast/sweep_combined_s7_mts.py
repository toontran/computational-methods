"""Sweep combined vs combined_s7 across mixed-tail-{sharp,balanced,soft}
for varying peek/current ratios. h1 = round(half_win * h1_mult),
h2 = round(half_win * h2_mult). At h2=0, combined and combined_s7 must match.
"""

from __future__ import annotations

import sys
import time
import numpy as np

import cex_restricted_space_probe as probe
import half_window_sliding_hmean_experiment as exp


HALF_WIN = 32
SEED = 0
ROW_SHUFFLE_SEED = 0
N = 1024

MATRICES = ("mixed-tail-sharp", "mixed-tail-balanced", "mixed-tail-soft")

# (h1_mult, h2_mult) — at h2_mult=0 we expect exact equivalence; otherwise,
# combined_s7 includes the peek window in score, combined ignores it.
RATIOS = (
    (1.0, 0.0),
    (1.0, 0.25),
    (1.0, 0.5),
    (1.0, 1.0),
    (1.0, 1.5),
    (1.0, 2.0),
)


def run(args, A, V_exact, sigma1, policy, h1_mult, h2_mult):
    args.h1_mult = h1_mult
    args.h2_mult = h2_mult
    np.random.seed(args.seed)
    r = exp.run_pair_stream(A, V_exact, sigma1, args, policy,
                            half_win=HALF_WIN, sliding=True)
    last = r["rows"][-1]
    return {
        "cos1_sq": last["cos1"] ** 2,
        "cos2_sq": last["cos2"] ** 2,
        "n_rounds": len(r["rows"]),
        "elapsed": r["elapsed"],
    }


def main():
    sys.argv = ["x", "--seed", str(SEED)]
    args = exp.parse_args()
    args.row_shuffle_seed = ROW_SHUFFLE_SEED

    print(f"half_win={HALF_WIN}  seed={SEED}  row_shuffle_seed={ROW_SHUFFLE_SEED}  n={N}")
    print()

    rows = []
    for matrix in MATRICES:
        A, V_exact, _, sigma1 = probe.generate_matrix_input(
            matrix, n=N, preset="fast", seed=0,
            shuffle_rows=True, row_shuffle_seed=ROW_SHUFFLE_SEED,
        )
        for h1m, h2m in RATIOS:
            h1 = max(1, int(round(HALF_WIN * h1m)))
            h2 = max(0, int(round(HALF_WIN * h2m)))
            t0 = time.time()
            rec_c = run(args, A, V_exact, sigma1, "combined", h1m, h2m)
            rec_s = run(args, A, V_exact, sigma1, "combined_s7", h1m, h2m)
            rows.append({
                "matrix": matrix, "h1": h1, "h2": h2,
                "combined_c1": rec_c["cos1_sq"], "combined_c2": rec_c["cos2_sq"],
                "s7_c1": rec_s["cos1_sq"], "s7_c2": rec_s["cos2_sq"],
                "n_rounds": rec_c["n_rounds"],
                "elapsed_s": time.time() - t0,
            })

    # Print table
    print(f"{'matrix':<20} {'h1':>3} {'h2':>3} {'rounds':>6}  "
          f"{'comb cos1²':>11} {'comb cos2²':>11}  "
          f"{'s7 cos1²':>9} {'s7 cos2²':>9}  "
          f"{'Δcos1²':>8} {'Δcos2²':>8}")
    print("-" * 113)
    for r in rows:
        d1 = r["s7_c1"] - r["combined_c1"]
        d2 = r["s7_c2"] - r["combined_c2"]
        print(f"{r['matrix']:<20} {r['h1']:>3} {r['h2']:>3} {r['n_rounds']:>6}  "
              f"{r['combined_c1']:>11.4f} {r['combined_c2']:>11.4f}  "
              f"{r['s7_c1']:>9.4f} {r['s7_c2']:>9.4f}  "
              f"{d1:>+8.4f} {d2:>+8.4f}")

    # Sanity: h2=0 rows must have Δ ≈ 0
    print()
    h2_zero_rows = [r for r in rows if r["h2"] == 0]
    max_d1 = max(abs(r["s7_c1"] - r["combined_c1"]) for r in h2_zero_rows)
    max_d2 = max(abs(r["s7_c2"] - r["combined_c2"]) for r in h2_zero_rows)
    print(f"h2=0 sanity: max |Δcos1²|={max_d1:.2e}  max |Δcos2²|={max_d2:.2e}  "
          f"({'PASS' if max(max_d1, max_d2) < 1e-12 else 'FAIL'})")


if __name__ == "__main__":
    main()
