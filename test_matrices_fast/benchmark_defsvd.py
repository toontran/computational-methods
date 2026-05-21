"""Benchmark DefSVD (and iSVD / Combined baselines) across the standard
matrix family used by benchmarks.txt.

Matches the (n, r, win, seed) defaults of run_all_experiments.py so the
align / rel_err_sval columns are directly comparable to the existing rows
in test_matrices_fast/benchmarks.txt.

Usage:
    python benchmark_defsvd.py
    python benchmark_defsvd.py --matrices static-cex mixed-tail-sharp
    python benchmark_defsvd.py --output benchmarks_defsvd.txt

Methods:
    isvd-ref          : reference iSVD (top-r SVD of [B; A_w]).
    defsvd-symm       : C2 with symmetric pre-deflation (carry + window).
    defsvd-carryonly  : C2 with carry-only pre-deflation (FD spirit).
    combined-ref      : optimizer-based direction policy (slow reference).
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

from cex_structured_new_py import (  # noqa: E402
    window_iter_basis_streaming,
    projected_subspace_svd,
)
from cex_restricted_space_probe import generate_matrix_input  # noqa: E402


BENCHMARK_MATRICES = [
    "static-cex",
    "diffuse-diffuse",
    "mixed-tail-soft",
    "mixed-tail-balanced",
    "mixed-tail-sharp",
    "residual-spiky-shocks",
    "alternative-data-signals",
    "futures-term-structure",
    "crowded-strategy",
    "rates-cross-currency",
    "options-vol-surface",
    "risk-residual-panel",
    "macro-factor-panel",
    "realized-vol-corr",
    "etf-basket-basis",
    "execution-cost-slippage",
    "stat-arb-spreads",
    "intraday-liquidity-shape",
]


def _new_state(V_hat, s_new, V_r, S_r, end_row, prev_sketch, extra=None):
    s_new = np.asarray(s_new).reshape(-1)
    state = {
        "V": V_hat,
        "s": s_new,
        "s2": s_new ** 2,
        "rows_seen": end_row,
        "prev_basis": V_r,
        "prev_s2": (np.diag(S_r) ** 2) if S_r is not None else None,
        "prev_sketch": prev_sketch,
        "H": np.full(s_new.shape, np.nan),
        "score": s_new ** 2,
    }
    if extra:
        state.update(extra)
    return state


def run_streaming(A, V_exact, sigma1, r, win, mode, deflate_window, l=1):
    n = A.shape[1]
    mA = A.shape[0]
    t0 = time.time()

    V_r = None
    S_r = None
    state = None
    delta_sum_sq = 0.0
    delta_prev_sq = 0.0   # used by DefSVD-CumOrthInflate (previous block's σ_{r+1}²)
    A_seen = np.zeros((0, n))

    for start0 in range(0, mA, win):
        end0 = min(start0 + win, mA)
        A_block = A[start0:end0, :]

        prev_sketch = None if (V_r is None or S_r is None) else S_r @ V_r.T
        M_gain = A_block if prev_sketch is None else np.vstack([prev_sketch, A_block])

        if mode == "Combined":
            V_hat, s_new, _, _, state_new = window_iter_basis_streaming(
                A_block, r, n, state, V_r, 8, 200, 1e-8
            )
            state_new["prev_sketch"] = prev_sketch
            V_hat, s_new = projected_subspace_svd(M_gain, V_hat)
            state_new["V"] = V_hat
            state_new["s"] = s_new
            state_new["s2"] = s_new ** 2

        elif mode == "iSVD":
            _, s, Vh = la.svd(M_gain, full_matrices=False, lapack_driver="gesdd")
            rr = min(r, len(s))
            V_hat = Vh.T[:, :rr]
            s_new = s[:rr]
            state_new = _new_state(V_hat, s_new, V_r, S_r, end0, prev_sketch)

        elif mode == "FD":
            # Liberty FD: same basis as iSVD, per-step σ²-shift on the carry.
            _, s, Vh = la.svd(M_gain, full_matrices=False, lapack_driver="gesdd")
            rr = min(r, len(s))
            V_hat = Vh.T[:, :rr]
            delta_sq = float(s[rr] ** 2) if len(s) > rr else 0.0
            s_new = np.sqrt(np.maximum(s[:rr] ** 2 - delta_sq, 0.0))
            state_new = _new_state(V_hat, s_new, V_r, S_r, end0, prev_sketch,
                                   extra={"fd_delta_sq": delta_sq})

        elif mode == "DefSVD":
            if V_r is None or S_r is None:
                B_def = None
            else:
                s_sketch = np.diag(S_r)
                s_sketch_def = np.sqrt(np.maximum(s_sketch ** 2 - delta_sum_sq, 0.0))
                B_def = (V_r * s_sketch_def).T

            if deflate_window:
                U_w, s_w, Vh_w = la.svd(A_block, full_matrices=False, lapack_driver="gesdd")
                s_w_def = np.sqrt(np.maximum(s_w ** 2 - delta_sum_sq, 0.0))
                A_w_def = (U_w * s_w_def) @ Vh_w
            else:
                A_w_def = A_block

            M_def = A_w_def if B_def is None else np.vstack([B_def, A_w_def])
            _, s_def, Vh_def = la.svd(M_def, full_matrices=False, lapack_driver="gesdd")
            V_def_full = Vh_def.T
            rr = min(r, V_def_full.shape[1])
            V_hat_raw = V_def_full[:, :rr]
            delta_this_sq = float(s_def[rr] ** 2) if len(s_def) > rr else 0.0
            V_hat, s_new = projected_subspace_svd(M_gain, V_hat_raw)
            state_new = _new_state(
                V_hat, s_new, V_r, S_r, end0, prev_sketch,
                extra={"delta_sum_sq_in": delta_sum_sq, "delta_this_sq": delta_this_sq},
            )
            delta_sum_sq += delta_this_sq

        elif mode == "DefSVD-RecalCarry":
            # Direction-aware carry deflation. Identical to defsvd-carryonly
            # (window pristine, "fresh data is sacred") except the carry's
            # total deflation mass (r * delta_sum_sq) is redistributed across
            # carried directions by reinforcement Delta_i = ||A_block @ v_i||^2.
            if V_r is None or S_r is None:
                B_def = None
                recal_extra = {}
            else:
                # C4 reinforcement of each carried direction (clean on V_r,
                # computed before any basis rotation).
                delta_i = np.sum((A_block @ V_r) ** 2, axis=0)  # length r
                mean_delta = float(np.mean(delta_i))
                if mean_delta == 0.0:
                    delta_tilde = np.zeros_like(delta_i)
                else:
                    delta_tilde = delta_i / mean_delta
                w = 1.0 / (1.0 + delta_tilde)
                w = w / np.sum(w)                       # sum w = 1
                r_carry = delta_i.shape[0]
                defl = (r_carry * delta_sum_sq) * w      # sum defl = r * delta_sum_sq
                s_sketch = np.diag(S_r)
                s_sketch_def = np.sqrt(np.maximum(s_sketch ** 2 - defl, 0.0))
                B_def = (V_r * s_sketch_def).T
                recal_extra = {
                    "delta_i": delta_i.tolist(),
                    "recal_weights": w.tolist(),
                    "defl": defl.tolist(),
                }

            # WINDOW STAYS PRISTINE (ignore deflate_window for this mode).
            A_w_def = A_block

            M_def = A_w_def if B_def is None else np.vstack([B_def, A_w_def])
            _, s_def, Vh_def = la.svd(M_def, full_matrices=False, lapack_driver="gesdd")
            V_def_full = Vh_def.T
            rr = min(r, V_def_full.shape[1])
            V_hat_raw = V_def_full[:, :rr]
            delta_this_sq = float(s_def[rr] ** 2) if len(s_def) > rr else 0.0
            V_hat, s_new = projected_subspace_svd(M_gain, V_hat_raw)
            extra = {"delta_sum_sq_in": delta_sum_sq, "delta_this_sq": delta_this_sq}
            extra.update(recal_extra)
            state_new = _new_state(
                V_hat, s_new, V_r, S_r, end0, prev_sketch, extra=extra,
            )
            delta_sum_sq += delta_this_sq

        elif mode == "DefSVD-OrthDef":
            # Same as DefSVD-carryonly but window is SVD'd in the subspace orthogonal to V_r before deflation.
            if V_r is None or S_r is None:
                B_def = None
            else:
                s_sketch = np.diag(S_r)
                s_sketch_def = np.sqrt(np.maximum(s_sketch ** 2 - delta_sum_sq, 0.0))
                B_def = (V_r * s_sketch_def).T

            if V_r is None:
                A_block_outside = A_block
            else:
                A_block_outside = A_block - (A_block @ V_r) @ V_r.T

            U_o, s_o, Vh_o = la.svd(A_block_outside, full_matrices=False, lapack_driver="gesdd")
            s_o_def = np.sqrt(np.maximum(s_o ** 2 - delta_sum_sq, 0.0))
            A_w_def = (U_o * s_o_def) @ Vh_o

            M_def = A_w_def if B_def is None else np.vstack([B_def, A_w_def])
            _, s_def, Vh_def = la.svd(M_def, full_matrices=False, lapack_driver="gesdd")
            V_def_full = Vh_def.T
            rr = min(r, V_def_full.shape[1])
            V_hat_raw = V_def_full[:, :rr]
            delta_this_sq = float(s_def[rr] ** 2) if len(s_def) > rr else 0.0
            V_hat, s_new = projected_subspace_svd(M_gain, V_hat_raw)
            state_new = _new_state(
                V_hat, s_new, V_r, S_r, end0, prev_sketch,
                extra={
                    "delta_sum_sq_in": delta_sum_sq,
                    "delta_this_sq": delta_this_sq,
                    "n_outside_dirs": int(s_o.size),
                },
            )
            delta_sum_sq += delta_this_sq

        elif mode == "DefSVD-CumOrthDef":
            # Same as DefSVD-OrthDef but per-outside-direction deflation uses
            # nonlocal history: each outside direction is deflated by
            # max(delta_sum_sq - E_prior_j, 0), where E_prior_j is the energy
            # of that direction already explained by previously-seen rows.
            if V_r is None or S_r is None:
                B_def = None
            else:
                s_sketch = np.diag(S_r)
                s_sketch_def = np.sqrt(np.maximum(s_sketch ** 2 - delta_sum_sq, 0.0))
                B_def = (V_r * s_sketch_def).T

            if V_r is None:
                A_block_outside = A_block
            else:
                A_block_outside = A_block - (A_block @ V_r) @ V_r.T

            U_o, s_o, Vh_o = la.svd(A_block_outside, full_matrices=False, lapack_driver="gesdd")
            V_o = Vh_o.T

            if A_seen.shape[0] == 0:
                E_prior = np.zeros(s_o.shape[0])
            else:
                E_prior = np.sum((A_seen @ V_o) ** 2, axis=0)
            delta_per_j = np.maximum(delta_sum_sq - E_prior, 0.0)
            s_o_def = np.sqrt(np.maximum(s_o ** 2 - delta_per_j, 0.0))

            total_deflation_per_j = E_prior + delta_per_j
            import sys as _sys
            exceed_mask = E_prior >= delta_sum_sq
            n_exceed = int(exceed_mask.sum())
            if n_exceed > 0:
                max_extra_defl = float(delta_per_j[exceed_mask].max())
                sum_extra_defl = float(delta_per_j[exceed_mask].sum())
            else:
                max_extra_defl = 0.0
                sum_extra_defl = 0.0
            _sys.stderr.write(
                f"# [CumOrthDef] Δ²={delta_sum_sq:.4g}  E_prior min={float(E_prior.min()):.4g} max={float(E_prior.max()):.4g}  "
                f"n_dirs_with_E_prior>=Δ²={n_exceed}/{len(E_prior)}  "
                f"extra_defl_on_those: max={max_extra_defl:.4g} sum={sum_extra_defl:.4g}  "
                f"max_gap={float((total_deflation_per_j - delta_sum_sq).max()):.4g}\n"
            )

            A_w_def = (U_o * s_o_def) @ Vh_o

            M_def = A_w_def if B_def is None else np.vstack([B_def, A_w_def])
            _, s_def, Vh_def = la.svd(M_def, full_matrices=False, lapack_driver="gesdd")
            V_def_full = Vh_def.T
            rr = min(r, V_def_full.shape[1])
            V_hat_raw = V_def_full[:, :rr]
            delta_this_sq = float(s_def[rr] ** 2) if len(s_def) > rr else 0.0
            V_hat, s_new = projected_subspace_svd(M_gain, V_hat_raw)
            state_new = _new_state(
                V_hat, s_new, V_r, S_r, end0, prev_sketch,
                extra={
                    "delta_sum_sq_in": delta_sum_sq,
                    "delta_this_sq": delta_this_sq,
                    "n_outside_dirs": int(s_o.size),
                },
            )
            delta_sum_sq += delta_this_sq

        elif mode == "DefSVD-CumOrthInflate":
            # ASYMMETRIC mirror of CumOrthDef: inflate ONLY the window's
            # outside-V_r component by max(δ_prev² - E_prior_j, 0). Carry
            # left untouched. The window-only asymmetry parallels the
            # carry-only deflate asymmetry that made FD-spirit DefSVD work.
            B_def = None if (V_r is None or S_r is None) else (V_r * np.diag(S_r)).T

            if V_r is None:
                A_block_outside = A_block
            else:
                A_block_outside = A_block - (A_block @ V_r) @ V_r.T

            U_o, s_o, Vh_o = la.svd(A_block_outside, full_matrices=False, lapack_driver="gesdd")
            V_o = Vh_o.T

            if A_seen.shape[0] == 0:
                E_prior = np.zeros(s_o.shape[0])
            else:
                E_prior = np.sum((A_seen @ V_o) ** 2, axis=0)

            inflate_per_j = np.maximum(delta_prev_sq - E_prior, 0.0)
            s_o_inf = np.sqrt(s_o ** 2 + inflate_per_j)
            A_w_inf = (U_o * s_o_inf) @ Vh_o

            M_def = A_w_inf if B_def is None else np.vstack([B_def, A_w_inf])
            _, s_def, Vh_def = la.svd(M_def, full_matrices=False, lapack_driver="gesdd")
            V_def_full = Vh_def.T
            rr = min(r, V_def_full.shape[1])
            V_hat_raw = V_def_full[:, :rr]
            delta_this_sq = float(s_def[rr] ** 2) if len(s_def) > rr else 0.0
            V_hat, s_new = projected_subspace_svd(M_gain, V_hat_raw)
            state_new = _new_state(
                V_hat, s_new, V_r, S_r, end0, prev_sketch,
                extra={
                    "delta_prev_sq_in": delta_prev_sq,
                    "delta_this_sq": delta_this_sq,
                    "max_inflate_per_j": float(inflate_per_j.max()) if inflate_per_j.size else 0.0,
                },
            )
            delta_prev_sq = delta_this_sq

        elif mode == "DefSVD-CumOrthInflate-Symm":
            # SYMMETRIC mirror of CumOrthDef: inflate BOTH carry and
            # outside σ² to a uniform per-direction target of δ_prev².
            # Carry uses per-direction E_prior on V_r columns; outside
            # uses per-direction E_prior on V_o columns. Mathematically
            # this is the "every direction reaches at least δ_prev²
            # total presence (history + this block's inflate)" mirror of
            # CumOrthDef's "every direction loses at most Δ² total".
            if V_r is None or S_r is None:
                B_def = None
            else:
                s_sketch = np.diag(S_r)
                if A_seen.shape[0] == 0:
                    E_prior_carry = np.zeros(s_sketch.shape[0])
                else:
                    E_prior_carry = np.sum((A_seen @ V_r) ** 2, axis=0)
                inflate_carry_per_j = np.maximum(delta_prev_sq - E_prior_carry, 0.0)
                s_sketch_inf = np.sqrt(s_sketch ** 2 + inflate_carry_per_j)
                B_def = (V_r * s_sketch_inf).T

            if V_r is None:
                A_block_outside = A_block
            else:
                A_block_outside = A_block - (A_block @ V_r) @ V_r.T

            U_o, s_o, Vh_o = la.svd(A_block_outside, full_matrices=False, lapack_driver="gesdd")
            V_o = Vh_o.T

            if A_seen.shape[0] == 0:
                E_prior_out = np.zeros(s_o.shape[0])
            else:
                E_prior_out = np.sum((A_seen @ V_o) ** 2, axis=0)
            inflate_out_per_j = np.maximum(delta_prev_sq - E_prior_out, 0.0)
            s_o_inf = np.sqrt(s_o ** 2 + inflate_out_per_j)
            A_w_inf = (U_o * s_o_inf) @ Vh_o

            M_def = A_w_inf if B_def is None else np.vstack([B_def, A_w_inf])
            _, s_def, Vh_def = la.svd(M_def, full_matrices=False, lapack_driver="gesdd")
            V_def_full = Vh_def.T
            rr = min(r, V_def_full.shape[1])
            V_hat_raw = V_def_full[:, :rr]
            delta_this_sq = float(s_def[rr] ** 2) if len(s_def) > rr else 0.0
            V_hat, s_new = projected_subspace_svd(M_gain, V_hat_raw)
            state_new = _new_state(
                V_hat, s_new, V_r, S_r, end0, prev_sketch,
                extra={
                    "delta_prev_sq_in": delta_prev_sq,
                    "delta_this_sq": delta_this_sq,
                    "max_inflate_carry": float(inflate_carry_per_j.max()) if V_r is not None and inflate_carry_per_j.size else 0.0,
                    "max_inflate_out": float(inflate_out_per_j.max()) if inflate_out_per_j.size else 0.0,
                },
            )
            delta_prev_sq = delta_this_sq

        elif mode == "DefSVD-CumOrthInflate-CarryToo":
            # Forces non-trivial carry inflation: carry gets a UNIFORM
            # δ_prev² boost (no E_prior clamp, since carry's E_prior
            # always swamps δ_prev² and would zero out the inflation).
            # Outside still uses per-direction max(δ_prev² - E_prior_j, 0).
            if V_r is None or S_r is None:
                B_def = None
            else:
                s_sketch = np.diag(S_r)
                s_sketch_inf = np.sqrt(s_sketch ** 2 + delta_prev_sq)
                B_def = (V_r * s_sketch_inf).T

            if V_r is None:
                A_block_outside = A_block
            else:
                A_block_outside = A_block - (A_block @ V_r) @ V_r.T

            U_o, s_o, Vh_o = la.svd(A_block_outside, full_matrices=False, lapack_driver="gesdd")
            V_o = Vh_o.T

            if A_seen.shape[0] == 0:
                E_prior_out = np.zeros(s_o.shape[0])
            else:
                E_prior_out = np.sum((A_seen @ V_o) ** 2, axis=0)
            inflate_out_per_j = np.maximum(delta_prev_sq - E_prior_out, 0.0)
            s_o_inf = np.sqrt(s_o ** 2 + inflate_out_per_j)
            A_w_inf = (U_o * s_o_inf) @ Vh_o

            M_def = A_w_inf if B_def is None else np.vstack([B_def, A_w_inf])
            _, s_def, Vh_def = la.svd(M_def, full_matrices=False, lapack_driver="gesdd")
            V_def_full = Vh_def.T
            rr = min(r, V_def_full.shape[1])
            V_hat_raw = V_def_full[:, :rr]
            delta_this_sq = float(s_def[rr] ** 2) if len(s_def) > rr else 0.0
            V_hat, s_new = projected_subspace_svd(M_gain, V_hat_raw)
            state_new = _new_state(
                V_hat, s_new, V_r, S_r, end0, prev_sketch,
                extra={
                    "delta_prev_sq_in": delta_prev_sq,
                    "delta_this_sq": delta_this_sq,
                    "carry_inflate": float(delta_prev_sq),
                },
            )
            delta_prev_sq = delta_this_sq

        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        A_seen = np.vstack([A_seen, A_block])
        V_r = V_hat
        S_r = np.diag(s_new)
        state = state_new

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
    parser.add_argument("--matrices", nargs="+", default=BENCHMARK_MATRICES)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--skip-combined", action="store_true",
        help="Skip the Combined optimizer-based reference (it is slow).",
    )
    args = parser.parse_args()

    methods = [
        ("isvd-ref", "iSVD", True),
        ("fd-ref", "FD", True),
        ("defsvd-symm", "DefSVD", True),
        ("defsvd-carryonly", "DefSVD", False),
        ("defsvd-recal-carry", "DefSVD-RecalCarry", False),
        ("defsvd-orth", "DefSVD-OrthDef", True),
        ("defsvd-cum-orth", "DefSVD-CumOrthDef", True),
        ("defsvd-cum-orth-inflate", "DefSVD-CumOrthInflate", True),
        ("defsvd-cum-orth-inflate-symm", "DefSVD-CumOrthInflate-Symm", True),
        ("defsvd-cum-orth-inflate-carrytoo", "DefSVD-CumOrthInflate-CarryToo", True),
    ]
    if not args.skip_combined:
        methods.append(("combined-ref", "Combined", True))

    header = ("matrix", "method", "mean_align", "mean_relerr_sval", "elapsed")
    print("\t".join(header), flush=True)
    rows = []

    for matrix in args.matrices:
        try:
            np.random.seed(args.seed)
            A, V_exact, _, sigma1 = generate_matrix_input(
                matrix, n=args.n, preset=args.preset, seed=args.seed,
            )
        except (ValueError, RuntimeError, TypeError) as e:
            print(f"# SKIP {matrix}: {e}", file=sys.stderr, flush=True)
            continue

        for label, mode, deflate_window in methods:
            np.random.seed(args.seed)
            try:
                align, rel_err, elapsed = run_streaming(
                    A, V_exact, sigma1, args.rank, args.win, mode, deflate_window,
                )
            except Exception as e:  # noqa: BLE001
                print(f"# FAIL {matrix} {label}: {e}", file=sys.stderr, flush=True)
                continue
            row = (matrix, label, f"{align:.6f}", f"{rel_err:.8f}", f"{elapsed:.3f}")
            print("\t".join(row), flush=True)
            rows.append(row)

    if args.output:
        with open(args.output, "a") as f:
            for row in rows:
                f.write("\t".join(row) + "\n")


if __name__ == "__main__":
    main()
