"""Carry-trajectory probe (INFRA-06).

Per backlog item INFRA-06 in summary/overview/score_family_workflow.txt §5,
this probe runs the streaming SVD block-by-block on each matrix in the §6
suite and dumps the carry-state trajectory:

  s_top                    : state.s[0]                  (top sval of carry)
  sk_F2_low                : sum(state.s ** 2)           (rank-r CARRY F-norm²)
  spectral_concentration   : s[0]² / sum(s_i²)
  carry_drift              : ‖V_t V_t^T − V_{t−1} V_{t−1}^T‖_F  (NaN at b1)

Output:
  summary/infra_carry_trajectory/{matrix}_win{2*half_win}.csv
  summary/infra_carry_trajectory/trajectories.png

This directly informs FAM-04 (carry-confidence multipliers): different
multiplier shapes (s_top², spectral concentration, carry-drift) make different
sense depending on the trajectory profile — the probe surfaces the choice.

Convention for `carry_drift` at block 1: V_{t-1} is undefined (no prior carry),
so we emit NaN.

Run from test_matrices_fast/:
    python carry_trajectory_probe.py
"""

import argparse
import csv
import os
import time

import numpy as np

import cex_restricted_space_probe as probe
from second_slot_tail_bias_diagnostic import make_state


# Matrix suite mirrors the §6 table of score_design_overview.txt and the
# 7-matrix S6 sweep (summary/bench_matrix_sweep_r_sk_g_S6/). risk-residual-panel
# is included per the INFRA-06 acceptance ("if listed" — it appears in the §7
# table as a peer regime to residual-spiky-shocks).
DEFAULT_MATRICES = [
    "mixed-tail-sharp",
    "mixed-tail-balanced",
    "mixed-tail-soft",
    "static-cex",
    "diffuse-diffuse",
    "etf-basket-basis",
    "residual-spiky-shocks",
    "risk-residual-panel",
]


# --------------- streaming driver ---------------

def run_carry_trajectory(A, V_exact, args, half_win, policy="future_hmean_r_sk_g"):
    """Drive the streaming algorithm in sliding mode and capture carry per block.

    Implementation note: the carry trajectory is mostly invariant to the slot-2
    score (per INFRA-06). What matters is that the rank-r CARRY update rule
    (iSVD-style: state.V, state.s = right-SVD of (V_selected.T @ M_gain)) is
    consistently applied. We use future_hmean_r_sk_g/S6 to match the bench
    convention; the resulting state per block is what FAM-04 will multiply.
    """
    work_dtype = np.float64 if args.dtype == "float64" else np.float32
    n = A.shape[0]
    rank = int(args.rank)

    state = None
    prev_VVt = None
    rows = []
    t0 = time.time()

    step = half_win  # sliding mode
    pair_count = 0
    for start0 in range(0, n - half_win, step):
        mid0 = start0 + half_win
        end0 = min(mid0 + half_win, n)
        if end0 - mid0 < half_win:
            break
        pair_count += 1
        block_id = pair_count
        A_half1 = np.asarray(A[start0:mid0, :], dtype=work_dtype)

        # Build M_gain = [B_top; A_cur] (same as the experiment harness).
        if state is None:
            M_gain = A_half1
            V_init = probe.row_norm_seed(A_half1, rank)
            rows_seen = A_half1.shape[0]
        else:
            B_top = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
            M_gain = np.vstack([B_top, A_half1]).astype(work_dtype, copy=False)
            V_init = probe.row_norm_seed(A_half1, rank)
            rows_seen = state["rows_seen"] + A_half1.shape[0]

        # We only need V_selected — the actual choice of slot-2 is not what
        # drives the carry shape (the carry update is a left-projection SVD).
        # Use the iSVD policy as the simplest deterministic update so the
        # trajectory is reproducible across runs and not noisy from the
        # candidate-pool optimizer.
        Mg = np.asarray(M_gain, dtype=np.float64)
        _, _, Vh = np.linalg.svd(Mg, full_matrices=False)
        V_selected = np.ascontiguousarray(Vh[:rank, :].T)

        # Use combined-style scoring details just to fill the H/score arrays
        # that make_state expects. (Values are not used downstream by us.)
        H_selected = np.zeros(rank, dtype=float)
        score_selected = np.zeros(rank, dtype=float)

        # Carry update: rank-r truncated left-projected SVD of M_gain restricted
        # to span(V_selected) — same path the bench uses.
        state, _V_r, _s_new = make_state(
            M_gain, V_selected, H_selected, score_selected, rows_seen
        )

        # Extract carry metrics from the freshly-updated state.
        s = np.asarray(state["s"], dtype=np.float64).reshape(-1)
        V = np.asarray(state["V"], dtype=np.float64)

        s_top = float(s[0]) if s.size > 0 else float("nan")
        sk_F2_low = float(np.sum(s * s))
        if sk_F2_low > 0.0:
            spectral_concentration = float(s[0] * s[0] / sk_F2_low) if s.size > 0 else float("nan")
        else:
            spectral_concentration = float("nan")

        VVt = V @ V.T
        if prev_VVt is None:
            carry_drift = float("nan")
        else:
            carry_drift = float(np.linalg.norm(VVt - prev_VVt, ord="fro"))
        prev_VVt = VVt

        rows.append({
            "block": block_id,
            "rows_seen": rows_seen,
            "s_top": s_top,
            "sk_F2_low": sk_F2_low,
            "spectral_concentration": spectral_concentration,
            "carry_drift": carry_drift,
        })

    elapsed = time.time() - t0
    return rows, elapsed


# --------------- I/O ---------------

def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_summary_plot(per_matrix_rows, out_png, win_total):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrices = list(per_matrix_rows.keys())
    n_mat = len(matrices)
    # Layout: 2 columns of subplots, one row per matrix; 4 lines per subplot.
    # Use a wide grid: 4 metrics × n_matrices, with a single subplot per
    # matrix and twin axes for s_top vs sk_F2_low (different scales).
    n_cols = 2
    n_rows = (n_mat + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0 * n_cols, 3.0 * n_rows),
                             squeeze=False)

    for idx, m in enumerate(matrices):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]
        rows = per_matrix_rows[m]
        if not rows:
            ax.set_title(f"{m} (no data)")
            continue
        blocks = [row["block"] for row in rows]
        s_top = [row["s_top"] for row in rows]
        sk_F2 = [row["sk_F2_low"] for row in rows]
        spec_c = [row["spectral_concentration"] for row in rows]
        drift = [row["carry_drift"] for row in rows]

        # Left axis: s_top and sk_F2_low (raw scale)
        l1, = ax.plot(blocks, s_top, "-o", color="tab:blue", markersize=3, label="s_top")
        l2, = ax.plot(blocks, sk_F2, "-s", color="tab:orange", markersize=3, label="sk_F2_low")
        ax.set_xlabel("block")
        ax.set_ylabel("magnitude (s_top, sk_F2_low)")
        ax.set_title(m)
        ax.grid(True, alpha=0.3)

        # Right axis: spectral concentration (∈[1/r, 1]) and carry_drift (∈[0, sqrt(2r)])
        ax2 = ax.twinx()
        l3, = ax2.plot(blocks, spec_c, "--^", color="tab:green", markersize=3, label="spec_conc")
        l4, = ax2.plot(blocks, drift, ":x", color="tab:red", markersize=4, label="carry_drift")
        ax2.set_ylabel("spec_conc / carry_drift")

        if idx == 0:
            ax.legend(handles=[l1, l2, l3, l4], loc="best", fontsize=8)

    # Hide any unused subplots
    for k in range(n_mat, n_rows * n_cols):
        r = k // n_cols
        c = k % n_cols
        axes[r][c].axis("off")

    fig.suptitle(f"INFRA-06: carry trajectory — half_win={win_total // 2}, full_win={win_total}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# --------------- entry point ---------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices", nargs="+", default=DEFAULT_MATRICES)
    parser.add_argument("--half-win", type=int, default=32,
                        help="Half window; full window = 2*half_win (bench convention).")
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-rows", action="store_true", default=True)
    parser.add_argument("--row-shuffle-seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--r-sig", type=int, default=2)
    parser.add_argument("--alpha-sig", type=float, default=0.003)
    parser.add_argument("--alpha-tail", type=float, default=0.0145)
    parser.add_argument("--tail-scale", type=float, default=0.99)
    parser.add_argument("--sigma1", type=float, default=0.991)
    parser.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    parser.add_argument("--out-dir", default="summary/infra_carry_trajectory")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    win_total = 2 * args.half_win

    per_matrix_rows = {}
    t_global = time.time()
    for matrix in args.matrices:
        print(f"[carry-traj] running {matrix} ...", flush=True)
        try:
            A, V_exact, _, _ = probe.generate_matrix_input(
                matrix=matrix,
                n=args.n,
                preset=args.preset,
                seed=args.seed,
                r_sig=args.r_sig,
                alpha_sig=args.alpha_sig,
                alpha_tail=args.alpha_tail,
                tail_scale=args.tail_scale,
                sigma1=args.sigma1,
                v_type=args.v_type,
                shuffle_rows=args.shuffle_rows,
                row_shuffle_seed=args.row_shuffle_seed,
            )
        except TypeError:
            # Some matrix generators do not accept the mixed-tail kwargs.
            A, V_exact, _, _ = probe.generate_matrix_input(
                matrix=matrix,
                n=args.n,
                preset=args.preset,
                seed=args.seed,
                shuffle_rows=args.shuffle_rows,
                row_shuffle_seed=args.row_shuffle_seed,
            )
        A = np.asarray(A, dtype=np.float64)
        V_exact = np.asarray(V_exact, dtype=np.float64)
        rows, elapsed = run_carry_trajectory(A, V_exact, args, args.half_win)
        per_matrix_rows[matrix] = rows

        csv_path = os.path.join(args.out_dir, f"{matrix}_win{win_total}.csv")
        write_csv(csv_path, rows)
        print(f"  wrote {csv_path}  ({len(rows)} blocks, elapsed={elapsed:.2f}s)", flush=True)

    out_png = os.path.join(args.out_dir, "trajectories.png")
    make_summary_plot(per_matrix_rows, out_png, win_total)
    print(f"[carry-traj] wrote {out_png}", flush=True)
    print(f"[carry-traj] total elapsed {time.time() - t_global:.2f}s", flush=True)


if __name__ == "__main__":
    main()
