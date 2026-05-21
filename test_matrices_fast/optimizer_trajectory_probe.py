"""Optimizer trajectory probe (INFRA-04 driver).

Run the tracked-ascent wrapper (`optimizer_trajectory.tracked_ascend`) on the
canonical 3-matrix probe set (static-cex, mixed-tail-sharp, diffuse-diffuse)
× blocks {6, 12, 31} × variants {S6, S6_GM} × ranks {1, 2}, emit per-cell CSV
trajectories, and produce a summary plot.

Per cell at rank r > 1: greedy deflation — track slot-1's trajectory, then fix
v1, deflate B_union, track slot-2's trajectory. Each slot's trajectory is its
own CSV row stream, tagged by `slot` column.

Per-iter CSV columns (one row per emitted iter — including iter=-1 init):
  tag           {matrix}|b{block}|{variant}|r{rank}|slot{k}
  matrix
  block
  variant
  rank
  slot          1-indexed slot index (1 for first ascent, 2 for deflated)
  start_id      restart index (we run a single warm-restart per slot here;
                see notes — multi-restart could be wired by repeating the
                tracked_ascend call per start)
  iter          -1 for init, 0..maxit-1 for accepted steps / ls_fail
  phase         "init" | "step" | "ls_fail"
  accepted      0 / 1
  alpha         line-search step size (NaN at init / ls_fail)
  score         score at v
  step_norm     ||step_z||₂   (NaN at init)
  grad_tan_norm ||grad_tan_z||₂
  grad_full_norm ||∇score||₂  (ambient gradient norm, before basis projection)
  cos2_max      max principal cos² of v vs V_oracle (= cos2 of closest axis)
                — at r=1 reduces to scalar cos²(v, v_oracle_proj)
  cos2_mean     mean of principal cos² across the r principal angles
  cos2_each_j   per-axis principal cos² (j = 0..r-1)
  angle_each_j_rad
                per-axis principal angle in radians (j = 0..r-1)
  v_proj_on_oracle  ⟨v, V_oracle V_oracleᵀ v⟩ — fraction of v's energy in the
                oracle subspace (rank-r). For slot-2 the relevant
                "drift" projection is on the projected-oracle complement
                of v1; see synthesis.

Outputs (under summary/infra_optimizer_trajectory/):
  {matrix}_b{block}_{variant}_r{rank}.csv    one trajectory per cell
                                             (slot1 + slot2 rows concatenated
                                             when r=2)
  trajectories.png                           cos²_max vs iter, faceted by
                                             matrix, color = (variant, block)
  synthesis.md                               qualitative observations

Backlog: INFRA-04 in summary/overview/score_family_workflow.txt §5.
Resolves: toolkit §8 (b) in summary/overview/diagnostic_toolkit.txt.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

import cex_restricted_space_probe as probe
from future_hmean_optimizer_diagnostic import (
    orth_basis_against,
    rowspace_basis,
)
from hmean_evidence_score import per_block_constants, stream_to_block
from optimizer_trajectory import tracked_ascend
from r_sk_g_score import make_r_sk_g_optimizer
from subspace_metrics import principal_angles


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _project_unit(vec, B):
    if vec is None or B is None or B.size == 0:
        return None
    p = B @ (B.T @ vec)
    nv = float(np.linalg.norm(p))
    return None if nv <= 1e-30 else p / nv


def _oracle_frame_in_union(V_exact, B_union, rank):
    """Project the top-rank columns of V_exact into B_union and orthonormalize."""
    Vex = np.asarray(V_exact, dtype=np.float64)[:, :rank]
    if B_union is None or B_union.size == 0:
        return None
    V_or_proj = B_union @ (B_union.T @ Vex)
    Q_or, _ = np.linalg.qr(V_or_proj)
    return np.ascontiguousarray(Q_or[:, :rank], dtype=np.float64)


def _starts_for_slot(snap, V_state, V_exact, B_search, B_union, slot, prev_v, oracle_warm):
    """Build a list of starts for the given slot. Each is in B_search coords
    after projection, but we return ambient-space vectors and let the caller
    project. Mirrors r_sk_g_score.py:analyze_block construction."""
    V_default = snap["V_default"]
    starts = []
    if slot == 1:
        # slot-1 warm-starts: combined v1, sketch_v1, mgain_svd v1 (i.e. V_default columns)
        starts.append(V_default[:, 0])
        if V_default.shape[1] >= 2:
            starts.append(V_default[:, 1])
        if V_state is not None:
            for j in range(min(V_state.shape[1], 4)):
                starts.append(V_state[:, j])
    else:
        # slot-2: same warm-starts, but project away from prev_v
        if V_default.shape[1] >= 2:
            starts.append(V_default[:, 1])
        starts.append(V_default[:, 0])
        if V_state is not None:
            for j in range(min(V_state.shape[1], 4)):
                starts.append(V_state[:, j])
    if oracle_warm:
        for j in range(min(2, V_exact.shape[1])):
            v_proj = _project_unit(V_exact[:, j], B_union)
            if v_proj is not None:
                starts.append(v_proj)
    return [s for s in starts if s is not None]


def _z_starts_for(starts_ambient, B_search, rng, n_random):
    """Ambient → basis-coord starts, plus n_random uniform starts on the unit
    sphere of the basis. Mirrors `optimize_future_hmean_in_basis`."""
    q = B_search.shape[1]
    z_starts = []
    for v0 in starts_ambient:
        z = B_search.T @ np.asarray(v0, dtype=np.float64).reshape(-1)
        nz = float(np.linalg.norm(z))
        if nz > 1e-12:
            z_starts.append(z / nz)
    for _ in range(max(0, int(n_random))):
        z = rng.standard_normal(q)
        nz = float(np.linalg.norm(z))
        if nz > 1e-12:
            z_starts.append(z / nz)
    return z_starts


# --------------------------------------------------------------------------
# Trajectory record assembly
# --------------------------------------------------------------------------


def _trajectory_record(rec, V_oracle, slot):
    """Convert a tracker dict into the CSV-row form (with cos² / angles)."""
    v = np.asarray(rec["v"], dtype=np.float64).reshape(-1)
    nv = float(np.linalg.norm(v))
    v_unit = v / nv if nv > 1e-30 else v
    if V_oracle is not None and V_oracle.size:
        cos2, angles = principal_angles(v_unit, V_oracle)
        v_proj_on_oracle = float(np.sum((V_oracle.T @ v_unit) ** 2))
    else:
        cos2 = np.array([])
        angles = np.array([])
        v_proj_on_oracle = float("nan")
    out = {
        "iter": int(rec["iter"]),
        "phase": rec["phase"],
        "accepted": 1 if rec["accepted"] else 0,
        "alpha": float(rec["alpha"]),
        "score": float(rec["score"]),
        "step_norm": float(np.linalg.norm(rec["step_z"])),
        "grad_tan_norm": float(np.linalg.norm(rec["grad_tan_z"])),
        "grad_full_norm": float(np.linalg.norm(rec["grad_full"])),
        "cos2_max": float(cos2[0]) if cos2.size else float("nan"),
        "cos2_mean": float(np.mean(cos2)) if cos2.size else float("nan"),
        "v_proj_on_oracle": v_proj_on_oracle,
        "_cos2": cos2,
        "_angles": angles,
        "slot": int(slot),
    }
    return out


# --------------------------------------------------------------------------
# Per-cell probe
# --------------------------------------------------------------------------


def probe_cell(args, matrix, A, V_exact, snap, block_id, variant, rank):
    """Track one (matrix, block, variant, rank) cell.

    Returns list of CSV-row dicts (concatenation of slot-1 and slot-2 if r=2).
    """
    half_win = int(args.half_win)
    A_cur = np.asarray(snap["A_cur"], dtype=np.float64)
    A_fut = np.asarray(snap["A_fut"], dtype=np.float64)
    A_sketch_arr = np.asarray(snap["A_sketch"], dtype=np.float64)
    A_sketch = A_sketch_arr if A_sketch_arr.size else None
    state = snap["state"]

    consts = per_block_constants(A, block_id, half_win)
    c_sk = consts["c_sk"]
    cur_F2 = float(consts["cur_F2"])
    fut_F2 = float(consts["fut_F2"])
    if A_sketch is not None:
        sk_F2_low = float(np.sum(A_sketch ** 2))
    else:
        sk_F2_low = 0.0
    cur_op2 = float(np.linalg.svd(A_cur, compute_uv=False)[0] ** 2) if A_cur.size else 0.0
    fut_op2 = float(np.linalg.svd(A_fut, compute_uv=False)[0] ** 2) if A_fut.size else 0.0
    if A_sketch is not None:
        if state is not None and state.get("s") is not None and np.asarray(state["s"]).size:
            sk_op2_low = float(np.asarray(state["s"], dtype=np.float64)[0] ** 2)
        else:
            sk_op2_low = float(np.linalg.svd(A_sketch, compute_uv=False)[0] ** 2)
    else:
        sk_op2_low = 0.0

    V_state = None
    if state is not None and state.get("V") is not None:
        Vs = np.asarray(state["V"], dtype=np.float64)
        if Vs.size:
            V_state = Vs

    if A_sketch is not None:
        union_stack = np.vstack([A_sketch, A_cur, A_fut])
    else:
        union_stack = np.vstack([A_cur, A_fut])
    B_union = rowspace_basis(union_stack)

    alpha, beta, gamma = float(args.alpha), float(args.beta), float(args.gamma)

    # Build the value-grad fn the production optimizer would use.
    vg = make_r_sk_g_optimizer(
        A_cur, A_fut, A_sketch, c_sk,
        variant=variant, alpha=alpha, beta=beta, gamma=gamma, V_state=V_state,
        cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
        cur_op2=cur_op2, fut_op2=fut_op2, sk_op2_low=sk_op2_low,
    )

    rows_out = []

    # Track slot 1.
    B_search = B_union
    starts_ambient_1 = _starts_for_slot(
        snap, V_state, V_exact, B_search, B_union, slot=1, prev_v=None,
        oracle_warm=bool(args.from_oracle_warm),
    )
    rng_starts = np.random.default_rng(args.seed + 91000 + block_id)
    z_starts_1 = _z_starts_for(starts_ambient_1, B_search, rng_starts, args.union_random_starts)
    if not z_starts_1:
        return rows_out

    V_oracle_1 = _oracle_frame_in_union(V_exact, B_union, 1)

    # We track the BEST trajectory across restarts: pick the one whose final
    # score is highest. (Tracker still emits each restart's full trajectory; we
    # filter and keep only the best at write time.) For the synthesis, the
    # "best run" is what the production optimizer would commit.
    best_v1 = None
    best_v1_score = -np.inf
    best_v1_rows = None

    for start_idx, z0 in enumerate(z_starts_1):
        traj_records = []
        tracker = lambda r, idx=start_idx, _list=traj_records: _list.append(_trajectory_record(r, V_oracle_1, slot=1))
        res = tracked_ascend(
            vg, B_search, z0,
            maxit=int(args.max_iter),
            tol=float(args.union_tol),
            tracker=tracker,
        )
        if res["score"] > best_v1_score:
            best_v1_score = res["score"]
            best_v1 = res["vec"]
            best_v1_rows = (start_idx, traj_records)

    if best_v1 is None or best_v1_rows is None:
        return rows_out

    start_id_1, traj_1 = best_v1_rows
    for r in traj_1:
        r["start_id"] = start_id_1
    rows_out.extend(traj_1)

    # Track slot 2 if rank >= 2.
    if int(rank) >= 2:
        B_search_2 = orth_basis_against(B_union, best_v1)
        if B_search_2.shape[1] < 1:
            return rows_out
        # Oracle frame for slot-2: top-2 oracle frame projected and orthonormalized,
        # then the "slot 2" comparison uses the FULL rank-2 oracle frame (since
        # the per-iter v is rank-1, principal_angles on a vector vs a 2-frame
        # gives the closest principal angle out of the two — that's the right
        # invariant for slot-2 drift).
        V_oracle_2_full = _oracle_frame_in_union(V_exact, B_union, 2)

        starts_ambient_2 = _starts_for_slot(
            snap, V_state, V_exact, B_search_2, B_union, slot=2, prev_v=best_v1,
            oracle_warm=bool(args.from_oracle_warm),
        )
        rng_starts_2 = np.random.default_rng(args.seed + 92000 + block_id)
        z_starts_2 = _z_starts_for(
            starts_ambient_2, B_search_2, rng_starts_2, args.union_random_starts
        )
        if not z_starts_2:
            return rows_out

        best_v2_score = -np.inf
        best_v2_rows = None
        for start_idx, z0 in enumerate(z_starts_2):
            traj_records = []
            # For slot-2 we use the FULL rank-2 oracle frame as the reference;
            # cos2_max measures alignment to the oracle 2-plane and cos2[0]
            # specifically picks the closer axis. This isolates "drift off the
            # 2-plane" from "rotation within the 2-plane".
            tracker = lambda r, idx=start_idx, _list=traj_records: _list.append(
                _trajectory_record(r, V_oracle_2_full, slot=2)
            )
            res = tracked_ascend(
                vg, B_search_2, z0,
                maxit=int(args.max_iter),
                tol=float(args.union_tol),
                tracker=tracker,
            )
            if res["score"] > best_v2_score:
                best_v2_score = res["score"]
                best_v2_rows = (start_idx, traj_records)

        if best_v2_rows is not None:
            start_id_2, traj_2 = best_v2_rows
            for r in traj_2:
                r["start_id"] = start_id_2
            rows_out.extend(traj_2)

    return rows_out


# --------------------------------------------------------------------------
# Per-matrix runner
# --------------------------------------------------------------------------


def run_matrix(args, matrix, blocks_to_report, variants, ranks):
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=matrix, n=args.n, preset=args.preset, seed=args.seed,
        r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
        shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, dtype=np.float64)
    V_exact = np.asarray(V_exact, dtype=np.float64)
    blocks = sorted(set(int(b) for b in blocks_to_report))
    target = max(blocks)
    # snapshots requires rank>=max(ranks); use args.rank for the streaming phase
    # but we'll compute rank-r tracking from V_exact within each cell.
    snapshots = stream_to_block(
        args, A, V_exact, work_dtype, max(int(args.rank), max(ranks)),
        target, set(blocks),
    )

    out_per_cell = {}
    for b in blocks:
        if b not in snapshots:
            print(f"  [{matrix}] block {b}: no snapshot; skipped")
            continue
        for variant in variants:
            for r in ranks:
                t0 = time.time()
                rows = probe_cell(
                    args, matrix, A, V_exact, snapshots[b], b, variant, r,
                )
                elapsed = time.time() - t0
                key = (b, variant, int(r))
                out_per_cell[key] = rows
                # Light progress line.
                if rows:
                    final = rows[-1]
                    iters_used = max((row["iter"] for row in rows if row["slot"] == max(rs["slot"] for rs in rows)), default=-1)
                    print(
                        f"  [{matrix}] b{b:>2} {variant} r={r}: "
                        f"slots={sorted(set(rw['slot'] for rw in rows))} "
                        f"slot1_final_cos2={next((rw['cos2_max'] for rw in reversed(rows) if rw['slot']==1), float('nan')):.3f} "
                        + (f"slot2_final_cos2={next((rw['cos2_max'] for rw in reversed(rows) if rw['slot']==2), float('nan')):.3f} " if r >= 2 else "")
                        + f"iters_used={iters_used}; {elapsed:.2f}s"
                    )
                else:
                    print(f"  [{matrix}] b{b:>2} {variant} r={r}: empty trajectory; {elapsed:.2f}s")
    return out_per_cell


# --------------------------------------------------------------------------
# CSV writing
# --------------------------------------------------------------------------


def _csv_fieldnames(rank):
    base = [
        "tag", "matrix", "block", "variant", "rank", "slot", "start_id",
        "iter", "phase", "accepted", "alpha",
        "score", "step_norm", "grad_tan_norm", "grad_full_norm",
        "cos2_max", "cos2_mean", "v_proj_on_oracle",
    ]
    base += [f"cos2_each_{j}" for j in range(rank)]
    base += [f"angle_each_{j}_rad" for j in range(rank)]
    return base


def write_cell_csv(path, matrix, block, variant, rank, rows):
    fieldnames = _csv_fieldnames(rank)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row_out = {
                "tag": f"{matrix}|b{int(block):02d}|{variant}|r{int(rank)}|slot{int(r['slot'])}",
                "matrix": matrix,
                "block": int(block),
                "variant": variant,
                "rank": int(rank),
                "slot": int(r["slot"]),
                "start_id": int(r.get("start_id", 0)),
                "iter": int(r["iter"]),
                "phase": r["phase"],
                "accepted": int(r["accepted"]),
                "alpha": float(r["alpha"]),
                "score": float(r["score"]),
                "step_norm": float(r["step_norm"]),
                "grad_tan_norm": float(r["grad_tan_norm"]),
                "grad_full_norm": float(r["grad_full_norm"]),
                "cos2_max": float(r["cos2_max"]),
                "cos2_mean": float(r["cos2_mean"]),
                "v_proj_on_oracle": float(r["v_proj_on_oracle"]),
            }
            cos2 = r.get("_cos2", np.array([]))
            angles = r.get("_angles", np.array([]))
            for j in range(rank):
                row_out[f"cos2_each_{j}"] = float(cos2[j]) if j < cos2.size else float("nan")
                row_out[f"angle_each_{j}_rad"] = float(angles[j]) if j < angles.size else float("nan")
            w.writerow(row_out)


# --------------------------------------------------------------------------
# Trajectory plot (cos²_max vs iter, faceted by matrix)
# --------------------------------------------------------------------------


def plot_trajectories(out_dir, all_results, matrices, variants, blocks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_matrix = len(matrices)
    fig, axes = plt.subplots(
        1, max(n_matrix, 1), figsize=(5.2 * max(n_matrix, 1), 4.0), sharey=True,
    )
    if n_matrix == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    # Build a deterministic color/style key for (variant, block).
    style_keys = [(v, b) for v in variants for b in blocks]
    color_for = {k: cmap(i % 10) for i, k in enumerate(style_keys)}

    for ax, matrix in zip(axes, matrices):
        per_cell = all_results.get(matrix, {})
        plotted_any = False
        for (b, variant, rank), rows in sorted(per_cell.items()):
            if not rows:
                continue
            # We plot slot-1 trajectory for r=1 and slot-2 (the harder one) for r=2.
            slot_to_plot = 2 if int(rank) >= 2 else 1
            slot_rows = [r for r in rows if r["slot"] == slot_to_plot]
            if not slot_rows:
                continue
            iters = [r["iter"] for r in slot_rows]
            cos2 = [r["cos2_max"] for r in slot_rows]
            label = f"{variant} b{b} r{rank} s{slot_to_plot}"
            color = color_for.get((variant, b), "gray")
            ls = "-" if int(rank) == 1 else "--"
            ax.plot(iters, cos2, marker="o", markersize=3, linestyle=ls,
                    color=color, label=label, alpha=0.85, linewidth=1.3)
            plotted_any = True
        ax.set_title(matrix)
        ax.set_xlabel("iter")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        if plotted_any:
            ax.legend(fontsize=7, loc="upper right", ncol=2)
    axes[0].set_ylabel(r"max principal cos$^2$(v, V_oracle)")
    fig.suptitle("Optimizer trajectory: cos²_max vs iter (slot to plot: 2 if r>=2 else 1)")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "trajectories.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# Argument parsing / main
# --------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="INFRA-04 optimizer-trajectory probe")
    p.add_argument(
        "--matrices", nargs="*", default=None,
        help="Default: static-cex mixed-tail-sharp diffuse-diffuse",
    )
    p.add_argument(
        "--blocks", nargs="+", type=int, default=[6, 12, 31],
        help="Blocks to probe (default 6 12 31).",
    )
    p.add_argument(
        "--variants", nargs="+", default=["S6", "S6_GM"],
        help="Score variants to track (default S6 S6_GM).",
    )
    p.add_argument(
        "--ranks", nargs="+", type=int, default=[1, 2],
        help="Ranks to probe (default 1 2).",
    )
    p.add_argument("--max-iter", type=int, default=120)
    p.add_argument(
        "--from-oracle-warm", action="store_true", default=False,
        help="Add the oracle projections to the warm-start pool (NOT default; "
             "see toolkit §9 — `--no-oracle-warmstart` is the default policy).",
    )
    p.add_argument(
        "--no-oracle-warmstart", dest="from_oracle_warm", action="store_false",
        help=argparse.SUPPRESS,  # alias for clarity / explicit reset.
    )
    p.add_argument(
        "--out-dir", default="summary/infra_optimizer_trajectory",
    )

    # Optimizer / streaming args (mirrored from r_sk_g_score.py)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--half-win", type=int, default=32)
    p.add_argument("--rank", type=int, default=2,
                   help="Streaming rank (carry rank); independent of --ranks.")
    p.add_argument("--preset", default="fast")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-rows", action="store_true", default=True)
    p.add_argument("--row-shuffle-seed", type=int, default=0)
    p.add_argument("--old-memory-size", type=int, default=32)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--q0", type=int, default=8)
    p.add_argument("--qmax", type=int, default=48)
    p.add_argument("--krylov-depth", type=int, default=2)
    p.add_argument("--residual-tol", type=float, default=0.01)
    p.add_argument("--expansion-maxit", type=int, default=8)
    p.add_argument("--num-restarts", type=int, default=3)
    p.add_argument("--maxit", type=int, default=120)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--post-expansion-maxit", type=int, default=80)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--patience-rel-tol", type=float, default=1e-5)
    p.add_argument("--union-maxit", type=int, default=120)
    p.add_argument("--union-tol", type=float, default=1e-9)
    p.add_argument("--union-random-starts", type=int, default=24)
    p.add_argument("--r-sig", type=int, default=2)
    p.add_argument("--alpha-sig", type=float, default=0.003)
    p.add_argument("--alpha-tail", type=float, default=0.0145)
    p.add_argument("--tail-scale", type=float, default=0.99)
    p.add_argument("--sigma1", type=float, default=0.991)
    p.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    return p.parse_args()


def main():
    args = parse_args()
    matrices = args.matrices if args.matrices else [
        "static-cex", "mixed-tail-sharp", "diffuse-diffuse",
    ]
    variants = list(args.variants)
    ranks = sorted(set(int(r) for r in args.ranks))
    blocks = sorted(set(int(b) for b in args.blocks))

    os.makedirs(args.out_dir, exist_ok=True)
    print(
        f"optimizer_trajectory_probe: variants={variants} ranks={ranks} "
        f"blocks={blocks} matrices={matrices} max_iter={args.max_iter} "
        f"from_oracle_warm={args.from_oracle_warm}"
    )

    all_results = {}
    summary_meta = {
        "args": {
            "variants": variants,
            "ranks": ranks,
            "blocks": blocks,
            "matrices": matrices,
            "max_iter": int(args.max_iter),
            "from_oracle_warm": bool(args.from_oracle_warm),
            "seed": int(args.seed),
            "half_win": int(args.half_win),
            "n": int(args.n),
        },
        "per_cell": {},
    }
    for matrix in matrices:
        print(f"== {matrix} ==")
        per_cell = run_matrix(args, matrix, blocks, variants, ranks)
        all_results[matrix] = per_cell

        for (b, variant, rank), rows in per_cell.items():
            out_path = os.path.join(
                args.out_dir,
                f"{matrix}_b{b:02d}_{variant}_r{rank}.csv",
            )
            write_cell_csv(out_path, matrix, b, variant, rank, rows)
            # Per-cell summary metadata for the JSON.
            slot_summary = {}
            for slot in sorted(set(r["slot"] for r in rows)) if rows else []:
                slot_rows = [r for r in rows if r["slot"] == slot]
                init_row = next((r for r in slot_rows if r["iter"] == -1), None)
                final_row = slot_rows[-1] if slot_rows else None
                slot_summary[f"slot{slot}"] = {
                    "init_score": float(init_row["score"]) if init_row is not None else None,
                    "init_cos2_max": float(init_row["cos2_max"]) if init_row is not None else None,
                    "final_score": float(final_row["score"]) if final_row is not None else None,
                    "final_cos2_max": float(final_row["cos2_max"]) if final_row is not None else None,
                    "n_iters_recorded": len(slot_rows),
                    "stop_phase": final_row["phase"] if final_row is not None else None,
                }
            summary_meta["per_cell"][f"{matrix}|b{b:02d}|{variant}|r{rank}"] = {
                "csv_path": out_path,
                "slots": slot_summary,
            }

    # Cross-matrix summary plot.
    plot_path = plot_trajectories(args.out_dir, all_results, matrices, variants, blocks)
    print(f"wrote {plot_path}")

    json_path = os.path.join(args.out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_meta, f, indent=2, sort_keys=True, default=float)
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
