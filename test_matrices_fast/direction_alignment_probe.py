"""Direction alignment probe.

Take any direction v (or rank-r frame V) at (matrix, block) and report:
  (1) cos² / principal angles vs. a fixed reference panel of "kinds of
      directions" (oracle, FD/iSVD carry, per-block top SVs, top rows),
  (2) row-response bias of v itself (relH1, eff_frac, top-row share)
      on each window {B, A_cur, A_fut, visible}.

This complements the per-block diagnostics (which slice by score
component) and DIAG-01 / INFRA-06 / INFRA-07 (which fix candidate panels)
by allowing arbitrary input directions to be analyzed against the same
common reference set.

Two-tier API so streaming-to-block runs once:

    pack  = build_reference_pack(matrix, block, args)   # streams once
    rows  = analyze_direction(v_or_V, pack, label="...")

The CLI (default: smoke run on mixed-tail-sharp b31) wires three input
directions through the panel:
    sketch_v1     ←  V_state[:, 0]  (FD/carry)
    S6_v1         ←  S6 slot-1 winner
    oracle_v1     ←  oracle_v1_proj (V_exact[:, 0] projected into B_union)

Outputs (per matrix, per block):
    summary/infra_direction_alignment/<matrix>_b<block>_alignment.csv
    summary/infra_direction_alignment/<matrix>_b<block>_rowbias.csv
    summary/infra_direction_alignment/<matrix>_b<block>.txt    (pretty)

Cross-references:
    diagnostic_toolkit.txt §4 (landscape / structure probes), §6
    (per-row alignment invariants), §6c (component time-series).
    state_align column should match the §6 invariant printed by
    r_sk_g_score.py — free sanity check.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

import numpy as np

import cex_restricted_space_probe as probe
from future_hmean_optimizer_diagnostic import orth_basis_against, rowspace_basis
from hmean_evidence_score import per_block_constants, stream_to_block
from oracle_entropy_audit import entropy_stats
from r_sk_g_score import _state_V, optimize_r_sk_g_in_basis
from row_cheat_baseline import oracle_frame_proj, top_r_rows_frame
from subspace_metrics import principal_angles


DEFAULT_MATRICES = ["mixed-tail-sharp"]
DEFAULT_BLOCKS = [31]
DEFAULT_INPUTS = ["sketch_v1", "sketch_v2", "combined_v1", "combined_v2",
                  "S6_v1", "oracle_v1_proj", "oracle_v2_proj",
                  "oracle_v1_exact", "oracle_v2_exact"]


# --------------------------------------------------------------------------
# Reference pack
# --------------------------------------------------------------------------


@dataclass
class ReferencePack:
    matrix: str
    block: int
    rank: int
    A: np.ndarray
    V_exact: np.ndarray
    snap: dict
    A_cur: np.ndarray
    A_fut: np.ndarray
    A_sketch: Optional[np.ndarray]
    M_gain: np.ndarray
    V_state: Optional[np.ndarray]
    B_union: np.ndarray
    references: dict   # label -> (n,) or (n, r) orthonormal frame
    windows: dict      # window label -> matrix
    sk_F2_low: float
    cur_F2: float
    fut_F2: float
    oracle_reach: list = field(default_factory=list)


def _top_r_right_sv(M: np.ndarray, r: int) -> Optional[np.ndarray]:
    if M is None:
        return None
    M = np.asarray(M, dtype=np.float64)
    if M.size == 0:
        return None
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    if Vt.shape[0] == 0:
        return None
    take = min(r, Vt.shape[0])
    return np.ascontiguousarray(Vt[:take, :].T)


def _normalize_or_none(v) -> Optional[np.ndarray]:
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float64)
    if a.size == 0:
        return None
    if a.ndim == 1:
        nrm = float(np.linalg.norm(a))
        return None if nrm <= 1e-30 else a / nrm
    # Frame: QR-orthonormalize.
    Q, R = np.linalg.qr(a)
    diagR = np.abs(np.diag(R))
    if diagR.size == 0:
        return None
    tol = max(diagR.max() * 1e-12, 1e-30)
    keep = np.where(diagR > tol)[0]
    if keep.size == 0:
        return None
    return np.ascontiguousarray(Q[:, keep])


def _build_reference_panel(snap, V_exact, V_state, A_sketch, A_cur, A_fut,
                            M_gain, B_union, rank) -> dict:
    """Return label -> (n,) or (n, r) frame for the reference panel.

    Vectors are normalized; frames are orthonormal. None entries are
    pruned by the caller.
    """
    refs = {}

    # Oracle: BOTH the un-projected (population truth) and the projected
    # (best achievable in B_union = rowspan([B; A_cur; A_fut])) versions.
    # cos²(v, oracle_v?_exact) = ||proj_of_V_exact[:,k] in B_union||² ×
    # cos²(v, oracle_v?_proj) for v in B_union — i.e. the gap between
    # the two columns at fixed v measures how much of V_exact[:, k] lives
    # OUTSIDE the visible window's row span.
    for k in range(min(rank, V_exact.shape[1])):
        v = V_exact[:, k]
        nv = float(np.linalg.norm(v))
        if nv <= 1e-30:
            continue
        v = v / nv
        # Un-projected oracle: V_exact[:, k] directly (population truth).
        refs[f"oracle_v{k+1}_exact"] = v
        # Projected oracle: V_exact[:, k] projected into the visible
        # row span and re-normalized — the slotwise reachable target.
        if B_union is not None and B_union.size:
            p = B_union @ (B_union.T @ v)
            np_norm = float(np.linalg.norm(p))
            if np_norm > 1e-30:
                refs[f"oracle_v{k+1}_proj"] = p / np_norm
            else:
                refs[f"oracle_v{k+1}_proj"] = None
        else:
            refs[f"oracle_v{k+1}_proj"] = v
    # Frames: both flavors as well.
    if V_exact.shape[1] >= 1:
        refs["oracle_frame_exact"] = V_exact[:, :min(rank, V_exact.shape[1])]
    refs["oracle_frame_proj"] = oracle_frame_proj(V_exact, B_union, rank)

    # FD / carry directions.
    if V_state is not None and V_state.size:
        for j in range(min(rank, V_state.shape[1])):
            refs[f"V_state_v{j+1}"] = V_state[:, j]
        refs["V_state_frame"] = V_state[:, :min(rank, V_state.shape[1])]

    # Per-block top right SVs of various windows. M_gain top-SVs ≡ iSVD
    # candidate (mgain_svd_v?), kept under that label for cross-tool
    # consistency with INFRA-07 / DIAG-01.
    sources = {
        "Acur":         A_cur,
        "Afut":         A_fut,
        "mgain":        M_gain,                                  # [B_top; A_cur]
        "visible":      np.vstack([A_cur, A_fut]),
        "B_top":        A_sketch if (A_sketch is not None and A_sketch.size) else None,
    }
    for src_label, M in sources.items():
        F = _top_r_right_sv(M, rank)
        if F is None:
            continue
        for j in range(F.shape[1]):
            refs[f"{src_label}_topSV_v{j+1}"] = F[:, j]
        if F.shape[1] >= 1:
            refs[f"{src_label}_topSV_frame"] = F

    # Top-r rows (orthonormalized).
    for label, M in (("Acur_rowcheat", A_cur), ("Afut_rowcheat", A_fut)):
        F = top_r_rows_frame(M, rank)
        if F is None:
            continue
        for j in range(F.shape[1]):
            refs[f"{label}_v{j+1}"] = F[:, j]
        if F.shape[1] >= 1:
            refs[f"{label}_frame"] = F

    # Drop None entries; normalize vectors.
    out = {}
    for k, V in refs.items():
        nV = _normalize_or_none(V)
        if nV is not None:
            out[k] = nV
    return out


def build_reference_pack(args, matrix: str, block: int) -> ReferencePack:
    # cex_restricted_space_probe.entropy_iter_basis_forget uses the global
    # numpy RNG (np.random.standard_normal) for restart seeds and feasibility
    # noise — without seeding, two consecutive runs of stream_to_block give
    # different V_state. Seed the global RNG with a stable, block-derived
    # value so per-block snapshots are reproducible.
    np.random.seed(int(args.seed) + 7919 * int(block))
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    try:
        A, V_exact, _, _ = probe.generate_matrix_input(
            matrix=matrix, n=args.n, preset=args.preset, seed=args.seed,
            r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
            tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
            shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
        )
    except TypeError:
        A, V_exact, _, _ = probe.generate_matrix_input(
            matrix=matrix, n=args.n, preset=args.preset, seed=args.seed,
            shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
        )
    A = np.asarray(A, np.float64)
    V_exact = np.asarray(V_exact, np.float64)

    snaps = stream_to_block(args, A, V_exact, work_dtype, int(args.rank), block, {block})
    if block not in snaps:
        raise RuntimeError(f"stream_to_block did not produce block {block} for {matrix}")

    snap = snaps[block]
    A_cur = snap["A_cur"]
    A_fut = snap["A_fut"]
    A_sketch_arr = snap["A_sketch"]
    A_sketch = A_sketch_arr if A_sketch_arr.size else None
    M_gain = snap["M_gain"]
    state = snap["state"]
    V_state = _state_V(state)

    if A_sketch is not None:
        union_stack = np.vstack([A_sketch, A_cur, A_fut])
    else:
        union_stack = np.vstack([A_cur, A_fut])
    B_union = rowspace_basis(union_stack)

    consts = per_block_constants(A, block, int(args.half_win))
    cur_F2 = float(consts["cur_F2"])
    fut_F2 = float(consts["fut_F2"])
    sk_F2_low = float(np.sum(np.asarray(A_sketch, dtype=np.float64) ** 2)) if A_sketch is not None else 0.0

    references = _build_reference_panel(
        snap, V_exact, V_state, A_sketch, A_cur, A_fut, M_gain, B_union, int(args.rank)
    )

    windows = {
        "B":       A_sketch if A_sketch is not None else np.zeros((0, A.shape[1])),
        "A_cur":   A_cur,
        "A_fut":   A_fut,
        "visible": np.vstack([A_cur, A_fut]),
    }

    pack = ReferencePack(
        matrix=matrix,
        block=block,
        rank=int(args.rank),
        A=A,
        V_exact=V_exact,
        snap=snap,
        A_cur=A_cur,
        A_fut=A_fut,
        A_sketch=A_sketch,
        M_gain=M_gain,
        V_state=V_state,
        B_union=B_union,
        references=references,
        windows=windows,
        sk_F2_low=sk_F2_low,
        cur_F2=cur_F2,
        fut_F2=fut_F2,
    )
    # Oracle reachability per slot: ||P_{B_union} V_exact[:, k]||². Equals
    # the upper bound on cos²(v, oracle_v_exact) achievable by any v in
    # B_union. Stored as an attribute so callers can print it next to the
    # alignment table (the gap "exact vs proj" lives here, not in v).
    reach = []
    for k in range(min(int(args.rank), V_exact.shape[1])):
        v = V_exact[:, k]
        nv = float(np.linalg.norm(v))
        if nv <= 1e-30:
            reach.append(float("nan"))
            continue
        v = v / nv
        if B_union is None or B_union.size == 0:
            reach.append(1.0)
            continue
        p = B_union @ (B_union.T @ v)
        reach.append(float(np.dot(p, p)))
    pack.oracle_reach = reach
    return pack


# --------------------------------------------------------------------------
# Per-direction analysis
# --------------------------------------------------------------------------


def _alignment_row(input_label: str, ref_label: str, V_in, V_ref, V_state):
    cos2, _ = principal_angles(V_in, V_ref)
    if cos2.size == 0:
        return None
    cos2_top = float(cos2[0])
    mean = float(np.mean(cos2))
    n_pa = int(cos2.size)
    # state_align cross-check: ||V_state^T v||² for r=1 v; compare to the
    # cos²(v, V_state_frame) row when present.
    if V_state is not None and V_state.size and V_in.ndim == 1:
        sa = float(np.sum((V_state.T @ V_in) ** 2))
    else:
        sa = float("nan")
    return {
        "input_label": input_label,
        "reference_label": ref_label,
        "cos2_top": cos2_top,
        "mean_cos2": mean,
        "n_principal_angles": n_pa,
        "state_align_check": sa,
    }


def analyze_direction(v_or_V, pack: ReferencePack, label: str):
    """Return (alignment_rows, rowbias_rows) for one input direction/frame."""
    V_in = _normalize_or_none(v_or_V)
    if V_in is None:
        return [], []

    V_state = pack.V_state
    align_rows = []
    for ref_label, V_ref in pack.references.items():
        row = _alignment_row(label, ref_label, V_in, V_ref, V_state)
        if row is not None:
            align_rows.append(row)

    rowbias_rows = []
    if V_in.ndim == 1:
        for win, Aw in pack.windows.items():
            stats = entropy_stats(Aw, V_in)
            top3_idx = ""
            if Aw is not None and Aw.size:
                y = (Aw @ V_in)
                e = y * y
                if e.sum() > 0:
                    idx = np.argsort(-e)[:3]
                    top3_idx = ",".join(str(int(i)) for i in idx)
            rowbias_rows.append({
                "input_label":   label,
                "window":        win,
                "rows":          stats["rows"],
                "relH1":         stats["relH1"],
                "eff_support":   stats["eff_support"],
                "eff_frac":      stats["eff_frac"],
                "top1_share":    stats["top1_share"],
                "top3_row_idx":  top3_idx,
                "energy":        stats["energy"],
            })
    return align_rows, rowbias_rows


# --------------------------------------------------------------------------
# Default input panel (for the CLI smoke run)
# --------------------------------------------------------------------------


def _build_default_input_panel(args, pack: ReferencePack) -> dict:
    out = {}
    if pack.V_state is not None and pack.V_state.size:
        out["sketch_v1"] = pack.V_state[:, 0]
        if pack.V_state.shape[1] >= 2 and pack.rank >= 2:
            out["sketch_v2"] = pack.V_state[:, 1]

    # Combined-step optimizer choices: v1 = top SV of A_cur (M3), v2 =
    # the "outside-window" pick (per project_v2_outside_window_is_carried_state).
    Vd = pack.snap.get("V_default")
    if Vd is not None:
        Vd = np.asarray(Vd, dtype=np.float64)
        if Vd.shape[1] >= 1:
            out["combined_v1"] = Vd[:, 0]
        if Vd.shape[1] >= 2 and pack.rank >= 2:
            out["combined_v2"] = Vd[:, 1]

    # Oracle inputs in BOTH flavors.
    for k in range(min(pack.rank, pack.V_exact.shape[1])):
        v = pack.V_exact[:, k]
        nv = float(np.linalg.norm(v))
        if nv <= 1e-30:
            continue
        v = v / nv
        out[f"oracle_v{k+1}_exact"] = v
        if pack.B_union is not None and pack.B_union.size:
            p = pack.B_union @ (pack.B_union.T @ v)
            np_norm = float(np.linalg.norm(p))
            if np_norm > 1e-30:
                out[f"oracle_v{k+1}_proj"] = p / np_norm

    # S6 slot-1 winner via the per-block optimizer in B_union.
    s6 = _s6_slot1(args, pack)
    if s6 is not None:
        out["S6_v1"] = s6
    return out


def _s6_slot1(args, pack: ReferencePack):
    """Run S6 slot-1 sphere ascent in B_union; return the winning unit vector or None."""
    A_sketch_for = pack.A_sketch
    consts = per_block_constants(pack.A, pack.block, int(args.half_win))
    c_sk = float(consts["c_sk"])
    cur_F2 = float(pack.cur_F2)
    fut_F2 = float(pack.fut_F2)
    starts = []
    if pack.snap.get("V_default") is not None:
        Vd = np.asarray(pack.snap["V_default"], dtype=np.float64)
        for j in range(min(pack.rank, Vd.shape[1])):
            starts.append(Vd[:, j])
    if pack.V_state is not None:
        for j in range(min(pack.rank, pack.V_state.shape[1])):
            starts.append(pack.V_state[:, j])
    if pack.B_union is not None and pack.B_union.size:
        for j in range(min(pack.rank, pack.V_exact.shape[1])):
            v = pack.V_exact[:, j]
            nv = float(np.linalg.norm(v))
            if nv > 1e-30:
                p = pack.B_union @ (pack.B_union.T @ (v / nv))
                if float(np.linalg.norm(p)) > 1e-30:
                    starts.append(p / float(np.linalg.norm(p)))
    rng = np.random.default_rng(args.seed + 880000 + 31 * pack.block)
    result = optimize_r_sk_g_in_basis(
        pack.A_cur, pack.A_fut, A_sketch_for, c_sk,
        pack.B_union, starts, rng,
        args.union_maxit, args.union_tol, args.union_random_starts,
        variant="S6", alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        V_state=pack.V_state, cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=pack.sk_F2_low,
    )
    return None if result is None else result["vec"]


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def write_csv(path, rows, fields):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(x):
        return "nan"
    return f"{x:.4f}"


def write_pretty(path, matrix, block, align_rows, rowbias_rows, ref_labels,
                  input_labels, oracle_reach=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# direction_alignment_probe — {matrix} b{block}\n\n")
        if oracle_reach:
            f.write("Oracle reachability ||P_B_union V_exact[:,k]||² per slot\n")
            f.write("(upper bound on cos²(v, oracle_v?_exact) for any v in B_union;\n")
            f.write(" = 1 means the oracle slot lives entirely in the visible row span):\n")
            for k, m in enumerate(oracle_reach):
                f.write(f"  slot {k+1}:  {_fmt(m)}\n")
            f.write("\n")
        f.write("Alignment table (cos² of top principal angle vs. each reference)\n")
        f.write("Note: oracle_v?_exact = un-projected V_exact[:, k] (population truth);\n")
        f.write("      oracle_v?_proj  = V_exact[:, k] projected into B_union (reachable).\n")
        f.write("-" * 72 + "\n")
        # Header: input | refs (cos2_top per cell)
        ref_label_order = sorted(ref_labels)
        widths = max(20, max(len(r) for r in ref_label_order) + 1) if ref_label_order else 20
        f.write(f"{'reference':<{widths}}")
        for inp in input_labels:
            f.write(f" {inp:>14}")
        f.write("\n")
        # Build a {(input, ref): cos2_top} lookup.
        cell = {}
        for r in align_rows:
            cell[(r["input_label"], r["reference_label"])] = r["cos2_top"]
        for ref in ref_label_order:
            f.write(f"{ref:<{widths}}")
            for inp in input_labels:
                f.write(f" {_fmt(cell.get((inp, ref), float('nan'))):>14}")
            f.write("\n")
        f.write("\n")
        f.write("state_align (cross-check: cos²(v, V_state_v1) row should equal\n")
        f.write("the §6 invariant printed by r_sk_g_score.py)\n\n")

        if rowbias_rows:
            f.write("Row-bias of input directions (per window)\n")
            f.write("-" * 72 + "\n")
            f.write(f"{'input':<14} {'window':<8} {'rows':>5} {'relH1':>7} "
                    f"{'eff_frac':>9} {'top1':>6} {'top3_idx':<24}\n")
            for r in rowbias_rows:
                f.write(
                    f"{r['input_label']:<14} {r['window']:<8} {int(r['rows']):>5} "
                    f"{_fmt(r['relH1']):>7} {_fmt(r['eff_frac']):>9} "
                    f"{_fmt(r['top1_share']):>6} {r['top3_row_idx']:<24}\n"
                )
            f.write("\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrices", nargs="+", default=DEFAULT_MATRICES)
    p.add_argument("--blocks", nargs="+", type=int, default=DEFAULT_BLOCKS)
    p.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS,
                   help="Subset of the default input panel to evaluate. "
                        "Default panel: sketch_v1[/v2], oracle_v1[/v2], S6_v1.")
    p.add_argument("--out-dir", default="summary/infra_direction_alignment")

    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--half-win", type=int, default=32)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--preset", default="fast")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-rows", action="store_true", default=True)
    p.add_argument("--row-shuffle-seed", type=int, default=0)
    p.add_argument("--old-memory-size", type=int, default=32)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--q0", type=int, default=4)
    p.add_argument("--qmax", type=int, default=16)
    p.add_argument("--krylov-depth", type=int, default=2)
    p.add_argument("--residual-tol", type=float, default=0.01)
    p.add_argument("--expansion-maxit", type=int, default=4)
    p.add_argument("--num-restarts", type=int, default=1)
    p.add_argument("--maxit", type=int, default=40)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--post-expansion-maxit", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--patience-rel-tol", type=float, default=1e-5)
    p.add_argument("--union-maxit", type=int, default=80)
    p.add_argument("--union-tol", type=float, default=1e-9)
    p.add_argument("--union-random-starts", type=int, default=8)
    p.add_argument("--r-sig", type=int, default=2)
    p.add_argument("--alpha-sig", type=float, default=0.003)
    p.add_argument("--alpha-tail", type=float, default=0.0145)
    p.add_argument("--tail-scale", type=float, default=0.99)
    p.add_argument("--sigma1", type=float, default=0.991)
    p.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    return p.parse_args()


ALIGN_FIELDS = [
    "input_label", "reference_label", "cos2_top",
    "mean_cos2", "n_principal_angles", "state_align_check",
]
ROWBIAS_FIELDS = [
    "input_label", "window", "rows",
    "relH1", "eff_support", "eff_frac", "top1_share",
    "top3_row_idx", "energy",
]


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    requested_inputs = list(args.inputs)

    for matrix in args.matrices:
        for block in args.blocks:
            print(f"[direction-align] {matrix} b{block}", flush=True)
            pack = build_reference_pack(args, matrix, block)
            input_panel = _build_default_input_panel(args, pack)

            # Filter to requested inputs (preserving order).
            chosen = [(label, input_panel[label]) for label in requested_inputs
                      if label in input_panel and input_panel[label] is not None]
            if not chosen:
                print(f"  (no input directions available for {matrix} b{block})")
                continue

            align_rows = []
            rowbias_rows = []
            for label, v in chosen:
                a, b = analyze_direction(v, pack, label)
                align_rows.extend(a)
                rowbias_rows.extend(b)

            stem = os.path.join(args.out_dir, f"{matrix}_b{block}")
            write_csv(stem + "_alignment.csv", align_rows, ALIGN_FIELDS)
            write_csv(stem + "_rowbias.csv", rowbias_rows, ROWBIAS_FIELDS)
            write_pretty(
                stem + ".txt", matrix, block, align_rows, rowbias_rows,
                ref_labels=set(pack.references.keys()),
                input_labels=[label for label, _ in chosen],
                oracle_reach=pack.oracle_reach,
            )
            print(f"  wrote {stem}_alignment.csv  +  {stem}_rowbias.csv  +  {stem}.txt",
                  flush=True)


if __name__ == "__main__":
    main()
