"""DIAG-02 forward-replication tail-conspiracy probe.

Diagnostic-only. For high-scoring streaming candidates V_default[:, slot] at
selected blocks, log:

    - current combined score
    - per-window relH1 and effective support on A_cur (visible) and
      A_fut (next unseen window) via row-energy distribution
    - top-row energy share on each window
    - overlap of dominant rows between A_cur and A_fut (Jaccard on top-k indices,
      and weighted L1 row-mass overlap)
    - response on the next unseen window (u_fut) and Pearson(|A_cur v|,|A_fut v|)
    - subsequent carry-alignment decay: state_align(v) at the producing block
      and at the next block's state V

Produces per-(matrix, block, slot, candidate) rows. Candidates per slot:
    - "v_default"    : streaming combined-score winner (score_favoured)
    - "v_oracle_proj": V_exact[:, slot] projected into A_cur/A_fut union, deflated
                       against earlier slot oracles (control)

The hypothesis under test: HM3-missed failures are due to within-window row
concentration that appears stable only because rare rows align in the visible
sample but break on A_fut. Signature would be (relative to oracle):
    (a) high top_share_cur AND
    (b) low top_share_fut OR low Jaccard_topk OR low replicability_abs_pearson AND
    (c) substantial cos2 align gap (winner has much larger align_exact than oracle).

Run:
  python summary/diag02_tail_conspiracy_forward/probe.py
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

# Run as a script from anywhere by anchoring imports to the project root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cex_restricted_space_probe as probe  # noqa: E402
from future_hmean_optimizer_diagnostic import rowspace_basis  # noqa: E402
from hmean_evidence_score import per_block_constants, stream_to_block  # noqa: E402


# --------------------------- small numerics ----------------------------------

def unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= 1e-30:
        return None
    return np.ascontiguousarray(v / n)


def project_unit(v, B):
    if v is None or B is None or B.size == 0:
        return None
    return unit(B @ (B.T @ v))


def deflate_unit(v, q):
    v = unit(v)
    q = unit(q)
    if v is None or q is None:
        return None
    return unit(v - q * float(q @ v))


def row_energy_dist(A, v):
    """Return (p, energy) where p_i = (A v)_i^2 / ||A v||^2."""
    A = np.asarray(A, dtype=np.float64)
    if A.size == 0:
        return None, 0.0
    y = A @ v
    e = y * y
    S = float(np.sum(e))
    if S <= 1e-30:
        return np.zeros_like(e), 0.0
    return e / S, S


def relH1_and_support(p):
    """Normalized entropy in [0,1] and effective support exp(H)."""
    if p is None or p.size == 0:
        return None, None
    H = -float(np.sum(p * np.log(np.maximum(p, 1e-300))))
    return float(H / np.log(max(p.size, 2))), float(np.exp(H))


def topk_indices(p, k):
    if p is None:
        return np.array([], dtype=int)
    k = max(1, min(k, p.size))
    return np.argpartition(-p, k - 1)[:k]


def jaccard(a, b):
    if a.size == 0 and b.size == 0:
        return 1.0
    sa, sb = set(a.tolist()), set(b.tolist())
    u = len(sa | sb)
    if u == 0:
        return 1.0
    return float(len(sa & sb) / u)


def pearson_abs_response(A_cur, A_fut, v):
    if A_cur.shape != A_fut.shape:
        # Different row counts: cannot pair, skip.
        return None
    x = np.abs(np.asarray(A_cur, dtype=np.float64) @ v)
    y = np.abs(np.asarray(A_fut, dtype=np.float64) @ v)
    if x.size < 2:
        return None
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx <= 1e-30 or sy <= 1e-30:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def state_align(V_state, v):
    if V_state is None or V_state.size == 0 or v is None:
        return None
    p = V_state.T @ v
    return float(np.dot(p, p))


# ---------------------- per-row record construction --------------------------

def slot_record(matrix, block, slot, label, role, v, snap, snap_next, A,
                V_exact, args):
    if v is None:
        return None
    A_cur = snap["A_cur"]
    A_fut = snap["A_fut"]
    A_sketch = snap["A_sketch"] if snap["A_sketch"].size else None

    # Combined-score score_total
    comp = probe.combined_score_component_details(
        snap["M_gain"], A_cur, v, A.shape[0],
        state_prev=snap["state"], old_row_memory=snap["old_row_memory"],
    )

    # Per-window distributions
    p_cur, S_cur = row_energy_dist(A_cur, v)
    p_fut, S_fut = row_energy_dist(A_fut, v)
    p_sk, S_sk = (row_energy_dist(A_sketch, v) if A_sketch is not None
                  else (None, 0.0))
    relH1_cur, supp_cur = relH1_and_support(p_cur)
    relH1_fut, supp_fut = relH1_and_support(p_fut)
    relH1_sk, supp_sk = (relH1_and_support(p_sk) if p_sk is not None
                         else (None, None))

    top_share_cur = float(np.max(p_cur)) if p_cur is not None else None
    top_share_fut = float(np.max(p_fut)) if p_fut is not None else None
    top_share_sk = float(np.max(p_sk)) if p_sk is not None else None

    # Dominant-row overlap between A_cur and A_fut
    k = int(args.topk)
    idx_cur = topk_indices(p_cur, k)
    idx_fut = topk_indices(p_fut, k)
    jac_topk = jaccard(idx_cur, idx_fut)
    # Weighted overlap: sum_i min(p_cur_i, p_fut_i) — only meaningful when
    # the row-index spaces are comparable (same row count). They are, both
    # are size half_win.
    if (p_cur is not None and p_fut is not None
            and p_cur.size == p_fut.size):
        weighted_overlap = float(np.sum(np.minimum(p_cur, p_fut)))
    else:
        weighted_overlap = None

    # Normalised responses (per_block_constants for u-fractions)
    consts = per_block_constants(A, block, A_cur.shape[0])
    cur_F2 = consts["cur_F2"]
    fut_F2 = consts["fut_F2"]
    u_cur = float(S_cur / max(cur_F2, 1e-30))
    u_fut = float(S_fut / max(fut_F2, 1e-30))
    asym = abs(u_cur - u_fut) / max(u_cur, u_fut, 1e-30)

    # Replicability across the two windows
    rep = pearson_abs_response(A_cur, A_fut, v)

    # Carry-alignment now and at next block (decay)
    V_state_now = None
    if snap["state"] is not None and snap["state"].get("V") is not None:
        V_state_now = np.asarray(snap["state"]["V"], dtype=np.float64)
    sa_now = state_align(V_state_now, v)

    sa_next = None
    if snap_next is not None and snap_next.get("state") is not None:
        Vn = snap_next["state"].get("V")
        if Vn is not None:
            sa_next = state_align(np.asarray(Vn, dtype=np.float64), v)

    # Oracle alignments
    align_o_slot = float((v @ unit(V_exact[:, slot])) ** 2)

    return {
        "matrix": matrix,
        "block": int(block),
        "slot": int(slot),
        "candidate": label,
        "role": role,
        "score_combined": float(comp["score_total"]),
        "phi": float(comp.get("phi", 0.0)),
        "energy_A_cur": float(comp.get("new_y2_sq", 0.0)),
        "u_cur": u_cur,
        "u_fut": u_fut,
        "between_asym": float(asym),
        "relH1_cur": relH1_cur,
        "eff_support_cur": supp_cur,
        "relH1_fut": relH1_fut,
        "eff_support_fut": supp_fut,
        "relH1_sketch": relH1_sk,
        "eff_support_sketch": supp_sk,
        "top_share_cur": top_share_cur,
        "top_share_fut": top_share_fut,
        "top_share_sketch": top_share_sk,
        "topk_jaccard_cur_fut": float(jac_topk),
        "weighted_row_overlap_cur_fut": weighted_overlap,
        "replicability_pearson": rep,
        "state_align_now": sa_now,
        "state_align_next": sa_next,
        "state_align_decay": (None if (sa_now is None or sa_next is None)
                              else float(sa_now - sa_next)),
        "align_exact_oslot": align_o_slot,
    }


# ---------------------- main probe loop --------------------------------------

def run_matrix(args, matrix, target_blocks):
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=matrix, n=args.n, preset=args.preset, seed=args.seed,
        r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
        shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, dtype=np.float64)
    V_exact = np.asarray(V_exact, dtype=np.float64)
    rank = int(args.rank)

    # We also want block b+1 snapshots so we can read the next state.
    blocks_to_report = set(target_blocks)
    blocks_to_report.update(b + 1 for b in target_blocks)
    final_block = max(blocks_to_report)
    snaps = stream_to_block(args, A, V_exact, work_dtype, rank,
                            final_block, blocks_to_report)

    rows = []
    for b in target_blocks:
        if b not in snaps:
            continue
        snap = snaps[b]
        snap_next = snaps.get(b + 1)
        A_cur = snap["A_cur"]
        A_fut = snap["A_fut"]
        A_sketch = snap["A_sketch"] if snap["A_sketch"].size else None
        # Build a basis for the "visible+future" union for projecting the oracle
        if A_sketch is not None:
            B_union = rowspace_basis(np.vstack([A_sketch, A_cur, A_fut]))
        else:
            B_union = rowspace_basis(np.vstack([A_cur, A_fut]))

        for slot in range(rank):
            v_def = unit(snap["V_default"][:, slot])
            # Oracle direction projected and deflated against earlier oracle
            # slot directions to give a clean control.
            v_orc = project_unit(unit(V_exact[:, slot]), B_union)
            for j in range(slot):
                v_orc = deflate_unit(v_orc, project_unit(unit(V_exact[:, j]),
                                                         B_union))
            for label, v, role in [
                ("v_default", v_def, "score_favoured"),
                ("v_oracle_proj", v_orc, "oracle"),
            ]:
                r = slot_record(matrix, b, slot, label, role, v,
                                snap, snap_next, A, V_exact, args)
                if r is not None:
                    rows.append(r)
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="summary/diag02_tail_conspiracy_forward")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--half-win", type=int, default=32)
    p.add_argument("--rank", type=int, default=2)
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
    p.add_argument("--r-sig", type=int, default=2)
    p.add_argument("--alpha-sig", type=float, default=0.003)
    p.add_argument("--alpha-tail", type=float, default=0.0145)
    p.add_argument("--tail-scale", type=float, default=0.99)
    p.add_argument("--sigma1", type=float, default=0.991)
    p.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    p.add_argument("--topk", type=int, default=4,
                   help="row-count for top-row overlap (Jaccard).")
    p.add_argument("--blocks", type=int, nargs="+", default=[1, 6, 12, 31])
    p.add_argument("--matrices", nargs="+", default=[
        "mixed-tail-sharp", "mixed-tail-balanced", "mixed-tail-soft",
        "etf-basket-basis",
    ])
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    target_blocks = sorted(set(int(b) for b in args.blocks))
    rows = []
    for matrix in args.matrices:
        t_m0 = time.time()
        rows.extend(run_matrix(args, matrix, target_blocks))
        print(f"[diag02] {matrix} done in {time.time() - t_m0:.1f}s")

    # Persist
    json_path = os.path.join(out_dir, "probe.json")
    csv_path = os.path.join(out_dir, "probe.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[diag02] wrote {json_path}")
    print(f"[diag02] wrote {csv_path}")
    print(f"[diag02] total {time.time() - t0:.1f}s, rows={len(rows)}")


if __name__ == "__main__":
    main()
