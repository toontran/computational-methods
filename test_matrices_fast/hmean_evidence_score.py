"""HM-evidence score: weighted harmonic mean of u_k(v) = raw_k(v) · N_k / ||A_k||_F².

For each candidate v in the union span at block k:

    raw_sk(v) = ||A_sketch v||^2           (A_sketch = state["s"] · state["V"]^T)
    raw_g1(v) = ||A_cur   v||^2
    raw_g2(v) = ||A_fut   v||^2

    u_sk(v) = (N_sk_total_rows  / ||A_sk_total||_F^2) · raw_sk(v)
    u_g1(v) = (N_block          / ||A_cur||_F^2     ) · raw_g1(v)
    u_g2(v) = (N_block          / ||A_fut||_F^2     ) · raw_g2(v)

    HM_evi(v) = W / (w_sk/u_sk + w_g1/u_g1 + w_g2/u_g2),
                W = w_sk + w_g1 + w_g2 = r + 2 N_block

    score(v) = HM_evi(v) · relH1(v)

Drop-in policy for use with the existing `optimize_future_hmean_in_basis`
scaffolding from `future_hmean_optimizer_diagnostic.py`.
"""

import argparse
import csv
import json
import os
import time

import numpy as np

import cex_restricted_space_probe as probe
import half_window_sliding_hmean_experiment as hm
from future_hmean_optimizer_diagnostic import (
    combined_score,
    optimize_future_hmean_in_basis,
    orth_basis_against,
    rowspace_basis,
)
from hmean_combinations_optimizer_diagnostic import (
    candidate_denoms,
    combination_value_grad,
    optimize_combination_in_basis,
)
from second_slot_tail_bias_diagnostic import make_state, raw_oracle_columns


# --------------------------------------------------------------------------
# Score / gradient
# --------------------------------------------------------------------------


def entropy_relH1_value_grad(A_cur, v):
    """Same relH1 used in combined HM policies (entropy of A_cur v energies)."""
    A_cur = np.asarray(A_cur, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    y = A_cur @ v
    e = y * y
    S = max(float(np.sum(e)), 1e-30)
    p = e / S
    p_pos = np.maximum(p, 1e-300)
    H = -float(np.sum(p * np.log(p_pos)))
    rel = max(H / np.log(max(len(e), 2)), 0.0)
    dH_de = -(np.log(p_pos) + H) / S
    grad_H = A_cur.T @ (2.0 * y * dH_de)
    grad_rel = grad_H / np.log(max(len(e), 2))
    return float(rel), np.ascontiguousarray(grad_rel, dtype=np.float64)


def hm_evi_value_grad(A_sketch, A_cur, A_fut, c_sk, c_g1, c_g2, w_sk, w_g1, w_g2, v):
    """Return (score, grad, u_sk, u_g1, u_g2, hm_evi, relH1).

    Score = HM_evi(v) · relH1(v) where HM_evi is the weighted harmonic mean of
    u_k(v) with weights (w_sk, w_g1, w_g2).
    """
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    eps = 1e-30

    A_sk = np.asarray(A_sketch, dtype=np.float64) if A_sketch is not None and np.asarray(A_sketch).size else None
    A_c = np.asarray(A_cur, dtype=np.float64)
    A_f = np.asarray(A_fut, dtype=np.float64)

    # Raw responses.
    if A_sk is not None:
        y_sk = A_sk @ v
        raw_sk = float(np.dot(y_sk, y_sk))
    else:
        y_sk = None
        raw_sk = 0.0

    y_c = A_c @ v
    raw_g1 = float(np.dot(y_c, y_c))
    y_f = A_f @ v
    raw_g2 = float(np.dot(y_f, y_f))

    # Evidence units.
    u_sk = c_sk * raw_sk if A_sk is not None else 0.0
    u_g1 = c_g1 * raw_g1
    u_g2 = c_g2 * raw_g2

    W = float(w_sk + w_g1 + w_g2)

    if (A_sk is not None and u_sk <= eps) or u_g1 <= eps or u_g2 <= eps:
        # Bottleneck collapses HM to ~0; gradient pushes away from the wall.
        # Use a tiny floor so optimization can still proceed without NaNs.
        u_sk_safe = max(u_sk, eps)
        u_g1_safe = max(u_g1, eps)
        u_g2_safe = max(u_g2, eps)
    else:
        u_sk_safe = u_sk
        u_g1_safe = u_g1
        u_g2_safe = u_g2

    if A_sk is None:
        # Sketch absent (block 1): drop sketch term entirely.
        D = w_g1 / u_g1_safe + w_g2 / u_g2_safe
        W_eff = float(w_g1 + w_g2)
    else:
        D = w_sk / u_sk_safe + w_g1 / u_g1_safe + w_g2 / u_g2_safe
        W_eff = W

    HM_evi = W_eff / max(D, eps)

    # relH1 factor.
    relH1, grad_relH1 = entropy_relH1_value_grad(A_c, v)

    # d(HM_evi)/dv = (HM_evi^2 / W_eff) · 2 · Σ (w_k / (c_k raw_k^2)) · A_k^T A_k v
    coeff = (HM_evi * HM_evi) / max(W_eff, eps) * 2.0
    grad_hm = np.zeros_like(v)
    if A_sk is not None and raw_sk > eps and c_sk > 0:
        grad_hm += coeff * (w_sk / (c_sk * raw_sk * raw_sk)) * (A_sk.T @ y_sk)
    if raw_g1 > eps:
        grad_hm += coeff * (w_g1 / (c_g1 * raw_g1 * raw_g1)) * (A_c.T @ y_c)
    if raw_g2 > eps:
        grad_hm += coeff * (w_g2 / (c_g2 * raw_g2 * raw_g2)) * (A_f.T @ y_f)

    score = HM_evi * relH1
    grad = relH1 * grad_hm + HM_evi * grad_relH1

    return (
        float(score),
        np.ascontiguousarray(grad, dtype=np.float64),
        float(u_sk),
        float(u_g1),
        float(u_g2),
        float(HM_evi),
        float(relH1),
    )


# --------------------------------------------------------------------------
# Adapter into existing optimizer scaffold
# --------------------------------------------------------------------------


def make_evi_optimizer(A_cur, A_fut, A_sketch, c_sk, c_g1, c_g2, w_sk, w_g1, w_g2):
    def value_grad(_unused_cur, _unused_fut, v):
        del _unused_cur, _unused_fut
        score, grad, u_sk, u_g1, u_g2, hm_evi, relH1 = hm_evi_value_grad(
            A_sketch, A_cur, A_fut, c_sk, c_g1, c_g2, w_sk, w_g1, w_g2, v
        )
        # Match the optimizer's expected signature: (val, grad, gain1_share, gain2_share, relH1).
        # We repurpose the share slots to carry u_g1, u_g2 for diagnostics.
        return score, grad, u_g1, u_g2, relH1

    return value_grad


def optimize_hm_evi_in_basis(
    A_cur, A_fut, A_sketch, c_sk, c_g1, c_g2, w_sk, w_g1, w_g2,
    B, starts, rng, maxit, tol, random_starts,
):
    original = optimize_future_hmean_in_basis.__globals__["future_hmean_value_grad"]
    optimize_future_hmean_in_basis.__globals__["future_hmean_value_grad"] = make_evi_optimizer(
        A_cur, A_fut, A_sketch, c_sk, c_g1, c_g2, w_sk, w_g1, w_g2
    )
    try:
        return optimize_future_hmean_in_basis(
            A_cur, A_fut, B, starts, rng, maxit=maxit, tol=tol, random_starts=random_starts
        )
    finally:
        optimize_future_hmean_in_basis.__globals__["future_hmean_value_grad"] = original


# --------------------------------------------------------------------------
# Streaming driver (single matrix, multiple blocks)
# --------------------------------------------------------------------------


def per_block_constants(A, block_id, half_win):
    sk_end = (block_id - 1) * half_win
    A_sk_full = A[:sk_end] if sk_end > 0 else None
    A_cur = A[sk_end:sk_end + half_win]
    A_fut = A[sk_end + half_win:sk_end + 2 * half_win]
    if A_sk_full is None or A_sk_full.size == 0:
        c_sk = 0.0
        sk_F2 = 0.0
        N_sk = 0
    else:
        sk_F2 = float(np.sum(A_sk_full * A_sk_full))
        N_sk = sk_end
        c_sk = N_sk / sk_F2 if sk_F2 > 0 else 0.0
    cur_F2 = float(np.sum(A_cur * A_cur))
    fut_F2 = float(np.sum(A_fut * A_fut))
    c_g1 = half_win / cur_F2 if cur_F2 > 0 else 0.0
    c_g2 = half_win / fut_F2 if fut_F2 > 0 else 0.0
    return {
        "N_sk": N_sk, "sk_F2": sk_F2, "c_sk": c_sk,
        "cur_F2": cur_F2, "c_g1": c_g1,
        "fut_F2": fut_F2, "c_g2": c_g2,
        "A_cur": A_cur, "A_fut": A_fut,
    }


def stream_to_block(args, A, V_exact, work_dtype, rank, target_block, all_blocks_to_report):
    """Stream forward, calling combined-score optimizer at each block to advance the carry.

    Returns dict block_id -> per-block snapshot used by analyze_block.
    """
    half_win = int(args.half_win)
    state = None
    old_row_memory = None
    snapshots = {}

    for block_id in range(1, target_block + 1):
        sk_end = (block_id - 1) * half_win
        A_cur = np.asarray(A[sk_end:sk_end + half_win], dtype=work_dtype)
        A_fut = np.asarray(A[sk_end + half_win:sk_end + 2 * half_win], dtype=work_dtype)
        if state is None:
            A_sketch = np.zeros((0, A.shape[1]), dtype=work_dtype)
            M_gain = A_cur
            rows_seen = A_cur.shape[0]
        else:
            B_top = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
            A_sketch = B_top
            M_gain = np.vstack([B_top, A_cur]).astype(work_dtype, copy=False)
            rows_seen = state["rows_seen"] + A_cur.shape[0]

        V_init = probe.row_norm_seed(A_cur, rank)
        V_score, _, _, _, diag = probe.entropy_iter_basis_forget(
            M_gain=M_gain,
            active_r=rank,
            rows_ref=A.shape[0],
            V_init=np.asarray(V_init, dtype=work_dtype),
            q0=args.q0, qmax=args.qmax, krylov_depth=args.krylov_depth,
            residual_tol=args.residual_tol, expansion_maxit=args.expansion_maxit,
            num_restarts=args.num_restarts, maxit=args.maxit, tol=args.tol,
            rng=np.random.default_rng(args.seed),
            verbose=False, state_prev=state, A_block=A_cur, rows_total=rows_seen,
            reduced_optimizer="cex", basis_selection="greedy",
            work_dtype=work_dtype, expansion_direction="residual",
            reuse_line_search_grad=True, expansion_warm_start=True,
            post_expansion_maxit=args.post_expansion_maxit,
            score_variant="combined", old_row_memory=old_row_memory,
            combined_rank=None, patience=args.patience,
            patience_rel_tol=args.patience_rel_tol,
        )
        V_default = np.ascontiguousarray(np.asarray(V_score[:, :rank], dtype=np.float64))

        if block_id in all_blocks_to_report:
            snapshots[block_id] = {
                "A_cur": np.asarray(A_cur, dtype=np.float64),
                "A_fut": np.asarray(A_fut, dtype=np.float64),
                "A_sketch": np.asarray(A_sketch, dtype=np.float64),
                "M_gain": np.asarray(M_gain, dtype=np.float64),
                "rows_seen": int(rows_seen),
                "state": state,
                "old_row_memory": old_row_memory,
                "V_default": V_default,
                "diag": diag,
            }

        # Advance carry.
        score_selected = np.zeros(rank, dtype=float)
        H_selected = np.zeros(rank, dtype=float)
        for j in range(rank):
            score_selected[j], _, H_selected[j] = probe.score_full_vector_details_forget(
                M_gain, A_cur, V_default[:, j], A.shape[0],
                state_prev=state, score_variant="combined", old_row_memory=None,
            )
        rows_seen_full = sk_end + half_win
        state, V_r, _ = make_state(M_gain, V_default, H_selected, score_selected, rows_seen_full)
        old_row_memory, _ = probe.select_old_row_memory(
            np.asarray(A[:rows_seen_full, :], dtype=work_dtype),
            V_r.astype(work_dtype, copy=False),
            args.old_memory_size if args.old_memory_size > 0 else half_win,
            np.random.default_rng(args.seed + sk_end + half_win),
            return_indices=True,
        )

    return snapshots


def analyze_block(args, matrix, A, V_exact, snap, block_id):
    rank = int(args.rank)
    half_win = int(args.half_win)
    A_cur = snap["A_cur"]
    A_fut = snap["A_fut"]
    A_sketch = snap["A_sketch"]
    M_gain = snap["M_gain"]
    state = snap["state"]
    old_row_memory = snap["old_row_memory"]
    V_default = snap["V_default"]
    diag = snap["diag"]

    consts = per_block_constants(A, block_id, half_win)
    c_sk = consts["c_sk"]
    c_g1 = consts["c_g1"]
    c_g2 = consts["c_g2"]
    weight_mode = getattr(args, "weights", "fixed")
    if weight_mode == "c":
        # User's proposal: HM_evi = (Σc)/Σ(c_k/raw_k) on raws.
        # Implemented in u-form with w_k = c_k² so that w_k/u_k = c_k²/(c_k·raw_k) = c_k/raw_k.
        w_sk = float(c_sk * c_sk)
        w_g1 = float(c_g1 * c_g1)
        w_g2 = float(c_g2 * c_g2)
    elif weight_mode == "c-on-u":
        # Alternative: HM_evi over u_k with weights c_k → equivalent to unweighted HM of raws (scaled).
        w_sk = float(c_sk)
        w_g1 = float(c_g1)
        w_g2 = float(c_g2)
    else:
        w_sk = float(rank)
        w_g1 = float(half_win)
        w_g2 = float(half_win)

    # Subspace bases for projected oracle candidates.
    union_stack = np.vstack([A_sketch, A_cur, A_fut]) if A_sketch.size else np.vstack([A_cur, A_fut])
    B_union = rowspace_basis(union_stack)

    def project_unit(vec, B):
        if vec is None or B is None or B.size == 0:
            return None
        p = B @ (B.T @ vec)
        nv = float(np.linalg.norm(p))
        return None if nv <= 1e-30 else p / nv

    oracle_v1 = V_exact[:, 0] / max(np.linalg.norm(V_exact[:, 0]), 1e-30)
    oracle_v2 = V_exact[:, 1] / max(np.linalg.norm(V_exact[:, 1]), 1e-30)
    oracle_v1_proj = project_unit(oracle_v1, B_union)
    oracle_v2_proj = project_unit(oracle_v2, B_union)

    # Existing HM-triplet (normalized) best for comparison.
    Q_oracle, raw_oracle = raw_oracle_columns(M_gain, V_exact, rank, np.float64)
    pool = hm.build_candidates(V_default, Q_oracle, raw_oracle, M_gain, A_cur, A_fut)
    pool = {k: pool.get(k) for k in hm.ONLINE_POOL}
    weights_existing = (state["rows_seen"] if state is not None else 0, A_cur.shape[0], A_fut.shape[0])
    denoms, _ = candidate_denoms(pool, A_cur, A_fut, A_sketch if A_sketch.size else None)
    if A_sketch.size:
        union_for_search = np.vstack([A_sketch, A_cur, A_fut]).astype(np.float64, copy=False)
    else:
        union_for_search = np.vstack([A_cur, A_fut]).astype(np.float64, copy=False)
    B_search = orth_basis_against(rowspace_basis(union_for_search), V_default[:, 0])

    starts = [V_default[:, 1]]
    starts.extend([v for v in pool.values() if v is not None])
    Vbasis = diag.get("Vbasis_final")
    if Vbasis is not None:
        Vb = np.asarray(Vbasis, dtype=np.float64)
        for j in range(min(Vb.shape[1], 8)):
            starts.append(Vb[:, j])

    triplet_norm = optimize_combination_in_basis(
        "future_hmean_triplet_online",
        A_cur, A_fut, A_sketch if A_sketch.size else None, denoms, weights_existing,
        B_search, starts,
        np.random.default_rng(args.seed + 7777 + block_id),
        args.union_maxit, args.union_tol, args.union_random_starts,
    )

    denoms_raw = {k: 1.0 for k in ("sketch", "gain1", "gain2", "sketch_gain1", "sketch_gain2", "sketch_raw_for_concat")}
    starts_raw = list(starts) + ([oracle_v1_proj, oracle_v2_proj] if oracle_v1_proj is not None else [])
    triplet_raw = optimize_combination_in_basis(
        "future_hmean_triplet_online",
        A_cur, A_fut, A_sketch if A_sketch.size else None, denoms_raw, weights_existing,
        B_search, starts_raw,
        np.random.default_rng(args.seed + 9001 + block_id),
        args.union_maxit, args.union_tol, args.union_random_starts,
    )

    starts_evi = list(starts) + ([oracle_v1_proj, oracle_v2_proj] if oracle_v1_proj is not None else [])
    A_sketch_for_evi = A_sketch if A_sketch.size else None
    evi_best = optimize_hm_evi_in_basis(
        A_cur, A_fut, A_sketch_for_evi,
        c_sk, c_g1, c_g2, w_sk, w_g1, w_g2,
        B_search, starts_evi,
        np.random.default_rng(args.seed + 31337 + block_id),
        args.union_maxit, args.union_tol, args.union_random_starts,
    )

    candidates = {
        "combined_optimizer_v2": V_default[:, 1],
        "hm_triplet_norm_best": None if triplet_norm is None else triplet_norm["vec"],
        "hm_triplet_raw_best": None if triplet_raw is None else triplet_raw["vec"],
        "hm_triplet_evidence_best": None if evi_best is None else evi_best["vec"],
        "oracle_v1_proj_S+G1+G2": oracle_v1_proj,
        "oracle_v2_proj_S+G1+G2": oracle_v2_proj,
    }

    rows = []
    for label, v in candidates.items():
        if v is None:
            continue
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        nv = float(np.linalg.norm(v))
        if nv <= 1e-30:
            continue
        v = v / nv
        score, _, u_sk, u_g1, u_g2, hm_evi, relH1 = hm_evi_value_grad(
            A_sketch_for_evi, A_cur, A_fut, c_sk, c_g1, c_g2, w_sk, w_g1, w_g2, v
        )
        comb = combined_score(M_gain, A_cur, v, A.shape[0], state, old_row_memory)
        align_v1 = float(np.dot(v, oracle_v1) ** 2)
        align_v2 = float(np.dot(v, oracle_v2) ** 2)
        rows.append({
            "matrix": matrix, "block": block_id, "label": label,
            "u_sk": u_sk, "u_g1": u_g1, "u_g2": u_g2,
            "hm_evi": hm_evi, "relH1": relH1, "score": score,
            "combined_score": comb,
            "align_v1": align_v1, "align_v2": align_v2,
        })

    info = {
        "matrix": matrix, "block_id": block_id, "half_win": half_win, "rank": rank,
        "N_sk": consts["N_sk"], "sk_F2": consts["sk_F2"],
        "cur_F2": consts["cur_F2"], "fut_F2": consts["fut_F2"],
        "c_sk": c_sk, "c_g1": c_g1, "c_g2": c_g2,
        "w_sk": w_sk, "w_g1": w_g1, "w_g2": w_g2,
        "union_dim": int(B_union.shape[1]),
    }
    return info, rows


# --------------------------------------------------------------------------
# Gradient check (finite differences)
# --------------------------------------------------------------------------


def gradient_check(A, V_exact, args, matrix, block_id):
    """Verify analytic gradient of HM_evi · relH1 against finite differences."""
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    rank = int(args.rank)
    half_win = int(args.half_win)
    blocks = {block_id}
    snaps = stream_to_block(args, A, V_exact, work_dtype, rank, block_id, blocks)
    snap = snaps[block_id]
    consts = per_block_constants(A, block_id, half_win)
    c_sk, c_g1, c_g2 = consts["c_sk"], consts["c_g1"], consts["c_g2"]
    weight_mode = getattr(args, "weights", "fixed")
    if weight_mode == "c":
        w_sk, w_g1, w_g2 = float(c_sk * c_sk), float(c_g1 * c_g1), float(c_g2 * c_g2)
    elif weight_mode == "c-on-u":
        w_sk, w_g1, w_g2 = float(c_sk), float(c_g1), float(c_g2)
    else:
        w_sk, w_g1, w_g2 = float(rank), float(half_win), float(half_win)

    rng = np.random.default_rng(0)
    n = A.shape[1]
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)

    score0, grad, *_ = hm_evi_value_grad(
        snap["A_sketch"], snap["A_cur"], snap["A_fut"],
        c_sk, c_g1, c_g2, w_sk, w_g1, w_g2, v,
    )

    h = 1e-6
    sample = rng.choice(n, size=20, replace=False)
    fd_at = np.zeros(len(sample))
    for k, i in enumerate(sample):
        ei = np.zeros(n); ei[i] = 1.0
        s_p, *_ = hm_evi_value_grad(snap["A_sketch"], snap["A_cur"], snap["A_fut"], c_sk, c_g1, c_g2, w_sk, w_g1, w_g2, v + h*ei)
        s_m, *_ = hm_evi_value_grad(snap["A_sketch"], snap["A_cur"], snap["A_fut"], c_sk, c_g1, c_g2, w_sk, w_g1, w_g2, v - h*ei)
        fd_at[k] = (s_p - s_m) / (2 * h)
    grad_at = grad[sample]
    abs_err = np.abs(grad_at - fd_at)
    rel_err = float(np.max(abs_err) / max(np.max(np.abs(fd_at)), 1e-30))
    print(f"  gradient check at block {block_id}: score={score0:.6e}  max |g-fd|={float(np.max(abs_err)):.4e}  rel={rel_err:.4e}")
    print(f"    sample analytic g: {grad_at[:5]}")
    print(f"    sample fd       g: {fd_at[:5]}")
    return rel_err


# --------------------------------------------------------------------------
# Main entry: per-matrix runner
# --------------------------------------------------------------------------


def run_matrix(args, matrix, blocks_to_report):
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=matrix, n=args.n, preset=args.preset, seed=args.seed,
        r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
        shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, np.float64)
    V_exact = np.asarray(V_exact, np.float64)
    target = max(blocks_to_report)
    snapshots = stream_to_block(args, A, V_exact, work_dtype, int(args.rank), target, set(blocks_to_report))
    out_rows = []
    out_info = {}
    for b in sorted(blocks_to_report):
        if b not in snapshots:
            continue
        info, rows = analyze_block(args, matrix, A, V_exact, snapshots[b], b)
        out_info[b] = info
        out_rows.extend(rows)
    return out_info, out_rows


def write_text(path, infos, rows):
    by_block = {}
    for r in rows:
        by_block.setdefault(r["block"], []).append(r)
    with open(path, "w", encoding="utf-8") as f:
        for block_id in sorted(by_block.keys()):
            info = infos[block_id]
            f.write(f"== block {block_id}  matrix={info['matrix']}  N_sk={info['N_sk']}  union_dim={info['union_dim']} ==\n")
            f.write(f"  c_sk={info['c_sk']:.4e}  c_g1={info['c_g1']:.4e}  c_g2={info['c_g2']:.4e}\n")
            f.write(f"  w_sk={info['w_sk']}  w_g1={info['w_g1']}  w_g2={info['w_g2']}\n")
            f.write(f"  {'label':<28} {'u_sk':>8} {'u_g1':>8} {'u_g2':>8}  {'HM_evi':>9} {'relH1':>7}  "
                    f"{'score':>10} {'combined':>10}  {'align_v1':>9} {'align_v2':>9}\n")
            for r in by_block[block_id]:
                f.write(
                    f"  {r['label']:<28} {r['u_sk']:>8.4f} {r['u_g1']:>8.4f} {r['u_g2']:>8.4f}  "
                    f"{r['hm_evi']:>9.4e} {r['relH1']:>7.4f}  {r['score']:>10.4e} {r['combined_score']:>10.4e}  "
                    f"{r['align_v1']:>9.4f} {r['align_v2']:>9.4f}\n"
                )
            f.write("\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="mixed-tail-sharp")
    p.add_argument("--matrices", nargs="*", default=None,
                   help="If given, run multiple matrices and summarise.")
    p.add_argument("--out-prefix", default="summary/hmean_evidence_score")
    p.add_argument("--blocks", nargs="+", type=int, default=[2, 6, 12])
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
    p.add_argument("--union-maxit", type=int, default=120)
    p.add_argument("--union-tol", type=float, default=1e-9)
    p.add_argument("--union-random-starts", type=int, default=24)
    p.add_argument("--r-sig", type=int, default=2)
    p.add_argument("--alpha-sig", type=float, default=0.003)
    p.add_argument("--alpha-tail", type=float, default=0.0145)
    p.add_argument("--tail-scale", type=float, default=0.99)
    p.add_argument("--sigma1", type=float, default=0.991)
    p.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    p.add_argument("--gradient-check", action="store_true")
    p.add_argument("--weights", choices=("fixed", "c", "c-on-u"), default="fixed",
                   help="HM weight scheme. 'fixed': (rank, half_win, half_win) on u (current). "
                        "'c': u-form with w=c² → equivalent to (Σc)/Σ(c_k/raw_k) on raws (user proposal). "
                        "'c-on-u': u-form with w=c → equivalent to unweighted HM of raws (scaled).")
    return p.parse_args()


def main():
    args = parse_args()
    matrices = args.matrices if args.matrices else [args.matrix]

    if args.gradient_check:
        # Run gradient check on the primary matrix at each requested block.
        work_dtype = np.float32 if args.dtype == "float32" else np.float64
        A, V_exact, _, _ = probe.generate_matrix_input(
            matrix=args.matrix, n=args.n, preset=args.preset, seed=args.seed,
            r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
            tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
            shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
        )
        A = np.asarray(A, np.float64)
        V_exact = np.asarray(V_exact, np.float64)
        for b in sorted(args.blocks):
            gradient_check(A, V_exact, args, args.matrix, b)
        return

    overall_rows = []
    overall_infos = {}
    for matrix in matrices:
        t0 = time.time()
        infos, rows = run_matrix(args, matrix, args.blocks)
        for b, info in infos.items():
            overall_infos[(matrix, b)] = info
        overall_rows.extend(rows)
        print(f"done matrix={matrix} blocks={list(infos.keys())} elapsed={time.time()-t0:.2f}s")

    csv_path = args.out_prefix + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if overall_rows:
            w = csv.DictWriter(f, fieldnames=list(overall_rows[0].keys()))
            w.writeheader()
            w.writerows(overall_rows)
    json_path = args.out_prefix + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "infos": {f"{m}|{b}": info for (m, b), info in overall_infos.items()},
            "rows": overall_rows,
        }, f, indent=2, sort_keys=True, default=float)
    txt_path = args.out_prefix + ".txt"
    write_text(txt_path, {b: overall_infos[(matrices[0], b)] for b in sorted({r['block'] for r in overall_rows if r['matrix'] == matrices[0]})}, [r for r in overall_rows if r['matrix'] == matrices[0]])
    # Per-matrix txt
    for matrix in matrices:
        mtxt = args.out_prefix + f"_{matrix}.txt"
        rows_m = [r for r in overall_rows if r['matrix'] == matrix]
        infos_m = {b: overall_infos[(matrix, b)] for b in sorted({r['block'] for r in rows_m})}
        if rows_m:
            write_text(mtxt, infos_m, rows_m)
    print(f"wrote {csv_path} {json_path} {txt_path}")


if __name__ == "__main__":
    main()
