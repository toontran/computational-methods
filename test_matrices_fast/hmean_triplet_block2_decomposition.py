"""Block-2 decomposition probe for the HM-triplet (sketch + gain1 + gain2) policy.

For the chosen matrix (default: residual-spiky-shocks), stream up to block 2 and
expose, for several v2 candidates, the decomposition

  v = alpha * oracle_proj_unit + sqrt(1 - alpha^2) * outside_dir

where oracle_proj is the projection of V_exact[:, 1] onto rowspan(sketch + gain1
+ gain2). The outside direction is then decomposed into orthogonal pieces along
sketch, gain1\sketch, gain2\(sketch + gain1) and matched against the remaining
exact right singular vectors V_exact[:, j != 1].

Both the HM-triplet policy score and the combined-forgetting score are reported
per candidate so we can see which trade-off the HM-triplet objective is making
versus the combined optimizer pick.
"""

import argparse
import csv
import json
from typing import Dict, List, Optional

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


def normed(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= 1e-30:
        return None
    return v / n


def project_coeffs(B, v):
    if B is None or B.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(B, dtype=np.float64).T @ np.asarray(v, dtype=np.float64).reshape(-1)


def proj_mass(B, v):
    c = project_coeffs(B, v)
    return float(np.dot(c, c))


def orthogonalize_against_basis(C, *bases):
    """Orthonormal basis of column-span(C) within (combined complement of bases)."""
    C = np.asarray(C, dtype=np.float64)
    if C.size == 0:
        return np.zeros((C.shape[0], 0), dtype=np.float64) if C.ndim == 2 else None
    out = C.copy()
    for B in bases:
        if B is None or B.size == 0:
            continue
        B = np.asarray(B, dtype=np.float64)
        out = out - B @ (B.T @ out)
    if out.size == 0:
        return np.zeros((C.shape[0], 0), dtype=np.float64)
    Q, R = np.linalg.qr(out)
    diag = np.abs(np.diag(R))
    if diag.size == 0:
        return np.zeros((C.shape[0], 0), dtype=np.float64)
    keep = diag > max(float(diag.max()) * 1e-10, 1e-30)
    return np.ascontiguousarray(Q[:, keep], dtype=np.float64)


def stream_block(args, A, work_dtype, state, old_row_memory, A_cur, A_sketch, M_gain, rows_seen, n_total):
    """Run the combined-score optimizer on one block and return V_default and diag."""
    rank = int(args.rank)
    V_init = probe.row_norm_seed(A_cur, rank)
    V_score, _, _, _, diag = probe.entropy_iter_basis_forget(
        M_gain=M_gain,
        active_r=rank,
        rows_ref=n_total,
        V_init=np.asarray(V_init, dtype=work_dtype),
        q0=args.q0,
        qmax=args.qmax,
        krylov_depth=args.krylov_depth,
        residual_tol=args.residual_tol,
        expansion_maxit=args.expansion_maxit,
        num_restarts=args.num_restarts,
        maxit=args.maxit,
        tol=args.tol,
        rng=np.random.default_rng(args.seed),
        verbose=False,
        state_prev=state,
        A_block=A_cur,
        rows_total=rows_seen,
        reduced_optimizer="cex",
        basis_selection="greedy",
        work_dtype=work_dtype,
        expansion_direction="residual",
        reuse_line_search_grad=True,
        expansion_warm_start=True,
        post_expansion_maxit=args.post_expansion_maxit,
        score_variant="combined",
        old_row_memory=old_row_memory,
        combined_rank=None,
        patience=args.patience,
        patience_rel_tol=args.patience_rel_tol,
    )
    V_default = np.ascontiguousarray(np.asarray(V_score[:, :rank], dtype=np.float64))
    return V_default, diag


def update_carry(args, A, work_dtype, state, V_default, M_gain, A_cur, mid0, rank, n_total, end0):
    """Mirror hmean_combinations_optimizer_diagnostic state-advance logic."""
    score_selected = np.zeros(rank, dtype=float)
    H_selected = np.zeros(rank, dtype=float)
    for j in range(rank):
        score_selected[j], _, H_selected[j] = probe.score_full_vector_details_forget(
            M_gain,
            A_cur,
            V_default[:, j],
            n_total,
            state_prev=state,
            score_variant="combined",
            old_row_memory=None,
        )
    rows_seen = mid0  # rows actually streamed up to mid0
    state_new, V_r, _ = make_state(M_gain, V_default, H_selected, score_selected, rows_seen)
    old_row_memory_new, _ = probe.select_old_row_memory(
        np.asarray(A[:mid0, :], dtype=work_dtype),
        V_r.astype(work_dtype, copy=False),
        args.old_memory_size if args.old_memory_size > 0 else int(args.half_win),
        np.random.default_rng(args.seed + end0),
        return_indices=True,
    )
    return state_new, old_row_memory_new


def build_block2_state(args, matrix):
    """Stream block 1 only and return (state, old_row_memory, A, V_exact)."""
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
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
    A = np.asarray(A, dtype=np.float64)
    V_exact = np.asarray(V_exact, dtype=np.float64)
    rank = int(args.rank)
    half_win = int(args.half_win)

    state = None
    old_row_memory = None

    # Block 1: build sketch state.
    start0 = 0
    mid0 = start0 + half_win
    end0 = mid0 + half_win

    A_cur1 = np.asarray(A[start0:mid0, :], dtype=work_dtype)
    M_gain1 = A_cur1
    V_init1 = probe.row_norm_seed(A_cur1, rank)
    V_score1, _, _, _, _ = probe.entropy_iter_basis_forget(
        M_gain=M_gain1,
        active_r=rank,
        rows_ref=A.shape[0],
        V_init=np.asarray(V_init1, dtype=work_dtype),
        q0=args.q0,
        qmax=args.qmax,
        krylov_depth=args.krylov_depth,
        residual_tol=args.residual_tol,
        expansion_maxit=args.expansion_maxit,
        num_restarts=args.num_restarts,
        maxit=args.maxit,
        tol=args.tol,
        rng=np.random.default_rng(args.seed),
        verbose=False,
        state_prev=None,
        A_block=A_cur1,
        rows_total=A_cur1.shape[0],
        reduced_optimizer="cex",
        basis_selection="greedy",
        work_dtype=work_dtype,
        expansion_direction="residual",
        reuse_line_search_grad=True,
        expansion_warm_start=True,
        post_expansion_maxit=args.post_expansion_maxit,
        score_variant="combined",
        old_row_memory=None,
        combined_rank=None,
        patience=args.patience,
        patience_rel_tol=args.patience_rel_tol,
    )
    V_default1 = np.ascontiguousarray(np.asarray(V_score1[:, :rank], dtype=np.float64))
    score_selected1 = np.zeros(rank, dtype=float)
    H_selected1 = np.zeros(rank, dtype=float)
    for j in range(rank):
        score_selected1[j], _, H_selected1[j] = probe.score_full_vector_details_forget(
            M_gain1,
            A_cur1,
            V_default1[:, j],
            A.shape[0],
            state_prev=None,
            score_variant="combined",
            old_row_memory=None,
        )
    state, V_r1, _ = make_state(M_gain1, V_default1, H_selected1, score_selected1, A_cur1.shape[0])
    old_row_memory, _ = probe.select_old_row_memory(
        np.asarray(A[:mid0, :], dtype=work_dtype),
        V_r1.astype(work_dtype, copy=False),
        args.old_memory_size if args.old_memory_size > 0 else half_win,
        np.random.default_rng(args.seed + end0),
        return_indices=True,
    )

    return {
        "A": A,
        "V_exact": V_exact,
        "state": state,
        "old_row_memory": old_row_memory,
        "V_default1": V_default1,
        "work_dtype": work_dtype,
    }


def candidate_metrics(label, v, ctx):
    if v is None:
        return None
    v = np.ascontiguousarray(np.asarray(v, dtype=np.float64).reshape(-1))
    nv = float(np.linalg.norm(v))
    if nv <= 1e-30:
        return None
    v = v / nv

    # Decompose against the v1 axis (first column of V_default at block 2) to
    # mirror the way the optimizer searches for v2 orthogonal to v1.
    v1 = ctx["v1"]
    v_perp_v1 = v - v1 * float(np.dot(v1, v))
    perp_norm = float(np.linalg.norm(v_perp_v1))

    # Oracle projection direction.
    p_unit = ctx["oracle_proj_unit"]
    cos_oracle = float(np.dot(v, p_unit)) if p_unit is not None else 0.0
    mass_oracle = cos_oracle * cos_oracle

    # Outside direction (orthogonal to oracle projection).
    if p_unit is not None:
        outside = v - cos_oracle * p_unit
    else:
        outside = v.copy()
    outside_norm = float(np.linalg.norm(outside))
    outside_unit = outside / outside_norm if outside_norm > 1e-30 else None

    # Subspace masses for v itself.
    sk_share = proj_mass(ctx["B_sketch"], v)
    g1_share = proj_mass(ctx["B_gain1"], v)
    g2_share = proj_mass(ctx["B_gain2"], v)
    union_share = proj_mass(ctx["B_union"], v)

    # Exclusive (orthogonalised) shares: sketch first, then gain1\sketch, then
    # gain2\(sketch + gain1). These sum to the union share.
    sk_excl = proj_mass(ctx["B_sketch_excl"], v)
    g1_excl = proj_mass(ctx["B_gain1_excl"], v)
    g2_excl = proj_mass(ctx["B_gain2_excl"], v)

    # Outside-direction subspace alignment.
    if outside_unit is not None:
        out_sk = proj_mass(ctx["B_sketch"], outside_unit)
        out_g1 = proj_mass(ctx["B_gain1"], outside_unit)
        out_g2 = proj_mass(ctx["B_gain2"], outside_unit)
        out_union = proj_mass(ctx["B_union"], outside_unit)
        out_sk_excl = proj_mass(ctx["B_sketch_excl"], outside_unit)
        out_g1_excl = proj_mass(ctx["B_gain1_excl"], outside_unit)
        out_g2_excl = proj_mass(ctx["B_gain2_excl"], outside_unit)
    else:
        out_sk = out_g1 = out_g2 = np.nan
        out_union = np.nan
        out_sk_excl = out_g1_excl = out_g2_excl = np.nan

    # Singular direction alignment: cos^2 of v and of outside_dir vs V_exact[:, j].
    sing_cos2_v = {}
    sing_cos2_outside = {}
    for j in range(ctx["V_exact"].shape[1]):
        col = ctx["V_exact"][:, j]
        cn = float(np.linalg.norm(col))
        if cn <= 1e-30:
            continue
        u = col / cn
        sing_cos2_v[j] = float(np.dot(v, u) ** 2)
        if outside_unit is not None:
            sing_cos2_outside[j] = float(np.dot(outside_unit, u) ** 2)
        else:
            sing_cos2_outside[j] = np.nan

    # HM-triplet policy score and combined score.
    hm_val, _, hm_parts = combination_value_grad(
        "future_hmean_triplet_online",
        ctx["A_cur"], ctx["A_fut"], ctx["A_sketch"],
        ctx["denoms"], ctx["weights"], v,
    )
    sk_sh = float(hm_parts.get("sketch_share", np.nan))
    g1_sh = float(hm_parts.get("gain1_share", np.nan))
    g2_sh = float(hm_parts.get("gain2_share", np.nan))
    rel_h1 = float(hm_parts.get("relH1", np.nan))
    # Raw energies (numerators) along sketch, gain1, gain2 separately.
    A_sk = np.asarray(ctx["A_sketch"], dtype=np.float64)
    A_c = np.asarray(ctx["A_cur"], dtype=np.float64)
    A_f = np.asarray(ctx["A_fut"], dtype=np.float64)
    raw_sketch = float(np.dot(A_sk @ v, A_sk @ v)) if A_sk.size else np.nan
    raw_gain1 = float(np.dot(A_c @ v, A_c @ v))
    raw_gain2 = float(np.dot(A_f @ v, A_f @ v))
    # Per-eigendirection breakdown of raw_sk: raw_sk = sum_i s_i^2 * cos^2(V_i, v)
    sketch_V_arr = ctx.get("sketch_V")
    sketch_s2_arr = ctx.get("sketch_s2")
    if sketch_V_arr is not None and sketch_V_arr.shape[1] and sketch_s2_arr is not None:
        cos2_per_dir = np.array([float(np.dot(sketch_V_arr[:, i], v)) ** 2
                                 for i in range(sketch_V_arr.shape[1])], dtype=np.float64)
        raw_sk_components = sketch_s2_arr * cos2_per_dir
    else:
        cos2_per_dir = np.zeros(0, dtype=np.float64)
        raw_sk_components = np.zeros(0, dtype=np.float64)
    raw_triplet = [raw_sketch, raw_gain1, raw_gain2]
    raw_finite = [x for x in raw_triplet if np.isfinite(x) and x > 0]
    if len(raw_finite) == 3:
        hm_raw_gain_factor = 3.0 / sum(1.0 / x for x in raw_triplet)
    elif any(np.isfinite(x) and x <= 0 for x in raw_triplet):
        hm_raw_gain_factor = 0.0
    else:
        hm_raw_gain_factor = np.nan
    hm_raw_score = hm_raw_gain_factor * rel_h1 if np.isfinite(hm_raw_gain_factor) else np.nan
    # HM-triplet decomposes as gain_factor * entropy_factor.
    finite_shares = [s for s in (sk_sh, g1_sh, g2_sh) if np.isfinite(s) and s > 0]
    if len(finite_shares) == 3:
        hm_gain_factor = 3.0 / (1.0 / sk_sh + 1.0 / g1_sh + 1.0 / g2_sh)
    else:
        hm_gain_factor = 0.0 if any(s <= 0 for s in (sk_sh, g1_sh, g2_sh)) else np.nan

    comb_details = probe.combined_score_component_details(
        ctx["M_gain"], ctx["A_cur"], v,
        ctx["rows_ref"], state_prev=ctx["state"], old_row_memory=ctx["old_row_memory"],
    )
    comb = float(comb_details["score_total"])
    comb_gain_factor = float(comb_details["gain2"])
    comb_phi = float(comb_details["phi"])
    comb_c = float(comb_details["combined_c"])
    comb_pooled_relH = float(comb_details["pooled_rel_H"])
    comb_pooled_H = float(comb_details["pooled_H"])
    comb_pooled_y2_sq = float(comb_details["pooled_y2_sq"])
    comb_pooled_y4_4 = float(comb_details["pooled_y4_4"])

    return {
        "label": label,
        "norm": nv,
        "cos_v1": float(np.dot(v, v1)),
        "perp_v1_norm": perp_norm,
        "mass_in_oracle_proj": mass_oracle,
        "mass_outside_oracle_proj": 1.0 - mass_oracle,
        "share_sketch": sk_share,
        "share_gain1": g1_share,
        "share_gain2": g2_share,
        "share_union": union_share,
        "share_sketch_excl": sk_excl,
        "share_gain1_excl": g1_excl,
        "share_gain2_excl": g2_excl,
        "outside_share_sketch": out_sk,
        "outside_share_gain1": out_g1,
        "outside_share_gain2": out_g2,
        "outside_share_union": out_union,
        "outside_share_sketch_excl": out_sk_excl,
        "outside_share_gain1_excl": out_g1_excl,
        "outside_share_gain2_excl": out_g2_excl,
        "outside_norm_in_unit": outside_norm,
        "raw_sketch": raw_sketch,
        "raw_gain1": raw_gain1,
        "raw_gain2": raw_gain2,
        "cos2_V_v_per_dir": cos2_per_dir.tolist(),
        "raw_sk_components_per_dir": raw_sk_components.tolist(),
        "share_sketch_pool": sk_sh,
        "share_gain1_pool": g1_sh,
        "share_gain2_pool": g2_sh,
        "hm_gain_factor": hm_gain_factor,
        "hm_entropy_factor": rel_h1,
        "hm_raw_gain_factor": hm_raw_gain_factor,
        "hm_raw_score": hm_raw_score,
        "combined_gain_factor": comb_gain_factor,
        "combined_phi_factor": comb_phi,
        "combined_c": comb_c,
        "combined_pooled_relH": comb_pooled_relH,
        "combined_pooled_H": comb_pooled_H,
        "combined_pooled_y2_sq": comb_pooled_y2_sq,
        "combined_pooled_y4_4": comb_pooled_y4_4,
        "hm_triplet_score": hm_val,
        "hm_relH1": hm_parts.get("relH1", np.nan),
        "hm_sketch_share": hm_parts.get("sketch_share", np.nan),
        "hm_gain1_share": hm_parts.get("gain1_share", np.nan),
        "hm_gain2_share": hm_parts.get("gain2_share", np.nan),
        "combined_score": comb,
        "sing_cos2_v": sing_cos2_v,
        "sing_cos2_outside": sing_cos2_outside,
    }


def analyze_block(args, A, V_exact, state, old_row_memory, work_dtype, block_id, matrix):
    rank = int(args.rank)
    half_win = int(args.half_win)
    start0 = (block_id - 1) * half_win
    mid0 = start0 + half_win
    end0 = mid0 + half_win

    A_cur = np.asarray(A[start0:mid0, :], dtype=work_dtype)
    A_fut = np.asarray(A[mid0:end0, :], dtype=work_dtype)
    if state is None:
        A_sketch = np.zeros((0, A.shape[1]), dtype=work_dtype)
        M_gain = A_cur
        rows_seen = A_cur.shape[0]
    else:
        B_top = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
        A_sketch = B_top
        M_gain = np.vstack([B_top, A_cur]).astype(work_dtype, copy=False)
        rows_seen = state["rows_seen"] + A_cur.shape[0]

    V_default, diag = stream_block(
        args, A, work_dtype, state, old_row_memory,
        A_cur, A_sketch, M_gain, rows_seen, A.shape[0],
    )

    # Build subspace bases.
    B_sketch = rowspace_basis(np.asarray(A_sketch, dtype=np.float64))
    B_gain1 = rowspace_basis(np.asarray(A_cur, dtype=np.float64))
    B_gain2 = rowspace_basis(np.asarray(A_fut, dtype=np.float64))
    union_stack = np.vstack([
        np.asarray(A_sketch, dtype=np.float64),
        np.asarray(A_cur, dtype=np.float64),
        np.asarray(A_fut, dtype=np.float64),
    ])
    B_union = rowspace_basis(union_stack)
    # Sketch + block1 only.
    s_g1_stack = np.vstack([
        np.asarray(A_sketch, dtype=np.float64),
        np.asarray(A_cur, dtype=np.float64),
    ])
    B_sketch_gain1 = rowspace_basis(s_g1_stack)

    # Exclusive (Gram-Schmidt) decomposition of the union.
    B_sketch_excl = B_sketch  # already orthonormal
    B_gain1_excl = orthogonalize_against_basis(B_gain1, B_sketch_excl)
    B_gain2_excl = orthogonalize_against_basis(B_gain2, B_sketch_excl, B_gain1_excl)

    # Oracle v2 and its projection on the union.
    oracle_v2 = normed(V_exact[:, 1])
    if B_union.size:
        oracle_proj = B_union @ (B_union.T @ oracle_v2)
    else:
        oracle_proj = np.zeros_like(oracle_v2)
    oracle_proj_norm = float(np.linalg.norm(oracle_proj))
    oracle_proj_unit = oracle_proj / oracle_proj_norm if oracle_proj_norm > 1e-30 else None

    def project_unit(vec, B):
        if vec is None or B is None or B.size == 0:
            return None, 0.0
        p = B @ (B.T @ vec)
        n = float(np.linalg.norm(p))
        if n <= 1e-30:
            return None, 0.0
        return p / n, n

    oracle_v1 = normed(V_exact[:, 0])
    oracle_v1_proj_S_G1, oracle_v1_proj_S_G1_norm = project_unit(oracle_v1, B_sketch_gain1)
    oracle_v1_proj_union, oracle_v1_proj_union_norm = project_unit(oracle_v1, B_union)
    oracle_v2_proj_S_G1, oracle_v2_proj_S_G1_norm = project_unit(oracle_v2, B_sketch_gain1)

    # HM-triplet policy normalisers (online pool).
    Q_oracle, raw_oracle = raw_oracle_columns(M_gain, V_exact, rank, np.float64)
    candidates_pool = hm.build_candidates(V_default, Q_oracle, raw_oracle, M_gain, A_cur, A_fut)
    candidates_pool = {k: candidates_pool.get(k) for k in hm.ONLINE_POOL}
    prior_seen = 0 if state is None else int(state["rows_seen"])
    weights = (prior_seen, A_cur.shape[0], A_fut.shape[0])
    denoms, _ = candidate_denoms(candidates_pool, A_cur, A_fut, A_sketch)

    # Best-in-union HM-triplet optimum (the same search the diagnostic uses).
    union_for_search = np.vstack([A_cur, A_fut]).astype(np.float64, copy=False)
    B_search = orth_basis_against(rowspace_basis(union_for_search), V_default[:, 0])
    starts = [V_default[:, 1]]
    starts.extend([v for v in candidates_pool.values() if v is not None])
    Vbasis = diag.get("Vbasis_final")
    if Vbasis is not None:
        Vb = np.asarray(Vbasis, dtype=np.float64)
        for j in range(min(Vb.shape[1], 8)):
            starts.append(Vb[:, j])
    triplet_best = optimize_combination_in_basis(
        "future_hmean_triplet_online",
        A_cur, A_fut, A_sketch, denoms, weights,
        B_search, starts,
        np.random.default_rng(args.seed + 7777),
        args.union_maxit, args.union_tol, args.union_random_starts,
    )
    triplet_best_v = None if triplet_best is None else triplet_best["vec"]

    # Same optimizer but with denominators set to 1 — i.e. HM of the raw
    # energies ||A_k v||^2 directly. Gradient flows through quad_share_value_grad
    # with denom=1 (gradient = 2 A^T A v) and hmean_many_value_grad uses the
    # standard d(HM)/dxi = HM^2 / (3 xi^2).
    denoms_raw = {
        "sketch": 1.0,
        "gain1": 1.0,
        "gain2": 1.0,
        "sketch_gain1": 1.0,
        "sketch_gain2": 1.0,
        "sketch_raw_for_concat": 1.0,
    }
    starts_raw = list(starts)
    if oracle_proj_unit is not None:
        starts_raw.append(oracle_proj_unit)
    triplet_raw_best = optimize_combination_in_basis(
        "future_hmean_triplet_online",
        A_cur, A_fut, A_sketch, denoms_raw, weights,
        B_search, starts_raw,
        np.random.default_rng(args.seed + 9001),
        args.union_maxit, args.union_tol, args.union_random_starts,
    )
    triplet_raw_best_v = None if triplet_raw_best is None else triplet_raw_best["vec"]

    # Sketch eigendecomposition (built earlier in info block but needed in ctx).
    if state is None:
        sketch_s_local = np.zeros(rank, dtype=np.float64)
        sketch_V_local = np.zeros((A.shape[1], 0), dtype=np.float64)
    else:
        sketch_s_local = np.asarray(state["s"], dtype=np.float64).reshape(-1)
        sketch_V_local = np.asarray(state["V"], dtype=np.float64)
    sketch_s2_local = sketch_s_local ** 2

    # Assemble candidate set.
    ctx = {
        "v1": np.ascontiguousarray(V_default[:, 0]),
        "B_sketch": B_sketch,
        "sketch_V": sketch_V_local,
        "sketch_s2": sketch_s2_local,
        "B_gain1": B_gain1,
        "B_gain2": B_gain2,
        "B_union": B_union,
        "B_sketch_excl": B_sketch_excl,
        "B_gain1_excl": B_gain1_excl,
        "B_gain2_excl": B_gain2_excl,
        "oracle_proj_unit": oracle_proj_unit,
        "V_exact": V_exact,
        "A_cur": np.asarray(A_cur, dtype=np.float64),
        "A_fut": np.asarray(A_fut, dtype=np.float64),
        "A_sketch": np.asarray(A_sketch, dtype=np.float64),
        "M_gain": np.asarray(M_gain, dtype=np.float64),
        "denoms": denoms,
        "weights": weights,
        "rows_ref": A.shape[0],
        "state": state,
        "old_row_memory": old_row_memory,
    }

    cands = [
        ("combined_optimizer_v2",        V_default[:, 1]),
        ("hm_triplet_best_in_union",     triplet_best_v),
        ("hm_triplet_raw_best_in_union", triplet_raw_best_v),
        ("oracle_v1_unprojected",        oracle_v1),
        ("oracle_v1_proj_S+G1",          oracle_v1_proj_S_G1),
        ("oracle_v1_proj_S+G1+G2",       oracle_v1_proj_union),
        ("oracle_v2_unprojected",        oracle_v2),
        ("oracle_v2_proj_S+G1",          oracle_v2_proj_S_G1),
        ("oracle_v2_proj_S+G1+G2",       oracle_proj_unit),
        ("opt2_pool_candidate",          candidates_pool.get("opt2")),
        ("q2_vs_q1oracle_pool",          candidates_pool.get("q2_vs_q1oracle")),
        ("mgain_deflated_svd_pool",      candidates_pool.get("mgain_deflated_svd")),
        ("block_complement_pool",        candidates_pool.get("block_complement")),
    ]

    rows = []
    for label, v in cands:
        rec = candidate_metrics(label, v, ctx)
        if rec is None:
            continue
        rows.append(rec)

    # Sketch evidence diagnostics: state["s"] are the singular values of the
    # carried operator (rows_seen rows summarised in rank-r). state["V"] are the
    # carried right-singular directions. raw_sk(v) = sum_i s_i^2 cos^2(V_i, v).
    if state is None:
        sketch_s = np.zeros(rank, dtype=np.float64)
        sketch_V = np.zeros((A.shape[1], 0), dtype=np.float64)
    else:
        sketch_s = np.asarray(state["s"], dtype=np.float64).reshape(-1)
        sketch_V = np.asarray(state["V"], dtype=np.float64)
    sketch_s2 = sketch_s ** 2
    sketch_frob_sq = float(sketch_s2.sum())

    cos2_V_v1 = np.zeros(sketch_V.shape[1], dtype=np.float64)
    cos2_V_v2 = np.zeros(sketch_V.shape[1], dtype=np.float64)
    raw_sk_breakdown_v1 = np.zeros(sketch_V.shape[1], dtype=np.float64)
    raw_sk_breakdown_v2 = np.zeros(sketch_V.shape[1], dtype=np.float64)
    if oracle_v1 is not None and sketch_V.shape[1]:
        for i in range(sketch_V.shape[1]):
            ci = float(np.dot(sketch_V[:, i], oracle_v1)) ** 2
            cos2_V_v1[i] = ci
            raw_sk_breakdown_v1[i] = sketch_s2[i] * ci
    if oracle_v2 is not None and sketch_V.shape[1]:
        for i in range(sketch_V.shape[1]):
            ci = float(np.dot(sketch_V[:, i], oracle_v2)) ** 2
            cos2_V_v2[i] = ci
            raw_sk_breakdown_v2[i] = sketch_s2[i] * ci

    # Cosine of the principal angles between span(state.V) and span(V_exact[:,:rank]).
    if sketch_V.shape[1] and oracle_v1 is not None and oracle_v2 is not None:
        Vex = V_exact[:, :rank]
        sing_vals = np.linalg.svd(sketch_V.T @ Vex, compute_uv=False)
        principal_cos = np.pad(np.asarray(sing_vals, dtype=np.float64), (0, max(rank - sing_vals.size, 0)))
    else:
        principal_cos = np.zeros(rank, dtype=np.float64)

    info = {
        "matrix": matrix,
        "n": int(A.shape[1]),
        "block_id": int(block_id),
        "half_win": int(half_win),
        "rank": int(rank),
        "rows_seen": int(rows_seen),
        "carried_rows_seen": int(0 if state is None else state.get("rows_seen", 0)),
        "sketch_s": sketch_s.tolist(),
        "sketch_s2": sketch_s2.tolist(),
        "sketch_frob_sq": sketch_frob_sq,
        "principal_cos_V_vs_Vexact": principal_cos.tolist(),
        "cos2_V_oracle_v1": cos2_V_v1.tolist(),
        "cos2_V_oracle_v2": cos2_V_v2.tolist(),
        "raw_sk_breakdown_v1": raw_sk_breakdown_v1.tolist(),
        "raw_sk_breakdown_v2": raw_sk_breakdown_v2.tolist(),
        "raw_sk_predicted_v1_full": float(raw_sk_breakdown_v1.sum()),
        "raw_sk_predicted_v2_full": float(raw_sk_breakdown_v2.sum()),
        "sketch_dim": int(B_sketch.shape[1]),
        "gain1_dim": int(B_gain1.shape[1]),
        "gain2_dim": int(B_gain2.shape[1]),
        "union_dim": int(B_union.shape[1]),
        "sketch_excl_dim": int(B_sketch_excl.shape[1]),
        "gain1_excl_dim": int(B_gain1_excl.shape[1]),
        "gain2_excl_dim": int(B_gain2_excl.shape[1]),
        "oracle_proj_norm": float(oracle_proj_norm),
        "oracle_proj_norm_sq_in_union": float(oracle_proj_norm ** 2),
        "oracle_v1_mass_S+G1": float(oracle_v1_proj_S_G1_norm ** 2),
        "oracle_v1_mass_S+G1+G2": float(oracle_v1_proj_union_norm ** 2),
        "oracle_v2_mass_S+G1": float(oracle_v2_proj_S_G1_norm ** 2),
        "oracle_v2_mass_S+G1+G2": float(oracle_proj_norm ** 2),
        "sketch_gain1_dim": int(B_sketch_gain1.shape[1]),
        "denoms": denoms,
        "weights": list(weights),
    }
    return info, rows, V_default, A_cur, M_gain, mid0, end0


def run(args, matrix):
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
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
    A = np.asarray(A, dtype=np.float64)
    V_exact = np.asarray(V_exact, dtype=np.float64)
    rank = int(args.rank)

    blocks_to_report = sorted(set(int(b) for b in args.blocks))
    max_block = max(blocks_to_report)
    state = None
    old_row_memory = None
    out_per_block = {}

    for block_id in range(1, max_block + 1):
        info, rows, V_default, A_cur, M_gain, mid0, end0 = analyze_block(
            args, A, V_exact, state, old_row_memory, work_dtype, block_id, matrix
        )
        if block_id in blocks_to_report:
            out_per_block[block_id] = (info, rows)
        # Advance carry state for next block.
        state, old_row_memory = update_carry(
            args, A, work_dtype, state, V_default, M_gain, A_cur, mid0, rank, A.shape[0], end0
        )
    return out_per_block


def write_text(path, info, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"HM-triplet block-{info['block_id']} decomposition\n")
        f.write("=" * (32 + len(str(info['block_id']))) + "\n\n")
        for k in [
            "matrix", "block_id", "half_win", "rank", "rows_seen", "carried_rows_seen",
            "sketch_dim", "gain1_dim", "gain2_dim", "union_dim", "sketch_gain1_dim",
            "sketch_excl_dim", "gain1_excl_dim", "gain2_excl_dim",
            "sketch_s", "sketch_s2", "sketch_frob_sq",
            "principal_cos_V_vs_Vexact",
            "cos2_V_oracle_v1", "cos2_V_oracle_v2",
            "raw_sk_breakdown_v1", "raw_sk_breakdown_v2",
            "raw_sk_predicted_v1_full", "raw_sk_predicted_v2_full",
            "oracle_proj_norm", "oracle_proj_norm_sq_in_union",
            "oracle_v1_mass_S+G1", "oracle_v1_mass_S+G1+G2",
            "oracle_v2_mass_S+G1", "oracle_v2_mass_S+G1+G2",
            "denoms", "weights",
        ]:
            f.write(f"  {k}: {info[k]}\n")
        f.write("\n")
        f.write(
            "Sketch evidence: raw_sk(v) = sum_i s_i^2 * cos^2(V_i, v) where "
            "V is state['V'] (rank-r carried right-singular directions) and "
            "s_i are the carried singular values. The s_i grow with accumulated "
            "evidence (~ sqrt(rows_seen) for true-aligned components); cos^2(V_i, "
            "oracle_j) tracks alignment between the carried subspace and the "
            "true singular vectors.\n\n"
        )
        f.write(
            "Per-candidate decomposition. Shares are squared lengths after "
            "normalising the candidate vector. _excl rows are the orthogonalised "
            "(Gram-Schmidt) shares: sketch first, then gain1\\sketch, then "
            "gain2\\(sketch+gain1); they sum to the in-union share.\n\n"
        )
        for rec in rows:
            f.write(f"-- {rec['label']} --\n")
            f.write(
                "  scores: hm_triplet={hm:.6e}  combined={cb:.6e}  relH1={rh:.4f}\n".format(
                    hm=rec["hm_triplet_score"], cb=rec["combined_score"], rh=rec["hm_relH1"]
                )
            )
            f.write(
                "  hm_factor_split: gain_factor={gf:.6e}  entropy_factor(relH1)={ef:.6f}  product={pr:.6e}\n".format(
                    gf=rec["hm_gain_factor"], ef=rec["hm_entropy_factor"],
                    pr=rec["hm_gain_factor"] * rec["hm_entropy_factor"],
                )
            )
            f.write(
                "  hm_raw_split (NO pool norm): gain_factor={gf:.6e}  entropy={ef:.6f}  product={pr:.6e}\n".format(
                    gf=rec["hm_raw_gain_factor"], ef=rec["hm_entropy_factor"],
                    pr=rec["hm_raw_score"],
                )
            )
            f.write(
                "  raw_energies (||A_k v||^2): sketch={s:.6e}  gain1={g1:.6e}  gain2={g2:.6e}\n".format(
                    s=rec["raw_sketch"], g1=rec["raw_gain1"], g2=rec["raw_gain2"]
                )
            )
            comps = rec.get("raw_sk_components_per_dir", [])
            cos2 = rec.get("cos2_V_v_per_dir", [])
            if comps:
                f.write(
                    "  raw_sk components (s_i^2 * cos^2(V_i,v)): "
                    + " ".join(f"i={i}: {c:.4e}" for i, c in enumerate(comps))
                    + "  cos^2(V_i,v): "
                    + " ".join(f"i={i}: {c:.4f}" for i, c in enumerate(cos2))
                    + f"  sum={sum(comps):.4e}\n"
                )
            f.write(
                "  pool_shares (raw / pool_max): sketch={s:.4e}  gain1={g1:.4e}  gain2={g2:.4e}\n".format(
                    s=rec["share_sketch_pool"], g1=rec["share_gain1_pool"], g2=rec["share_gain2_pool"]
                )
            )
            f.write(
                "  combined_factor_split: gain2(M_gain)={gf:.6e}  phi={ph:.6e}  product={pr:.6e}  c={c:+.4f}\n".format(
                    gf=rec["combined_gain_factor"], ph=rec["combined_phi_factor"],
                    pr=rec["combined_gain_factor"] * rec["combined_phi_factor"],
                    c=rec["combined_c"],
                )
            )
            f.write(
                "  combined_pooled: relH={rh:.4f}  H={h:.4f}  y2_sq={y2:.4e}  y4_4={y4:.4e}\n".format(
                    rh=rec["combined_pooled_relH"], h=rec["combined_pooled_H"],
                    y2=rec["combined_pooled_y2_sq"], y4=rec["combined_pooled_y4_4"],
                )
            )
            f.write(
                "  hm_shares: sketch={s:.4f}  gain1={g1:.4f}  gain2={g2:.4f}\n".format(
                    s=rec["hm_sketch_share"], g1=rec["hm_gain1_share"], g2=rec["hm_gain2_share"]
                )
            )
            f.write(
                "  axis_decomp: in_oracle_proj={oin:.4f}  outside={oout:.4f}  cos_v1={c1:+.4f}\n".format(
                    oin=rec["mass_in_oracle_proj"],
                    oout=rec["mass_outside_oracle_proj"],
                    c1=rec["cos_v1"],
                )
            )
            f.write(
                "  v_subspace_share: sketch={s:.4f}  gain1={g1:.4f}  gain2={g2:.4f}  union={u:.4f}\n".format(
                    s=rec["share_sketch"], g1=rec["share_gain1"], g2=rec["share_gain2"], u=rec["share_union"]
                )
            )
            f.write(
                "  v_subspace_share_excl: sketch={s:.4f}  gain1\\S={g1:.4f}  gain2\\(S+G1)={g2:.4f}\n".format(
                    s=rec["share_sketch_excl"], g1=rec["share_gain1_excl"], g2=rec["share_gain2_excl"]
                )
            )
            f.write(
                "  outside_subspace_share: sketch={s:.4f}  gain1={g1:.4f}  gain2={g2:.4f}  union={u:.4f}\n".format(
                    s=rec["outside_share_sketch"], g1=rec["outside_share_gain1"],
                    g2=rec["outside_share_gain2"], u=rec["outside_share_union"]
                )
            )
            f.write(
                "  outside_subspace_share_excl: sketch={s:.4f}  gain1\\S={g1:.4f}  gain2\\(S+G1)={g2:.4f}\n".format(
                    s=rec["outside_share_sketch_excl"], g1=rec["outside_share_gain1_excl"],
                    g2=rec["outside_share_gain2_excl"]
                )
            )
            top_v = sorted(rec["sing_cos2_v"].items(), key=lambda kv: -kv[1])[:5]
            top_o = sorted(rec["sing_cos2_outside"].items(), key=lambda kv: -kv[1])[:5]
            f.write("  v_singular_align (top 5 cos2 vs V_exact[:, j]): "
                    + ", ".join(f"j={j}:{c:.4f}" for j, c in top_v) + "\n")
            f.write("  outside_singular_align (top 5 cos2 vs V_exact[:, j]): "
                    + ", ".join(f"j={j}:{c:.4f}" for j, c in top_o) + "\n")
            f.write("\n")

        # Per-component sketch/gain1/gain2 table (raw energies).
        f.write("Sketch / gain1 / gain2 raw energies and shares\n")
        f.write("----------------------------------------------\n")
        f.write(
            f"  denoms (HM-triplet pool): sketch={info['denoms']['sketch']:.4e}  "
            f"gain1={info['denoms']['gain1']:.4e}  gain2={info['denoms']['gain2']:.4e}\n\n"
        )
        f.write(
            f"{'label':<28} {'raw_sk':>12} {'raw_g1':>12} {'raw_g2':>12}  "
            f"{'sh_sk':>9} {'sh_g1':>9} {'sh_g2':>10}  {'min_share':>10}\n"
        )
        for rec in rows:
            shares = [rec["share_sketch_pool"], rec["share_gain1_pool"], rec["share_gain2_pool"]]
            finite = [s for s in shares if np.isfinite(s)]
            min_share = min(finite) if finite else np.nan
            f.write(
                "{lbl:<28} {rs:>12.4e} {r1:>12.4e} {r2:>12.4e}  "
                "{ss:>9.4f} {s1:>9.4f} {s2:>10.4f}  {ms:>10.4e}\n".format(
                    lbl=rec["label"][:28],
                    rs=rec["raw_sketch"], r1=rec["raw_gain1"], r2=rec["raw_gain2"],
                    ss=rec["share_sketch_pool"], s1=rec["share_gain1_pool"], s2=rec["share_gain2_pool"],
                    ms=min_share,
                )
            )
        f.write("\n")

        # Side-by-side factor table for entropy vs gain.
        f.write("Entropy-vs-gain factor table (per candidate)\n")
        f.write("--------------------------------------------\n")
        header = (
            f"{'label':<28} {'hm_score':>12} {'hm_gain':>12} {'hm_entropy':>11}  "
            f"{'comb_score':>12} {'comb_gain2':>12} {'comb_phi':>11} {'comb_relH':>10}\n"
        )
        f.write(header)
        for rec in rows:
            f.write(
                "{lbl:<28} {hms:>12.4e} {hmg:>12.4e} {hme:>11.4f}  "
                "{cbs:>12.4e} {cbg:>12.4e} {cbp:>11.4f} {cbr:>10.4f}\n".format(
                    lbl=rec["label"][:28],
                    hms=rec["hm_triplet_score"],
                    hmg=rec["hm_gain_factor"],
                    hme=rec["hm_entropy_factor"],
                    cbs=rec["combined_score"],
                    cbg=rec["combined_gain_factor"],
                    cbp=rec["combined_phi_factor"],
                    cbr=rec["combined_pooled_relH"],
                )
            )
        f.write("\n")

        # Unnormalised HM (no pool denominators).
        f.write("Unnormalised HM (no pool norm) score table\n")
        f.write("------------------------------------------\n")
        f.write(
            f"{'label':<28} {'hm_raw_score':>14} {'hm_raw_gain':>14} {'hm_entropy':>11}  "
            f"{'min_raw':>12} {'argmin':>8}\n"
        )
        for rec in rows:
            raws = [rec["raw_sketch"], rec["raw_gain1"], rec["raw_gain2"]]
            names = ["sk", "g1", "g2"]
            finite_pairs = [(n, r) for n, r in zip(names, raws) if np.isfinite(r)]
            if finite_pairs:
                argmin_name, min_raw = min(finite_pairs, key=lambda x: x[1])
            else:
                argmin_name, min_raw = "", float("nan")
            f.write(
                "{lbl:<28} {hms:>14.4e} {hmg:>14.4e} {hme:>11.4f}  "
                "{mr:>12.4e} {ar:>8}\n".format(
                    lbl=rec["label"][:28],
                    hms=rec["hm_raw_score"],
                    hmg=rec["hm_raw_gain_factor"],
                    hme=rec["hm_entropy_factor"],
                    mr=min_raw, ar=argmin_name,
                )
            )


def write_csv(path, info, rows):
    fields = [
        "matrix", "label",
        "hm_triplet_score", "combined_score", "hm_relH1",
        "hm_sketch_share", "hm_gain1_share", "hm_gain2_share",
        "cos_v1", "mass_in_oracle_proj", "mass_outside_oracle_proj",
        "share_sketch", "share_gain1", "share_gain2", "share_union",
        "share_sketch_excl", "share_gain1_excl", "share_gain2_excl",
        "outside_share_sketch", "outside_share_gain1", "outside_share_gain2", "outside_share_union",
        "outside_share_sketch_excl", "outside_share_gain1_excl", "outside_share_gain2_excl",
    ]
    n_sing = max((max(rec["sing_cos2_v"].keys(), default=-1) for rec in rows), default=-1) + 1
    for j in range(n_sing):
        fields.append(f"v_cos2_V{j}")
    for j in range(n_sing):
        fields.append(f"outside_cos2_V{j}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in rows:
            row = {k: rec.get(k, "") for k in fields if not k.startswith("v_cos2_V") and not k.startswith("outside_cos2_V")}
            row["matrix"] = info["matrix"]
            for j in range(n_sing):
                row[f"v_cos2_V{j}"] = rec["sing_cos2_v"].get(j, "")
                row[f"outside_cos2_V{j}"] = rec["sing_cos2_outside"].get(j, "")
            writer.writerow(row)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="residual-spiky-shocks")
    p.add_argument("--out-prefix", default="summary/hmean_triplet_block2_decomposition")
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
    p.add_argument("--blocks", nargs="+", type=int, default=[2])
    return p.parse_args()


def main():
    args = parse_args()
    out_per_block = run(args, args.matrix)
    json_payload = {"matrix": args.matrix, "blocks": {}}
    paths = []
    for block_id in sorted(out_per_block.keys()):
        info, rows = out_per_block[block_id]
        suffix = f"_block{block_id}"
        csv_path = args.out_prefix + suffix + ".csv"
        json_path_block = args.out_prefix + suffix + ".json"
        txt_path = args.out_prefix + suffix + ".txt"
        write_csv(csv_path, info, rows)
        serial_rows = []
        for rec in rows:
            out = dict(rec)
            out["sing_cos2_v"] = {str(k): v for k, v in rec["sing_cos2_v"].items()}
            out["sing_cos2_outside"] = {str(k): v for k, v in rec["sing_cos2_outside"].items()}
            serial_rows.append(out)
        with open(json_path_block, "w", encoding="utf-8") as f:
            json.dump({"info": info, "rows": serial_rows}, f, indent=2, sort_keys=True, default=float)
        write_text(txt_path, info, rows)
        json_payload["blocks"][str(block_id)] = {"info": info, "rows": serial_rows}
        paths.extend([csv_path, json_path_block, txt_path])
    combined_json_path = args.out_prefix + "_combined.json"
    with open(combined_json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, sort_keys=True, default=float)
    print("wrote " + " ".join(paths + [combined_json_path]))


if __name__ == "__main__":
    main()
