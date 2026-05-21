#!/usr/bin/env python3
"""Tail-conspiracy/sketch-bias diagnostics for the combined algorithm.

This is a Python counterpart to the MATLAB `cex_structured_combined_hybrid.m`
tail-conspiracy plots.  It emits machine-readable JSONL/CSV plus a short text
interpretation that is easy to inspect in this workspace.

The diagnostic tracks both optimizer-selected slots:

  opt1 = first direction selected by the combined optimizer in a block
  opt2 = second direction selected by the combined optimizer in a block

For each block it records:

  - opt1/opt2 cosine similarity to oracle1/oracle2;
  - opt1/opt2 tail mass;
  - per-block persistence of opt1/opt2 tail components;
  - previous-slot score ratios under the current objective;
  - actual_B, zero_B, and oracle_B score decomposition for both slots;
  - opt-vs-oracle score margins.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

import cex_restricted_space_probe as probe


def as_float(x):
    if x is None:
        return float("nan")
    return float(np.asarray(x, dtype=float))


def safe_ratio(num, den):
    den = float(den)
    if abs(den) <= 1e-30:
        return float("nan")
    return float(num) / den


def orth_against(v, Q):
    out = np.asarray(v, dtype=np.float64).reshape(-1)
    if Q is not None and np.asarray(Q).size:
        Qq = probe.orthonormalize_columns(np.asarray(Q, dtype=np.float64), dtype=np.float64)
        out = out - Qq @ (Qq.T @ out)
    nrm = float(np.linalg.norm(out))
    if nrm <= 1e-30:
        return None
    return np.ascontiguousarray(out / nrm)


def tail_component(v, Q_exact):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    out = v - Q_exact @ (Q_exact.T @ v)
    nrm = float(np.linalg.norm(out))
    if nrm <= 1e-30:
        return None
    return np.ascontiguousarray(out / nrm)


def tail_mass(v, Q_exact):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(v))
    if nrm <= 1e-30:
        return float("nan")
    v = v / nrm
    signal = float(np.linalg.norm(Q_exact @ (Q_exact.T @ v)) ** 2)
    return max(0.0, 1.0 - signal)


def abs_cos(a, b):
    if a is None or b is None:
        return float("nan")
    return abs(float(np.asarray(a).reshape(-1) @ np.asarray(b).reshape(-1)))


def make_state(M_gain, V_selected, rows_seen, A_block, rows_ref, state_prev, old_row_memory):
    scores = []
    H_vals = []
    for j in range(V_selected.shape[1]):
        score, _, H = probe.score_full_vector_details_forget(
            M_gain,
            A_block,
            V_selected[:, j],
            rows_ref,
            state_prev=state_prev,
            score_variant="combined",
            old_row_memory=old_row_memory,
        )
        scores.append(score)
        H_vals.append(H)
    _, s_new, Vt_new, _ = probe.left_projected_operator_svd_factors(V_selected.T, M_gain)
    V_carried = np.ascontiguousarray(Vt_new.T[:, : V_selected.shape[1]], dtype=np.float64)
    s_new = np.asarray(s_new[: V_selected.shape[1]], dtype=np.float64)
    state = {
        "V": np.ascontiguousarray(V_carried.astype(np.float32)),
        "s": np.asarray(s_new, dtype=np.float32),
        "s2": np.asarray(s_new, dtype=np.float32) ** 2,
        "H": np.asarray(H_vals, dtype=np.float32),
        "score": np.asarray(scores, dtype=np.float32),
        "rows_seen": int(rows_seen),
    }
    return state, V_carried, s_new


def frame_score_sum(V2, M_score, A_block, rows_ref, state_prev, old_row_memory):
    comps = [
        score_candidate(V2[:, j], M_score, A_block, rows_ref, state_prev, old_row_memory)
        for j in range(V2.shape[1])
    ]
    return float(sum(c["score"] for c in comps)), comps


def best_random_internal_frame(
    V_internal,
    M_score,
    A_block,
    rows_ref,
    state_prev,
    old_row_memory,
    rng,
    starts,
    seeds=(),
):
    Q_internal = probe.orthonormalize_columns(V_internal, dtype=np.float64)
    q = Q_internal.shape[1]
    best_V = None
    best_score = -np.inf
    for seed in seeds:
        if seed is None or not np.asarray(seed).size:
            continue
        seed_arr = np.asarray(seed, dtype=np.float64)
        if seed_arr.shape[0] != Q_internal.shape[0] or seed_arr.shape[1] < 2:
            continue
        Z = probe.orthonormalize_columns(Q_internal.T @ seed_arr[:, :2], dtype=np.float64)
        if Z.shape[1] < 2:
            continue
        V2 = np.ascontiguousarray(Q_internal @ Z[:, :2], dtype=np.float64)
        score, _ = frame_score_sum(V2, M_score, A_block, rows_ref, state_prev, old_row_memory)
        if score > best_score:
            best_score = score
            best_V = V2
    for _ in range(max(0, int(starts))):
        Z = probe.orthonormalize_columns(rng.standard_normal((q, 2)), dtype=np.float64)
        if Z.shape[1] < 2:
            continue
        V2 = np.ascontiguousarray(Q_internal @ Z[:, :2], dtype=np.float64)
        score, _ = frame_score_sum(V2, M_score, A_block, rows_ref, state_prev, old_row_memory)
        if score > best_score:
            best_score = score
            best_V = V2
    if best_V is None:
        best_V = Q_internal[:, :2]
    return best_V


def rank2_variant_frame(
    variant,
    M_gain,
    V_internal,
    V_carried,
    Q_oracle,
    A_block,
    rows_ref,
    state_prev,
    old_row_memory,
    rng,
    random_starts,
):
    final_rank = 2
    M_zero = np.asarray(A_block, dtype=np.float64)
    if variant == "left_projected":
        V2 = np.asarray(V_carried[:, :final_rank], dtype=np.float64)
    elif variant == "internal_projected_svd":
        V_proj, _ = probe.projected_subspace_svd(np.asarray(M_gain, dtype=np.float64), V_internal)
        V2 = np.asarray(V_proj[:, :final_rank], dtype=np.float64)
    elif variant == "internal_current_svd":
        V_proj, _ = probe.projected_subspace_svd(M_zero, V_internal)
        V2 = np.asarray(V_proj[:, :final_rank], dtype=np.float64)
    elif variant == "internal_first2":
        V2 = np.asarray(V_internal[:, :final_rank], dtype=np.float64)
    elif variant == "internal_score_top2":
        scores = []
        for j in range(V_internal.shape[1]):
            comp = score_candidate(V_internal[:, j], M_gain, A_block, rows_ref, state_prev, old_row_memory)
            scores.append(comp["score"])
        order = np.argsort(np.asarray(scores))[::-1][:final_rank]
        V2 = np.asarray(V_internal[:, order], dtype=np.float64)
    elif variant == "internal_zeroB_score_top2":
        scores = []
        for j in range(V_internal.shape[1]):
            comp = score_candidate(V_internal[:, j], M_zero, A_block, rows_ref, state_prev, old_row_memory)
            scores.append(comp["score"])
        order = np.argsort(np.asarray(scores))[::-1][:final_rank]
        V2 = np.asarray(V_internal[:, order], dtype=np.float64)
    elif variant == "internal_actualB_score_opt":
        V_proj, _ = probe.projected_subspace_svd(np.asarray(M_gain, dtype=np.float64), V_internal)
        V2 = best_random_internal_frame(
            V_internal,
            np.asarray(M_gain, dtype=np.float64),
            A_block,
            rows_ref,
            state_prev,
            old_row_memory,
            rng,
            random_starts,
            seeds=(V_carried[:, :final_rank], V_internal[:, :final_rank], V_proj[:, :final_rank]),
        )
    elif variant == "internal_zeroB_score_opt":
        V_proj, _ = probe.projected_subspace_svd(M_zero, V_internal)
        V2 = best_random_internal_frame(
            V_internal,
            M_zero,
            A_block,
            rows_ref,
            state_prev,
            old_row_memory,
            rng,
            random_starts,
            seeds=(V_internal[:, :final_rank], V_proj[:, :final_rank]),
        )
    elif variant == "oracle_projected":
        V2 = np.asarray(Q_oracle[:, :final_rank], dtype=np.float64)
    elif variant == "opt1_oracle2":
        oracle2_vs_opt1 = orth_against(Q_oracle[:, 1], V_internal[:, :1])
        if oracle2_vs_opt1 is None:
            oracle2_vs_opt1 = Q_oracle[:, 1]
        V2 = np.column_stack([V_internal[:, 0], oracle2_vs_opt1])
    else:
        raise ValueError(f"Unknown compression variant: {variant}")
    return probe.orthonormalize_columns(np.ascontiguousarray(V2, dtype=np.float64), dtype=np.float64)


def compression_variant_record(
    matrix,
    internal_rank,
    block,
    rows,
    variant,
    V2,
    V_internal,
    Q_oracle,
    M_gain,
    A_block,
    rows_ref,
    state_prev,
    old_row_memory,
):
    oracle1 = np.ascontiguousarray(Q_oracle[:, 0], dtype=np.float64)
    oracle2 = np.ascontiguousarray(Q_oracle[:, 1], dtype=np.float64)
    variant_cos = probe.subspace_principal_cosines(V2, Q_oracle)
    Q2 = probe.orthonormalize_columns(V2, dtype=np.float64)
    Q_internal = probe.orthonormalize_columns(V_internal, dtype=np.float64)
    col_scores = [
        score_candidate(V2[:, j], M_gain, A_block, rows_ref, state_prev, old_row_memory)
        for j in range(V2.shape[1])
    ]
    score_sum = float(sum(c["score"] for c in col_scores))
    gain2_sum = float(sum(c["gain2"] for c in col_scores))
    phi_mean = float(np.mean([c["phi"] for c in col_scores]))
    return {
        "matrix": matrix,
        "internal_rank": int(internal_rank),
        "final_rank": 2,
        "block": int(block),
        "row_start": int(rows[0]),
        "row_end": int(rows[1]),
        "variant": variant,
        "variant_subspace_cos1": float(variant_cos[0]),
        "variant_subspace_cos2": float(variant_cos[1]),
        "variant_oracle1_projection": float(np.linalg.norm(Q2.T @ oracle1)),
        "variant_oracle2_projection": float(np.linalg.norm(Q2.T @ oracle2)),
        "variant_col1_vs_oracle1": abs_cos(V2[:, 0], oracle1),
        "variant_col2_vs_oracle2": abs_cos(V2[:, 1], oracle2),
        "variant_actualB_score_sum": score_sum,
        "variant_actualB_gain2_sum": gain2_sum,
        "variant_actualB_phi_mean": phi_mean,
        "internal_oracle1_projection": float(np.linalg.norm(Q_internal.T @ oracle1)),
        "internal_oracle2_projection": float(np.linalg.norm(Q_internal.T @ oracle2)),
    }


def score_candidate(v, M_gain, A_block, rows_ref, state_prev, old_row_memory):
    comp = probe.combined_score_component_details(
        M_gain,
        A_block,
        v,
        rows_ref,
        state_prev=state_prev,
        old_row_memory=old_row_memory,
    )
    return {
        "score": float(comp["score_total"]),
        "gain2": float(comp["gain2"]),
        "phi": float(comp["phi"]),
        "relH": float(comp["pooled_rel_H"]),
    }


def prefix_record(prefix, comp):
    return {
        f"{prefix}_score": comp["score"],
        f"{prefix}_gain2": comp["gain2"],
        f"{prefix}_phi": comp["phi"],
        f"{prefix}_relH": comp["relH"],
    }


def score_contexts(v, label, contexts, A_block, rows_ref, old_row_memory):
    out = {}
    for ctx_name, (M_ctx, state_ctx) in contexts.items():
        comp = score_candidate(v, M_ctx, A_block, rows_ref, state_ctx, old_row_memory)
        out.update(prefix_record(f"{label}_{ctx_name}", comp))
    return out


def build_record(
    matrix,
    method,
    block,
    rows,
    V_selected,
    V_carried,
    Q_oracle,
    Q_exact,
    contexts,
    A_block,
    rows_ref,
    old_row_memory,
    prev_selected,
):
    opt1 = np.ascontiguousarray(V_selected[:, 0], dtype=np.float64)
    opt2 = np.ascontiguousarray(V_selected[:, 1], dtype=np.float64)
    carried1 = np.ascontiguousarray(V_carried[:, 0], dtype=np.float64)
    carried2 = np.ascontiguousarray(V_carried[:, 1], dtype=np.float64)
    oracle1 = np.ascontiguousarray(Q_oracle[:, 0], dtype=np.float64)
    oracle2 = np.ascontiguousarray(Q_oracle[:, 1], dtype=np.float64)
    oracle2_vs_opt1 = orth_against(oracle2, opt1[:, None])
    if oracle2_vs_opt1 is None:
        oracle2_vs_opt1 = oracle2

    subspace_cos = probe.subspace_principal_cosines(V_selected, Q_oracle)
    carried_cos = probe.subspace_principal_cosines(V_carried, Q_oracle)

    rec = {
        "matrix": matrix,
        "method": method,
        "block": int(block),
        "row_start": int(rows[0]),
        "row_end": int(rows[1]),
        "subspace_cos1": float(subspace_cos[0]),
        "subspace_cos2": float(subspace_cos[1]),
        "carried_subspace_cos1": float(carried_cos[0]),
        "carried_subspace_cos2": float(carried_cos[1]),
        "opt1_vs_oracle1": abs_cos(opt1, oracle1),
        "opt1_vs_oracle2": abs_cos(opt1, oracle2),
        "opt2_vs_oracle1": abs_cos(opt2, oracle1),
        "opt2_vs_oracle2": abs_cos(opt2, oracle2),
        "carried1_vs_oracle1": abs_cos(carried1, oracle1),
        "carried1_vs_oracle2": abs_cos(carried1, oracle2),
        "carried2_vs_oracle1": abs_cos(carried2, oracle1),
        "carried2_vs_oracle2": abs_cos(carried2, oracle2),
        "opt1_tail_mass": tail_mass(opt1, Q_exact),
        "opt2_tail_mass": tail_mass(opt2, Q_exact),
        "carried1_tail_mass": tail_mass(carried1, Q_exact),
        "carried2_tail_mass": tail_mass(carried2, Q_exact),
    }

    if prev_selected is None:
        rec.update({
            "prev_opt1_dot_opt1": float("nan"),
            "prev_opt2_dot_opt2": float("nan"),
            "prev_opt1_tail_cos": float("nan"),
            "prev_opt2_tail_cos": float("nan"),
            "prev_opt1_score_ratio": float("nan"),
            "prev_opt2_score_ratio": float("nan"),
        })
    else:
        prev_opt1 = np.asarray(prev_selected[:, 0], dtype=np.float64)
        prev_opt2 = np.asarray(prev_selected[:, 1], dtype=np.float64)
        prev_opt1_tail = tail_component(prev_opt1, Q_exact)
        prev_opt2_tail = tail_component(prev_opt2, Q_exact)
        opt1_tail = tail_component(opt1, Q_exact)
        opt2_tail = tail_component(opt2, Q_exact)
        actual_M, actual_state = contexts["actualB"]
        prev1_comp = score_candidate(prev_opt1, actual_M, A_block, rows_ref, actual_state, old_row_memory)
        prev2_comp = score_candidate(prev_opt2, actual_M, A_block, rows_ref, actual_state, old_row_memory)
        opt1_comp = score_candidate(opt1, actual_M, A_block, rows_ref, actual_state, old_row_memory)
        opt2_comp = score_candidate(opt2, actual_M, A_block, rows_ref, actual_state, old_row_memory)
        rec.update({
            "prev_opt1_dot_opt1": abs_cos(prev_opt1, opt1),
            "prev_opt2_dot_opt2": abs_cos(prev_opt2, opt2),
            "prev_opt1_tail_cos": abs_cos(prev_opt1_tail, opt1_tail),
            "prev_opt2_tail_cos": abs_cos(prev_opt2_tail, opt2_tail),
            "prev_opt1_score_ratio": safe_ratio(prev1_comp["score"], opt1_comp["score"]),
            "prev_opt2_score_ratio": safe_ratio(prev2_comp["score"], opt2_comp["score"]),
        })

    for label, vec in [
        ("opt1", opt1),
        ("opt2", opt2),
        ("oracle1", oracle1),
        ("oracle2", oracle2),
        ("oracle2_vs_opt1", oracle2_vs_opt1),
    ]:
        rec.update(score_contexts(vec, label, contexts, A_block, rows_ref, old_row_memory))

    rec["opt1_actual_margin_vs_oracle1"] = rec["opt1_actualB_score"] - rec["oracle1_actualB_score"]
    rec["opt2_actual_margin_vs_oracle2"] = rec["opt2_actualB_score"] - rec["oracle2_actualB_score"]
    rec["opt2_actual_margin_vs_oracle2_vs_opt1"] = (
        rec["opt2_actualB_score"] - rec["oracle2_vs_opt1_actualB_score"]
    )
    rec["opt2_zero_margin_vs_oracle2_vs_opt1"] = (
        rec["opt2_zeroB_score"] - rec["oracle2_vs_opt1_zeroB_score"]
    )
    rec["opt1_sketch_score_share"] = safe_ratio(
        rec["opt1_actualB_score"] - rec["opt1_zeroB_score"],
        rec["opt1_actualB_score"],
    )
    rec["opt2_sketch_score_share"] = safe_ratio(
        rec["opt2_actualB_score"] - rec["opt2_zeroB_score"],
        rec["opt2_actualB_score"],
    )
    rec["opt1_actual_over_zero"] = safe_ratio(rec["opt1_actualB_score"], rec["opt1_zeroB_score"])
    rec["opt2_actual_over_zero"] = safe_ratio(rec["opt2_actualB_score"], rec["opt2_zeroB_score"])
    return rec


def run_matrix(matrix, method, args):
    np.random.seed(args.seed)
    A, V_exact, _, sigma1 = probe.generate_matrix_input(
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
    n = A.shape[0]
    rank = int(args.rank)
    if rank != 2:
        raise ValueError("This diagnostic currently expects --rank 2.")
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    state = None
    oracle_state = None
    old_row_memory = None
    prev_selected = None
    V_carried = None
    records = []
    if method == "combined":
        combined_rank = rank
    elif method == "hybrid":
        combined_rank = max(0, min(int(args.hybrid_combined_rank), rank))
    else:
        raise ValueError(f"Unknown method: {method}")

    for block, start0 in enumerate(range(0, n, args.win), start=1):
        end0 = min(start0 + args.win, n)
        A_block = np.ascontiguousarray(A[start0:end0, :], dtype=work_dtype)
        rows_seen = end0 if state is None else int(state["rows_seen"]) + A_block.shape[0]

        if state is None:
            M_actual = A_block
        else:
            B_actual = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
            M_actual = np.ascontiguousarray(np.vstack([B_actual, A_block]), dtype=work_dtype)
        M_zero = A_block
        if oracle_state is None:
            M_oracle = A_block
            oracle_rows_seen = A_block.shape[0]
        else:
            B_oracle = oracle_state["s"].astype(work_dtype)[:, None] * oracle_state["V"].astype(work_dtype).T
            M_oracle = np.ascontiguousarray(np.vstack([B_oracle, A_block]), dtype=work_dtype)
            oracle_rows_seen = int(oracle_state["rows_seen"]) + A_block.shape[0]

        V_init = probe.row_norm_seed(A_block, rank)
        V_score, _, _, _, _ = probe.entropy_iter_basis_forget(
            M_gain=M_actual,
            active_r=rank,
            rows_ref=n,
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
            A_block=A_block,
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
            combined_rank=combined_rank,
        )
        V_selected = np.ascontiguousarray(np.asarray(V_score[:, :rank], dtype=np.float64))
        Q_oracle, _ = probe.projected_true_span_oracle(M_actual, V_exact[:, :rank], rank, dtype=np.float64)
        Q_exact = probe.orthonormalize_columns(V_exact[:, :rank], dtype=np.float64)

        contexts = {
            "actualB": (M_actual, state),
            "zeroB": (M_zero, state),
            "oracleB": (M_oracle, oracle_state if oracle_state is not None else state),
        }
        state, V_carried, _ = make_state(
            M_actual, V_selected, rows_seen, A_block, n, state, old_row_memory
        )
        record = build_record(
            matrix,
            method,
            block,
            (start0 + 1, end0),
            V_selected,
            V_carried,
            Q_oracle,
            Q_exact,
            contexts,
            A_block,
            n,
            old_row_memory,
            prev_selected,
        )
        records.append(record)

        # Oracle-forced state for oracleB scoring in the next block.
        Q_oracle_state, _ = probe.projected_true_span_oracle(M_oracle, V_exact[:, :rank], rank, dtype=np.float64)
        oracle_state, _, _ = make_state(
            M_oracle,
            np.ascontiguousarray(Q_oracle_state[:, :rank], dtype=np.float64),
            oracle_rows_seen,
            A_block,
            n,
            oracle_state,
            old_row_memory,
        )

        old_row_memory = probe.select_old_row_memory(
            A[:end0, :].astype(work_dtype, copy=False),
            V_carried.astype(work_dtype, copy=False),
            args.old_memory_size,
            np.random.default_rng(args.seed + end0),
            return_indices=False,
        )
        prev_selected = V_selected

    final_align = float(np.linalg.norm((V_carried @ V_carried.T) @ V_exact[:, :1], ord="fro"))
    return {
        "matrix": matrix,
        "method": method,
        "combined_rank": int(combined_rank),
        "sigma1": float(sigma1),
        "final_mean_align": final_align,
        "records": records,
    }


def run_internal_rank_matrix(matrix, internal_rank, args):
    np.random.seed(args.seed)
    A, V_exact, _, sigma1 = probe.generate_matrix_input(
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
    n = A.shape[0]
    final_rank = int(args.rank)
    if final_rank != 2:
        raise ValueError("Internal-rank diagnostic currently expects final --rank 2.")
    internal_rank = max(final_rank, int(internal_rank))
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    state = None
    old_row_memory = None
    V_carried = None
    records = []
    variant_records = []
    compression_variants = list(args.compression_variants)

    for block, start0 in enumerate(range(0, n, args.win), start=1):
        end0 = min(start0 + args.win, n)
        A_block = np.ascontiguousarray(A[start0:end0, :], dtype=work_dtype)
        rows_seen = end0 if state is None else int(state["rows_seen"]) + A_block.shape[0]
        if state is None:
            M_actual = A_block
        else:
            B_actual = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
            M_actual = np.ascontiguousarray(np.vstack([B_actual, A_block]), dtype=work_dtype)

        V_init = probe.row_norm_seed(A_block, internal_rank)
        V_score, _, _, _, _ = probe.entropy_iter_basis_forget(
            M_gain=M_actual,
            active_r=internal_rank,
            rows_ref=n,
            V_init=np.asarray(V_init, dtype=work_dtype),
            q0=max(args.q0, internal_rank),
            qmax=max(args.qmax, internal_rank),
            krylov_depth=args.krylov_depth,
            residual_tol=args.residual_tol,
            expansion_maxit=args.expansion_maxit,
            num_restarts=args.num_restarts,
            maxit=args.maxit,
            tol=args.tol,
            rng=np.random.default_rng(args.seed),
            verbose=False,
            state_prev=state,
            A_block=A_block,
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
            combined_rank=internal_rank,
        )
        V_internal = np.ascontiguousarray(np.asarray(V_score[:, :internal_rank], dtype=np.float64))
        Q_oracle, _ = probe.projected_true_span_oracle(
            M_actual, V_exact[:, :final_rank], final_rank, dtype=np.float64
        )
        oracle1 = np.ascontiguousarray(Q_oracle[:, 0], dtype=np.float64)
        oracle2 = np.ascontiguousarray(Q_oracle[:, 1], dtype=np.float64)
        internal_cos = probe.subspace_principal_cosines(V_internal, Q_oracle)

        state_prev_for_block = state
        state, V_carried, _ = make_state(
            M_actual, V_internal, rows_seen, A_block, n, state, old_row_memory
        )
        V_final2 = np.ascontiguousarray(V_carried[:, :final_rank], dtype=np.float64)
        final_cos = probe.subspace_principal_cosines(V_final2, Q_oracle)
        Q_internal = probe.orthonormalize_columns(V_internal, dtype=np.float64)
        Q_final2 = probe.orthonormalize_columns(V_final2, dtype=np.float64)
        rec = {
            "matrix": matrix,
            "internal_rank": int(internal_rank),
            "final_rank": int(final_rank),
            "block": int(block),
            "row_start": int(start0 + 1),
            "row_end": int(end0),
            "internal_subspace_cos1": float(internal_cos[0]),
            "internal_subspace_cos2": float(internal_cos[1]),
            "internal_oracle1_projection": float(np.linalg.norm(Q_internal.T @ oracle1)),
            "internal_oracle2_projection": float(np.linalg.norm(Q_internal.T @ oracle2)),
            "internal_max_col_vs_oracle2": float(np.max(np.abs(V_internal.T @ oracle2))),
            "compressed_subspace_cos1": float(final_cos[0]),
            "compressed_subspace_cos2": float(final_cos[1]),
            "compressed_oracle1_projection": float(np.linalg.norm(Q_final2.T @ oracle1)),
            "compressed_oracle2_projection": float(np.linalg.norm(Q_final2.T @ oracle2)),
            "compressed_opt1_vs_oracle1": abs_cos(V_final2[:, 0], oracle1),
            "compressed_opt2_vs_oracle2": abs_cos(V_final2[:, 1], oracle2),
        }
        records.append(rec)

        for variant in compression_variants:
            V_variant = rank2_variant_frame(
                variant,
                M_actual,
                V_internal,
                V_carried,
                Q_oracle,
                A_block,
                n,
                state_prev_for_block,
                old_row_memory,
                np.random.default_rng(args.seed + 1000003 * int(internal_rank) + block),
                args.variant_random_starts,
            )
            variant_records.append(
                compression_variant_record(
                    matrix,
                    internal_rank,
                    block,
                    (start0 + 1, end0),
                    variant,
                    V_variant,
                    V_internal,
                    Q_oracle,
                    M_actual,
                    A_block,
                    n,
                    state_prev_for_block,
                    old_row_memory,
                )
            )

        old_row_memory = probe.select_old_row_memory(
            A[:end0, :].astype(work_dtype, copy=False),
            V_carried.astype(work_dtype, copy=False),
            args.old_memory_size,
            np.random.default_rng(args.seed + end0),
            return_indices=False,
        )

    return {
        "matrix": matrix,
        "internal_rank": int(internal_rank),
        "final_rank": int(final_rank),
        "sigma1": float(sigma1),
        "records": records,
        "variant_records": variant_records,
    }


def finite_mean(values):
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def finite_min(values):
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return float("nan")
    return float(np.min(vals))


def interpret(result):
    rows = result["records"]
    usable = [r for r in rows if r["block"] >= 3]
    final = rows[-1]
    summary = {
        "matrix": result["matrix"],
        "method": result.get("method", "combined"),
        "combined_rank": result.get("combined_rank"),
        "final_mean_align": result["final_mean_align"],
        "final_opt1_vs_oracle1": final["opt1_vs_oracle1"],
        "final_opt2_vs_oracle2": final["opt2_vs_oracle2"],
        "final_subspace_cos2": final["subspace_cos2"],
        "mean_opt1_vs_oracle1_b3plus": finite_mean(r["opt1_vs_oracle1"] for r in usable),
        "mean_opt2_vs_oracle2_b3plus": finite_mean(r["opt2_vs_oracle2"] for r in usable),
        "min_opt2_vs_oracle2_b3plus": finite_min(r["opt2_vs_oracle2"] for r in usable),
        "mean_prev_opt2_tail_cos_b3plus": finite_mean(r["prev_opt2_tail_cos"] for r in usable),
        "mean_prev_opt2_score_ratio_b3plus": finite_mean(r["prev_opt2_score_ratio"] for r in usable),
        "mean_opt2_sketch_score_share_b3plus": finite_mean(r["opt2_sketch_score_share"] for r in usable),
        "mean_opt2_actual_over_zero_b3plus": finite_mean(r["opt2_actual_over_zero"] for r in usable),
        "mean_opt2_actual_margin_vs_oracle2_vs_opt1_b3plus": finite_mean(
            r["opt2_actual_margin_vs_oracle2_vs_opt1"] for r in usable
        ),
        "mean_opt2_zero_margin_vs_oracle2_vs_opt1_b3plus": finite_mean(
            r["opt2_zero_margin_vs_oracle2_vs_opt1"] for r in usable
        ),
    }
    story_holds = (
        summary["mean_opt2_vs_oracle2_b3plus"] < 0.75
        and summary["mean_prev_opt2_tail_cos_b3plus"] > 0.75
        and summary["mean_prev_opt2_score_ratio_b3plus"] > 0.75
        and summary["mean_opt2_actual_over_zero_b3plus"] > 2.0
        and summary["mean_opt2_actual_margin_vs_oracle2_vs_opt1_b3plus"] > 0.0
        and summary["mean_opt2_zero_margin_vs_oracle2_vs_opt1_b3plus"] < 0.0
    )
    control_converges = (
        summary["final_subspace_cos2"] > 0.95
        and summary["final_opt2_vs_oracle2"] > 0.80
    )
    summary["story_holds_by_threshold"] = bool(story_holds)
    summary["control_converges_by_threshold"] = bool(control_converges)
    return summary


def write_outputs(results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    interpretations = []
    for result in results:
        matrix = result["matrix"]
        method = result.get("method", "combined")
        jsonl_path = out_dir / f"{matrix}_{method}_tail_diagnostics.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for rec in result["records"]:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
                all_records.append(rec)
        interpretations.append(interpret(result))

    csv_path = out_dir / "tail_diagnostics_all.csv"
    if all_records:
        fieldnames = sorted({k for rec in all_records for k in rec.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)

    interp_path = out_dir / "tail_diagnostics_interpretation.json"
    with interp_path.open("w", encoding="utf-8") as f:
        json.dump(interpretations, f, indent=2, sort_keys=True)

    txt_path = out_dir / "tail_diagnostics_interpretation.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        for s in interpretations:
            f.write(f"matrix={s['matrix']}\n")
            f.write(f"  method={s['method']} combined_rank={s['combined_rank']}\n")
            f.write(f"  final mean_align={s['final_mean_align']:.6f}\n")
            f.write(f"  final opt1_vs_oracle1={s['final_opt1_vs_oracle1']:.6f}\n")
            f.write(f"  final opt2_vs_oracle2={s['final_opt2_vs_oracle2']:.6f}\n")
            f.write(f"  final subspace_cos2={s['final_subspace_cos2']:.6f}\n")
            f.write(f"  mean opt2_vs_oracle2 b>=3={s['mean_opt2_vs_oracle2_b3plus']:.6f}\n")
            f.write(f"  mean prev_opt2_tail_cos b>=3={s['mean_prev_opt2_tail_cos_b3plus']:.6f}\n")
            f.write(f"  mean prev_opt2_score_ratio b>=3={s['mean_prev_opt2_score_ratio_b3plus']:.6f}\n")
            f.write(f"  mean opt2 actual/zero b>=3={s['mean_opt2_actual_over_zero_b3plus']:.6f}\n")
            f.write(
                "  mean margin opt2-orthogonalized-oracle2: "
                f"actualB={s['mean_opt2_actual_margin_vs_oracle2_vs_opt1_b3plus']:.6f} "
                f"zeroB={s['mean_opt2_zero_margin_vs_oracle2_vs_opt1_b3plus']:.6f}\n"
            )
            f.write(f"  story_holds_by_threshold={s['story_holds_by_threshold']}\n")
            f.write(f"  control_converges_by_threshold={s['control_converges_by_threshold']}\n\n")
    return interp_path, txt_path, csv_path


def interpret_internal_rank(result):
    rows = result["records"]
    final = rows[-1]
    oracle2_internal = final["internal_oracle2_projection"]
    oracle2_compressed = final["compressed_oracle2_projection"]
    compressed_cos2 = final["compressed_subspace_cos2"]
    if oracle2_internal < 0.75:
        verdict = "oracle2_absent_internally"
    elif compressed_cos2 < 0.75 or oracle2_compressed < 0.75:
        verdict = "oracle2_present_but_compression_drops_rank2"
    else:
        verdict = "oracle2_present_and_final_rank2_recovers"
    return {
        "matrix": result["matrix"],
        "internal_rank": result["internal_rank"],
        "final_rank": result["final_rank"],
        "final_internal_oracle2_projection": oracle2_internal,
        "final_internal_subspace_cos2": final["internal_subspace_cos2"],
        "final_compressed_oracle2_projection": oracle2_compressed,
        "final_compressed_subspace_cos2": compressed_cos2,
        "final_compressed_opt2_vs_oracle2": final["compressed_opt2_vs_oracle2"],
        "verdict": verdict,
    }


def write_internal_rank_outputs(results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    all_variant_records = []
    for result in results:
        all_records.extend(result["records"])
        all_variant_records.extend(result.get("variant_records", []))

    csv_path = out_dir / "internal_rank_diagnostics_all.csv"
    if all_records:
        fieldnames = sorted({k for rec in all_records for k in rec.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)

    summaries = [interpret_internal_rank(result) for result in results]
    json_path = out_dir / "internal_rank_diagnostics_interpretation.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)

    txt_path = out_dir / "internal_rank_diagnostics_interpretation.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        for s in summaries:
            f.write(f"matrix={s['matrix']} internal_rank={s['internal_rank']} final_rank={s['final_rank']}\n")
            f.write(f"  final internal_oracle2_projection={s['final_internal_oracle2_projection']:.6f}\n")
            f.write(f"  final internal_subspace_cos2={s['final_internal_subspace_cos2']:.6f}\n")
            f.write(f"  final compressed_oracle2_projection={s['final_compressed_oracle2_projection']:.6f}\n")
            f.write(f"  final compressed_subspace_cos2={s['final_compressed_subspace_cos2']:.6f}\n")
            f.write(f"  final compressed_opt2_vs_oracle2={s['final_compressed_opt2_vs_oracle2']:.6f}\n")
            f.write(f"  verdict={s['verdict']}\n\n")
    variant_csv_path = out_dir / "compression_variant_diagnostics_all.csv"
    if all_variant_records:
        fieldnames = sorted({k for rec in all_variant_records for k in rec.keys()})
        with variant_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_variant_records)

    variant_summaries = []
    grouped = {}
    for rec in all_variant_records:
        key = (rec["matrix"], rec["internal_rank"], rec["variant"])
        grouped.setdefault(key, []).append(rec)
    for key, rows in sorted(grouped.items()):
        matrix, internal_rank, variant = key
        final = max(rows, key=lambda r: r["block"])
        if final["internal_oracle2_projection"] < 0.75:
            verdict = "oracle2_absent_internally"
        elif final["variant_subspace_cos2"] >= 0.75 and final["variant_oracle2_projection"] >= 0.75:
            verdict = "variant_recovers_rank2"
        else:
            verdict = "variant_drops_rank2"
        variant_summaries.append({
            "matrix": matrix,
            "internal_rank": int(internal_rank),
            "variant": variant,
            "final_internal_oracle2_projection": final["internal_oracle2_projection"],
            "final_variant_oracle2_projection": final["variant_oracle2_projection"],
            "final_variant_subspace_cos2": final["variant_subspace_cos2"],
            "final_variant_col2_vs_oracle2": final["variant_col2_vs_oracle2"],
            "final_variant_actualB_score_sum": final["variant_actualB_score_sum"],
            "final_variant_actualB_gain2_sum": final["variant_actualB_gain2_sum"],
            "final_variant_actualB_phi_mean": final["variant_actualB_phi_mean"],
            "verdict": verdict,
        })

    variant_json_path = out_dir / "compression_variant_diagnostics_interpretation.json"
    with variant_json_path.open("w", encoding="utf-8") as f:
        json.dump(variant_summaries, f, indent=2, sort_keys=True)

    variant_txt_path = out_dir / "compression_variant_diagnostics_interpretation.txt"
    with variant_txt_path.open("w", encoding="utf-8") as f:
        for s in variant_summaries:
            f.write(
                f"matrix={s['matrix']} internal_rank={s['internal_rank']} "
                f"variant={s['variant']}\n"
            )
            f.write(f"  final internal_oracle2_projection={s['final_internal_oracle2_projection']:.6f}\n")
            f.write(f"  final variant_oracle2_projection={s['final_variant_oracle2_projection']:.6f}\n")
            f.write(f"  final variant_subspace_cos2={s['final_variant_subspace_cos2']:.6f}\n")
            f.write(f"  final variant_col2_vs_oracle2={s['final_variant_col2_vs_oracle2']:.6f}\n")
            f.write(f"  final variant_actualB_score_sum={s['final_variant_actualB_score_sum']:.6f}\n")
            f.write(f"  final variant_actualB_gain2_sum={s['final_variant_actualB_gain2_sum']:.6f}\n")
            f.write(f"  final variant_actualB_phi_mean={s['final_variant_actualB_phi_mean']:.6f}\n")
            f.write(f"  verdict={s['verdict']}\n\n")
    return json_path, txt_path, csv_path, variant_json_path, variant_txt_path, variant_csv_path


def plot_internal_rank_outputs(results, out_dir):
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for result in results:
        records = result["records"]
        variant_records = result.get("variant_records", [])
        if not records:
            continue
        matrix = result["matrix"]
        internal_rank = result["internal_rank"]
        x = np.asarray([r["block"] for r in records], dtype=float)
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
        fig.suptitle(f"{matrix} / internal rank {internal_rank}")
        axes[0].plot(
            x,
            [r["internal_oracle2_projection"] for r in records],
            marker="o",
            linewidth=1.4,
            markersize=3,
            label="internal_oracle2_projection",
        )
        axes[0].plot(
            x,
            [r["compressed_oracle2_projection"] for r in records],
            marker="o",
            linewidth=1.4,
            markersize=3,
            label="left_projected_oracle2_projection",
        )
        axes[0].set_ylim(-0.02, 1.02)
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc="best", fontsize=8)

        by_variant = {}
        for rec in variant_records:
            by_variant.setdefault(rec["variant"], []).append(rec)
        for variant, rows in sorted(by_variant.items()):
            rows = sorted(rows, key=lambda r: r["block"])
            axes[1].plot(
                [r["block"] for r in rows],
                [r["variant_subspace_cos2"] for r in rows],
                marker="o",
                linewidth=1.3,
                markersize=3,
                label=variant,
            )
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].set_xlabel("block")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="best", fontsize=8)
        plot_path = out_dir / f"{matrix}_internal_rank_{internal_rank}_compression_variants.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        paths.append(plot_path)
    return paths


def plot_outputs(results, out_dir):
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []
    panels = [
        (
            "alignment and tail mass",
            ["opt1_vs_oracle1", "opt2_vs_oracle2", "opt1_tail_mass", "opt2_tail_mass"],
        ),
        (
            "previous-slot persistence",
            ["prev_opt1_tail_cos", "prev_opt2_tail_cos", "prev_opt1_score_ratio", "prev_opt2_score_ratio"],
        ),
        (
            "actual vs zero-B scores",
            [
                "opt1_actualB_score",
                "opt1_zeroB_score",
                "oracle1_actualB_score",
                "opt2_actualB_score",
                "opt2_zeroB_score",
                "oracle2_vs_opt1_actualB_score",
            ],
        ),
        (
            "sketch share and margins",
            [
                "opt1_sketch_score_share",
                "opt2_sketch_score_share",
                "opt1_actual_margin_vs_oracle1",
                "opt2_actual_margin_vs_oracle2_vs_opt1",
            ],
        ),
    ]

    for result in results:
        records = result["records"]
        if not records:
            continue
        matrix = result["matrix"]
        method = result.get("method", "combined")
        x = np.asarray([r["block"] for r in records], dtype=float)
        fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True, constrained_layout=True)
        fig.suptitle(f"{matrix} / {method}")
        for ax, (title, fields) in zip(axes, panels):
            for field in fields:
                vals = np.asarray([r.get(field, np.nan) for r in records], dtype=float)
                if np.isfinite(vals).any():
                    ax.plot(x, vals, marker="o", linewidth=1.4, markersize=3, label=field)
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
        axes[-1].set_xlabel("block")
        plot_path = out_dir / f"{matrix}_{method}_tail_diagnostics.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        plot_paths.append(plot_path)
    return plot_paths


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrices", nargs="+", default=["mixed-tail-sharp", "diffuse-diffuse", "static-cex"])
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--win", type=int, default=32)
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
    p.add_argument("--num-restarts", type=int, default=8)
    p.add_argument("--maxit", type=int, default=120)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--post-expansion-maxit", type=int, default=80)
    p.add_argument("--r-sig", type=int, default=2)
    p.add_argument("--alpha-sig", type=float, default=0.003)
    p.add_argument("--alpha-tail", type=float, default=0.0145)
    p.add_argument("--tail-scale", type=float, default=0.99)
    p.add_argument("--sigma1", type=float, default=0.991)
    p.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    p.add_argument("--output-dir", default="summary/tail_conspiracy_python")
    p.add_argument("--methods", nargs="+", choices=("combined", "hybrid"), default=["combined"])
    p.add_argument(
        "--internal-ranks",
        nargs="+",
        type=int,
        default=[],
        help="Also run internal-rank oversampling diagnostics and compress/evaluate final rank 2.",
    )
    p.add_argument(
        "--only-internal-ranks",
        action="store_true",
        help="Skip combined/hybrid diagnostics and run only --internal-ranks.",
    )
    p.add_argument(
        "--compression-variants",
        nargs="+",
        choices=(
            "left_projected",
            "internal_projected_svd",
            "internal_current_svd",
            "internal_first2",
            "internal_score_top2",
            "internal_zeroB_score_top2",
            "internal_actualB_score_opt",
            "internal_zeroB_score_opt",
            "oracle_projected",
            "opt1_oracle2",
        ),
        default=[
            "left_projected",
            "internal_projected_svd",
            "internal_current_svd",
            "internal_first2",
            "internal_score_top2",
            "internal_zeroB_score_top2",
            "internal_actualB_score_opt",
            "internal_zeroB_score_opt",
            "oracle_projected",
            "opt1_oracle2",
        ],
        help="Rank-2 compression comparators evaluated from the same internal frame.",
    )
    p.add_argument(
        "--variant-random-starts",
        type=int,
        default=64,
        help="Random Stiefel starts for objective-aware internal compression variants.",
    )
    p.add_argument("--plot-internal", action="store_true", help="Write internal-rank compression trajectory plots.")
    p.add_argument(
        "--hybrid-combined-rank",
        type=int,
        default=1,
        help="Hybrid split: first this many rank slots use the combined optimizer; remaining slots use deflated SVD.",
    )
    p.add_argument("--plot", action="store_true", help="Write four-panel PNG diagnostics for each matrix/method.")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    results = []
    out_dir = Path(args.output_dir)
    if not args.only_internal_ranks:
        for method in args.methods:
            for matrix in args.matrices:
                print(f"running method={method} matrix={matrix}")
                results.append(run_matrix(matrix, method, args))
        interp_json, interp_txt, csv_path = write_outputs(results, out_dir)
        print(f"wrote {interp_json}")
        print(f"wrote {interp_txt}")
        print(f"wrote {csv_path}")
        if args.plot:
            for path in plot_outputs(results, out_dir):
                print(f"wrote {path}")
    if args.internal_ranks:
        internal_results = []
        for internal_rank in args.internal_ranks:
            for matrix in args.matrices:
                print(f"running internal_rank={internal_rank} matrix={matrix}")
                internal_results.append(run_internal_rank_matrix(matrix, internal_rank, args))
        ir_json, ir_txt, ir_csv, cv_json, cv_txt, cv_csv = write_internal_rank_outputs(internal_results, out_dir)
        print(f"wrote {ir_json}")
        print(f"wrote {ir_txt}")
        print(f"wrote {ir_csv}")
        print(f"wrote {cv_json}")
        print(f"wrote {cv_txt}")
        print(f"wrote {cv_csv}")
        if args.plot_internal:
            for path in plot_internal_rank_outputs(internal_results, out_dir):
                print(f"wrote {path}")
    print(f"elapsed={time.time() - t0:.3f}s")


if __name__ == "__main__":
    main()
