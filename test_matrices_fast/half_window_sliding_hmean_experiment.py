import argparse
import csv
import json
import time

import numpy as np

import cex_restricted_space_probe as probe
from second_slot_tail_bias_diagnostic import make_state, orth_against, raw_oracle_columns, svd_complement


LABELS = [
    "opt2",
    "q2_vs_q1oracle",
    "q2_raw_projected",
    "half2_complement",
    "mgain_deflated_svd",
    "opt2_outside",
    "block_complement",
    "prev_opt2",
]


# Value-only candidate pool (default). The ORACLE-AWARE pool also includes
# "q2_vs_q1oracle", which is built from V_exact[:,1] projected into the row
# span (see second_slot_tail_bias_diagnostic.raw_oracle_columns); that entry
# leaks the true second right singular vector and turns this baseline into an
# upper bound, not a value-only production policy. INFRA-10 (2026-04-28)
# stripped it from the default pool. Use --online-include-oracle to opt back
# into the ceiling reference.
ONLINE_POOL_VALUE_ONLY = ("opt2", "mgain_deflated_svd", "block_complement", "prev_opt2")
ONLINE_POOL_ORACLE_AWARE = ("opt2", "q2_vs_q1oracle", "mgain_deflated_svd", "block_complement", "prev_opt2")
ONLINE_POOL = ONLINE_POOL_VALUE_ONLY
ONLINE_HMEAN_POLICIES = (
    "future_hmean_online",
    "future_hmean_online_joint",
    "future_hmean_triplet_online",
    "future_hmean_nested_online",
    "future_hmean_pairwise_online",
    "future_hmean_weighted_online",
    "future_hmean_online_no_phi",
    "future_hmean_triplet_online_no_phi",
    "future_hmean_nested_online_no_phi",
    "future_hmean_pairwise_online_no_phi",
    "future_hmean_weighted_online_no_phi",
    "future_hmean_evidence",
    "future_hmean_r_sk_g",
)


def normed(v):
    if v is None:
        return None
    out = np.asarray(v, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(out))
    if nrm <= 1e-30:
        return None
    return np.ascontiguousarray(out / nrm)


def hmean(a, b, eps=1e-30):
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    a = max(float(a), 0.0)
    b = max(float(b), 0.0)
    return float((2.0 * a * b) / max(a + b, eps))


def hmean_many(values, weights=None, eps=1e-30):
    vals = np.asarray(values, dtype=np.float64)
    if weights is None:
        w = np.ones_like(vals, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(vals) & np.isfinite(w) & (w > 0.0)
    if not np.any(valid):
        return np.nan
    vals = np.maximum(vals[valid], 0.0)
    w = w[valid]
    if np.any(vals <= eps):
        return 0.0
    return float(np.sum(w) / max(float(np.sum(w / vals)), eps))


def response_shape(A_block, v):
    if v is None:
        return {"relH": np.nan, "max_frac": np.nan, "top4_frac": np.nan}
    y = np.asarray(A_block, dtype=np.float64) @ np.asarray(v, dtype=np.float64)
    e = y * y
    total = max(float(np.sum(e)), 1e-30)
    p = e / total
    p_pos = p[p > 0.0]
    H = -float(np.sum(p_pos * np.log(p_pos))) if p_pos.size else np.nan
    relH = H / np.log(max(len(e), 2)) if np.isfinite(H) else np.nan
    sorted_p = np.sort(p)[::-1]
    return {
        "relH": relH,
        "max_frac": float(sorted_p[0]) if sorted_p.size else np.nan,
        "top4_frac": float(np.sum(sorted_p[: min(4, sorted_p.size)])) if sorted_p.size else np.nan,
    }


def block_svd_complement(A_block, Q):
    _, _, Vh = np.linalg.svd(np.asarray(A_block, dtype=np.float64), full_matrices=False)
    best = None
    best_gain = -np.inf
    for row in Vh[: min(16, Vh.shape[0])]:
        cand = orth_against(row, Q)
        if cand is None:
            continue
        gain = float(np.linalg.norm(np.asarray(A_block, dtype=np.float64) @ cand) ** 2)
        if gain > best_gain:
            best_gain = gain
            best = cand
    return best


def outside_component(v, Q_oracle):
    if v is None or Q_oracle is None:
        return None
    q = np.asarray(Q_oracle, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    out = vv - q @ (q.T @ vv)
    return normed(out)


def span_objective(W, A_cur, A_fut):
    Y = np.asarray(A_cur, dtype=np.float64) @ W
    Z = np.asarray(A_fut, dtype=np.float64) @ W
    a = float(np.sum(Y * Y))
    b = float(np.sum(Z * Z))
    hm = 2.0 * a * b / max(a + b, 1e-30)
    e = np.sum(Y * Y, axis=1)
    S = float(np.sum(e))
    if S <= 1e-30:
        return 0.0, a, b, 0.0
    p = e / S
    p_pos = p[p > 0.0]
    H_nat = -float(np.sum(p_pos * np.log(p_pos))) if p_pos.size else 0.0
    m = max(len(e), 2)
    rel = H_nat / np.log(m)
    rel = max(rel, 0.0)
    return hm * rel, a, b, rel


def _candidate_span_basis(V_init, candidate_vecs):
    cols = [np.asarray(V_init[:, j], dtype=np.float64).reshape(-1) for j in range(V_init.shape[1])]
    for c in candidate_vecs:
        if c is None:
            continue
        arr = np.asarray(c, dtype=np.float64).reshape(-1)
        nrm = float(np.linalg.norm(arr))
        if nrm > 1e-30:
            cols.append(arr / nrm)
    if not cols:
        return None
    M = np.column_stack(cols)
    Q, R = np.linalg.qr(M)
    diag = np.abs(np.diag(R))
    if diag.size == 0:
        return None
    cutoff = 1e-10 * float(diag.max())
    keep = diag > cutoff
    B = Q[:, keep]
    return np.ascontiguousarray(B)


def joint_span_polish(W_init, A_cur, A_fut, candidate_vecs, maxit=40, step_init=1.0, tol=1e-12):
    W0 = np.ascontiguousarray(np.asarray(W_init, dtype=np.float64).copy())
    Acc = np.asarray(A_cur, dtype=np.float64)
    Afc = np.asarray(A_fut, dtype=np.float64)
    J0, _, _, _ = span_objective(W0, Acc, Afc)

    B = _candidate_span_basis(W0, candidate_vecs)
    if B is None or B.shape[1] < W0.shape[1]:
        return W0, J0, J0

    C0 = B.T @ W0
    Qc, _ = np.linalg.qr(C0)
    C = Qc[:, : W0.shape[1]]
    Acb = Acc @ B
    Afb = Afc @ B
    AcbTAcb = Acb.T @ Acb
    AfbTAfb = Afb.T @ Afb

    def eval_C(Cmat):
        Wmat = B @ Cmat
        return span_objective(Wmat, Acc, Afc)

    J_best, a0, b0, h0 = eval_C(C)
    if not np.isfinite(J_best):
        return W0, J0, J0
    C_best = C.copy()

    for _ in range(maxit):
        Y = Acb @ C
        Z = Afb @ C
        a = float(np.sum(Y * Y))
        b = float(np.sum(Z * Z))
        denom2 = max((a + b) ** 2, 1e-30)
        ga = 2.0 * (AcbTAcb @ C)
        gb = 2.0 * (AfbTAfb @ C)
        grad = 2.0 * (b * b * ga + a * a * gb) / denom2
        CtG = C.T @ grad
        Xi = grad - C @ (0.5 * (CtG + CtG.T))
        xi_norm = float(np.linalg.norm(Xi))
        if xi_norm < tol:
            break
        step = step_init
        improved = False
        for _ in range(25):
            C_try = C + step * Xi
            Qt, _ = np.linalg.qr(C_try)
            C_try = Qt[:, : C.shape[1]]
            J_try, _, _, _ = eval_C(C_try)
            if np.isfinite(J_try) and J_try > J_best + 1e-12:
                C = C_try
                J_best = J_try
                C_best = C_try.copy()
                improved = True
                break
            step *= 0.5
        if not improved:
            break
    W_out = B @ C_best
    Qw, _ = np.linalg.qr(W_out)
    return np.ascontiguousarray(Qw[:, : W0.shape[1]]), J_best, J0


def oracle_projection_norms(M_gain, V_exact, rank, dtype):
    _, Q_row = probe.projected_true_span_oracle(
        np.asarray(M_gain, dtype=dtype),
        np.asarray(V_exact, dtype=dtype)[:, : int(rank)],
        int(rank),
        dtype=dtype,
    )
    norms = []
    for j in range(min(int(rank), np.asarray(V_exact).shape[1])):
        v = np.asarray(V_exact, dtype=dtype)[:, j]
        norms.append(float(np.linalg.norm(probe.project_onto_span(v, Q_row))))
    return np.asarray(norms, dtype=float)


def frame_tail_mass(V, V_exact, rank):
    Vq = probe.orthonormalize_columns(np.asarray(V, dtype=np.float64)[:, :rank], dtype=np.float64)
    Qsig = probe.orthonormalize_columns(np.asarray(V_exact, dtype=np.float64)[:, :rank], dtype=np.float64)
    sig_mass = float(np.linalg.norm(Qsig @ (Qsig.T @ Vq), ord="fro") ** 2 / max(rank, 1))
    return max(0.0, 1.0 - sig_mass)


def build_candidates(V_selected, Q_oracle, raw_oracle, M_gain, A_half1, A_half2, prev_opt2=None):
    v1 = np.asarray(V_selected[:, :1], dtype=np.float64)
    opt2 = normed(V_selected[:, 1]) if V_selected.shape[1] > 1 else None
    q2_raw = None
    if len(raw_oracle) > 1:
        q2_raw = normed(raw_oracle[1])
    elif Q_oracle.shape[1] > 1:
        q2_raw = normed(Q_oracle[:, 1])
    q2_qr = orth_against(q2_raw, Q_oracle[:, :1]) if q2_raw is not None else None
    half2_comp = block_svd_complement(A_half2, v1)
    block_comp = block_svd_complement(A_half1, v1)
    mgain_deflated = svd_complement(M_gain, v1)
    opt2_out = outside_component(opt2, Q_oracle)
    prev_opt2_cand = None
    if prev_opt2 is not None:
        prev_opt2_cand = orth_against(prev_opt2, v1)
    return {
        "opt2": opt2,
        "q2_vs_q1oracle": q2_qr,
        "q2_raw_projected": q2_raw,
        "half2_complement": half2_comp,
        "mgain_deflated_svd": mgain_deflated,
        "opt2_outside": opt2_out,
        "block_complement": block_comp,
        "prev_opt2": prev_opt2_cand,
    }


def rank2_svd_frame(v1, chosen_v2, M_gain, rank=2):
    """Replace Gram-Schmidt with rank-2 SVD step within span{v1, chosen_v2}.

    Builds an orthonormal basis Qc of the 2D span, then takes the right
    singular vectors of M_gain @ Qc as the ordered frame. Returns columns
    sorted by descending singular value of M_gain restricted to the span.
    """
    v1 = np.asarray(v1, dtype=np.float64).reshape(-1)
    if chosen_v2 is None:
        return None
    v2 = np.asarray(chosen_v2, dtype=np.float64).reshape(-1)
    C = np.column_stack([v1, v2])
    Qc, Rc = np.linalg.qr(C)
    diag = np.abs(np.diag(Rc))
    if diag.size < 2 or diag[1] < 1e-10 * max(diag[0], 1e-30):
        return None
    Mc = np.asarray(M_gain, dtype=np.float64) @ Qc
    _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
    return np.ascontiguousarray(Qc @ Vt.T[:, :rank])


def score_half_candidates(candidates, A_half1, A_half2, A_sketch_prior=None, hm_weights=None):
    records = {}
    A_sketch = A_sketch_prior
    if hm_weights is None:
        hm_weights = (0, A_half1.shape[0], A_half2.shape[0])
    sketch_seen_rows, current_rows, future_rows = [int(w) for w in hm_weights]
    for label, v in candidates.items():
        if v is None:
            records[label] = None
            continue
        sketch_gain = np.nan
        if A_sketch is not None and np.asarray(A_sketch).size > 0:
            sketch_gain = float(np.linalg.norm(np.asarray(A_sketch, dtype=np.float64) @ v) ** 2)
        gain1 = float(np.linalg.norm(np.asarray(A_half1, dtype=np.float64) @ v) ** 2)
        gain2 = float(np.linalg.norm(np.asarray(A_half2, dtype=np.float64) @ v) ** 2)
        sketch_gain_for_concat = sketch_gain if np.isfinite(sketch_gain) else 0.0
        shape1 = response_shape(A_half1, v)
        shape2 = response_shape(A_half2, v)
        records[label] = {
            "vec": normed(v),
            "sketch_gain": sketch_gain,
            "gain1": gain1,
            "gain2": gain2,
            "sketch_gain1": sketch_gain_for_concat + gain1,
            "sketch_gain2": sketch_gain_for_concat + gain2,
            "relH1": shape1["relH"],
            "relH2": shape2["relH"],
            "max_frac2": shape2["max_frac"],
            "top4_frac2": shape2["top4_frac"],
        }
    finite = [r for r in records.values() if r is not None]
    if not finite:
        return records
    finite_sketch = [r for r in finite if np.isfinite(r["sketch_gain"])]
    max_sketch_gain = max((r["sketch_gain"] for r in finite_sketch), default=np.nan)
    max_gain1 = max(r["gain1"] for r in finite)
    max_gain2 = max(r["gain2"] for r in finite)
    max_sketch_gain1 = max(r["sketch_gain1"] for r in finite)
    max_sketch_gain2 = max(r["sketch_gain2"] for r in finite)
    half2_comp = candidates.get("half2_complement")
    for rec in finite:
        if np.isfinite(max_sketch_gain) and max_sketch_gain > 1e-30:
            rec["sketch_gain_share"] = rec["sketch_gain"] / max(max_sketch_gain, 1e-30)
        else:
            rec["sketch_gain_share"] = 1.0
        rec["gain1_share"] = rec["gain1"] / max(max_gain1, 1e-30)
        rec["gain2_share"] = rec["gain2"] / max(max_gain2, 1e-30)
        rec["sketch_gain1_share"] = rec["sketch_gain1"] / max(max_sketch_gain1, 1e-30)
        rec["sketch_gain2_share"] = rec["sketch_gain2"] / max(max_sketch_gain2, 1e-30)
        relH1 = max(float(rec["relH1"]), 0.0) if np.isfinite(rec["relH1"]) else 0.0
        relH2 = max(float(rec["relH2"]), 0.0) if np.isfinite(rec["relH2"]) else 0.0
        rec["obj_half_hmean"] = hmean(rec["gain1_share"], rec["gain2_share"]) * relH2
        rec["obj_future_online"] = hmean(rec["gain1_share"], rec["gain2_share"]) * relH1
        sk = rec["sketch_gain_share"]
        g1 = rec["gain1_share"]
        g2 = rec["gain2_share"]
        sg1 = rec["sketch_gain1_share"]
        sg2 = rec["sketch_gain2_share"]
        rec["obj_future_hmean_triplet_online"] = hmean_many([sk, g1, g2]) * relH1
        rec["obj_future_hmean_nested_online"] = hmean(sg1, g2) * relH1
        rec["obj_future_hmean_pairwise_online"] = hmean(sg1, sg2) * relH1
        rec["obj_future_hmean_weighted_online"] = hmean_many([sk, g1, g2], weights=hm_weights) * relH1
        rec["obj_future_online_no_phi"] = hmean(g1, g2)
        rec["obj_future_hmean_triplet_online_no_phi"] = hmean_many([sk, g1, g2])
        rec["obj_future_hmean_nested_online_no_phi"] = hmean(sg1, g2)
        rec["obj_future_hmean_pairwise_online_no_phi"] = hmean(sg1, sg2)
        rec["obj_future_hmean_weighted_online_no_phi"] = hmean_many([sk, g1, g2], weights=hm_weights)
        rec["sketch_seen_rows"] = sketch_seen_rows
        rec["current_rows"] = current_rows
        rec["future_rows"] = future_rows
        if half2_comp is None:
            rec["half2_comp_align2"] = np.nan
            rec["obj_half_hmean_guard"] = rec["obj_half_hmean"]
        else:
            align2 = float(np.dot(rec["vec"], half2_comp) ** 2)
            rec["half2_comp_align2"] = align2
            rec["obj_half_hmean_guard"] = rec["obj_half_hmean"] * max(1.0 - align2, 0.0)
    return records


def choose_second_slot(policy, records, fallback):
    if policy == "combined":
        return "opt2", fallback
    if policy in ("future_hmean_online", "future_hmean_online_joint"):
        key = "obj_future_online"
    elif policy == "future_hmean_online_no_phi":
        key = "obj_future_online_no_phi"
    elif policy == "future_hmean_triplet_online":
        key = "obj_future_hmean_triplet_online"
    elif policy == "future_hmean_triplet_online_no_phi":
        key = "obj_future_hmean_triplet_online_no_phi"
    elif policy == "future_hmean_nested_online":
        key = "obj_future_hmean_nested_online"
    elif policy == "future_hmean_nested_online_no_phi":
        key = "obj_future_hmean_nested_online_no_phi"
    elif policy == "future_hmean_pairwise_online":
        key = "obj_future_hmean_pairwise_online"
    elif policy == "future_hmean_pairwise_online_no_phi":
        key = "obj_future_hmean_pairwise_online_no_phi"
    elif policy == "future_hmean_weighted_online":
        key = "obj_future_hmean_weighted_online"
    elif policy == "future_hmean_weighted_online_no_phi":
        key = "obj_future_hmean_weighted_online_no_phi"
    elif policy == "half_hmean":
        key = "obj_half_hmean"
    else:
        key = "obj_half_hmean_guard"
    vals = {
        label: rec[key]
        for label, rec in records.items()
        if rec is not None and np.isfinite(rec.get(key, np.nan))
    }
    if not vals:
        return "opt2", fallback
    label = max(vals, key=vals.get)
    return label, records[label]["vec"]


def rel_err_sval(s_est, sigma1):
    top_sval_est = float(np.asarray(s_est, dtype=float).reshape(-1)[0])
    return abs(top_sval_est - float(sigma1)) / max(float(sigma1), 1e-30)


def run_pair_stream(A, V_exact, sigma1, args, policy, half_win, sliding):
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    n = A.shape[0]
    rank = int(args.rank)
    state = None
    old_row_memory = None
    V_r = None
    prev_opt2 = None
    rows = []
    t0 = time.time()

    step = half_win if sliding else 2 * half_win
    pair_count = 0
    for start0 in range(0, n - half_win, step):
        mid0 = start0 + half_win
        end0 = min(mid0 + half_win, n)
        if end0 - mid0 < half_win:
            break
        pair_count += 1
        if args.max_pairs is not None and pair_count > args.max_pairs:
            break
        block_id = pair_count
        A_half1 = np.asarray(A[start0:mid0, :], dtype=work_dtype)
        A_half2 = np.asarray(A[mid0:end0, :], dtype=work_dtype)

        if state is None:
            M_sketch = None
            M_gain = A_half1
            V_init = probe.row_norm_seed(A_half1, rank)
            rows_seen = A_half1.shape[0]
        else:
            B_top = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
            M_sketch = B_top
            M_gain = np.vstack([B_top, A_half1]).astype(work_dtype, copy=False)
            V_init = probe.row_norm_seed(A_half1, rank)
            rows_seen = state["rows_seen"] + A_half1.shape[0]

        if policy == "isvd":
            Mg = np.asarray(M_gain, dtype=np.float64)
            _, _, Vh = np.linalg.svd(Mg, full_matrices=False)
            V_default = np.ascontiguousarray(Vh[:rank, :].T)
        else:
            combined_rank = 1 if policy == "hybrid" else None
            V_score, _, _, _, diag_basis = probe.entropy_iter_basis_forget(
                M_gain=M_gain,
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
                A_block=A_half1,
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
                patience=args.patience,
                patience_rel_tol=args.patience_rel_tol,
            )
            V_default = np.ascontiguousarray(np.asarray(V_score[:, :rank], dtype=np.float64))

        Q_oracle, raw_oracle = raw_oracle_columns(M_gain, V_exact, rank, np.float64)

        if policy in ("isvd", "hybrid"):
            candidates = {}
            records = {}
            chosen_label = policy
            chosen_v2 = None
        else:
            candidates = build_candidates(
                V_default, Q_oracle, raw_oracle, M_gain, A_half1, A_half2, prev_opt2=prev_opt2
            )
            if policy in ONLINE_HMEAN_POLICIES:
                pool = (
                    ONLINE_POOL_ORACLE_AWARE
                    if getattr(args, "online_include_oracle", False)
                    else ONLINE_POOL_VALUE_ONLY
                )
                candidates = {k: candidates.get(k) for k in pool}
            else:
                candidates = {
                    k: v for k, v in candidates.items() if k not in ("block_complement", "prev_opt2")
                }
            prior_seen = 0 if state is None else int(state["rows_seen"])
            hm_weights = (prior_seen, A_half1.shape[0], A_half2.shape[0])
            records = score_half_candidates(
                candidates,
                A_half1,
                A_half2,
                A_sketch_prior=M_sketch,
                hm_weights=hm_weights,
            )
            chosen_label, chosen_v2 = choose_second_slot(policy, records, candidates.get("opt2"))
            if policy == "future_hmean_evidence":
                from future_hmean_optimizer_diagnostic import orth_basis_against, rowspace_basis
                from hmean_evidence_score import optimize_hm_evi_in_basis

                A_sketch_for_evi = (
                    np.asarray(M_sketch, dtype=np.float64)
                    if M_sketch is not None and np.asarray(M_sketch).size
                    else None
                )
                A_h1_f = np.asarray(A_half1, dtype=np.float64)
                A_h2_f = np.asarray(A_half2, dtype=np.float64)
                sk_end = mid0 - A_half1.shape[0]
                if sk_end > 0:
                    sk_F2 = float(np.sum(A[:sk_end] * A[:sk_end]))
                    c_sk = (sk_end / sk_F2) if sk_F2 > 0 else 0.0
                else:
                    c_sk = 0.0
                cur_F2 = float(np.sum(A_h1_f * A_h1_f))
                fut_F2 = float(np.sum(A_h2_f * A_h2_f))
                c_g1 = A_half1.shape[0] / cur_F2 if cur_F2 > 0 else 0.0
                c_g2 = A_half2.shape[0] / fut_F2 if fut_F2 > 0 else 0.0

                if A_sketch_for_evi is not None:
                    union_for_search = np.vstack([A_sketch_for_evi, A_h1_f, A_h2_f])
                else:
                    union_for_search = np.vstack([A_h1_f, A_h2_f])
                B_search = orth_basis_against(rowspace_basis(union_for_search), V_default[:, 0])

                starts_evi = [V_default[:, 1]]
                for v in candidates.values():
                    if v is not None:
                        starts_evi.append(v)
                Vbasis = diag_basis.get("Vbasis_final") if diag_basis is not None else None
                if Vbasis is not None:
                    Vb = np.asarray(Vbasis, dtype=np.float64)
                    for j in range(min(Vb.shape[1], 8)):
                        starts_evi.append(Vb[:, j])

                evi_best = optimize_hm_evi_in_basis(
                    A_h1_f, A_h2_f, A_sketch_for_evi,
                    c_sk, c_g1, c_g2, rank, A_half1.shape[0], A_half2.shape[0],
                    B_search, starts_evi,
                    np.random.default_rng(args.seed + 31337 + block_id),
                    args.maxit, args.tol, 24,
                )
                if evi_best is not None and evi_best.get("vec") is not None:
                    chosen_v2 = np.asarray(evi_best["vec"], dtype=np.float64).reshape(-1)
                    chosen_label = "evi_best"

            if policy == "future_hmean_r_sk_g":
                from future_hmean_optimizer_diagnostic import orth_basis_against, rowspace_basis
                from r_sk_g_score import optimize_r_sk_g_in_basis

                A_sketch_for_rsk = (
                    np.asarray(M_sketch, dtype=np.float64)
                    if M_sketch is not None and np.asarray(M_sketch).size
                    else None
                )
                A_h1_f = np.asarray(A_half1, dtype=np.float64)
                A_h2_f = np.asarray(A_half2, dtype=np.float64)
                sk_end = mid0 - A_half1.shape[0]
                if sk_end > 0:
                    sk_F2 = float(np.sum(A[:sk_end] * A[:sk_end]))
                    c_sk = (sk_end / sk_F2) if sk_F2 > 0 else 0.0
                else:
                    c_sk = 0.0

                # F-norms for the S6 score. sk_F2_low is the rank-r CARRY (NOT
                # the full prefix Frobenius); see f_hm3_score_implementation_context.
                cur_F2_rsk = float(np.sum(A_h1_f * A_h1_f))
                fut_F2_rsk = float(np.sum(A_h2_f * A_h2_f))
                if A_sketch_for_rsk is not None:
                    sk_F2_low_rsk = float(np.sum(A_sketch_for_rsk * A_sketch_for_rsk))
                else:
                    sk_F2_low_rsk = 0.0

                rsk_variant = getattr(args, "rsk_variant", "S6")
                rsk_no_deflate = getattr(args, "rsk_no_deflate", False)

                # Op-norm-squared per-block constants for S6_OP (AB-02 weighting
                # ablation). Only computed when needed to avoid wasted SVDs.
                # A_sketch is rank-r (= state.s · state.V^T) so its top SV is just
                # state.s[0] (state.s sorted descending by streaming SVD invariant).
                cur_op2_rsk = None
                fut_op2_rsk = None
                sk_op2_low_rsk = None
                if rsk_variant == "S6_OP":
                    cur_op2_rsk = float(np.linalg.svd(A_h1_f, compute_uv=False)[0] ** 2) \
                        if A_h1_f.size else 0.0
                    fut_op2_rsk = float(np.linalg.svd(A_h2_f, compute_uv=False)[0] ** 2) \
                        if A_h2_f.size else 0.0
                    if A_sketch_for_rsk is not None:
                        if state is not None and state.get("s") is not None and np.asarray(state["s"]).size:
                            sk_op2_low_rsk = float(np.asarray(state["s"], dtype=np.float64)[0] ** 2)
                        else:
                            sk_op2_low_rsk = float(np.linalg.svd(A_sketch_for_rsk,
                                                                  compute_uv=False)[0] ** 2)
                    else:
                        sk_op2_low_rsk = 0.0

                if A_sketch_for_rsk is not None:
                    union_for_search = np.vstack([A_sketch_for_rsk, A_h1_f, A_h2_f])
                else:
                    union_for_search = np.vstack([A_h1_f, A_h2_f])
                B_union_rsk = rowspace_basis(union_for_search)
                if rsk_no_deflate:
                    B_search = B_union_rsk
                else:
                    B_search = orth_basis_against(B_union_rsk, V_default[:, 0])

                starts_rsk = [V_default[:, 1]]
                for v in candidates.values():
                    if v is not None:
                        starts_rsk.append(v)
                Vbasis = diag_basis.get("Vbasis_final") if diag_basis is not None else None
                if Vbasis is not None:
                    Vb = np.asarray(Vbasis, dtype=np.float64)
                    for j in range(min(Vb.shape[1], 8)):
                        starts_rsk.append(Vb[:, j])

                V_state_rsk = None
                if state is not None and state.get("V") is not None:
                    Vs_arr = np.asarray(state["V"], dtype=np.float64)
                    if Vs_arr.size:
                        V_state_rsk = Vs_arr
                        # Sketch-init: include the carried right singular vectors
                        # as warm-starts. For S5 these are the only on-sphere
                        # points that exceed the gate; for S6 they have non-zero
                        # u_sk and serve as a useful basin (P1 in context).
                        for j in range(min(Vs_arr.shape[1], 2)):
                            starts_rsk.append(Vs_arr[:, j])

                # P1 (S6): also seed from the rank-2 SVD of M_gain = [B_top; A_cur].
                # These have decent u_sk and reasonable u_g1 — combined_v's are
                # ~zero on u_g2 so they alone are a bad basin under S6.
                M_gain_arr = np.asarray(M_gain, dtype=np.float64)
                if M_gain_arr.size:
                    try:
                        _, _, Vt_mg = np.linalg.svd(M_gain_arr, full_matrices=False)
                        for j in range(min(Vt_mg.shape[0], 2)):
                            starts_rsk.append(Vt_mg[j])
                    except np.linalg.LinAlgError:
                        pass

                rsk_best = optimize_r_sk_g_in_basis(
                    A_h1_f, A_h2_f, A_sketch_for_rsk, c_sk,
                    B_search, starts_rsk,
                    np.random.default_rng(args.seed + 41000 + block_id),
                    args.maxit, args.tol, 24,
                    variant=rsk_variant,
                    alpha=getattr(args, "rsk_alpha", 1.0),
                    beta=getattr(args, "rsk_beta", 2.0),
                    gamma=getattr(args, "rsk_gamma", 1.0),
                    V_state=V_state_rsk,
                    cur_F2=cur_F2_rsk,
                    fut_F2=fut_F2_rsk,
                    sk_F2_low=sk_F2_low_rsk,
                    cur_op2=cur_op2_rsk,
                    fut_op2=fut_op2_rsk,
                    sk_op2_low=sk_op2_low_rsk,
                )
                if rsk_best is not None and rsk_best.get("vec") is not None:
                    chosen_v2 = np.asarray(rsk_best["vec"], dtype=np.float64).reshape(-1)
                    chosen_label = "rsk_g_best"

        winner_oracle_mass = np.nan
        if chosen_v2 is not None and Q_oracle is not None and np.asarray(Q_oracle).size > 0:
            Q = np.asarray(Q_oracle, dtype=np.float64)
            w = np.asarray(chosen_v2, dtype=np.float64).reshape(-1)
            wn = float(np.linalg.norm(w))
            if wn > 1e-30:
                w = w / wn
                winner_oracle_mass = float(np.linalg.norm(Q @ (Q.T @ w)) ** 2)

        svd_frame_used = False
        if policy in ONLINE_HMEAN_POLICIES and chosen_v2 is not None:
            V_svd = rank2_svd_frame(V_default[:, 0], chosen_v2, M_gain, rank=rank)
            if V_svd is not None and V_svd.shape[1] >= rank:
                V_selected = np.ascontiguousarray(V_svd[:, :rank])
                svd_frame_used = True
        if not svd_frame_used:
            V_selected = np.ascontiguousarray(V_default.copy())
            if chosen_v2 is not None:
                V_selected[:, 1] = chosen_v2
            V_selected = probe.orthonormalize_columns(V_selected[:, :rank], dtype=np.float64)[:, :rank]

        joint_J_before = np.nan
        joint_J_after = np.nan
        polish_diag = {}
        if policy == "future_hmean_online_joint":
            candidate_vecs = [v for v in candidates.values() if v is not None]
            V_before = V_selected.copy()
            W_polished, J_after, J_before = joint_span_polish(
                V_selected, A_half1, A_half2, candidate_vecs, maxit=40, step_init=1.0
            )
            V_selected = probe.orthonormalize_columns(
                np.ascontiguousarray(W_polished, dtype=np.float64), dtype=np.float64
            )[:, :rank]
            joint_J_before = J_before
            joint_J_after = J_after

            Ve2 = np.asarray(V_exact, dtype=np.float64)[:, :2]
            def _col_diag(tag, V):
                for j in range(min(2, V.shape[1])):
                    cj = V[:, j]
                    dot0 = float(np.dot(cj, Ve2[:, 0]))
                    dot1 = float(np.dot(cj, Ve2[:, 1]))
                    out = cj - Ve2 @ (Ve2.T @ cj)
                    out_mass = float(np.dot(out, out))
                    polish_diag[f"{tag}_c{j}_Ve0"] = dot0
                    polish_diag[f"{tag}_c{j}_Ve1"] = dot1
                    polish_diag[f"{tag}_c{j}_outmass"] = out_mass
                    if out_mass > 1e-8:
                        out_n = out / np.sqrt(out_mass)
                        best_lbl = None
                        best_ov = 0.0
                        for lbl, cand in candidates.items():
                            if cand is None:
                                continue
                            ov = abs(float(np.dot(out_n, np.asarray(cand, dtype=np.float64).reshape(-1))))
                            if ov > best_ov:
                                best_ov = ov
                                best_lbl = lbl
                        polish_diag[f"{tag}_c{j}_out_bestlbl"] = best_lbl or ""
                        polish_diag[f"{tag}_c{j}_out_bestov"] = best_ov
                    else:
                        polish_diag[f"{tag}_c{j}_out_bestlbl"] = ""
                        polish_diag[f"{tag}_c{j}_out_bestov"] = 0.0

            _col_diag("pre", V_before)
            _col_diag("post", V_selected)

        score_selected = np.zeros(rank, dtype=float)
        H_selected = np.zeros(rank, dtype=float)
        for j in range(rank):
            score_selected[j], _, H_selected[j] = probe.score_full_vector_details_forget(
                M_gain,
                A_half1,
                V_selected[:, j],
                n,
                state_prev=state,
                score_variant="combined",
                old_row_memory=old_row_memory,
            )

        cos = probe.subspace_principal_cosines(V_selected, Q_oracle)
        exact_cos = probe.subspace_principal_cosines(V_selected, V_exact[:, :rank])
        oracle_proj_norm = oracle_projection_norms(M_gain, V_exact, rank, np.float64)
        tail_mass = frame_tail_mass(V_selected, V_exact, rank)

        _, s_new, Vt_new, _ = probe.left_projected_operator_svd_factors(V_selected.T, M_gain)
        V_carried = np.ascontiguousarray(np.asarray(Vt_new.T[:, :rank], dtype=np.float64))
        car_oracle_cos = probe.subspace_principal_cosines(V_carried, Q_oracle)
        car_exact_cos = probe.subspace_principal_cosines(V_carried, V_exact[:, :rank])
        sel_car_cos = probe.subspace_principal_cosines(V_selected, V_carried)
        car_tail_mass = frame_tail_mass(V_carried, V_exact, rank)

        survive_subspace = np.nan
        survive_v2 = np.nan
        prev2_score_ratio = np.nan
        if prev_opt2 is not None:
            survive_subspace = float(np.linalg.norm(V_selected @ (V_selected.T @ prev_opt2)))
            survive_v2 = abs(float(V_selected[:, 1] @ prev_opt2))
            prev_score = probe.combined_score_component_details(
                M_gain, A_half1, prev_opt2, n, state_prev=state, old_row_memory=old_row_memory
            )["score_total"]
            curr_score = probe.combined_score_component_details(
                M_gain, A_half1, V_selected[:, 1], n, state_prev=state, old_row_memory=old_row_memory
            )["score_total"]
            prev2_score_ratio = float(prev_score / max(curr_score, 1e-30))

        chosen_rec = records.get(chosen_label)
        rows.append(
            {
                "pair": pair_count,
                "block": block_id,
                "rows_seen": end0 if not sliding else mid0,
                "policy": policy,
                "mode": "sliding" if sliding else "split_probe",
                "half_win": half_win,
                "selected_label": chosen_label,
                "selected_half_hmean": np.nan if chosen_rec is None else chosen_rec["obj_half_hmean"],
                "selected_half_hmean_guard": np.nan if chosen_rec is None else chosen_rec["obj_half_hmean_guard"],
                "selected_future_hmean_triplet_online": (
                    np.nan if chosen_rec is None else chosen_rec["obj_future_hmean_triplet_online"]
                ),
                "selected_future_hmean_nested_online": (
                    np.nan if chosen_rec is None else chosen_rec["obj_future_hmean_nested_online"]
                ),
                "selected_future_hmean_pairwise_online": (
                    np.nan if chosen_rec is None else chosen_rec["obj_future_hmean_pairwise_online"]
                ),
                "selected_future_hmean_weighted_online": (
                    np.nan if chosen_rec is None else chosen_rec["obj_future_hmean_weighted_online"]
                ),
                "selected_future_online_no_phi": (
                    np.nan if chosen_rec is None else chosen_rec["obj_future_online_no_phi"]
                ),
                "selected_future_hmean_triplet_online_no_phi": (
                    np.nan if chosen_rec is None else chosen_rec["obj_future_hmean_triplet_online_no_phi"]
                ),
                "selected_future_hmean_nested_online_no_phi": (
                    np.nan if chosen_rec is None else chosen_rec["obj_future_hmean_nested_online_no_phi"]
                ),
                "selected_future_hmean_pairwise_online_no_phi": (
                    np.nan if chosen_rec is None else chosen_rec["obj_future_hmean_pairwise_online_no_phi"]
                ),
                "selected_future_hmean_weighted_online_no_phi": (
                    np.nan if chosen_rec is None else chosen_rec["obj_future_hmean_weighted_online_no_phi"]
                ),
                "cos1": float(cos[0]) if len(cos) > 0 else np.nan,
                "cos2": float(cos[1]) if len(cos) > 1 else np.nan,
                "exact_cos1": float(exact_cos[0]) if len(exact_cos) > 0 else np.nan,
                "exact_cos2": float(exact_cos[1]) if len(exact_cos) > 1 else np.nan,
                "car_cos1": float(car_oracle_cos[0]) if len(car_oracle_cos) > 0 else np.nan,
                "car_cos2": float(car_oracle_cos[1]) if len(car_oracle_cos) > 1 else np.nan,
                "car_exact_cos1": float(car_exact_cos[0]) if len(car_exact_cos) > 0 else np.nan,
                "car_exact_cos2": float(car_exact_cos[1]) if len(car_exact_cos) > 1 else np.nan,
                "sel_car_cos1": float(sel_car_cos[0]) if len(sel_car_cos) > 0 else np.nan,
                "sel_car_cos2": float(sel_car_cos[1]) if len(sel_car_cos) > 1 else np.nan,
                "oracle_proj_norm1": float(oracle_proj_norm[0]) if len(oracle_proj_norm) > 0 else np.nan,
                "oracle_proj_norm2": float(oracle_proj_norm[1]) if len(oracle_proj_norm) > 1 else np.nan,
                "tail_mass": tail_mass,
                "car_tail_mass": car_tail_mass,
                "survive_subspace": survive_subspace,
                "survive_v2": survive_v2,
                "prev2_score_ratio": prev2_score_ratio,
                "relerr_sval": rel_err_sval(s_new[:rank], sigma1),
                "winner_oracle_mass": winner_oracle_mass,
                "svd_frame_used": int(svd_frame_used),
                "joint_J_before": joint_J_before,
                "joint_J_after": joint_J_after,
                **polish_diag,
            }
        )

        state, V_r, _ = make_state(M_gain, V_selected, H_selected, score_selected, rows_seen)
        seen_for_memory = A[:mid0, :] if sliding else A[:end0, :]
        old_row_memory, _ = probe.select_old_row_memory(
            np.asarray(seen_for_memory, dtype=work_dtype),
            V_r.astype(work_dtype, copy=False),
            args.old_memory_size if args.old_memory_size > 0 else half_win,
            np.random.default_rng(args.seed + end0),
            return_indices=True,
        )
        prev_opt2 = np.ascontiguousarray(V_selected[:, 1].copy())

    mean_align = float(np.linalg.norm((V_r @ V_r.T) @ V_exact[:, :1], "fro")) if V_r is not None else np.nan
    return {
        "policy": policy,
        "mode": "sliding" if sliding else "split_probe",
        "half_win": half_win,
        "rows": rows,
        "mean_align": mean_align,
        "mean_relerr_sval": rows[-1]["relerr_sval"] if rows else np.nan,
        "elapsed": time.time() - t0,
    }


def summarize_result(result):
    rows = result["rows"]
    final = rows[-1]
    label_counts = {}
    for row in rows:
        label_counts[row["selected_label"]] = label_counts.get(row["selected_label"], 0) + 1
    return {
        "mode": result["mode"],
        "policy": result["policy"],
        "half_win": result["half_win"],
        "steps": len(rows),
        "mean_align": result["mean_align"],
        "mean_relerr_sval": result["mean_relerr_sval"],
        "mean_cos1": float(np.nanmean([r["cos1"] for r in rows])),
        "mean_cos2": float(np.nanmean([r["cos2"] for r in rows])),
        "mean_exact_cos1": float(np.nanmean([r["exact_cos1"] for r in rows])),
        "mean_exact_cos2": float(np.nanmean([r["exact_cos2"] for r in rows])),
        "mean_winner_oracle_mass": float(np.nanmean([r.get("winner_oracle_mass", np.nan) for r in rows])),
        "final_cos": [final["cos1"], final["cos2"]],
        "final_exact_cos": [final["exact_cos1"], final["exact_cos2"]],
        "final_car_exact_cos": [final["car_exact_cos1"], final["car_exact_cos2"]],
        "final_oracle_proj_norm": [final["oracle_proj_norm1"], final["oracle_proj_norm2"]],
        "final_tail_mass": final["tail_mass"],
        "selected_label_counts": label_counts,
        "elapsed": result.get("elapsed"),
        "sec_per_step": (result.get("elapsed") / len(rows)) if rows and result.get("elapsed") is not None else None,
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path, summaries):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Half-window HM experiment\n")
        f.write("=========================\n\n")
        for s in summaries:
            f.write(
                f"mode={s['mode']} policy={s['policy']} half_win={s['half_win']} steps={s['steps']} "
                f"mean_align={s['mean_align']:.6f} mean_relerr_sval={s['mean_relerr_sval']:.8f} "
                f"mean_cos2={s['mean_cos2']:.6f} final_cos=[{s['final_cos'][0]:.6f} {s['final_cos'][1]:.6f}] "
                f"final_exact_cos=[{s['final_exact_cos'][0]:.6f} {s['final_exact_cos'][1]:.6f}] "
                f"final_car_exact_cos=[{s['final_car_exact_cos'][0]:.6f} {s['final_car_exact_cos'][1]:.6f}] "
                f"final_oracle_proj_norm=[{s['final_oracle_proj_norm'][0]:.6f} {s['final_oracle_proj_norm'][1]:.6f}] "
                f"final_tail_mass={s['final_tail_mass']:.6f} "
                f"labels={json.dumps(s['selected_label_counts'], sort_keys=True)}\n"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="mixed-tail-sharp")
    parser.add_argument("--half-win", type=int, default=16)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[
            "combined",
            "half_hmean",
            "half_hmean_guard",
            "future_hmean_online",
            "future_hmean_triplet_online",
            "future_hmean_nested_online",
            "future_hmean_pairwise_online",
            "future_hmean_weighted_online",
            "future_hmean_online_no_phi",
            "future_hmean_triplet_online_no_phi",
            "future_hmean_nested_online_no_phi",
            "future_hmean_pairwise_online_no_phi",
            "future_hmean_weighted_online_no_phi",
            "future_hmean_online_joint",
        ],
    )
    parser.add_argument("--rsk-variant", choices=("S1", "S2", "S3", "S4", "S5", "S6", "S6_GM", "D0", "S6_OP"), default="S4",
                        help="Variant for the future_hmean_r_sk_g policy.")
    parser.add_argument("--rsk-alpha", type=float, default=1.0)
    parser.add_argument("--rsk-beta", type=float, default=2.0)
    parser.add_argument("--rsk-gamma", type=float, default=1.0)
    parser.add_argument("--rsk-no-deflate", action="store_true",
                        help="Skip the V_default[:,0] deflation in S6 streaming wiring (P2').")
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-rows", action="store_true", default=True)
    parser.add_argument("--row-shuffle-seed", type=int, default=0)
    parser.add_argument("--old-memory-size", type=int, default=32)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--q0", type=int, default=8)
    parser.add_argument("--qmax", type=int, default=48)
    parser.add_argument("--krylov-depth", type=int, default=2)
    parser.add_argument("--residual-tol", type=float, default=0.01)
    parser.add_argument("--expansion-maxit", type=int, default=8)
    parser.add_argument("--num-restarts", type=int, default=3)
    parser.add_argument("--maxit", type=int, default=120)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--post-expansion-maxit", type=int, default=80)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--patience-rel-tol", type=float, default=1e-5)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--online-include-oracle",
        action="store_true",
        default=False,
        help=(
            "Include the oracle-derived q2_vs_q1oracle candidate in the online "
            "candidate pool. Default OFF (value-only). When ON the resulting "
            "future_hmean_online numbers are an oracle-aware UPPER BOUND, not a "
            "value-only baseline (INFRA-10)."
        ),
    )
    parser.add_argument("--r-sig", type=int, default=2)
    parser.add_argument("--alpha-sig", type=float, default=0.003)
    parser.add_argument("--alpha-tail", type=float, default=0.0145)
    parser.add_argument("--tail-scale", type=float, default=0.99)
    parser.add_argument("--sigma1", type=float, default=0.991)
    parser.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    parser.add_argument("--json-out", default="summary/half_window_sliding_hmean_experiment.json")
    parser.add_argument("--csv-out", default="summary/half_window_sliding_hmean_experiment.csv")
    parser.add_argument("--text-out", default="summary/half_window_sliding_hmean_experiment.txt")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    t0 = time.time()
    A, V_exact, _, sigma1 = probe.generate_matrix_input(
        matrix=args.matrix,
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

    results = []
    for policy in args.policies:
        split_result = run_pair_stream(A, V_exact, sigma1, args, policy, args.half_win, sliding=False)
        sliding_result = run_pair_stream(A, V_exact, sigma1, args, policy, args.half_win, sliding=True)
        results.extend([split_result, sliding_result])
        for result in (split_result, sliding_result):
            print(
                f"half_window_run mode={result['mode']} policy={policy} half_win={args.half_win} "
                f"mean_align={result['mean_align']:.6f} mean_relerr_sval={result['mean_relerr_sval']:.8f}"
            )

    summaries = [summarize_result(r) for r in results]
    all_rows = [row for result in results for row in result["rows"]]
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump({"summaries": summaries, "results": results}, f, indent=2, sort_keys=True)
    write_csv(args.csv_out, all_rows)
    write_text(args.text_out, summaries)
    print(
        f"wrote_json={args.json_out} wrote_csv={args.csv_out} wrote_text={args.text_out} "
        f"elapsed={time.time() - t0:.3f}"
    )


if __name__ == "__main__":
    main()
