import argparse
import csv
import json
import os
import time

import numpy as np

import cex_restricted_space_probe as probe
import half_window_sliding_hmean_experiment as hm
from future_hmean_optimizer_diagnostic import orth_basis_against, rowspace_basis
from hmean_combinations_optimizer_diagnostic import (
    candidate_denoms,
    combination_value_grad,
    optimize_combination_in_basis,
)
from second_slot_tail_bias_diagnostic import make_state, raw_oracle_columns


GRADIENT_POLICIES = {
    "gradient_triplet": "future_hmean_triplet_online",
    "gradient_nested": "future_hmean_nested_online",
    "gradient_weighted": "future_hmean_weighted_online",
}

RERANK_POLICIES = {
    "rerank_nested_oracle": "future_hmean_nested_online",
    "rerank_nested_no_oracle": "future_hmean_nested_online",
}

DEFAULT_POLICIES = [
    "combined",
    "rerank_nested_oracle",
    "rerank_nested_no_oracle",
    "gradient_triplet",
    "gradient_nested",
    "gradient_weighted",
]

TAIL_MATRICES = [
    "mixed-tail-sharp",
    "mixed-tail-balanced",
    "mixed-tail-soft",
    "diffuse-diffuse",
    "static-cex",
    "etf-basket-basis",
    "residual-spiky-shocks",
    "risk-residual-panel",
]

ORACLE_SEED_LABELS = {"q2_vs_q1oracle", "q2_raw_projected", "opt2_outside"}
ALLOWED_SEED_LABELS = {"opt2", "mgain_deflated_svd", "block_complement", "prev_opt2", "half2_complement"}


def orthonormalize_stack(cols, d):
    good = []
    for c in cols:
        if c is None:
            continue
        arr = np.asarray(c, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.size == 0:
            continue
        for j in range(arr.shape[1]):
            v = arr[:, j]
            if float(np.linalg.norm(v)) > 1e-12:
                good.append(v)
    if not good:
        return np.zeros((d, 0), dtype=np.float64)
    M = np.column_stack(good)
    Q, R = np.linalg.qr(M)
    diag = np.abs(np.diag(R))
    if diag.size == 0:
        return np.zeros((d, 0), dtype=np.float64)
    keep = diag > max(float(diag.max()) * 1e-10, 1e-30)
    return np.ascontiguousarray(Q[:, keep], dtype=np.float64)


def build_search_basis(A_cur, A_fut, A_sketch, v1):
    union = np.vstack([np.asarray(A_cur, dtype=np.float64), np.asarray(A_fut, dtype=np.float64)])
    q_union = rowspace_basis(union)
    q_sketch = rowspace_basis(A_sketch) if A_sketch is not None else None
    q_full = orthonormalize_stack([q_union, q_sketch], union.shape[1])
    return orth_basis_against(q_full, v1)


def relerr_svals(s_est, s_true, rank):
    s_est = np.asarray(s_est, dtype=np.float64).reshape(-1)[:rank]
    s_true = np.asarray(s_true, dtype=np.float64).reshape(-1)[:rank]
    if s_est.size == 0 or s_true.size == 0:
        return np.nan
    k = min(s_est.size, s_true.size)
    return float(np.linalg.norm(s_est[:k] - s_true[:k]) / max(float(np.linalg.norm(s_true[:k])), 1e-30))


def vector_mass(A, v):
    if A is None:
        return np.nan
    q = rowspace_basis(A)
    if q.size == 0:
        return 0.0
    vv = np.asarray(v, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(q @ (q.T @ vv)) ** 2 / max(float(np.dot(vv, vv)), 1e-30))


def candidate_seed_list(candidates, V_default, diag, include_oracle=False):
    seeds = [V_default[:, 1]]
    for label, vec in candidates.items():
        if vec is None:
            continue
        if include_oracle or label in ALLOWED_SEED_LABELS:
            seeds.append(vec)
    Vbasis = diag.get("Vbasis_final")
    if Vbasis is not None:
        Vbasis = np.asarray(Vbasis, dtype=np.float64)
        for j in range(min(Vbasis.shape[1], 8)):
            seeds.append(Vbasis[:, j])
    return seeds


def select_second_slot(policy, V_default, candidates, records):
    if policy == "combined":
        return "combined", V_default[:, 1], {}
    if policy == "rerank_nested_oracle":
        label, v = hm.choose_second_slot(RERANK_POLICIES[policy], records, candidates.get("opt2"))
        return label, v, {}
    if policy == "rerank_nested_no_oracle":
        no_oracle = {k: v for k, v in candidates.items() if k not in ORACLE_SEED_LABELS}
        recs = {k: records.get(k) for k in no_oracle}
        label, v = hm.choose_second_slot(RERANK_POLICIES[policy], recs, no_oracle.get("opt2"))
        return label, v, {}
    raise ValueError(f"Unsupported rerank policy: {policy}")


def run_stream(A, V_exact, sigma_true, args, matrix, policy):
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    n = A.shape[0]
    rank = int(args.rank)
    half_win = int(args.half_win)
    state = None
    old_row_memory = None
    prev_opt2 = None
    rows = []
    t0 = time.time()

    for block_id, start0 in enumerate(range(0, n - half_win, half_win), start=1):
        if args.max_pairs is not None and block_id > args.max_pairs:
            break
        mid0 = start0 + half_win
        end0 = min(mid0 + half_win, n)
        if end0 - mid0 < half_win:
            break
        A_cur = np.asarray(A[start0:mid0, :], dtype=work_dtype)
        A_fut = np.asarray(A[mid0:end0, :], dtype=work_dtype)

        if state is None:
            A_sketch = None
            M_gain = A_cur
            V_init = probe.row_norm_seed(A_cur, rank)
            rows_seen = A_cur.shape[0]
        else:
            B_top = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
            A_sketch = B_top
            M_gain = np.vstack([B_top, A_cur]).astype(work_dtype, copy=False)
            V_init = probe.row_norm_seed(A_cur, rank)
            rows_seen = state["rows_seen"] + A_cur.shape[0]

        V_score, _, _, _, diag = probe.entropy_iter_basis_forget(
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
        v1 = V_default[:, 0]

        Q_oracle, raw_oracle = raw_oracle_columns(M_gain, V_exact, rank, np.float64)
        candidates_all = hm.build_candidates(
            V_default, Q_oracle, raw_oracle, M_gain, A_cur, A_fut, prev_opt2=prev_opt2
        )
        oracle_rerank_pool = {k: candidates_all.get(k) for k in hm.ONLINE_POOL}
        gradient_pool = {k: candidates_all.get(k) for k in ALLOWED_SEED_LABELS}
        prior_seen = 0 if state is None else int(state["rows_seen"])
        weights = (prior_seen, A_cur.shape[0], A_fut.shape[0])
        gradient_denoms, _ = candidate_denoms(gradient_pool, A_cur, A_fut, A_sketch)
        rerank_denoms, rerank_records = candidate_denoms(oracle_rerank_pool, A_cur, A_fut, A_sketch)

        chosen_label = ""
        ascent = {}
        policy_score_at_v2 = np.nan
        policy_parts = {}
        if policy in GRADIENT_POLICIES:
            grad_policy = GRADIENT_POLICIES[policy]
            B = build_search_basis(A_cur, A_fut, A_sketch, v1)
            seeds = candidate_seed_list(gradient_pool, V_default, diag, include_oracle=False)
            best = optimize_combination_in_basis(
                grad_policy,
                A_cur,
                A_fut,
                A_sketch,
                gradient_denoms,
                weights,
                B,
                seeds,
                np.random.default_rng(args.seed + 1009 * block_id + 37 * list(GRADIENT_POLICIES).index(policy)),
                maxit=args.ascent_maxit,
                tol=args.ascent_tol,
                random_starts=args.ascent_random_starts,
            )
            if best is None:
                chosen_v2 = V_default[:, 1]
                chosen_label = "gradient_fallback_opt2"
                ascent = {"stop": {"reason": "no_basis", "iters": 0, "grad_norm": np.nan}, "score": np.nan}
            else:
                chosen_v2 = best["vec"]
                chosen_label = policy
                ascent = best
            policy_score_at_v2, _, policy_parts = combination_value_grad(
                grad_policy, A_cur, A_fut, A_sketch, gradient_denoms, weights, chosen_v2
            )
        else:
            if policy == "rerank_nested_no_oracle":
                rerank_pool = {k: v for k, v in {**oracle_rerank_pool, "half2_complement": candidates_all.get("half2_complement")}.items() if k not in ORACLE_SEED_LABELS}
                rerank_denoms, rerank_records = candidate_denoms(rerank_pool, A_cur, A_fut, A_sketch)
            else:
                rerank_pool = oracle_rerank_pool
            chosen_label, chosen_v2, _ = select_second_slot(policy, V_default, rerank_pool, rerank_records)
            if policy in RERANK_POLICIES:
                policy_score_at_v2, _, policy_parts = combination_value_grad(
                    RERANK_POLICIES[policy], A_cur, A_fut, A_sketch, rerank_denoms, weights, chosen_v2
                )

        V_selected = np.ascontiguousarray(V_default.copy())
        if chosen_v2 is not None:
            V_selected[:, 1] = chosen_v2
        if args.rank2_svd_reframe and chosen_v2 is not None and policy in GRADIENT_POLICIES:
            V_svd = hm.rank2_svd_frame(v1, chosen_v2, M_gain, rank=rank)
            if V_svd is not None and V_svd.shape[1] >= rank:
                V_selected = np.ascontiguousarray(V_svd[:, :rank])
            else:
                V_selected = probe.orthonormalize_columns(V_selected[:, :rank], dtype=np.float64)[:, :rank]
        else:
            V_selected = probe.orthonormalize_columns(V_selected[:, :rank], dtype=np.float64)[:, :rank]

        score_selected = np.zeros(rank, dtype=float)
        H_selected = np.zeros(rank, dtype=float)
        for j in range(rank):
            score_selected[j], _, H_selected[j] = probe.score_full_vector_details_forget(
                M_gain,
                A_cur,
                V_selected[:, j],
                n,
                state_prev=state,
                score_variant="combined",
                old_row_memory=old_row_memory,
            )

        state, V_r, _ = make_state(M_gain, V_selected, H_selected, score_selected, rows_seen)
        v2 = np.asarray(chosen_v2 if chosen_v2 is not None else V_selected[:, 1], dtype=np.float64)
        state_V = np.asarray(state["V"], dtype=np.float64)
        exact_cos = probe.subspace_principal_cosines(V_selected, V_exact[:, :rank])
        car_exact_cos = probe.subspace_principal_cosines(state_V[:, :rank], V_exact[:, :rank])
        tail_mass = hm.frame_tail_mass(state_V[:, :rank], V_exact, rank)
        relerr = relerr_svals(state["s"], sigma_true, rank)
        row = {
            "matrix": matrix,
            "policy": policy,
            "block": block_id,
            "rows_seen": mid0,
            "selected_label": chosen_label,
            "v2_align_exact": float(abs(np.dot(v2, V_exact[:, 1])) ** 2) if V_exact.shape[1] > 1 else np.nan,
            "selected_exact_cos1": float(exact_cos[0]) if len(exact_cos) > 0 else np.nan,
            "selected_exact_cos2": float(exact_cos[1]) if len(exact_cos) > 1 else np.nan,
            "state_V0_align_exact": float(abs(np.dot(state_V[:, 0], V_exact[:, 0])) ** 2),
            "state_V1_align_exact": float(abs(np.dot(state_V[:, 1], V_exact[:, 1])) ** 2) if state_V.shape[1] > 1 else np.nan,
            "state_exact_cos1": float(car_exact_cos[0]) if len(car_exact_cos) > 0 else np.nan,
            "state_exact_cos2": float(car_exact_cos[1]) if len(car_exact_cos) > 1 else np.nan,
            "relerr_sval": relerr,
            "tail_mass": tail_mass,
            "gradient_ascent_iterations": ascent.get("stop", {}).get("iters", np.nan),
            "gradient_ascent_grad_norm": ascent.get("stop", {}).get("grad_norm", np.nan),
            "gradient_ascent_stop": ascent.get("stop", {}).get("reason", ""),
            "policy_score_at_v2": policy_score_at_v2,
            "gain1_share_at_v2": policy_parts.get("gain1_share", np.nan),
            "gain2_share_at_v2": policy_parts.get("gain2_share", np.nan),
            "sketch_share_at_v2": policy_parts.get("sketch_share", np.nan),
            "relH1_at_v2": policy_parts.get("relH1", np.nan),
            "v2_state_mass": vector_mass(A_sketch, v2),
            "v2_current_mass": vector_mass(A_cur, v2),
            "v2_future_mass": vector_mass(A_fut, v2),
            "optimizer_subspace_dim_v2": int(np.asarray(diag.get("subspace_dims", [np.nan]))[1])
            if len(np.asarray(diag.get("subspace_dims", []))) > 1 else np.nan,
            "optimizer_grad_perp_v2": float(np.asarray(diag.get("grad_perp_ratio", [np.nan]))[1])
            if len(np.asarray(diag.get("grad_perp_ratio", []))) > 1 else np.nan,
        }
        rows.append(row)

        old_row_memory, _ = probe.select_old_row_memory(
            np.asarray(A[:mid0, :], dtype=work_dtype),
            V_r.astype(work_dtype, copy=False),
            args.old_memory_size if args.old_memory_size > 0 else half_win,
            np.random.default_rng(args.seed + end0),
            return_indices=True,
        )
        prev_opt2 = np.ascontiguousarray(V_selected[:, 1].copy())

    return {"matrix": matrix, "policy": policy, "rows": rows, "elapsed": time.time() - t0}


def summarize_result(result):
    rows = result["rows"]
    if not rows:
        return {"matrix": result["matrix"], "policy": result["policy"], "steps": 0}
    final = rows[-1]
    grad_rows = [r for r in rows if r["policy"].startswith("gradient_")]
    grad_good = [
        r for r in grad_rows
        if np.isfinite(float(r["gradient_ascent_grad_norm"])) and float(r["gradient_ascent_grad_norm"]) <= 1e-6
    ]
    return {
        "matrix": result["matrix"],
        "policy": result["policy"],
        "steps": len(rows),
        "mean_v2_align_exact": float(np.nanmean([r["v2_align_exact"] for r in rows])),
        "mean_state_V1_align_exact": float(np.nanmean([r["state_V1_align_exact"] for r in rows])),
        "final_state_V1_align_exact": final["state_V1_align_exact"],
        "mean_state_exact_cos2": float(np.nanmean([r["state_exact_cos2"] for r in rows])),
        "final_state_exact_cos2": final["state_exact_cos2"],
        "mean_relerr_sval": float(np.nanmean([r["relerr_sval"] for r in rows])),
        "final_relerr_sval": final["relerr_sval"],
        "final_tail_mass": final["tail_mass"],
        "mean_policy_score_at_v2": float(np.nanmean([r["policy_score_at_v2"] for r in rows])),
        "grad_tol_fraction": (len(grad_good) / len(grad_rows)) if grad_rows else np.nan,
        "elapsed": result["elapsed"],
        "sec_per_step": result["elapsed"] / max(len(rows), 1),
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_synthesis(path, summaries, args):
    fields = [
        "matrix",
        "policy",
        "steps",
        "mean_v2_align_exact",
        "mean_state_V1_align_exact",
        "final_state_V1_align_exact",
        "mean_state_exact_cos2",
        "final_state_exact_cos2",
        "mean_relerr_sval",
        "final_tail_mass",
        "grad_tol_fraction",
        "sec_per_step",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("HM gradient-ascent streaming synthesis\n")
        f.write("======================================\n\n")
        f.write(
            f"n={args.n} half_win={args.half_win} rank={args.rank} "
            f"ascent_maxit={args.ascent_maxit} ascent_random_starts={args.ascent_random_starts} "
            f"rank2_svd_reframe={int(args.rank2_svd_reframe)}\n\n"
        )
        f.write("Summary\n")
        f.write("-------\n")
        f.write(" ".join(f"{x:<32}" for x in fields) + "\n")
        for s in summaries:
            vals = []
            for field in fields:
                v = s.get(field, np.nan)
                vals.append(f"{v:<32.6f}" if isinstance(v, float) and np.isfinite(v) else f"{str(v):<32}")
            f.write(" ".join(vals) + "\n")
        f.write("\nNotes\n")
        f.write("-----\n")
        f.write("gradient_* policies run HM-combination gradient ascent for v2 and feed [v1,v2] into make_state.\n")
        f.write("rerank_nested_oracle is the existing oracle-leaking rerank reference; rerank_nested_no_oracle excludes q2/q2_raw/opt2_outside from candidate selection.\n")
        f.write("combined is the combined-score-only baseline.\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices", nargs="+", default=TAIL_MATRICES)
    parser.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    parser.add_argument("--out-dir", default="summary/hmean_gradient_ascent_streaming")
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--half-win", type=int, default=32)
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
    parser.add_argument("--ascent-maxit", type=int, default=80)
    parser.add_argument("--ascent-tol", type=float, default=1e-9)
    parser.add_argument("--ascent-random-starts", type=int, default=8)
    parser.add_argument("--rank2-svd-reframe", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--r-sig", type=int, default=2)
    parser.add_argument("--alpha-sig", type=float, default=0.003)
    parser.add_argument("--alpha-tail", type=float, default=0.0145)
    parser.add_argument("--tail-scale", type=float, default=0.99)
    parser.add_argument("--sigma1", type=float, default=0.991)
    parser.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    all_summaries = []
    all_rows = []
    t0 = time.time()
    for matrix in args.matrices:
        A, V_exact, svec, _ = probe.generate_matrix_input(
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
        sigma_true = np.asarray(svec, dtype=np.float64)[: args.rank]
        for policy in args.policies:
            result = run_stream(A, V_exact, sigma_true, args, matrix, policy)
            rows = result["rows"]
            summary = summarize_result(result)
            all_summaries.append(summary)
            all_rows.extend(rows)
            base = os.path.join(args.out_dir, f"{matrix}_{policy}")
            write_csv(base + ".csv", rows)
            with open(base + ".json", "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "rows": rows}, f, indent=2, sort_keys=True)
            print(f"done matrix={matrix} policy={policy} steps={len(rows)} elapsed={result['elapsed']:.3f}")

    write_csv(os.path.join(args.out_dir, "all_rows.csv"), all_rows)
    with open(os.path.join(args.out_dir, "summaries.json"), "w", encoding="utf-8") as f:
        json.dump({"summaries": all_summaries}, f, indent=2, sort_keys=True)
    write_synthesis(os.path.join(args.out_dir, "synthesis.txt"), all_summaries, args)
    print(f"wrote {args.out_dir} elapsed={time.time() - t0:.3f}")


if __name__ == "__main__":
    main()
