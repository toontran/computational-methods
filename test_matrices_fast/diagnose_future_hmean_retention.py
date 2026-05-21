import argparse
import csv
import json
from collections import Counter, defaultdict

import numpy as np

import cex_restricted_space_probe as probe
import half_window_sliding_hmean_experiment as hm
from second_slot_tail_bias_diagnostic import make_state, raw_oracle_columns


def rowspace_mass(A, v):
    A = np.asarray(A, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if A.size == 0:
        return np.nan
    _, s, Vh = np.linalg.svd(A, full_matrices=False)
    if s.size == 0:
        return np.nan
    keep = s > max(float(s[0]) * 1e-10, 1e-30)
    if not np.any(keep):
        return 0.0
    Q = Vh[keep, :].T
    return float(np.linalg.norm(Q @ (Q.T @ v)) ** 2 / max(float(np.dot(v, v)), 1e-30))


def rowspace_project(A, v):
    A = np.asarray(A, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if A.size == 0:
        return None
    _, s, Vh = np.linalg.svd(A, full_matrices=False)
    if s.size == 0:
        return None
    keep = s > max(float(s[0]) * 1e-10, 1e-30)
    if not np.any(keep):
        return None
    Q = Vh[keep, :].T
    return Q @ (Q.T @ v)


def future_hmean_abs_score(A_half1, A_half2, v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    gain1 = float(np.linalg.norm(np.asarray(A_half1, dtype=np.float64) @ v) ** 2)
    gain2 = float(np.linalg.norm(np.asarray(A_half2, dtype=np.float64) @ v) ** 2)
    relH1 = hm.response_shape(A_half1, v)["relH"]
    relH1 = max(float(relH1), 0.0) if np.isfinite(relH1) else 0.0
    return hm.hmean(gain1, gain2) * relH1


def projected_future_score(A_half1, A_half2, union_block, v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    p = rowspace_project(union_block, v)
    if p is None:
        return np.nan, np.nan, np.nan
    p_norm = float(np.linalg.norm(p))
    if p_norm <= 1e-30:
        return np.nan, 0.0, np.nan
    score_v = future_hmean_abs_score(A_half1, A_half2, v)
    score_u = future_hmean_abs_score(A_half1, A_half2, p / p_norm)
    ratio = score_u / max(score_v, 1e-300) if np.isfinite(score_v) and score_v > 0.0 else np.nan
    return score_u, p_norm * p_norm, ratio


def state_space_mass(state, v):
    if state is None:
        return np.nan
    Q = np.asarray(state["V"], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(Q @ (Q.T @ v)) ** 2 / max(float(np.dot(v, v)), 1e-30))


def safe_rec(records, label, key):
    rec = records.get(label)
    if rec is None:
        return np.nan
    return rec.get(key, np.nan)


def run_matrix(args, matrix):
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
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
    V_exact = np.asarray(V_exact, dtype=np.float64)

    state = None
    old_row_memory = None
    prev_opt2 = None
    rank = int(args.rank)
    half_win = int(args.half_win)
    rows = []

    for block_id, start0 in enumerate(range(0, A.shape[0] - half_win, half_win), start=1):
        if args.max_pairs is not None and block_id > args.max_pairs:
            break
        mid0 = start0 + half_win
        end0 = min(mid0 + half_win, A.shape[0])
        if end0 - mid0 < half_win:
            break

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

        V_score, _, _, _, _ = probe.entropy_iter_basis_forget(
            M_gain=M_gain,
            active_r=rank,
            rows_ref=A.shape[0],
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
            combined_rank=None,
            patience=args.patience,
            patience_rel_tol=args.patience_rel_tol,
        )
        V_default = np.ascontiguousarray(np.asarray(V_score[:, :rank], dtype=np.float64))

        Q_oracle, raw_oracle = raw_oracle_columns(M_gain, V_exact, rank, np.float64)
        candidates = hm.build_candidates(
            V_default, Q_oracle, raw_oracle, M_gain, A_half1, A_half2, prev_opt2=prev_opt2
        )
        candidates = {k: candidates.get(k) for k in hm.ONLINE_POOL}
        prior_seen = 0 if state is None else int(state["rows_seen"])
        records = hm.score_half_candidates(
            candidates,
            A_half1,
            A_half2,
            A_sketch_prior=M_sketch,
            hm_weights=(prior_seen, A_half1.shape[0], A_half2.shape[0]),
        )
        chosen_label, chosen_v2 = hm.choose_second_slot(
            "future_hmean_online", records, candidates.get("opt2")
        )

        svd_frame_used = False
        if chosen_v2 is not None:
            V_svd = hm.rank2_svd_frame(V_default[:, 0], chosen_v2, M_gain, rank=rank)
            if V_svd is not None and V_svd.shape[1] >= rank:
                V_selected = np.ascontiguousarray(V_svd[:, :rank])
                svd_frame_used = True
            else:
                V_selected = V_default.copy()
                V_selected[:, 1] = chosen_v2
                V_selected = probe.orthonormalize_columns(V_selected[:, :rank], dtype=np.float64)[:, :rank]
        else:
            V_selected = probe.orthonormalize_columns(V_default[:, :rank], dtype=np.float64)[:, :rank]

        chosen = records.get(chosen_label)
        selected_vec = chosen["vec"] if chosen is not None else V_selected[:, 1]
        selected_frame_v2 = V_selected[:, 1]
        union_block = np.vstack([A_half1, A_half2]).astype(work_dtype, copy=False)
        selected_abs_score = future_hmean_abs_score(A_half1, A_half2, selected_vec)
        selected_proj_abs_score, selected_proj_mass, selected_proj_score_ratio = projected_future_score(
            A_half1, A_half2, union_block, selected_vec
        )
        frame_v2_abs_score = future_hmean_abs_score(A_half1, A_half2, selected_frame_v2)
        frame_v2_proj_abs_score, frame_v2_proj_mass, frame_v2_proj_score_ratio = projected_future_score(
            A_half1, A_half2, union_block, selected_frame_v2
        )
        prev_align = np.nan
        frame_prev_align = np.nan
        if prev_opt2 is not None:
            prev_align = float(abs(np.dot(selected_vec, prev_opt2)) ** 2)
            frame_prev_align = float(abs(np.dot(selected_frame_v2, prev_opt2)) ** 2)

        candidate_pairs = [
            (rec.get("sketch_gain_share"), rec.get("gain2_share"))
            for rec in records.values()
            if rec is not None
            and np.isfinite(rec.get("sketch_gain_share", np.nan))
            and np.isfinite(rec.get("gain2_share", np.nan))
        ]
        if len(candidate_pairs) >= 2:
            arr = np.asarray(candidate_pairs, dtype=np.float64)
            corr = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
        else:
            corr = np.nan

        exact_cos = probe.subspace_principal_cosines(V_selected, V_exact[:, :rank])
        rows.append(
            {
                "matrix": matrix,
                "block": block_id,
                "rows_seen": mid0,
                "selected_label": chosen_label,
                "svd_frame_used": int(svd_frame_used),
                "selected_sketch_share": safe_rec(records, chosen_label, "sketch_gain_share"),
                "selected_gain1_share": safe_rec(records, chosen_label, "gain1_share"),
                "selected_gain2_share": safe_rec(records, chosen_label, "gain2_share"),
                "selected_obj_future_online": safe_rec(records, chosen_label, "obj_future_online"),
                "selected_prev_align_candidate": prev_align,
                "selected_prev_align_after_svd": frame_prev_align,
                "selected_sketch_space_mass": state_space_mass(state, selected_vec),
                "selected_current_rowspace_mass": rowspace_mass(A_half1, selected_vec),
                "selected_future_rowspace_mass": rowspace_mass(A_half2, selected_vec),
                "selected_union_rowspace_mass": rowspace_mass(union_block, selected_vec),
                "selected_abs_future_hmean_score": selected_abs_score,
                "selected_projected_abs_future_hmean_score": selected_proj_abs_score,
                "selected_projection_mass": selected_proj_mass,
                "selected_projected_score_ratio": selected_proj_score_ratio,
                "frame_v2_sketch_space_mass": state_space_mass(state, selected_frame_v2),
                "frame_v2_current_rowspace_mass": rowspace_mass(A_half1, selected_frame_v2),
                "frame_v2_future_rowspace_mass": rowspace_mass(A_half2, selected_frame_v2),
                "frame_v2_union_rowspace_mass": rowspace_mass(union_block, selected_frame_v2),
                "frame_v2_abs_future_hmean_score": frame_v2_abs_score,
                "frame_v2_projected_abs_future_hmean_score": frame_v2_proj_abs_score,
                "frame_v2_projection_mass": frame_v2_proj_mass,
                "frame_v2_projected_score_ratio": frame_v2_proj_score_ratio,
                "default_v1_current_rowspace_mass": rowspace_mass(A_half1, V_default[:, 0]),
                "default_v1_future_rowspace_mass": rowspace_mass(A_half2, V_default[:, 0]),
                "default_v1_union_rowspace_mass": rowspace_mass(union_block, V_default[:, 0]),
                "default_v2_current_rowspace_mass": rowspace_mass(A_half1, V_default[:, 1]),
                "default_v2_future_rowspace_mass": rowspace_mass(A_half2, V_default[:, 1]),
                "default_v2_union_rowspace_mass": rowspace_mass(union_block, V_default[:, 1]),
                "prevopt2_candidate_exists": int(records.get("prev_opt2") is not None),
                "prevopt2_sketch_share": safe_rec(records, "prev_opt2", "sketch_gain_share"),
                "prevopt2_gain1_share": safe_rec(records, "prev_opt2", "gain1_share"),
                "prevopt2_gain2_share": safe_rec(records, "prev_opt2", "gain2_share"),
                "prevopt2_obj_future_online": safe_rec(records, "prev_opt2", "obj_future_online"),
                "candidate_sketch_future_corr": corr,
                "exact_cos1": float(exact_cos[0]) if len(exact_cos) > 0 else np.nan,
                "exact_cos2": float(exact_cos[1]) if len(exact_cos) > 1 else np.nan,
            }
        )

        score_selected = np.zeros(rank, dtype=float)
        H_selected = np.zeros(rank, dtype=float)
        for j in range(rank):
            score_selected[j], _, H_selected[j] = probe.score_full_vector_details_forget(
                M_gain,
                A_half1,
                V_selected[:, j],
                A.shape[0],
                state_prev=state,
                score_variant="combined",
                old_row_memory=old_row_memory,
            )
        state, V_r, _ = make_state(M_gain, V_selected, H_selected, score_selected, rows_seen)
        seen_for_memory = A[:mid0, :]
        old_row_memory, _ = probe.select_old_row_memory(
            np.asarray(seen_for_memory, dtype=work_dtype),
            V_r.astype(work_dtype, copy=False),
            args.old_memory_size if args.old_memory_size > 0 else half_win,
            np.random.default_rng(args.seed + end0),
            return_indices=True,
        )
        prev_opt2 = np.ascontiguousarray(V_selected[:, 1].copy())

    return rows


def summarize(rows):
    by_matrix = defaultdict(list)
    for row in rows:
        by_matrix[row["matrix"]].append(row)
    summaries = []
    for matrix, rs in sorted(by_matrix.items()):
        after_first = [r for r in rs if r["block"] > 1]
        label_counts = Counter(r["selected_label"] for r in rs)
        prev_selected = sum(1 for r in rs if r["selected_label"] == "prev_opt2")
        summaries.append(
            {
                "matrix": matrix,
                "blocks": len(rs),
                "selected_labels": dict(sorted(label_counts.items())),
                "prev_opt2_selected": prev_selected,
                "mean_selected_sketch_share": float(np.nanmean([r["selected_sketch_share"] for r in after_first])),
                "mean_selected_gain1_share": float(np.nanmean([r["selected_gain1_share"] for r in rs])),
                "mean_selected_gain2_share": float(np.nanmean([r["selected_gain2_share"] for r in rs])),
                "mean_selected_prev_align_candidate": float(np.nanmean([r["selected_prev_align_candidate"] for r in after_first])),
                "mean_selected_prev_align_after_svd": float(np.nanmean([r["selected_prev_align_after_svd"] for r in after_first])),
                "mean_selected_sketch_space_mass": float(np.nanmean([r["selected_sketch_space_mass"] for r in after_first])),
                "mean_selected_current_rowspace_mass": float(np.nanmean([r["selected_current_rowspace_mass"] for r in rs])),
                "mean_selected_future_rowspace_mass": float(np.nanmean([r["selected_future_rowspace_mass"] for r in rs])),
                "mean_selected_union_rowspace_mass": float(np.nanmean([r["selected_union_rowspace_mass"] for r in rs])),
                "mean_selected_projected_score_ratio": float(np.nanmean([r["selected_projected_score_ratio"] for r in rs])),
                "mean_selected_projection_mass": float(np.nanmean([r["selected_projection_mass"] for r in rs])),
                "mean_frame_v2_sketch_space_mass": float(np.nanmean([r["frame_v2_sketch_space_mass"] for r in after_first])),
                "mean_frame_v2_current_rowspace_mass": float(np.nanmean([r["frame_v2_current_rowspace_mass"] for r in rs])),
                "mean_frame_v2_future_rowspace_mass": float(np.nanmean([r["frame_v2_future_rowspace_mass"] for r in rs])),
                "mean_frame_v2_union_rowspace_mass": float(np.nanmean([r["frame_v2_union_rowspace_mass"] for r in rs])),
                "mean_frame_v2_projected_score_ratio": float(np.nanmean([r["frame_v2_projected_score_ratio"] for r in rs])),
                "mean_frame_v2_projection_mass": float(np.nanmean([r["frame_v2_projection_mass"] for r in rs])),
                "mean_default_v1_union_rowspace_mass": float(np.nanmean([r["default_v1_union_rowspace_mass"] for r in rs])),
                "mean_default_v2_union_rowspace_mass": float(np.nanmean([r["default_v2_union_rowspace_mass"] for r in rs])),
                "mean_prevopt2_gain1_share": float(np.nanmean([r["prevopt2_gain1_share"] for r in after_first])),
                "mean_prevopt2_gain2_share": float(np.nanmean([r["prevopt2_gain2_share"] for r in after_first])),
                "mean_candidate_sketch_future_corr": float(np.nanmean([r["candidate_sketch_future_corr"] for r in after_first])),
                "final_exact_cos2": rs[-1]["exact_cos2"] if rs else np.nan,
            }
        )
    return summaries


def write_report(path, summaries):
    with open(path, "w", encoding="utf-8") as f:
        f.write("future_hmean_online retention diagnostic\n")
        f.write("========================================\n\n")
        f.write("Question: why can the original online policy retain previous directions even though its HM score uses only A_w and A_{w+1}?\n\n")
        f.write("Mechanism under test:\n")
        f.write("  M_gain = [sketch; A_w] after block 1, so the sketch affects V_default, candidates, rank-2 SVD reframe, and next state.\n")
        f.write("  The selected second-slot candidate is still gated by HM(||A_w v||^2, ||A_{w+1} v||^2) * relH1.\n\n")
        fields = [
            "matrix",
            "blocks",
            "prev_opt2_selected",
            "mean_selected_sketch_share",
            "mean_selected_gain1_share",
            "mean_selected_gain2_share",
            "mean_selected_prev_align_candidate",
            "mean_selected_prev_align_after_svd",
            "mean_selected_sketch_space_mass",
            "mean_selected_current_rowspace_mass",
            "mean_selected_future_rowspace_mass",
            "mean_selected_union_rowspace_mass",
            "mean_selected_projected_score_ratio",
            "mean_selected_projection_mass",
            "mean_frame_v2_sketch_space_mass",
            "mean_frame_v2_current_rowspace_mass",
            "mean_frame_v2_future_rowspace_mass",
            "mean_frame_v2_union_rowspace_mass",
            "mean_frame_v2_projected_score_ratio",
            "mean_frame_v2_projection_mass",
            "mean_default_v1_union_rowspace_mass",
            "mean_default_v2_union_rowspace_mass",
            "mean_prevopt2_gain1_share",
            "mean_prevopt2_gain2_share",
            "mean_candidate_sketch_future_corr",
            "final_exact_cos2",
        ]
        f.write("Summary by matrix\n")
        f.write("-----------------\n")
        f.write(" ".join(f"{x:<34}" for x in fields) + "\n")
        for s in summaries:
            vals = []
            for field in fields:
                v = s[field]
                if isinstance(v, float):
                    vals.append(f"{v:<34.4f}")
                else:
                    vals.append(f"{str(v):<34}")
            f.write(" ".join(vals) + "\n")
        f.write("\nInterpretation guide\n")
        f.write("--------------------\n")
        f.write("High selected_sketch_space_mass with high selected_gain1/gain2 means a prior direction is retained only when it reappears in current/future evidence.\n")
        f.write("High selected_prev_align_after_svd relative to selected_prev_align_candidate means the rank-2 SVD reframe reintroduces carried memory after candidate selection.\n")
        f.write("Low/negative candidate_sketch_future_corr means direct sketch gain is not a reliable proxy for future gain, explaining why sketch-in-HM can over-bias selection.\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices", nargs="+", default=[
        "mixed-tail-sharp",
        "mixed-tail-balanced",
        "mixed-tail-soft",
        "diffuse-diffuse",
        "static-cex",
        "etf-basket-basis",
        "residual-spiky-shocks",
        "risk-residual-panel",
    ])
    parser.add_argument("--out-prefix", default="summary/future_hmean_retention_diagnostic")
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
    all_rows = []
    for matrix in args.matrices:
        rows = run_matrix(args, matrix)
        all_rows.extend(rows)
        print(f"done {matrix} blocks={len(rows)}")

    csv_path = args.out_prefix + ".csv"
    if all_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    summaries = summarize(all_rows)
    json_path = args.out_prefix + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summaries": summaries, "rows": all_rows}, f, indent=2, sort_keys=True)
    text_path = args.out_prefix + ".txt"
    write_report(text_path, summaries)
    print(f"wrote {csv_path} {json_path} {text_path}")


if __name__ == "__main__":
    main()
