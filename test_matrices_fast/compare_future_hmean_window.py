import argparse
import csv
import json
import time

import numpy as np

import cex_restricted_space_probe as probe
from mixed_tail_sharp_objective_probe import (
    LABELS,
    block_svd_complement,
    component,
    enrich_records,
    future_gain,
    normed,
    outside_component,
)
from second_slot_tail_bias_diagnostic import make_state, orth_against, raw_oracle_columns, svd_complement


def fmt(vals, precision=6):
    arr = np.asarray(vals, dtype=float).reshape(-1)
    if arr.size == 0:
        return ""
    return " ".join(f"{x:.{precision}f}" for x in arr)


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


def build_candidates(V_selected, Q_oracle, raw_oracle, M_gain, A_block):
    v1 = np.asarray(V_selected[:, :1], dtype=np.float64)
    opt2 = normed(V_selected[:, 1]) if V_selected.shape[1] > 1 else None
    q2_raw = None
    if len(raw_oracle) > 1:
        q2_raw = normed(raw_oracle[1])
    elif Q_oracle.shape[1] > 1:
        q2_raw = normed(Q_oracle[:, 1])
    q2_qr = orth_against(q2_raw, Q_oracle[:, :1]) if q2_raw is not None else None
    block_comp = block_svd_complement(A_block, v1)
    mgain_deflated = svd_complement(M_gain, v1)
    opt2_out = outside_component(opt2, Q_oracle)
    return {
        "opt2": opt2,
        "q2_vs_q1oracle": q2_qr,
        "q2_raw_projected": q2_raw,
        "block_complement": block_comp,
        "mgain_deflated_svd": mgain_deflated,
        "opt2_outside": opt2_out,
    }


def score_candidates(candidates, A, block_idx, win, M_gain, A_block, n, state, old_row_memory, future_horizon):
    records = {}
    for label, v in candidates.items():
        rec = component(v, M_gain, A_block, n, state, old_row_memory)
        if rec is None:
            records[label] = None
            continue
        rec["future_gain"] = future_gain(A, block_idx, win, v, future_horizon)
        records[label] = rec

    q2_qr = candidates.get("q2_vs_q1oracle")
    block_comp = candidates.get("block_complement")
    mgain_deflated = candidates.get("mgain_deflated_svd")
    for rec in [r for r in records.values() if r is not None]:
        rec["oracle_qr_align2"] = np.nan if q2_qr is None else float(np.dot(rec["vec"], q2_qr) ** 2)
        rec["oracle_mass"] = np.nan
        rec["complement_align2"] = np.nan if block_comp is None else float(np.dot(rec["vec"], block_comp) ** 2)
        rec["mgain_svd_align2"] = np.nan if mgain_deflated is None else float(np.dot(rec["vec"], mgain_deflated) ** 2)
    enrich_records(records)
    return records


def pick_second_slot(policy, records, fallback):
    if policy == "combined":
        return "opt2", fallback
    vals = {
        label: rec["obj_future_hmean"]
        for label, rec in records.items()
        if rec is not None and np.isfinite(rec.get("obj_future_hmean", np.nan))
    }
    if not vals:
        return "opt2", fallback
    label = max(vals, key=vals.get)
    chosen = records[label]["vec"]
    return label, chosen


def rel_err_sval(s_est, sigma1):
    top_sval_est = float(np.asarray(s_est, dtype=float).reshape(-1)[0])
    return abs(top_sval_est - float(sigma1)) / max(float(sigma1), 1e-30)


def common_checkpoint_rows(n):
    checkpoints = [64, 128, 192, 256, 320, 384, 512, 640, 768, 832, 896, 960, n]
    return [x for x in checkpoints if x <= n]


def run_trajectory(A, V_exact, sigma1, policy, win, args):
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    n = A.shape[0]
    rank = int(args.rank)
    state = None
    old_row_memory = None
    V_r = None
    prev_opt2 = None
    prev_selected = None
    prev_carried = None
    rows = []
    t0 = time.time()

    for block_idx, start0 in enumerate(range(0, n, win), start=1):
        end0 = min(start0 + win, n)
        A_block = np.asarray(A[start0:end0, :], dtype=work_dtype)
        if state is None:
            M_gain = A_block
            V_init = probe.row_norm_seed(A_block, rank)
            rows_seen = A_block.shape[0]
        else:
            B_top = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
            M_gain = np.vstack([B_top, A_block]).astype(work_dtype, copy=False)
            V_init = probe.row_norm_seed(A_block, rank)
            rows_seen = state["rows_seen"] + A_block.shape[0]

        V_score, _, H_score, score_score, _ = probe.entropy_iter_basis_forget(
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
        )

        Q_oracle, raw_oracle = raw_oracle_columns(M_gain, V_exact, rank, np.float64)
        V_default = np.ascontiguousarray(np.asarray(V_score[:, :rank], dtype=np.float64))
        candidates = build_candidates(V_default, Q_oracle, raw_oracle, M_gain, A_block)
        records = score_candidates(
            candidates, A, block_idx, win, M_gain, A_block, n, state, old_row_memory, args.future_horizon
        )
        chosen_label, chosen_v2 = pick_second_slot(policy, records, candidates["opt2"])

        V_selected = np.ascontiguousarray(V_default.copy())
        if chosen_v2 is not None:
            V_selected[:, 1] = chosen_v2
        V_selected = probe.orthonormalize_columns(V_selected[:, :rank], dtype=np.float64)[:, :rank]

        score_selected = np.zeros(rank, dtype=float)
        H_selected = np.zeros(rank, dtype=float)
        for j in range(rank):
            score_selected[j], _, H_selected[j] = probe.score_full_vector_details_forget(
                M_gain,
                A_block,
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
        carried_tail_mass = frame_tail_mass(V_carried, V_exact, rank)
        final_relerr = rel_err_sval(s_new[:rank], sigma1)

        survive_subspace = np.nan
        survive_v2 = np.nan
        prev2_score_ratio = np.nan
        if prev_opt2 is not None:
            survive_subspace = float(np.linalg.norm(V_selected @ (V_selected.T @ prev_opt2)))
            survive_v2 = abs(float(V_selected[:, 1] @ prev_opt2))
            prev_score = probe.combined_score_component_details(
                M_gain, A_block, prev_opt2, n, state_prev=state, old_row_memory=old_row_memory
            )["score_total"]
            curr_score = probe.combined_score_component_details(
                M_gain, A_block, V_selected[:, 1], n, state_prev=state, old_row_memory=old_row_memory
            )["score_total"]
            prev2_score_ratio = float(prev_score / max(curr_score, 1e-30))

        prev_sel_vs_curr = np.nan
        prev_car_vs_curr = np.nan
        if prev_selected is not None:
            prev_sel_vs_curr = float(np.linalg.norm(V_selected @ (V_selected.T @ prev_selected[:, 1])))
        if prev_carried is not None:
            prev_car_vs_curr = float(np.linalg.norm(V_selected @ (V_selected.T @ prev_carried[:, 1])))

        selected_rec = records.get(chosen_label)
        opt2_rec = records.get("opt2")
        rows.append(
            {
                "policy": policy,
                "win": win,
                "block": block_idx,
                "rows_seen": rows_seen,
                "selected_label": chosen_label,
                "selected_future_hmean": np.nan if selected_rec is None else selected_rec["obj_future_hmean"],
                "selected_full_score": np.nan if selected_rec is None else selected_rec["full_score"],
                "selected_block_score": np.nan if selected_rec is None else selected_rec["block_score"],
                "selected_oracle_mass": np.nan if selected_rec is None else selected_rec["oracle_mass"],
                "selected_oracle_qr_align2": np.nan if selected_rec is None else selected_rec["oracle_qr_align2"],
                "opt2_future_hmean": np.nan if opt2_rec is None else opt2_rec["obj_future_hmean"],
                "opt2_full_score": np.nan if opt2_rec is None else opt2_rec["full_score"],
                "future_hmean_gap_vs_opt2": (
                    np.nan
                    if selected_rec is None or opt2_rec is None
                    else selected_rec["obj_future_hmean"] - opt2_rec["obj_future_hmean"]
                ),
                "full_score_gap_vs_opt2": (
                    np.nan if selected_rec is None or opt2_rec is None else selected_rec["full_score"] - opt2_rec["full_score"]
                ),
                "cos1": float(cos[0]) if len(cos) > 0 else np.nan,
                "cos2": float(cos[1]) if len(cos) > 1 else np.nan,
                "exact_cos1": float(exact_cos[0]) if len(exact_cos) > 0 else np.nan,
                "exact_cos2": float(exact_cos[1]) if len(exact_cos) > 1 else np.nan,
                "oracle_proj_norm1": float(oracle_proj_norm[0]) if len(oracle_proj_norm) > 0 else np.nan,
                "oracle_proj_norm2": float(oracle_proj_norm[1]) if len(oracle_proj_norm) > 1 else np.nan,
                "tail_mass": tail_mass,
                "car_cos1": float(car_oracle_cos[0]) if len(car_oracle_cos) > 0 else np.nan,
                "car_cos2": float(car_oracle_cos[1]) if len(car_oracle_cos) > 1 else np.nan,
                "car_exact_cos1": float(car_exact_cos[0]) if len(car_exact_cos) > 0 else np.nan,
                "car_exact_cos2": float(car_exact_cos[1]) if len(car_exact_cos) > 1 else np.nan,
                "sel_car_cos1": float(sel_car_cos[0]) if len(sel_car_cos) > 0 else np.nan,
                "sel_car_cos2": float(sel_car_cos[1]) if len(sel_car_cos) > 1 else np.nan,
                "car_tail_mass": carried_tail_mass,
                "survive_subspace": survive_subspace,
                "survive_v2": survive_v2,
                "prev2_score_ratio": prev2_score_ratio,
                "prev_sel_v2_in_curr": prev_sel_vs_curr,
                "prev_car_v2_in_curr": prev_car_vs_curr,
                "relerr_sval": final_relerr,
            }
        )

        state, V_r, s_state = make_state(M_gain, V_selected, H_selected, score_selected, rows_seen)
        old_row_memory, _ = probe.select_old_row_memory(
            A[:end0, :].astype(work_dtype, copy=False),
            V_r.astype(work_dtype, copy=False),
            args.old_memory_size if args.old_memory_size > 0 else win,
            np.random.default_rng(args.seed + end0),
            return_indices=True,
        )
        prev_opt2 = np.ascontiguousarray(V_selected[:, 1].copy())
        prev_selected = V_selected
        prev_carried = V_carried

    mean_align = float(np.linalg.norm((V_r @ V_r.T) @ V_exact[:, :1], "fro"))
    result = {
        "policy": policy,
        "win": win,
        "blocks": len(rows),
        "mean_align": mean_align,
        "mean_relerr_sval": rows[-1]["relerr_sval"],
        "elapsed": time.time() - t0,
        "rows": rows,
    }
    return result


def summarize_result(result):
    rows = result["rows"]
    counts = {}
    for row in rows:
        counts[row["selected_label"]] = counts.get(row["selected_label"], 0) + 1
    final = rows[-1]
    return {
        "policy": result["policy"],
        "win": result["win"],
        "blocks": result["blocks"],
        "mean_align": result["mean_align"],
        "mean_relerr_sval": result["mean_relerr_sval"],
        "mean_cos2": float(np.nanmean([r["cos2"] for r in rows])),
        "final_cos": [final["cos1"], final["cos2"]],
        "final_exact_cos": [final["exact_cos1"], final["exact_cos2"]],
        "final_car_cos": [final["car_cos1"], final["car_cos2"]],
        "final_car_exact_cos": [final["car_exact_cos1"], final["car_exact_cos2"]],
        "final_oracle_proj_norm": [final["oracle_proj_norm1"], final["oracle_proj_norm2"]],
        "final_tail_mass": final["tail_mass"],
        "selected_label_counts": counts,
    }


def rows_at_checkpoints(results, checkpoints):
    out = []
    for result in results:
        by_rows = {row["rows_seen"]: row for row in result["rows"]}
        for rows_seen in checkpoints:
            if rows_seen in by_rows:
                row = dict(by_rows[rows_seen])
                row["checkpoint_rows"] = rows_seen
                out.append(row)
    return out


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path, summaries, checkpoint_rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Future-hmean vs doubled-window combined comparison\n")
        f.write("===============================================\n\n")
        f.write("Summaries\n")
        f.write("---------\n")
        for s in summaries:
            f.write(
                f"policy={s['policy']} win={s['win']} blocks={s['blocks']} "
                f"mean_align={s['mean_align']:.6f} mean_relerr_sval={s['mean_relerr_sval']:.8f} "
                f"mean_cos2={s['mean_cos2']:.6f} final_cos=[{fmt(s['final_cos'])}] "
                f"final_exact_cos=[{fmt(s['final_exact_cos'])}] final_car_cos=[{fmt(s['final_car_cos'])}] "
                f"final_car_exact_cos=[{fmt(s['final_car_exact_cos'])}] "
                f"final_oracle_proj_norm=[{fmt(s['final_oracle_proj_norm'])}] final_tail_mass={s['final_tail_mass']:.6f} "
                f"selected_labels={json.dumps(s['selected_label_counts'], sort_keys=True)}\n"
            )
        f.write("\nCheckpoint rows\n")
        f.write("---------------\n")
        for row in checkpoint_rows:
            f.write(
                f"policy={row['policy']} win={row['win']} rows_seen={row['rows_seen']} block={row['block']} "
                f"selected_label={row['selected_label']} selected_future_hmean={row['selected_future_hmean']:.6f} "
                f"future_hmean_gap_vs_opt2={row['future_hmean_gap_vs_opt2']:.6f} "
                f"full_score_gap_vs_opt2={row['full_score_gap_vs_opt2']:.6f} "
                f"cos=[{row['cos1']:.6f} {row['cos2']:.6f}] "
                f"exact_cos=[{row['exact_cos1']:.6f} {row['exact_cos2']:.6f}] "
                f"car_cos=[{row['car_cos1']:.6f} {row['car_cos2']:.6f}] "
                f"car_exact_cos=[{row['car_exact_cos1']:.6f} {row['car_exact_cos2']:.6f}] "
                f"sel_car_cos=[{row['sel_car_cos1']:.6f} {row['sel_car_cos2']:.6f}] "
                f"oracle_proj_norm=[{row['oracle_proj_norm1']:.6f} {row['oracle_proj_norm2']:.6f}] "
                f"tail_mass={row['tail_mass']:.6f} car_tail_mass={row['car_tail_mass']:.6f} "
                f"survive_subspace={row['survive_subspace']:.6f} survive_v2={row['survive_v2']:.6f} "
                f"prev2_score_ratio={row['prev2_score_ratio']:.6f} relerr_sval={row['relerr_sval']:.8f}\n"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="mixed-tail-sharp")
    parser.add_argument("--wins", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--policies", nargs="+", default=["combined", "future_hmean"])
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-rows", action="store_true", default=True)
    parser.add_argument("--row-shuffle-seed", type=int, default=0)
    parser.add_argument("--old-memory-size", type=int, default=32)
    parser.add_argument("--future-horizon", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--q0", type=int, default=8)
    parser.add_argument("--qmax", type=int, default=48)
    parser.add_argument("--krylov-depth", type=int, default=2)
    parser.add_argument("--residual-tol", type=float, default=0.01)
    parser.add_argument("--expansion-maxit", type=int, default=8)
    parser.add_argument("--num-restarts", type=int, default=8)
    parser.add_argument("--maxit", type=int, default=120)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--post-expansion-maxit", type=int, default=80)
    parser.add_argument("--r-sig", type=int, default=2)
    parser.add_argument("--alpha-sig", type=float, default=0.003)
    parser.add_argument("--alpha-tail", type=float, default=0.0145)
    parser.add_argument("--tail-scale", type=float, default=0.99)
    parser.add_argument("--sigma1", type=float, default=0.991)
    parser.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    parser.add_argument("--json-out", default="summary/future_hmean_vs_window_comparison.json")
    parser.add_argument("--csv-out", default="summary/future_hmean_vs_window_comparison_checkpoints.csv")
    parser.add_argument("--text-out", default="summary/future_hmean_vs_window_comparison.txt")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    np.random.seed(args.seed)
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
    for win in args.wins:
        for policy in args.policies:
            result = run_trajectory(A, V_exact, sigma1, policy, win, args)
            results.append(result)
            print(
                f"comparison_run matrix={args.matrix} policy={policy} win={win} "
                f"mean_align={result['mean_align']:.6f} mean_relerr_sval={result['mean_relerr_sval']:.8f}"
            )

    summaries = [summarize_result(r) for r in results]
    checkpoint_rows = rows_at_checkpoints(results, common_checkpoint_rows(args.n))
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump({"summaries": summaries, "results": results}, f, indent=2, sort_keys=True)
    write_csv(args.csv_out, checkpoint_rows)
    write_text(args.text_out, summaries, checkpoint_rows)
    print(
        f"wrote_json={args.json_out} wrote_csv={args.csv_out} wrote_text={args.text_out} "
        f"elapsed={time.time() - t0:.3f}"
    )


if __name__ == "__main__":
    main()
