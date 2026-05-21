import argparse
import csv
import json
import time

import numpy as np

import cex_restricted_space_probe as probe
from second_slot_tail_bias_diagnostic import (
    make_state,
    orth_against,
    raw_oracle_columns,
    svd_complement,
)


def normed(v):
    if v is None:
        return None
    out = np.asarray(v, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(out))
    if nrm <= 1e-30:
        return None
    return np.ascontiguousarray(out / nrm)


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


def response_shape(A_block, v):
    if v is None:
        return {
            "current_gain2": np.nan,
            "current_relH": np.nan,
            "max_frac": np.nan,
            "top4_frac": np.nan,
        }
    y = np.asarray(A_block, dtype=np.float64) @ np.asarray(v, dtype=np.float64)
    e = y * y
    gain2 = max(float(np.sum(e)), 1e-30)
    p = e / gain2
    p_pos = p[p > 0.0]
    H = -float(np.sum(p_pos * np.log(p_pos))) if p_pos.size else np.nan
    relH = H / np.log(max(len(e), 2)) if np.isfinite(H) else np.nan
    sorted_p = np.sort(p)[::-1]
    return {
        "current_gain2": gain2,
        "current_relH": relH,
        "max_frac": float(sorted_p[0]) if sorted_p.size else np.nan,
        "top4_frac": float(np.sum(sorted_p[: min(4, sorted_p.size)])) if sorted_p.size else np.nan,
    }


def block_only_component(v, A_block, rows_ref, state_prev, old_row_memory):
    if v is None:
        return None
    comp = probe.combined_score_component_details(
        A_block,
        A_block,
        v,
        rows_ref,
        state_prev=state_prev,
        old_row_memory=old_row_memory,
    )
    shape = response_shape(A_block, v)
    return {
        "score": float(comp["score_total"]),
        "gain2": float(comp["gain2"]),
        "phi": float(comp["phi"]),
        "pooled_relH": float(comp["pooled_rel_H"]),
        "current_relH": shape["current_relH"],
        "max_frac": shape["max_frac"],
        "top4_frac": shape["top4_frac"],
    }


def full_score(v, M_gain, A_block, rows_ref, state_prev, old_row_memory):
    if v is None:
        return np.nan
    comp = probe.combined_score_component_details(
        M_gain,
        A_block,
        v,
        rows_ref,
        state_prev=state_prev,
        old_row_memory=old_row_memory,
    )
    return float(comp["score_total"])


def oracle_metrics(v, Q_oracle, raw_oracle):
    if v is None or Q_oracle is None or np.asarray(Q_oracle).size == 0:
        return {"oracle_mass": np.nan, "q2_absdot": np.nan}
    vv = normed(v)
    Q = np.asarray(Q_oracle, dtype=np.float64)
    oracle_mass = float(np.linalg.norm(Q @ (Q.T @ vv)) ** 2)
    q2 = raw_oracle[1] if len(raw_oracle) > 1 else (Q[:, 1] if Q.shape[1] > 1 else None)
    if q2 is None:
        q2_absdot = np.nan
    else:
        q2_absdot = abs(float(np.dot(vv, normed(q2))))
    return {"oracle_mass": oracle_mass, "q2_absdot": q2_absdot}


def safe_ratio(num, den):
    return float(num / max(float(den), 1e-30))


def summarize_matrix(matrix, rows, candidate_labels):
    blocks = len(rows)
    best_gain_counts = {label: 0 for label in candidate_labels}
    best_score_counts = {label: 0 for label in candidate_labels}
    metrics = {
        label: {
            "gain_share": [],
            "score_share": [],
            "oracle_mass": [],
            "q2_absdot": [],
            "stability": [],
            "current_relH": [],
            "max_frac": [],
            "top4_frac": [],
            "full_score_share": [],
        }
        for label in candidate_labels
    }
    prev_vecs = {}
    streaks = []
    current_streak_label = None
    current_streak_len = 0

    for row in rows:
        recs = row["records"]
        finite_gain = {k: v["gain2"] for k, v in recs.items() if v is not None and np.isfinite(v["gain2"])}
        finite_score = {k: v["score"] for k, v in recs.items() if v is not None and np.isfinite(v["score"])}
        finite_full = {k: v["full_score"] for k, v in recs.items() if v is not None and np.isfinite(v["full_score"])}
        if not finite_gain:
            continue
        best_gain = max(finite_gain.values())
        best_gain_label = max(finite_gain, key=finite_gain.get)
        best_gain_counts[best_gain_label] += 1
        if current_streak_label == best_gain_label:
            current_streak_len += 1
        else:
            if current_streak_label is not None:
                streaks.append((current_streak_label, current_streak_len))
            current_streak_label = best_gain_label
            current_streak_len = 1
        if finite_score:
            best_score = max(finite_score.values())
            best_score_label = max(finite_score, key=finite_score.get)
            best_score_counts[best_score_label] += 1
        else:
            best_score = np.nan
        best_full = max(finite_full.values()) if finite_full else np.nan
        for label in candidate_labels:
            rec = recs.get(label)
            if rec is None:
                continue
            metrics[label]["gain_share"].append(safe_ratio(rec["gain2"], best_gain))
            metrics[label]["score_share"].append(safe_ratio(rec["score"], best_score))
            metrics[label]["oracle_mass"].append(rec["oracle_mass"])
            metrics[label]["q2_absdot"].append(rec["q2_absdot"])
            metrics[label]["current_relH"].append(rec["current_relH"])
            metrics[label]["max_frac"].append(rec["max_frac"])
            metrics[label]["top4_frac"].append(rec["top4_frac"])
            if np.isfinite(best_full):
                metrics[label]["full_score_share"].append(safe_ratio(rec["full_score"], best_full))
            vec = rec.get("vec")
            if label in prev_vecs and vec is not None:
                metrics[label]["stability"].append(abs(float(np.dot(prev_vecs[label], vec))))
            if vec is not None:
                prev_vecs[label] = vec
    if current_streak_label is not None:
        streaks.append((current_streak_label, current_streak_len))

    out = {
        "matrix": matrix,
        "blocks": blocks,
        "best_gain_counts": best_gain_counts,
        "best_score_counts": best_score_counts,
        "longest_best_gain_streak": max((s[1] for s in streaks), default=0),
        "longest_best_gain_streak_label": max(streaks, key=lambda s: s[1])[0] if streaks else "",
        "labels": {},
    }
    for label in candidate_labels:
        vals = {}
        for key, arr in metrics[label].items():
            a = np.asarray([x for x in arr if np.isfinite(x)], dtype=float)
            vals[f"mean_{key}"] = float(np.mean(a)) if a.size else np.nan
            vals[f"min_{key}"] = float(np.min(a)) if a.size else np.nan
        support = np.asarray(metrics[label]["gain_share"], dtype=float)
        oracle = np.asarray(metrics[label]["oracle_mass"], dtype=float)
        stable = np.asarray(metrics[label]["stability"], dtype=float)
        relH = np.asarray(metrics[label]["current_relH"], dtype=float)
        support = support[np.isfinite(support)]
        oracle = oracle[np.isfinite(oracle)]
        stable = stable[np.isfinite(stable)]
        relH = relH[np.isfinite(relH)]
        vals["repeat_support_index"] = float(np.mean(support * support)) if support.size else np.nan
        vals["oracle_repeat_index"] = float(np.mean(support * oracle[: support.size])) if support.size and oracle.size == support.size else np.nan
        vals["stable_support_index"] = float(np.mean(support[1:] * stable)) if support.size > 1 and stable.size == support.size - 1 else np.nan
        vals["anti_spike_support_index"] = float(np.mean(support * relH[: support.size])) if support.size and relH.size == support.size else np.nan
        out["labels"][label] = vals
    return out


def run_matrix(matrix, args):
    np.random.seed(args.seed)
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
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    n = A.shape[0]
    rank = int(args.rank)
    state = None
    old_row_memory = None
    V_r = None
    prev_opt2 = None
    rows = []
    candidate_labels = [
        "opt2",
        "q2_vs_q1oracle",
        "svd_complement_mgain",
        "block_svd_complement",
        "prev_opt2",
    ]

    for block_idx, start0 in enumerate(range(0, n, args.win), start=1):
        end0 = min(start0 + args.win, n)
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
        V_selected = np.ascontiguousarray(np.asarray(V_score[:, :rank], dtype=np.float64))
        v1 = V_selected[:, :1]
        q2_raw = raw_oracle[1] if len(raw_oracle) > 1 else (Q_oracle[:, 1] if Q_oracle.shape[1] > 1 else None)
        candidates = {
            "opt2": normed(V_selected[:, 1]) if V_selected.shape[1] > 1 else None,
            "q2_vs_q1oracle": orth_against(q2_raw, Q_oracle[:, :1]) if q2_raw is not None else None,
            "svd_complement_mgain": svd_complement(M_gain, v1),
            "block_svd_complement": block_svd_complement(A_block, v1),
            "prev_opt2": prev_opt2,
        }

        records = {}
        for label, v in candidates.items():
            rec = block_only_component(v, A_block, n, state, old_row_memory)
            if rec is None:
                records[label] = None
                continue
            rec.update(oracle_metrics(v, Q_oracle, raw_oracle))
            rec["full_score"] = full_score(v, M_gain, A_block, n, state, old_row_memory)
            rec["vec"] = normed(v)
            records[label] = rec
        rows.append({"block": block_idx, "records": records})

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
        state, V_r, _ = make_state(M_gain, V_selected, H_selected, score_selected, rows_seen)
        old_row_memory, _ = probe.select_old_row_memory(
            A[:end0, :].astype(work_dtype, copy=False),
            V_r.astype(work_dtype, copy=False),
            args.old_memory_size,
            np.random.default_rng(args.seed + end0),
            return_indices=True,
        )
        prev_opt2 = normed(V_selected[:, 1].copy()) if V_selected.shape[1] > 1 else None

    return summarize_matrix(matrix, rows, candidate_labels)


def write_csv(path, summaries):
    labels = ["opt2", "q2_vs_q1oracle", "svd_complement_mgain", "block_svd_complement", "prev_opt2"]
    fields = [
        "matrix",
        "label",
        "blocks",
        "best_gain_count",
        "best_score_count",
        "mean_gain_share",
        "mean_score_share",
        "mean_full_score_share",
        "mean_oracle_mass",
        "mean_q2_absdot",
        "mean_stability",
        "mean_current_relH",
        "mean_max_frac",
        "mean_top4_frac",
        "repeat_support_index",
        "oracle_repeat_index",
        "stable_support_index",
        "anti_spike_support_index",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            for label in labels:
                vals = summary["labels"][label]
                writer.writerow(
                    {
                        "matrix": summary["matrix"],
                        "label": label,
                        "blocks": summary["blocks"],
                        "best_gain_count": summary["best_gain_counts"].get(label, 0),
                        "best_score_count": summary["best_score_counts"].get(label, 0),
                        **{k: vals.get(k, np.nan) for k in fields if k.startswith("mean_")},
                        "repeat_support_index": vals.get("repeat_support_index", np.nan),
                        "oracle_repeat_index": vals.get("oracle_repeat_index", np.nan),
                        "stable_support_index": vals.get("stable_support_index", np.nan),
                        "anti_spike_support_index": vals.get("anti_spike_support_index", np.nan),
                    }
                )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrices",
        nargs="+",
        default=["static-cex", "diffuse-diffuse", "mixed-tail-sharp", "residual-spiky-shocks"],
    )
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--win", type=int, default=32)
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
    parser.add_argument("--json-out", default="summary/objective_design_repeated_block_support.json")
    parser.add_argument("--csv-out", default="summary/objective_design_repeated_block_support.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    summaries = []
    for matrix in args.matrices:
        summary = run_matrix(matrix, args)
        summaries.append(summary)
        q2 = summary["labels"]["q2_vs_q1oracle"]
        bsvd = summary["labels"]["block_svd_complement"]
        print(
            "repeated_support_summary "
            f"matrix={matrix} blocks={summary['blocks']} "
            f"best_gain={summary['best_gain_counts']} "
            f"q2_repeat={q2['repeat_support_index']:.6f} "
            f"q2_oracle_repeat={q2['oracle_repeat_index']:.6f} "
            f"q2_stable_support={q2['stable_support_index']:.6f} "
            f"block_svd_repeat={bsvd['repeat_support_index']:.6f} "
            f"block_svd_oracle_repeat={bsvd['oracle_repeat_index']:.6f}"
        )
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)
    write_csv(args.csv_out, summaries)
    print(f"wrote_json={args.json_out} wrote_csv={args.csv_out} elapsed={time.time() - t0:.3f}")


if __name__ == "__main__":
    main()
