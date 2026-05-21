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


LABELS = [
    "opt2",
    "q2_vs_q1oracle",
    "svd_complement_mgain",
    "block_svd_complement",
    "prev_opt2",
]


ALL_MATRICES = [
    "static-cex",
    "diffuse-diffuse",
    "mixed-tail-soft",
    "mixed-tail-balanced",
    "mixed-tail-sharp",
    "residual-spiky-shocks",
    "alternative-data-signals",
    "crowded-strategy",
    "execution-cost-slippage",
    "etf-basket-basis",
    "futures-term-structure",
    "intraday-liquidity-shape",
    "macro-factor-panel",
    "rates-cross-currency",
    "stat-arb-spreads",
    "options-vol-surface",
    "risk-residual-panel",
    "realized-vol-corr",
]


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
        return {"current_relH": np.nan, "max_frac": np.nan, "top4_frac": np.nan}
    y = np.asarray(A_block, dtype=np.float64) @ np.asarray(v, dtype=np.float64)
    e = y * y
    total = max(float(np.sum(e)), 1e-30)
    p = e / total
    p_pos = p[p > 0.0]
    H = -float(np.sum(p_pos * np.log(p_pos))) if p_pos.size else np.nan
    relH = H / np.log(max(len(e), 2)) if np.isfinite(H) else np.nan
    sorted_p = np.sort(p)[::-1]
    return {
        "current_relH": relH,
        "max_frac": float(sorted_p[0]) if sorted_p.size else np.nan,
        "top4_frac": float(np.sum(sorted_p[: min(4, sorted_p.size)])) if sorted_p.size else np.nan,
    }


def component(v, M_gain, A_block, rows_ref, state_prev, old_row_memory):
    if v is None:
        return None
    block = probe.combined_score_component_details(
        A_block,
        A_block,
        v,
        rows_ref,
        state_prev=state_prev,
        old_row_memory=old_row_memory,
    )
    full = probe.combined_score_component_details(
        M_gain,
        A_block,
        v,
        rows_ref,
        state_prev=state_prev,
        old_row_memory=old_row_memory,
    )
    shape = response_shape(A_block, v)
    return {
        "vec": normed(v),
        "block_gain2": float(block["gain2"]),
        "block_score": float(block["score_total"]),
        "block_phi": float(block["phi"]),
        "full_score": float(full["score_total"]),
        "full_gain2": float(full["gain2"]),
        "old_y2": float(block["old_y2_sq"]) if np.isfinite(block["old_y2_sq"]) else 0.0,
        "pooled_relH": float(block["pooled_rel_H"]),
        "current_relH": shape["current_relH"],
        "max_frac": shape["max_frac"],
        "top4_frac": shape["top4_frac"],
    }


def future_gain(A, block_idx, win, v, horizon):
    if v is None or horizon <= 0:
        return np.nan
    start = block_idx * win
    end = min(start + horizon * win, A.shape[0])
    if start >= end:
        return np.nan
    y = np.asarray(A[start:end, :], dtype=np.float64) @ np.asarray(v, dtype=np.float64)
    return float(np.dot(y, y))


def oracle_eval_metrics(v, Q_oracle):
    if v is None or Q_oracle is None or np.asarray(Q_oracle).size == 0:
        return {"oracle_mass": np.nan}
    vv = normed(v)
    Q = np.asarray(Q_oracle, dtype=np.float64)
    return {"oracle_mass": float(np.linalg.norm(Q @ (Q.T @ vv)) ** 2)}


def rank_objectives(records):
    finite = [r for r in records.values() if r is not None]
    if not finite:
        return

    max_block_gain = max(r["block_gain2"] for r in finite)
    max_block_score = max(r["block_score"] for r in finite)
    max_full_score = max(r["full_score"] for r in finite)
    max_old_y2 = max(r["old_y2"] for r in finite)
    max_future_gain = max((r["future_gain"] for r in finite if np.isfinite(r["future_gain"])), default=np.nan)
    block_svd_vec = records.get("block_svd_complement", {}).get("vec") if records.get("block_svd_complement") else None

    for rec in finite:
        rec["block_gain_share"] = rec["block_gain2"] / max(max_block_gain, 1e-30)
        rec["block_score_share"] = rec["block_score"] / max(max_block_score, 1e-30)
        rec["full_score_share"] = rec["full_score"] / max(max_full_score, 1e-30)
        rec["old_share"] = rec["old_y2"] / max(max_old_y2, 1e-30) if max_old_y2 > 0.0 else 0.0
        rec["future_share"] = (
            rec["future_gain"] / max(max_future_gain, 1e-30) if np.isfinite(max_future_gain) else np.nan
        )
        if block_svd_vec is None or rec["vec"] is None:
            rec["block_svd_align2"] = np.nan
        else:
            rec["block_svd_align2"] = float(np.dot(rec["vec"], block_svd_vec) ** 2)
        relH = max(float(rec["current_relH"]), 0.0) if np.isfinite(rec["current_relH"]) else 0.0
        not_spiky = relH * max(1.0 - float(rec["top4_frac"]), 0.0)
        not_block_svd = max(1.0 - float(rec["block_svd_align2"]), 0.0) if np.isfinite(rec["block_svd_align2"]) else 1.0
        full_adaptive_not_svd = (
            max(1.0 - float(rec["block_svd_align2"]) * (1.0 - rec["full_score_share"]), 0.0)
            if np.isfinite(rec["block_svd_align2"])
            else 1.0
        )
        old_adaptive_not_svd = (
            max(1.0 - float(rec["block_svd_align2"]) * (1.0 - rec["old_share"]), 0.0)
            if np.isfinite(rec["block_svd_align2"])
            else 1.0
        )
        future_adaptive_not_svd = (
            max(1.0 - float(rec["block_svd_align2"]) * (1.0 - rec["future_share"]), 0.0)
            if np.isfinite(rec["block_svd_align2"]) and np.isfinite(rec["future_share"])
            else np.nan
        )
        rec["obj_block_score"] = rec["block_score_share"]
        rec["obj_block_antispike"] = rec["block_score_share"] * relH
        rec["obj_block_concentration_guard"] = rec["block_score_share"] * not_spiky
        rec["obj_full_block_hmean"] = hmean(rec["block_score_share"], rec["full_score_share"]) * relH
        rec["obj_old_block_hmean"] = hmean(rec["block_gain_share"], rec["old_share"]) * relH
        rec["obj_complement_penalty"] = rec["block_score_share"] * relH * not_block_svd
        rec["obj_old_complement_penalty"] = hmean(rec["block_gain_share"], rec["old_share"]) * relH * not_block_svd
        rec["obj_future_hmean"] = hmean(rec["block_gain_share"], rec["future_share"]) * relH
        rec["obj_future_complement_penalty"] = hmean(rec["block_gain_share"], rec["future_share"]) * relH * not_block_svd
        rec["obj_adaptive_full_complement"] = rec["block_score_share"] * relH * full_adaptive_not_svd
        rec["obj_adaptive_old_complement"] = rec["block_score_share"] * relH * old_adaptive_not_svd
        rec["obj_adaptive_future_complement"] = (
            hmean(rec["block_gain_share"], rec["future_share"]) * relH * future_adaptive_not_svd
            if np.isfinite(future_adaptive_not_svd)
            else np.nan
        )


def summarize_objectives(matrix, rows, objective_names):
    out = {
        "matrix": matrix,
        "blocks": len(rows),
        "objectives": {},
    }
    for obj in objective_names:
        counts = {label: 0 for label in LABELS}
        valid_decisions = 0
        q2_rank = []
        q2_over_block_svd = []
        q2_margin_block_svd = []
        q2_over_opt2 = []
        q2_margin_opt2 = []
        winner_oracle_mass = []
        oracle90_wins = 0
        oracle99_wins = 0
        for row in rows:
            vals = {
                label: rec[obj]
                for label, rec in row["records"].items()
                if rec is not None and np.isfinite(rec.get(obj, np.nan))
            }
            if not vals:
                continue
            valid_decisions += 1
            winner = max(vals, key=vals.get)
            counts[winner] += 1
            ordered = sorted(vals, key=vals.get, reverse=True)
            if "q2_vs_q1oracle" in vals:
                q2_rank.append(ordered.index("q2_vs_q1oracle") + 1)
            if "q2_vs_q1oracle" in vals and "block_svd_complement" in vals:
                q2_margin_block_svd.append(vals["q2_vs_q1oracle"] - vals["block_svd_complement"])
                if vals["block_svd_complement"] > 1e-12:
                    q2_over_block_svd.append(vals["q2_vs_q1oracle"] / vals["block_svd_complement"])
            if "q2_vs_q1oracle" in vals and "opt2" in vals:
                q2_margin_opt2.append(vals["q2_vs_q1oracle"] - vals["opt2"])
                if vals["opt2"] > 1e-12:
                    q2_over_opt2.append(vals["q2_vs_q1oracle"] / vals["opt2"])
            rec = row["records"].get(winner)
            if rec is not None:
                oracle_mass = rec.get("oracle_mass", np.nan)
                winner_oracle_mass.append(oracle_mass)
                if np.isfinite(oracle_mass) and oracle_mass >= 0.90:
                    oracle90_wins += 1
                if np.isfinite(oracle_mass) and oracle_mass >= 0.99:
                    oracle99_wins += 1
        out["objectives"][obj] = {
            "winner_counts": counts,
            "valid_decisions": valid_decisions,
            "q2_win_count": counts.get("q2_vs_q1oracle", 0),
            "block_svd_win_count": counts.get("block_svd_complement", 0),
            "opt2_win_count": counts.get("opt2", 0),
            "oracle90_win_count": oracle90_wins,
            "oracle99_win_count": oracle99_wins,
            "mean_q2_rank": float(np.mean(q2_rank)) if q2_rank else np.nan,
            "mean_q2_over_block_svd": float(np.mean(q2_over_block_svd)) if q2_over_block_svd else np.nan,
            "mean_q2_margin_block_svd": float(np.mean(q2_margin_block_svd)) if q2_margin_block_svd else np.nan,
            "mean_q2_over_opt2": float(np.mean(q2_over_opt2)) if q2_over_opt2 else np.nan,
            "mean_q2_margin_opt2": float(np.mean(q2_margin_opt2)) if q2_margin_opt2 else np.nan,
            "mean_winner_oracle_mass": float(np.nanmean(winner_oracle_mass)) if winner_oracle_mass else np.nan,
        }
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
            rec = component(v, M_gain, A_block, n, state, old_row_memory)
            if rec is None:
                records[label] = None
                continue
            rec["future_gain"] = future_gain(A, block_idx, args.win, v, args.future_horizon)
            rec.update(oracle_eval_metrics(v, Q_oracle))
            records[label] = rec
        rank_objectives(records)
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

    objective_names = [
        "obj_block_score",
        "obj_block_antispike",
        "obj_block_concentration_guard",
        "obj_full_block_hmean",
        "obj_old_block_hmean",
        "obj_complement_penalty",
        "obj_old_complement_penalty",
        "obj_future_hmean",
        "obj_future_complement_penalty",
        "obj_adaptive_full_complement",
        "obj_adaptive_old_complement",
        "obj_adaptive_future_complement",
    ]
    return summarize_objectives(matrix, rows, objective_names)


def write_csv(path, summaries):
    fields = [
        "matrix",
        "objective",
        "blocks",
        "valid_decisions",
        "q2_win_count",
        "block_svd_win_count",
        "opt2_win_count",
        "oracle90_win_count",
        "oracle99_win_count",
        "mean_q2_rank",
        "mean_q2_over_block_svd",
        "mean_q2_margin_block_svd",
        "mean_q2_over_opt2",
        "mean_q2_margin_opt2",
        "mean_winner_oracle_mass",
        "winner_counts_json",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            for objective, rec in summary["objectives"].items():
                writer.writerow(
                    {
                        "matrix": summary["matrix"],
                        "objective": objective,
                        "blocks": summary["blocks"],
                        "valid_decisions": rec["valid_decisions"],
                        "q2_win_count": rec["q2_win_count"],
                        "block_svd_win_count": rec["block_svd_win_count"],
                        "opt2_win_count": rec["opt2_win_count"],
                        "oracle90_win_count": rec["oracle90_win_count"],
                        "oracle99_win_count": rec["oracle99_win_count"],
                        "mean_q2_rank": rec["mean_q2_rank"],
                        "mean_q2_over_block_svd": rec["mean_q2_over_block_svd"],
                        "mean_q2_margin_block_svd": rec["mean_q2_margin_block_svd"],
                        "mean_q2_over_opt2": rec["mean_q2_over_opt2"],
                        "mean_q2_margin_opt2": rec["mean_q2_margin_opt2"],
                        "mean_winner_oracle_mass": rec["mean_winner_oracle_mass"],
                        "winner_counts_json": json.dumps(rec["winner_counts"], sort_keys=True),
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
    parser.add_argument("--json-out", default="summary/objective_design_oracle_free_objective_probe.json")
    parser.add_argument("--csv-out", default="summary/objective_design_oracle_free_objective_probe.csv")
    parser.add_argument("--all-matrices", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.all_matrices:
        args.matrices = ALL_MATRICES
    t0 = time.time()
    summaries = []
    for matrix in args.matrices:
        summary = run_matrix(matrix, args)
        summaries.append(summary)
        print(f"oracle_free_matrix matrix={matrix} blocks={summary['blocks']}")
        for objective, rec in summary["objectives"].items():
            print(
                "oracle_free_objective "
                f"matrix={matrix} objective={objective} "
                f"q2_wins={rec['q2_win_count']}/{rec['valid_decisions']} "
                f"block_svd_wins={rec['block_svd_win_count']}/{rec['valid_decisions']} "
                f"opt2_wins={rec['opt2_win_count']}/{rec['valid_decisions']} "
                f"oracle90_wins={rec['oracle90_win_count']}/{rec['valid_decisions']} "
                f"mean_q2_rank={rec['mean_q2_rank']:.3f} "
                f"mean_winner_oracle_mass={rec['mean_winner_oracle_mass']:.6f}"
            )
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)
    write_csv(args.csv_out, summaries)
    print(f"wrote_json={args.json_out} wrote_csv={args.csv_out} elapsed={time.time() - t0:.3f}")


if __name__ == "__main__":
    main()
