import argparse
import csv
import json
import time

import numpy as np

import cex_restricted_space_probe as probe
from second_slot_tail_bias_diagnostic import (
    intervention_frame,
    make_state,
    orth_against,
    raw_oracle_columns,
    svd_complement,
)


LABELS = [
    "opt2",
    "q2_vs_q1oracle",
    "q2_raw_projected",
    "block_complement",
    "mgain_deflated_svd",
    "opt2_outside",
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


def outside_component(v, Q_oracle):
    if v is None or Q_oracle is None:
        return None
    q = np.asarray(Q_oracle, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    out = vv - q @ (q.T @ vv)
    return normed(out)


def future_gain(A, block_idx, win, v, horizon):
    if v is None or horizon <= 0:
        return np.nan
    start = block_idx * win
    end = min(start + horizon * win, A.shape[0])
    if start >= end:
        return np.nan
    y = np.asarray(A[start:end, :], dtype=np.float64) @ np.asarray(v, dtype=np.float64)
    return float(np.dot(y, y))


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
        "full_score": float(full["score_total"]),
        "full_gain2": float(full["gain2"]),
        "block_score": float(block["score_total"]),
        "block_gain2": float(block["gain2"]),
        "block_phi": float(block["phi"]),
        "pooled_relH": float(block["pooled_rel_H"]),
        "old_y2": float(block["old_y2_sq"]) if np.isfinite(block["old_y2_sq"]) else 0.0,
        "current_relH": shape["current_relH"],
        "max_frac": shape["max_frac"],
        "top4_frac": shape["top4_frac"],
    }


def enrich_records(records):
    finite = [r for r in records.values() if r is not None]
    if not finite:
        return

    max_block_gain = max(r["block_gain2"] for r in finite)
    max_block_score = max(r["block_score"] for r in finite)
    max_full_score = max(r["full_score"] for r in finite)
    max_future_gain = max((r["future_gain"] for r in finite if np.isfinite(r["future_gain"])), default=np.nan)

    for rec in finite:
        rec["block_gain_share"] = rec["block_gain2"] / max(max_block_gain, 1e-30)
        rec["block_score_share"] = rec["block_score"] / max(max_block_score, 1e-30)
        rec["full_score_share"] = rec["full_score"] / max(max_full_score, 1e-30)
        rec["future_share"] = (
            rec["future_gain"] / max(max_future_gain, 1e-30) if np.isfinite(max_future_gain) else np.nan
        )
        relH = max(float(rec["current_relH"]), 0.0) if np.isfinite(rec["current_relH"]) else 0.0
        not_complement = (
            max(1.0 - float(rec["complement_align2"]), 0.0) if np.isfinite(rec["complement_align2"]) else 1.0
        )
        rec["obj_full_score"] = rec["full_score"]
        rec["obj_block_score"] = rec["block_score"]
        rec["obj_block_gain"] = rec["block_gain2"]
        rec["obj_hard_complement_block"] = rec["block_score_share"] * relH * not_complement
        rec["obj_future_hmean"] = hmean(rec["block_gain_share"], rec["future_share"]) * relH


def order_line(obj_name, records):
    vals = {
        label: rec[obj_name]
        for label, rec in records.items()
        if rec is not None and np.isfinite(rec.get(obj_name, np.nan))
    }
    if not vals:
        return ""
    ordered = sorted(vals.items(), key=lambda item: item[1], reverse=True)
    return " > ".join(f"{label}({value:.6f})" for label, value in ordered)


def candidate_row(block_idx, mode, label, rec):
    return {
        "block": block_idx,
        "mode": mode,
        "label": label,
        "full_score": np.nan if rec is None else rec["full_score"],
        "block_score": np.nan if rec is None else rec["block_score"],
        "block_gain2": np.nan if rec is None else rec["block_gain2"],
        "hard_complement_block": np.nan if rec is None else rec["obj_hard_complement_block"],
        "future_hmean": np.nan if rec is None else rec["obj_future_hmean"],
        "oracle_qr_align2": np.nan if rec is None else rec["oracle_qr_align2"],
        "oracle_mass": np.nan if rec is None else rec["oracle_mass"],
        "complement_align2": np.nan if rec is None else rec["complement_align2"],
        "mgain_svd_align2": np.nan if rec is None else rec["mgain_svd_align2"],
        "current_relH": np.nan if rec is None else rec["current_relH"],
        "max_frac": np.nan if rec is None else rec["max_frac"],
        "top4_frac": np.nan if rec is None else rec["top4_frac"],
    }


def build_records(A, V_exact, mode, args):
    A = np.asarray(A, dtype=np.float64)
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    n = A.shape[0]
    rank = int(args.rank)
    state = None
    old_row_memory = None
    V_r = None
    block_rows = []
    csv_rows = []

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
        V_selected = intervention_frame(mode, block_idx, V_score[:, :rank], Q_oracle, raw_oracle, rank)
        V_selected = probe.orthonormalize_columns(np.asarray(V_selected[:, :rank], dtype=np.float64), dtype=np.float64)[
            :, :rank
        ]
        v1 = V_selected[:, :1]
        opt2 = normed(V_selected[:, 1]) if V_selected.shape[1] > 1 else None
        q2_raw = None
        if len(raw_oracle) > 1:
            q2_raw = normed(raw_oracle[1])
        elif Q_oracle.shape[1] > 1:
            q2_raw = normed(Q_oracle[:, 1])
        q2_qr = orth_against(q2_raw, Q_oracle[:, :1]) if q2_raw is not None else None
        block_comp = block_svd_complement(A_block, v1)
        mgain_deflated = svd_complement(M_gain, v1)
        opt2_outside = outside_component(opt2, Q_oracle)

        candidates = {
            "opt2": opt2,
            "q2_vs_q1oracle": q2_qr,
            "q2_raw_projected": q2_raw,
            "block_complement": block_comp,
            "mgain_deflated_svd": mgain_deflated,
            "opt2_outside": opt2_outside,
        }

        records = {}
        for label, v in candidates.items():
            rec = component(v, M_gain, A_block, n, state, old_row_memory)
            if rec is None:
                records[label] = None
                continue
            rec["future_gain"] = future_gain(A, block_idx, args.win, v, args.future_horizon)
            rec["oracle_qr_align2"] = np.nan if q2_qr is None else float(np.dot(rec["vec"], q2_qr) ** 2)
            rec["oracle_mass"] = float(np.linalg.norm(Q_oracle @ (Q_oracle.T @ rec["vec"])) ** 2)
            rec["complement_align2"] = np.nan if block_comp is None else float(np.dot(rec["vec"], block_comp) ** 2)
            rec["mgain_svd_align2"] = (
                np.nan if mgain_deflated is None else float(np.dot(rec["vec"], mgain_deflated) ** 2)
            )
            records[label] = rec
        enrich_records(records)
        for label in LABELS:
            csv_rows.append(candidate_row(block_idx, mode, label, records.get(label)))

        block_rows.append(
            {
                "block": block_idx,
                "mode": mode,
                "objective_orderings": {
                    "full_score": order_line("obj_full_score", records),
                    "block_score": order_line("obj_block_score", records),
                    "block_gain": order_line("obj_block_gain", records),
                    "hard_complement_block": order_line("obj_hard_complement_block", records),
                    "future_hmean": order_line("obj_future_hmean", records),
                },
                "records": records,
            }
        )

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

    return block_rows, csv_rows


def json_ready(results):
    out = {}
    for mode, blocks in results.items():
        mode_rows = []
        for row in blocks:
            row_out = {
                "block": row["block"],
                "mode": row["mode"],
                "objective_orderings": dict(row["objective_orderings"]),
                "records": {},
            }
            for label, rec in row["records"].items():
                if rec is None:
                    row_out["records"][label] = None
                    continue
                clean = {k: v for k, v in rec.items() if k != "vec"}
                row_out["records"][label] = clean
            mode_rows.append(row_out)
        out[mode] = mode_rows
    return out


def write_csv(path, rows):
    fields = [
        "mode",
        "block",
        "label",
        "full_score",
        "block_score",
        "block_gain2",
        "hard_complement_block",
        "future_hmean",
        "oracle_qr_align2",
        "oracle_mass",
        "complement_align2",
        "mgain_svd_align2",
        "current_relH",
        "max_frac",
        "top4_frac",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path, results):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Mixed-tail-sharp per-block objective probe\n")
        f.write("=========================================\n\n")
        for mode, blocks in results.items():
            f.write(f"Mode: {mode}\n")
            f.write(f"{'-' * (6 + len(mode))}\n")
            for row in blocks:
                f.write(f"block={row['block']}\n")
                for name, ordering in row["objective_orderings"].items():
                    f.write(f"  {name}: {ordering}\n")
                for label in LABELS:
                    rec = row["records"].get(label)
                    if rec is None:
                        f.write(f"  {label}: unavailable\n")
                        continue
                    f.write(
                        "  "
                        f"{label}: full_score={rec['full_score']:.6f} "
                        f"block_score={rec['block_score']:.6f} "
                        f"block_gain2={rec['block_gain2']:.6f} "
                        f"hard_complement_block={rec['obj_hard_complement_block']:.6f} "
                        f"future_hmean={rec['obj_future_hmean']:.6f} "
                        f"oracle_qr_align2={rec['oracle_qr_align2']:.6f} "
                        f"oracle_mass={rec['oracle_mass']:.6f} "
                        f"complement_align2={rec['complement_align2']:.6f} "
                        f"mgain_svd_align2={rec['mgain_svd_align2']:.6f} "
                        f"current_relH={rec['current_relH']:.6f} "
                        f"max_frac={rec['max_frac']:.6f} "
                        f"top4_frac={rec['top4_frac']:.6f}\n"
                    )
                f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interventions",
        nargs="+",
        default=["normal", "force-b1", "force-b1b2"],
        choices=("normal", "force-b1", "force-b1b2", "force-second-b1"),
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
    parser.add_argument("--json-out", default="summary/mixed_tail_sharp_per_block_objective_probe.json")
    parser.add_argument("--csv-out", default="summary/mixed_tail_sharp_per_block_objective_probe.csv")
    parser.add_argument("--text-out", default="summary/mixed_tail_sharp_per_block_objective_probe.txt")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix="mixed-tail-sharp",
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
    results = {}
    csv_rows = []
    for mode in args.interventions:
        block_rows, mode_csv_rows = build_records(A, V_exact, mode, args)
        results[mode] = block_rows
        csv_rows.extend(mode_csv_rows)
        print(f"objective_probe mode={mode} blocks={len(block_rows)}")
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(json_ready(results), f, indent=2, sort_keys=True)
    write_csv(args.csv_out, csv_rows)
    write_text(args.text_out, results)
    print(
        f"wrote_json={args.json_out} wrote_csv={args.csv_out} wrote_text={args.text_out} "
        f"elapsed={time.time() - t0:.3f}"
    )


if __name__ == "__main__":
    main()
