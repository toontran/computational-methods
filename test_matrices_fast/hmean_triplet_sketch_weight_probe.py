"""Multi-block diagnostic: does raw_sketch < raw_gain1, raw_gain2 hold for the
projected oracle across blocks, and what would the corresponding sketch weight
look like?

For each block w >= 2 (so a sketch carries from block w-1), compute the
projection of V_exact[:, 0] and V_exact[:, 1] onto rowspan(sketch + A_cur +
A_fut) and report the per-subspace raw energies. From those we derive:
  - implied sketch weight = raw_sk / mean(raw_g1, raw_g2): the multiplicative
    factor in a weighted HM that would let the projected oracle sit at the
    interior optimum (gradients balanced).
  - sketch confidence proxies: principal cosines between the sketch basis
    state['V'] and V_exact[:, :rank]; ratio of sketch singular values to top
    singular values of the rows seen so far; ratio of state's stored singular
    values squared to the diagonal of (sketch_rowspace @ M_gain^T M_gain @
    sketch_rowspace) -- i.e. how well the carried sketch energies match the
    current pooled response.
"""

import argparse
import csv
import json

import numpy as np

import cex_restricted_space_probe as probe
from future_hmean_optimizer_diagnostic import rowspace_basis
from second_slot_tail_bias_diagnostic import make_state


def normed(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= 1e-30:
        return None
    return v / n


def project_unit(v, B):
    if v is None or B is None or B.size == 0:
        return None
    p = B @ (B.T @ v)
    n = float(np.linalg.norm(p))
    if n <= 1e-30:
        return None
    return p / n


def raw_energies(v, A_sketch, A_cur, A_fut):
    Asks = np.asarray(A_sketch, dtype=np.float64) if A_sketch is not None else None
    Ac = np.asarray(A_cur, dtype=np.float64)
    Af = np.asarray(A_fut, dtype=np.float64)
    raw_sk = float(np.dot(Asks @ v, Asks @ v)) if Asks is not None and Asks.size else np.nan
    raw_g1 = float(np.dot(Ac @ v, Ac @ v))
    raw_g2 = float(np.dot(Af @ v, Af @ v))
    return raw_sk, raw_g1, raw_g2


def hmean_raw(raws):
    valid = [x for x in raws if np.isfinite(x) and x > 0]
    if len(valid) != 3:
        return float("nan")
    return 3.0 / sum(1.0 / x for x in raws)


def hmean_weighted(raws, weights):
    valid = [(w, x) for w, x in zip(weights, raws) if np.isfinite(x) and x > 0 and w > 0]
    if len(valid) != 3:
        return float("nan")
    ws = sum(w for w, _ in valid)
    rec = sum(w / x for w, x in valid)
    return ws / rec


def stream_blocks(args):
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    A, V_exact, _, _ = probe.generate_matrix_input(
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
    rank = int(args.rank)
    half_win = int(args.half_win)

    state = None
    old_row_memory = None
    rows = []

    for block_id, start0 in enumerate(range(0, A.shape[0] - half_win, half_win), start=1):
        if args.max_blocks is not None and block_id > args.max_blocks:
            break
        mid0 = start0 + half_win
        end0 = min(mid0 + half_win, A.shape[0])
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

        if state is not None:
            # Subspace bases.
            B_sketch = rowspace_basis(np.asarray(A_sketch, dtype=np.float64))
            B_union = rowspace_basis(np.vstack([
                np.asarray(A_sketch, dtype=np.float64),
                np.asarray(A_cur, dtype=np.float64),
                np.asarray(A_fut, dtype=np.float64),
            ]))

            v1_proj = project_unit(normed(V_exact[:, 0]), B_union)
            v2_proj = project_unit(normed(V_exact[:, 1]), B_union)

            # Sketch confidence: principal angles between the carried sketch
            # basis state['V'] (right factors) and V_exact[:, :rank].
            sketch_basis = np.asarray(state["V"], dtype=np.float64)
            sketch_singvals = np.asarray(state["s"], dtype=np.float64)
            cos_sing = np.linalg.svd(sketch_basis.T @ V_exact[:, :rank], compute_uv=False)
            cos_sing = np.clip(cos_sing, -1.0, 1.0)

            # Top singular values of all rows seen (cumulative true SVD reference).
            # Use only top-rank singular values for cost.
            seen = np.asarray(A[:start0, :], dtype=np.float64)
            if seen.shape[0] >= 1:
                # Use SVD top-r via partial method for speed.
                _, svals_seen, _ = np.linalg.svd(seen, full_matrices=False)
                sigma_top = svals_seen[:rank]
            else:
                sigma_top = np.full(rank, np.nan)
            sketch_quality = np.full(rank, np.nan)
            for j in range(rank):
                if j < len(sigma_top) and sigma_top[j] > 1e-30 and j < len(sketch_singvals):
                    sketch_quality[j] = float(sketch_singvals[j] / sigma_top[j])

            for label, vproj in [("v1", v1_proj), ("v2", v2_proj)]:
                if vproj is None:
                    continue
                raw_sk, raw_g1, raw_g2 = raw_energies(vproj, A_sketch, A_cur, A_fut)
                raw_g_avg = 0.5 * (raw_g1 + raw_g2)
                ratio_sk_to_g = raw_sk / max(raw_g_avg, 1e-30)
                # Implied weight that equalises the gradient contributions of
                # the three terms in the harmonic mean at this candidate. For a
                # weighted HM = (sum w_i) / sum(w_i / x_i), the partial
                # derivative w.r.t. x_i is w_i HM^2 / (x_i^2 sum w_i). Equal
                # gradient magnitudes per direction therefore require
                # w_i / x_i^2 to be constant. Pinning w_g1 = w_g2 = 1 gives
                # w_sk = (raw_sk / sqrt(raw_g1 raw_g2))^2.
                if raw_g1 > 0 and raw_g2 > 0:
                    suggested_w_sk = (raw_sk / np.sqrt(raw_g1 * raw_g2)) ** 2
                else:
                    suggested_w_sk = float("nan")

                hm_uniform = hmean_raw([raw_sk, raw_g1, raw_g2])
                hm_dropw = hmean_raw([raw_g1, raw_g2])  # only block1 + block2
                hm_with_w = hmean_weighted([raw_sk, raw_g1, raw_g2],
                                           [suggested_w_sk, 1.0, 1.0])

                rows.append({
                    "matrix": args.matrix,
                    "block": int(block_id),
                    "rows_seen_before": int(start0),
                    "oracle": label,
                    "raw_sketch": raw_sk,
                    "raw_gain1": raw_g1,
                    "raw_gain2": raw_g2,
                    "raw_gain_avg_g1g2": raw_g_avg,
                    "ratio_sk_to_avg_g": ratio_sk_to_g,
                    "implied_w_sketch_for_balance": suggested_w_sk,
                    "hm_uniform_raw": hm_uniform,
                    "hm_drop_sketch_raw": hm_dropw,
                    "hm_with_implied_w_raw": hm_with_w,
                    "sketch_principal_cos": cos_sing.tolist(),
                    "sketch_singvals": sketch_singvals.tolist(),
                    "sigma_top_seen": sigma_top.tolist(),
                    "sketch_quality_ratio": sketch_quality.tolist(),
                    "sketch_dim": int(B_sketch.shape[1]),
                    "union_dim": int(B_union.shape[1]),
                })

        # Update state for next block.
        score_selected = np.zeros(rank, dtype=float)
        H_selected = np.zeros(rank, dtype=float)
        for j in range(rank):
            score_selected[j], _, H_selected[j] = probe.score_full_vector_details_forget(
                M_gain,
                A_cur,
                V_default[:, j],
                A.shape[0],
                state_prev=state,
                score_variant="combined",
                old_row_memory=old_row_memory,
            )
        state, V_r, _ = make_state(M_gain, V_default, H_selected, score_selected, rows_seen)
        old_row_memory, _ = probe.select_old_row_memory(
            np.asarray(A[:mid0, :], dtype=work_dtype),
            V_r.astype(work_dtype, copy=False),
            args.old_memory_size if args.old_memory_size > 0 else half_win,
            np.random.default_rng(args.seed + end0),
            return_indices=True,
        )

    return rows


def write_csv(path, rows):
    flat_rows = []
    for r in rows:
        out = dict(r)
        for k, v in list(out.items()):
            if isinstance(v, list):
                for j, x in enumerate(v):
                    out[f"{k}_{j}"] = x
                del out[k]
        flat_rows.append(out)
    if not flat_rows:
        return
    fields = sorted({k for r in flat_rows for k in r.keys()})
    fields = ["matrix", "block", "oracle", "rows_seen_before"] + [f for f in fields if f not in {"matrix", "block", "oracle", "rows_seen_before"}]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in flat_rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def write_text(path, rows, args):
    by_block = {}
    for r in rows:
        by_block.setdefault(r["block"], []).append(r)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"HM-triplet sketch-weight probe -- matrix={args.matrix}\n")
        f.write(f"half_win={args.half_win}  rank={args.rank}  blocks_analyzed={len(by_block)}\n\n")
        f.write(
            "Per-block raw energies of oracle projected onto rowspan(sketch + block1 + block2).\n"
            "Implied weight w_sketch makes the harmonic-mean gradient balanced at the projected oracle\n"
            "(setting w_g1 = w_g2 = 1; w_sk = (raw_sk / sqrt(raw_g1 raw_g2))^2).\n\n"
        )
        f.write(
            f"{'block':>5} {'oracle':>6} {'raw_sk':>11} {'raw_g1':>11} {'raw_g2':>11}  "
            f"{'sk/avg_g':>9} {'w_sk*':>9}  {'hm_uni':>11} {'hm_drop_sk':>12} {'hm_with_w':>11}  "
            f"{'sk_cos0':>8} {'sk_cos1':>8} {'sk_qual0':>9} {'sk_qual1':>9}\n"
        )
        for block in sorted(by_block):
            for r in by_block[block]:
                cos = r["sketch_principal_cos"]
                qual = r["sketch_quality_ratio"]
                f.write(
                    "{b:>5} {o:>6} {rs:>11.4e} {r1:>11.4e} {r2:>11.4e}  "
                    "{rt:>9.4f} {w:>9.4f}  {hu:>11.4e} {hd:>12.4e} {hw:>11.4e}  "
                    "{c0:>8.4f} {c1:>8.4f} {q0:>9.4f} {q1:>9.4f}\n".format(
                        b=block, o=r["oracle"],
                        rs=r["raw_sketch"], r1=r["raw_gain1"], r2=r["raw_gain2"],
                        rt=r["ratio_sk_to_avg_g"], w=r["implied_w_sketch_for_balance"],
                        hu=r["hm_uniform_raw"], hd=r["hm_drop_sketch_raw"], hw=r["hm_with_implied_w_raw"],
                        c0=cos[0] if len(cos) > 0 else float("nan"),
                        c1=cos[1] if len(cos) > 1 else float("nan"),
                        q0=qual[0] if len(qual) > 0 else float("nan"),
                        q1=qual[1] if len(qual) > 1 else float("nan"),
                    )
                )
        f.write("\n")
        # Aggregate stats: per oracle, mean of ratio and implied weight.
        f.write("Aggregate (mean / median over blocks)\n")
        for oracle in ("v1", "v2"):
            sub = [r for r in rows if r["oracle"] == oracle]
            if not sub:
                continue
            ratios = [r["ratio_sk_to_avg_g"] for r in sub]
            weights = [r["implied_w_sketch_for_balance"] for r in sub]
            f.write(
                f"  oracle={oracle}: mean_ratio_sk_to_avg_g={np.nanmean(ratios):.4f}  "
                f"median={np.nanmedian(ratios):.4f}  "
                f"mean_implied_w_sk={np.nanmean(weights):.4f}  median={np.nanmedian(weights):.4f}\n"
            )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="mixed-tail-sharp")
    p.add_argument("--out-prefix", default="summary/hmean_triplet_sketch_weight_probe")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--half-win", type=int, default=32)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--max-blocks", type=int, default=None)
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
    p.add_argument("--r-sig", type=int, default=2)
    p.add_argument("--alpha-sig", type=float, default=0.003)
    p.add_argument("--alpha-tail", type=float, default=0.0145)
    p.add_argument("--tail-scale", type=float, default=0.99)
    p.add_argument("--sigma1", type=float, default=0.991)
    p.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    return p.parse_args()


def main():
    args = parse_args()
    rows = stream_blocks(args)
    write_csv(args.out_prefix + ".csv", rows)
    write_text(args.out_prefix + ".txt", rows, args)
    with open(args.out_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2, sort_keys=True, default=float)
    print(f"wrote {args.out_prefix}.csv {args.out_prefix}.txt {args.out_prefix}.json")


if __name__ == "__main__":
    main()
