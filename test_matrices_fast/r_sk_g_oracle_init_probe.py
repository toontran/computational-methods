"""Oracle-initialized HM3 ascent probe.

Question: HM3 score at oracle_v_proj is below S4_best by a small margin (0.005)
on most blocks. Is that a local-optimum / search-basis artefact, or is the oracle
genuinely a saddle / non-stationary point of HM3?

Procedure (per matrix × per block × per oracle ∈ {v1_proj, v2_proj}):
  1. Stream to the target block. Build B_union = rowspace_basis(A_sketch ∪ A_cur ∪ A_fut).
  2. Project the full oracle onto B_union (oracle_v_proj).
  3. Run S4 ascent inside B_union with the single warm start = oracle_v_proj,
     random_starts = 0. Compare score before / after, and the angle moved.
  4. Decompose the final direction v_end against:
       - top-k right singular vectors of A_sketch / A_cur / A_fut / A_union
       - the rows of A_cur and A_fut (which row directions does v_end most align with?)
     Same for the "outside" component v_outside = (v_end − ⟨v_start, v_end⟩ v_start)
     normalized — what direction did the optimizer ESCAPE TOWARD?
"""

import argparse
import json
import os

import numpy as np

import cex_restricted_space_probe as probe
from future_hmean_optimizer_diagnostic import rowspace_basis
from hmean_evidence_score import per_block_constants, stream_to_block
from r_sk_g_score import optimize_r_sk_g_in_basis, r_sk_g_value_grad


def _normed(v, eps=1e-30):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= eps:
        return None
    return v / n


def _project_unit(vec, B):
    if vec is None or B is None or B.size == 0:
        return None
    p = B @ (B.T @ vec)
    return _normed(p)


def _top_right_singvecs(M, k):
    """Return Vh^T[:, :k] of SVD(M); empty if M has no rows."""
    M = np.asarray(M, dtype=np.float64)
    if M.size == 0 or M.shape[0] == 0:
        return np.zeros((M.shape[1] if M.ndim == 2 else 0, 0), dtype=np.float64)
    _, _, Vh = np.linalg.svd(M, full_matrices=False)
    return np.ascontiguousarray(Vh[:k].T, dtype=np.float64)


def _top_alignments(v, basis, k=4):
    """Return the (col_index, cos2) for the top-k columns of basis aligned with v."""
    if basis is None or basis.size == 0:
        return []
    al = (basis.T @ v) ** 2
    order = np.argsort(-al)[:k]
    return [(int(j), float(al[j])) for j in order]


def _row_alignments(A, v, k=4):
    """Find the rows of A whose normalized direction is most aligned with v."""
    if A is None or A.size == 0:
        return []
    A = np.asarray(A, dtype=np.float64)
    norms = np.linalg.norm(A, axis=1)
    valid = norms > 1e-30
    if not np.any(valid):
        return []
    A_unit = np.zeros_like(A)
    A_unit[valid] = A[valid] / norms[valid, None]
    al = (A_unit @ v) ** 2
    al[~valid] = 0.0
    order = np.argsort(-al)[:k]
    return [(int(j), float(al[j]), float(norms[j])) for j in order]


def probe_one(args, matrix, block_id, k_svd=4, k_rows=4):
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=matrix, n=args.n, preset=args.preset, seed=args.seed,
        r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
        shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, np.float64)
    V_exact = np.asarray(V_exact, np.float64)

    snaps = stream_to_block(args, A, V_exact, work_dtype, int(args.rank), block_id, {block_id})
    snap = snaps[block_id]
    A_cur = snap["A_cur"]
    A_fut = snap["A_fut"]
    A_sketch = snap["A_sketch"]
    state = snap["state"]
    consts = per_block_constants(A, block_id, int(args.half_win))
    c_sk = consts["c_sk"]

    A_sketch_for = A_sketch if A_sketch.size else None

    if A_sketch_for is not None:
        union_stack = np.vstack([A_sketch, A_cur, A_fut])
    else:
        union_stack = np.vstack([A_cur, A_fut])
    B_union = rowspace_basis(union_stack)

    oracle_v1 = _normed(V_exact[:, 0])
    oracle_v2 = _normed(V_exact[:, 1])
    oracle_v1_proj = _project_unit(oracle_v1, B_union)
    oracle_v2_proj = _project_unit(oracle_v2, B_union)

    V_state = None
    if state is not None and state.get("V") is not None:
        Vs = np.asarray(state["V"], dtype=np.float64)
        if Vs.size:
            V_state = Vs

    # Per-block SVD bases (right singular vectors).
    svd_sketch = _top_right_singvecs(A_sketch_for if A_sketch_for is not None else np.zeros((0, A.shape[1])), k_svd)
    svd_cur = _top_right_singvecs(A_cur, k_svd)
    svd_fut = _top_right_singvecs(A_fut, k_svd)
    svd_union = _top_right_singvecs(union_stack, k_svd)

    out = {
        "matrix": matrix, "block": block_id,
        "n_union": int(B_union.shape[1]),
        "c_sk": float(c_sk),
        "starts": {},
    }

    for label, v_start in (("oracle_v1_proj", oracle_v1_proj), ("oracle_v2_proj", oracle_v2_proj)):
        if v_start is None:
            continue
        # Score before ascent.
        score_start, _, r_sk_s, raw_g1_s, raw_g2_s, hm_g_s, sat_s, st_align_s = r_sk_g_value_grad(
            A_sketch_for, A_cur, A_fut, c_sk, v_start,
            variant="S4", V_state=V_state,
        )

        result = optimize_r_sk_g_in_basis(
            A_cur, A_fut, A_sketch_for, c_sk,
            B_union, [v_start],
            np.random.default_rng(args.seed + 99000 + block_id + (1 if label.endswith("v1_proj") else 2)),
            args.union_maxit, args.union_tol, 0,  # random_starts = 0 → only the oracle warm start
            variant="S4", V_state=V_state,
        )
        v_end = None if result is None else _normed(result["vec"])
        if v_end is None:
            continue

        score_end, _, r_sk_e, raw_g1_e, raw_g2_e, hm_g_e, sat_e, st_align_e = r_sk_g_value_grad(
            A_sketch_for, A_cur, A_fut, c_sk, v_end,
            variant="S4", V_state=V_state,
        )

        cos_start_end = float(np.dot(v_start, v_end))
        cos2_start_end = cos_start_end * cos_start_end

        # Outside direction (v_end ⊥ v_start, then renormalized).
        outside = v_end - cos_start_end * v_start
        v_outside = _normed(outside)

        block = {
            "score_start": float(score_start),
            "score_end": float(score_end),
            "score_gain": float(score_end - score_start),
            "r_sk_start": float(r_sk_s), "r_sk_end": float(r_sk_e),
            "raw_g1_start": float(raw_g1_s), "raw_g1_end": float(raw_g1_e),
            "raw_g2_start": float(raw_g2_s), "raw_g2_end": float(raw_g2_e),
            "hm3_start": float(sat_s), "hm3_end": float(sat_e),
            "cos_start_end": cos_start_end,
            "cos2_start_end": cos2_start_end,
            "stop": result.get("stop"),
            "decomp_v_end": {
                "top_svd_sketch": _top_alignments(v_end, svd_sketch, k_svd),
                "top_svd_cur": _top_alignments(v_end, svd_cur, k_svd),
                "top_svd_fut": _top_alignments(v_end, svd_fut, k_svd),
                "top_svd_union": _top_alignments(v_end, svd_union, k_svd),
                "top_rows_cur": _row_alignments(A_cur, v_end, k_rows),
                "top_rows_fut": _row_alignments(A_fut, v_end, k_rows),
            },
            "decomp_v_outside": None,
        }
        if v_outside is not None:
            block["decomp_v_outside"] = {
                "top_svd_sketch": _top_alignments(v_outside, svd_sketch, k_svd),
                "top_svd_cur": _top_alignments(v_outside, svd_cur, k_svd),
                "top_svd_fut": _top_alignments(v_outside, svd_fut, k_svd),
                "top_svd_union": _top_alignments(v_outside, svd_union, k_svd),
                "top_rows_cur": _row_alignments(A_cur, v_outside, k_rows),
                "top_rows_fut": _row_alignments(A_fut, v_outside, k_rows),
            }
        # Cross-alignment to the OTHER oracle.
        v_other = oracle_v2_proj if label == "oracle_v1_proj" else oracle_v1_proj
        if v_other is not None:
            block["cos2_v_end_to_other_oracle"] = float(np.dot(v_end, v_other) ** 2)
        out["starts"][label] = block

    return out


def fmt_topk(items, fmt_score="{:.3f}"):
    return ", ".join(f"#{j} ({fmt_score.format(c)})" for j, c, *rest in items)


def fmt_rows(items):
    return ", ".join(f"row#{j} cos²={c:.3f} ‖row‖={n:.3f}" for j, c, n in items)


def write_text(out_path, results):
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(f"==== matrix={entry['matrix']}  block={entry['block']}  "
                    f"union_dim={entry['n_union']}  c_sk={entry['c_sk']:.4e} ====\n")
            for label, b in entry["starts"].items():
                f.write(f"  --- start = {label} ---\n")
                f.write(f"    score:   start={b['score_start']:.5e}  end={b['score_end']:.5e}  "
                        f"Δ={b['score_gain']:+.4e}\n")
                f.write(f"    hm3:     start={b['hm3_start']:.4f}      end={b['hm3_end']:.4f}\n")
                f.write(f"    r_sk:    start={b['r_sk_start']:+.4f}      end={b['r_sk_end']:+.4f}\n")
                f.write(f"    raw_g1:  start={b['raw_g1_start']:.4f}      end={b['raw_g1_end']:.4f}\n")
                f.write(f"    raw_g2:  start={b['raw_g2_start']:.4f}      end={b['raw_g2_end']:.4f}\n")
                f.write(f"    cos²(start, end) = {b['cos2_start_end']:.4f}   "
                        f"(angle moved → outside-mass = {1.0 - b['cos2_start_end']:.4f})\n")
                if "cos2_v_end_to_other_oracle" in b:
                    f.write(f"    cos²(end, other_oracle_proj) = "
                            f"{b['cos2_v_end_to_other_oracle']:.4f}\n")
                f.write(f"    stop: {b['stop']}\n")
                d = b["decomp_v_end"]
                f.write(f"    v_end  ↔ SVD_sketch: {fmt_topk(d['top_svd_sketch'])}\n")
                f.write(f"    v_end  ↔ SVD_cur:    {fmt_topk(d['top_svd_cur'])}\n")
                f.write(f"    v_end  ↔ SVD_fut:    {fmt_topk(d['top_svd_fut'])}\n")
                f.write(f"    v_end  ↔ SVD_union:  {fmt_topk(d['top_svd_union'])}\n")
                f.write(f"    v_end  ↔ rows(A_cur):{fmt_rows(d['top_rows_cur'])}\n")
                f.write(f"    v_end  ↔ rows(A_fut):{fmt_rows(d['top_rows_fut'])}\n")
                if b["decomp_v_outside"]:
                    do = b["decomp_v_outside"]
                    f.write(f"    v_outside (v_end ⊥ v_start, ‖outside‖={1.0 - b['cos2_start_end']:.4f}):\n")
                    f.write(f"      ↔ SVD_sketch: {fmt_topk(do['top_svd_sketch'])}\n")
                    f.write(f"      ↔ SVD_cur:    {fmt_topk(do['top_svd_cur'])}\n")
                    f.write(f"      ↔ SVD_fut:    {fmt_topk(do['top_svd_fut'])}\n")
                    f.write(f"      ↔ SVD_union:  {fmt_topk(do['top_svd_union'])}\n")
                    f.write(f"      ↔ rows(A_cur):{fmt_rows(do['top_rows_cur'])}\n")
                    f.write(f"      ↔ rows(A_fut):{fmt_rows(do['top_rows_fut'])}\n")
            f.write("\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrices", nargs="+", default=["static-cex", "mixed-tail-sharp", "diffuse-diffuse"])
    p.add_argument("--blocks", nargs="+", type=int, default=[6, 12, 31])
    p.add_argument("--out-prefix", default="summary/r_sk_g_oracle_init_probe")
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
    p.add_argument("--union-maxit", type=int, default=200)
    p.add_argument("--union-tol", type=float, default=1e-12)
    p.add_argument("--r-sig", type=int, default=2)
    p.add_argument("--alpha-sig", type=float, default=0.003)
    p.add_argument("--alpha-tail", type=float, default=0.0145)
    p.add_argument("--tail-scale", type=float, default=0.99)
    p.add_argument("--sigma1", type=float, default=0.991)
    p.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    results = []
    for matrix in args.matrices:
        for block in args.blocks:
            r = probe_one(args, matrix, block)
            results.append(r)
            print(f"done matrix={matrix} block={block}")
    json_path = args.out_prefix + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    txt_path = args.out_prefix + ".txt"
    write_text(txt_path, results)
    print(f"wrote {json_path} {txt_path}")


if __name__ == "__main__":
    main()
