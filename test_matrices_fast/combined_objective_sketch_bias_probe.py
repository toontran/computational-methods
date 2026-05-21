"""Sketch-vs-block gain bias probe for the combined streaming objective.

Goal: collect evidence that the combined objective
    score(v) = gain2(v) * phi(v)
        gain2(v) = ||M_gain v||^2 = ||A_sketch v||^2 + ||A_cur v||^2
is biased toward the sketch term — i.e. it preferentially picks
directions with large ||A_sketch v||^2 even when an oracle direction
would be better in the current block.

For the combined objective there is one sketch (B_top = state.s · state.V^T)
and one block (A_cur). For each of four candidates per block,

    candidates = { combined_v1, combined_v2,           # streaming-optimizer V_default columns
                   oracle_v1_proj, oracle_v2_proj }    # V_exact[:,j] projected onto rowspan([A_sk; A_cur])

we report

    gain_sketch = ||A_sketch v||^2
    gain_block  = ||A_cur    v||^2
    gain_total  = gain_sketch + gain_block
    sketch_frac = gain_sketch / gain_total
    phi, score_total
    align_v1 = (v · V_exact[:,0])^2,  align_v2 = (v · V_exact[:,1])^2

Reuses snapshots from `hmean_evidence_score.stream_to_block`.
"""

import argparse
import csv
import os
import time

import numpy as np

import cex_restricted_space_probe as probe
from future_hmean_optimizer_diagnostic import rowspace_basis
from hmean_evidence_score import per_block_constants, stream_to_block


def project_unit(vec, B):
    if vec is None or B is None or B.size == 0:
        return None
    p = B @ (B.T @ vec)
    nv = float(np.linalg.norm(p))
    return None if nv <= 1e-30 else p / nv


def deflate_unit(v, v1):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    nv = float(np.linalg.norm(v))
    if nv <= 1e-30:
        return None
    v = v / nv
    r = v - float(np.dot(v1, v)) * v1
    nr = float(np.linalg.norm(r))
    return None if nr <= 1e-30 else r / nr


def joint_optimizer_at_snapshot(args, A_shape0, snap, work_dtype, rank):
    """Run basis_selection='joint' on the same snapshot M_gain.

    Diagnostic only: does NOT advance the carry. The streaming carry is
    still derived from the greedy optimizer in `stream_to_block`.
    """
    M_gain = np.asarray(snap["M_gain"], dtype=work_dtype)
    A_cur = np.asarray(snap["A_cur"], dtype=work_dtype)
    state_prev = snap["state"]
    old_row_memory = snap["old_row_memory"]
    rows_seen = snap["rows_seen"]

    V_init = probe.row_norm_seed(A_cur, rank)
    V_score, _, _, _, _ = probe.entropy_iter_basis_forget(
        M_gain=M_gain,
        active_r=rank,
        rows_ref=A_shape0,
        V_init=np.asarray(V_init, dtype=work_dtype),
        q0=args.q0, qmax=args.qmax, krylov_depth=args.krylov_depth,
        residual_tol=args.residual_tol, expansion_maxit=args.expansion_maxit,
        num_restarts=max(args.num_restarts, 4),
        maxit=args.maxit, tol=args.tol,
        rng=np.random.default_rng(args.seed + 99991),
        verbose=False, state_prev=state_prev, A_block=A_cur, rows_total=rows_seen,
        reduced_optimizer="cex",
        basis_selection="joint",
        joint_solver="riemannian",
        joint_warm_start_greedy=True,
        joint_default_svd_start=True,
        work_dtype=work_dtype, expansion_direction="residual",
        reuse_line_search_grad=True, expansion_warm_start=True,
        post_expansion_maxit=args.post_expansion_maxit,
        score_variant="combined", old_row_memory=old_row_memory,
        combined_rank=None,
        patience=args.patience,
        patience_rel_tol=args.patience_rel_tol,
    )
    return np.ascontiguousarray(np.asarray(V_score[:, :rank], dtype=np.float64))


def gains_for(v, A_sketch, A_cur, V_state=None):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    nv = float(np.linalg.norm(v))
    if nv <= 1e-30:
        return None
    v = v / nv
    if A_sketch is None or A_sketch.size == 0:
        gain_sketch = 0.0
    else:
        ys = A_sketch @ v
        gain_sketch = float(np.dot(ys, ys))
    yc = A_cur @ v
    gain_block = float(np.dot(yc, yc))
    if V_state is None or V_state.size == 0:
        state_align = 0.0
    else:
        proj = V_state.T @ v
        state_align = float(np.dot(proj, proj))
    return v, gain_sketch, gain_block, state_align


def slot2_landscape_block1(snap, V_exact, A_shape0):
    """E2: at block 1, evaluate combined-score in the deflation complement
    of combined_v_1 along several candidate directions to disentangle
    'phi-bias' vs 'orthogonality rotation' for the slot-2 failure."""
    A_cur = np.asarray(snap["A_cur"], dtype=np.float64)
    A_sketch = snap["A_sketch"] if snap["A_sketch"].size else None
    M_gain = np.asarray(snap["M_gain"], dtype=np.float64)
    V_default = snap["V_default"]
    state = snap["state"]
    old_row_memory = snap["old_row_memory"]

    union_stack = np.asarray(A_cur) if A_sketch is None else np.vstack([A_sketch, A_cur])
    B_union = rowspace_basis(union_stack)

    oracle_v1 = V_exact[:, 0] / max(np.linalg.norm(V_exact[:, 0]), 1e-30)
    oracle_v2 = V_exact[:, 1] / max(np.linalg.norm(V_exact[:, 1]), 1e-30)
    oracle_v1_proj = project_unit(oracle_v1, B_union)
    oracle_v2_proj = project_unit(oracle_v2, B_union)

    v1 = V_default[:, 0] / max(np.linalg.norm(V_default[:, 0]), 1e-30)
    samples = {}
    samples["combined_v2"] = V_default[:, 1] / max(np.linalg.norm(V_default[:, 1]), 1e-30)
    if oracle_v1_proj is not None:
        d = deflate_unit(oracle_v1_proj, v1)
        if d is not None:
            samples["oracle_v1_proj_perp"] = d
    if oracle_v2_proj is not None:
        d = deflate_unit(oracle_v2_proj, v1)
        if d is not None:
            samples["oracle_v2_proj_perp"] = d
    rng = np.random.default_rng(0)
    n = B_union.shape[0]
    for k in range(5):
        r = rng.standard_normal(n)
        r = B_union @ (B_union.T @ r)  # project onto union span
        d = deflate_unit(r, v1)
        if d is not None:
            samples[f"random_{k}"] = d

    rows = []
    for label, v in samples.items():
        if v is None:
            continue
        out = gains_for(v, A_sketch, A_cur, V_state=None)
        if out is None:
            continue
        v_unit, gain_sketch, gain_block, _ = out
        gain_total = gain_sketch + gain_block
        comp = probe.combined_score_component_details(
            M_gain, A_cur, v_unit, A_shape0,
            state_prev=state, old_row_memory=old_row_memory,
        )
        algn_o2 = (
            float(np.dot(v_unit, oracle_v2_proj) ** 2)
            if oracle_v2_proj is not None else float("nan")
        )
        algn_v1 = float(np.dot(v_unit, v1) ** 2)
        rows.append({
            "label": label,
            "gain_total": gain_total,
            "phi": comp["phi"],
            "score": comp["score_total"],
            "algn_v1_check": algn_v1,
            "algn_o2": algn_o2,
        })
    return rows


def analyze_block(matrix, A, V_exact, snap, block_id, half_win):
    A_cur = snap["A_cur"]
    A_sketch = snap["A_sketch"] if snap["A_sketch"].size else None
    M_gain = snap["M_gain"]
    state = snap["state"]
    old_row_memory = snap["old_row_memory"]
    V_default = snap["V_default"]

    if A_sketch is not None:
        union_stack = np.vstack([A_sketch, A_cur])
    else:
        union_stack = np.asarray(A_cur)
    B_union = rowspace_basis(union_stack)

    oracle_v1 = V_exact[:, 0] / max(np.linalg.norm(V_exact[:, 0]), 1e-30)
    oracle_v2 = V_exact[:, 1] / max(np.linalg.norm(V_exact[:, 1]), 1e-30)
    oracle_v1_proj = project_unit(oracle_v1, B_union)
    oracle_v2_proj = project_unit(oracle_v2, B_union)

    V_state = None
    if state is not None:
        Vs = state.get("V")
        if Vs is not None:
            Vs_arr = np.asarray(Vs, dtype=np.float64)
            if Vs_arr.size:
                V_state = Vs_arr

    V_joint = snap.get("V_joint")
    candidates = {
        "combined_v1":     V_default[:, 0],
        "combined_v2":     V_default[:, 1],
        "joint_v1":        None if V_joint is None else V_joint[:, 0],
        "joint_v2":        None if V_joint is None else V_joint[:, 1],
        "oracle_v1_proj":  oracle_v1_proj,
        "oracle_v2_proj":  oracle_v2_proj,
    }

    rows = []
    for label, v in candidates.items():
        if v is None:
            continue
        out = gains_for(v, A_sketch, A_cur, V_state=V_state)
        if out is None:
            continue
        v_unit, gain_sketch, gain_block, state_align = out
        gain_total = gain_sketch + gain_block
        sketch_frac = gain_sketch / max(gain_total, 1e-30)
        comp = probe.combined_score_component_details(
            M_gain, A_cur, v_unit, A.shape[0],
            state_prev=state, old_row_memory=old_row_memory,
        )
        align_o1 = (
            float(np.dot(v_unit, oracle_v1_proj) ** 2)
            if oracle_v1_proj is not None else float("nan")
        )
        align_o2 = (
            float(np.dot(v_unit, oracle_v2_proj) ** 2)
            if oracle_v2_proj is not None else float("nan")
        )
        rows.append({
            "matrix": matrix, "block": block_id, "label": label,
            "gain_sketch": gain_sketch,
            "gain_block": gain_block,
            "gain_total": gain_total,
            "sketch_frac": sketch_frac,
            "phi": comp["phi"],
            "score_total": comp["score_total"],
            "state_align": state_align,
            "align_o1": align_o1,
            "align_o2": align_o2,
        })

    consts = per_block_constants(A, block_id, half_win)
    info = {
        "matrix": matrix, "block": block_id, "half_win": half_win,
        "N_sk": consts["N_sk"], "sk_F2": consts["sk_F2"],
        "cur_F2": consts["cur_F2"],
        "sketch_rows": A_sketch.shape[0] if A_sketch is not None else 0,
        "block_rows": int(A_cur.shape[0]),
        "union_dim": int(B_union.shape[1]),
    }
    return info, rows


def run_matrix(args, matrix, blocks_to_report):
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=matrix, n=args.n, preset=args.preset, seed=args.seed,
        r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
        shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, np.float64)
    V_exact = np.asarray(V_exact, np.float64)
    target = max(blocks_to_report)
    snapshots = stream_to_block(args, A, V_exact, work_dtype, int(args.rank), target, set(blocks_to_report))
    out_rows = []
    out_info = {}
    landscape_rows = {}
    for b in sorted(blocks_to_report):
        if b not in snapshots:
            continue
        snap = snapshots[b]
        snap["V_joint"] = joint_optimizer_at_snapshot(args, A.shape[0], snap, work_dtype, int(args.rank))
        info, rows = analyze_block(matrix, A, V_exact, snap, b, int(args.half_win))
        out_info[b] = info
        out_rows.extend(rows)
        if b == 1:
            landscape_rows[b] = slot2_landscape_block1(snap, V_exact, A.shape[0])
    return out_info, out_rows, landscape_rows


def write_landscape(path, landscape_rows):
    if not landscape_rows:
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write("# E2: slot-2 landscape inside the deflation complement of combined_v_1.\n")
        f.write("# All directions below are unit, orthogonal to combined_v_1.\n\n")
        for block_id in sorted(landscape_rows.keys()):
            f.write(f"== block {block_id} ==\n")
            f.write(
                f"  {'label':<22} {'gain_tot':>10} {'phi':>10} {'score':>11} "
                f"{'algn_v1_chk':>12} {'algn_o2':>9}\n"
            )
            for r in landscape_rows[block_id]:
                f.write(
                    f"  {r['label']:<22} {r['gain_total']:>10.4f} "
                    f"{r['phi']:>10.4f} {r['score']:>11.4f} "
                    f"{r['algn_v1_check']:>12.2e} {r['algn_o2']:>9.4f}\n"
                )
            f.write("\n")


def write_text(path, infos, rows):
    by_block = {}
    for r in rows:
        by_block.setdefault(r["block"], []).append(r)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Combined-objective sketch-vs-block gain decomposition.\n")
        f.write("# score(v) = (||A_sketch v||^2 + ||A_cur v||^2) * phi(v)\n")
        f.write("# sketch_frac = gain_sketch / (gain_sketch + gain_block).\n\n")
        for block_id in sorted(by_block.keys()):
            info = infos[block_id]
            f.write(
                f"== block {block_id}  matrix={info['matrix']}  "
                f"sketch_rows={info['sketch_rows']}  block_rows={info['block_rows']}  "
                f"sk_F2={info['sk_F2']:.4e}  cur_F2={info['cur_F2']:.4e}  "
                f"union_dim={info['union_dim']} ==\n"
            )
            f.write(
                f"  {'label':<18} {'gain_sk':>10} {'gain_blk':>10} {'gain_tot':>10} "
                f"{'sk_frac':>8} {'st_algn':>8} {'phi':>8} {'score':>11} "
                f"{'algn_o1':>9} {'algn_o2':>9}\n"
            )
            for r in by_block[block_id]:
                f.write(
                    f"  {r['label']:<18} "
                    f"{r['gain_sketch']:>10.4f} {r['gain_block']:>10.4f} {r['gain_total']:>10.4f} "
                    f"{r['sketch_frac']:>8.4f} {r['state_align']:>8.4f} "
                    f"{r['phi']:>8.4f} {r['score_total']:>11.4f} "
                    f"{r['align_o1']:>9.4f} {r['align_o2']:>9.4f}\n"
                )
            picks = {r["label"]: r for r in by_block[block_id]}
            for o, c in [("oracle_v1_proj", "combined_v1"), ("oracle_v2_proj", "combined_v2")]:
                ro = picks.get(o); rc = picks.get(c)
                if ro is None or rc is None:
                    continue
                # Symmetric decomposition of Δscore = Δ(gain2 * phi) into
                # gain-driven and phi-driven contributions:
                #   Δscore = Δgain2 * mean(phi) + Δphi * mean(gain2)
                d_gain = ro['gain_total'] - rc['gain_total']
                d_phi = ro['phi'] - rc['phi']
                mean_phi = 0.5 * (ro['phi'] + rc['phi'])
                mean_gain = 0.5 * (ro['gain_total'] + rc['gain_total'])
                via_gain = d_gain * mean_phi
                via_phi = d_phi * mean_gain
                d_score = ro['score_total'] - rc['score_total']
                f.write(
                    f"  Δ({o} − {c}): "
                    f"gain_sk={ro['gain_sketch'] - rc['gain_sketch']:+.4f}  "
                    f"gain_blk={ro['gain_block'] - rc['gain_block']:+.4f}  "
                    f"score={d_score:+.4f}  "
                    f"(via_gain={via_gain:+.4f}, via_phi={via_phi:+.4f})  "
                    f"algn_o1={ro['align_o1'] - rc['align_o1']:+.4f}  "
                    f"algn_o2={ro['align_o2'] - rc['align_o2']:+.4f}\n"
                )
            f.write("\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="mixed-tail-sharp")
    p.add_argument("--out-prefix", default="summary/combined_obj_sketch_bias")
    p.add_argument("--blocks", nargs="+", type=int, default=[2, 6, 12, 31])
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
    p.add_argument("--r-sig", type=int, default=2)
    p.add_argument("--alpha-sig", type=float, default=0.003)
    p.add_argument("--alpha-tail", type=float, default=0.0145)
    p.add_argument("--tail-scale", type=float, default=0.99)
    p.add_argument("--sigma1", type=float, default=0.991)
    p.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    infos, rows, landscape_rows = run_matrix(args, args.matrix, args.blocks)
    print(f"done matrix={args.matrix} blocks={sorted(infos.keys())} elapsed={time.time()-t0:.2f}s")

    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    csv_path = args.out_prefix + f"_{args.matrix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    txt_path = args.out_prefix + f"_{args.matrix}.txt"
    write_text(txt_path, infos, rows)
    landscape_path = args.out_prefix + f"_{args.matrix}_landscape.txt"
    write_landscape(landscape_path, landscape_rows)
    print(f"wrote {csv_path}, {txt_path}, {landscape_path}")


if __name__ == "__main__":
    main()
