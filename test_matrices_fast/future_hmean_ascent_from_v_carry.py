"""Run future-HM gradient ascent from v_carry on the unit sphere ⊥ v1.
Track score and rowspace mass of v over iterations to characterize the path.
"""
import argparse
import csv
import json

import numpy as np

import cex_restricted_space_probe as probe
from second_slot_tail_bias_diagnostic import make_state
from future_hmean_optimizer_diagnostic import (
    optimize_future_hmean_in_basis,
    rowspace_basis,
    orth_basis_against,
    future_hmean_value_grad,
)


def project_to_v1_perp(v, v1):
    v1n = v1 / max(float(np.linalg.norm(v1)), 1e-30)
    out = v - float(v1n @ v) * v1n
    n = float(np.linalg.norm(out))
    if n <= 1e-30:
        return None
    return out / n


def ascend_future_hm(v0, v1, A_cur, A_fut, maxit=2000, tol=1e-10, log_every=50, Q_window=None):
    v = project_to_v1_perp(np.asarray(v0, dtype=np.float64), v1)
    if v is None:
        return None
    val, grad, a, b, rel = future_hmean_value_grad(A_cur, A_fut, v)
    log = []
    for it in range(maxit):
        gtan = grad - float(v @ grad) * v
        v1n = v1 / max(float(np.linalg.norm(v1)), 1e-30)
        gtan = gtan - float(v1n @ gtan) * v1n
        gnorm = float(np.linalg.norm(gtan))
        if it % log_every == 0 or gnorm <= tol:
            mass = float(np.linalg.norm(Q_window.T @ v) ** 2) if Q_window is not None else np.nan
            log.append({"iter": it, "score": float(val), "a": float(a), "b": float(b),
                        "relH": float(rel), "grad_norm": gnorm,
                        "window_mass": mass})
        if gnorm <= tol:
            break
        alpha = 1.0
        improved = False
        for _ in range(60):
            vt = v + alpha * gtan / max(gnorm, 1e-30)
            vt_p = project_to_v1_perp(vt, v1)
            if vt_p is None:
                alpha *= 0.5
                continue
            val_t, grad_t, a_t, b_t, rel_t = future_hmean_value_grad(A_cur, A_fut, vt_p)
            if val_t > val + 1e-14:
                v, val, grad, a, b, rel = vt_p, val_t, grad_t, a_t, b_t, rel_t
                improved = True
                break
            alpha *= 0.5
        if not improved:
            mass = float(np.linalg.norm(Q_window.T @ v) ** 2) if Q_window is not None else np.nan
            log.append({"iter": it + 1, "score": float(val), "a": float(a), "b": float(b),
                        "relH": float(rel), "grad_norm": gnorm,
                        "window_mass": mass, "stopped": "line_search_fail"})
            break
    return v, val, log


def run_one(args):
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=args.matrix, n=args.n, preset=args.preset, seed=args.seed,
        r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
        shuffle_rows=True, row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, dtype=np.float64)
    V_exact = np.asarray(V_exact, dtype=np.float64)
    rank = int(args.rank)
    half_win = int(args.half_win)
    state = None
    old_row_memory = None
    out_blocks = []

    for block_id, start0 in enumerate(range(0, A.shape[0] - half_win, half_win), start=1):
        if args.max_pairs is not None and block_id > args.max_pairs:
            break
        mid0 = start0 + half_win
        end0 = min(mid0 + half_win, A.shape[0])
        if end0 - mid0 < half_win:
            break
        A_cur = np.asarray(A[start0:mid0, :], dtype=work_dtype)
        A_fut = np.asarray(A[mid0:end0, :], dtype=work_dtype)
        if state is None:
            M_gain = A_cur
            V_init = probe.row_norm_seed(A_cur, rank)
            rows_seen = A_cur.shape[0]
        else:
            B_top = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
            M_gain = np.vstack([B_top, A_cur]).astype(work_dtype, copy=False)
            V_init = probe.row_norm_seed(A_cur, rank)
            rows_seen = state["rows_seen"] + A_cur.shape[0]
        V_score, _, _, _, diag = probe.entropy_iter_basis_forget(
            M_gain=M_gain, active_r=rank, rows_ref=A.shape[0],
            V_init=np.asarray(V_init, dtype=work_dtype),
            q0=args.q0, qmax=args.qmax, krylov_depth=args.krylov_depth,
            residual_tol=args.residual_tol, expansion_maxit=args.expansion_maxit,
            num_restarts=args.num_restarts, maxit=args.maxit, tol=args.tol,
            rng=np.random.default_rng(args.seed), verbose=False,
            state_prev=state, A_block=A_cur, rows_total=rows_seen,
            reduced_optimizer="cex", basis_selection="greedy", work_dtype=work_dtype,
            expansion_direction="residual", reuse_line_search_grad=True,
            expansion_warm_start=True,
            post_expansion_maxit=args.post_expansion_maxit,
            score_variant="combined", old_row_memory=old_row_memory,
            combined_rank=None, patience=args.patience,
            patience_rel_tol=args.patience_rel_tol,
        )
        V_default = np.ascontiguousarray(np.asarray(V_score[:, :rank], dtype=np.float64))
        v1 = V_default[:, 0]
        v2 = V_default[:, 1] if V_default.shape[1] >= 2 else None
        if state is not None and v2 is not None and block_id in args.blocks:
            union = np.vstack([A_cur, A_fut]).astype(np.float64, copy=False)
            Q_window = rowspace_basis(union)
            B_union = orth_basis_against(Q_window, v1)
            starts_v = [V_default[:, 1]]
            if V_exact.shape[1] >= 2:
                starts_v.append(V_exact[:, 1])
            best = optimize_future_hmean_in_basis(
                np.asarray(A_cur, dtype=np.float64),
                np.asarray(A_fut, dtype=np.float64),
                B_union, starts_v,
                np.random.default_rng(args.seed + 1009 * block_id),
                maxit=80, tol=1e-9, random_starts=8,
            )
            v_inwindow = None if best is None else best["vec"]

            v_end, sc_end, log = ascend_future_hm(
                v2, v1,
                np.asarray(A_cur, dtype=np.float64),
                np.asarray(A_fut, dtype=np.float64),
                maxit=args.ascent_maxit, tol=1e-12, log_every=args.log_every,
                Q_window=Q_window,
            )
            align_to_inwindow = (
                np.nan if v_end is None or v_inwindow is None
                else float(abs(np.dot(v_end, v_inwindow)) ** 2)
            )
            align_to_v_carry = (
                np.nan if v_end is None
                else float(abs(np.dot(v_end, v2)) ** 2)
            )
            inwindow_score = (
                np.nan if v_inwindow is None
                else float(future_hmean_value_grad(
                    np.asarray(A_cur, dtype=np.float64),
                    np.asarray(A_fut, dtype=np.float64),
                    v_inwindow,
                )[0])
            )
            out_blocks.append({
                "matrix": args.matrix,
                "block": block_id,
                "v_inwindow_score": inwindow_score,
                "ascent_end_score": float(sc_end) if v_end is not None else np.nan,
                "ascent_end_align_v_inwindow": align_to_inwindow,
                "ascent_end_align_v_carry": align_to_v_carry,
                "log": log,
            })

        score_selected = np.zeros(rank, dtype=float)
        H_selected = np.zeros(rank, dtype=float)
        for j in range(rank):
            score_selected[j], _, H_selected[j] = probe.score_full_vector_details_forget(
                M_gain, A_cur, V_default[:, j], A.shape[0], state_prev=state,
                score_variant="combined", old_row_memory=old_row_memory,
            )
        state, V_r, _ = make_state(M_gain, V_default, H_selected, score_selected, rows_seen)
        old_row_memory, _ = probe.select_old_row_memory(
            np.asarray(A[:mid0, :], dtype=work_dtype),
            V_r.astype(work_dtype, copy=False),
            args.old_memory_size if args.old_memory_size > 0 else half_win,
            np.random.default_rng(args.seed + end0), return_indices=True,
        )
    return out_blocks


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="mixed-tail-sharp")
    p.add_argument("--out-prefix", default="summary/future_hmean_ascent_from_v_carry")
    p.add_argument("--blocks", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--half-win", type=int, default=32)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--preset", default="fast")
    p.add_argument("--seed", type=int, default=0)
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
    p.add_argument("--ascent-maxit", type=int, default=4000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--max-pairs", type=int, default=8)
    p.add_argument("--r-sig", type=int, default=2)
    p.add_argument("--alpha-sig", type=float, default=0.003)
    p.add_argument("--alpha-tail", type=float, default=0.0145)
    p.add_argument("--tail-scale", type=float, default=0.99)
    p.add_argument("--sigma1", type=float, default=0.991)
    p.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    return p.parse_args()


def main():
    args = parse_args()
    out = run_one(args)
    json_path = args.out_prefix + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True, default=float)
    print(f"wrote {json_path}")
    for blk in out:
        print(f"\nblock {blk['block']}: v_inwindow_score={blk['v_inwindow_score']:.6e}")
        print(f"  ascent end score={blk['ascent_end_score']:.6e}, align²(inwindow)={blk['ascent_end_align_v_inwindow']:.4f}, align²(v_carry)={blk['ascent_end_align_v_carry']:.4f}")
        for entry in blk["log"]:
            print(f"  it={entry['iter']:5d}  score={entry['score']:.4e}  a={entry['a']:.4e}  b={entry['b']:.4e}  relH={entry['relH']:.4f}  ||g||={entry['grad_norm']:.4e}  win_mass={entry['window_mass']:.4f}")


if __name__ == "__main__":
    main()
