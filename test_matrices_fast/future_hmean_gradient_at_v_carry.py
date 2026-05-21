"""Compute the future-HM gradient at v_carry, u (=window-projection of v_carry, normalized),
and v_inwindow on the tangent sphere ⊥ v1.

Question: is v_carry a local max of future-HM, or is the gradient non-zero
with a component pointing toward the window?

By the identity future_HM(v) = ||P v||^2 * future_HM(u) where P projects onto
rowspan([A_w; A_{w+1}]) and u = P v / ||P v||, we expect non-zero gradient
at v_carry with a component aligned with P v_carry (which lies in the window).

Decompose the tangent gradient at v_carry into:
  - in-window-tangent component (gradient projected into rowspan(window) ∩ v1^perp ∩ v_carry^perp)
  - out-of-window-tangent component (the rest)

Compare magnitudes.
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


def tangent_proj(grad, v, v1=None):
    g = grad - float(v @ grad) * v
    if v1 is not None:
        v1n = v1 / max(float(np.linalg.norm(v1)), 1e-30)
        g = g - float(v1n @ g) * v1n
    return g


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
    rows = []

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

        if state is not None and v2 is not None:
            # In-window optimum
            union = np.vstack([A_cur, A_fut]).astype(np.float64, copy=False)
            Q_window = rowspace_basis(union)
            B_union = orth_basis_against(Q_window, v1)
            starts_v = [V_default[:, 1]]
            if V_exact.shape[1] >= 2:
                starts_v.append(V_exact[:, 1])
            if diag.get("Vbasis_final") is not None:
                Vbasis = np.asarray(diag["Vbasis_final"], dtype=np.float64)
                for j in range(min(Vbasis.shape[1], 8)):
                    starts_v.append(Vbasis[:, j])
            best = optimize_future_hmean_in_basis(
                A_cur, A_fut, B_union, starts_v,
                np.random.default_rng(args.seed + 1009 * block_id),
                maxit=80, tol=1e-9, random_starts=12,
            )
            v_inwindow = None if best is None else best["vec"]

            # u = window-projection of v_carry, renormalized
            Pv = Q_window @ (Q_window.T @ v2)
            mass_v_in_win = float(np.linalg.norm(Pv) ** 2)
            u = Pv / max(np.linalg.norm(Pv), 1e-30) if mass_v_in_win > 1e-30 else None

            rec = {"matrix": args.matrix, "block": block_id, "rows_seen": mid0,
                   "mass_v_carry_in_window": mass_v_in_win}

            def _eval(name, v):
                if v is None:
                    rec[f"{name}_score"] = np.nan
                    rec[f"{name}_grad_norm_full"] = np.nan
                    rec[f"{name}_tangent_grad_norm"] = np.nan
                    rec[f"{name}_tangent_grad_into_window"] = np.nan
                    rec[f"{name}_tangent_grad_out_of_window"] = np.nan
                    return
                val, grad, a, b, rel = future_hmean_value_grad(np.asarray(A_cur, dtype=np.float64),
                                                                np.asarray(A_fut, dtype=np.float64),
                                                                v)
                gtan = tangent_proj(grad, v, v1)
                # Split gtan into "in-window" and "out-of-window" parts
                # (in tangent space ⊥ v, ⊥ v1)
                gtan_in = Q_window @ (Q_window.T @ gtan)
                gtan_out = gtan - gtan_in
                rec[f"{name}_score"] = float(val)
                rec[f"{name}_a_gain"] = float(a)
                rec[f"{name}_b_gain"] = float(b)
                rec[f"{name}_relH"] = float(rel)
                rec[f"{name}_grad_norm_full"] = float(np.linalg.norm(grad))
                rec[f"{name}_tangent_grad_norm"] = float(np.linalg.norm(gtan))
                rec[f"{name}_tangent_grad_into_window"] = float(np.linalg.norm(gtan_in))
                rec[f"{name}_tangent_grad_out_of_window"] = float(np.linalg.norm(gtan_out))

            _eval("v_carry", v2)
            _eval("u_proj", u)
            _eval("v_inwindow", v_inwindow)

            # Sanity check identity: future_HM(v_carry) = ||P v||^2 * future_HM(u)
            if u is not None and v_inwindow is not None:
                rec["identity_check"] = (
                    rec["v_carry_score"] / max(rec["u_proj_score"], 1e-300)
                    if rec["u_proj_score"] > 0 else np.nan
                )
                rec["expected_ratio"] = mass_v_in_win

            rows.append(rec)

        # Update state
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
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="mixed-tail-sharp")
    p.add_argument("--out-prefix", default="summary/future_hmean_gradient_at_v_carry")
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
    rows = run_one(args)
    if not rows:
        print("no rows")
        return
    json_path = args.out_prefix + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True, default=float)
    csv_path = args.out_prefix + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path} {json_path}")


if __name__ == "__main__":
    main()
