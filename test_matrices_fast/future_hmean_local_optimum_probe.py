"""Probe whether the streaming v2 is a local optimum, and characterize the
combined-score landscape between v_carry (streaming v2) and v_inwindow
(in-window future-HM peak).

For each block:
  1. Run the streaming optimizer (just like future_hmean_optimizer_diagnostic).
  2. Find v_inwindow = in-window future-HM optimum (basis = rowspan([A_w; A_{w+1}]) deflated against v1).
  3. Decompose M_gain into B_top (sketch) and A_cur. Report state["s"][1]^2.
  4. Evaluate combined-score and its TANGENT gradient norm (sphere, orthogonal to v1) at:
       v_carry, v_inwindow, and v_inwindow_proj (v_inwindow projected onto M_gain rowspace,
       which is what would actually be reachable from the optimizer's search basis).
  5. Walk a great-circle arc between v_carry and v_inwindow_proj and report combined-score
     along the arc — does the score have a saddle/barrier between them?
  6. Run gradient ascent on the FULL n-dim sphere (no basis restriction) starting from
     each of v_carry, v_inwindow, v_inwindow_proj. Report where each ends up.
  7. Run gradient ascent on full sphere starting from many random directions.
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
)


def _onb(M, tol=1e-10):
    M = np.asarray(M, dtype=np.float64)
    if M.size == 0 or M.shape[1] == 0:
        return np.zeros((M.shape[0] if M.ndim == 2 else 0, 0), dtype=np.float64)
    Q, R = np.linalg.qr(M)
    diag = np.abs(np.diag(R))
    if diag.size == 0:
        return np.zeros((M.shape[0], 0), dtype=np.float64)
    keep = diag > max(float(diag.max()) * tol, 1e-30)
    return np.ascontiguousarray(Q[:, keep], dtype=np.float64)


def project_to_v1_perp(v, v1):
    v1n = v1 / max(float(np.linalg.norm(v1)), 1e-30)
    out = v - float(v1n @ v) * v1n
    n = float(np.linalg.norm(out))
    if n <= 1e-30:
        return None
    return out / n


def combined_score_and_grad_full(M_gain, A_block, v, rows_ref, state_prev, old_row_memory):
    """Combined score and FULL n-dim gradient at v (||v||=1)."""
    M = np.asarray(M_gain, dtype=np.float64)
    A = np.asarray(A_block, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    R = None if old_row_memory is None else np.asarray(old_row_memory, dtype=np.float64)

    g = M @ v
    gain2 = max(float(np.dot(g, g)), 1e-30)
    grad_gain2 = 2.0 * (M.T @ g)

    y = A @ v
    pooled_y2 = float(np.dot(y, y))
    pooled_y4 = float(np.sum((y * y) ** 2))
    rows_block = A.shape[0]
    rows_entropy = rows_block
    cy = A.T @ y
    y3 = y * y * y
    cy3 = A.T @ y3
    if R is not None and R.size and state_prev is not None:
        if R.ndim == 1:
            R = R.reshape(1, -1)
        r = R @ v
        pooled_y2 += float(np.dot(r, r))
        pooled_y4 += float(np.sum((r * r) ** 2))
        cy = cy + R.T @ r
        cy3 = cy3 + R.T @ (r * r * r)
        rows_entropy += R.shape[0]
    pooled_y2 = max(pooled_y2, 1e-30)
    pooled_y4 = max(pooled_y4, 1e-30)
    rows_entropy = max(rows_entropy, 2)
    rows_ref_eff = max(int(rows_ref), rows_entropy)
    n_old = 0 if state_prev is None else int(state_prev.get("rows_seen", 0))
    rows_seen = min(max(n_old + rows_block, 1), rows_ref_eff)
    c = np.log(rows_seen / rows_ref_eff) / (2.0 * np.log(rows_entropy))
    log_phi = c * (np.log(pooled_y4) - 2.0 * np.log(pooled_y2))
    phi = float(np.exp(log_phi))
    score = float(gain2 * phi)
    grad_log_phi = 4.0 * c * (cy3 / pooled_y4 - cy / pooled_y2)
    grad = phi * grad_gain2 + score * grad_log_phi
    return score, grad, gain2, phi, c


def tangent_grad_norm(grad, v, v1=None):
    """Tangent gradient norm on unit sphere, optionally constrained orthogonal to v1."""
    g = grad - float(v @ grad) * v
    if v1 is not None:
        v1n = v1 / max(float(np.linalg.norm(v1)), 1e-30)
        g = g - float(v1n @ g) * v1n
    return float(np.linalg.norm(g)), g


def gradient_ascent_full(v0, v1, M_gain, A_cur, rows_ref, state_prev, old_row_memory,
                         maxit=200, tol=1e-10):
    """Riemannian gradient ascent on the unit sphere orthogonal to v1, FULL R^n basis."""
    v = project_to_v1_perp(np.asarray(v0, dtype=np.float64), v1)
    if v is None:
        return None, np.nan, np.nan
    score, grad, _, _, _ = combined_score_and_grad_full(M_gain, A_cur, v, rows_ref, state_prev, old_row_memory)
    gnorm, gtan = tangent_grad_norm(grad, v, v1)
    for it in range(maxit):
        if gnorm <= tol:
            break
        alpha = 1.0
        improved = False
        for _ in range(40):
            vt = v + alpha * gtan / max(gnorm, 1e-30)
            vt = project_to_v1_perp(vt, v1)
            if vt is None:
                alpha *= 0.5
                continue
            score_t, grad_t, _, _, _ = combined_score_and_grad_full(M_gain, A_cur, vt, rows_ref, state_prev, old_row_memory)
            if score_t > score + 1e-12:
                v, score, grad = vt, score_t, grad_t
                gnorm, gtan = tangent_grad_norm(grad, v, v1)
                improved = True
                break
            alpha *= 0.5
        if not improved:
            break
    return v, score, gnorm


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
            sketch_block = None
        else:
            B_top = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
            sketch_block = B_top
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
            # Find in-window optimum
            union = np.vstack([A_cur, A_fut]).astype(np.float64, copy=False)
            B_union = orth_basis_against(rowspace_basis(union), v1)
            starts = [V_default[:, 1]]
            if V_exact.shape[1] >= 2:
                starts.append(V_exact[:, 1])
            if diag.get("Vbasis_final") is not None:
                Vbasis = np.asarray(diag["Vbasis_final"], dtype=np.float64)
                for j in range(min(Vbasis.shape[1], 8)):
                    starts.append(Vbasis[:, j])
            best = optimize_future_hmean_in_basis(
                A_cur, A_fut, B_union, starts,
                np.random.default_rng(args.seed + 1009 * block_id),
                maxit=80, tol=1e-9, random_starts=12,
            )
            v_inwindow = None if best is None else best["vec"]

            # Project v_inwindow onto M_gain rowspace (the search space the optimizer can reach)
            Q_mgain = rowspace_basis(np.asarray(M_gain, dtype=np.float64))
            if v_inwindow is not None:
                v_inwindow_proj = Q_mgain @ (Q_mgain.T @ v_inwindow)
                nrm = float(np.linalg.norm(v_inwindow_proj))
                if nrm > 1e-30:
                    v_inwindow_proj = v_inwindow_proj / nrm
                else:
                    v_inwindow_proj = None
            else:
                v_inwindow_proj = None

            rec = {"matrix": args.matrix, "block": block_id, "rows_seen": mid0}
            rec["state_s1"] = float(state["s"][0]) if state is not None else np.nan
            rec["state_s2"] = float(state["s"][1]) if state is not None and len(state["s"]) >= 2 else np.nan
            rec["state_s2_squared"] = rec["state_s2"] ** 2

            # Score and gradient at v_carry (= optimizer v2)
            sc, gr, gain2_c, phi_c, c_c = combined_score_and_grad_full(
                M_gain, A_cur, v2, A.shape[0], state, old_row_memory)
            gn_c, _ = tangent_grad_norm(gr, v2, v1)
            rec["v_carry_score"] = sc
            rec["v_carry_gain2"] = gain2_c
            rec["v_carry_phi"] = phi_c
            rec["v_carry_grad_norm"] = gn_c

            if v_inwindow is not None:
                sc, gr, gain2_w, phi_w, c_w = combined_score_and_grad_full(
                    M_gain, A_cur, v_inwindow, A.shape[0], state, old_row_memory)
                gn_w, _ = tangent_grad_norm(gr, v_inwindow, v1)
                rec["v_inwindow_score"] = sc
                rec["v_inwindow_gain2"] = gain2_w
                rec["v_inwindow_phi"] = phi_w
                rec["v_inwindow_grad_norm"] = gn_w

                # arc walk: v_carry -> v_inwindow on great circle (orthogonal to v1)
                # parametrize v(t) = cos(t)*v_carry + sin(t) * (v_inwindow_perp normalized)
                v_carry_perp_v1 = project_to_v1_perp(v2, v1)
                v_in_perp_v1 = project_to_v1_perp(v_inwindow, v1)
                if v_carry_perp_v1 is not None and v_in_perp_v1 is not None:
                    e1 = v_carry_perp_v1
                    u = v_in_perp_v1 - float(e1 @ v_in_perp_v1) * e1
                    nu = float(np.linalg.norm(u))
                    if nu > 1e-30:
                        e2 = u / nu
                        ts = np.linspace(0.0, np.pi, 11)
                        scores = []
                        for t in ts:
                            vt = np.cos(t) * e1 + np.sin(t) * e2
                            sc_t, _, gain2_t, phi_t, _ = combined_score_and_grad_full(
                                M_gain, A_cur, vt, A.shape[0], state, old_row_memory)
                            scores.append((float(t), float(sc_t), float(gain2_t), float(phi_t)))
                        rec["arc"] = scores

            if v_inwindow_proj is not None:
                sc, gr, gain2_wp, phi_wp, c_wp = combined_score_and_grad_full(
                    M_gain, A_cur, v_inwindow_proj, A.shape[0], state, old_row_memory)
                gn_wp, _ = tangent_grad_norm(gr, v_inwindow_proj, v1)
                rec["v_inwindow_proj_score"] = sc
                rec["v_inwindow_proj_gain2"] = gain2_wp
                rec["v_inwindow_proj_phi"] = phi_wp
                rec["v_inwindow_proj_grad_norm"] = gn_wp

                # Run gradient ascent on full sphere from v_inwindow_proj
                v_end, sc_end, gn_end = gradient_ascent_full(
                    v_inwindow_proj, v1, M_gain, A_cur, A.shape[0], state, old_row_memory,
                    maxit=300, tol=1e-9)
                if v_end is not None:
                    rec["ascent_from_inwindow_proj_end_score"] = sc_end
                    rec["ascent_from_inwindow_proj_end_grad"] = gn_end
                    rec["ascent_from_inwindow_proj_align_carry"] = float(abs(np.dot(v_end, v2)) ** 2)

            # Run gradient ascent on full sphere from v_inwindow itself (allowing escape from S4)
            if v_inwindow is not None:
                v_end, sc_end, gn_end = gradient_ascent_full(
                    v_inwindow, v1, M_gain, A_cur, A.shape[0], state, old_row_memory,
                    maxit=300, tol=1e-9)
                if v_end is not None:
                    rec["ascent_from_inwindow_end_score"] = sc_end
                    rec["ascent_from_inwindow_end_grad"] = gn_end
                    rec["ascent_from_inwindow_align_carry"] = float(abs(np.dot(v_end, v2)) ** 2)
                    rec["ascent_from_inwindow_align_inwindow"] = float(abs(np.dot(v_end, v_inwindow)) ** 2)

            # Random restarts on full sphere
            n = A.shape[1]
            rng = np.random.default_rng(args.seed + 10007 * block_id)
            best_random = None
            best_random_align_carry = np.nan
            for _ in range(args.random_restarts):
                v0 = rng.standard_normal(n)
                v0 /= max(float(np.linalg.norm(v0)), 1e-30)
                v_end, sc_end, gn_end = gradient_ascent_full(
                    v0, v1, M_gain, A_cur, A.shape[0], state, old_row_memory,
                    maxit=300, tol=1e-9)
                if v_end is not None and (best_random is None or sc_end > best_random):
                    best_random = sc_end
                    best_random_align_carry = float(abs(np.dot(v_end, v2)) ** 2)
            rec["random_restart_best_score"] = best_random
            rec["random_restart_align_carry"] = best_random_align_carry

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
    p.add_argument("--out-prefix", default="summary/future_hmean_local_optimum_probe")
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
    p.add_argument("--random-restarts", type=int, default=8)
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
    flat = []
    arc_rows = []
    for r in rows:
        rcopy = {k: v for k, v in r.items() if k != "arc"}
        flat.append(rcopy)
        if "arc" in r:
            for t, sc, g2, ph in r["arc"]:
                arc_rows.append({"matrix": r["matrix"], "block": r["block"],
                                 "t": t, "score": sc, "gain2": g2, "phi": ph})
    csv_path = args.out_prefix + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    if arc_rows:
        with open(args.out_prefix + "_arc.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(arc_rows[0].keys()))
            w.writeheader()
            w.writerows(arc_rows)
    print(f"wrote {csv_path} {json_path} (and arc csv)")


if __name__ == "__main__":
    main()
