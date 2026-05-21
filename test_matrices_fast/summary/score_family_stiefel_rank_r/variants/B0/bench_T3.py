"""FAM-01 B0 — rank-r joint Stiefel HM3 streaming bench (T3).

Backlog: summary/overview/score_family_workflow.txt §5 [FAM-01] (B0 = HM3
rank-r joint frame). Five-matrix partial-pass set:
    - residual-spiky-shocks   (PASS-with-overshoot)
    - etf-basket-basis        (FAIL-slot-2-only)
    - mixed-tail-sharp        (FAIL-slot-2-only)
    - mixed-tail-balanced     (FAIL-slot-2-only)
    - mixed-tail-soft         (FAIL-slot-2-only)
diffuse-diffuse and static-cex are SKIPPED (FAIL-subspace; see
score_family_workflow.txt §5 [FAM-01] notes — they redirect to evidence
augmentation, not optimizer changes).

Acceptance (FAM-01 T3):
    - cos0² (slot-1) lifts over greedy S6 by ≥0.05 on at least 2
      high-entropy matrices, AND
    - no §6 matrix regresses cos0² by >0.02.
    cos1² is informational only (slot-2 evidence did not pass S-2).

Baselines:
    - greedy S6 (policy=future_hmean_r_sk_g, --rsk-variant S6)
    - value-only future_hmean_online (INFRA-10 default pool)

Compares against B0 (this file): a streaming policy that, at every
block, jointly optimizes Z ∈ Stiefel(d, 2) restricted to B_union =
rowspace([A_sketch; A_cur; A_fut]) using frame_S6_value_grad's analytic
Stiefel ascent. Both columns of Z are free (joint frame); slot-1 is NOT
anchored to V_default[:,0]. The frame is then re-ordered by the rank-2
SVD of M_gain @ Z so the carry stays consistent with the existing
streaming-state machinery.

Window: half_win = 32 (matches the §6 baseline harness; bench JSONs are
labeled `_win64` by the existing convention which encodes the FULL
window 2*half_win).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Optional

import numpy as np

# Make the test_matrices_fast root importable.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cex_restricted_space_probe as probe
import half_window_sliding_hmean_experiment as harness
from future_hmean_optimizer_diagnostic import orth_basis_against, rowspace_basis
from second_slot_tail_bias_diagnostic import make_state, raw_oracle_columns
from stiefel_grad_check import (
    frame_S6_value_grad,
    polar_retract,
    stiefel_tangent_project,
)


MATRICES_T3 = [
    "residual-spiky-shocks",
    "etf-basket-basis",
    "mixed-tail-sharp",
    "mixed-tail-balanced",
    "mixed-tail-soft",
]

HALF_WIN = 32
N_ROWS = 1024
RANK = 2


# ---------------------------------------------------------------------------
# B0 joint Stiefel ascent on B_union
# ---------------------------------------------------------------------------


def _orth_proj_into_basis(Z, B):
    """Project columns of Z into span(B), then QR-orthonormalize. Returns Q
    with up to 2 columns (drops linearly-dependent columns)."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    # Project then orthonormalize.
    Zp = B @ (B.T @ Z)
    Q, R = np.linalg.qr(Zp)
    diag = np.abs(np.diag(R))
    keep = diag > 1e-10 * max(diag.max(initial=0.0), 1e-30)
    Q = Q[:, keep]
    return Q


def _expand_to_two_cols(Z, B, rng):
    """Ensure Z has 2 orthonormal columns inside span(B). If only 1 column,
    append a random direction in B \\ span(Z)."""
    Z = _orth_proj_into_basis(Z, B)
    while Z.shape[1] < 2:
        v = B @ rng.standard_normal(B.shape[1])
        if Z.shape[1]:
            v = v - Z @ (Z.T @ v)
        n = float(np.linalg.norm(v))
        if n <= 1e-12:
            continue
        v = v / n
        Z = np.column_stack([Z, v]) if Z.shape[1] else v[:, None]
    return Z[:, :2]


def _stiefel_frame_ascent(Z_init, *, A_sk, A_cur, A_fut, sk_F2, cur_F2,
                          fut_F2, B_union, max_iter=200, step0=1.0,
                          tol=1e-11, rng=None):
    """Joint Stiefel(d,2) ascent of frame_S6 restricted to B_union.

    Mirrors probe_frame_oracle_gap._stiefel_frame_ascent, kept local to
    avoid an import cycle with that module.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    Z = _expand_to_two_cols(Z_init, B_union, rng)

    def value_grad(V):
        return frame_S6_value_grad(A_sk, A_cur, A_fut, sk_F2, cur_F2, fut_F2, V)

    score, _ = value_grad(Z)
    score = float(score)
    for _ in range(max_iter):
        _, G = value_grad(Z)
        # Constrain the search to B_union before projecting onto T_Z St(d,2).
        G = B_union @ (B_union.T @ G)
        Gt = stiefel_tangent_project(Z, G)
        gnorm = float(np.linalg.norm(Gt))
        if gnorm <= tol:
            break
        accepted = False
        for step in (step0, step0 / 2.0, step0 / 4.0, step0 / 8.0,
                     step0 / 16.0, step0 / 64.0):
            Z_try = polar_retract(Z, step * Gt)
            score_try, _ = value_grad(Z_try)
            score_try = float(score_try)
            if score_try > score + tol:
                Z = Z_try
                score = score_try
                accepted = True
                break
        if not accepted:
            break
    return Z, score


def _b0_select_frame(*, A_sketch, A_h1, A_h2, sk_F2_low, cur_F2, fut_F2,
                     B_union, M_gain, V_default, V_state, n_rand_starts,
                     rng):
    """Run multi-start joint Stiefel ascent and return the best Z."""
    starts = []
    # 1) The default streaming frame (V_default[:, :2]).
    if V_default is not None and V_default.shape[1] >= 2:
        starts.append(V_default[:, :2])
    # 2) M_gain top-2 right singular vectors (rank-2 SVD warm-start).
    Mg = np.asarray(M_gain, dtype=np.float64)
    if Mg.size:
        try:
            _, _, Vt_mg = np.linalg.svd(Mg, full_matrices=False)
            if Vt_mg.shape[0] >= 2:
                starts.append(Vt_mg[:2].T)
        except np.linalg.LinAlgError:
            pass
    # 3) Carried state's top-2 V columns.
    if V_state is not None:
        Vs = np.asarray(V_state, dtype=np.float64)
        if Vs.shape[1] >= 2:
            starts.append(Vs[:, :2])
        elif Vs.shape[1] == 1 and V_default is not None and V_default.shape[1] >= 2:
            starts.append(np.column_stack([Vs[:, 0], V_default[:, 1]]))
    # 4) Random Stiefel(d,2) starts inside B_union.
    for _ in range(int(n_rand_starts)):
        G = rng.standard_normal((B_union.shape[1], 2))
        Z = B_union @ G
        Q, _ = np.linalg.qr(Z)
        starts.append(Q[:, :2])

    best = None
    best_score = -np.inf
    for Z0 in starts:
        try:
            Z, s = _stiefel_frame_ascent(
                Z0, A_sk=A_sketch, A_cur=A_h1, A_fut=A_h2,
                sk_F2=sk_F2_low, cur_F2=cur_F2, fut_F2=fut_F2,
                B_union=B_union, rng=rng,
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[B0] ascent failed: {exc}")
            continue
        if s > best_score:
            best_score = float(s)
            best = Z
    return best, float(best_score)


# ---------------------------------------------------------------------------
# B0 streaming loop. Mirrors run_pair_stream's bookkeeping but the policy
# branch jointly optimizes Z and skips the slot-1-anchored chosen_v2 path.
# ---------------------------------------------------------------------------


def run_b0_stream(A, V_exact, sigma1, args, half_win, sliding,
                  n_rand_starts=4):
    work_dtype = np.float64
    n = A.shape[0]
    rank = int(args.rank)
    state = None
    old_row_memory = None
    V_r = None
    rows = []
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    step = half_win if sliding else 2 * half_win
    pair_count = 0
    for start0 in range(0, n - half_win, step):
        mid0 = start0 + half_win
        end0 = min(mid0 + half_win, n)
        if end0 - mid0 < half_win:
            break
        pair_count += 1
        block_id = pair_count
        A_h1 = np.asarray(A[start0:mid0, :], dtype=work_dtype)
        A_h2 = np.asarray(A[mid0:end0, :], dtype=work_dtype)

        if state is None:
            M_sketch = None
            M_gain = A_h1
            rows_seen = A_h1.shape[0]
        else:
            B_top = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
            M_sketch = B_top
            M_gain = np.vstack([B_top, A_h1]).astype(work_dtype, copy=False)
            rows_seen = state["rows_seen"] + A_h1.shape[0]

        # Default greedy V_score from the same combined-score iterator the
        # harness uses; this is the warm-start basin for B0.
        V_init = probe.row_norm_seed(A_h1, rank)
        V_score, _, _, _, _ = probe.entropy_iter_basis_forget(
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
            A_block=A_h1,
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

        # B_union and F-norms (rank-r CARRY for sketch — match S6 wiring).
        cur_F2 = float(np.sum(A_h1 * A_h1))
        fut_F2 = float(np.sum(A_h2 * A_h2))
        if M_sketch is not None and np.asarray(M_sketch).size:
            A_sketch = np.asarray(M_sketch, dtype=np.float64)
            sk_F2_low = float(np.sum(A_sketch * A_sketch))
        else:
            A_sketch = None
            sk_F2_low = 0.0

        if A_sketch is not None:
            union_for_search = np.vstack([A_sketch, A_h1, A_h2])
        else:
            union_for_search = np.vstack([A_h1, A_h2])
        B_union = rowspace_basis(union_for_search)

        V_state = None
        if state is not None and state.get("V") is not None:
            Vs_arr = np.asarray(state["V"], dtype=np.float64)
            if Vs_arr.size:
                V_state = Vs_arr

        Z_best, _ = _b0_select_frame(
            A_sketch=A_sketch, A_h1=A_h1, A_h2=A_h2,
            sk_F2_low=sk_F2_low, cur_F2=cur_F2, fut_F2=fut_F2,
            B_union=B_union, M_gain=M_gain,
            V_default=V_default, V_state=V_state,
            n_rand_starts=n_rand_starts, rng=rng,
        )

        # Re-order columns of Z by descending singular value of M_gain @ Z so
        # the streaming carry/order convention matches `rank2_svd_frame`.
        if Z_best is None:
            V_selected = V_default[:, :rank].copy()
        else:
            Mz = np.asarray(M_gain, dtype=np.float64) @ Z_best
            _, _, Vt = np.linalg.svd(Mz, full_matrices=False)
            V_selected = np.ascontiguousarray(Z_best @ Vt.T[:, :rank])
        V_selected = probe.orthonormalize_columns(
            V_selected[:, :rank], dtype=np.float64
        )[:, :rank]

        # Diagnostics.
        Q_oracle, _ = raw_oracle_columns(M_gain, V_exact, rank, np.float64)
        cos = probe.subspace_principal_cosines(V_selected, Q_oracle)
        exact_cos = probe.subspace_principal_cosines(V_selected, V_exact[:, :rank])
        oracle_proj_norm = harness.oracle_projection_norms(M_gain, V_exact, rank, np.float64)
        tail_mass = harness.frame_tail_mass(V_selected, V_exact, rank)

        score_selected = np.zeros(rank, dtype=float)
        H_selected = np.zeros(rank, dtype=float)
        for j in range(rank):
            score_selected[j], _, H_selected[j] = probe.score_full_vector_details_forget(
                M_gain, A_h1, V_selected[:, j], n,
                state_prev=state, score_variant="combined",
                old_row_memory=old_row_memory,
            )

        _, s_new, Vt_new, _ = probe.left_projected_operator_svd_factors(V_selected.T, M_gain)
        V_carried = np.ascontiguousarray(np.asarray(Vt_new.T[:, :rank], dtype=np.float64))
        car_exact_cos = probe.subspace_principal_cosines(V_carried, V_exact[:, :rank])

        rows.append({
            "pair": pair_count, "block": block_id,
            "rows_seen": end0 if not sliding else mid0,
            "policy": "future_hmean_b0_rank_r",
            "mode": "sliding" if sliding else "split_probe",
            "half_win": half_win,
            "cos1": float(cos[0]) if len(cos) > 0 else np.nan,
            "cos2": float(cos[1]) if len(cos) > 1 else np.nan,
            "exact_cos1": float(exact_cos[0]) if len(exact_cos) > 0 else np.nan,
            "exact_cos2": float(exact_cos[1]) if len(exact_cos) > 1 else np.nan,
            "car_exact_cos1": float(car_exact_cos[0]) if len(car_exact_cos) > 0 else np.nan,
            "car_exact_cos2": float(car_exact_cos[1]) if len(car_exact_cos) > 1 else np.nan,
            "oracle_proj_norm1": float(oracle_proj_norm[0]) if len(oracle_proj_norm) > 0 else np.nan,
            "oracle_proj_norm2": float(oracle_proj_norm[1]) if len(oracle_proj_norm) > 1 else np.nan,
            "tail_mass": float(tail_mass),
            "relerr_sval": harness.rel_err_sval(s_new[:rank], sigma1),
        })

        state, V_r, _ = make_state(M_gain, V_selected, H_selected,
                                   score_selected, rows_seen)
        seen_for_memory = A[:mid0, :] if sliding else A[:end0, :]
        old_row_memory, _ = probe.select_old_row_memory(
            np.asarray(seen_for_memory, dtype=work_dtype),
            V_r.astype(work_dtype, copy=False),
            args.old_memory_size if args.old_memory_size > 0 else half_win,
            np.random.default_rng(args.seed + end0),
            return_indices=True,
        )

    final = rows[-1] if rows else None
    return {
        "policy": "future_hmean_b0_rank_r",
        "mode": "sliding" if sliding else "split_probe",
        "half_win": half_win,
        "rows": rows,
        "final_exact_cos": [final["exact_cos1"], final["exact_cos2"]] if final else [np.nan, np.nan],
        "final_cos": [final["cos1"], final["cos2"]] if final else [np.nan, np.nan],
        "final_tail_mass": final["tail_mass"] if final else np.nan,
        "elapsed": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Baselines via the full harness — greedy S6, value-only online.
# ---------------------------------------------------------------------------


def _make_args(matrix, half_win, seed=0):
    """Build a Namespace matching half_window_sliding_hmean_experiment's
    defaults but pinned to the matrix/window we want."""
    ns = argparse.Namespace(
        matrix=matrix,
        half_win=half_win,
        policies=["combined"],
        rsk_variant="S6",
        rsk_alpha=1.0, rsk_beta=2.0, rsk_gamma=1.0,
        rsk_no_deflate=False,
        force_oracle_v2=False, force_oracle_frame=False,
        n=N_ROWS, rank=RANK, preset="fast", seed=seed,
        shuffle_rows=True, row_shuffle_seed=0,
        old_memory_size=32, dtype="float32",
        q0=8, qmax=48, krylov_depth=2,
        residual_tol=0.01, expansion_maxit=8, num_restarts=3,
        maxit=120, tol=1e-8, post_expansion_maxit=80,
        patience=5, patience_rel_tol=1e-5, max_pairs=None,
        online_include_oracle=False,
        r_sig=2, alpha_sig=0.003, alpha_tail=0.0145,
        tail_scale=0.99, sigma1=0.991, v_type="rand",
        json_out="", csv_out="", text_out="",
    )
    return ns


def _final_exact_cos2(result):
    final = result["rows"][-1] if result.get("rows") else None
    if not final:
        return [np.nan, np.nan]
    return [
        float(final.get("exact_cos1", np.nan)) ** 2 if not np.isnan(final.get("exact_cos1", np.nan)) else np.nan,
        float(final.get("exact_cos2", np.nan)) ** 2 if not np.isnan(final.get("exact_cos2", np.nan)) else np.nan,
    ]


def run_baselines(matrix, half_win, seed):
    ns = _make_args(matrix, half_win, seed=seed)
    A, V_exact, _, sigma1 = probe.generate_matrix_input(
        matrix=matrix, n=ns.n, preset=ns.preset, seed=ns.seed,
        r_sig=ns.r_sig, alpha_sig=ns.alpha_sig,
        alpha_tail=ns.alpha_tail, tail_scale=ns.tail_scale,
        sigma1=ns.sigma1, v_type=ns.v_type,
        shuffle_rows=ns.shuffle_rows,
        row_shuffle_seed=ns.row_shuffle_seed,
    )
    A = np.asarray(A, dtype=np.float64)
    V_exact = np.asarray(V_exact, dtype=np.float64)

    out = {}

    # Greedy S6 — sliding mode (matches §6 baseline).
    ns.rsk_variant = "S6"
    s6 = harness.run_pair_stream(A, V_exact, sigma1, ns,
                                 "future_hmean_r_sk_g", half_win,
                                 sliding=True)
    out["greedy_S6"] = s6

    # Value-only online — sliding mode.
    online = harness.run_pair_stream(A, V_exact, sigma1, ns,
                                     "future_hmean_online", half_win,
                                     sliding=True)
    out["value_only_online"] = online

    # B0 — sliding mode.
    b0 = run_b0_stream(A, V_exact, sigma1, ns, half_win, sliding=True,
                       n_rand_starts=4)
    out["B0"] = b0

    return out, A, V_exact


def _final_metrics(result):
    rows = result.get("rows", [])
    if not rows:
        return {"cos0_sq": np.nan, "cos1_sq": np.nan,
                "tail_mass": np.nan, "elapsed": np.nan, "steps": 0}
    final = rows[-1]
    return {
        "cos0_sq": float(final["exact_cos1"]) ** 2 if np.isfinite(final["exact_cos1"]) else np.nan,
        "cos1_sq": float(final["exact_cos2"]) ** 2 if np.isfinite(final["exact_cos2"]) else np.nan,
        "tail_mass": float(final.get("tail_mass", np.nan)),
        "elapsed": float(result.get("elapsed", np.nan)),
        "steps": len(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",
                    default=os.path.join(os.path.dirname(__file__), "bench"))
    ap.add_argument("--half-win", type=int, default=HALF_WIN)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--matrices", nargs="+", default=MATRICES_T3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    full_window = 2 * args.half_win
    summary = {"half_win": args.half_win,
               "matrices": [], "table": {}}

    for matrix in args.matrices:
        t0 = time.time()
        print(f"[B0/T3] running {matrix} (half_win={args.half_win})")
        results, A, V_exact = run_baselines(matrix, args.half_win, args.seed)
        per_matrix = {}
        for name, res in results.items():
            per_matrix[name] = _final_metrics(res)
        per_matrix["delta_cos0_sq_B0_minus_S6"] = (
            per_matrix["B0"]["cos0_sq"] - per_matrix["greedy_S6"]["cos0_sq"]
        )
        per_matrix["delta_cos1_sq_B0_minus_S6"] = (
            per_matrix["B0"]["cos1_sq"] - per_matrix["greedy_S6"]["cos1_sq"]
        )
        summary["table"][matrix] = per_matrix
        summary["matrices"].append(matrix)

        # Write per-matrix JSON with all rows.
        out_path_json = os.path.join(args.out_dir, f"{matrix}_win{full_window}.json")
        out_payload = {
            "matrix": matrix,
            "half_win": args.half_win,
            "seed": args.seed,
            "results": {k: v for k, v in results.items()},
            "metrics": per_matrix,
        }
        with open(out_path_json, "w", encoding="utf-8") as fh:
            json.dump(out_payload, fh, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else None)

        # Per-matrix CSV: row-by-row B0 trajectory.
        out_path_csv = os.path.join(args.out_dir, f"{matrix}_win{full_window}.csv")
        all_rows = []
        for name, res in results.items():
            for r in res.get("rows", []):
                row = {"variant": name, **r}
                all_rows.append(row)
        if all_rows:
            keys = sorted({k for r in all_rows for k in r.keys()})
            with open(out_path_csv, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                for r in all_rows:
                    w.writerow(r)

        elapsed = time.time() - t0
        print(f"[B0/T3] {matrix}: greedy_S6 cos0²={per_matrix['greedy_S6']['cos0_sq']:.4f}"
              f" cos1²={per_matrix['greedy_S6']['cos1_sq']:.4f}"
              f" | online_VO cos0²={per_matrix['value_only_online']['cos0_sq']:.4f}"
              f" cos1²={per_matrix['value_only_online']['cos1_sq']:.4f}"
              f" | B0 cos0²={per_matrix['B0']['cos0_sq']:.4f}"
              f" cos1²={per_matrix['B0']['cos1_sq']:.4f}"
              f" | Δcos0²={per_matrix['delta_cos0_sq_B0_minus_S6']:+.4f}"
              f" elapsed={elapsed:.1f}s")

    # Summary JSON.
    sum_path = os.path.join(args.out_dir, f"T3_summary_win{full_window}.json")
    with open(sum_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else None)

    # Plain-text report.
    txt_path = os.path.join(args.out_dir, f"T3_summary_win{full_window}.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("FAM-01 B0 — T3 streaming bench summary\n")
        fh.write(f"half_win={args.half_win} (full window={full_window}), seed={args.seed}\n\n")
        hdr = (f"{'matrix':<26} | {'S6 cos0²':>9} {'S6 cos1²':>9}"
               f" | {'VO cos0²':>9} {'VO cos1²':>9}"
               f" | {'B0 cos0²':>9} {'B0 cos1²':>9}"
               f" | {'Δcos0² B0-S6':>13} {'Δcos1² B0-S6':>13}\n")
        fh.write(hdr)
        fh.write("-" * len(hdr) + "\n")
        for matrix in args.matrices:
            t = summary["table"][matrix]
            fh.write(
                f"{matrix:<26} | {t['greedy_S6']['cos0_sq']:9.4f} {t['greedy_S6']['cos1_sq']:9.4f}"
                f" | {t['value_only_online']['cos0_sq']:9.4f} {t['value_only_online']['cos1_sq']:9.4f}"
                f" | {t['B0']['cos0_sq']:9.4f} {t['B0']['cos1_sq']:9.4f}"
                f" | {t['delta_cos0_sq_B0_minus_S6']:+13.4f} {t['delta_cos1_sq_B0_minus_S6']:+13.4f}\n"
            )

    print(f"wrote: {sum_path}")
    print(f"wrote: {txt_path}")


if __name__ == "__main__":
    main()
