"""Sequential oracle-evidence sweep — V1 → V2 → V3 (→ V4 if needed).

Per the 2026-04-28 reframe: balance is not enough; the score's argmax is
not the oracle. This probe tests, in order:

  V1  hm_x_futdir     base HM3 × Rayleigh on A_fut direction
                      (fut energy is rotation-symmetric within fut row-
                      space; multiply by ||A_fut^T A_fut Z||_F² /
                      ||A_fut Z||_F² to reward alignment with the leading
                      A_fut-spectrum directions).
  V2  hm_x_crosscorr  base HM3 × <A_cur Z, A_fut Z>_F /
                      (||A_cur Z||_F ||A_fut Z||_F)
                      (rewards directions where current and future agree).
  V3  hm              base HM3 only — joint Stiefel(d, 2) optimization.
                      Tests whether the slot-by-slot greedy chain is the
                      identifiability bottleneck rather than the score
                      content.

For each variant: run BOTH greedy (slot-1 then deflate slot-2, mirroring
r_sk_g_score) AND joint Stiefel(d, 2) ascent. If joint < greedy at any
matrix, the joint optimizer is broken; if joint ≥ greedy and Z_winner
still > Z_oracle, the score is the bottleneck (not optimization).

Speed: analytic gradient on Z (no finite differences) + polar retraction
on Stiefel — one to two orders of magnitude faster than the FD version
in probe_frame_oracle_gap.py.

Outputs go to summary/score_family_aggregator_ablation/oracle_evidence_sweep/.
"""
from __future__ import annotations

import argparse
import json
import os
from math import sqrt

import numpy as np

import cex_restricted_space_probe as probe
import half_window_sliding_hmean_experiment  # noqa: F401
from future_hmean_optimizer_diagnostic import orth_basis_against, rowspace_basis
from hmean_evidence_score import per_block_constants, stream_to_block
from r_sk_g_score import _state_V

from probe_e2_landscape import make_default_args


# -------------------- score + analytic gradient --------------------

def _ensure_2d(Z):
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    return Z


def frame_score_and_grad(Z, *, A_sk, A_cur, A_fut, sk_F2, cur_F2, fut_F2,
                         ablation, oracle_Vr=None, oracle_lambda=0.0):
    """Frame score f(Z) and Euclidean gradient ∂f/∂Z (same shape as Z).

    Z may be d×1 (rank-1 / greedy) or d×2 (joint). Same formulas in both
    cases because we use Frobenius/trace forms throughout.
    """
    eps = 1e-30
    Z = _ensure_2d(Z)
    have_sk = A_sk is not None and sk_F2 > 0.0

    # Per-window responses Y_X = A_X Z and energies e_X = ||Y_X||_F².
    if have_sk:
        Y_sk = A_sk @ Z
        e_sk = float(np.sum(Y_sk * Y_sk))
        u_sk = e_sk / sk_F2
    else:
        Y_sk = None; e_sk = 0.0; u_sk = 0.0
    Y_c = A_cur @ Z; e_c = float(np.sum(Y_c * Y_c))
    Y_f = A_fut @ Z; e_f = float(np.sum(Y_f * Y_f))
    u_c = e_c / cur_F2 if cur_F2 > 0 else 0.0
    u_f = e_f / fut_F2 if fut_F2 > 0 else 0.0

    # Harmonic mean and its gradient.
    if have_sk:
        if u_sk <= eps or u_c <= eps or u_f <= eps:
            return 0.0, np.zeros_like(Z)
        inv_sum = 1.0 / u_sk + 1.0 / u_c + 1.0 / u_f
        base = 3.0 / inv_sum
        n_terms = 3.0
    else:
        if u_c <= eps or u_f <= eps:
            return 0.0, np.zeros_like(Z)
        inv_sum = 1.0 / u_c + 1.0 / u_f
        base = 2.0 / inv_sum
        n_terms = 2.0

    # ∂base/∂u_X = base² / (n_terms · u_X²).
    # ∂u_X/∂Z = 2 · A_X^T Y_X / ||A_X||_F².
    coef = (base * base) / n_terms
    grad_base = np.zeros_like(Z)
    if have_sk:
        grad_base += (coef / (u_sk * u_sk)) * (2.0 / sk_F2) * (A_sk.T @ Y_sk)
    grad_base += (coef / (u_c * u_c)) * (2.0 / cur_F2) * (A_cur.T @ Y_c)
    grad_base += (coef / (u_f * u_f)) * (2.0 / fut_F2) * (A_fut.T @ Y_f)

    # V4: optional oracle-subspace reward, additive: λ · ||V_r^T Z||_F² / 2.
    # Independent of `ablation` (composes with any base score). For Z with
    # p columns, max of ||V_r^T Z||_F² is min(r, p); we normalize by 2.
    def _add_oracle_term(score, grad):
        if oracle_Vr is None or oracle_lambda <= 0.0:
            return score, grad
        VtZ = oracle_Vr.T @ Z
        bonus = float(np.sum(VtZ * VtZ)) / 2.0
        # ∂(||V_r^T Z||_F²)/∂Z = 2 V_r V_r^T Z.
        d_bonus = oracle_Vr @ VtZ  # = V_r V_r^T Z (since V_r orthonormal)
        return score + oracle_lambda * bonus, grad + oracle_lambda * d_bonus

    if ablation == "hm":
        return _add_oracle_term(float(base), grad_base)

    if ablation == "hm_x_energy":
        E = e_sk + e_c + e_f
        grad_E = 2.0 * (A_cur.T @ Y_c + A_fut.T @ Y_f)
        if have_sk:
            grad_E = grad_E + 2.0 * (A_sk.T @ Y_sk)
        return _add_oracle_term(float(base * E), grad_base * E + base * grad_E)

    if ablation == "hm_x_crosscorr":
        nc = sqrt(max(e_c, 0.0)); nf = sqrt(max(e_f, 0.0))
        if nc <= eps or nf <= eps:
            return 0.0, np.zeros_like(Z)
        ip = float(np.sum(Y_c * Y_f))
        cc = ip / (nc * nf)
        if cc <= 0.0:
            return 0.0, np.zeros_like(Z)
        # ∂ip/∂Z, ∂nc/∂Z, ∂nf/∂Z
        d_ip = A_cur.T @ Y_f + A_fut.T @ Y_c
        d_nc = (A_cur.T @ Y_c) / nc
        d_nf = (A_fut.T @ Y_f) / nf
        denom = nc * nf
        d_cc = d_ip / denom - ip * (d_nc * nf + nc * d_nf) / (denom * denom)
        return _add_oracle_term(float(base * cc), grad_base * cc + base * d_cc)

    if ablation == "hm_x_futdir":
        # Rayleigh-style directional term on A_fut:
        #   r(Z) = ||A_fut^T A_fut Z||_F² / ||A_fut Z||_F²
        #        = trace(Z^T M² Z) / trace(Z^T M Z)        with M = A_fut^T A_fut.
        # Reaches ||A_fut||²_op when Z is in the leading eigenspace of M.
        if e_f <= eps:
            return 0.0, np.zeros_like(Z)
        AtY = A_fut.T @ Y_f                       # M Z, shape (d, k)
        AAY = A_fut @ AtY                         # A_fut M Z, used for d_num
        num = float(np.sum(AtY * AtY))            # ||M Z||_F² = trace(Z^T M² Z)
        r = num / e_f
        # ∂num/∂Z = 2 M² Z = 2 A_fut^T A_fut · A_fut^T A_fut Z
        d_num = 2.0 * (A_fut.T @ AAY)
        d_ef = 2.0 * AtY                          # ∂||A_fut Z||²/∂Z = 2 A_fut^T A_fut Z
        d_r = d_num / e_f - num * d_ef / (e_f * e_f)
        return _add_oracle_term(float(base * r), grad_base * r + base * d_r)

    raise ValueError(f"unknown ablation {ablation!r}")


def frame_score_only(Z, **kw):
    return frame_score_and_grad(Z, **kw)[0]


# -------------------- Riemannian ascent --------------------

def _project_to_basis(Z, B):
    """Project Z columns into span(B) and orthonormalize."""
    W = B.T @ Z
    Q, R = np.linalg.qr(W)
    keep = np.abs(np.diag(R)) > 1e-12
    Q = Q[:, keep]
    if Q.shape[1] == 0:
        return None
    return B @ Q


def stiefel_ascent(Z0, B, *, score_kw, n_iter=400, tol=1e-12):
    """Riemannian gradient ascent on Stiefel(k, p) where k = dim(B),
    p = #cols of Z0. Parameterizes Z = B · W; uses polar retraction.
    Returns (Z_best, score_best)."""
    Z0 = _ensure_2d(Z0)
    p_target = Z0.shape[1]
    Z = _project_to_basis(Z0, B)
    if Z is None or Z.shape[1] != p_target:
        return None, -np.inf
    score, grad_full = frame_score_and_grad(Z, **score_kw)
    eta = 0.25
    for _ in range(n_iter):
        # Pull gradient back to W coords.
        grad_W = B.T @ grad_full
        # Tangent projection on Stiefel(k, p): G_tan = G - W · sym(W^T G).
        W = B.T @ Z
        WtG = W.T @ grad_W
        sym = 0.5 * (WtG + WtG.T)
        G_tan = grad_W - W @ sym
        gnorm = float(np.linalg.norm(G_tan))
        if gnorm < tol:
            break
        improved = False
        eta_try = eta
        for _ls in range(20):
            W_unproj = W + eta_try * G_tan
            U, _, Vt = np.linalg.svd(W_unproj, full_matrices=False)
            W_new = U @ Vt
            Z_new = B @ W_new
            score_new, grad_new = frame_score_and_grad(Z_new, **score_kw)
            if score_new > score + 1e-14 * (1.0 + abs(score)):
                Z = Z_new; score = score_new; grad_full = grad_new
                eta = min(eta_try * 1.2, 5.0)
                improved = True
                break
            eta_try *= 0.5
            if eta_try < 1e-12:
                break
        if not improved:
            break
    return Z, float(score)


def best_of_starts(starts, B, *, score_kw, n_iter=400):
    best_Z = None; best_s = -np.inf
    for Z0 in starts:
        if Z0 is None:
            continue
        Z, s = stiefel_ascent(Z0, B, score_kw=score_kw, n_iter=n_iter)
        if Z is not None and s > best_s:
            best_Z = Z; best_s = s
    return best_Z, best_s


# -------------------- greedy and joint --------------------

def greedy_pair(B_union, *, score_kw, rng, n_starts=12):
    """Optimize v1 over B_union (rank-1), then v2 over B_def(v1). Returns
    Z = QR([v1, v2]) and (score_v1, score_v2, frame_score_at_Z)."""
    d = B_union.shape[0]
    k = B_union.shape[1]
    # Rank-1 starts
    starts1 = [B_union @ rng.standard_normal((k, 1)) for _ in range(n_starts)]
    v1, s1 = best_of_starts(starts1, B_union, score_kw=score_kw, n_iter=300)
    if v1 is None:
        return None, None
    B2 = orth_basis_against(B_union, v1[:, 0])
    if B2 is None or B2.size == 0:
        return None, None
    starts2 = [B2 @ rng.standard_normal((B2.shape[1], 1)) for _ in range(n_starts)]
    v2, s2 = best_of_starts(starts2, B2, score_kw=score_kw, n_iter=300)
    if v2 is None:
        return None, None
    Z, _ = np.linalg.qr(np.column_stack([v1[:, 0], v2[:, 0]]))
    s_frame = frame_score_only(Z, **score_kw)
    return Z, {"s_v1_rank1": float(s1), "s_v2_rank1": float(s2), "s_frame": float(s_frame)}


def joint_pair(B_union, *, score_kw, rng, warm_starts, n_starts=12):
    """Joint Stiefel(k, 2) ascent. Combines warm starts (oracle, sketch)
    with random restarts."""
    k = B_union.shape[1]
    starts = list(warm_starts)
    for _ in range(n_starts):
        starts.append(B_union @ rng.standard_normal((k, 2)))
    Z_best, s_best = best_of_starts(starts, B_union, score_kw=score_kw, n_iter=400)
    return Z_best, s_best


# -------------------- per-matrix driver --------------------

def _project_unit(v, B):
    if v is None or B is None or B.size == 0:
        return None
    p = B @ (B.T @ v)
    n = float(np.linalg.norm(p))
    return None if n <= 1e-30 else p / n


def run_for_matrix(matrix, *, block, ablation, n_starts=12,
                    oracle_lambda=0.0, oracle_r=2):
    args = make_default_args(matrix, block=block)
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=matrix, n=args.n, preset=args.preset, seed=args.seed,
        r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
        shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, np.float64); V_exact = np.asarray(V_exact, np.float64)
    snapshots = stream_to_block(args, A, V_exact, work_dtype, int(args.rank), block, {block})
    snap = snapshots[block]
    consts = per_block_constants(A, block, args.half_win)

    A_cur = np.asarray(snap["A_cur"], np.float64)
    A_fut = np.asarray(snap["A_fut"], np.float64)
    A_sk_arr = np.asarray(snap["A_sketch"], np.float64)
    A_sk = A_sk_arr if A_sk_arr.size else None
    state = snap["state"]; V_state = _state_V(state)
    cur_F2 = float(consts["cur_F2"]); fut_F2 = float(consts["fut_F2"])
    sk_F2 = float(np.sum(A_sk_arr ** 2)) if A_sk is not None else 0.0

    if A_sk is not None:
        union_stack = np.vstack([A_sk_arr, A_cur, A_fut])
    else:
        union_stack = np.vstack([A_cur, A_fut])
    B_union = rowspace_basis(union_stack)

    o1 = _project_unit(V_exact[:, 0], B_union)
    o2 = _project_unit(V_exact[:, 1], B_union)
    if o1 is None or o2 is None:
        return None
    Z_oracle, _ = np.linalg.qr(np.column_stack([o1, o2]))

    Vr = V_exact[:, :oracle_r] if (oracle_lambda > 0.0 and oracle_r > 0) else None
    score_kw = dict(A_sk=A_sk, A_cur=A_cur, A_fut=A_fut,
                    sk_F2=sk_F2, cur_F2=cur_F2, fut_F2=fut_F2,
                    ablation=ablation, oracle_Vr=Vr,
                    oracle_lambda=oracle_lambda)
    s_oracle = frame_score_only(Z_oracle, **score_kw)

    rng = np.random.default_rng(int(args.seed) + 0x91A2B3 + block)

    # --- greedy ---
    Z_greedy, gd = greedy_pair(B_union, score_kw=score_kw, rng=rng, n_starts=n_starts)
    s_greedy_frame = gd["s_frame"] if gd is not None else float("nan")

    # --- joint, warm-started with oracle, sketch state, greedy ---
    warm = [Z_oracle]
    if V_state is not None and V_state.shape[1] >= 2:
        warm.append(V_state[:, :2])
    if Z_greedy is not None:
        warm.append(Z_greedy)
    Z_joint, s_joint = joint_pair(B_union, score_kw=score_kw, rng=rng,
                                   warm_starts=warm, n_starts=n_starts)

    # Principal cosines.
    def pa2(Za, Zb):
        if Za is None or Zb is None:
            return None
        sv = np.linalg.svd(Za.T @ Zb, compute_uv=False)
        return (sv ** 2).tolist()

    return {
        "matrix": matrix, "block": block, "ablation": ablation,
        "oracle_lambda": float(oracle_lambda), "oracle_r": int(oracle_r),
        "score_oracle": float(s_oracle),
        "score_greedy": float(s_greedy_frame),
        "score_joint": float(s_joint) if Z_joint is not None else float("nan"),
        "delta_greedy": float(s_greedy_frame - s_oracle),
        "delta_joint": float(s_joint - s_oracle) if Z_joint is not None else float("nan"),
        "joint_minus_greedy": float((s_joint if Z_joint is not None else float("nan")) - s_greedy_frame),
        "pa2_greedy_vs_oracle": pa2(Z_greedy, Z_oracle),
        "pa2_joint_vs_oracle": pa2(Z_joint, Z_oracle),
    }


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", nargs="+", default=[
        "diffuse-diffuse", "residual-spiky-shocks", "mixed-tail-soft",
    ])
    ap.add_argument("--block", type=int, default=31)
    ap.add_argument("--ablations", nargs="+",
                    default=["hm", "hm_x_futdir", "hm_x_crosscorr"])
    ap.add_argument("--n-starts", type=int, default=12)
    ap.add_argument("--oracle-lambdas", nargs="+", type=float, default=[0.0],
                    help="V4 oracle-subspace reward weights (sweep).")
    ap.add_argument("--oracle-r", type=int, default=2,
                    help="V4 rank of V_exact prefix used in the oracle term.")
    ap.add_argument("--out-dir",
                    default="summary/score_family_aggregator_ablation/oracle_evidence_sweep")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for ab in args.ablations:
        for lam in args.oracle_lambdas:
            tag = f"{ab}" + (f" λ={lam:g}" if lam > 0 else "")
            print(f"=== {tag} ===", flush=True)
            for m in args.matrices:
                r = run_for_matrix(m, block=args.block, ablation=ab,
                                   n_starts=args.n_starts,
                                   oracle_lambda=lam,
                                   oracle_r=args.oracle_r)
                if r is None:
                    print(f"  {m}: SKIP (oracle projection empty)")
                    continue
                rows.append(r)
                print(f"  {m:<24} oracle={r['score_oracle']:.4f}  "
                      f"greedy={r['score_greedy']:.4f}  joint={r['score_joint']:.4f}  "
                      f"Δgreedy={r['delta_greedy']:+.4f}  Δjoint={r['delta_joint']:+.4f}  "
                      f"j−g={r['joint_minus_greedy']:+.4f}",
                      flush=True)

    json_path = os.path.join(args.out_dir, f"sweep_b{args.block}.json")
    with open(json_path, "w") as fh:
        json.dump(rows, fh, indent=2, default=float)

    txt_path = os.path.join(args.out_dir, f"sweep_b{args.block}.txt")
    with open(txt_path, "w") as fh:
        fh.write("# Oracle-evidence sweep — V1 → V2 → V3\n")
        fh.write(f"# block={args.block}\n")
        fh.write("# Δ > 0 (winner > oracle) on a regression matrix → score is bottleneck\n")
        fh.write("# joint < greedy → joint optimizer is broken; joint ≥ greedy expected\n\n")
        fh.write(f"{'ablation':<16} {'λ':>8} {'matrix':<24} "
                 f"{'oracle':>10} {'greedy':>10} {'joint':>10} "
                 f"{'Δgreedy':>10} {'Δjoint':>10} {'j−g':>10}\n")
        for r in rows:
            fh.write(f"{r['ablation']:<16} {r['oracle_lambda']:>8.4g} {r['matrix']:<24} "
                     f"{r['score_oracle']:>10.4f} {r['score_greedy']:>10.4f} "
                     f"{r['score_joint']:>10.4f} "
                     f"{r['delta_greedy']:>+10.4f} {r['delta_joint']:>+10.4f} "
                     f"{r['joint_minus_greedy']:>+10.4f}\n")
    print(f"\nwrote {json_path}\nwrote {txt_path}")


if __name__ == "__main__":
    main()
