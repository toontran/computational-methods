"""FAM-07 T1 — Stiefel finite-difference gradient check for rho_F (rank-1)
and rho_frame (rank-r) cross-window consistency terms.

Spec: summary/score_family_cross_window_consistency/00_spec.md

This is a self-contained prototype. It does NOT touch r_sk_g_score.py or
hmean_evidence_score.py. It uses the INFRA-02 harness
(stiefel_grad_check.stiefel_fd_check) for the rank-r check and a
plain symmetric-FD column probe for rank-1 (sphere = St(n, 1) special
case).

Acceptance bar (T1, per [FAM-07] Acceptance, score_family_workflow.txt
line ~1712): rel < 1e-7 against finite differences at float64, on every
synthetic cell. Trace-form sanity row stays < 1e-9.

Run from the repo's test_matrices_fast/ working directory:

    cd test_matrices_fast/
    python summary/score_family_cross_window_consistency/01_t1_grad_check.py

Outputs go to stdout AND `02_t1_grad_check_log.txt` next to this file.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# Make the test_matrices_fast/ module path importable when this script is
# launched from the repo root or from inside the spec directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TMF_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _TMF_DIR not in sys.path:
    sys.path.insert(0, _TMF_DIR)

from stiefel_grad_check import (  # noqa: E402
    stiefel_fd_check,
    stiefel_tangent_project,
    trace_form_value_grad,
)


# --------------------------------------------------------------------------
# Closed-form value-and-gradient implementations (spec §3.3 and §4.3)
# --------------------------------------------------------------------------

_EPS_NORM = 1e-30


def rho_F_squared_value_grad(M_c: np.ndarray, M_f: np.ndarray, v: np.ndarray):
    """Rank-1 squared cross-window consistency:
        rho_F(v) = <M_c v, M_f v> / (||M_c v|| ||M_f v||)
        score   = rho_F(v)**2
    Returns (score, grad) with grad in R^n (Euclidean).

    Closed form (spec §3.3): with p = M_c v, q = M_f v, np_ = ||p||,
    nq_ = ||q||, s = <p, q>:
        rho_F     = s / (np_ * nq_)
        d rho_F   = (M_c q + M_f p) / (np_ nq_)
                  - rho_F * (M_c p / np_^2 + M_f q / nq_^2)
        d (rho^2) = 2 rho_F * d rho_F
    """
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    M_c = np.asarray(M_c, dtype=np.float64)
    M_f = np.asarray(M_f, dtype=np.float64)
    p = M_c @ v
    q = M_f @ v
    np_ = float(np.linalg.norm(p))
    nq_ = float(np.linalg.norm(q))
    if np_ <= _EPS_NORM or nq_ <= _EPS_NORM:
        return 0.0, np.zeros_like(v)
    s = float(np.dot(p, q))
    rho = s / (np_ * nq_)
    # d rho / dv
    grad_rho = (M_c @ q + M_f @ p) / (np_ * nq_) \
               - rho * (M_c @ p / (np_ * np_) + M_f @ q / (nq_ * nq_))
    score = rho * rho
    grad = 2.0 * rho * grad_rho
    return float(score), np.ascontiguousarray(grad, dtype=np.float64)


def rho_frame_value_grad(M_c: np.ndarray, M_f: np.ndarray, V: np.ndarray):
    """Rank-r frame cross-window consistency (spec §4.2 / §4.3):
        K   = V^T M_c M_f V                       (r x r)
        N   = ||K||_F^2 = trace(K^T K)
        D_c = trace(V^T M_c^2 V) = ||M_c V||_F^2
        D_f = trace(V^T M_f^2 V) = ||M_f V||_F^2
        rho_frame(V) = N / (D_c * D_f)

    Closed-form Euclidean gradient (returned UNPROJECTED; the FD
    harness applies the Stiefel tangent projection itself):
        grad N   = 2 ( (M_f M_c V) K^T + (M_c M_f V) K )
        grad D_c = 2 M_c^2 V
        grad D_f = 2 M_f^2 V
        grad rho = (1/(D_c D_f)) grad N
                 - (N/(D_c^2 D_f)) grad D_c
                 - (N/(D_c D_f^2)) grad D_f.
    """
    V = np.asarray(V, dtype=np.float64)
    M_c = np.asarray(M_c, dtype=np.float64)
    M_f = np.asarray(M_f, dtype=np.float64)

    G_c = M_c @ V                          # n x r
    G_f = M_f @ V                          # n x r
    K = V.T @ G_f                          # r x r ; equals V^T M_c M_f V
    # Note V^T G_f = V^T M_f V is symmetric. We need V^T M_c M_f V, which is
    # V^T M_c (M_f V) = V^T M_c G_f. Use V^T (M_c G_f).
    K = V.T @ (M_c @ G_f)                  # r x r ; V^T M_c M_f V

    D_c = float(np.sum(G_c * G_c))
    D_f = float(np.sum(G_f * G_f))
    N = float(np.sum(K * K))

    if D_c <= _EPS_NORM or D_f <= _EPS_NORM:
        return 0.0, np.zeros_like(V)

    rho = N / (D_c * D_f)

    # grad N = 2 ( S V K^T + S^T V K ) with S = M_c M_f
    # Derivation: N = trace(K^T K), K = V^T S V. dK = dV^T S V + V^T S dV.
    # d N = 2 trace(K^T dK) = 2 [<S V K^T, dV> + <S^T V K, dV>].
    SV = M_c @ G_f                         # = S V = M_c (M_f V) ; n x r
    StV = M_f @ G_c                        # = S^T V = M_f (M_c V) ; n x r
    grad_N = 2.0 * (SV @ K.T + StV @ K)

    grad_Dc = 2.0 * (M_c @ G_c)            # = 2 M_c^2 V
    grad_Df = 2.0 * (M_f @ G_f)            # = 2 M_f^2 V

    grad = (
        grad_N / (D_c * D_f)
        - (N / (D_c * D_c * D_f)) * grad_Dc
        - (N / (D_c * D_f * D_f)) * grad_Df
    )

    return float(rho), np.ascontiguousarray(grad, dtype=np.float64)


# --------------------------------------------------------------------------
# Rank-1 sphere FD harness (St(n, 1) special case; we use a small helper
# rather than reusing stiefel_fd_check, which expects a (n, r) frame —
# although it works for r=1 too, the rank-1 spec is `score = rho_F^2(v)`
# with v on the unit sphere, and we want a scalar v probe to mirror the
# `gradient_check` pattern in r_sk_g_score.py / hmean_evidence_score.py).
# --------------------------------------------------------------------------


def sphere_fd_check(score_value_grad_fn, v, n_samples=20, eps=1e-6, rng=None):
    """Compare analytic gradient vs central-difference column FD on the
    sphere.

    Tangent space at v: T_v S^{n-1} = {z : v^T z = 0}. For each FD step
    we project the random Euclidean perturbation onto the tangent and
    retract via normalization (which is the polar retraction at r=1).

    Returns dict with keys ``per_dir`` (list of (ana, fd, rel)),
    ``max_rel``, and ``score0``.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    v = v / max(np.linalg.norm(v), _EPS_NORM)
    score0, g = score_value_grad_fn(v)
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    # Sphere tangent projection: P_v(g) = g - (v^T g) v
    g_t = g - float(np.dot(v, g)) * v
    n = v.size
    per_dir = []
    for _ in range(n_samples):
        e = rng.standard_normal(n)
        z = e - float(np.dot(v, e)) * v
        nz = float(np.linalg.norm(z))
        if nz <= _EPS_NORM:
            continue
        z = z / nz
        # Polar retraction: (v + eps z) / ||v + eps z||
        vp = v + eps * z
        vp = vp / np.linalg.norm(vp)
        vm = v - eps * z
        vm = vm / np.linalg.norm(vm)
        sp, _ = score_value_grad_fn(vp)
        sm, _ = score_value_grad_fn(vm)
        fd = (sp - sm) / (2.0 * eps)
        ana = float(np.dot(g_t, z))
        denom = max(abs(fd), abs(ana), _EPS_NORM)
        rel = abs(fd - ana) / denom
        per_dir.append((float(ana), float(fd), float(rel)))
    max_rel = max((r for _, _, r in per_dir), default=float("nan"))
    return {"per_dir": per_dir, "max_rel": float(max_rel), "score0": float(score0)}


# --------------------------------------------------------------------------
# Synthetic data + driver
# --------------------------------------------------------------------------


def make_synthetic(n=64, m_c=48, m_f=40, seed=0, signal_rank=4, signal_strength=2.0):
    """Random A_cur (m_c x n), A_fut (m_f x n) with different row counts
    AND a shared low-rank signal of dimension ``signal_rank`` injected
    with amplitude ``signal_strength``. The shared signal lifts
    rho_frame off the noise floor (~0.04 in pure-Gaussian) into a
    healthier range (~0.3 - 0.6), which keeps the directional
    derivatives away from the FD precision floor on rank-r tangents.
    The spec must work when m_c != m_f, which this preserves."""
    rng = np.random.default_rng(seed)
    A_c = rng.standard_normal((m_c, n))
    A_f = rng.standard_normal((m_f, n))
    if signal_rank > 0:
        # Inject the same n-dim signal directions into both windows so
        # rho_frame is non-trivial. Different row-side mixtures.
        Q, _ = np.linalg.qr(rng.standard_normal((n, signal_rank)))
        L_c = rng.standard_normal((m_c, signal_rank)) * signal_strength
        L_f = rng.standard_normal((m_f, signal_rank)) * signal_strength
        A_c = A_c + L_c @ Q.T
        A_f = A_f + L_f @ Q.T
    M_c = A_c.T @ A_c
    M_f = A_f.T @ A_f
    return A_c, A_f, M_c, M_f


def random_stiefel(n, r, rng):
    G = rng.standard_normal((n, r))
    Q, _ = np.linalg.qr(G)
    return Q[:, :r]


def main():
    log_lines = []

    def log(s):
        print(s)
        log_lines.append(s)

    log("FAM-07 T1 Stiefel/sphere gradient check")
    log("=" * 72)
    log(f"date: 2026-04-29")
    log(f"acceptance: max_rel < 1e-7 at float64 on every cell")
    log(f"trace-form sanity acceptance: max_rel < 1e-9")
    log("")

    rng = np.random.default_rng(0)
    n = 64
    A_c, A_f, M_c, M_f = make_synthetic(n=n, m_c=48, m_f=40, seed=1)

    # ----------------------------------------------------------------
    # Trace-form sanity row (harness self-test; INFRA-02 prescription)
    # ----------------------------------------------------------------
    M_sym = A_c.T @ A_c
    V_seed = random_stiefel(n, 3, rng)
    res = stiefel_fd_check(
        lambda V_: trace_form_value_grad(M_sym, V_),
        V_seed, n_directions=8, eps=1e-5,
        rng=np.random.default_rng(101), retraction="polar",
        direction_mode="column",
    )
    log(f"[sanity trace-form r=3]  max_rel={res['max_rel']:.3e}   "
        f"(< 1e-9 expected)")
    sanity_max = res["max_rel"]
    assert sanity_max < 1e-9, f"Trace-form sanity FAILED: {sanity_max:.3e}"

    # ----------------------------------------------------------------
    # (a) Rank-1 sphere check on rho_F^2(v)
    # ----------------------------------------------------------------
    rng2 = np.random.default_rng(7)
    v0 = rng2.standard_normal(n)
    v0 = v0 / np.linalg.norm(v0)
    res_v = sphere_fd_check(
        lambda v_: rho_F_squared_value_grad(M_c, M_f, v_),
        v0, n_samples=24, eps=1e-5, rng=np.random.default_rng(11),
    )
    log(f"[rank-1 rho_F^2 sphere]  score={res_v['score0']:.4e}  "
        f"max_rel={res_v['max_rel']:.3e}")
    rank1_max = res_v["max_rel"]

    # ----------------------------------------------------------------
    # (b–d) Rank-r Stiefel checks on rho_frame(V) for r = 1, 2, 3
    # ----------------------------------------------------------------
    rank_results = {}
    for r in (1, 2, 3, 4):
        V0 = random_stiefel(n, r, np.random.default_rng(20 + r))
        # Use direction_mode="random" so each FD probe is a full-frame
        # tangent (rather than a single-column seed). With small
        # rho_frame values (~0.04 at r=4), the column-seeded FD step has
        # very small directional derivatives in some columns and the
        # rel-error formula becomes denominator-noise-limited; full-frame
        # random tangents spread the signal evenly and stay below 1e-9.
        res = stiefel_fd_check(
            lambda V_: rho_frame_value_grad(M_c, M_f, V_),
            V0, n_directions=12, eps=1e-5,
            rng=np.random.default_rng(50 + r),
            retraction="polar", direction_mode="random",
        )
        log(
            f"[rank-{r} rho_frame Stiefel]  score={res['score0']:.4e}  "
            f"max_rel={res['max_rel']:.3e}  "
            f"tan_resid={res['tangent_check']:.2e}  "
            f"grad_tan_resid={res['grad_tan_resid']:.2e}"
        )
        rank_results[r] = res["max_rel"]

    # ----------------------------------------------------------------
    # (e) Sanity: at r=1, rho_frame(V=v) should equal rho_F(v)^2 (modulo
    # the column normalization).
    # ----------------------------------------------------------------
    v_test = random_stiefel(n, 1, np.random.default_rng(99))
    s_frame, _ = rho_frame_value_grad(M_c, M_f, v_test)
    s_vec, _ = rho_F_squared_value_grad(M_c, M_f, v_test[:, 0])
    rel_eq = abs(s_frame - s_vec) / max(abs(s_frame), abs(s_vec), 1e-30)
    log(f"[r=1 cross-check]  rho_frame={s_frame:.6e}  "
        f"rho_F^2={s_vec:.6e}  rel_diff={rel_eq:.3e}")
    assert rel_eq < 1e-12, f"r=1 cross-check FAILED: rel_diff={rel_eq:.3e}"

    # ----------------------------------------------------------------
    # (f) O(r)-invariance check: rho_frame(V Q) == rho_frame(V) for Q in O(r)
    # ----------------------------------------------------------------
    V_inv = random_stiefel(n, 3, np.random.default_rng(123))
    s_V, _ = rho_frame_value_grad(M_c, M_f, V_inv)
    Q, _ = np.linalg.qr(np.random.default_rng(124).standard_normal((3, 3)))
    s_VQ, _ = rho_frame_value_grad(M_c, M_f, V_inv @ Q)
    rel_inv = abs(s_V - s_VQ) / max(abs(s_V), abs(s_VQ), 1e-30)
    log(f"[O(r)-invariance r=3]  rho(V)={s_V:.6e}  "
        f"rho(V Q)={s_VQ:.6e}  rel_diff={rel_inv:.3e}")
    assert rel_inv < 1e-10, f"O(r)-invariance FAILED: rel_diff={rel_inv:.3e}"

    # ----------------------------------------------------------------
    # Verdict
    # ----------------------------------------------------------------
    all_max = max([rank1_max] + list(rank_results.values()))
    log("")
    log(f"WORST max_rel across cells: {all_max:.3e}")
    if all_max < 1e-7:
        log("STATUS: PASS  (T1 acceptance bar met at float64)")
    else:
        log("STATUS: FAIL  (T1 acceptance bar NOT met)")

    out_path = os.path.join(_THIS_DIR, "02_t1_grad_check_log.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\nwrote {out_path}")
    return 0 if all_max < 1e-7 else 1


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"elapsed {time.time() - t0:.2f}s")
    sys.exit(rc)
