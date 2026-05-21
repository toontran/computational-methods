"""Optimizer trajectory tracker (INFRA-04).

Tracked-ascent wrapper around the basis-restricted sphere optimizer used by
the score family (S6 / S6_GM / etc.). Re-implements the inner loop of
`optimize_future_hmean_in_basis` (future_hmean_optimizer_diagnostic.py:72)
verbatim — projected-gradient ascent in the basis coords `z` with backtracking
line-search — but exposes a per-iteration tracker callback so we can log
trajectory state for diagnosing the P4 plateau drift in FAM-01.

Per iter, the tracker receives:
  iter        — 0-indexed iteration number
  v           — n-vector (basis-lifted unit vector at iteration end)
  score       — score(v) at iteration end
  grad_full   — n-vector ambient score gradient at v
  grad_tan_z  — q-vector tangent gradient in basis coords (the "step direction")
  step        — q-vector actual step taken in basis coords (alpha * grad_tan_z)
  z           — q-vector current basis coord
  alpha       — line-search step size accepted (NaN on initial state, on
                tol-stop, or on line_search_fail)

The wrapper is single-restart: callers control the starting point and pass it
in as `z0`; restarts are an outer-loop concern handled in
`optimizer_trajectory_probe.py`.

Backlog: INFRA-04 in summary/overview/score_family_workflow.txt §5.
Resolves: toolkit §8 (b) in summary/overview/diagnostic_toolkit.txt.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


__all__ = [
    "tracked_ascend",
]


def tracked_ascend(
    value_grad_fn: Callable,
    B: np.ndarray,
    z0: np.ndarray,
    *,
    maxit: int = 120,
    tol: float = 1e-9,
    armijo_c1: float = 1e-4,
    line_search_max: int = 30,
    initial_alpha: float = 1.0,
    tracker: Optional[Callable] = None,
):
    """Projected-gradient sphere ascent in basis coords.

    Mirrors the inner loop of `optimize_future_hmean_in_basis` so that
    trajectories are directly comparable to the production optimizer. The
    tracker callback is invoked once at iteration -1 (initial state) and once
    per accepted iteration.

    Parameters
    ----------
    value_grad_fn : callable(_unused_cur, _unused_fut, v) -> (val, grad, *_extra)
        Same signature as the production scoring functions hooked through
        `make_r_sk_g_optimizer`. The first two positional args are unused
        (legacy from the future_hmean signature) — pass None / None or whatever
        the wrapper passes; they exist only to mimic the production optimizer's
        call site.
    B : (n, q) ndarray
        Basis whose column span we restrict the search to. Should be
        orthonormal.
    z0 : (q,) ndarray
        Initial basis coord; need not be unit-norm — wrapper normalizes.
    tracker : callable(record_dict) or None
        If not None, called once per emitted state with a dict containing the
        per-iter quantities documented in the module docstring.

    Returns
    -------
    dict with keys:
        vec, score, z, iters, stop_reason, grad_norm_final
    """
    B = np.ascontiguousarray(np.asarray(B, dtype=np.float64))
    n, q = B.shape

    z = np.asarray(z0, dtype=np.float64).reshape(-1).copy()
    nz = float(np.linalg.norm(z))
    if nz <= 1e-30:
        raise ValueError("z0 has near-zero norm; cannot start ascent")
    z /= nz

    v = B @ z
    val, grad_full, *_extra = value_grad_fn(None, None, v)
    val = float(val)
    grad_full = np.asarray(grad_full, dtype=np.float64).reshape(-1)

    gz = B.T @ grad_full
    gtan = gz - z * float(z @ gz)
    gnorm = float(np.linalg.norm(gtan))

    if tracker is not None:
        tracker(
            {
                "iter": -1,
                "v": v.copy(),
                "score": val,
                "grad_full": grad_full.copy(),
                "grad_tan_z": gtan.copy(),
                "step_z": np.zeros(q, dtype=np.float64),
                "z": z.copy(),
                "alpha": float("nan"),
                "accepted": False,
                "phase": "init",
            }
        )

    stop_reason = "maxit"
    accepted_iters = 0

    for it in range(int(maxit)):
        gz = B.T @ grad_full
        gtan = gz - z * float(z @ gz)
        gnorm = float(np.linalg.norm(gtan))
        if gnorm <= tol:
            stop_reason = "grad_tol"
            break

        alpha = float(initial_alpha)
        accepted = False
        for ls_iter in range(int(line_search_max)):
            zt = z + alpha * gtan
            nzt = float(np.linalg.norm(zt))
            if nzt > 1e-30:
                zt = zt / nzt
                vt = B @ zt
                val_t, grad_t, *_extra = value_grad_fn(None, None, vt)
                val_t = float(val_t)
                if np.isfinite(val_t) and val_t >= val + armijo_c1 * alpha * float(gtan @ gtan):
                    step_z = alpha * gtan
                    z = np.ascontiguousarray(zt)
                    val = val_t
                    grad_full = np.asarray(grad_t, dtype=np.float64).reshape(-1)
                    accepted = True
                    accepted_iters += 1
                    if tracker is not None:
                        tracker(
                            {
                                "iter": it,
                                "v": vt.copy(),
                                "score": val,
                                "grad_full": grad_full.copy(),
                                "grad_tan_z": gtan.copy(),
                                "step_z": step_z.copy(),
                                "z": z.copy(),
                                "alpha": float(alpha),
                                "accepted": True,
                                "phase": "step",
                            }
                        )
                    break
            alpha *= 0.5
        if not accepted:
            stop_reason = "line_search_fail"
            if tracker is not None:
                tracker(
                    {
                        "iter": it,
                        "v": v.copy() if False else (B @ z),
                        "score": val,
                        "grad_full": grad_full.copy(),
                        "grad_tan_z": gtan.copy(),
                        "step_z": np.zeros(q, dtype=np.float64),
                        "z": z.copy(),
                        "alpha": float("nan"),
                        "accepted": False,
                        "phase": "ls_fail",
                    }
                )
            break
    else:
        # Loop hit maxit without break.
        pass

    return {
        "vec": np.ascontiguousarray(B @ z, dtype=np.float64),
        "score": float(val),
        "z": np.ascontiguousarray(z, dtype=np.float64),
        "iters": int(accepted_iters),
        "stop_reason": stop_reason,
        "grad_norm_final": float(gnorm),
    }


# --------------------------------------------------------------------------
# Self-test (smoke): a simple Rayleigh-quotient ascent on a small SPD matrix.
# --------------------------------------------------------------------------


def _self_test():
    """Sanity: ascend `v ↦ <v, Mv>` on the unit sphere; the maximum is the
    top eigenvector and the score equals the top eigenvalue. Should converge
    in a handful of iterations and the tracker should record monotone score.
    """
    rng = np.random.default_rng(0)
    n = 8
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eig = np.array([5.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.2, 0.1])
    M = Q @ np.diag(eig) @ Q.T
    M = 0.5 * (M + M.T)

    def vg(_a, _b, v):
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        Mv = M @ v
        return float(v @ Mv), 2.0 * Mv

    history = []
    B = np.eye(n)
    z0 = rng.standard_normal(n)

    res = tracked_ascend(
        vg, B, z0, maxit=200, tol=1e-12, tracker=lambda r: history.append((r["iter"], r["score"]))
    )
    # final score should be ~ top eigenvalue 5.0
    assert abs(res["score"] - 5.0) < 1e-6, f"converged to {res['score']}, expected 5.0"
    # alignment check
    v_final = res["vec"]
    cos2 = float((v_final @ Q[:, 0]) ** 2)
    assert cos2 > 1.0 - 1e-6, f"top-eig alignment cos² = {cos2}"
    # tracker monotonicity (after init record at iter=-1)
    scores = [s for (i, s) in history if i >= 0]
    assert all(s_b - s_a >= -1e-12 for s_a, s_b in zip(scores[:-1], scores[1:])), \
        "tracked scores not monotone"
    print("optimizer_trajectory self-test: OK"
          f" (iters={res['iters']}, final={res['score']:.6f}, stop={res['stop_reason']})")


if __name__ == "__main__":
    _self_test()
