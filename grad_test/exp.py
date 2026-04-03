import argparse
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
from numpy.linalg import norm
from scipy.linalg import hadamard, qr, svd

from threadpoolctl import threadpool_limits
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

seed = 0
rng = np.random.default_rng(seed)
np.random.seed(seed)   # for legacy / scipy calls

# ============================================================
# Timing diagnostics
# ============================================================

@dataclass
class TimeStats:
    totals: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    def add(self, key: str, dt: float):
        self.totals[key] = self.totals.get(key, 0.0) + dt
        self.counts[key] = self.counts.get(key, 0) + 1

    def merge(self, other: "TimeStats"):
        for k, v in other.totals.items():
            self.totals[k] = self.totals.get(k, 0.0) + v
        for k, v in other.counts.items():
            self.counts[k] = self.counts.get(k, 0) + v

    def report(self, sort_by="time"):
        items = []
        for k in self.totals:
            total = self.totals[k]
            cnt = self.counts.get(k, 0)
            avg = total / cnt if cnt > 0 else 0.0
            items.append((k, total, cnt, avg))

        if sort_by == "time":
            items.sort(key=lambda x: -x[1])
        elif sort_by == "count":
            items.sort(key=lambda x: -x[2])
        else:
            items.sort(key=lambda x: x[0])

        lines = []
        for k, total, cnt, avg in items:
            lines.append(f"{k:40s} total={total:10.6f}s   count={cnt:8d}   avg={avg:10.6e}s")
        return "\n".join(lines)


@contextmanager
def timed(stats: Optional[TimeStats], key: str):
    if stats is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        stats.add(key, time.perf_counter() - t0)


# ============================================================
# Matvec diagnostics
# ============================================================

@dataclass
class MatvecStats:
    matvec_count: int = 0
    rmatvec_count: int = 0
    matvec_time: float = 0.0
    rmatvec_time: float = 0.0

    def add_matvec(self, dt: float, cols: int = 1):
        self.matvec_count += int(cols)
        self.matvec_time += float(dt)

    def add_rmatvec(self, dt: float, cols: int = 1):
        self.rmatvec_count += int(cols)
        self.rmatvec_time += float(dt)

    def merge(self, other: "MatvecStats"):
        self.matvec_count += other.matvec_count
        self.rmatvec_count += other.rmatvec_count
        self.matvec_time += other.matvec_time
        self.rmatvec_time += other.rmatvec_time

    @property
    def matvec_pairs(self) -> int:
        return min(self.matvec_count, self.rmatvec_count)

    @property
    def total_time(self) -> float:
        return self.matvec_time + self.rmatvec_time

    def as_dict(self) -> Dict[str, float]:
        return {
            "MATVEC_COUNT": self.matvec_count,
            "RMATVEC_COUNT": self.rmatvec_count,
            "MATVEC_PAIRS": self.matvec_pairs,
            "matvec_time": self.matvec_time,
            "rmatvec_time": self.rmatvec_time,
            "matvec_total_time": self.total_time,
        }


def _num_rhs(x: np.ndarray) -> int:
    return 1 if x.ndim == 1 else int(x.shape[1])


def matvec(M: np.ndarray, v: np.ndarray, mvstats: Optional[MatvecStats] = None) -> np.ndarray:
    t0 = time.perf_counter()
    out = M @ v
    if mvstats is not None:
        mvstats.add_matvec(time.perf_counter() - t0, cols=_num_rhs(v))
    return out


def rmatvec(M: np.ndarray, u: np.ndarray, mvstats: Optional[MatvecStats] = None) -> np.ndarray:
    t0 = time.perf_counter()
    out = M.T @ u
    if mvstats is not None:
        mvstats.add_rmatvec(time.perf_counter() - t0, cols=_num_rhs(u))
    return out


def avg_matvec_pair_time(M: np.ndarray, trials: int = 50, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = M.shape[1]
    times: List[float] = []
    for _ in range(trials):
        v = rng.standard_normal(n)
        v /= max(norm(v), 1e-30)
        t0 = time.perf_counter()
        u = M @ v
        _ = M.T @ u
        times.append(time.perf_counter() - t0)
    arr = np.array(times, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "trials": int(trials),
    }


def plain_power_iteration(M: np.ndarray, iters: int = 50, seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(M.shape[1])
    v /= max(norm(v), 1e-30)
    mvstats = MatvecStats()
    t0 = time.perf_counter()
    for _ in range(iters):
        u = matvec(M, v, mvstats)
        u /= max(norm(u), 1e-30)
        v = rmatvec(M, u, mvstats)
        v /= max(norm(v), 1e-30)
    wall = time.perf_counter() - t0
    sigma = float(norm(M @ v))
    return {
        "iters": int(iters),
        "sigma_est": sigma,
        "wall_time": wall,
        "matvec_stats": mvstats,
        "vector": v,
    }


# ============================================================
# Stop + progress diagnostics
# ============================================================

@dataclass
class StopDiagnostics:
    reason: str = "unknown"
    iters: int = 0

    grad_norm: float = np.inf
    grad_tol: float = np.nan

    step_norm: float = np.inf
    step_tol: float = np.nan

    f_change: float = np.inf
    f_threshold: float = np.nan

    tr_radius: float = np.nan
    rho: float = np.nan
    pred: float = np.nan
    ared: float = np.nan

    accepted: bool = False
    line_search_alpha: float = np.nan
    line_search_steps: int = 0

    solver: str = ""
    note: str = ""

    def as_dict(self):
        out: Dict[str, Any] = {
            "reason": self.reason,
            "iters": self.iters,
            "grad_norm": self.grad_norm,
            "grad_tol": self.grad_tol,
            "step_norm": self.step_norm,
            "step_tol": self.step_tol,
            "accepted": self.accepted,
            "solver": self.solver,
        }
        if np.isfinite(self.f_change):
            out["f_change"] = self.f_change
        if np.isfinite(self.f_threshold):
            out["f_threshold"] = self.f_threshold
            out["f_ratio"] = self.f_change / self.f_threshold if self.f_threshold > 0 else np.inf
        if np.isfinite(self.tr_radius):
            out["tr_radius"] = self.tr_radius
        if np.isfinite(self.rho):
            out["rho"] = self.rho
        if np.isfinite(self.pred):
            out["pred"] = self.pred
        if np.isfinite(self.ared):
            out["ared"] = self.ared
        if np.isfinite(self.line_search_alpha):
            out["line_search_alpha"] = self.line_search_alpha
        if self.line_search_steps:
            out["line_search_steps"] = self.line_search_steps
        if self.note:
            out["note"] = self.note
        return out


@dataclass
class ProgressDiagnostics:
    first_score: float = np.nan
    last_score: float = np.nan
    best_score: float = -np.inf

    last_f_change: float = np.nan
    last_step_norm: float = np.nan
    last_cos_step: float = np.nan
    last_armijo_margin: float = np.nan

    min_f_change: float = np.inf
    max_f_change: float = -np.inf

    min_step_norm: float = np.inf
    max_step_norm: float = -np.inf

    num_updates: int = 0

    def init_score(self, f0: float):
        self.first_score = f0
        self.last_score = f0
        self.best_score = max(self.best_score, f0)

    def update(self, f_old, f_new, v_old, v_new, armijo_margin=np.nan):
        step_norm = norm(v_new - v_old)
        cos_step = abs(float(v_old @ v_new))
        f_change = f_new - f_old

        if not np.isfinite(self.first_score):
            self.first_score = f_old
        self.last_score = f_new
        self.best_score = max(self.best_score, f_new)

        self.last_f_change = f_change
        self.last_step_norm = step_norm
        self.last_cos_step = cos_step
        self.last_armijo_margin = armijo_margin

        self.min_f_change = min(self.min_f_change, f_change)
        self.max_f_change = max(self.max_f_change, f_change)

        self.min_step_norm = min(self.min_step_norm, step_norm)
        self.max_step_norm = max(self.max_step_norm, step_norm)

        self.num_updates += 1

    def no_update(self):
        self.last_f_change = 0.0
        self.last_step_norm = 0.0
        self.last_cos_step = 1.0
        self.last_armijo_margin = np.nan

    def as_dict(self):
        total_gain = np.nan
        if np.isfinite(self.first_score) and np.isfinite(self.last_score):
            total_gain = self.last_score - self.first_score
        return {
            "first_score": self.first_score,
            "last_score": self.last_score,
            "best_score": self.best_score,
            "total_gain": total_gain,
            "last_f_change": self.last_f_change,
            "last_step_norm": self.last_step_norm,
            "last_cos_step": self.last_cos_step,
            "last_armijo_margin": self.last_armijo_margin,
            "min_f_change": self.min_f_change,
            "max_f_change": self.max_f_change,
            "min_step_norm": self.min_step_norm,
            "max_step_norm": self.max_step_norm,
            "num_updates": self.num_updates,
        }


# ============================================================
# Options
# ============================================================

@dataclass
class ContinuationOptions:
    num_stages: int = 8
    max_subdivide: int = 8
    num_hedge: int = 6
    perturb_scale_small: float = 1e-3
    perturb_scale_large: float = 1e-2
    local_maxit: int = 200
    local_tol: float = 1e-8
    local_ls_max: int = 20
    accept_armijo: float = 1e-4
    fail_gtan: float = 5e-4
    fail_cos: float = 0.50
    min_dc: float = 1e-4
    progress_f_tol: float = 1e-12
    progress_step_tol: float = 1e-10
    verbose: bool = True
    time_stats: Optional[TimeStats] = None
    matvec_stats: Optional[MatvecStats] = None


@dataclass
class EntropyOptions:
    solver: str = "trust_region"   # "trust_region", "newton", "rbfgs", "rcg"
    num_restarts: int = 12
    maxit: int = 200
    tol: float = 1e-8
    grad_tol: float = 1e-8
    step_tol: float = 1e-10
    f_tol: float = 1e-12
    progress_f_tol: float = 1e-12
    progress_step_tol: float = 1e-10
    tr_radius0: float = 0.25
    tr_radius_max: float = 2.0
    tr_eta1: float = 0.10
    tr_eta2: float = 0.75
    tr_shrink: float = 0.25
    tr_expand: float = 2.0
    cg_maxit: int = 100
    cg_tol: float = 1e-10
    line_search_c1: float = 1e-4
    line_search_beta: float = 0.5
    line_search_maxit: int = 20
    use_negcurv_escape: bool = True
    negcurv_when_grad_below: float = 1e-5
    negcurv_tol: float = -1e-8
    negcurv_iters: int = 30
    negcurv_step_scales: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.35, 0.5)
    warm_start_weight: float = 0.90
    mix_alphas: Tuple[float, ...] = (1.0, 0.98, 0.9, 0.7, 0.4, 0.0)
    num_random_mixtures: int = 4
    verbose: bool = False
    time_stats: Optional[TimeStats] = None
    matvec_stats: Optional[MatvecStats] = None


# ============================================================
# Basic helpers
# ============================================================

def safe_norm(x: np.ndarray, eps: float = 1e-30) -> float:
    return float(max(norm(x), eps))


def kahan_sum(x: np.ndarray) -> float:
    s = 0.0
    c = 0.0
    for y in np.ravel(x):
        y2 = float(y) - c
        t = s + y2
        c = (t - s) - y2
        s = t
    return s


def project_feasible(x: np.ndarray, Q: np.ndarray) -> np.ndarray:
    if Q.size == 0:
        return x.copy()
    return x - Q @ (Q.T @ x)


def tangent_proj(x: np.ndarray, v: np.ndarray, Q: np.ndarray) -> np.ndarray:
    y = project_feasible(x, Q)
    y = y - v * float(v @ y)
    return y


def retract_feasible(x: np.ndarray, Q: np.ndarray, eps: float = 1e-14) -> Optional[np.ndarray]:
    y = project_feasible(x, Q)
    ny = norm(y)
    if ny <= eps:
        return None
    return y / ny


def feasible_random(d: int, Q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    for _ in range(100):
        v = retract_feasible(rng.standard_normal(d), Q)
        if v is not None:
            return v
    raise RuntimeError("Could not generate feasible random vector.")


def transport_tangent(x: np.ndarray, v_old: np.ndarray, v_new: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return tangent_proj(x, v_new, Q)


def target_c_from_win_n(win: int, n: int) -> float:
    return 2.0 * math.log(win / n) / math.log(win)


def continuation_grid(c_target: float, num_stages: int) -> np.ndarray:
    return np.linspace(0.0, c_target, num_stages + 1)


# ============================================================
# Entropy-score objective
# ============================================================

def entropy_logscore_grad_c(
    M: np.ndarray,
    v: np.ndarray,
    c: float,
    stats: Optional[TimeStats] = None,
    mvstats: Optional[MatvecStats] = None,
):
    with timed(stats, "entropy_logscore_grad_c"):
        y = matvec(M, v, mvstats)
        y2 = safe_norm(y)
        y4_4 = max(float(np.sum(y**4)), 1e-30)
        y4 = y4_4 ** 0.25

        logf = (1.0 - c) * math.log(y2) + c * math.log(y4)

        g2 = rmatvec(M, y, mvstats) / (y2**2)
        g4 = rmatvec(M, y**3, mvstats) / y4_4
        grad = (1.0 - c) * g2 + c * g4

        H2 = -(math.log(y4_4) - 2.0 * math.log(y2**2))
        return logf, grad, y2, H2


def entropy_logscore_grad(
    M: np.ndarray,
    v: np.ndarray,
    win: int,
    n: int,
    stats: Optional[TimeStats] = None,
    mvstats: Optional[MatvecStats] = None,
):
    with timed(stats, "entropy_logscore_grad"):
        c = target_c_from_win_n(win, n)
        return entropy_logscore_grad_c(M, v, c, stats=stats, mvstats=mvstats)


def entropy_score_fast_c(
    M: np.ndarray,
    v: np.ndarray,
    c: float,
    stats: Optional[TimeStats] = None,
    mvstats: Optional[MatvecStats] = None,
) -> float:
    with timed(stats, "entropy_score_fast_c"):
        return float(math.exp(entropy_logscore_grad_c(M, v, c, stats=stats, mvstats=mvstats)[0]))


def entropy_score_fast(
    M: np.ndarray,
    v: np.ndarray,
    win: int,
    n: int,
    stats: Optional[TimeStats] = None,
    mvstats: Optional[MatvecStats] = None,
) -> float:
    with timed(stats, "entropy_score_fast"):
        return float(math.exp(entropy_logscore_grad(M, v, win, n, stats=stats, mvstats=mvstats)[0]))


# ============================================================
# Finite-difference Riemannian Hessian actions
# ============================================================

def riem_grad(
    M: np.ndarray,
    v: np.ndarray,
    Q: np.ndarray,
    win: int,
    n: int,
    stats: Optional[TimeStats] = None,
    mvstats: Optional[MatvecStats] = None,
) -> np.ndarray:
    _, gradE, _, _ = entropy_logscore_grad(M, v, win, n, stats=stats, mvstats=mvstats)
    return tangent_proj(gradE, v, Q)


def riem_grad_c(
    M: np.ndarray,
    v: np.ndarray,
    Q: np.ndarray,
    c: float,
    stats: Optional[TimeStats] = None,
    mvstats: Optional[MatvecStats] = None,
) -> np.ndarray:
    _, gradE, _, _ = entropy_logscore_grad_c(M, v, c, stats=stats, mvstats=mvstats)
    return tangent_proj(gradE, v, Q)


def riem_hess_mult(
    M: np.ndarray,
    v: np.ndarray,
    Q: np.ndarray,
    win: int,
    n: int,
    p: np.ndarray,
    h: float = 1e-6,
    stats: Optional[TimeStats] = None,
    mvstats: Optional[MatvecStats] = None,
) -> np.ndarray:
    with timed(stats, "riem_hess_mult"):
        p = tangent_proj(p, v, Q)
        np_ = norm(p)
        if np_ <= 1e-16:
            return np.zeros_like(v)
        p = p / np_
        vp = retract_feasible(v + h * p, Q)
        vm = retract_feasible(v - h * p, Q)
        if vp is None or vm is None:
            return np.zeros_like(v)
        gp = riem_grad(M, vp, Q, win, n, stats=stats, mvstats=mvstats)
        gm = riem_grad(M, vm, Q, win, n, stats=stats, mvstats=mvstats)
        return (gp - gm) / (2.0 * h)


# ============================================================
# Trust-region and line search
# ============================================================

def tau_to_boundary(eta: np.ndarray, p: np.ndarray, Delta: float) -> float:
    a = float(p @ p)
    b = 2.0 * float(eta @ p)
    c = float(eta @ eta) - Delta**2
    disc = max(b*b - 4.0*a*c, 0.0)
    return (-b + math.sqrt(disc)) / (2.0 * a)


def armijo_ascent_step(
    M: np.ndarray,
    v: np.ndarray,
    Q: np.ndarray,
    win: int,
    n: int,
    f: float,
    g: np.ndarray,
    eta: np.ndarray,
    opts: EntropyOptions,
):
    with timed(opts.time_stats, "armijo_ascent_step"):
        accepted = False
        v_new = v
        f_new = f
        g_new = g
        used_alpha = np.nan
        ls_steps = 0
        armijo_margin = np.nan

        eta = tangent_proj(eta, v, Q)
        if norm(eta) <= opts.step_tol:
            return accepted, v_new, f_new, g_new, used_alpha, ls_steps, armijo_margin

        slope = float(g @ eta)
        if slope <= 0:
            return accepted, v_new, f_new, g_new, used_alpha, ls_steps, armijo_margin

        alpha = 1.0
        for j in range(opts.line_search_maxit):
            ls_steps = j + 1
            vt = retract_feasible(v + alpha * eta, Q)
            if vt is not None:
                ft, gradEt, _, _ = entropy_logscore_grad(M, vt, win, n, stats=opts.time_stats, mvstats=opts.matvec_stats)
                rhs = f + opts.line_search_c1 * alpha * slope
                if ft >= rhs:
                    accepted = True
                    v_new = vt
                    f_new = ft
                    g_new = tangent_proj(gradEt, vt, Q)
                    used_alpha = alpha
                    armijo_margin = ft - rhs
                    return accepted, v_new, f_new, g_new, used_alpha, ls_steps, armijo_margin
            alpha *= opts.line_search_beta

        return accepted, v_new, f_new, g_new, used_alpha, ls_steps, armijo_margin


def tr_steihaug_step(
    M: np.ndarray,
    v: np.ndarray,
    Q: np.ndarray,
    win: int,
    n: int,
    g: np.ndarray,
    Delta: float,
    opts: EntropyOptions,
):
    with timed(opts.time_stats, "tr_steihaug_step"):
        eta = np.zeros_like(v)
        r = -g.copy()
        p = -r.copy()
        hit_boundary = False

        if norm(r) <= opts.cg_tol:
            return eta, 0.0, hit_boundary

        for _ in range(opts.cg_maxit):
            Hp = -riem_hess_mult(M, v, Q, win, n, p, stats=opts.time_stats, mvstats=opts.matvec_stats)
            pHp = float(p @ Hp)

            if pHp <= 0:
                tau = tau_to_boundary(eta, p, Delta)
                eta = eta + tau * p
                hit_boundary = True
                break

            alpha = float(r @ r) / pHp
            eta_next = eta + alpha * p
            if norm(eta_next) >= Delta:
                tau = tau_to_boundary(eta, p, Delta)
                eta = eta + tau * p
                hit_boundary = True
                break

            r_next = r + alpha * Hp
            eta = eta_next

            if norm(r_next) <= opts.cg_tol:
                r = r_next
                break

            beta = float(r_next @ r_next) / max(float(r @ r), 1e-30)
            p = -r_next + beta * p
            r = r_next

        Heta = riem_hess_mult(M, v, Q, win, n, eta, stats=opts.time_stats, mvstats=opts.matvec_stats)
        pred = float(g @ eta + 0.5 * eta @ Heta)
        return eta, pred, hit_boundary


def truncated_newton_direction(
    M: np.ndarray,
    v: np.ndarray,
    Q: np.ndarray,
    win: int,
    n: int,
    g: np.ndarray,
    opts: EntropyOptions,
) -> np.ndarray:
    with timed(opts.time_stats, "truncated_newton_direction"):
        lam = 1e-6
        eta = np.zeros_like(v)
        r = -g.copy()
        p = r.copy()

        for _ in range(opts.cg_maxit):
            Hp = riem_hess_mult(M, v, Q, win, n, p, stats=opts.time_stats, mvstats=opts.matvec_stats) + lam * p
            pHp = float(p @ Hp)
            if abs(pHp) <= 1e-20:
                break
            alpha = float(r @ r) / pHp
            eta = eta + alpha * p
            r_new = r - alpha * Hp
            if norm(r_new) <= opts.cg_tol:
                break
            beta = float(r_new @ r_new) / max(float(r @ r), 1e-30)
            p = r_new + beta * p
            r = r_new
        return tangent_proj(eta, v, Q)


def estimate_negative_curvature_direction(
    M: np.ndarray,
    v: np.ndarray,
    Q: np.ndarray,
    win: int,
    n: int,
    opts: EntropyOptions,
    rng: np.random.Generator,
):
    with timed(opts.time_stats, "estimate_negative_curvature_direction"):
        best_dir = None
        best_curv = np.inf
        d = len(v)
        for _ in range(opts.negcurv_iters):
            z = tangent_proj(rng.standard_normal(d), v, Q)
            nz = norm(z)
            if nz <= 1e-14:
                continue
            z /= nz
            Hz = riem_hess_mult(M, v, Q, win, n, z, stats=opts.time_stats, mvstats=opts.matvec_stats)
            curv = float(z @ Hz)
            if curv < best_curv:
                best_curv = curv
                best_dir = z
        return best_dir, best_curv


def try_negative_curvature_escape(
    M: np.ndarray,
    v: np.ndarray,
    Q: np.ndarray,
    win: int,
    n: int,
    opts: EntropyOptions,
    rng: np.random.Generator,
):
    with timed(opts.time_stats, "try_negative_curvature_escape"):
        f0, _, _, _ = entropy_logscore_grad(M, v, win, n, stats=opts.time_stats, mvstats=opts.matvec_stats)
        z, curv = estimate_negative_curvature_direction(M, v, Q, win, n, opts, rng)
        if z is None or curv >= opts.negcurv_tol:
            return False, v, f0, {"curv": curv, "reason": "no_negative_curvature"}

        best_v = v
        best_f = f0
        best_step = np.nan
        best_sign = 0.0
        for s in opts.negcurv_step_scales:
            for sign in (-1.0, 1.0):
                vt = retract_feasible(v + sign * s * z, Q)
                if vt is None:
                    continue
                ft, _, _, _ = entropy_logscore_grad(M, vt, win, n, stats=opts.time_stats, mvstats=opts.matvec_stats)
                if ft > best_f:
                    best_f = ft
                    best_v = vt
                    best_step = s
                    best_sign = sign
        return best_f > f0, best_v, best_f, {
            "curv": curv,
            "reason": "improved" if best_f > f0 else "no_improvement",
            "best_step": best_step,
            "best_sign": best_sign,
        }


# ============================================================
# L-BFGS
# ============================================================

def update_lbfgs_hist(s_hist, y_hist, svec, yvec, mem):
    s_hist = list(s_hist)
    y_hist = list(y_hist)
    s_hist.append(svec.copy())
    y_hist.append(yvec.copy())
    if len(s_hist) > mem:
        s_hist = s_hist[-mem:]
        y_hist = y_hist[-mem:]
    return s_hist, y_hist


def lbfgs_two_loop(g, s_hist, y_hist):
    q = g.copy()
    alpha = []
    rho = []

    for s, y in zip(reversed(s_hist), reversed(y_hist)):
        rhoi = 1.0 / max(float(y @ s), 1e-30)
        ai = rhoi * float(s @ q)
        q = q - ai * y
        rho.append(rhoi)
        alpha.append(ai)

    if len(s_hist) > 0:
        s = s_hist[-1]
        y = y_hist[-1]
        gamma = float(s @ y) / max(float(y @ y), 1e-30)
    else:
        gamma = 1.0
    r = gamma * q

    for i, (s, y) in enumerate(zip(s_hist, y_hist)):
        rhoi = rho[-1 - i]
        ai = alpha[-1 - i]
        beta = rhoi * float(y @ r)
        r = r + s * (ai - beta)

    return r


# ============================================================
# Shared progress stop helper
# ============================================================

def progress_stop_check(f_old: float, f_new: float, step_norm: float, f_tol: float, step_tol: float):
    f_change = abs(f_new - f_old)
    f_threshold = f_tol * max(1.0, abs(f_old))
    stop_f = f_change <= f_threshold
    stop_step = step_norm <= step_tol
    return stop_f, stop_step, f_change, f_threshold


# ============================================================
# Structured starts and manifold optimizer
# ============================================================

def build_structured_starts(M, Q, V_init, k, opts: EntropyOptions, rng):
    with timed(opts.time_stats, "build_structured_starts.total"):
        d = M.shape[1]
        starts = []

        if V_init is not None and V_init.shape[1] >= k:
            v = retract_feasible(V_init[:, k - 1], Q)
            if v is not None:
                starts.append(v)

        with timed(opts.time_stats, "build_structured_starts.svd"):
            with threadpool_limits(limits=8, user_api="blas"):
                _, _, Vh = svd(M, full_matrices=False)
        Vsvd = Vh.T
        num_v = min(Vsvd.shape[1], 4)

        for j in range(min(num_v, 3)):
            vj = retract_feasible(Vsvd[:, j], Q)
            if vj is not None:
                starts.append(vj)

        for j in range(min(num_v, 3)):
            vj = retract_feasible(Vsvd[:, j], Q)
            if vj is None:
                continue
            for a in opts.mix_alphas:
                noise = feasible_random(d, Q, rng)
                v = retract_feasible(a * vj + math.sqrt(max(0.0, 1.0 - a*a)) * noise, Q)
                if v is not None:
                    starts.append(v)

        if V_init is not None and V_init.shape[1] >= k:
            vw = retract_feasible(V_init[:, k - 1], Q)
            if vw is not None:
                for _ in range(opts.num_random_mixtures):
                    noise = feasible_random(d, Q, rng)
                    a = opts.warm_start_weight
                    v = retract_feasible(a * vw + math.sqrt(max(0.0, 1.0 - a*a)) * noise, Q)
                    if v is not None:
                        starts.append(v)

        while len(starts) < opts.num_restarts:
            starts.append(feasible_random(d, Q, rng))

        return starts[:opts.num_restarts]


def optimize_on_feasible_sphere(M, v0, Q, win, n, opts: EntropyOptions, rng):
    with timed(opts.time_stats, "optimize_on_feasible_sphere.total"):
        v = retract_feasible(v0, Q)
        if v is None:
            v = feasible_random(M.shape[1], Q, rng)

        logf, gradE, _, _ = entropy_logscore_grad(M, v, win, n, stats=opts.time_stats, mvstats=opts.matvec_stats)
        g = tangent_proj(gradE, v, Q)

        solver = opts.solver
        tr_radius = opts.tr_radius0
        s_hist, y_hist = [], []
        mem = 10
        p_prev = None
        g_prev = None
        v_prev_saved = None

        stop = StopDiagnostics(
            reason="not_started",
            grad_tol=opts.grad_tol,
            step_tol=opts.step_tol,
            solver=solver,
        )
        prog = ProgressDiagnostics()
        prog.init_score(logf)

        for it in range(opts.maxit):
            with timed(opts.time_stats, f"optimize_on_feasible_sphere.iter[{solver}]"):
                gnorm = norm(g)

                if gnorm <= opts.grad_tol:
                    if opts.use_negcurv_escape:
                        did_escape, v_new, f_new, esc_info = try_negative_curvature_escape(M, v, Q, win, n, opts, rng)
                        if did_escape and f_new > logf + opts.f_tol:
                            prog.update(logf, f_new, v, v_new, np.nan)
                            v = v_new
                            logf = f_new
                            _, gradE, _, _ = entropy_logscore_grad(M, v, win, n, stats=opts.time_stats, mvstats=opts.matvec_stats)
                            g = tangent_proj(gradE, v, Q)
                            p_prev = None
                            g_prev = None
                            continue
                        stop.note = f"negcurv_escape={esc_info}"

                    stop.reason = "grad_tol"
                    stop.iters = it + 1
                    stop.grad_norm = gnorm
                    stop.accepted = True
                    break

                if solver == "trust_region":
                    eta, pred, hit_boundary = tr_steihaug_step(M, v, Q, win, n, g, tr_radius, opts)
                    eta_norm = norm(eta)

                    if eta_norm <= opts.step_tol:
                        stop.reason = "step_tol"
                        stop.iters = it + 1
                        stop.grad_norm = gnorm
                        stop.step_norm = eta_norm
                        stop.tr_radius = tr_radius
                        stop.pred = pred
                        stop.accepted = False
                        break

                    v_trial = retract_feasible(v + eta, Q)
                    if v_trial is None:
                        tr_radius = max(opts.tr_shrink * tr_radius, 1e-12)
                        stop.reason = "trial_retraction_failed"
                        stop.iters = it + 1
                        stop.grad_norm = gnorm
                        stop.step_norm = eta_norm
                        stop.tr_radius = tr_radius
                        stop.pred = pred
                        stop.accepted = False
                        continue

                    f_trial, _, _, _ = entropy_logscore_grad(M, v_trial, win, n, stats=opts.time_stats)
                    ared = f_trial - logf
                    rho = ared / max(abs(pred), 1e-16)

                    if rho < opts.tr_eta1:
                        tr_radius = max(opts.tr_shrink * tr_radius, 1e-12)
                        stop.reason = "tr_rejected_step"
                        stop.iters = it + 1
                        stop.grad_norm = gnorm
                        stop.step_norm = eta_norm
                        stop.tr_radius = tr_radius
                        stop.pred = pred
                        stop.ared = ared
                        stop.rho = rho
                        stop.accepted = False
                    else:
                        v_old = v.copy()
                        g_old = g.copy()
                        f_old = logf

                        v = v_trial
                        logf = f_trial
                        prog.update(f_old, logf, v_old, v, np.nan)

                        _, gradE, _, _ = entropy_logscore_grad(M, v, win, n, stats=opts.time_stats, mvstats=opts.matvec_stats)
                        g = tangent_proj(gradE, v, Q)

                        svec = tangent_proj(v - v_old, v, Q)
                        yvec = tangent_proj(g - transport_tangent(g_old, v_old, v, Q), v, Q)
                        if float(svec @ yvec) > 1e-12 * norm(svec) * norm(yvec):
                            s_hist, y_hist = update_lbfgs_hist(s_hist, y_hist, svec, yvec, mem)

                        if rho > opts.tr_eta2 and hit_boundary:
                            tr_radius = min(opts.tr_expand * tr_radius, opts.tr_radius_max)

                        stop_f, stop_step, f_change, f_threshold = progress_stop_check(
                            f_old, logf, prog.last_step_norm, opts.progress_f_tol, opts.progress_step_tol
                        )

                        if stop_f:
                            stop.reason = "progress_f_tol"
                            stop.iters = it + 1
                            stop.grad_norm = norm(g)
                            stop.step_norm = prog.last_step_norm
                            stop.f_change = f_change
                            stop.f_threshold = f_threshold
                            stop.tr_radius = tr_radius
                            stop.rho = rho
                            stop.pred = pred
                            stop.ared = ared
                            stop.accepted = True
                            break

                        if stop_step:
                            stop.reason = "progress_step_tol"
                            stop.iters = it + 1
                            stop.grad_norm = norm(g)
                            stop.step_norm = prog.last_step_norm
                            stop.f_change = f_change
                            stop.f_threshold = f_threshold
                            stop.tr_radius = tr_radius
                            stop.rho = rho
                            stop.pred = pred
                            stop.ared = ared
                            stop.accepted = True
                            break

                        f_change_std = abs(logf - f_old)
                        f_threshold_std = opts.f_tol * max(1.0, abs(f_old))
                        if f_change_std <= f_threshold_std:
                            stop.reason = "f_tol"
                            stop.iters = it + 1
                            stop.grad_norm = norm(g)
                            stop.step_norm = eta_norm
                            stop.f_change = f_change_std
                            stop.f_threshold = f_threshold_std
                            stop.tr_radius = tr_radius
                            stop.rho = rho
                            stop.pred = pred
                            stop.ared = ared
                            stop.accepted = True
                            break

                elif solver == "newton":
                    eta = truncated_newton_direction(M, v, Q, win, n, g, opts)
                    eta_norm = norm(eta)
                    if eta_norm <= opts.step_tol:
                        stop.reason = "step_tol"
                        stop.iters = it + 1
                        stop.grad_norm = gnorm
                        stop.step_norm = eta_norm
                        stop.accepted = False
                        break

                    f_old = logf
                    v_old = v.copy()
                    accepted, v, logf, g, used_alpha, ls_steps, armijo_margin = armijo_ascent_step(M, v, Q, win, n, logf, g, eta, opts)
                    stop.line_search_alpha = used_alpha
                    stop.line_search_steps = ls_steps
                    if not accepted:
                        did_escape, v_new, f_new, esc_info = try_negative_curvature_escape(M, v, Q, win, n, opts, rng)
                        if did_escape and f_new > logf:
                            prog.update(logf, f_new, v, v_new, np.nan)
                            v = v_new
                            logf = f_new
                            _, gradE, _, _ = entropy_logscore_grad(M, v, win, n, stats=opts.time_stats, mvstats=opts.matvec_stats)
                            g = tangent_proj(gradE, v, Q)
                        else:
                            stop.reason = "newton_no_progress"
                            stop.iters = it + 1
                            stop.grad_norm = gnorm
                            stop.step_norm = eta_norm
                            stop.accepted = False
                            stop.note = f"negcurv_escape={esc_info}"
                            break
                    else:
                        prog.update(f_old, logf, v_old, v, armijo_margin)
                        stop_f, stop_step, f_change, f_threshold = progress_stop_check(
                            f_old, logf, prog.last_step_norm, opts.progress_f_tol, opts.progress_step_tol
                        )
                        if stop_f:
                            stop.reason = "progress_f_tol"
                            stop.iters = it + 1
                            stop.grad_norm = norm(g)
                            stop.step_norm = prog.last_step_norm
                            stop.f_change = f_change
                            stop.f_threshold = f_threshold
                            stop.accepted = True
                            break
                        if stop_step:
                            stop.reason = "progress_step_tol"
                            stop.iters = it + 1
                            stop.grad_norm = norm(g)
                            stop.step_norm = prog.last_step_norm
                            stop.f_change = f_change
                            stop.f_threshold = f_threshold
                            stop.accepted = True
                            break

                elif solver == "rbfgs":
                    eta = lbfgs_two_loop(g, s_hist, y_hist) if len(s_hist) > 0 else g.copy()
                    eta = tangent_proj(eta, v, Q)
                    if float(g @ eta) <= 0:
                        eta = g.copy()

                    eta_norm = norm(eta)
                    if eta_norm <= opts.step_tol:
                        stop.reason = "step_tol"
                        stop.iters = it + 1
                        stop.grad_norm = gnorm
                        stop.step_norm = eta_norm
                        stop.accepted = False
                        break

                    v_old = v.copy()
                    g_old = g.copy()
                    f_old = logf

                    accepted, v, logf, g, used_alpha, ls_steps, armijo_margin = armijo_ascent_step(M, v, Q, win, n, logf, g, eta, opts)
                    stop.line_search_alpha = used_alpha
                    stop.line_search_steps = ls_steps
                    if not accepted:
                        accepted, v, logf, g, used_alpha, ls_steps2, armijo_margin = armijo_ascent_step(M, v, Q, win, n, logf, g, g, opts)
                        stop.line_search_alpha = used_alpha
                        stop.line_search_steps += ls_steps2
                        if not accepted:
                            stop.reason = "rbfgs_line_search_failed"
                            stop.iters = it + 1
                            stop.grad_norm = gnorm
                            stop.step_norm = eta_norm
                            stop.accepted = False
                            break

                    prog.update(f_old, logf, v_old, v, armijo_margin)

                    svec = tangent_proj(v - v_old, v, Q)
                    yvec = tangent_proj(g - transport_tangent(g_old, v_old, v, Q), v, Q)
                    if float(svec @ yvec) > 1e-12 * norm(svec) * norm(yvec):
                        s_hist, y_hist = update_lbfgs_hist(s_hist, y_hist, svec, yvec, mem)

                    stop_f, stop_step, f_change, f_threshold = progress_stop_check(
                        f_old, logf, prog.last_step_norm, opts.progress_f_tol, opts.progress_step_tol
                    )
                    if stop_f:
                        stop.reason = "progress_f_tol"
                        stop.iters = it + 1
                        stop.grad_norm = norm(g)
                        stop.step_norm = prog.last_step_norm
                        stop.f_change = f_change
                        stop.f_threshold = f_threshold
                        stop.accepted = True
                        break
                    if stop_step:
                        stop.reason = "progress_step_tol"
                        stop.iters = it + 1
                        stop.grad_norm = norm(g)
                        stop.step_norm = prog.last_step_norm
                        stop.f_change = f_change
                        stop.f_threshold = f_threshold
                        stop.accepted = True
                        break

                    f_change_std = abs(logf - f_old)
                    f_threshold_std = opts.f_tol * max(1.0, abs(f_old))
                    if f_change_std <= f_threshold_std:
                        stop.reason = "f_tol"
                        stop.iters = it + 1
                        stop.grad_norm = norm(g)
                        stop.step_norm = norm(svec)
                        stop.f_change = f_change_std
                        stop.f_threshold = f_threshold_std
                        stop.accepted = True
                        break

                elif solver == "rcg":
                    if p_prev is None:
                        eta = g.copy()
                    else:
                        beta_pr = float(g @ (g - g_prev)) / max(float(g_prev @ g_prev), 1e-16)
                        beta_pr = max(beta_pr, 0.0)
                        eta = g + beta_pr * transport_tangent(p_prev, v_prev_saved, v, Q)
                        eta = tangent_proj(eta, v, Q)
                        if float(g @ eta) <= 1e-14:
                            eta = g.copy()

                    eta_norm = norm(eta)
                    if eta_norm <= opts.step_tol:
                        stop.reason = "step_tol"
                        stop.iters = it + 1
                        stop.grad_norm = gnorm
                        stop.step_norm = eta_norm
                        stop.accepted = False
                        break

                    v_prev_saved = v.copy()
                    g_prev = g.copy()
                    p_prev = eta.copy()

                    f_old = logf
                    v_old = v.copy()
                    accepted, v, logf, g, used_alpha, ls_steps, armijo_margin = armijo_ascent_step(M, v, Q, win, n, logf, g, eta, opts)
                    stop.line_search_alpha = used_alpha
                    stop.line_search_steps = ls_steps
                    if not accepted:
                        accepted, v, logf, g, used_alpha, ls_steps2, armijo_margin = armijo_ascent_step(M, v, Q, win, n, logf, g, g, opts)
                        stop.line_search_alpha = used_alpha
                        stop.line_search_steps += ls_steps2
                        if not accepted:
                            stop.reason = "rcg_line_search_failed"
                            stop.iters = it + 1
                            stop.grad_norm = gnorm
                            stop.step_norm = eta_norm
                            stop.accepted = False
                            break
                        p_prev = None
                        g_prev = None

                    prog.update(f_old, logf, v_old, v, armijo_margin)

                    stop_f, stop_step, f_change, f_threshold = progress_stop_check(
                        f_old, logf, prog.last_step_norm, opts.progress_f_tol, opts.progress_step_tol
                    )
                    if stop_f:
                        stop.reason = "progress_f_tol"
                        stop.iters = it + 1
                        stop.grad_norm = norm(g)
                        stop.step_norm = prog.last_step_norm
                        stop.f_change = f_change
                        stop.f_threshold = f_threshold
                        stop.accepted = True
                        break
                    if stop_step:
                        stop.reason = "progress_step_tol"
                        stop.iters = it + 1
                        stop.grad_norm = norm(g)
                        stop.step_norm = prog.last_step_norm
                        stop.f_change = f_change
                        stop.f_threshold = f_threshold
                        stop.accepted = True
                        break

                else:
                    raise ValueError(f"Unknown solver: {solver}")

        if stop.reason in ("unknown", "not_started"):
            stop.reason = "maxit"
            stop.iters = opts.maxit
            stop.grad_norm = norm(g)

        return v, logf, stop, prog


def entropy_iter_basis_manifold(M, r, win, n, V_init=None, opts=None, rng=None):
    if opts is None:
        opts = EntropyOptions()
    if rng is None:
        rng = np.random.default_rng(0)

    with timed(opts.time_stats, "entropy_iter_basis_manifold.total"):
        d = M.shape[1]
        V_out = np.zeros((d, r))
        s_out = np.zeros(r)
        H_out = -np.inf * np.ones(r)
        score_out = -np.inf * np.ones(r)

        Q = np.zeros((d, 0))

        for k in range(1, r + 1):
            with timed(opts.time_stats, f"entropy_iter_basis_manifold.vector_{k}"):
                starts = build_structured_starts(M, Q, V_init, k, opts, rng)
                best_logf = -np.inf
                best_v = None
                best_y2 = 0.0
                best_H = -np.inf

                for j, v0 in enumerate(starts):
                    with timed(opts.time_stats, "entropy_iter_basis_manifold.restart"):
                        v_loc, _, stop, prog = optimize_on_feasible_sphere(M, v0, Q, win, n, opts, rng)
                        logf_chk, _, y2_chk, H_chk = entropy_logscore_grad(M, v_loc, win, n, stats=opts.time_stats, mvstats=opts.matvec_stats)
                        if opts.verbose:
                            out = {"restart": j}
                            out.update(stop.as_dict())
                            out.update(prog.as_dict())
                            print(out)
                        if logf_chk > best_logf:
                            best_logf = logf_chk
                            best_v = v_loc
                            best_y2 = y2_chk
                            best_H = H_chk

                if best_v is None:
                    best_v = feasible_random(d, Q, rng)
                    best_logf, _, best_y2, best_H = entropy_logscore_grad(M, best_v, win, n, stats=opts.time_stats)

                Q = np.column_stack([Q, best_v])
                V_out[:, k - 1] = best_v
                s_out[k - 1] = best_y2
                H_out[k - 1] = best_H
                score_out[k - 1] = math.exp(best_logf)

        return V_out, s_out, H_out, score_out


# ============================================================
# Simpler multi-restart projected ascent ("basic" variant)
# ============================================================

def batched_retract_feasible(V: np.ndarray, Q: np.ndarray, eps: float = 1e-14):
    """
    Project columns of V to feasible space (orthogonal to Q) and normalize.
    Returns:
        Vn: normalized feasible columns where possible
        valid: boolean mask of columns with norm > eps
        norms: column norms after projection
    """
    if Q.size != 0:
        Vp = V - Q @ (Q.T @ V)
    else:
        Vp = V.copy()

    norms = np.linalg.norm(Vp, axis=0)
    valid = norms > eps

    Vn = Vp.copy()
    if np.any(valid):
        Vn[:, valid] /= norms[valid][None, :]
    return Vn, valid, norms


def entropy_logscore_grad_batched(
    M: np.ndarray,
    V: np.ndarray,
    win: int,
    n: int,
    mvstats: Optional[MatvecStats] = None,
):
    """
    Batched version of entropy_logscore_grad for V with columns as restart vectors.

    Inputs:
        M: shape (m, d)
        V: shape (d, R)

    Returns:
        logf: shape (R,)
        grad: shape (d, R)
        y2: shape (R,)
        H2: shape (R,)
    """
    c = target_c_from_win_n(win, n)

    Y = matvec(M, V, mvstats)             # (m, R)
    y2_sq = np.sum(Y * Y, axis=0)
    y2_sq = np.maximum(y2_sq, 1e-30)
    y2 = np.sqrt(y2_sq)

    y4_4 = np.sum(Y**4, axis=0)
    y4_4 = np.maximum(y4_4, 1e-30)
    y4 = y4_4 ** 0.25

    logf = (1.0 - c) * np.log(y2) + c * np.log(y4)

    g2 = rmatvec(M, Y, mvstats) / y2_sq[None, :]
    g4 = rmatvec(M, Y**3, mvstats) / y4_4[None, :]
    grad = (1.0 - c) * g2 + c * g4

    H2 = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    return logf, grad, y2, H2


def entropy_iter_basis_basic(
    M,
    r,
    win,
    n,
    V_init=None,
    num_restarts=8,
    maxit=200,
    tol=1e-8,
    progress_f_tol=1e-12,
    progress_step_tol=1e-10,
    rng=None,
    stats: Optional[TimeStats] = None,
    verbose: bool = True,
):
    with timed(stats, "entropy_iter_basis_basic.total"):
        if rng is None:
            rng = np.random.default_rng(0)

        d = M.shape[1]
        V_out = np.zeros((d, r))
        s_out = np.zeros(r)
        H_out = -np.inf * np.ones(r)
        score_out = -np.inf * np.ones(r)
        Q = np.zeros((d, 0))

        with timed(stats, "entropy_iter_basis_basic.svd"):
            _, _, Vh = svd(M, full_matrices=False)
        Vsvd = Vh.T
        num_top = min(4, Vsvd.shape[1])
        alpha_grid = [0.98, 0.9, 0.75, 0.5, 0.25, 0.0]

        for k in range(1, r + 1):
            with timed(stats, f"entropy_iter_basis_basic.vector_{k}"):
                # ------------------------------------------------------------
                # Build initial restart matrix V0 with columns as restarts
                # ------------------------------------------------------------
                V0 = np.zeros((d, num_restarts))
                stop_list = []
                prog_list = []

                for restart in range(num_restarts):
                    stop = StopDiagnostics(
                        reason="not_started",
                        grad_tol=tol,
                        step_tol=progress_step_tol,
                        solver="basic_batched",
                    )
                    prog = ProgressDiagnostics()

                    v_prev = None
                    if V_init is not None and V_init.shape[1] >= k:
                        v_prev = V_init[:, k - 1]

                    restart_type = (restart % 5) + 1
                    restart_block = restart // 5

                    if restart_type == 1:
                        if v_prev is not None:
                            xi = project_feasible(rng.standard_normal(d), Q)
                            nxi = norm(xi)
                            if nxi > 1e-14:
                                xi /= nxi
                            alpha = alpha_grid[restart_block % len(alpha_grid)]
                            v0 = alpha * v_prev + math.sqrt(max(0.0, 1.0 - alpha**2)) * xi
                        else:
                            v0 = Vsvd[:, 0]
                    elif restart_type == 2:
                        j = restart_block % num_top
                        v0 = Vsvd[:, j]
                    elif restart_type == 3:
                        j1 = restart_block % num_top
                        j2 = (restart_block + 1) % num_top
                        alpha = alpha_grid[restart_block % len(alpha_grid)]
                        v0 = alpha * Vsvd[:, j1] + math.sqrt(max(0.0, 1.0 - alpha**2)) * Vsvd[:, j2]
                    elif restart_type == 4:
                        j = restart_block % num_top
                        v0 = Vsvd[:, j] + 1e-2 * rng.standard_normal(d)
                    else:
                        v0 = rng.standard_normal(d)

                    v = retract_feasible(v0, Q)
                    if v is None:
                        v = feasible_random(d, Q, rng)

                    V0[:, restart] = v
                    stop_list.append(stop)
                    prog_list.append(prog)

                V = V0.copy()

                # initialize scores/progress
                with timed(stats, "entropy_iter_basis_basic.init_eval"):
                    logf, gradE, y2, H2 = entropy_logscore_grad_batched(M, V, win, n, mvstats=stats.matvec_stats if stats is not None and hasattr(stats, "matvec_stats") else None)

                for j in range(num_restarts):
                    prog_list[j].init_score(float(logf[j]))

                active = np.ones(num_restarts, dtype=bool)

                # ------------------------------------------------------------
                # Lockstep batched optimization over restarts
                # ------------------------------------------------------------
                for it in range(maxit):
                    if not np.any(active):
                        break

                    with timed(stats, "entropy_iter_basis_basic.iter"):
                        # Batched tangent projection
                        G = gradE.copy()

                        if Q.size != 0:
                            G = G - Q @ (Q.T @ G)

                        vg_dot = np.sum(V * G, axis=0)
                        G = G - V * vg_dot[None, :]

                        gnorms = np.linalg.norm(G, axis=0)

                        # Gradient-based stopping
                        grad_done = active & (gnorms <= tol)
                        for j in np.where(grad_done)[0]:
                            stop_list[j].reason = "grad_tol"
                            stop_list[j].iters = it + 1
                            stop_list[j].grad_norm = float(gnorms[j])
                            stop_list[j].accepted = True
                        active[grad_done] = False

                        if not np.any(active):
                            break

                        # Prepare masked backtracking search
                        search_mask = active.copy()
                        alpha = np.ones(num_restarts)
                        accepted = np.zeros(num_restarts, dtype=bool)

                        V_old = V.copy()
                        logf_old = logf.copy()

                        accepted_alpha = np.full(num_restarts, np.nan)
                        accepted_armijo_margin = np.full(num_restarts, np.nan)
                        ls_steps = np.zeros(num_restarts, dtype=int)

                        V_new = V.copy()
                        logf_new = logf.copy()
                        gradE_new = gradE.copy()
                        y2_new = y2.copy()
                        H2_new = H2.copy()

                        for ls_it in range(20):
                            if not np.any(search_mask):
                                break

                            ls_steps[search_mask] = ls_it + 1

                            V_trial = V.copy()
                            V_trial[:, search_mask] = (
                                V[:, search_mask] + G[:, search_mask] * alpha[search_mask][None, :]
                            )

                            V_trial, valid, _ = batched_retract_feasible(V_trial, Q)

                            # any invalid active columns keep shrinking
                            invalid_cols = np.zeros(num_restarts, dtype=bool)
                            invalid_cols[search_mask] = ~valid[search_mask]
                            search_mask[invalid_cols] = True
                            alpha[invalid_cols] *= 0.5

                            eval_mask = search_mask & valid
                            if not np.any(eval_mask):
                                continue

                            with timed(stats, "entropy_iter_basis_basic.line_search_eval"):
                                logf_trial, gradE_trial, y2_trial, H2_trial = entropy_logscore_grad_batched(
                                    M, V_trial[:, eval_mask], win, n,
                                    mvstats=stats.matvec_stats if stats is not None and hasattr(stats, "matvec_stats") else None
                                )

                            rhs = logf[eval_mask] + 1e-4 * alpha[eval_mask] * np.sum(G[:, eval_mask] * G[:, eval_mask], axis=0)
                            good = logf_trial >= rhs

                            idx_eval = np.where(eval_mask)[0]
                            idx_good = idx_eval[good]
                            idx_bad = idx_eval[~good]

                            if len(idx_good) > 0:
                                accepted[idx_good] = True
                                accepted_alpha[idx_good] = alpha[idx_good]
                                accepted_armijo_margin[idx_good] = logf_trial[good] - rhs[good]

                                V_new[:, idx_good] = V_trial[:, idx_good]
                                logf_new[idx_good] = logf_trial[good]
                                gradE_new[:, idx_good] = gradE_trial[:, good]
                                y2_new[idx_good] = y2_trial[good]
                                H2_new[idx_good] = H2_trial[good]

                            if len(idx_bad) > 0:
                                alpha[idx_bad] *= 0.5

                            search_mask[idx_good] = False

                        # handle line-search failures
                        failed = active & (~accepted)
                        for j in np.where(failed)[0]:
                            prog_list[j].no_update()
                            stop_list[j].reason = "line_search_failed"
                            stop_list[j].iters = it + 1
                            stop_list[j].grad_norm = float(gnorms[j])
                            stop_list[j].accepted = False
                            stop_list[j].line_search_steps = int(ls_steps[j])
                        active[failed] = False

                        # commit accepted updates and progress checks
                        good_updates = active & accepted
                        if np.any(good_updates):
                            for j in np.where(good_updates)[0]:
                                prog_list[j].update(
                                    float(logf_old[j]),
                                    float(logf_new[j]),
                                    V_old[:, j],
                                    V_new[:, j],
                                    float(accepted_armijo_margin[j]),
                                )

                                stop_list[j].line_search_alpha = float(accepted_alpha[j])
                                stop_list[j].line_search_steps = int(ls_steps[j])

                                stop_f, stop_step, f_change, f_threshold = progress_stop_check(
                                    float(logf_old[j]),
                                    float(logf_new[j]),
                                    float(prog_list[j].last_step_norm),
                                    progress_f_tol,
                                    progress_step_tol,
                                )

                                if stop_f:
                                    stop_list[j].reason = "progress_f_tol"
                                    stop_list[j].iters = it + 1
                                    stop_list[j].grad_norm = float(gnorms[j])
                                    stop_list[j].step_norm = float(prog_list[j].last_step_norm)
                                    stop_list[j].f_change = float(f_change)
                                    stop_list[j].f_threshold = float(f_threshold)
                                    stop_list[j].accepted = True
                                    active[j] = False
                                elif stop_step:
                                    stop_list[j].reason = "progress_step_tol"
                                    stop_list[j].iters = it + 1
                                    stop_list[j].grad_norm = float(gnorms[j])
                                    stop_list[j].step_norm = float(prog_list[j].last_step_norm)
                                    stop_list[j].f_change = float(f_change)
                                    stop_list[j].f_threshold = float(f_threshold)
                                    stop_list[j].accepted = True
                                    active[j] = False

                        # update batched state
                        V = V_new
                        logf = logf_new
                        gradE = gradE_new
                        y2 = y2_new
                        H2 = H2_new

                # mark remaining as maxit
                for j in range(num_restarts):
                    if stop_list[j].reason in ("unknown", "not_started"):
                        stop_list[j].reason = "maxit"
                        stop_list[j].iters = maxit
                        # recompute tangent grad norm from current batch state
                        gcol = gradE[:, j].copy()
                        if Q.size != 0:
                            gcol = gcol - Q @ (Q.T @ gcol)
                        gcol = gcol - V[:, j] * float(V[:, j] @ gcol)
                        stop_list[j].grad_norm = float(norm(gcol))
                        stop_list[j].accepted = False

                # ------------------------------------------------------------
                # pick best restart
                # ------------------------------------------------------------
                scores = np.exp(logf)
                best_j = int(np.argmax(scores))

                if verbose:
                    for j in range(num_restarts):
                        out_diag = {"restart": j}
                        out_diag.update(stop_list[j].as_dict())
                        out_diag.update(prog_list[j].as_dict())
                        out_diag["final_grad_norm"] = float(stop_list[j].grad_norm)
                        out_diag["grad_tol"] = float(tol)
                        out_diag["accepted_alpha"] = None if not np.isfinite(stop_list[j].line_search_alpha) else float(stop_list[j].line_search_alpha)
                        out_diag["final_logscore"] = float(logf[j])
                        print(out_diag)

                best_v = V[:, best_j].copy()
                best_s = float(y2[best_j])
                best_H = float(H2[best_j])
                best_score = float(scores[best_j])

                Q = np.column_stack([Q, best_v])
                V_out[:, k - 1] = best_v
                s_out[k - 1] = best_s
                H_out[k - 1] = best_H
                score_out[k - 1] = best_score

        return V_out, s_out, H_out, score_out


# ============================================================
# Continuation variant
# ============================================================

def projected_top_right_singular_vector_power(M, Q, maxit=50, tol=1e-10, rng=None, stats: Optional[TimeStats] = None):
    with timed(stats, "projected_top_right_singular_vector_power"):
        if rng is None:
            rng = np.random.default_rng(0)

        d = M.shape[1]
        v = feasible_random(d, Q, rng)
        for _ in range(maxit):
            v_new = M.T @ (M @ v)
            v_new = retract_feasible(v_new, Q)
            if v_new is None:
                break
            if norm(v_new - v) <= tol:
                v = v_new
                break
            v = v_new
        return v


def default_info():
    return {
        "score": -np.inf,
        "end_reason": "not_started",
        "last_gtan": np.inf,
        "cos_prev": 1.0,
        "subdivides": 0,
        "stop_diag": None,
        "prog_diag": None,
    }


def local_optimize_at_c(M, Q, v_seed, c_target, opts: ContinuationOptions):
    with timed(opts.time_stats, "local_optimize_at_c.total"):
        v = retract_feasible(v_seed, Q)
        if v is None:
            raise RuntimeError("Seed is infeasible.")

        info = default_info()
        prev_v = v.copy()

        stop = StopDiagnostics(
            reason="not_started",
            grad_tol=opts.local_tol,
            step_tol=opts.progress_step_tol,
            solver="continuation_local",
        )
        prog = ProgressDiagnostics()

        f0, _, _, _ = entropy_logscore_grad_c(M, v, c_target, stats=opts.time_stats)
        prog.init_score(f0)

        for it in range(opts.local_maxit):
            with timed(opts.time_stats, "local_optimize_at_c.iter"):
                logf, gradE, _, _ = entropy_logscore_grad_c(M, v, c_target, stats=opts.time_stats)
                g = tangent_proj(gradE, v, Q)
                gnorm = norm(g)

                info["score"] = math.exp(logf)
                info["last_gtan"] = gnorm

                if gnorm <= opts.local_tol:
                    info["end_reason"] = "small_tangent_grad"
                    stop.reason = "grad_tol"
                    stop.iters = it + 1
                    stop.grad_norm = gnorm
                    stop.accepted = True
                    info["stop_diag"] = stop
                    info["prog_diag"] = prog
                    return v, info

                alpha = 1.0
                accepted = False
                used_alpha = np.nan
                ls_steps = 0
                for j in range(opts.local_ls_max):
                    ls_steps = j + 1
                    vt = retract_feasible(v + alpha * g, Q)
                    if vt is not None:
                        ft, _, _, _ = entropy_logscore_grad_c(M, vt, c_target, stats=opts.time_stats)
                        rhs = logf + opts.accept_armijo * alpha * float(g @ g)
                        if ft >= rhs:
                            prev_v = v.copy()
                            v = vt
                            accepted = True
                            used_alpha = alpha
                            prog.update(logf, ft, prev_v, v, ft - rhs)
                            break
                    alpha *= 0.5

                stop.line_search_alpha = used_alpha
                stop.line_search_steps = ls_steps

                if not accepted:
                    prog.no_update()
                    info["end_reason"] = "line_search_failed"
                    info["cos_prev"] = abs(float(prev_v @ v))
                    stop.reason = "line_search_failed"
                    stop.iters = it + 1
                    stop.grad_norm = gnorm
                    stop.accepted = False
                    info["stop_diag"] = stop
                    info["prog_diag"] = prog
                    return v, info

                stop_f, stop_step, f_change, f_threshold = progress_stop_check(
                    logf, prog.last_score, prog.last_step_norm, opts.progress_f_tol, opts.progress_step_tol
                )

                if stop_f:
                    info["end_reason"] = "progress_f_tol"
                    info["cos_prev"] = abs(float(prev_v @ v))
                    stop.reason = "progress_f_tol"
                    stop.iters = it + 1
                    stop.grad_norm = gnorm
                    stop.step_norm = prog.last_step_norm
                    stop.f_change = f_change
                    stop.f_threshold = f_threshold
                    stop.accepted = True
                    info["stop_diag"] = stop
                    info["prog_diag"] = prog
                    return v, info

                if stop_step:
                    info["end_reason"] = "progress_step_tol"
                    info["cos_prev"] = abs(float(prev_v @ v))
                    stop.reason = "progress_step_tol"
                    stop.iters = it + 1
                    stop.grad_norm = gnorm
                    stop.step_norm = prog.last_step_norm
                    stop.f_change = f_change
                    stop.f_threshold = f_threshold
                    stop.accepted = True
                    info["stop_diag"] = stop
                    info["prog_diag"] = prog
                    return v, info

        info["end_reason"] = "maxit"
        info["cos_prev"] = abs(float(prev_v @ v))
        stop.reason = "maxit"
        stop.iters = opts.local_maxit
        stop.grad_norm = info["last_gtan"]
        stop.accepted = False
        info["stop_diag"] = stop
        info["prog_diag"] = prog
        return v, info


def make_hedged_seeds(v_center, Q, opts: ContinuationOptions, rng):
    with timed(opts.time_stats, "make_hedged_seeds"):
        d = len(v_center)
        seeds = [retract_feasible(v_center, Q)]
        for j in range(opts.num_hedge):
            z = tangent_proj(rng.standard_normal(d), v_center, Q)
            nz = norm(z)
            if nz <= 1e-14:
                continue
            z /= nz
            epsj = opts.perturb_scale_small if (j % 2 == 0) else opts.perturb_scale_large
            vt = retract_feasible(v_center + epsj * z, Q)
            if vt is not None:
                seeds.append(vt)
        return seeds


def stage_failure_flag(info, dc, opts: ContinuationOptions):
    if info["end_reason"] == "line_search_failed":
        return True
    if info["last_gtan"] > opts.fail_gtan and dc > opts.min_dc:
        return True
    if info["cos_prev"] < opts.fail_cos and dc > opts.min_dc:
        return True
    return False


def continuation_stage_once(M, Q, v_prev, c_target, opts: ContinuationOptions, rng):
    with timed(opts.time_stats, "continuation_stage_once"):
        seeds = make_hedged_seeds(v_prev, Q, opts, rng)

        best_score = -np.inf
        v_best = None
        info_best = default_info()

        for idx, v_seed in enumerate(seeds):
            if v_seed is None:
                continue
            v_loc, info_loc = local_optimize_at_c(M, Q, v_seed, c_target, opts)
            if opts.verbose and info_loc.get("stop_diag", None) is not None:
                out = {"local_seed": idx}
                out.update(info_loc["stop_diag"].as_dict())
                if info_loc.get("prog_diag") is not None:
                    out.update(info_loc["prog_diag"].as_dict())
                print(out)
            if info_loc["score"] > best_score:
                best_score = info_loc["score"]
                v_best = v_loc
                info_best = info_loc

        if v_best is None:
            v_best = v_prev
            info_best["score"] = entropy_score_fast_c(M, v_best, c_target, stats=opts.time_stats)
            info_best["end_reason"] = "fallback_prev"

        info_best["cos_prev"] = abs(float(v_prev @ v_best))
        return v_best, info_best


def continuation_stage_recursive(M, Q, v_prev, c_prev, c_next, opts: ContinuationOptions, depth: int, rng):
    with timed(opts.time_stats, "continuation_stage_recursive"):
        v_try, info_try = continuation_stage_once(M, Q, v_prev, c_next, opts, rng)
        dc = abs(c_next - c_prev)
        fail_flag = stage_failure_flag(info_try, dc, opts)

        if (not fail_flag) or (depth >= opts.max_subdivide) or (dc <= opts.min_dc):
            info_try["subdivides"] = depth
            return v_try, info_try

        c_mid = 0.5 * (c_prev + c_next)

        if opts.verbose:
            print(
                f"  stage failure detected on [{c_prev:.4e}, {c_next:.4e}], "
                f"inserting midpoint {c_mid:.4e} "
                f"(reason={info_try['end_reason']}, "
                f"last_gtan={info_try['last_gtan']:.3e}, "
                f"cos_prev={info_try['cos_prev']:.3e})"
            )

        v_mid, info1 = continuation_stage_recursive(M, Q, v_prev, c_prev, c_mid, opts, depth + 1, rng)
        v_out, info2 = continuation_stage_recursive(M, Q, v_mid, c_mid, c_next, opts, depth + 1, rng)
        info2["subdivides"] = max(info1["subdivides"], info2["subdivides"])
        return v_out, info2


def entropy_continuation_basis(M, r, win, n, V_init=None, opts=None, rng=None):
    if opts is None:
        opts = ContinuationOptions()
    if rng is None:
        rng = np.random.default_rng(0)

    with timed(opts.time_stats, "entropy_continuation_basis.total"):
        d = M.shape[1]
        V_out = np.zeros((d, r))
        s_out = np.zeros(r)
        H_out = -np.inf * np.ones(r)
        score_out = -np.inf * np.ones(r)

        Q = np.zeros((d, 0))
        c_target = target_c_from_win_n(win, n)
        c_grid = continuation_grid(c_target, opts.num_stages)

        for k in range(1, r + 1):
            with timed(opts.time_stats, f"entropy_continuation_basis.vector_{k}"):
                if opts.verbose:
                    print(f"\n==== extracting vector k={k}/{r} ====")

                v_spec = projected_top_right_singular_vector_power(M, Q, 50, 1e-10, rng=rng, stats=opts.time_stats)
                v_spec = retract_feasible(v_spec, Q)

                v0_candidates = [v_spec] if v_spec is not None else []

                if V_init is not None and V_init.shape[1] >= k:
                    v_init = retract_feasible(V_init[:, k - 1], Q)
                    if v_init is not None:
                        v0_candidates.append(v_init)

                best0 = None
                best0_score = -np.inf
                for vj in v0_candidates:
                    s0 = entropy_score_fast_c(M, vj, 0.0, stats=opts.time_stats)
                    if s0 > best0_score:
                        best0_score = s0
                        best0 = vj

                if best0 is None:
                    best0 = feasible_random(d, Q, rng)

                v_cur = best0
                c_cur = 0.0

                if opts.verbose:
                    print(f"k={k} spectral-start score(c=0)={entropy_score_fast_c(M, v_cur, 0.0, stats=opts.time_stats, mvstats=opts.matvec_stats):.12e}")

                for t in range(1, len(c_grid)):
                    with timed(opts.time_stats, "entropy_continuation_basis.stage"):
                        c_next = float(c_grid[t])
                        v_new, info_track = continuation_stage_recursive(M, Q, v_cur, c_cur, c_next, opts, 0, rng)
                        v_cur = v_new
                        c_cur = c_next

                        if opts.verbose:
                            print(
                                f"k={k} continued to c={c_cur:.12e} "
                                f"final_score={info_track['score']:.12e} "
                                f"end={info_track['end_reason']} "
                                f"subdivides={info_track['subdivides']} "
                                f"last_gtan={info_track['last_gtan']:.3e} "
                                f"cos_prev={info_track['cos_prev']:.3e}"
                            )

                logf, _, y2_fin, H_fin = entropy_logscore_grad_c(M, v_cur, c_target, stats=opts.time_stats)

                Q = np.column_stack([Q, v_cur])
                V_out[:, k - 1] = v_cur
                s_out[k - 1] = y2_fin
                H_out[k - 1] = H_fin
                score_out[k - 1] = math.exp(logf)

        return V_out, s_out, H_out, score_out


# ============================================================
# Synthetic experiment
# ============================================================

def build_ground_truth(n=1024, r_sig=1, V_type="id", sigma1=0.991, alpha_sig=0.003, alpha_tail=0.0145, tail_scale=0.99):
    if n % 2 != 0:
        raise ValueError("Hadamard construction requires n to be even.")

    k = n
    U0 = np.zeros((n, n), dtype=float)

    H = hadamard(n).astype(float)
    U0[:, :r_sig] = H[:, :r_sig] / math.sqrt(n)

    a_tail = math.sqrt(1.0 - r_sig / n)
    b_tail = 1.0 / math.sqrt(n)
    for j in range(r_sig, n):
        col = np.zeros(n, dtype=float)
        idx_large = j - r_sig
        if idx_large <= n - r_sig - 1:
            col[idx_large] = a_tail
        else:
            raise ValueError("Tail index out of range.")
        col[n - r_sig:] = b_tail
        U0[:, j] = col

    Qtmp, _ = qr(U0, mode="economic")
    for j in range(r_sig):
        if float(Qtmp[:, j] @ U0[:, j]) < 0:
            Qtmp[:, j] *= -1.0
    U = Qtmp[:, :k]

    if V_type == "id":
        V = np.eye(n, k)
    elif V_type == "U":
        V = U.copy()
    elif V_type == "rand":
        Vrand, _ = qr(np.random.default_rng(0).standard_normal((n, k)), mode="economic")
        V = Vrand
    else:
        raise ValueError('Unknown V_type. Use "id", "U", or "rand".')

    sig_block = sigma1 * (np.arange(1, r_sig + 1, dtype=float) ** (-alpha_sig))
    tail_block = tail_scale * (np.arange(1, (k - r_sig) + 1, dtype=float) ** (-alpha_tail))
    svec = np.concatenate([sig_block, tail_block])
    svec[0] = sigma1
    S = np.diag(svec)

    return U, S, V, svec


def run_streaming_experiment(
    n=1024,
    r=1,
    l=1,
    win=1000,
    mode="EntropyScore",
    optimizer="manifold",
    V_type="id",
    r_sig=1,
    sigma_vals=(0.991,),
    num_exper=1,
    cont_opts=None,
    entropy_opts=None,
    seed=0,
    matvec_trials=50,
    baseline_iters=50,
):
    global_stats = TimeStats()
    global_stats.matvec_stats = MatvecStats()

    with timed(global_stats, "run_streaming_experiment.total"):
        rng = np.random.default_rng(seed)

        if cont_opts is None:
            cont_opts = ContinuationOptions()
        if entropy_opts is None:
            entropy_opts = EntropyOptions()

        cont_opts.time_stats = global_stats
        entropy_opts.time_stats = global_stats
        cont_opts.matvec_stats = global_stats.matvec_stats
        entropy_opts.matvec_stats = global_stats.matvec_stats

        num_svals = len(sigma_vals)
        alignment_results = np.zeros((num_svals, num_exper))
        relerr_sval_results = np.zeros((num_svals, num_exper))
        Delta_results = np.zeros((num_svals, num_exper))
        DeltaComp_results = np.zeros((num_svals, num_exper))
        low_sval_indicator = np.zeros((num_svals, num_exper))
        baseline_times = np.zeros((num_svals, num_exper))
        baseline_sigma = np.zeros((num_svals, num_exper))
        predicted_matvec_times = np.zeros((num_svals, num_exper))
        raw_pair_means = np.zeros((num_svals, num_exper))

        for i, sigma1 in enumerate(sigma_vals):
            with timed(global_stats, "run_streaming_experiment.per_sigma"):
                U, S, V, svec = build_ground_truth(n=n, r_sig=r_sig, V_type=V_type, sigma1=sigma1)

                k = n
                E_opt = float(np.sum(svec[r:]**2)) if r < k else 0.0
                Delta_comp = float(np.sum(svec[:r]**2) - np.sum(svec[r:2*r]**2))
                DeltaComp_results[i, :] = Delta_comp

                for e in range(num_exper):
                    with timed(global_stats, "run_streaming_experiment.per_experiment"):
                        p = rng.permutation(n)
                        A = U @ S @ V.T
                        A = A[p, :]

                        mA = A.shape[0]
                        S_r = None
                        V_r = None

                        for start_row in range(0, mA, win):
                            with timed(global_stats, "run_streaming_experiment.per_block"):
                                block_t0 = time.perf_counter()

                                end_row = min(start_row + win, mA)
                                A_block = A[start_row:end_row, :]

                                if V_r is None:
                                    M = A_block
                                else:
                                    B_top = S_r @ V_r.T
                                    M = np.vstack([B_top, A_block])

                                raw_pair_info = avg_matvec_pair_time(M, trials=matvec_trials, seed=seed + e)
                                baseline_info = plain_power_iteration(M, iters=baseline_iters, seed=seed + e)

                                if mode == "EntropyScore":
                                    rr = min(r, M.shape[1])
                                    V_init = None if V_r is None else V_r

                                    if optimizer == "basic":
                                        V_new, s_new, H_new, score_new = entropy_iter_basis_basic(
                                            M, rr, win, n, V_init=V_init,
                                            num_restarts=8, maxit=200, tol=1e-8,
                                            progress_f_tol=1e-12, progress_step_tol=1e-10,
                                            rng=rng, stats=global_stats, verbose=True
                                        )
                                    elif optimizer == "continuation":
                                        V_new, s_new, H_new, score_new = entropy_continuation_basis(
                                            M, rr, win, n, V_init=V_init, opts=cont_opts, rng=rng
                                        )
                                    elif optimizer == "manifold":
                                        V_new, s_new, H_new, score_new = entropy_iter_basis_manifold(
                                            M, rr, win, n, V_init=V_init, opts=entropy_opts, rng=rng
                                        )
                                    else:
                                        raise ValueError(f"Unknown optimizer: {optimizer}")

                                    S_r = np.diag(s_new)
                                    V_r = V_new

                                    with timed(global_stats, "run_streaming_experiment.window_svd"):
                                        with threadpool_limits(limits=8, user_api="blas"):
                                            _, _, Vh = svd(M, full_matrices=False)
                                    v_ = Vh.T
                                    e1_proj = v_ @ (v_.T @ V[:, 0])
                                    e1_proj = e1_proj / safe_norm(e1_proj)
                                    score_e1_proj = entropy_score_fast(M, e1_proj, win, n, stats=global_stats, mvstats=global_stats.matvec_stats)

                                    print(f"rows {start_row + 1}:{end_row}")
                                    print("score of v1 projection onto window space:", score_e1_proj)
                                    print("s:", s_new)
                                    print("H:", H_new)
                                    print("scores:", score_new)
                                    print("score_e1_proj:", score_e1_proj)
                                    print(f"V(1,1)={V_new[0,0]:.5f}")
                                    print(f"should be: {e1_proj[0]:.5f}")

                                    block_dt = time.perf_counter() - block_t0
                                    optimizer_pairs = global_stats.matvec_stats.matvec_pairs
                                    predicted_time_from_matvecs = optimizer_pairs * raw_pair_info["mean"]
                                    baseline_times[i, e] = baseline_info["wall_time"]
                                    baseline_sigma[i, e] = baseline_info["sigma_est"]
                                    predicted_matvec_times[i, e] = predicted_time_from_matvecs
                                    raw_pair_means[i, e] = raw_pair_info["mean"]

                                    print(f"block optimizer wall time: {block_dt:.6f}s")
                                    print("raw_matvec_pair_time:", raw_pair_info)
                                    print("plain_power_iteration:", {
                                        "iters": baseline_info["iters"],
                                        "sigma_est": baseline_info["sigma_est"],
                                        "wall_time": baseline_info["wall_time"],
                                        **baseline_info["matvec_stats"].as_dict(),
                                    })
                                    print("optimizer_matvec_counts:", global_stats.matvec_stats.as_dict())
                                    print("predicted_time_from_matvecs:", predicted_time_from_matvecs)
                                    print("\nTiming diagnostics so far:")
                                    print(global_stats.report(sort_by="time"))
                                    break

                                elif mode in ("iSVD", "FD"):
                                    with timed(global_stats, "run_streaming_experiment.isvd_fd_svd"):
                                        with threadpool_limits(limits=8, user_api="blas"):
                                            _, S_hat, V_hat = svd(M, full_matrices=False)
                                    s = np.diag(S_hat) if S_hat.ndim == 2 else S_hat
                                    rr = min(r, len(s))

                                    if mode == "iSVD":
                                        S_r = np.diag(s[:rr])
                                        V_r = V_hat[:, :rr]
                                    else:
                                        delta = s[rr]**2 if len(s) > rr else 0.0
                                        s1 = s[:rr]
                                        s1_shr = np.sqrt(np.maximum(s1**2 - delta, 0.0))
                                        S_r = np.diag(s1_shr)
                                        V_r = V_hat[:, :rr]
                                else:
                                    raise ValueError(f"Unknown mode: {mode}")

                        ll = min(l, V_r.shape[1])
                        align = norm((np.eye(V_r.shape[0]) - V_r @ V_r.T) @ V[:, :ll], ord="fro") / math.sqrt(ll)
                        top_sval_est = 0.0 if S_r is None else float(S_r[0, 0])
                        rel_err_sval = abs(top_sval_est - sigma1) / sigma1
                        E_alg = norm(A, ord="fro")**2 if V_r is None else norm(A - A @ V_r @ V_r.T, ord="fro")**2
                        Delta = E_alg - E_opt

                        alignment_results[i, e] = align
                        relerr_sval_results[i, e] = rel_err_sval
                        Delta_results[i, e] = Delta
                        low_sval_indicator[i, e] = float(top_sval_est <= 0.99)

        return {
            "sigma1": np.array(sigma_vals, dtype=float),
            "mean_align": np.mean(alignment_results, axis=1),
            "std_align": np.std(alignment_results, axis=1, ddof=0),
            "mean_relerr_sval": np.mean(relerr_sval_results, axis=1),
            "std_relerr_sval": np.std(relerr_sval_results, axis=1, ddof=0),
            "low_sval_count": np.sum(low_sval_indicator, axis=1),
            "alignment_results": alignment_results,
            "relerr_sval_results": relerr_sval_results,
            "Delta_results": Delta_results,
            "DeltaComp_results": DeltaComp_results,
            "baseline_times": baseline_times,
            "baseline_sigma": baseline_sigma,
            "predicted_matvec_times": predicted_matvec_times,
            "raw_pair_means": raw_pair_means,
            "time_stats": global_stats,
            "matvec_stats": global_stats.matvec_stats,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True,
                        choices=["trust_region", "deform", "structured_restart"])

    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--win", type=int, default=1000)
    parser.add_argument("--sigma1", type=float, default=0.991)
    parser.add_argument("--matvec-trials", type=int, default=50)
    parser.add_argument("--baseline-iters", type=int, default=50)

    args = parser.parse_args()

    if args.mode == "trust_region":
        optimizer = "manifold"
        entropy_opts = EntropyOptions(
            solver="trust_region",
            verbose=True,
            progress_f_tol=1e-12,
            progress_step_tol=1e-10,
        )
        cont_opts = ContinuationOptions()

    elif args.mode == "deform":
        optimizer = "continuation"
        entropy_opts = EntropyOptions()
        cont_opts = ContinuationOptions(
            progress_f_tol=1e-12,
            progress_step_tol=1e-10,
            verbose=True,
        )

    elif args.mode == "structured_restart":
        optimizer = "basic"
        entropy_opts = EntropyOptions()
        cont_opts = ContinuationOptions()

    else:
        raise ValueError("Unknown mode")

    out = run_streaming_experiment(
        n=args.n,
        r=args.r,
        l=1,
        win=args.win,
        mode="EntropyScore",
        optimizer=optimizer,
        V_type="id",
        r_sig=1,
        sigma_vals=(args.sigma1,),
        num_exper=1,
        cont_opts=cont_opts,
        entropy_opts=entropy_opts,
        seed=0,
        matvec_trials=args.matvec_trials,
        baseline_iters=args.baseline_iters,
    )

    print("\nSummary:")
    for i, sigma1 in enumerate(out["sigma1"]):
        print(
            {
                "sigma1": float(sigma1),
                "mean_align": float(out["mean_align"][i]),
                "std_align": float(out["std_align"][i]),
                "mean_relerr_sval": float(out["mean_relerr_sval"][i]),
                "std_relerr_sval": float(out["std_relerr_sval"][i]),
                "low_sval_count": int(out["low_sval_count"][i]),
                "raw_pair_mean": float(out["raw_pair_means"][i, 0]),
                "baseline_time": float(out["baseline_times"][i, 0]),
                "baseline_sigma": float(out["baseline_sigma"][i, 0]),
                "predicted_matvec_time": float(out["predicted_matvec_times"][i, 0]),
            }
        )

    print("\nFinal timing diagnostics:")
    print(out["time_stats"].report(sort_by="time"))
