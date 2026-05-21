import requests
import io
import tarfile
import os
import sys
import importlib.util
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import requests
import re
import time
import gc
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

try:
    from threadpoolctl import threadpool_info
except Exception:
    threadpool_info = None

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from scipy.io import mmread
import scipy.sparse as sparse
from scipy.optimize import linear_sum_assignment
# from scipy.linalg import orthogonal_procrustes, subspace_angles, matrix_balance
from scipy.sparse._csr import csr_matrix

from scipy import stats

import matplotlib
import traceback
matplotlib.use('Agg')

import json
import primme


@dataclass
class TimeStats:
    totals: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    def add(self, key: str, dt: float):
        self.totals[key] = self.totals.get(key, 0.0) + dt
        self.counts[key] = self.counts.get(key, 0) + 1

    def snapshot(self):
        return dict(self.totals), dict(self.counts)

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


def stats_delta_summary(stats: Optional[TimeStats], totals_before, counts_before, keys):
    if stats is None:
        return {}
    out = {}
    for key in keys:
        total = stats.totals.get(key, 0.0) - totals_before.get(key, 0.0)
        count = stats.counts.get(key, 0) - counts_before.get(key, 0)
        if count > 0 or total > 0.0:
            out[key] = {
                "time": total,
                "count": count,
                "avg": total / count if count > 0 else None,
            }
    return out


@dataclass
class MatvecStats:
    matvec_calls: int = 0
    rmatvec_calls: int = 0
    matvec_rhs: int = 0
    rmatvec_rhs: int = 0
    matmat_calls: int = 0
    rmatmat_calls: int = 0
    matvec_time: float = 0.0
    rmatvec_time: float = 0.0
    matvec_rhs_sizes: Dict[int, int] = field(default_factory=dict)
    rmatvec_rhs_sizes: Dict[int, int] = field(default_factory=dict)

    def add_matvec(self, dt: float, cols: int = 1, via: str = "matvec"):
        cols = int(cols)
        self.matvec_calls += 1
        self.matvec_rhs += cols
        self.matvec_time += float(dt)
        self.matvec_rhs_sizes[cols] = self.matvec_rhs_sizes.get(cols, 0) + 1
        if via == "matmat":
            self.matmat_calls += 1

    def add_rmatvec(self, dt: float, cols: int = 1, via: str = "rmatvec"):
        cols = int(cols)
        self.rmatvec_calls += 1
        self.rmatvec_rhs += cols
        self.rmatvec_time += float(dt)
        self.rmatvec_rhs_sizes[cols] = self.rmatvec_rhs_sizes.get(cols, 0) + 1
        if via == "rmatmat":
            self.rmatmat_calls += 1

    def as_dict(self) -> Dict[str, float]:
        return {
            "MATVEC_CALLS": self.matvec_calls,
            "RMATVEC_CALLS": self.rmatvec_calls,
            "MATVEC_RHS": self.matvec_rhs,
            "RMATVEC_RHS": self.rmatvec_rhs,
            "MATMAT_CALLS": self.matmat_calls,
            "RMATMAT_CALLS": self.rmatmat_calls,
            "matvec_time": self.matvec_time,
            "rmatvec_time": self.rmatvec_time,
            "matvec_total_time": self.matvec_time + self.rmatvec_time,
            "matvec_rhs_sizes": dict(sorted(self.matvec_rhs_sizes.items())),
            "rmatvec_rhs_sizes": dict(sorted(self.rmatvec_rhs_sizes.items())),
        }


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
    accepted: bool = False
    line_search_alpha: float = np.nan
    line_search_steps: int = 0
    solver: str = ""
    note: str = ""

    def as_dict(self):
        out = {
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
    min_f_change: float = np.inf
    max_f_change: float = -np.inf
    min_step_norm: float = np.inf
    max_step_norm: float = -np.inf
    num_updates: int = 0

    def init_score(self, f0: float):
        self.first_score = f0
        self.last_score = f0
        self.best_score = max(self.best_score, f0)

    def update(self, f_old, f_new, v_old, v_new):
        step_norm = np.linalg.norm(v_new - v_old)
        f_change = f_new - f_old
        if not np.isfinite(self.first_score):
            self.first_score = f_old
        self.last_score = f_new
        self.best_score = max(self.best_score, f_new)
        self.last_f_change = f_change
        self.last_step_norm = step_norm
        self.min_f_change = min(self.min_f_change, f_change)
        self.max_f_change = max(self.max_f_change, f_change)
        self.min_step_norm = min(self.min_step_norm, step_norm)
        self.max_step_norm = max(self.max_step_norm, step_norm)
        self.num_updates += 1

    def no_update(self):
        self.last_f_change = 0.0
        self.last_step_norm = 0.0

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
            "min_f_change": self.min_f_change,
            "max_f_change": self.max_f_change,
            "min_step_norm": self.min_step_norm,
            "max_step_norm": self.max_step_norm,
            "num_updates": self.num_updates,
        }


def _num_rhs(x: np.ndarray) -> int:
    return 1 if np.asarray(x).ndim == 1 else int(np.asarray(x).shape[1])


def tracked_matvec(M: np.ndarray, v: np.ndarray, mvstats: Optional[MatvecStats] = None) -> np.ndarray:
    t0 = time.perf_counter()
    out = M @ v
    if mvstats is not None:
        mvstats.add_matvec(time.perf_counter() - t0, cols=_num_rhs(v), via="matvec")
    return out


def tracked_rmatvec(M: np.ndarray, u: np.ndarray, mvstats: Optional[MatvecStats] = None) -> np.ndarray:
    t0 = time.perf_counter()
    out = M.T @ u
    if mvstats is not None:
        mvstats.add_rmatvec(time.perf_counter() - t0, cols=_num_rhs(u), via="rmatvec")
    return out


def make_tracked_linear_operator(M: np.ndarray, mvstats: Optional[MatvecStats] = None, dtype=None):
    M_arr = np.asarray(M)
    op_dtype = M_arr.dtype if dtype is None else dtype

    def matvec(v):
        return tracked_matvec(M_arr, v, mvstats)

    def rmatvec(v):
        return tracked_rmatvec(M_arr, v, mvstats)

    def matmat(V):
        t0 = time.perf_counter()
        out = M_arr @ V
        if mvstats is not None:
            mvstats.add_matvec(time.perf_counter() - t0, cols=_num_rhs(V), via="matmat")
        return out

    def rmatmat(V):
        t0 = time.perf_counter()
        out = M_arr.T @ V
        if mvstats is not None:
            mvstats.add_rmatvec(time.perf_counter() - t0, cols=_num_rhs(V), via="rmatmat")
        return out

    return sp.sparse.linalg.LinearOperator(
        shape=M_arr.shape,
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
        dtype=op_dtype,
    )


def dense_mv_microbench(A, num_trials=10, seed=0):
    A_arr = np.asarray(A)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A_arr.shape[1]).astype(A_arr.dtype, copy=False)
    u = rng.standard_normal(A_arr.shape[0]).astype(A_arr.dtype, copy=False)

    _ = A_arr @ v
    _ = A_arr.T @ u

    t0 = time.perf_counter()
    for _ in range(num_trials):
        _ = A_arr @ v
    forward_total = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(num_trials):
        _ = A_arr.T @ u
    reverse_total = time.perf_counter() - t0

    return {
        "trials": num_trials,
        "avg_forward_time": forward_total / num_trials if num_trials > 0 else None,
        "avg_reverse_time": reverse_total / num_trials if num_trials > 0 else None,
    }


def runtime_state_snapshot():
    return {
        "threadpool_info": threadpool_info() if threadpool_info is not None else "threadpoolctl unavailable",
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "pid": os.getpid(),
    }


def arr_info(name, x):
    arr = np.asarray(x)
    return {
        "name": name,
        "shape": arr.shape,
        "ndim": arr.ndim,
        "dtype": str(arr.dtype),
        "strides": arr.strides,
        "c_contiguous": bool(arr.flags.c_contiguous),
        "f_contiguous": bool(arr.flags.f_contiguous),
        "owndata": bool(arr.flags.owndata),
    }


_entropy_exact_operand_debug_printed = False


def matrix_work_dtype(M):
    return np.asarray(M).dtype

def save_txt(filename, **kwargs):
    data = {}

    for key, value in kwargs.items():
        if isinstance(value, np.ndarray):
            data[key] = {
                "type": "ndarray",
                "value": value.tolist()
            }
        elif isinstance(value, (np.generic,)):  # catches np.float32, np.int64, etc.
            data[key] = {
                "type": "scalar",
                "value": value.item()
            }
        else:
            data[key] = {
                "type": "scalar",
                "value": value
            }

    with open(filename, "w") as f:
        json.dump(data, f)


def load_txt(filename):
    with open(filename, "r") as f:
        raw = json.load(f)

    out = {}
    for key, obj in raw.items():
        if obj["type"] == "ndarray":
            out[key] = np.array(obj["value"])
        else:
            out[key] = obj["value"]

    return out


def sample_4d_hyperboloid(num_points, a, b, c, d):
    np.random.seed(420)
    # Generate random values for theta, phi, and psi
    theta = np.random.uniform(-np.pi/2, np.pi/2, num_points)
    phi = np.random.uniform(0, 2*np.pi, num_points)
    psi = np.random.normal(0, 1, num_points)  # Using normal distribution for psi

    # Calculate the coordinates
    x = a * np.cos(theta) * np.cos(phi)
    y = b * np.cos(theta) * np.sin(phi)
    z = c * np.sin(theta)
    w = d * np.sinh(psi)

    return np.column_stack((x, y, z, w))

def sample_swiss_roll(num_points, noise=0.1):
    np.random.seed(42)
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(num_points))
    height = 2 * np.random.rand(num_points)
    x = t * np.cos(t)
    y = height
    z = t * np.sin(t)
    return np.column_stack((x, y, z))
#     w = noise * np.random.randn(num_points)
#     return np.column_stack((x, y, z, w))

def sample_torus(num_points, R=3, r=1):
    np.random.seed(42)
    theta = np.random.uniform(0, 2*np.pi, num_points)
    phi = np.random.uniform(0, 2*np.pi, num_points)
    x = (R + r*np.cos(phi)) * np.cos(theta)
    y = (R + r*np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return np.column_stack((x, y, z))
#     w = 0.1 * np.random.randn(num_points)
#     return np.column_stack((x, y, z, w))

def sample_gaussian_mixture(num_points_per_cluster=300, std_dev=1.0):
    np.random.seed(42)
    centers = np.array([
        [2, 2, 2],
        [-2, -2, -2],
        [2, -2, 2]
    ])
    total_points = num_points_per_cluster * len(centers)
    points = np.zeros((total_points, 3))
    for i in range(len(centers)):
        start_idx = i * num_points_per_cluster
        end_idx = (i + 1) * num_points_per_cluster
        cluster_points = std_dev * np.random.randn(num_points_per_cluster, 3) + centers[i]
        points[start_idx:end_idx] = cluster_points
    return points


class StreamingMatrix:
    def __init__(self, n, singular_values):
        if n & (n - 1) != 0:
            raise ValueError("n must be a power of 2")
        if len(singular_values) != n:
            raise ValueError("The number of singular values must match n")
        
        self.n = n
        self.S = np.diag(singular_values)
        self.V = None
        self.VSV_T = None

    def generate_unitary_matrix(self):
        # Initialize the new unitary matrix
        V = np.zeros((self.n, self.n), dtype=complex)
        
        # Set the first column
        V[:, 0] = 1 / np.sqrt(self.n)
        
        # Generate the remaining columns
        for j in range(1, self.n):
            v = np.random.random(self.n) * 0.1
            v[j-1] = 1
            v[j] = -1
            
            # Orthogonalize against previous columns
            for i in range(j):
                v = v - np.dot(np.conj(V[:, i]), v) * V[:, i]
            
            # Normalize
            V[:, j] = v / np.linalg.norm(v)
        
        return V

    def generate_matrix(self):
        if self.V is None:
            self.V = self.generate_unitary_matrix()
        
        # Compute VSV^T
        VS = np.dot(self.V, self.S)
        self.VSV_T = np.dot(VS, self.V.conj().T)

    def __getitem__(self, key):
        if self.VSV_T is None:
            self.generate_matrix()
        
        if isinstance(key, tuple):
            row_idx, col_idx = key
        else:
            row_idx, col_idx = key, slice(None)
        
        return self.VSV_T[row_idx, col_idx]

    def __len__(self):
        return self.n

    @property
    def shape(self):
        return (self.n, self.n)

    def get_singular_values(self):
        return np.diag(self.S)

    def get_unitary_matrix(self):
        if self.V is None:
            self.V = self.generate_unitary_matrix()
        return self.V

# class StreamingRBFKernel:
#     def __init__(self, points, lengthscale=1.0, kernel_noise_std=0.0, point_noise_std=0.0):
#         self.points = points.copy()
#         self.lengthscale = lengthscale
#         self.n = len(points)
#         self.kernel_noise_std = kernel_noise_std
#         self.points += point_noise_std * np.random.randn(*self.points.shape) * self.points
        
#     def calculate_row(self, i):
#         diff = self.points - self.points[i]
#         sq_dists = np.sum(diff**2, axis=1)
#         rbf = np.exp(-sq_dists / (2*self.lengthscale**2))
#         rbf += self.kernel_noise_std * np.random.randn(*rbf.shape) * rbf
#         return rbf
    
#     def __getitem__(self, key):
#         if isinstance(key, tuple):
#             row_idx, col_idx = key
#         else:
#             row_idx, col_idx = key, slice(None)
        
#         if isinstance(row_idx, int):
#             row = self.calculate_row(row_idx)
#             return row[col_idx]
#         elif isinstance(row_idx, slice):
#             start, stop, step = row_idx.indices(self.n)
#             rows = np.array([self.calculate_row(i) for i in range(start, stop, step)])
#             return rows[:, col_idx]
#         elif isinstance(row_idx, (list, np.ndarray)):
#             rows = np.array([self.calculate_row(i) for i in row_idx])
#             return rows[:, col_idx]
#         else:
#             raise IndexError("Invalid index type")

#     def __len__(self):
#         return self.n

#     @property
#     def shape(self):
#         return (self.n, self.n)


import os
import json
import time
import hashlib
import numpy as np
import scipy as sp


class StreamingRBFKernel:
    def __init__(
        self,
        points,
        lengthscale=1.0,
        kernel_noise_std=0.0,
        point_noise_std=0.0,
        block_size=1024,
        cache_dir=None,
        dtype=np.float32,
        verbose=False,
    ):
        self.points = np.asarray(points, dtype=np.float64).copy()
        self.lengthscale = float(lengthscale)
        self.n = len(points)
        self.block_size = int(block_size)
        self.dtype = np.dtype(dtype)
        self.verbose = verbose

        if point_noise_std > 0.0:
            self.points += (
                point_noise_std
                * np.random.randn(*self.points.shape)
                * self.points
            )

        if kernel_noise_std != 0.0:
            raise ValueError(
                "kernel_noise_std must be 0.0 for block-precomputed operator mode"
            )
        self.kernel_noise_std = 0.0

        self.cache_dir = cache_dir
        self._matvec_calls = 0
        self._matmat_calls = 0

        # Special-case in-memory cache when the whole kernel is a single block.
        self._single_block_memory = None
        self._single_block_range = None
        self.small_matrix_threshold = 4096
        self._full_matrix_memory = None

    def __len__(self):
        return self.n

    @property
    def shape(self):
        return (self.n, self.n)

    def _kernel_block(self, I, J):
        Xi = self.points[I]
        Xj = self.points[J]

        Xi_sq = np.sum(Xi * Xi, axis=1)[:, None]
        Xj_sq = np.sum(Xj * Xj, axis=1)[None, :]
        sq_dists = Xi_sq + Xj_sq - 2.0 * (Xi @ Xj.T)
        np.maximum(sq_dists, 0.0, out=sq_dists)

        return np.exp(-sq_dists / (2.0 * self.lengthscale**2)).astype(
            self.dtype, copy=False
        )

    def _maybe_promote_full_matrix_to_memory(self):
        if self._full_matrix_memory is not None:
            return
        if self.n > self.small_matrix_threshold:
            return
        self._full_matrix_memory = self._kernel_block(slice(None), slice(None))

    def calculate_row(self, i):
        return self._kernel_block(slice(i, i + 1), slice(None))[0]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_idx, col_idx = key
        else:
            row_idx, col_idx = key, slice(None)
        return self._kernel_block(row_idx, col_idx)

    def _cache_key(self):
        h = hashlib.sha1()
        h.update(str(self.n).encode())
        h.update(str(self.points.shape[1:] if self.points.ndim > 1 else ()).encode())
        h.update(str(self.lengthscale).encode())
        h.update(str(self.block_size).encode())
        h.update(str(self.dtype).encode())
        h.update(str(self.points.dtype).encode())

        # Small sample of points for sanity/versioning without hashing everything
        flat = np.ascontiguousarray(self.points.reshape(-1))
        sample_len = min(flat.size, 4096)
        h.update(flat[:sample_len].tobytes())
        return h.hexdigest()[:16]

    def _resolved_cache_dir(self):
        if self.cache_dir is None:
            raise ValueError("cache_dir is None; block precompute requires a cache_dir")
        return os.path.join(
            self.cache_dir,
            f"rbf_n{self.n}_ls{self.lengthscale:g}_{self._cache_key()}",
        )

    def _meta_path(self):
        return os.path.join(self._resolved_cache_dir(), "meta.json")

    def _block_path(self, i0, i1):
        return os.path.join(self._resolved_cache_dir(), f"block_{i0}_{i1}.npy")

    def _num_blocks(self):
        return (self.n + self.block_size - 1) // self.block_size

    def _maybe_promote_single_block_to_memory(self):
        if self._single_block_memory is not None:
            return

        if self._num_blocks() != 1:
            return

        i0, i1 = 0, self.n
        path = self._block_path(i0, i1)
        if not os.path.exists(path):
            return

        if self.verbose:
            print(f"[cache] loading single block into memory: {path}")

        self._single_block_memory = np.load(path, allow_pickle=False)
        self._single_block_range = (i0, i1)

    def precompute_blocks(self, overwrite=False):
        cache_root = self._resolved_cache_dir()
        os.makedirs(cache_root, exist_ok=True)

        meta = {
            "n": self.n,
            "lengthscale": self.lengthscale,
            "block_size": self.block_size,
            "dtype": str(self.dtype),
            "points_shape": tuple(self.points.shape),
        }

        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        t0 = time.time()
        total_blocks = self._num_blocks()

        for block_idx, i0 in enumerate(range(0, self.n, self.block_size), start=1):
            i1 = min(i0 + self.block_size, self.n)
            path = self._block_path(i0, i1)

            if (not overwrite) and os.path.exists(path):
                if self.verbose:
                    print(f"[cache] skip existing block {block_idx}/{total_blocks}: {i0}:{i1}")
                continue

            bt = time.time()
            K_block = self._kernel_block(slice(i0, i1), slice(None))
            np.save(path, K_block, allow_pickle=False)

            if self.verbose:
                gb = K_block.nbytes / (1024**3)
                print(
                    f"[cache] wrote block {block_idx}/{total_blocks}: {i0}:{i1} "
                    f"shape={K_block.shape} size={gb:.3f} GB "
                    f"time={time.time() - bt:.2f}s"
                )

        # If there is only one block, keep it in memory after precompute.
        self._maybe_promote_single_block_to_memory()

        if self.verbose:
            print(f"[cache] done in {time.time() - t0:.2f}s -> {cache_root}")

    def _load_block(self, i0, i1):
        self._maybe_promote_single_block_to_memory()

        if (
            self._single_block_memory is not None
            and self._single_block_range == (i0, i1)
        ):
            return self._single_block_memory

        path = self._block_path(i0, i1)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing cached block {path}. Run precompute_blocks() first."
            )
        return np.load(path, mmap_mode="r")

    def matvec(self, v):
        v = np.asarray(v, dtype=self.dtype).reshape(-1)
        if v.ndim != 1:
            raise ValueError("matvec expects a 1D vector")
        if v.shape[0] != self.n:
            raise ValueError(
                f"matvec dimension mismatch: kernel shape={self.shape}, vector shape={v.shape}"
            )

        self._matvec_calls += 1
        if self._matvec_calls % 50 == 0:
            print(f"matvec calls: {self._matvec_calls}")

        self._maybe_promote_full_matrix_to_memory()
        if self._full_matrix_memory is not None:
            return self._full_matrix_memory @ v

        out = np.zeros(self.n, dtype=self.dtype)
        for i0 in range(0, self.n, self.block_size):
            i1 = min(i0 + self.block_size, self.n)
            K_block = self._load_block(i0, i1)
            out[i0:i1] = K_block @ v

        return out

    def rmatvec(self, v):
        return self.matvec(v)

    def matmat(self, V):
        V = np.asarray(V, dtype=self.dtype)
        if V.ndim != 2:
            raise ValueError("matmat expects a 2D array")
        if V.shape[0] != self.n:
            raise ValueError(
                f"matmat dimension mismatch: kernel shape={self.shape}, matrix shape={V.shape}"
            )

        self._matmat_calls += 1
        if self._matmat_calls % 50 == 0:
            print(f"matmat calls: {self._matmat_calls}")

        self._maybe_promote_full_matrix_to_memory()
        if self._full_matrix_memory is not None:
            return self._full_matrix_memory @ V

        out = np.zeros((self.n, V.shape[1]), dtype=self.dtype)
        for i0 in range(0, self.n, self.block_size):
            i1 = min(i0 + self.block_size, self.n)
            K_block = self._load_block(i0, i1)
            out[i0:i1, :] = K_block @ V

        return out

    def rmatmat(self, V):
        return self.matmat(V)

    def __matmul__(self, other):
        other = np.asarray(other, dtype=self.dtype)

        if other.ndim == 1:
            if other.shape[0] != self.n:
                raise ValueError(
                    f"matmul dimension mismatch: kernel shape={self.shape}, vector shape={other.shape}"
                )
            return self.matvec(other)

        if other.ndim == 2:
            if other.shape[0] != self.n:
                raise ValueError(
                    f"matmul dimension mismatch: kernel shape={self.shape}, matrix shape={other.shape}"
                )
            return self.matmat(other)

        raise ValueError("matmul expects a 1D or 2D ndarray")

    def to_linear_operator(self):
        return sp.sparse.linalg.LinearOperator(
            shape=(self.n, self.n),
            matvec=self.matvec,
            rmatvec=self.rmatvec,
            matmat=self.matmat,
            rmatmat=self.rmatmat,
            dtype=self.dtype,
        )

    def stats(self):
        return {
            "matvec_calls": self._matvec_calls,
            "matmat_calls": self._matmat_calls,
            "cache_dir": self._resolved_cache_dir(),
            "single_block_in_memory": self._single_block_memory is not None,
        }

def abbreviate_phrase(phrase):
    # Remove parentheses and split the phrase into words
    words = phrase.replace('(', '').replace(')', '').split()
    
    # Take the first letter of each word, capitalize it, and join
    abbreviation = ''.join(word[0].upper() for word in words)
    
    return abbreviation

def analyze_correlation(approx_residuals, some_measure, 
                        dir_path, iteration,
                        n=40, name="Current Measure"):
    # Ensure we're only looking at the first n elements
    approx_residuals = approx_residuals[:n]
    some_measure = some_measure[:n]
    
    # Calculate Pearson correlation coefficient
    correlation, p_value = stats.pearsonr(some_measure, np.log10(approx_residuals))
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(some_measure, approx_residuals)
    plt.xlabel(name)
    plt.ylabel('Approximate Residual')
    plt.yscale('log')  # Use log scale for residuals due to their wide range
    
    # Add correlation line
    # z = np.polyfit(some_measure, np.log10(approx_residuals), 1)
    # p = np.poly1d(z)
    # plt.plot(some_measure, 10**p(some_measure), "r--", alpha=0.8)

    title = f'{name} vs Approximate Residual (First {n} Eigenvectors)\n'

    # title += f"Pearson correlation coefficient: {correlation:.4f}\n"
    
    if p_value < 0.05:
        title += f"p value = {p_value:.3f} < 0.05"
    else:
        title += f"p value = {p_value:.3f} >= 0.05"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(dir_path + f'/{abbreviate_phrase(name)}_vs_residual_window_{iteration+1}.png', bbox_inches='tight', pad_inches=0.5)
    plt.show()
    plt.close('all')

def download_and_read_matrix(url):
    # Download the file
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad responses

    # Create a file-like object from the response content
    file_like_object = io.BytesIO(response.content)

    # Open the tar.gz file
    with tarfile.open(fileobj=file_like_object, mode="r:gz") as tar:
        # Find the .mtx file in the archive
        mtx_file = [f for f in tar.getnames() if f.endswith('.mtx')][0]
        
        # Extract and read the .mtx file
        f = tar.extractfile(mtx_file)
        matrix = mmread(f)

    return sparse.csr_matrix(matrix)

def download_and_read_matrix_cached(url, cache_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if not os.path.exists(cache_path):
        response = requests.get(url)
        response.raise_for_status()
        with open(cache_path, "wb") as f:
            f.write(response.content)

    with tarfile.open(cache_path, "r:gz") as tar:
        mtx_file = [f for f in tar.getnames() if f.endswith('.mtx')][0]
        f = tar.extractfile(mtx_file)
        matrix = mmread(f)

    return sparse.csr_matrix(matrix)

def download_and_read_matrix(url, cache_path):
    if os.path.exists(cache_path):
        with tarfile.open(cache_path, "r:gz") as tar:
            mtx_file = [f for f in tar.getnames() if f.endswith('.mtx')][0]
            return sparse.csr_matrix(mmread(tar.extractfile(mtx_file)))

    response = requests.get(url)
    response.raise_for_status()

    with open(cache_path, "wb") as f:
        f.write(response.content)

    return download_and_read_matrix(url, cache_path)

# Optional: Soft thresholding function
def soft_thresholding(S, threshold=0):
    return np.maximum(S - threshold, 0)

def soft_thresholding_Ghashami(S, ):
    # Assuming S is descending order
    return np.sqrt(S**2 - S[-1]**2) 

# def soft_thresholding_Ghashami(S):
#     # Assuming S is descending order
#     S = S.copy()
#     S[-2] = 
#     return #np.sqrt(S**2 - S[-1]**2) 

def match_eigenpairs_by_norm(estimated_eigenvalues, estimated_eigenvectors, 
                                    exact_eigenvalues, exact_eigenvectors):
    n = estimated_eigenvectors.shape[1]
    
    # Normalize eigenvectors
    estimated_eigenvectors = estimated_eigenvectors / np.linalg.norm(estimated_eigenvectors, axis=0)
    exact_eigenvectors = exact_eigenvectors / np.linalg.norm(exact_eigenvectors, axis=0)
    
    # Compute the pairwise differences between estimated and exact eigenvectors
    differences = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Compute norm of the difference in eigenvectors
            differences[i, j] = np.linalg.norm(estimated_eigenvectors[:, i] - exact_eigenvectors[:, j])
    
    # Use the Hungarian algorithm to find the best matches
    row_ind, col_ind = linear_sum_assignment(differences)
    
    # Create the matches
    matches = {}
    for i, j in zip(row_ind, col_ind):
        angle = np.arccos(np.abs(np.dot(estimated_eigenvectors[:, i], exact_eigenvectors[:, j])))
        relative_diff = np.abs(estimated_eigenvalues[i] - exact_eigenvalues[j]) / np.abs(exact_eigenvalues[j])
        matches[i] = (j, estimated_eigenvalues[i], exact_eigenvalues[j], differences[i, j], angle, relative_diff)
    
    return matches

def match_eigenpairs_by_angle(estimated_eigenvalues, estimated_eigenvectors, 
                     exact_eigenvalues, exact_eigenvectors, 
                     angle_threshold=np.pi/4):
    n = len(estimated_eigenvalues)
    
    # Normalize eigenvectors
    estimated_eigenvectors = estimated_eigenvectors / np.linalg.norm(estimated_eigenvectors, axis=0)
    exact_eigenvectors = exact_eigenvectors / np.linalg.norm(exact_eigenvectors, axis=0)
    
    # Compute cosine similarities
    similarities = np.abs(np.dot(estimated_eigenvectors.T, exact_eigenvectors))
    
    # Use the Hungarian algorithm to find the best matches
    row_ind, col_ind = linear_sum_assignment(-similarities)
    # print(similarities.shape, estimated_eigenvectors.shape, )
    
    # Create the matches and check against the threshold
    matches = {}
    for i, j in zip(row_ind, col_ind):
        eps = 1e-6
        assert np.all(similarities[i, j] > -1.0 - eps) and np.all(similarities[i, j] < 1.0 + eps), "Invalid canonical correlation found" 
        angle = np.arccos(np.clip(similarities[i, j], -1.0, 1.0))
        norm_diff = np.linalg.norm(estimated_eigenvectors[:, i] - exact_eigenvectors[:, j])
        # if angle <= angle_threshold:
        relative_diff = np.abs(estimated_eigenvalues[i] - exact_eigenvalues[j]) / np.abs(exact_eigenvalues[j])
        matches[i] = (j, estimated_eigenvalues[i], exact_eigenvalues[j], norm_diff, angle, relative_diff)
    
    return matches

def match_eigenpairs_by_eigenvalues(estimated_eigenvalues, estimated_eigenvectors, 
                                    exact_eigenvalues, exact_eigenvectors, 
                                    relative_threshold=0.1):
    n = len(estimated_eigenvalues)
    
    # Compute the pairwise differences between estimated and exact eigenvalues
    differences = np.abs(estimated_eigenvalues[:, np.newaxis] - exact_eigenvalues)
    
    # Use the Hungarian algorithm to find the best matches
    row_ind, col_ind = linear_sum_assignment(differences)
    
    # Create the matches and check against the threshold
    matches = {}
    for i, j in zip(row_ind, col_ind):
        relative_diff = differences[i, j] / np.abs(exact_eigenvalues[j])
        # if relative_diff <= relative_threshold:
        # Compute the angle between eigenvectors for information
        cos_angle = np.abs(np.dot(estimated_eigenvectors[:, i], exact_eigenvectors[:, j]))
        eps = 1e-6
        assert np.all(cos_angle > -1.0 - eps) and np.all(cos_angle < 1.0 + eps), "Invalid canonical correlation found" 
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        norm_diff = np.linalg.norm(estimated_eigenvectors[:, i] - exact_eigenvectors[:, j])
        matches[i] = (j, estimated_eigenvalues[i], exact_eigenvalues[j], norm_diff, angle, relative_diff)
    
    return matches

def plot_spectrum_comparison(S, S_exact, 
                             A_norm, name, iteration, dir_path):
    # np.save('some_measure.npy', some_measure)
    
    plt.figure(figsize=(10, 6))
    plt.plot(S, label='Approximated Spectrum')
    plt.plot(S_exact, label='Exact Spectrum')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.title(f'Comparison of Approximated and Exact Spectra iteration {iteration}')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(dir_path + f'/spectrum_window_{iteration+1}.png', bbox_inches='tight', pad_inches=0.5)
    plt.show()
    plt.close('all')


    plt.figure(figsize=(10, 6))
    plt.plot(np.abs(S - S_exact[:len(S)]) / A_norm, label='$\\left\\|\\frac{S-S_\\text{exact}}{\|A\|_F}\\right\\|$')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.title(f'Difference between Approximated and Exact Spectra iteration {iteration}')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(dir_path + f'/diffspec_relA_window_{iteration+1}.png', bbox_inches='tight', pad_inches=0.5)
    plt.show()
    plt.close('all')

    plt.figure(figsize=(10, 6))
    plt.plot(np.abs(S - S_exact[:len(S)]) / np.abs(S_exact[:len(S)]), label='$\\left\\|\\frac{S-S_\\text{exact}}{|S|_\\text{exact}}\\right\\|$')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.title(f'Difference between Approximated and Exact Spectra iteration {iteration}')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(dir_path + f'/diffspec_relS_window_{iteration+1}.png', bbox_inches='tight', pad_inches=0.5)
    plt.show()
    plt.close('all')

def plot_residuals_old(A_csr, S, Vt, S_exact, Vt_exact, U_exact, 
                   A_norm, name, iteration, dir_path, is_sym_psd):
    approx_residuals = []
    # exact_residuals = []

    if is_sym_psd:
        for i in range(len(S)):
            # Approximated residual
            # approx_res = A.T @ (A @ Vt[i].T) - S[i]**2 * Vt[i].T
            approx_res = (A_csr @ Vt[i].T) - (S[i]) * Vt[i].T
            approx_residuals.append(np.linalg.norm(approx_res) / A_norm)
            
            # Exact residual
            # exact_res = A.T @ (A @ Vt_exact[i].T) - S_exact[i]**2 * Vt_exact[i].T
            # exact_residuals.append(np.linalg.norm(exact_res) / np.linalg.norm(S_exact[i]**2 * Vt_exact[i].T))
        
        approx_residuals = np.array(approx_residuals)
        # approx_residuals, exact_residuals = np.array(approx_residuals), np.array(exact_residuals)
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(approx_residuals, label='$\\frac{\|Av - \sigma v\|_2}{\|A\|_F}$')
        # plt.semilogy(exact_residuals, label='Exact Relative Residuals')
        plt.xlabel('Index')
        plt.ylabel('Residual Norm (log scale)')
        plt.title('')
        plt.legend()
        plt.grid(True)
        plt.savefig(dir_path + f'/residual_window_{iteration+1}.png', bbox_inches='tight', pad_inches=0.5)
        plt.show()
        plt.close('all')

    # matches = match_eigenpairs_by_eigenvalues(S, Vt.T, 
    #                                S_exact[:], Vt_exact[:].T)
    matches = match_eigenpairs_by_angle(S, Vt.T, 
                                   S_exact[:], Vt_exact[:].T)
    # matches = match_eigenpairs_by_norm(S, Vt.T, 
    #                                S_exact[:], Vt_exact[:].T)
    # for i in matches:
    #     print(f"Estimated eigenpair {i} matched with exact eigenpair {matches[i][0]}")
    #     print(f"Estimated eigenvalue: {matches[i][1]:.4f}, Exact eigenvalue: {matches[i][2]:.4f}")
    #     print(f"Angle between eigenvectors: {matches[i][3]:.4f} radians")
    #     print()

    approx_residuals = []
    diff = []
    
    for i in matches:
        # Approximated residual
        # approx_res = A.T @ (A @ Vt[i].T) - S[i]**2 * Vt[i].T
        k = matches[i][0]
        approx_res = (A_csr @ Vt[i].T) - (S[i]) * U_exact[:,k]
        approx_residuals.append(np.linalg.norm(approx_res) / A_norm)
        # diff.append(np.linalg.norm(Vt[i] - U_exact[:,k]))
        
    approx_residuals = np.array(approx_residuals)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(list(matches.keys()), approx_residuals, label='$\\frac{\|Av - \sigma u\|_2}{\|A\|_F}$')
    plt.xlabel('Index')
    plt.ylabel('Residual Norm (log scale)')
    plt.title('')
    # plt.legend()
    plt.grid(True)
    plt.savefig(dir_path + f'/residual_uexact_window_{iteration+1}.png', bbox_inches='tight', pad_inches=0.5)
    plt.show()
    plt.close('all')

    # plt.figure(figsize=(10, 6))
    # plt.semilogy(list(matches.keys()), [matches[i][3] for i in matches])
    # plt.xlabel('Index')
    # plt.ylabel('Angles')
    # plt.title('')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # plt.close('all')

    norm_diff = [matches[i][3] for i in matches]
    angles = [matches[i][4] for i in matches]
    rel_eig_diff = [matches[i][5] for i in matches]
    assert len(approx_residuals) == len(norm_diff), "Arrays must have the same length"

    
    analyze_correlation(approx_residuals, norm_diff, 
                        dir_path, iteration, 
                        name="Norm of Difference",)
    analyze_correlation(approx_residuals, angles, 
                        dir_path, iteration, 
                        name="Angle")
    analyze_correlation(approx_residuals, angles, dir_path, iteration,
                        name="(Relative) Eigenvalue Difference")


def plot_residuals(A_csr, S, Vt, S_exact, Vt_exact, U_exact, 
                   A_norm, name, iteration, dir_path, is_sym_psd):

    if is_sym_psd:
        approx_residuals = []
        for i in range(len(S)):
            # Approximated residual
            # approx_res = A.T @ (A @ Vt[i].T) - S[i]**2 * Vt[i].T
            approx_res = (A_csr @ Vt[i].T) - (S[i]) * Vt[i].T
            approx_residuals.append(np.linalg.norm(approx_res) / A_norm)
            
            # Exact residual
            # exact_res = A.T @ (A @ Vt_exact[i].T) - S_exact[i]**2 * Vt_exact[i].T
            # exact_residuals.append(np.linalg.norm(exact_res) / np.linalg.norm(S_exact[i]**2 * Vt_exact[i].T))
        
        approx_residuals = np.array(approx_residuals)
        # approx_residuals, exact_residuals = np.array(approx_residuals), np.array(exact_residuals)
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(approx_residuals, label='$\\frac{\|Av - \sigma v\|_2}{\|A\|_F}$')
        # plt.semilogy(exact_residuals, label='Exact Relative Residuals')
        plt.xlabel('Index')
        plt.ylabel('Residual Norm (sym pd)')
        plt.title('')
        plt.legend()
        plt.grid(True)
        plt.savefig(dir_path + f'/residual_window_{iteration+1}.png', bbox_inches='tight', pad_inches=0.5)
        plt.show()
        plt.close('all')

    approx_residuals = []
    for i in range(len(S)):
        # Approximated residual
        # approx_res = A.T @ (A @ Vt[i].T) - S[i]**2 * Vt[i].T
        u = A_csr @ Vt[i].T
        u = u / np.linalg.norm(u)
        approx_res = (A_csr.T @ u) - (S[i]) * Vt[i].T
        approx_residuals.append(np.linalg.norm(approx_res) / A_norm)
        
    approx_residuals = np.array(approx_residuals)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(approx_residuals, label='$\\frac{\|A^Tu - \sigma v\|_2}{\|A\|_F}$')
    plt.xlabel('Index')
    plt.ylabel('Residual Norm (no sym pd)')
    plt.title('')
    plt.legend()
    plt.grid(True)
    plt.savefig(dir_path + f'/residual_u_window_{iteration+1}.png', bbox_inches='tight', pad_inches=0.5)
    plt.show()
    plt.close('all')
    

def plot_canonical_angles(Vt, Vt_exact, iteration, dir_path):
    # Compute the singular values of Q1.T @ Q2
    print(Vt.shape, Vt_exact.shape)
    s = np.linalg.svd(Vt @ Vt_exact[:Vt.shape[0], :].T, compute_uv=False)
    
    # Compute the angles in radians
    eps = 1e-6
    assert np.all(s > -1.0 - eps) and np.all(s < 1.0 + eps), "Invalid canonical correlation found" 
    angles = np.arccos(np.clip(s, -1.0, 1.0))
    print("Subspace angle 2:", max(angles), np.mean(angles))
    
    epsilon = 1e-4
    s = -np.log(np.maximum(1 - s, epsilon))
    # print(s)
    # Create the boxplot
    fig, ax = plt.subplots(figsize=(10, 8))
    bp = ax.boxplot(s, whis=1.5)
    
    # Extract positions
    whiskers = [item.get_ydata()[1] for item in bp['whiskers']]
    caps = [item.get_ydata()[0] for item in bp['caps']]
    boxes = [item.get_ydata() for item in bp['boxes']][0]
    medians = [item.get_ydata()[0] for item in bp['medians']]
    fliers = bp['fliers'][0].get_ydata()
    
    # Calculate statistics
    min_val, max_val = np.min(s), np.max(s)
    q1, median_val, q3 = np.percentile(s, [25, 50, 75])
    lower_whisker, upper_whisker = whiskers[0], whiskers[1]
    
    # Function to add annotation with offset
    def add_annotation(x, y, text, offset=0, color='black', ha='left', va='center'):
        ax.annotate(text, (x, y), xytext=(offset, 0), textcoords='offset points',
                    ha=ha, va=va, color=color)
    
    def inverse_transform(s):
        return 1 - np.exp(-s)
    
    # Add annotations with adjusted positions
    add_annotation(0.6, min_val+0.1, f'Min: {inverse_transform(min_val):.2f}', offset=5, va='bottom')
    add_annotation(0.6, max_val-0.1, f'Max: {inverse_transform(max_val):.2f}', offset=5, va='top')
    add_annotation(1.1, q1, f'Q1: {inverse_transform(q1):.2f}', offset=5)
    add_annotation(1.1+0.1, median_val, f'Median: {inverse_transform(median_val):.2f}', offset=5)
    add_annotation(1.1, q3, f'Q3: {inverse_transform(q3):.2f}', offset=5)
    add_annotation(1.1+0.1, lower_whisker, f'Lower whisker: {inverse_transform(lower_whisker):.2f}', offset=5, va='bottom')
    add_annotation(1.1+0.1, upper_whisker, f'Upper whisker: {inverse_transform(upper_whisker):.2f}', offset=5, va='top')
    
    # Add outlier information
    if len(fliers) > 0:
        outlier_min, outlier_max = np.min(fliers), np.max(fliers)
        add_annotation(0.78, outlier_min, f'Min outlier: {inverse_transform(outlier_min):.2f}', offset=5, va='bottom', color='red')
        add_annotation(0.78, outlier_max, f'Max outlier: {inverse_transform(outlier_max):.2f}', offset=5, va='top', color='red')
        add_annotation(0.78, (outlier_min+outlier_max)/2, f'Number of outliers: {len(fliers)}', offset=5, va='center', color='red')
    
    # Customize the plot
    ax.set_title('Boxplot with Non-Overlapping Annotations')
    ax.set_ylabel('Values')
    ax.set_xlim(0.5, 1.5)  # Adjust x-axis limits to make room for annotations
    ax.set_ylim(0, -np.log(epsilon)+0.1)
    
    y_ticks = np.array([0, 0.5, 0.9, 0.99, 0.999, 0.9999, 1])
    ax.set_yticks(-np.log(1 - y_ticks + epsilon))
    ax.set_yticklabels([f"{y}" for y in y_ticks])
    
    plt.grid()
    plt.tight_layout()
    plt.savefig(dir_path + f'/angles_window_{iteration+1}.png', bbox_inches='tight', pad_inches=0.5)
    plt.show()
    plt.close('all')

def save_spectrum_comparison(S, S_exact, A_norm, name, iteration, dir_path, S_quotient=None,
                             score_history=None, save_in_text=True, extra_fields=None):
    os.makedirs(dir_path, exist_ok=True)

    ext = "txt" if save_in_text else "npz"

    # Save S_exact only at iteration 0
    filepath = os.path.join(dir_path, f"spectrum_data_{iteration}.{ext}")
    spectrum_kwargs = dict(S=S, S_quotient=S_quotient, score_history=score_history, iteration=iteration)
    if extra_fields:
        spectrum_kwargs.update(extra_fields)
    if iteration == 0 and S_exact is not None:
        if save_in_text and len(S_exact) > 0 and S_exact[0] != 0:
            spectrum_kwargs["S_exact"] = S_exact / S_exact[0]
        else:
            spectrum_kwargs["S_exact"] = S_exact

    if save_in_text:
        save_txt(filepath, **spectrum_kwargs)
    else:
        np.savez(filepath, **spectrum_kwargs, allow_pickle=True)

    # These comparison files still need S_exact available in memory at save time
    if S_exact is not None:
        diff_relA = np.abs(S - S_exact[:len(S)]) / A_norm
        filepath = os.path.join(dir_path, f"diffspec_relA_data_{iteration}.{ext}")
        if save_in_text:
            save_txt(filepath, diff=diff_relA, iteration=iteration)
        else:
            np.savez(filepath, diff=diff_relA, iteration=iteration, allow_pickle=True)

        diff_relS = np.abs(S - S_exact[:len(S)]) / np.abs(S_exact[:len(S)])
        filepath = os.path.join(dir_path, f"diffspec_relS_data_{iteration}.{ext}")
        if save_in_text:
            save_txt(filepath, diff=diff_relS, iteration=iteration)
        else:
            np.savez(filepath, diff=diff_relS, iteration=iteration, allow_pickle=True)

    print(f"Data saved successfully for iteration {iteration}")


def save_residuals(A_csr, S, Vt,
                   S_exact, A_norm, name, iteration, dir_path, is_sym_psd,
                   row_permutation, start_idx, end_idx, save_in_text=True):
    os.makedirs(dir_path, exist_ok=True)

    ext = "txt" if save_in_text else "npz"

    if is_sym_psd:
        approx_residuals_sym = []
        if A_csr.shape[1] < 5e4:
            for i in range(len(S)):
                approx_res = (A_csr @ Vt[i].T) / S_exact[0] - S[i] * Vt[i].T
                approx_residuals_sym.append(np.linalg.norm(approx_res) / A_norm)

        approx_residuals_sym = np.array(approx_residuals_sym)

        filepath = os.path.join(dir_path, f"residuals_sym_psd_data_{iteration}.{ext}")
        if save_in_text:
            save_txt(
                filepath,
                approx_residuals=approx_residuals_sym,
                iteration=iteration,
                A_norm=A_norm,
            )
        else:
            np.savez(
                filepath,
                approx_residuals=approx_residuals_sym,
                iteration=iteration,
                A_norm=A_norm,
                allow_pickle=True
            )

        window_indices = row_permutation[start_idx:end_idx]
        approx_residuals_sym = []
        if A_csr.shape[1] < 5e4:
            for i in range(len(S)):
                approx_res = (A_csr[window_indices, :] @ Vt[i].T) / S_exact[0] - S[i] * Vt[i, window_indices].T
                approx_residuals_sym.append(np.linalg.norm(approx_res) / A_norm)

        approx_residuals_sym = np.array(approx_residuals_sym)

        filepath = os.path.join(dir_path, f"residuals_sym_psd_data_truncated_{iteration}.{ext}")
        if save_in_text:
            save_txt(
                filepath,
                approx_residuals=approx_residuals_sym,
                iteration=iteration,
                A_norm=A_norm,
            )
        else:
            np.savez(
                filepath,
                approx_residuals=approx_residuals_sym,
                iteration=iteration,
                A_norm=A_norm,
                allow_pickle=True
            )

        approx_residuals_sym = []
        approx_residuals_sym_full = []
        S_truncated_Rayleigh_list = []
        S_truncated_Rayleigh_full_list = []

        if A_csr.shape[1] < 5e4:
            for i in range(len(S)):
                S_truncated_Rayleigh = np.dot(
                    Vt[i, window_indices].T,
                    (A_csr[window_indices, :] @ Vt[i].T) / S_exact[0]
                )
                sq_norm_V = np.dot(Vt[i, window_indices].T, Vt[i, window_indices].T)

                S_truncated_Rayleigh_full = np.dot(
                    Vt[i, row_permutation[:end_idx]].T,
                    (A_csr[row_permutation[:end_idx], :] @ Vt[i].T) / S_exact[0]
                )
                sq_norm_V_full = np.dot(
                    Vt[i, row_permutation[:end_idx]].T,
                    Vt[i, row_permutation[:end_idx]].T
                )

                if sq_norm_V == 0:
                    S_truncated_Rayleigh = np.nan
                else:
                    S_truncated_Rayleigh /= sq_norm_V

                if sq_norm_V_full == 0:
                    S_truncated_Rayleigh_full = np.nan
                else:
                    S_truncated_Rayleigh_full /= sq_norm_V_full

                approx_res = (
                    (A_csr[window_indices, :] @ Vt[i].T) / S_exact[0]
                ) - S_truncated_Rayleigh * Vt[i, window_indices].T

                approx_res_full = (
                    (A_csr[row_permutation[:end_idx], :] @ Vt[i].T) / S_exact[0]
                ) - S_truncated_Rayleigh * Vt[i, row_permutation[:end_idx]].T

                approx_residuals_sym.append(np.linalg.norm(approx_res) / A_norm)
                approx_residuals_sym_full.append(np.linalg.norm(approx_res_full) / A_norm)
                S_truncated_Rayleigh_list.append(S_truncated_Rayleigh)
                S_truncated_Rayleigh_full_list.append(S_truncated_Rayleigh_full)

        approx_residuals_sym = np.array(approx_residuals_sym)
        approx_residuals_sym_full = np.array(approx_residuals_sym_full)
        S_truncated_Rayleigh_list = np.array(S_truncated_Rayleigh_list)
        S_truncated_Rayleigh_full_list = np.array(S_truncated_Rayleigh_full_list)

        filepath = os.path.join(dir_path, f"residuals_sym_psd_data_truncated_Rayleigh_{iteration}.{ext}")
        if save_in_text:
            save_txt(
                filepath,
                approx_residuals=approx_residuals_sym,
                approx_residuals_full=approx_residuals_sym_full,
                iteration=iteration,
                S_truncated_Rayleigh_list=S_truncated_Rayleigh_list,
                S_truncated_Rayleigh_full_list=S_truncated_Rayleigh_full_list,
                A_norm=A_norm,
            )
        else:
            np.savez(
                filepath,
                approx_residuals=approx_residuals_sym,
                approx_residuals_full=approx_residuals_sym_full,
                iteration=iteration,
                S_truncated_Rayleigh_list=S_truncated_Rayleigh_list,
                S_truncated_Rayleigh_full_list=S_truncated_Rayleigh_full_list,
                A_norm=A_norm,
                allow_pickle=True
            )

    else:
        approx_residuals = []
        if A_csr.shape[1] < 5e4:
            for i in range(len(S)):
                u = (A_csr @ Vt[i].T) / S_exact[0]
                u = u / np.linalg.norm(u)
                approx_res = ((A_csr.T @ u) / S_exact[0]) - S[i] * Vt[i].T
                approx_residuals.append(np.linalg.norm(approx_res) / A_norm)

        approx_residuals = np.array(approx_residuals)

        filepath = os.path.join(dir_path, f"residuals_data_{iteration}.{ext}")
        if save_in_text:
            save_txt(filepath, approx_residuals=approx_residuals, iteration=iteration)
        else:
            np.savez(
                filepath,
                approx_residuals=approx_residuals,
                iteration=iteration,
                allow_pickle=True
            )

    print(f"Residuals data saved successfully for iteration {iteration}")


def save_residuals_reservoir(reservoir, reservoir_idx, row_permutation,
                             S, Vt, S_exact, A_norm, A_csr, S_quotient,
                             name, iteration, dir_path, save_in_text=True):
    os.makedirs(dir_path, exist_ok=True)

    ext = "txt" if save_in_text else "npz"

    print_memory_usage(f"Before residual reservoir, window {iteration+1}")
    reservoir_Vt = reservoir @ Vt.T
    regular_Vt = (A_csr @ Vt.T) / S_exact[0]

    Vt_permuted = Vt[:, row_permutation[reservoir_idx]]

    reservoir_residuals = []
    regular_residuals = []
    reservoir_residuals_quotient = []
    regular_residuals_quotient = []

    for i in range(len(S_quotient)):
        # reservoir_res = reservoir_Vt[:, i] - (S[i]) * Vt_permuted
        # reservoir_res_quotient = reservoir_Vt[:, i] - S_quotient[i] * Vt_permuted
        reservoir_res = reservoir_Vt[:, i] - S[i] * Vt_permuted[i]
        reservoir_res_quotient = reservoir_Vt[:, i] - S_quotient[i] * Vt_permuted[i]
        regular_res = regular_Vt[:, i] - np.dot(Vt[i,:], regular_Vt[:, i]) * Vt[i]
        # regular_res = regular_Vt[:, i] - S[i] * Vt[i]
        regular_res_quotient = regular_Vt[:, i] - S_quotient[i] * Vt[i]

        reservoir_residuals_quotient.append(np.linalg.norm(reservoir_res_quotient))
        regular_residuals_quotient.append(np.linalg.norm(regular_res_quotient))
        reservoir_residuals.append(np.linalg.norm(reservoir_res))
        regular_residuals.append(np.linalg.norm(regular_res))

    reservoir_residuals = np.array(reservoir_residuals)
    regular_residuals = np.array(regular_residuals)
    reservoir_residuals_quotient = np.array(reservoir_residuals_quotient)
    regular_residuals_quotient = np.array(regular_residuals_quotient)

    whole_space_regular_residuals = regular_Vt - S[:len(S_quotient)] * Vt.T
    whole_space_regular_residuals_2norm = np.linalg.norm(whole_space_regular_residuals, ord=2)
    whole_space_regular_residuals_fro = np.linalg.norm(whole_space_regular_residuals, ord="fro")

    whole_space_reservoir_residuals = reservoir_Vt - S[:len(S_quotient)] * Vt_permuted.T
    whole_space_reservoir_residuals_2norm = np.linalg.norm(whole_space_reservoir_residuals, ord=2)
    whole_space_reservoir_residuals_fro = np.linalg.norm(whole_space_reservoir_residuals, ord="fro")

    whole_space_regular_residuals_quotient = regular_Vt - S_quotient * Vt.T
    whole_space_regular_residuals_quotient_2norm = np.linalg.norm(whole_space_regular_residuals_quotient, ord=2)
    whole_space_regular_residuals_quotient_fro = np.linalg.norm(whole_space_regular_residuals_quotient, ord="fro")

    whole_space_reservoir_residuals_quotient = reservoir_Vt - S_quotient * Vt_permuted.T
    whole_space_reservoir_residuals_quotient_2norm = np.linalg.norm(whole_space_reservoir_residuals_quotient, ord=2)
    whole_space_reservoir_residuals_quotient_fro = np.linalg.norm(whole_space_reservoir_residuals_quotient, ord="fro")

    print("reservoir_residuals_quotient:", reservoir_residuals_quotient)
    print(f"2: {whole_space_reservoir_residuals_quotient_2norm:.3f}, fro: {whole_space_reservoir_residuals_quotient_fro:.3f}")
    print("regular_residuals_quotient:", regular_residuals_quotient)
    print(f"2: {whole_space_regular_residuals_quotient_2norm:.3f}, fro: {whole_space_regular_residuals_quotient_fro:.3f}")

    filepath = os.path.join(dir_path, f"reservoir_residuals_data_{iteration}.{ext}")
    if save_in_text:
        save_txt(
            filepath,
            reservoir_residuals=reservoir_residuals,
            regular_residuals=regular_residuals,
            reservoir_residuals_quotient=reservoir_residuals_quotient,
            regular_residuals_quotient=regular_residuals_quotient,
            whole_space_regular_residuals_2norm=whole_space_regular_residuals_2norm,
            whole_space_regular_residuals_fro=whole_space_regular_residuals_fro,
            whole_space_reservoir_residuals_2norm=whole_space_reservoir_residuals_2norm,
            whole_space_reservoir_residuals_fro=whole_space_reservoir_residuals_fro,
            whole_space_regular_residuals_quotient_2norm=whole_space_regular_residuals_quotient_2norm,
            whole_space_regular_residuals_quotient_fro=whole_space_regular_residuals_quotient_fro,
            whole_space_reservoir_residuals_quotient_2norm=whole_space_reservoir_residuals_quotient_2norm,
            whole_space_reservoir_residuals_quotient_fro=whole_space_reservoir_residuals_quotient_fro,
            iteration=iteration,
            A_norm=A_norm,
        )
    else:
        np.savez(
            filepath,
            reservoir_residuals=reservoir_residuals,
            regular_residuals=regular_residuals,
            reservoir_residuals_quotient=reservoir_residuals_quotient,
            regular_residuals_quotient=regular_residuals_quotient,
            whole_space_regular_residuals_2norm=whole_space_regular_residuals_2norm,
            whole_space_regular_residuals_fro=whole_space_regular_residuals_fro,
            whole_space_reservoir_residuals_2norm=whole_space_reservoir_residuals_2norm,
            whole_space_reservoir_residuals_fro=whole_space_reservoir_residuals_fro,
            whole_space_regular_residuals_quotient_2norm=whole_space_regular_residuals_quotient_2norm,
            whole_space_regular_residuals_quotient_fro=whole_space_regular_residuals_quotient_fro,
            whole_space_reservoir_residuals_quotient_2norm=whole_space_reservoir_residuals_quotient_2norm,
            whole_space_reservoir_residuals_quotient_fro=whole_space_reservoir_residuals_quotient_fro,
            iteration=iteration,
            A_norm=A_norm,
            allow_pickle=True
        )

    del reservoir_Vt, regular_Vt, reservoir_residuals, regular_residuals, reservoir_residuals_quotient, regular_residuals_quotient
    gc.collect()
    print_memory_usage(f"After residual reservoir, window {iteration+1}")


def save_canonical_angles(Vt, Vt_exact, iteration, dir_path, additional_label="", save_in_text=True):
    C = Vt @ Vt_exact[:Vt.shape[0], :].T
    s = np.linalg.svd(C, compute_uv=False)

    angles = np.arccos(np.clip(s, -1.0, 1.0))
    print("Subspace angle 2:", max(angles), np.mean(angles))

    epsilon = 1e-4
    s = -np.log(np.maximum(1 - s, epsilon))

    os.makedirs(dir_path, exist_ok=True)

    ext = "txt" if save_in_text else "npz"
    filepath = os.path.join(dir_path, f"canonical_angles{additional_label}_data_{iteration}.{ext}")

    if save_in_text:
        save_txt(filepath, s=s, iteration=iteration, C=C)
    else:
        np.savez(filepath, s=s, iteration=iteration, C=C, allow_pickle=True)

    print(f"Canonical angles data saved successfully for iteration {iteration}")


def save_leftout(Vt, S, Vt_exact, combined, iteration, dir_path, additional_label="", save_in_text=True,
                 extra_fields=None):
    current_total = np.linalg.norm(combined @ Vt_exact[:len(Vt), :].T, axis=0)
    keep = np.linalg.norm((S[:, None] * Vt) @ Vt_exact[:len(Vt), :].T, axis=0)
    throw = current_total - keep

    os.makedirs(dir_path, exist_ok=True)

    ext = "txt" if save_in_text else "npz"
    filepath = os.path.join(dir_path, f"leftout{additional_label}_data_{iteration}.{ext}")
    save_kwargs = dict(iteration=iteration, current_total=current_total, throw=throw)
    if extra_fields:
        save_kwargs.update(extra_fields)

    if save_in_text:
        save_txt(filepath, **save_kwargs)
    else:
        np.savez(
            filepath,
            **save_kwargs,
            allow_pickle=True
        )

    print(f"Leftout data saved successfully for iteration {iteration}")


def projected_subspace_svd_factors(Vt, combined):
    V_basis = orthonormalize_columns(np.asarray(Vt, dtype=np.float32).T, dtype=np.float32)
    if V_basis.size == 0:
        return (
            np.zeros((0, combined.shape[0]), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros((0, combined.shape[1]), dtype=np.float32),
            V_basis,
        )

    B_proj = np.asarray(combined, dtype=np.float32) @ V_basis
    U_proj, s_proj, Rt_proj = np.linalg.svd(B_proj, full_matrices=False)
    V_proj = np.ascontiguousarray(V_basis @ Rt_proj.T, dtype=np.float32)
    return (
        np.ascontiguousarray(U_proj.T, dtype=np.float32),
        np.asarray(s_proj, dtype=np.float32),
        V_proj.T,
        V_basis,
    )


def left_projected_operator_svd_factors(Vt, combined):
    dtype = np.result_type(np.asarray(Vt).dtype, np.asarray(combined).dtype)
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float64
    V_basis = orthonormalize_columns(np.asarray(Vt, dtype=dtype).T, dtype=dtype)
    if V_basis.size == 0:
        return (
            np.zeros((0, combined.shape[0]), dtype=dtype),
            np.zeros(0, dtype=dtype),
            np.zeros((0, combined.shape[1]), dtype=dtype),
            np.zeros((combined.shape[0], 0), dtype=dtype),
        )

    combined_arr = np.asarray(combined, dtype=dtype)
    Y = combined_arr @ V_basis
    # Modified Gram-Schmidt in float32 can lose orthogonality badly here, which
    # breaks the projector interpretation of U_hat U_hat^T. Use a stable QR.
    if Y.size == 0:
        U_hat = np.zeros((combined.shape[0], 0), dtype=dtype)
    else:
        U_hat, _ = np.linalg.qr(np.asarray(Y, dtype=np.float64), mode='reduced')
        U_hat = np.ascontiguousarray(U_hat.astype(dtype, copy=False))
    if U_hat.size == 0:
        return (
            np.zeros((0, combined.shape[0]), dtype=dtype),
            np.zeros(0, dtype=dtype),
            np.zeros((0, combined.shape[1]), dtype=dtype),
            U_hat,
        )

    compressed = np.ascontiguousarray(U_hat.T @ combined_arr, dtype=dtype)
    U_small, s_proj, Vt_proj = np.linalg.svd(compressed, full_matrices=False)
    U_proj = np.ascontiguousarray(U_hat @ U_small, dtype=dtype)
    return (
        np.ascontiguousarray(U_proj.T, dtype=dtype),
        np.asarray(s_proj, dtype=dtype),
        np.ascontiguousarray(Vt_proj, dtype=dtype),
        U_hat,
    )


def save_leftout_projected(Vt, Vt_exact, combined, iteration, dir_path, additional_label="", save_in_text=True):
    Ut_proj, S_proj, Vt_proj, V_basis = projected_subspace_svd_factors(Vt, combined)
    exact_vectors = Vt_exact[:len(Vt_proj), :].T
    current_total = np.linalg.norm(combined @ exact_vectors, axis=0)
    keep_surrogate = np.linalg.norm((S_proj[:, None] * Vt_proj) @ exact_vectors, axis=0)
    projected_operator = (Ut_proj.T * S_proj[None, :]) @ Vt_proj
    keep_projected_true = np.linalg.norm(projected_operator @ exact_vectors, axis=0)
    extra_fields = {
        "leftout_mode": "projected_subspace_svd",
        "keep_surrogate": keep_surrogate,
        "throw_surrogate": current_total - keep_surrogate,
        "keep_projected_true": keep_projected_true,
        "throw_projected_true": current_total - keep_projected_true,
        "projected_keep_gap": keep_projected_true - keep_surrogate,
        "subspace_dim": np.int64(V_basis.shape[1]),
        "projected_singular_values": S_proj,
    }
    save_leftout(Vt_proj, S_proj, Vt_exact, combined, iteration, dir_path,
                 additional_label=additional_label, save_in_text=save_in_text,
                 extra_fields=extra_fields)

    
# def make_operator(window, n, w, V, S):
#     def matvec(x):
#         x = x.reshape(-1)
#         result = np.zeros(n*w, dtype=np.float64)

#         # First (n-1)w rows: V S V^T part
#         V_old = V[:(n-1)*w, :]
#         x_old = x[:(n-1)*w]
#         result[:(n-1)*w] = V_old @ (S * (V_old.T @ x_old))

#         # Last block + cross terms
#         result[(n-1)*w:n*w] = window[:, (n-1)*w:n*w] @ x[(n-1)*w:n*w]
#         V_last = V[(n-1)*w:n*w, :]
        
#         cross_term_old = V_old @ (S * (V_last.T @ x[(n-1)*w:n*w]))
#         cross_term_new = V_last @ (S * (V_old.T @ x_old))
        
#         result[:(n-1)*w] += cross_term_old.real
#         result[(n-1)*w:n*w] += cross_term_new.real
        
#         return result
#     return sp.sparse.linalg.LinearOperator((n*w, n*w), matvec=matvec, rmatvec=matvec, dtype=np.float64)

def inverse_permutation(perm):
    inverse = [0] * len(perm)
    for i in range(len(perm)):
        inverse[perm[i]] = i
    return np.array(inverse)

# def save_residuals_reservoir(reservoir, reservoir_idx, row_permutation,
#                              S, Vt, A_norm, A_csr, S_quotient,
#                              name, iteration, dir_path):
#     # Create directory if it doesn't exist
#     os.makedirs(dir_path, exist_ok=True)

#     print_memory_usage(f"Before residual reservoir, window {iteration+1}")
    
#     reservoir_Vt = reservoir @ Vt.T  
#     regular_Vt = A_csr @ Vt.T  

#     Vt_permuted = Vt[:, row_permutation[reservoir_idx]]

#     #TODO: test this, then do rayleigh
#     reservoir_residuals = []
#     regular_residuals = []
#     reservoir_residuals_quotient = []
#     regular_residuals_quotient = []
#     for i in range(len(S_quotient)):
#         # print(reservoir.shape, Vt[i].shape, S[i])
#         reservoir_res = reservoir_Vt[:, i] - (S[i]) * Vt_permuted
#         reservoir_res_quotient = reservoir_Vt[:, i]- (S_quotient[i]) * Vt_permuted
#         # if A_csr.shape[1] < 5e4:
#         regular_res = regular_Vt[:, i] - (S[i]) * Vt[i]
#         regular_res_quotient = regular_Vt[:, i] - (S_quotient[i]) * Vt[i]
#         reservoir_residuals_quotient.append(np.linalg.norm(reservoir_res_quotient))
#         regular_residuals_quotient.append(np.linalg.norm(regular_res_quotient))
#         reservoir_residuals.append(np.linalg.norm(reservoir_res))
#         regular_residuals.append(np.linalg.norm(regular_res))
        
#     reservoir_residuals = np.array(reservoir_residuals)
#     regular_residuals = np.array(regular_residuals)
#     reservoir_residuals_quotient = np.array(reservoir_residuals_quotient)
#     regular_residuals_quotient = np.array(regular_residuals_quotient)

    
#     # save both fro and 2-norm
#     whole_space_regular_residuals = regular_Vt - S[:len(S_quotient)] * Vt.T
#     whole_space_regular_residuals_2norm = np.linalg.norm(whole_space_regular_residuals, ord=2)
#     whole_space_regular_residuals_fro = np.linalg.norm(whole_space_regular_residuals, ord='fro')
#     whole_space_reservoir_residuals = reservoir_Vt - S[:len(S_quotient)] * Vt_permuted.T
#     whole_space_reservoir_residuals_2norm = np.linalg.norm(whole_space_reservoir_residuals, ord=2)
#     whole_space_reservoir_residuals_fro = np.linalg.norm(whole_space_reservoir_residuals, ord='fro')

#     whole_space_regular_residuals_quotient = regular_Vt - S_quotient * Vt.T
#     whole_space_regular_residuals_quotient_2norm = np.linalg.norm(whole_space_regular_residuals_quotient, ord=2)
#     whole_space_regular_residuals_quotient_fro = np.linalg.norm(whole_space_regular_residuals_quotient, ord='fro')
#     whole_space_reservoir_residuals_quotient = reservoir_Vt - S_quotient * Vt_permuted.T
#     whole_space_reservoir_residuals_quotient_2norm = np.linalg.norm(whole_space_reservoir_residuals_quotient, ord=2)
#     whole_space_reservoir_residuals_quotient_fro = np.linalg.norm(whole_space_reservoir_residuals_quotient, ord='fro')

#     # print(S[0], reservoir_residuals[0], Vt[0, -10:], A_csr[0, -10:])
#     # print(S[2], reservoir_residuals[2], Vt[2, -10:])
#     # print(S[4], reservoir_residuals[4], Vt[4, -10:])
#     # if "rs_50" in name:
#     # print(reservoir_residuals)
#     print("reservoir_residuals_quotient:", reservoir_residuals_quotient)
#     print(f"2: {whole_space_reservoir_residuals_quotient_2norm:.3f}, fro: {whole_space_reservoir_residuals_quotient_fro:.3f}")
#     print("regular_residuals_quotient:", regular_residuals_quotient)
#     print(f"2: {whole_space_regular_residuals_quotient_2norm:.3f}, fro: {whole_space_regular_residuals_quotient_fro:.3f}")
     
#     # if iteration == 9:
#         # print(regular_res[:10], reservoir_res)
#         # inv_perm = inverse_permutation(row_permutation)
#         # print(inv_perm[reservoir_idx])
    
    
    
#     # Save symmetric PSD data
#     np.savez(os.path.join(dir_path, f'reservoir_residuals_data_{iteration}.npz'),
#                 reservoir_residuals=reservoir_residuals,
#                 regular_residuals=regular_residuals,
#                 reservoir_residuals_quotient=reservoir_residuals_quotient,
#                 regular_residuals_quotient=regular_residuals_quotient,
#                 whole_space_regular_residuals_2norm=whole_space_regular_residuals_2norm,
#                 whole_space_regular_residuals_fro=whole_space_regular_residuals_fro,
#                 whole_space_reservoir_residuals_2norm=whole_space_reservoir_residuals_2norm,
#                 whole_space_reservoir_residuals_fro=whole_space_reservoir_residuals_fro,
#                 whole_space_regular_residuals_quotient_2norm=whole_space_regular_residuals_quotient_2norm,
#                 whole_space_regular_residuals_quotient_fro=whole_space_regular_residuals_quotient_fro,
#                 whole_space_reservoir_residuals_quotient_2norm=whole_space_reservoir_residuals_quotient_2norm,
#                 whole_space_reservoir_residuals_quotient_fro=whole_space_reservoir_residuals_quotient_fro,
#                 iteration=iteration,
#                 A_norm=A_norm,
#                 allow_pickle=True)
#     del reservoir_Vt, regular_Vt, reservoir_residuals, regular_residuals, reservoir_residuals_quotient, regular_residuals_quotient
#     gc.collect()
#     print_memory_usage(f"After residual reservoir, window {iteration+1}")
    
def compute_svd(A, k, is_sparse=True, Vt=None):
    is_sparse = True
    if not is_sparse:
        print("Small matrix")
        return sp.linalg.svd(A, lapack_driver="gesdd", full_matrices=False)
    else:
        # u, s, vt = primme.svds(A, k=min(k + k//5, min(A.shape) - 1), which='LM', v0=Vt.T if Vt is not None else None)
        u, s, vt = sp.sparse.linalg.svds(A, k=min(k+k//5, min(A.shape)-1))
        s = s[::-1]
        vt = vt[::-1, :]
        u = u[:, ::-1]
        if Vt is not None:
            del Vt
        return u,s,vt

def make_operator(window, N, n, w, V, S):
    def matvec(x):
        x = x.reshape(-1)
        result = np.zeros(min(n*w,N), dtype=np.float64)

        # First (n-1)w rows: V S V^T part
        result[:(n-1)*w] = V[:(n-1)*w, :] @ (S * (V[:(n-1)*w, :].T @ x[:(n-1)*w]))
        cross_term_first = window[:, :(n-1)*w].T @ x[(n-1)*w:n*w]
        result[:(n-1)*w] += cross_term_first.real

        # Last block + cross terms        
        try:
            result[(n-1)*w:n*w] = window[:, :n*w] @ x 
        except:
            raise
        return result
    
    size = min(n*w, N)
    return sp.sparse.linalg.LinearOperator((size, size), matvec=matvec, rmatvec=matvec, dtype=np.float64)


def create_svd_operators(V11, S11, S11_sqrt, S11_invsqrt, window, V, S, n, w, N):
    def svd_matvec(x):
        x = x.reshape(-1)
        # result = np.zeros(N, dtype=np.complex128)
        result = np.zeros(N, dtype=np.float64)
        
        v_proj = V11.T @ x
        scaled_proj = S11_sqrt * v_proj
        result[:n*w] = V11 @ scaled_proj
        
        scaled_proj_inv = S11_invsqrt * v_proj
        temp = V11 @ scaled_proj_inv
#         print(V[:(n-1)*w, :].T.shape, temp.shape, w )
        v_old_proj = V[n*w:, :] @ (np.diag(S) @ (V[:(n-1)*w, :].T @ temp[:(n-1)*w]))
        window_proj = window[:, n*w:].T @ temp[(n-1)*w:]
        result[n*w:] = v_old_proj + window_proj
        return result

    def svd_rmatvec(x):
        x1, x2 = x[:n*w], x[n*w:]
        
        Vn_t_x2 = V[n*w:, :].T @ x2
        scaled_Vn_t_x2 = np.diag(S) @ Vn_t_x2
        A12_upper_x2 = V[:(n-1)*w, :] @ scaled_Vn_t_x2
        A12_lower_x2 = window[:, n*w:] @ x2
        A12_x2 = np.concatenate([A12_upper_x2, A12_lower_x2])
        
        V11_t_x1 = V11.T @ x1
        V11_t_A12_x2 = V11.T @ A12_x2
        
        temp = (np.diag(S11_sqrt) @ V11_t_x1) + (np.diag(S11_invsqrt) @ V11_t_A12_x2)
        result = V11 @ temp
        
        return result

    return sp.sparse.linalg.LinearOperator((N, min(n*w, N)), matvec=svd_matvec, rmatvec=svd_rmatvec, dtype=np.float64) #complex128

def make_operator_true(n, w, A, N):
    def matvec(x):
        x = x.reshape(-1)
        result = A[:n*w, :n*w] @ x
        return result
    size = min(n*w, N)
    return sp.sparse.linalg.LinearOperator((size, size), matvec=matvec, rmatvec=matvec, dtype=np.float64)

def create_svd_operators_true(V11, S11, S11_sqrt, S11_invsqrt, window, A, n, w, N):
    def svd_matvec(x):
        x = x.reshape(-1)
        # result = np.zeros(N, dtype=np.complex128)
        result = np.zeros(N, dtype=np.float64)
        
        v_proj = V11.T @ x
        scaled_proj = S11_sqrt * v_proj
        result[:n*w] = V11 @ scaled_proj
        
        scaled_proj_inv = S11_invsqrt * v_proj
        temp = V11 @ scaled_proj_inv
#         print(V[:(n-1)*w, :].T.shape, temp.shape, w )
        result[n*w:] = A[:n*w, n*w:].T @ temp
        return result

    def svd_rmatvec(x):
        x1, x2 = x[:n*w], x[n*w:]
        
        # Vn_t_x2 = V[n*w:, :].T @ x2
        # scaled_Vn_t_x2 = np.diag(S) @ Vn_t_x2
        # A12_upper_x2 = V[:(n-1)*w, :] @ scaled_Vn_t_x2
        # A12_lower_x2 = window[:, n*w:] @ x2
        # A12_x2 = np.concatenate([A12_upper_x2, A12_lower_x2])
        A12_x2 = A[:n*w, n*w:] @ x2
        
        V11_t_x1 = V11.T @ x1
        V11_t_A12_x2 = V11.T @ A12_x2
        
        temp = (np.diag(S11_sqrt) @ V11_t_x1) + (np.diag(S11_invsqrt) @ V11_t_A12_x2)
        result = V11 @ temp
        
        return result

    return sp.sparse.linalg.LinearOperator((N, min(N, n*w)), matvec=svd_matvec, rmatvec=svd_rmatvec, dtype=np.float64) #complex128


def inverse_permutation(perm):
    inverse = [0] * len(perm)
    for i in range(len(perm)):
        inverse[perm[i]] = i
    return inverse

def compose_permutations(p1, p2):
    # Applies p2 first, then p1
    if p1 is None:
        return p2
    else:
        return [p1[p2[i]] for i in range(len(p1))]

def nystrom_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W, 
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path, 
              col_permutation, track_U, reservoir_size, 
              num_Vs, with_S, reverse,
              threshold_factor, track_reconstruction_error, reconstruction_errors,
              use_true_matrix, m, return_row_order,
              total_S_reduced,
              Vt=None, S=None, inverse_perm=None,  V_focus=None,
            ):
    next_window = next_window[:, row_permutation]
    if isinstance(A_csr, csr_matrix):
        next_window = next_window.toarray()
    

    # print(next_window.shape)
    if j == 0:
        # Initial SVD for the first window
        w = first_window_size

        S11, V11 = sp.linalg.eig(next_window[:, :first_window_size])
        idx = np.argsort(S11)[::-1]
        S11, V11 = S11[idx], V11[:, idx]

        # Initial SVD
        S11_sqrt = np.sqrt(np.abs(S11))
        S11_invsqrt = np.zeros_like(S11)
        mask = S11 > threshold_factor * w * np.finfo(float).eps * np.max(np.abs(S11))
        S11_invsqrt[mask] = 1 / np.sqrt(S11[mask])

        upper = V11 @ np.diag(S11_sqrt) @ V11.T
        lower = next_window[:, w:].T @ V11 @ np.diag(S11_invsqrt) @ V11.T
        aug_matrix = np.vstack([upper, lower])

        U, S, Vt = sp.linalg.svd(aug_matrix, full_matrices=False)
        idx = np.argsort(S)[::-1]
        # keep = len(idx) # for now
        V = U[:, idx[:k]]
        S = S[idx[:k]]**2

    else:
         

        n = j + 1  # Current number of windows being processed

        # Correct eigenvectors for current row perm
        # Inverts the previous permutation, then applies current row permutation
        # print(V.shape, len(row_permutation), len(inverse_perm))
         
        V = Vt.T
        V = V[compose_permutations(inverse_perm, row_permutation), :]  
        print("Before recalculating:", (V[j*first_window_size:(j+1)*first_window_size,:] @ np.diag(S) @ V[:,:].T)[-3:,-5:])

        # Get new window
        current_window = next_window
        w = first_window_size
        N = A_csr.shape[0]

        if track_reconstruction_error:
            # TODO: check reconstruction of part of current_window
            reconstructed = V[:(n-1)*w, :] @ np.diag(S) @ V[(n-1)*w:n*w, :].T
            print(n, w, "Shapes:", reconstructed.shape)
            reconstruction_error = np.linalg.norm(current_window[:, :(n-1)*w] - reconstructed.T) / np.linalg.norm(current_window[:, :(n-1)*w])
            reconstruction_errors.append(reconstruction_error)
            print("Reconstruction error P_C:", reconstruction_error)
            np.savez(os.path.join(dir_path, f'reconstruction_error.npz'),
                    reconstruction_errors=np.array(reconstruction_errors),)

        # print("Window after:", prev_window[:, row_permutation][-3:,-5:], (V[(j-1)*first_window_size:(j)*first_window_size,:] @ np.diag(S) @ V[:,:].T)[-3:,-5:])
        # print("Norm after:", np.linalg.norm(prev_window[:, row_permutation] - V[(j-1)*first_window_size:(j)*first_window_size,:] @ np.diag(S) @ V[:,:].T))
        
         

        # Create operator for eigendecomposition
        if use_true_matrix:
            A11_op = make_operator_true(n, w, A_csr, N)
        else:
            A11_op = make_operator(current_window, N, n, w, V, S)
        if end_idx == m:
            keep = k
        else:
            keep = k #len(S) # for now
        S11, V11 = sp.sparse.linalg.eigs(A11_op, k=keep, which='LM', ncv=n*w - 1)

        if track_reconstruction_error:
            A11_reconstruct = V11 @ np.diag(S11) @ V11.T
            error = np.linalg.norm(A11_op.matmat(np.eye(A11_op.shape[0])) - A11_reconstruct, ord='fro')
            print("A11 reconstruction error:", error)        

        if end_idx == m:
            # If last window, stop
            S = S11[:k]
            V = V11[:, :k]
#                 print(f"Eigenvalues: {S}")
#                 print("Exact:", S_exact)
#                 print(f"V shape: {V.shape}")
            
        else:

            # Prepare for SVD
            S11_sqrt = np.sqrt(np.abs(S11))
            S11_invsqrt = np.zeros_like(S11)
            mask = S11 > threshold_factor * A11_op.shape[0] * np.finfo(float).eps * np.max(np.abs(S11))
            S11_invsqrt[mask] = 1 / np.sqrt(np.abs(S11[mask]))

            # Create SVD operator and compute decomposition
#                 print("V11.shape, S11.shape, S11_sqrt.shape, current_window.shape, V.shape, S.shape, n, w, N")
#                 print(V11.shape, S11.shape, S11_sqrt.shape, current_window.shape, V.shape, S.shape, n, w, N)

            if use_true_matrix:
                svd_op = create_svd_operators_true(V11, S11, S11_sqrt, S11_invsqrt, 
                                            current_window, A_csr, n, w, N)
            else:
                svd_op = create_svd_operators(V11, S11, S11_sqrt, S11_invsqrt, 
                                            current_window, V, S, n, w, N)
            U, S, Vt = sp.sparse.linalg.svds(svd_op, k=k, ncv=n*w - 1, which="LM")

            # Update V and S for next iteration
            idx = np.argsort(S)[::-1]
            S, V = S[idx]**2, U[:, idx]

        # if j == 4:
        #     temp = A11_op(np.ones(n*w)) - A_csr[:n*w, :n*w] @ np.ones(n*w)
        #     print("A11_op:", A11_op(np.ones(n*w))[:10], (A_csr[:n*w, :n*w] @ np.ones(n*w))[:10])
        #     G = np.vstack([(V11 @ np.diag(S11_sqrt) @ V11.T),
        #                    (np.vstack([V[:(n-1)*w, :] @ (np.diag(S) @ V[n*w:, :].T), 
        #                                current_window[:, n*w:]]).T @ V11 @ np.diag(S11_invsqrt) @ V11.T)])
        #     print("G:", (G@G.T)[:5, :5], A_csr[:5,:5])
        #     svd_G = np.linalg.svd(G.real, full_matrices=False)
        #     print(svd_G.S[:5]**2, S[:5])
         

    print(S[:10])
    print(S_exact[:10])

    print("V.shape, S.shape:", V.shape, S.shape)

    # invert row perm, compose with new row perm
    inverse_perm = inverse_permutation(row_permutation)
    temp = compose_permutations(row_permutation, inverse_permutation(row_permutation))
    # print("Inverse is correct:", [i for i in range(len(temp)) if i != temp[i]])
    print("Window before:", next_window[-3:,-5:], (V[j*first_window_size:(j+1)*first_window_size,:] @ np.diag(S) @ V[:,:].T)[-3:,-5:])
    print("Norm before:", np.linalg.norm(next_window - V[j*first_window_size:(j+1)*first_window_size,:] @ np.diag(S) @ V[:,:].T))
    # prev_window = A_csr[window_indices, :]
    # prev_permutation = row_permutation
     
#                 print(f"Eigenvalues: {S}")
#                 print(f"V shape: {V.shape}")

    # Plot
    # plot_spectrum_comparison(S, S_exact, 
    #                          A_norm, name, j, dir_path)
    # plot_residuals(A_csr, S, Vt, S_exact, Vt_exact, U_exact, 
    #                A_norm, name, j, dir_path, is_sym_psd) 
    # plot_canonical_angles(Vt, Vt_exact, 
    #                       j, dir_path)

    Vt = V[inverse_perm, :].T # VSV^T ~ A
    if num_Vs:
        if with_S:
            if reverse:
                indices = np.argsort(np.sum((Vt[:num_Vs, row_permutation[end_idx:]] * S[:num_Vs].reshape(-1,1))**2, axis=0)).reshape(-1)[::1]
            else:
                indices = np.argsort(np.sum((Vt[:num_Vs, row_permutation[end_idx:]] * S[:num_Vs].reshape(-1,1))**2, axis=0)).reshape(-1)[::-1]
        else:
            if reverse:
                indices = np.argsort(np.sum((Vt[:num_Vs, row_permutation[end_idx:]])**2, axis=0)).reshape(-1)[::1]
            else:
                indices = np.argsort(np.sum((Vt[:num_Vs, row_permutation[end_idx:]])**2, axis=0)).reshape(-1)[::-1]
#             print(indices)
        row_permutation[end_idx:] = row_permutation[end_idx:][indices] 

    # Plot
    save_spectrum_comparison(S+total_S_reduced, S_exact, 
                                A_norm, name, j, dir_path)
    save_residuals(A_csr, S+total_S_reduced, Vt, 
                    S_exact, A_norm, name, j, dir_path, is_sym_psd,
                    row_permutation, start_idx, end_idx) 

    if not Vt_exact is None:
        print("Reconstruction quality:", np.linalg.norm(Vt - Vt_exact[:Vt.shape[0], :], 'fro'))
        save_canonical_angles(Vt, Vt_exact, 
                                j, dir_path)
    if j == W - 1 and track_U and not U_exact is None and not is_sym_psd:
        print("No tracking U option available")
#             save_canonical_angles(U.T, U_exact.T, 
#                                   j, dir_path, additional_label="_U")
    

    if not S_exact is None:
        print("Relative error in S:", np.linalg.norm(S - S_exact[:Vt.shape[0]]) / A_norm)
    # X = np.linalg.pinv(Vt_exact[:Vt.shape[0],:].T) @ Vt.T 
    # Vt_reconstructed = Vt_exact[:Vt.shape[0],:].T @ X
    # print("Reconstruct Vt from Vt_exact:", np.linalg.norm(Vt.T - Vt_reconstructed, 'fro'))
    # print("Projection F-norm error:", np.linalg.norm(Vt.T @ Vt - Vt_exact[:Vt.shape[0], :].T @ Vt_exact[:Vt.shape[0], :], 'fro'))
    # print("Trace correlation", np.trace(Vt @ Vt_exact[:Vt.shape[0], :].T @ Vt_exact[:Vt.shape[0], :] @ Vt.T) / min(Vt.T.shape[1], Vt_exact[:Vt.shape[0], :].T.shape[1]))

    ret = [S, V.T, inverse_perm]
    if track_U:
        ret.append(U)
    # if track_discarded:
    #     ret.append(discarded_list)
    if return_row_order:
        ret.append(row_permutation)
    if total_S_reduced > 0:
        ret.append(total_S_reduced)
    return ret

def compute_eigenvector_error(A, s, v, v_tilde):
    # Ensure v and v_tilde are unit vectors
    v = v / np.linalg.norm(v)
    v_tilde = v_tilde / np.linalg.norm(v_tilde)
    
    # Compute the cosine of the angle between v and v_tilde
    cos_theta = np.dot(v, v_tilde)
    
    # Compute sin(theta)
    sin_theta = np.sqrt(1 - cos_theta**2)
    
    # Compute w (perpendicular component)
    # For small angles, we can approximate w as:
    w_unnormalized = v_tilde - cos_theta * v
    
    # Normalize w (it should be a unit vector)
    # For numerical stability, check if w_unnormalized is close to zero
    w_norm = np.linalg.norm(w_unnormalized)
    if w_norm < 1e-10:  # Numerical threshold
        # Vectors are nearly parallel
        return 0.0
    
    w = w_unnormalized / w_norm
    
    # Compute (Aw - sw)
    Aw_minus_sw = np.dot(A, w) - s * w
    
    # Compute the final result: (Aw - sw)·sin(θ)
    result = Aw_minus_sw * sin_theta
    
    # If you want the norm instead:
    result_norm = np.linalg.norm(result)
    
    return {
        'w': w,
        'sin_theta': sin_theta,
        'Aw_minus_sw': Aw_minus_sw,
        'result': result,
        'result_norm': result_norm
    }

# Optional: Soft thresholding function
def soft_thresholding(S, threshold=0):
    return np.maximum(S - threshold, 0)

def soft_thresholding_Ghashami(S, ):
    # Assuming S is descending order
    ret = np.sqrt(S**2 - S[-1]**2) 
    if np.sum(np.isnan(ret)) > 0:
        import pdb; pdb.set_trace()
    return ret

def isvd_partial_step_(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, Vt_exact,
              col_permutation, track_U, 
              track_discarded, discarded_list,
              reservoir_size, reservoir_idx, reservoir, reservoir_method,
              Vt=None, S=None, reserved=None,
              use_soft_threshold=False, use_Ghashami=False, dir_path="",
              save_in_text=True,):
    
    if not col_permutation is None:
        next_window = next_window[:, col_permutation]
    if isinstance(A_csr, csr_matrix):
        if min(*A_csr.shape) < 3e4:
            next_window = next_window.toarray()
            is_sparse = False
        else:
            is_sparse = True
    elif isinstance(A_csr, StreamingRBFKernel):
        if min(*A_csr.shape) < 3e4:
            is_sparse = False
        else:
            is_sparse = True
    elif max(*A_csr.shape) > 3e4:
        is_sparse = True
    else:
        is_sparse = False

    # import pdb;pdb.set_trace()
    print("Sparse:", is_sparse)
    
    # print(next_window.shape)
    
    if j == 0:
            # Initial SVD for the first window
        
        # Reverse the order to get largest singular values first
        # _, S, Vt = svds(next_window, k=r)
        # S = S[::-1]
        # Vt = Vt[::-1, :]
        
        # U_sketch, S, Vt = sp.linalg.svd(next_window, lapack_driver="gesdd", full_matrices=False)
        # U_sketch, S, Vt = sp.sparse.linalg.svds(next_window, k=k+k//5)
        
        del S, Vt
        gc.collect()
        print_memory_usage(f"Before, window {j+1}")
        start_time = time.time()
        combined = next_window
        U_sketch, S, Vt = compute_svd(combined, k, is_sparse=is_sparse, Vt=None)
        svd_time = time.time() - start_time
        print(f"SVD completed in {svd_time:.4f} seconds")
        if not track_U:
            del U_sketch
            gc.collect()

        print_memory_usage(f"After SVD, window {j+1}")
        
#             if track_discarded:
#                 print(l, S.shape, Vt.shape)
#                 discarded_list.append([S[l:], Vt[l:, :]])
#             print(S, Vt[0,:10])

        # # Test: get the last k instead
        # S = S[-k:]
        # Vt = Vt[-k:, :]

        S = S[:k]
        Vt = Vt[:k, :]
        if use_soft_threshold:
            if use_Ghashami:
                S = soft_thresholding_Ghashami(S)
            else:
                S = soft_thresholding(S, threshold=S[-1])
        
        # B = S.reshape(-1, 1) * Vt

        if track_U:
            U = U_sketch
         
    else:

        # Concatenate B[j-1] and the next window
        if isinstance(next_window, np.ndarray):
            combined = np.concatenate((S.reshape(-1, 1) * Vt, next_window), axis=0)
        else:
            combined = sp.sparse.vstack([S.reshape(-1, 1) * Vt, next_window])
        # 
        
        # U_sketch, S, Vt = sp.linalg.svd(combined, lapack_driver="gesdd", full_matrices=False)
        print("Computing SVD...")
        start_time = time.time()
        # del S, Vt
        gc.collect()
        print_memory_usage(f"Before, window {j+1}")
        print("k:", k)
        print("Vt shape before SVD:", Vt.shape if Vt is not None else "Vt is None")
        U_sketch, S, Vt = compute_svd(combined, k, is_sparse=is_sparse, Vt=Vt)
        gc.collect()
        svd_time = time.time() - start_time
        print(f"SVD completed in {svd_time:.4f} seconds")
        
        if not track_U:
            del U_sketch
            gc.collect()
        
        print_memory_usage(f"After SVD, window {j+1}")


        if track_discarded:
            print(f"Discarding: {S[first_window_size:].shape}/{S.shape}")
            discarded_list.append([S[first_window_size:], Vt[first_window_size:, :]])
        

        S = S[:k]
        Vt = Vt[:k, :]
        if use_soft_threshold:
            if use_Ghashami:
                S = soft_thresholding_Ghashami(S)
            else:
                S = soft_thresholding(S, threshold=S[-1])

        # Update B
        # B = S.reshape(-1, 1) * Vt
#             print("B", B[0,:10])

        if track_U:
                # Update U
            U_new = np.zeros((U.shape[0] + len(window_indices), U.shape[1] + len(window_indices)))
            U_new[:U.shape[0], :U.shape[1]] = U
            U_new[U.shape[0]:, U.shape[1]:] = np.eye(len(window_indices))
            U = U_new
            U = U @ U_sketch
#                 print("U", U.shape, U_sketch.shape)
            U = U[:, :k]
        
    if reservoir_size > 0:
        # Need to switch between reservoir strats here
        if reservoir_method == "uniform":
            if reserved is None:
                reservoir_idx = np.random.randint(0, end_idx, reservoir_size)
                 
                reservoir = next_window[reservoir_idx, :]
            else:
                for idx in range(start_idx, end_idx):
                    # Generate random index
                    temp = np.random.randint(0, idx + 1)
                    
                    # If j < s, replace element at position j
                    if temp < reservoir_size:
                        reservoir_idx[temp] = idx
                        reservoir[temp, :] = next_window[idx-start_idx, :]
        elif reservoir_method == "weighted":
            if reserved is None:
                reservoir_idx = np.random.randint(0, end_idx, reservoir_size)
                 
                reservoir = next_window[reservoir_idx, :]
            else:
                for idx in range(start_idx, end_idx):
                    # Generate random index
                    temp = np.random.randint(0, idx + 1)
                    
                    # If j < s, replace element at position j
                    if temp < reservoir_size:
                        reservoir_idx[temp] = idx
                        reservoir[temp, :] = next_window[idx-start_idx, :]
        elif reservoir_method == "greedy":
            # if j == 0:
            #     # Pre-sort the absolute values for each row of Vt in the window
            #     sorted_indices_by_row = []
            #     for row in range(k):
            #         # Sort window indices by descending absolute value for this row
            #         sorted_indices = np.argsort(-np.abs(Vt[row, window_indices]))
            #         sorted_indices_by_row.append(sorted_indices)
                
            #     # Now fill the reservoir using the pre-sorted indices
            #     for i in range(reservoir_size):
            #         row_idx = i % k       # Cycle through rows of Vt
            #         rank = i // k         # Which ranked element to select
                    
            #         # Get the pre-sorted indices for this row
            #         sorted_indices = sorted_indices_by_row[row_idx]
                    
            #         # Select the element with the appropriate rank
            #         if rank < len(sorted_indices):
            #             temp = sorted_indices[rank]
            #             reservoir_idx[i] = start_idx + temp
            #             reservoir[i, :] = next_window[temp]
            # else:
            #     # For each eigenvector, sort both window elements and reservoir elements together
            #     for row_idx in range(k):

            #         # Get current elements in reservoir belonging to this row
            #         row_reservoir_indices = [i for i in range(reservoir_size) if i % k == row_idx]
                    
            #         # Combine window indices with reservoir indices for this row
            #         combined_indices = np.concatenate([window_indices, row_permutation[reservoir_idx]])
                    
            #         # Sort by magnitude (descending)
            #         combined_magnitude = np.abs(Vt[row_idx, combined_indices])
            #         sorted_indices = np.argsort(-combined_magnitude)
                    
            #         # Keep track of which elements came from window vs reservoir
            #         is_from_window = np.concatenate([
            #             np.ones(len(window_indices), dtype=bool),
            #             np.zeros(len(reservoir_idx), dtype=bool)
            #         ])
                    
            #          
            #         # Fill this row's portion of the reservoir with top elements
            #         for rank, i in enumerate(row_reservoir_indices):
            #             if rank < len(sorted_indices):
            #                 idx = sorted_indices[rank]
            #                 if is_from_window[idx]:
            #                     # Element from window
            #                     window_idx = idx
            #                     reservoir_idx[i] = start_idx + window_idx
            #                     reservoir[i, :] = next_window[window_idx]
            #                 else:
            #                     # Element from reservoir
            #                     res_idx = idx - len(window_indices)
            #                     reservoir_idx[i] = reservoir_idx[res_idx]
            #                     reservoir[i, :] = reservoir[res_idx, :]
            if j == 0:
                # Pre-sort the absolute values for each row of Vt in the window
                sorted_indices_by_row = []
                for row in range(k):
                    # Sort window indices by descending absolute value for this row
                    sorted_indices = np.argsort(-np.abs(Vt[row, window_indices]))
                    sorted_indices_by_row.append(sorted_indices)
                
                # Track which elements are already selected
                selected_elements = set()
                
                # Now fill the reservoir using the pre-sorted indices
                i = 0  # Reservoir position counter
                row_idx = 0  # Start with first row
                rank = 0  # Start with highest ranked element
                # np.linalg.norm(A_csr[row_permutation[temp], :]-reservoir[9])
                # np.linalg.norm(next_window[temp]-reservoir[9])
                while i < reservoir_size and row_idx < k:
                    # Get the pre-sorted indices for this row
                    sorted_indices = sorted_indices_by_row[row_idx]
                    
                    # Find next unselected element for this row
                    while rank < len(sorted_indices):
                        temp = sorted_indices[rank]
                        element_idx = start_idx + temp
                        
                        # Check if this element is already selected
                        if element_idx not in selected_elements:
                            # Add to reservoir
                            reservoir_idx[i] = element_idx
                            if isinstance(next_window, np.ndarray):
                                reservoir[i, :] = next_window[temp]
                            else:
                                reservoir[i, :] = next_window[temp].toarray()
                            selected_elements.add(element_idx)
                            i += 1
                            break
                        
                        # Try next ranked element
                        rank += 1
                    
                    if rank >= len(sorted_indices):
                        break
                    
                    # Move to next row, reset rank if we've gone through all rows
                    row_idx = (row_idx + 1) % k
                    if row_idx == 0:
                        rank += 1  # Move to next rank for all rows
            else:
                # For subsequent iterations
                # Track which elements are already selected
                selected_elements = set()
                reservoir_copy = reservoir.copy()
                
                for row_idx in range(k):
                    # Get current elements in reservoir belonging to this row
                    row_reservoir_indices = {i:None for i in range(reservoir_size) if i % k == row_idx}
                    
                    # Combine window indices with reservoir indices 
                     
                    combined_indices = np.concatenate([window_indices, row_permutation[reservoir_idx]])
                    
                    # Sort by magnitude (descending)
                    combined_magnitude = np.abs(Vt[row_idx, combined_indices])
                    sorted_indices = np.argsort(-combined_magnitude)
                    
                    # Keep track of which elements came from window vs reservoir
                    is_from_window = np.concatenate([
                        np.ones(len(window_indices), dtype=bool),
                        np.zeros(len(reservoir_idx), dtype=bool)
                    ])
                    
                    # Fill this row's portion of the reservoir with top elements
                    rank = 0
                    for i in row_reservoir_indices:
                        # Find next unselected element
                        while rank < len(sorted_indices):
                            idx = sorted_indices[rank]
                            
                            if is_from_window[idx]:
                                # Element from window
                                window_idx = idx
                                element_idx = start_idx + window_idx
                                
                                # Check if already selected
                                if element_idx not in selected_elements:
                                    reservoir_idx[i] = element_idx
                                    if isinstance(next_window, np.ndarray):
                                        reservoir[i, :] = next_window[window_idx]
                                    else:
                                        reservoir[i, :] = next_window[window_idx].toarray()
                                    selected_elements.add(element_idx)
                                    break
                            else:
                                # Element from reservoir
                                res_idx = idx - len(window_indices)
                                element_idx = reservoir_idx[res_idx]
                                
                                # Check if already selected
                                if element_idx not in selected_elements:
                                    reservoir_idx[i] = element_idx
                                    reservoir[i,:] = reservoir_copy[res_idx, :]
                                    selected_elements.add(element_idx)
                                    break
                            
                            # Try next ranked element
                            rank += 1
                            
                        # If we couldn't find an unselected element, leave this position unchanged
                        if rank >= len(sorted_indices):
                            # Could handle this case differently - e.g., by trying elements from other rows
                            pass
                        
                        rank += 1  # Move to next rank for next iteration
            print("scores:")
            for row_idx in range(k):
                print(np.abs(Vt[row_idx, row_permutation[reservoir_idx[[j for j in range(reservoir_size) if j % Vt.shape[0] == row_idx]]]]), end=", ")
        elif reservoir_method == "current_window":
            reservoir_idx = np.arange(start_idx, end_idx)
            reservoir = next_window
        else:
            raise NotImplementedError 
        

    # np.linalg.norm(A_csr[row_permutation[reservoir_idx[0]]] - reservoir[0,:])
    print("Vt shape:", Vt.shape)
    if not Vt_exact is None:
        print("\nSubspace angles each eigenvector")
        print(np.sum((Vt @ Vt_exact[:Vt.shape[0], :].T) ** 2, axis=0))
        num_save_files = 50
        if j == 0 or (j * (num_save_files- 1)) // W != ((j - 1) * (num_save_files - 1)) // W:
            save_leftout(Vt, S, Vt_exact, combined, j, dir_path, save_in_text=save_in_text)
    del combined
    gc.collect()
    return Vt, S, reservoir, reservoir_idx 


def isvd_step_(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path,
              col_permutation, track_U,
              track_discarded, discarded_list,
              num_Vs, with_S, reverse, return_row_order,
              total_S_reduced,
              reservoir_size, reservoir_idx, reservoir, reservoir_method,
              Vt=None, S=None,  V_focus=None, reserved=None, adaptive_order_ours=False, return_ours=False,
              use_soft_threshold=False, use_Ghashami=False, save_in_text=True,
    ):
    step_start_time = time.time()
    block_rows = int(end_idx - start_idx)
    Vt, S, reservoir, reservoir_idx = isvd_partial_step_(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, Vt_exact,
              col_permutation, track_U,
              track_discarded, discarded_list,
              reservoir_size, reservoir_idx, reservoir, reservoir_method,
              Vt=Vt, S=S, reserved=reserved,
              use_soft_threshold=use_soft_threshold, use_Ghashami=use_Ghashami,
              dir_path=dir_path, save_in_text=save_in_text)
    rows_seen_after = int(end_idx)
    solve_elapsed = time.time() - step_start_time

    print("Vt shape:", Vt.shape)
    S_head = np.asarray(S)[: min(10, len(S))]
    print(f"iSVD active rank: {len(S)} / requested {k}, block rows={block_rows}, total rows={rows_seen_after}")
    print(f"iSVD basis solve time: {solve_elapsed:.2f}s")
    print(f"iSVD S (top {len(S_head)}):", S_head)
    isvd_diag = {
        "block_index_1based": int(j) + 1,
        "block_rows": block_rows,
        "rows_seen": rows_seen_after,
        "active_rank": int(len(S)),
        "requested_rank": int(k),
        "solve_time_s": float(solve_elapsed),
        "reservoir_size": int(reservoir_size),
        "S_head": np.asarray(S_head, dtype=np.float64).tolist(),
    }
    if S_exact is not None:
        S_exact_arr = np.asarray(S_exact, dtype=np.float64)
        S_exact_head = S_exact_arr[: len(S)]
        scale = float(S_exact_arr[0]) if S_exact_arr.size and S_exact_arr[0] != 0.0 else 1.0
        S_recovered = np.asarray(S, dtype=np.float64) * scale
        tr_S = float(np.sum(S_recovered))
        tr_S_exact = float(np.sum(S_exact_head))
        isvd_diag["trace_S_recovered"] = tr_S
        isvd_diag["trace_S_exact_topk"] = tr_S_exact
        if tr_S_exact != 0.0:
            isvd_diag["trace_relerr"] = (tr_S - tr_S_exact) / tr_S_exact
    print({"iSVD_diag": isvd_diag})

    # Compute with reservoir
    print_memory_usage(f"Before LS, window {j+1}")
    S_quotient = []
    for i in range(k):
        # np.linalg.norm(A_csr[row_permutation[0], :]-reservoir[0])
        # np.dot(Vt_exact[0, row_permutation[reservoir_idx]].T, A_csr[row_permutation[reservoir_idx], :] @ Vt_exact[0].T) / np.dot(Vt_exact[0, row_permutation[reservoir_idx]].T, Vt_exact[0, row_permutation[reservoir_idx]].T)
        S_truncated_Rayleigh = np.dot(Vt[i, row_permutation[reservoir_idx]].T, reservoir @ Vt[i].T)
        sq_norm_V = np.dot(Vt[i, row_permutation[reservoir_idx]].T, Vt[i, row_permutation[reservoir_idx]].T)
        #S_truncated_Rayleigh_full = np.dot(Vt[i, row_permutation[:end_idx]].T, A_csr[row_permutation[:end_idx], :] @ Vt[i].T)
        #sq_norm_V_full = np.dot(Vt[i, row_permutation[:end_idx]].T, Vt[i, row_permutation[:end_idx]].T)
        if sq_norm_V < 1e-16:
            S_truncated_Rayleigh = S[i]
        else:
            S_truncated_Rayleigh /= sq_norm_V
#                 S.append(S_truncated_Rayleigh)
        #if sq_norm_V_full == 0:
        #    S_truncated_Rayleigh_full = np.nan
        #else:
        #    S_truncated_Rayleigh_full /= sq_norm_V_full
        S_quotient.append(S_truncated_Rayleigh)
    S_quotient = np.array(S_quotient)

    print(S[:10])
    print(S_quotient[:10])
    print(S_exact[:10])
    print_memory_usage(f"After LS, window {j+1}")
     
    # Plot
    # plot_spectrum_comparison(S, S_exact, 
    #                          A_norm, name, j, dir_path)
    # plot_residuals(A_csr, S, Vt, S_exact, Vt_exact, U_exact, 
    #                A_norm, name, j, dir_path, is_sym_psd) 
    # plot_canonical_angles(Vt, Vt_exact, 
    #                       j, dir_path)

    if adaptive_order_ours:
        S_adaptive = S_quotient
    else:
        S_adaptive = S
    if num_Vs:
        if V_focus is None:
            if with_S:
                if reverse:
                    indices = np.argsort(np.sum((Vt[:num_Vs, row_permutation[end_idx:]] * S_adaptive[:num_Vs].reshape(-1,1))**2, axis=0)).reshape(-1)[::1]
                else:
                    indices = np.argsort(np.sum((Vt[:num_Vs, row_permutation[end_idx:]] * S_adaptive[:num_Vs].reshape(-1,1))**2, axis=0)).reshape(-1)[::-1]
            else:
                if reverse:
                    indices = np.argsort(np.sum((Vt[:num_Vs, row_permutation[end_idx:]])**2, axis=0)).reshape(-1)[::1]
                else:
                    indices = np.argsort(np.sum((Vt[:num_Vs, row_permutation[end_idx:]])**2, axis=0)).reshape(-1)[::-1]
#             print(indices)
        else:
            indices = np.argsort(np.sum((Vt[V_focus, row_permutation[end_idx:]])**2, axis=0)).reshape(-1)[::1]
        row_permutation[end_idx:] = row_permutation[end_idx:][indices]

    # Plot
    print_memory_usage(f"Before saving, window {j+1}")
    print("j:", j)
    num_save_files = 50
    if j == 0 or (j * (num_save_files- 1)) // W != ((j - 1) * (num_save_files - 1)) // W:
        save_spectrum_comparison(S+total_S_reduced, S_exact, 
                                    A_norm, name, j, dir_path, S_quotient=S_quotient, save_in_text=save_in_text)
        save_residuals(A_csr, S+total_S_reduced, Vt, 
                        S_exact, A_norm, name, j, dir_path, is_sym_psd,
                        row_permutation, start_idx, end_idx, save_in_text=save_in_text)
        if reservoir_size > 0:
            save_residuals_reservoir(reservoir, reservoir_idx, row_permutation,
                                        S, Vt, S_exact, A_norm, A_csr, S_quotient, 
                                        name, j, dir_path, save_in_text=save_in_text)
        
    # temp = compute_eigenvector_error(A_csr, S_exact[0], Vt_exact[0,:], Vt[0,:])
    # print(temp['result_norm'])
     
    print_memory_usage(f"Before canonical angles, window {j+1}")
    if not Vt_exact is None:
        print("Reconstruction quality:", np.linalg.norm(Vt - Vt_exact[:Vt.shape[0], :], 'fro'))
        num_save_files = 50
        if j == 0 or (j * (num_save_files- 1)) // W != ((j - 1) * (num_save_files - 1)) // W:
            save_canonical_angles(Vt, Vt_exact, 
                                    j, dir_path, save_in_text=save_in_text)
    if j == W - 1 and track_U and not U_exact is None and not is_sym_psd:
        num_save_files = 50
        if j == 0 or (j * (num_save_files- 1)) // W != ((j - 1) * (num_save_files - 1)) // W:
            save_canonical_angles(U.T, U_exact.T, 
                                    j, dir_path, additional_label="_U", save_in_text=save_in_text)
    print_memory_usage(f"After canonical angles, window {j+1}")

    if not S_exact is None:
        print("Relative error in S:", np.linalg.norm(S - S_exact[:Vt.shape[0]]) / A_norm)
        print("Relative error in S_quotient:", np.linalg.norm(S_quotient - S_exact[:Vt.shape[0]]) / A_norm)
    # X = np.linalg.pinv(Vt_exact[:Vt.shape[0],:].T) @ Vt.T 
    # Vt_reconstructed = Vt_exact[:Vt.shape[0],:].T @ X
    # print("Reconstruct Vt from Vt_exact:", np.linalg.norm(Vt.T - Vt_reconstructed, 'fro'))
    # print("Projection F-norm error:", np.linalg.norm(Vt.T @ Vt - Vt_exact[:Vt.shape[0], :].T @ Vt_exact[:Vt.shape[0], :], 'fro'))
    # print("Trace correlation", np.trace(Vt @ Vt_exact[:Vt.shape[0], :].T @ Vt_exact[:Vt.shape[0], :] @ Vt.T) / min(Vt.T.shape[1], Vt_exact[:Vt.shape[0], :].T.shape[1]))
    
    print_memory_usage(f"After saving, window {j+1}")
    # del U_sketch
    # gc.collect()
    # print_memory_usage(f"After collection, window {j+1}")

    if return_ours:
        ret = [S_quotient, Vt, reserved]
    else:
        ret = [S, Vt, reserved]
    if reservoir_size > 0:
        ret.append((reservoir_idx, reservoir))
    if track_U:
        ret.append(U)
    if track_discarded:
        ret.append(discarded_list)
    if return_row_order:
        ret.append(row_permutation)
    if total_S_reduced > 0:
        ret.append(total_S_reduced)
    return ret

def isvd_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path, 
              col_permutation, track_U, 
              track_discarded, discarded_list,
              num_Vs, with_S, reverse, return_row_order,
              total_S_reduced,
              reservoir_size, reservoir_idx, reservoir, reservoir_method,
              Vt=None, S=None,  V_focus=None, reserved=None, 
              use_soft_threshold=False, use_Ghashami=False,
              save_in_text=True,
    ):
    return isvd_step_(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path, 
              col_permutation, track_U, 
              track_discarded, discarded_list,
              num_Vs, with_S, reverse, return_row_order,
              total_S_reduced,
              reservoir_size, reservoir_idx, reservoir, reservoir_method,
              Vt=Vt, S=S,  V_focus=V_focus, reserved=reserved, adaptive_order_ours=False, return_ours=False,
              use_soft_threshold=use_soft_threshold, use_Ghashami=use_Ghashami, save_in_text=save_in_text,
    )

def isvd_ls_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path, 
              col_permutation, track_U, 
              track_discarded, discarded_list,
              num_Vs, with_S, reverse, return_row_order,
              total_S_reduced,
              reservoir_size, reservoir_idx, reservoir, reservoir_method,
              Vt=None, S=None,  V_focus=None, reserved=None, 
    ):
    return isvd_step_(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path, 
              col_permutation, track_U, 
              track_discarded, discarded_list,
              num_Vs, with_S, reverse, return_row_order,
              total_S_reduced,
              reservoir_size, reservoir_idx, reservoir, reservoir_method,
              Vt=Vt, S=S,  V_focus=V_focus, reserved=reserved, adaptive_order_ours=True, return_ours=False
    )

def isvd_ls_2_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path, 
              col_permutation, track_U, 
              track_discarded, discarded_list,
              num_Vs, with_S, reverse, return_row_order,
              total_S_reduced,
              reservoir_size, reservoir_idx, reservoir, reservoir_method,
              Vt=None, S=None,  V_focus=None, reserved=None, 
    ):
    return isvd_step_(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path, 
              col_permutation, track_U, 
              track_discarded, discarded_list,
              num_Vs, with_S, reverse, return_row_order,
              total_S_reduced,
              reservoir_size, reservoir_idx, reservoir, reservoir_method,
              Vt=Vt, S=S,  V_focus=V_focus, reserved=reserved, adaptive_order_ours=True, return_ours=True
    )


def make_A_operator(window, n, w, Vt, S):
    def matvec(x):
        x = x.reshape(-1)
        result = np.zeros(Vt.shape[1], dtype=np.float64)

        # First (n-1)w rows: 
        result[:(n-1)*w] = Vt[:, :(n-1)*w].T @ (S * (Vt[:, :(n-1)*w] @ x[:(n-1)*w]))
        result[:(n-1)*w] += window[:, :(n-1)*w].T @ x[(n-1)*w:n*w]
        # try:
        result[:(n-1)*w] += Vt[:, :(n-1)*w].T @ (S * (Vt[:, n*w:] @ x[n*w:]))

        # Current window:
        result[(n-1)*w:n*w] += window @ x

        # Last n*w rows
        result[n*w:] = Vt[:, n*w:].T @ (S * (Vt[:, :(n-1)*w] @ x[:(n-1)*w]))
        result[n*w:] += window[:, n*w:].T @ x[(n-1)*w:n*w]
        result[n*w:] += Vt[:, n*w:].T @ (S * (Vt[:, n*w:] @ x[n*w:]))
        return result
    
    size = Vt.shape[1]
    return sp.sparse.linalg.LinearOperator((size, size), matvec=matvec, rmatvec=matvec, dtype=np.float64)





def make_A_operator(window, n, w, Vt, S):
    def matvec(x):
        x = x.reshape(-1)
        result = np.zeros(Vt.shape[1], dtype=np.float64)

        # First (n-1)w rows: 
        result[:(n-1)*w] = Vt[:, :(n-1)*w].T @ (S * (Vt[:, :(n-1)*w] @ x[:(n-1)*w]))
        result[:(n-1)*w] += window[:, :(n-1)*w].T @ x[(n-1)*w:n*w]
        # try:
        result[:(n-1)*w] += Vt[:, :(n-1)*w].T @ (S * (Vt[:, n*w:] @ x[n*w:]))

        # Current window:
        result[(n-1)*w:n*w] += window @ x

        # Last n*w rows
        result[n*w:] = Vt[:, n*w:].T @ (S * (Vt[:, :(n-1)*w] @ x[:(n-1)*w]))
        result[n*w:] += window[:, n*w:].T @ x[(n-1)*w:n*w]
        result[n*w:] += Vt[:, n*w:].T @ (S * (Vt[:, n*w:] @ x[n*w:]))
        return result
    
    size = Vt.shape[1]
    return sp.sparse.linalg.LinearOperator((size, size), matvec=matvec, rmatvec=matvec, dtype=np.float64)


def isvd_new_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path, 
              col_permutation, track_U, 
              track_discarded, discarded_list,
              num_Vs, with_S, reverse, return_row_order,
              total_S_reduced,
              reservoir_size, reservoir_idx, reservoir, inverse_perm,
              Vt=None, S=None, V_focus=None, 
    ):
    if not col_permutation is None:
        next_window = next_window[:, col_permutation]

    next_window = next_window[:, row_permutation]
    if isinstance(A_csr, csr_matrix):
        next_window = next_window.toarray()
    
    # B = S.reshape(-1, 1) * Vt
    
    Vt = Vt[:, compose_permutations(inverse_perm, row_permutation)]  
    # # Concatenate B[j-1] and the next window
    # combined = np.concatenate((B, next_window), axis=0)
    # np.linalg.norm(next_window - Vt.T[row_permutation[start_idx:end_idx],:] @ np.diag(S) @ Vt)
    print(np.linalg.norm(next_window - Vt.T[start_idx:end_idx,:] @ np.diag(S) @ Vt))
    # print(np.linalg.norm(next_window - Vt.T[row_permutation[start_idx:end_idx],:] @ np.diag(S) @ Vt[:, row_permutation]))
    n = j + 1
    Vt = Vt.real
    S = S.real
    A_op = make_A_operator(next_window, n, first_window_size, Vt, S)
    keep = k #len(S)
    S, V = sp.sparse.linalg.lobpcg(A_op, Vt.T, largest=True)
    
    # Sort by magnitude of eigenvalues
    idx = np.argsort(abs(S))[::-1]
    S = S[idx]
    V = V[:, idx]
    # S, V = sp.sparse.linalg.eigs(A_op, k=keep, which='LM', ncv=next_window.shape[0]-1)
    Vt = V.T
    
    # Perform SVD on the combined matrix
    # Reverse the order to get largest singular values first
    # _, S, Vt = svds(combined, k=r)
    # S = S[::-1]
    # Vt = Vt[::-1, :]
    
    # U_sketch, S, Vt = sp.linalg.svd(combined, lapack_driver="gesdd", full_matrices=False)
    # if track_discarded:
    #     print(f"Discarding: {S[first_window_size:].shape}/{S.shape}")
    #     discarded_list.append([S[first_window_size:], Vt[first_window_size:, :]])
    
    # Optional: Apply soft thresholding to singular values
    # S = soft_thresholding(S)
    # total_S_reduced += S[-1]
#             S = soft_thresholding_Ghashami(S)
#             S = soft_thresholding_SS(S)

    S = S[:k]
    Vt = Vt[:k, :]

    # Update B
    # B = S.reshape(-1, 1) * Vt
#             print("B", B[0,:10])

#     if track_U:
#             # Update U
#         U_new = np.zeros((U.shape[0] + len(window_indices), U.shape[1] + len(window_indices)))
#         U_new[:U.shape[0], :U.shape[1]] = U
#         U_new[U.shape[0]:, U.shape[1]:] = np.eye(len(window_indices))
#         U = U_new
#         U = U @ U_sketch
# #                 print("U", U.shape, U_sketch.shape)
#         U = U[:, :k]
    
    if reservoir_size > 0:
        for idx in range(start_idx, end_idx):
            # Generate random index
            temp = np.random.randint(0, idx + 1)
            
            # If j < s, replace element at position j
            if temp < reservoir_size:
                reservoir_idx[temp] = idx
                reservoir[temp, :] = next_window[idx-start_idx, :]
         

    # Recalculate S based on Vt
#     S_quotient = []
#     for i in range(k):
#         S_truncated_Rayleigh = np.dot(Vt[i, window_indices].T, A_csr[window_indices, :] @ Vt[i].T)
#         sq_norm_V = np.dot(Vt[i, window_indices].T, Vt[i, window_indices].T)
#         #S_truncated_Rayleigh_full = np.dot(Vt[i, row_permutation[:end_idx]].T, A_csr[row_permutation[:end_idx], :] @ Vt[i].T)
#         #sq_norm_V_full = np.dot(Vt[i, row_permutation[:end_idx]].T, Vt[i, row_permutation[:end_idx]].T)
#         if sq_norm_V == 0:
#             S_truncated_Rayleigh = S[i]
#         else:
#             S_truncated_Rayleigh /= sq_norm_V
# #                 S.append(S_truncated_Rayleigh)
#         #if sq_norm_V_full == 0:
#         #    S_truncated_Rayleigh_full = np.nan
#         #else:
#         #    S_truncated_Rayleigh_full /= sq_norm_V_full
#         S_quotient.append(S_truncated_Rayleigh)
#     S_quotient = np.array(S_quotient)
    print(S[:10])
#     print(S_quotient[:10])
    print(S_exact[:10])
     
    # Plot
    # plot_spectrum_comparison(S, S_exact, 
    #                          A_norm, name, j, dir_path)
    # plot_residuals(A_csr, S, Vt, S_exact, Vt_exact, U_exact, 
    #                A_norm, name, j, dir_path, is_sym_psd) 
    # plot_canonical_angles(Vt, Vt_exact, 
    #                       j, dir_path)

    inverse_perm = inverse_permutation(row_permutation)
    
     
    Vt = Vt[:, inverse_perm]
    if num_Vs:
        if with_S:
            if reverse:
                indices = np.argsort(np.sum((Vt[:num_Vs, row_permutation[end_idx:]] * S[:num_Vs].reshape(-1,1))**2, axis=0)).reshape(-1)[::1]
            else:
                indices = np.argsort(np.sum((Vt[:num_Vs, row_permutation[end_idx:]] * S[:num_Vs].reshape(-1,1))**2, axis=0)).reshape(-1)[::-1]
        else:
            if reverse:
                indices = np.argsort(np.sum((Vt[:num_Vs, row_permutation[end_idx:]])**2, axis=0)).reshape(-1)[::1]
            else:
                indices = np.argsort(np.sum((Vt[:num_Vs, row_permutation[end_idx:]])**2, axis=0)).reshape(-1)[::-1]
#             print(indices)
        row_permutation[end_idx:] = row_permutation[end_idx:][indices]
    Vt = Vt[:, row_permutation]

    # Plot
    save_spectrum_comparison(S+total_S_reduced, S_exact, 
                                A_norm, name, j, dir_path)
    save_residuals(A_csr, S+total_S_reduced, Vt, 
                    A_norm, name, j, dir_path, is_sym_psd,
                    row_permutation, start_idx, end_idx)
    # if reservoir_size > 0:
    #     save_residuals_reservoir(reservoir, reservoir_idx, row_permutation,
    #                                 S, Vt, A_norm, A_csr, S_quotient, 
    #                                 name, j, dir_path) 

    if not Vt_exact is None:
        print("Reconstruction quality:", np.linalg.norm(Vt - Vt_exact[:Vt.shape[0], :], 'fro'))
        save_canonical_angles(Vt, Vt_exact, 
                                j, dir_path)
    # if j == W - 1 and track_U and not U_exact is None and not is_sym_psd:
    #     save_canonical_angles(U.T, U_exact.T, 
    #                             j, dir_path, additional_label="_U")
    

    if not S_exact is None:
        print("Relative error in S:", np.linalg.norm(S - S_exact[:Vt.shape[0]]) / A_norm)
        # print("Relative error in S_quotient:", np.linalg.norm(S_quotient - S_exact[:Vt.shape[0]]) / A_norm)
    # X = np.linalg.pinv(Vt_exact[:Vt.shape[0],:].T) @ Vt.T 
    # Vt_reconstructed = Vt_exact[:Vt.shape[0],:].T @ X
    # print("Reconstruct Vt from Vt_exact:", np.linalg.norm(Vt.T - Vt_reconstructed, 'fro'))
    # print("Projection F-norm error:", np.linalg.norm(Vt.T @ Vt - Vt_exact[:Vt.shape[0], :].T @ Vt_exact[:Vt.shape[0], :], 'fro'))
    # print("Trace correlation", np.trace(Vt @ Vt_exact[:Vt.shape[0], :].T @ Vt_exact[:Vt.shape[0], :] @ Vt.T) / min(Vt.T.shape[1], Vt_exact[:Vt.shape[0], :].T.shape[1]))

    ret = [S, Vt, inverse_perm]
    if track_U:
        ret.append(U)
    if track_discarded:
        ret.append(discarded_list)
    if return_row_order:
        ret.append(row_permutation)
    if total_S_reduced > 0:
        ret.append(total_S_reduced)
    return ret


def isvd_1by1_new_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path, 
              col_permutation, track_U, 
              track_discarded, discarded_list,
              num_Vs, with_S, reverse, return_row_order,
              total_S_reduced,
              reservoir_size, reservoir_idx, reservoir, current_eigenvector_idx,
              Vt=None, S=None,  V_focus=None, reserved=None,
    ):
    if not col_permutation is None:
        next_window = next_window[:, col_permutation]
    if isinstance(A_csr, csr_matrix):
        next_window = next_window.toarray()
    
    # print(next_window.shape)
    # print("j inside:", j)
    if j == 0:
            # Initial SVD for the first window
        
        # Reverse the order to get largest singular values first
        # _, S, Vt = svds(next_window, k=r)
        # S = S[::-1]
        # Vt = Vt[::-1, :]
        
        U_sketch, S, Vt = sp.linalg.svd(next_window, lapack_driver="gesdd", full_matrices=False)
        
#             if track_discarded:
#                 print(l, S.shape, Vt.shape)
#                 discarded_list.append([S[l:], Vt[l:, :]])
#             print(S, Vt[0,:10])

        # # Test: get the last k instead
        # S = S[-k:]
        # Vt = Vt[-k:, :]

        S = S[:k]
        Vt = Vt[:k, :]
        
        B = S.reshape(-1, 1) * Vt

        if track_U:
            U = U_sketch
        
            # 
    else:
        B = S[current_eigenvector_idx:].reshape(-1, 1) * Vt[current_eigenvector_idx:, :]

        # Deflate next_window based on current eigenvector idx
        for i in range(current_eigenvector_idx):
            Av = next_window @ Vt[i, :]
            # TODO: Can be done in parallel
            for row in range(next_window.shape[0]):
                next_window[row, :] -= Av[row] * Vt[i,:]

        # Concatenate B[j-1] and the next window
        combined = np.concatenate((B, next_window), axis=0)
        
        # Perform SVD on the combined matrix
        # Reverse the order to get largest singular values first
        # _, S, Vt = svds(combined, k=r)
        # S = S[::-1]
        # Vt = Vt[::-1, :]
        
        U_sketch, S_new, Vt_new = sp.linalg.svd(combined, lapack_driver="gesdd", full_matrices=False)
        if track_discarded: 
            print(f"Discarding: {S[first_window_size:].shape}/{S.shape}")
            discarded_list.append([S[first_window_size:], Vt[first_window_size:, :]])
        
        # Optional: Apply soft thresholding to singular values
        # S = soft_thresholding(S)
        # total_S_reduced += S[-1]
#             S = soft_thresholding_Ghashami(S)
#             S = soft_thresholding_SS(S)

        # # Test: get the last k instead
        # S = S[:k]
        # Vt = Vt[:k, :]

        S[current_eigenvector_idx:] = S_new[:k-current_eigenvector_idx]
        Vt[current_eigenvector_idx:, :] = Vt_new[:k-current_eigenvector_idx, :]

        # Update B
        B = S.reshape(-1, 1) * Vt

        # if track_U:
        #     U_new = np.zeros((U.shape[0] + len(window_indices), U.shape[1] + len(window_indices)))
        #     U_new[:U.shape[0], :U.shape[1]] = U
        #     U_new[U.shape[0]:, U.shape[1]:] = np.eye(len(window_indices))
        #     U = U_new
        #     U = U @ U_sketch
        #     U = U[:, :k]

     
    # TODO: use other reservoir methods?
    # if reservoir_size > 0:
    #     # TODO: Threshold for whether we should compute residual using this vector (might be too small to compute)
    #     # May threshold based on the best percentage we can get from sampling like this
    #     threshold = 1e-5
    #      
    #     if j == 0:
    #         for i in range(reservoir_size):
    #             temp = np.argmax(np.abs(Vt[i, window_indices]))
    #             reservoir_idx[i] = start_idx + temp
    #             reservoir[i, :] = next_window[temp]
    #              
    #     else:
    #          
    #         for i in range(reservoir_size):
    #             evec_window_magnitude = np.abs(Vt[i, window_indices])
    #             temp = np.argmax(evec_window_magnitude)
    #             if evec_window_magnitude[temp] > np.abs(Vt[i, row_permutation[reservoir_idx[i]]]):
    #                 reservoir_idx[i] = start_idx + temp
    #                 reservoir[i, :] = next_window[temp]  

    if reservoir_size > 0:
         
        if j == 0:
            for i in range(reservoir_size):
                temp = np.argmax(np.abs(Vt[i, window_indices]))
                reservoir_idx[i] = start_idx + temp
                reservoir[i, :] = next_window[temp]
                 
        else:
             
            for i in range(reservoir_size):
                evec_window_magnitude = np.abs(Vt[i, np.concatenate([window_indices, reservoir_idx])])
                temp = np.argmax(evec_window_magnitude)
                if evec_window_magnitude[temp] > np.abs(Vt[i, row_permutation[reservoir_idx[i]]]):
                    reservoir_idx[i] = start_idx + temp if temp < window_indices.shape[0] else reservoir_idx[temp-window_indices.shape[0]] 
                    reservoir[i, :] = next_window[temp] if temp < window_indices.shape[0] else reservoir[temp-window_indices.shape[0]] 


    # Compute with reservoir
    S_quotient = []
    for i in range(k):
        reservoir_threshold = 1e-12
        mask = np.abs(reservoir @ Vt[i].T) > reservoir_threshold 
        S_truncated_Rayleigh = np.dot(Vt[i, row_permutation[reservoir_idx[mask]]].T, reservoir[mask] @ Vt[i].T)
        sq_norm_V = np.dot(Vt[i, row_permutation[reservoir_idx]].T, Vt[i, row_permutation[reservoir_idx]].T)
        if sq_norm_V < 1e-16:
            S_truncated_Rayleigh = S[i]
        else:
            S_truncated_Rayleigh /= sq_norm_V
        S_quotient.append(S_truncated_Rayleigh)
    S_quotient = np.array(S_quotient)

    print(S[:10])
    print(S_quotient[:10])
    print(S_exact[:10])
     
    # Plot
    # plot_spectrum_comparison(S, S_exact, 
    #                          A_norm, name, j, dir_path)
    # plot_residuals(A_csr, S, Vt, S_exact, Vt_exact, U_exact, 
    #                A_norm, name, j, dir_path, is_sym_psd) 
    # plot_canonical_angles(Vt, Vt_exact, 
    #                       j, dir_path)

    reservoir_residuals_quotient = []
    regular_residuals_quotient = []
    for i in range(len(S)):
        reservoir_res_quotient = reservoir @ Vt[i] - (S_quotient[i]) * Vt[i, row_permutation[reservoir_idx]]
        regular_res_quotient = A_csr @ Vt[i] - (S_quotient[i]) * Vt[i]
        reservoir_residuals_quotient.append(np.linalg.norm(reservoir_res_quotient))
        regular_residuals_quotient.append(np.linalg.norm(regular_res_quotient))
    reservoir_residuals_quotient = np.array(reservoir_residuals_quotient) / A_norm
    regular_residuals_quotient = np.array(regular_residuals_quotient) / A_norm

    print(reservoir_residuals_quotient)
    print(regular_residuals_quotient)

    temp = current_eigenvector_idx
     
    # for i in range(current_eigenvector_idx, Vt.shape[0]): 
        # row_scores = np.sum(np.abs(Vt[current_eigenvector_idx+1:, row_permutation[end_idx:]]) * S[current_eigenvector_idx+1:].reshape(-1,1), axis=0)
        # row_scores /= np.abs(Vt[current_eigenvector_idx, row_permutation[end_idx:]]) * S[current_eigenvector_idx]
        # row_permutation[end_idx:] = row_permutation[end_idx:][np.argsort(row_scores)]

    temp = (Vt[current_eigenvector_idx:, row_permutation] * S[current_eigenvector_idx:].reshape(-1,1)) 
    temp_sum = np.zeros((temp.shape[1]), dtype=float)
    for x in range(temp.shape[0]):
        for y in range(temp.shape[0]):
            # if i != 0 and j != 0 :
            if x != y:
                temp_sum += temp[x, :] * temp[y, :]
                # print((temp[i, :] * temp[j, :]).sum(), end=",")
     
    temp_sum[end_idx:].sort() # Need to sort end_idx down only
    if not np.isclose(temp_sum.sum(), 0):
        # "V is not orthonormal?" 
        raise Exception("V is not orthonormal?" )
    current_sum = temp_sum[:end_idx].sum()
    pos_idx, neg_idx = row_permutation.shape[0]-1, end_idx
    new_order = []
    current_sum_order = [current_sum]
    for i in range(end_idx, len(temp_sum)):
        if current_sum <= 0:
            current_sum += temp_sum[pos_idx]
            new_order.append(pos_idx)
            pos_idx -= 1
        else:
            current_sum += temp_sum[neg_idx]
            new_order.append(neg_idx)
            neg_idx += 1

        if pos_idx < neg_idx-1:
            raise Exception("Problem with indices")
        current_sum_order.append(current_sum)
     
    row_permutation[end_idx:] = row_permutation[new_order]
     

        # current_scores = np.sum(np.abs(Vt[current_eigenvector_idx+1:, row_permutation[:end_idx]]) * S[current_eigenvector_idx+1:].reshape(-1,1))
        # current_scores /= np.sum(np.abs(Vt[current_eigenvector_idx, row_permutation[:end_idx]]) * S[current_eigenvector_idx])

        # next_window_scores = np.sum(np.abs(Vt[current_eigenvector_idx+1:, row_permutation[:end_idx+first_window_size]]) * S[current_eigenvector_idx+1:].reshape(-1,1))
        # next_window_scores /= np.sum(np.abs(Vt[current_eigenvector_idx, row_permutation[:end_idx+first_window_size]]) * S[current_eigenvector_idx])
        # # i = 0
        # print("scores:", current_scores, next_window_scores)
         
        # if reservoir_residuals_quotient[i] < threshold or next_window_scores > current_scores: # TODO: Adjust to concentration type threshold
        #     current_eigenvector_idx += 1
        # else:
        #     break


    # TODO: Order rows based on current eigenvector idx
    # row_scores = np.sum(np.abs(Vt[current_eigenvector_idx+1:, row_permutation[end_idx:]]) * S[current_eigenvector_idx+1:].reshape(-1,1), axis=0)
    # row_scores /= np.abs(Vt[current_eigenvector_idx, row_permutation[end_idx:]]) * S[current_eigenvector_idx]
    # # print(row_scores, np.sum(np.abs(Vt[:, row_permutation[end_idx:]]) * S[:].reshape(-1,1), axis=0))
    
    # # temp = temp @ temp.T
     
    # row_permutation[end_idx:] = row_permutation[end_idx:][np.argsort(row_scores)]

    # Plot
    print("j:", j)
    save_spectrum_comparison(S+total_S_reduced, S_exact, 
                                A_norm, name, j, dir_path, S_quotient=S_quotient)
    save_residuals(A_csr, S+total_S_reduced, Vt, 
                    A_norm, name, j, dir_path, is_sym_psd,
                    row_permutation, start_idx, end_idx)
    if reservoir_size > 0:
        save_residuals_reservoir(reservoir, reservoir_idx, row_permutation,
                                    S, Vt, A_norm, A_csr, S_quotient, 
                                    name, j, dir_path) 

    if not Vt_exact is None:
        print("Reconstruction quality:", np.linalg.norm(Vt - Vt_exact[:Vt.shape[0], :], 'fro'))
        save_canonical_angles(Vt, Vt_exact, 
                                j, dir_path)
    # if j == W - 1 and track_U and not U_exact is None and not is_sym_psd:
    #     save_canonical_angles(U.T, U_exact.T, 
    #                             j, dir_path, additional_label="_U")
    

    if not S_exact is None:
        print("Relative error in S:", np.linalg.norm(S - S_exact[:Vt.shape[0]]) / A_norm)
        print("Relative error in S_quotient:", np.linalg.norm(S_quotient - S_exact[:Vt.shape[0]]) / A_norm)
    # X = np.linalg.pinv(Vt_exact[:Vt.shape[0],:].T) @ Vt.T 
    # Vt_reconstructed = Vt_exact[:Vt.shape[0],:].T @ X
    # print("Reconstruct Vt from Vt_exact:", np.linalg.norm(Vt.T - Vt_reconstructed, 'fro'))
    # print("Projection F-norm error:", np.linalg.norm(Vt.T @ Vt - Vt_exact[:Vt.shape[0], :].T @ Vt_exact[:Vt.shape[0], :], 'fro'))
    # print("Trace correlation", np.trace(Vt @ Vt_exact[:Vt.shape[0], :].T @ Vt_exact[:Vt.shape[0], :] @ Vt.T) / min(Vt.T.shape[1], Vt_exact[:Vt.shape[0], :].T.shape[1]))

    # if current_eigenvector_idx == k:
    #     print("Done")
    #     return S, Vt, current_eigenvector_idx

    ret = [S, Vt, current_eigenvector_idx]
    if reservoir_size > 0:
        ret.append((reservoir_idx, reservoir))
    if track_U:
        ret.append(U)
    if track_discarded:
        ret.append(discarded_list)
    if return_row_order:
        ret.append(row_permutation)
    if total_S_reduced > 0:
        ret.append(total_S_reduced)
    return ret


def isvd_1by1_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path, 
              col_permutation, track_U, 
              track_discarded, discarded_list,
              num_Vs, with_S, reverse, return_row_order,
              total_S_reduced,
              reservoir_size, reservoir_idx, reservoir, current_eigenvector_idx,
              Vt=None, S=None,  V_focus=None, reserved=None,
    ):
    if not col_permutation is None:
        next_window = next_window[:, col_permutation]
    if isinstance(A_csr, csr_matrix):
        next_window = next_window.toarray()
    
    # print(next_window.shape)
    # print("j inside:", j)
    if j == 0:
            # Initial SVD for the first window
        
        # Reverse the order to get largest singular values first
        # _, S, Vt = svds(next_window, k=r)
        # S = S[::-1]
        # Vt = Vt[::-1, :]
        
        U_sketch, S, Vt = sp.linalg.svd(next_window, lapack_driver="gesdd", full_matrices=False)
        
#             if track_discarded:
#                 print(l, S.shape, Vt.shape)
#                 discarded_list.append([S[l:], Vt[l:, :]])
#             print(S, Vt[0,:10])

        # # Test: get the last k instead
        # S = S[-k:]
        # Vt = Vt[-k:, :]

        S = S[:k]
        Vt = Vt[:k, :]
        
        B = S.reshape(-1, 1) * Vt

        if track_U:
            U = U_sketch
        
            # 
    else:
        B = S[current_eigenvector_idx:].reshape(-1, 1) * Vt[current_eigenvector_idx:, :]

        # Deflate next_window based on current eigenvector idx
        for i in range(current_eigenvector_idx):
            Av = next_window @ Vt[i, :]
            # TODO: Can be done in parallel
            for row in range(next_window.shape[0]):
                next_window[row, :] -= Av[row] * Vt[i,:]

        # Concatenate B[j-1] and the next window
        combined = np.concatenate((B, next_window), axis=0)
        
        # Perform SVD on the combined matrix
        # Reverse the order to get largest singular values first
        # _, S, Vt = svds(combined, k=r)
        # S = S[::-1]
        # Vt = Vt[::-1, :]
        
        U_sketch, S_new, Vt_new = sp.linalg.svd(combined, lapack_driver="gesdd", full_matrices=False)
        if track_discarded: 
            print(f"Discarding: {S[first_window_size:].shape}/{S.shape}")
            discarded_list.append([S[first_window_size:], Vt[first_window_size:, :]])
        
        # Optional: Apply soft thresholding to singular values
        # S = soft_thresholding(S)
        # total_S_reduced += S[-1]
#             S = soft_thresholding_Ghashami(S)
#             S = soft_thresholding_SS(S)

        # # Test: get the last k instead
        # S = S[:k]
        # Vt = Vt[:k, :]

         
        S[current_eigenvector_idx:] = S_new[:k-current_eigenvector_idx]
        Vt[current_eigenvector_idx:, :] = Vt_new[:k-current_eigenvector_idx, :]
         

        # Update B
        B = S.reshape(-1, 1) * Vt

        # if track_U:
        #     U_new = np.zeros((U.shape[0] + len(window_indices), U.shape[1] + len(window_indices)))
        #     U_new[:U.shape[0], :U.shape[1]] = U
        #     U_new[U.shape[0]:, U.shape[1]:] = np.eye(len(window_indices))
        #     U = U_new
        #     U = U @ U_sketch
        #     U = U[:, :k]

     
    # TODO: use other reservoir methods?
    # if reservoir_size > 0:
    #      
    #     if j == 0:
    #         for i in range(reservoir_size):
    #             temp = np.argmax(np.abs(Vt[i, window_indices]))
    #             reservoir_idx[i] = start_idx + temp
    #             reservoir[i, :] = next_window[temp]
    #              
    #     else:
    #          
    #         for i in range(reservoir_size):
    #             evec_window_magnitude = np.abs(Vt[i, window_indices])
    #             temp = np.argmax(evec_window_magnitude)
    #             if evec_window_magnitude[temp] > np.abs(Vt[i, row_permutation[reservoir_idx[i]]]):
    #                 reservoir_idx[i] = start_idx + temp
    #                 reservoir[i, :] = next_window[temp]  
    
    if reservoir_size > 0:
         
        if j == 0:
            for i in range(reservoir_size):
                temp = np.argmax(np.abs(Vt[i, window_indices]))
                reservoir_idx[i] = start_idx + temp
                reservoir[i, :] = next_window[temp]
                 
        else:
             
            for i in range(reservoir_size):
                 
                evec_window_magnitude = np.abs(Vt[i, np.concatenate([window_indices, reservoir_idx])])
                temp = np.argmax(evec_window_magnitude)
                if evec_window_magnitude[temp] > np.abs(Vt[i, row_permutation[reservoir_idx[i]]]):
                    reservoir_idx[i] = start_idx + temp if temp < window_indices.shape[0] else reservoir_idx[temp-window_indices.shape[0]] 
                    reservoir[i, :] = next_window[temp] if temp < window_indices.shape[0] else reservoir[temp-window_indices.shape[0]] 


    # Compute with reservoir
    S_quotient = []
    for i in range(k):
        reservoir_threshold = 1e-12
        mask = np.abs(reservoir @ Vt[i].T) > reservoir_threshold 
        S_truncated_Rayleigh = np.dot(Vt[i, row_permutation[reservoir_idx[mask]]].T, reservoir[mask, :] @ Vt[i].T)
        sq_norm_V = np.dot(Vt[i, row_permutation[reservoir_idx[mask]]].T, Vt[i, row_permutation[reservoir_idx[mask]]].T)
        if sq_norm_V < 1e-16:
            S_truncated_Rayleigh = S[i]
        else:
            S_truncated_Rayleigh /= sq_norm_V
        S_quotient.append(S_truncated_Rayleigh)
    S_quotient = np.array(S_quotient)

    print(S[:10])
    print(S_quotient[:10])
    print(S_exact[:10])
     
    # Plot
    # plot_spectrum_comparison(S, S_exact, 
    #                          A_norm, name, j, dir_path)
    # plot_residuals(A_csr, S, Vt, S_exact, Vt_exact, U_exact, 
    #                A_norm, name, j, dir_path, is_sym_psd) 
    # plot_canonical_angles(Vt, Vt_exact, 
    #                       j, dir_path)

    reservoir_residuals_quotient = []
    regular_residuals_quotient = []
    for i in range(len(S)):
        reservoir_res_quotient = reservoir @ Vt[i] - (S_quotient[i]) * Vt[i, row_permutation[reservoir_idx]]
        regular_res_quotient = A_csr @ Vt[i] - (S_quotient[i]) * Vt[i]
        reservoir_residuals_quotient.append(np.linalg.norm(reservoir_res_quotient))
        regular_residuals_quotient.append(np.linalg.norm(regular_res_quotient))
    reservoir_residuals_quotient = np.array(reservoir_residuals_quotient) / A_norm
    regular_residuals_quotient = np.array(regular_residuals_quotient) / A_norm

    print(reservoir_residuals_quotient)
    print(regular_residuals_quotient)

    temp = current_eigenvector_idx
     
    for i in range(current_eigenvector_idx, Vt.shape[0]): 

        row_scores = np.sum(np.abs(Vt[current_eigenvector_idx+1:, row_permutation[end_idx:]]) * S[current_eigenvector_idx+1:].reshape(-1,1), axis=0)
        row_scores /= np.abs(Vt[current_eigenvector_idx, row_permutation[end_idx:]]) * S[current_eigenvector_idx]
        row_permutation[end_idx:] = row_permutation[end_idx:][np.argsort(row_scores)]

        current_scores = np.sum(np.abs(Vt[current_eigenvector_idx+1:, row_permutation[:end_idx]]) * S[current_eigenvector_idx+1:].reshape(-1,1))
        current_scores /= np.sum(np.abs(Vt[current_eigenvector_idx, row_permutation[:end_idx]]) * S[current_eigenvector_idx])

        next_window_scores = np.sum(np.abs(Vt[current_eigenvector_idx+1:, row_permutation[:end_idx+first_window_size]]) * S[current_eigenvector_idx+1:].reshape(-1,1))
        next_window_scores /= np.sum(np.abs(Vt[current_eigenvector_idx, row_permutation[:end_idx+first_window_size]]) * S[current_eigenvector_idx])
        # i = 0
        print("ev idx:", i)
        print("scores:", current_scores, next_window_scores)
         
        # May threshold based on the best percentage we can get from sampling like this
        threshold = 1e-10
        if reservoir_residuals_quotient[i] < threshold or next_window_scores > current_scores: # TODO: Adjust to concentration type threshold
            current_eigenvector_idx += 1
        else:
            break


    # TODO: Order rows based on current eigenvector idx
    # row_scores = np.sum(np.abs(Vt[current_eigenvector_idx+1:, row_permutation[end_idx:]]) * S[current_eigenvector_idx+1:].reshape(-1,1), axis=0)
    # row_scores /= np.abs(Vt[current_eigenvector_idx, row_permutation[end_idx:]]) * S[current_eigenvector_idx]
    # # print(row_scores, np.sum(np.abs(Vt[:, row_permutation[end_idx:]]) * S[:].reshape(-1,1), axis=0))
    # temp = (Vt[current_eigenvector_idx:, :] * S[current_eigenvector_idx:].reshape(-1,1)) 
    # temp_sum = np.zeros((temp.shape[1]), dtype=float)
    # for i in range(temp.shape[0]):
    #     for j in range(temp.shape[0]):
    #         # if i != 0 and j != 0 :
    #         if i != j:
    #             temp_sum += temp[i, :] * temp[j, :]
    # # temp = temp @ temp.T
     
    # row_permutation[end_idx:] = row_permutation[end_idx:][np.argsort(row_scores)]

    # Plot
    print("j:", j)
    save_spectrum_comparison(S+total_S_reduced, S_exact, 
                                A_norm, name, j, dir_path, S_quotient=S_quotient)
    save_residuals(A_csr, S+total_S_reduced, Vt, 
                    A_norm, name, j, dir_path, is_sym_psd,
                    row_permutation, start_idx, end_idx)
    if reservoir_size > 0:
        save_residuals_reservoir(reservoir, reservoir_idx, row_permutation,
                                    S, Vt, A_norm, A_csr, S_quotient, 
                                    name, j, dir_path) 

    if not Vt_exact is None:
        print("Reconstruction quality:", np.linalg.norm(Vt - Vt_exact[:Vt.shape[0], :], 'fro'))
        save_canonical_angles(Vt, Vt_exact, 
                                j, dir_path)
    # if j == W - 1 and track_U and not U_exact is None and not is_sym_psd:
    #     save_canonical_angles(U.T, U_exact.T, 
    #                             j, dir_path, additional_label="_U")
    

    if not S_exact is None:
        print("Relative error in S:", np.linalg.norm(S - S_exact[:Vt.shape[0]]) / A_norm)
        print("Relative error in S_quotient:", np.linalg.norm(S_quotient - S_exact[:Vt.shape[0]]) / A_norm)
    # X = np.linalg.pinv(Vt_exact[:Vt.shape[0],:].T) @ Vt.T 
    # Vt_reconstructed = Vt_exact[:Vt.shape[0],:].T @ X
    # print("Reconstruct Vt from Vt_exact:", np.linalg.norm(Vt.T - Vt_reconstructed, 'fro'))
    # print("Projection F-norm error:", np.linalg.norm(Vt.T @ Vt - Vt_exact[:Vt.shape[0], :].T @ Vt_exact[:Vt.shape[0], :], 'fro'))
    # print("Trace correlation", np.trace(Vt @ Vt_exact[:Vt.shape[0], :].T @ Vt_exact[:Vt.shape[0], :] @ Vt.T) / min(Vt.T.shape[1], Vt_exact[:Vt.shape[0], :].T.shape[1]))

    # if current_eigenvector_idx == k:
    #     print("Done")
    #     return S, Vt, current_eigenvector_idx

    ret = [S, Vt, current_eigenvector_idx]
    if reservoir_size > 0:
        ret.append((reservoir_idx, reservoir))
    if track_U:
        ret.append(U)
    if track_discarded:
        ret.append(discarded_list)
    if return_row_order:
        ret.append(row_permutation)
    if total_S_reduced > 0:
        ret.append(total_S_reduced)
    return ret

import psutil

def print_memory_usage(label):
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"{label}: RSS={mem_info.rss/1024/1024:.1f}MB, VMS={mem_info.vms/1024/1024:.1f}MB")


import gc

def compute_eig(A, k):
    if min(A.shape[0], A.shape[1]) <= 100:
        return sp.linalg.eig(A, lapack_driver="gesdd", full_matrices=False)
    else:
        u, s, vt = sp.sparse.linalg.eigs(A, k=k+k//5)
        s = s[::-1]
        vt = vt[::-1, :]
        u = u[:, ::-1]
        return u,s,vt
    
    inf_mask = np.isinf(S_quotient)
    S_quotient[inf_mask] = -np.inf  # Temporarily replace inf with -inf for sorting

    # Get indices that would sort eigenvalues in descending order
    idx = np.argsort(-S_quotient)  # Negative sign for descending order

    # Apply sorting to eigenvalues and eigenvectors
    S_quotient = S_quotient[idx]
    S_quotient[inf_mask] = np.inf
    V_coeffs = V_coeffs[:, idx]

def isvd_demix_step_(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path, 
              col_permutation, track_U, 
              track_discarded, discarded_list,
              num_Vs, with_S, reverse, return_row_order,
              total_S_reduced,
              reservoir_size, reservoir_idx, reservoir, reservoir_method, 
              Vt=None, S=None,  V_focus=None, reserved=None, adaptive_order_ours=False, return_ours=False,
              use_soft_threshold=False, use_Ghashami=False,
    ):
#     if not col_permutation is None:
#         next_window = next_window[:, col_permutation]
#     if isinstance(A_csr, csr_matrix):
#         if min(*A_csr.shape) < 3e4:
#             next_window = next_window.toarray()
#             is_sparse = False
#         else:
#             is_sparse = True
#     else:
#         is_sparse = False

#     print("Sparse:", is_sparse)
    
#     # print(next_window.shape)
#     print_memory_usage(f"Before, window {j+1}")
#     if j == 0:
#             # Initial SVD for the first window
        
#         # Reverse the order to get largest singular values first
#         # _, S, Vt = svds(next_window, k=r)
#         # S = S[::-1]
#         # Vt = Vt[::-1, :]
        
#         # U_sketch, S, Vt = sp.linalg.svd(next_window, lapack_driver="gesdd", full_matrices=False)
#         # U_sketch, S, Vt = sp.sparse.linalg.svds(next_window, k=k+k//5)
#         U_sketch, S, Vt = compute_svd(next_window, k, is_sparse=is_sparse)
#         print_memory_usage(f"After SVD, window {j+1}")
#         
# #             if track_discarded:
# #                 print(l, S.shape, Vt.shape)
# #                 discarded_list.append([S[l:], Vt[l:, :]])
# #             print(S, Vt[0,:10])

#         # # Test: get the last k instead
#         # S = S[-k:]
#         # Vt = Vt[-k:, :]

#         S = S[:k]
#         Vt = Vt[:k, :]
#         if use_soft_threshold:
#             if use_Ghashami:
#                 S = soft_thresholding_Ghashami(S)
#             else:
#                 S = soft_thresholding(S, threshold=S[-1])
        
#         B = S.reshape(-1, 1) * Vt

#         if track_U:
#             U = U_sketch
         
#     else:
#         B = S.reshape(-1, 1) * Vt

#         # Concatenate B[j-1] and the next window
#         
#         if isinstance(next_window, np.ndarray):
#             combined = np.concatenate((B, next_window), axis=0)
#         else:
#             combined = sp.sparse.vstack([B, next_window])
#         # 
        

#         # U_sketch, S, Vt = sp.linalg.svd(combined, lapack_driver="gesdd", full_matrices=False)
#         print("Computing SVD...")
#         start_time = time.time()
#         U_sketch, S, Vt = compute_svd(combined, k)
#         svd_time = time.time() - start_time
#         print(f"SVD completed in {svd_time:.4f} seconds")
#         print_memory_usage(f"After SVD, window {j+1}")
#         


#         if track_discarded:
#             print(f"Discarding: {S[first_window_size:].shape}/{S.shape}")
#             discarded_list.append([S[first_window_size:], Vt[first_window_size:, :]])
        

#         S = S[:k]
#         Vt = Vt[:k, :]
#         if use_soft_threshold:
#             if use_Ghashami:
#                 S = soft_thresholding_Ghashami(S)
#             else:
#                 S = soft_thresholding(S, threshold=S[-1])

#         # Update B
#         B = S.reshape(-1, 1) * Vt
# #             print("B", B[0,:10])

#         if track_U:
#                 # Update U
#             U_new = np.zeros((U.shape[0] + len(window_indices), U.shape[1] + len(window_indices)))
#             U_new[:U.shape[0], :U.shape[1]] = U
#             U_new[U.shape[0]:, U.shape[1]:] = np.eye(len(window_indices))
#             U = U_new
#             U = U @ U_sketch
# #                 print("U", U.shape, U_sketch.shape)
#             U = U[:, :k]
        
#     if reservoir_size > 0:
#         # Need to switch between reservoir strats here
#         if reservoir_method == "uniform":
#             if reserved is None:
#                 reservoir_idx = np.random.randint(0, end_idx, reservoir_size)
                 
#                 reservoir = next_window[reservoir_idx, :]
#             else:
#                 for idx in range(start_idx, end_idx):
#                     # Generate random index
#                     temp = np.random.randint(0, idx + 1)
                    
#                     # If j < s, replace element at position j
#                     if temp < reservoir_size:
#                         reservoir_idx[temp] = idx
#                         reservoir[temp, :] = next_window[idx-start_idx, :]
#         elif reservoir_method == "weighted":
#             if reserved is None:
#                 reservoir_idx = np.random.randint(0, end_idx, reservoir_size)
                 
#                 reservoir = next_window[reservoir_idx, :]
#             else:
#                 for idx in range(start_idx, end_idx):
#                     # Generate random index
#                     temp = np.random.randint(0, idx + 1)
                    
#                     # If j < s, replace element at position j
#                     if temp < reservoir_size:
#                         reservoir_idx[temp] = idx
#                         reservoir[temp, :] = next_window[idx-start_idx, :]
#         elif reservoir_method == "greedy":
#             # if j == 0:
#             #     # Pre-sort the absolute values for each row of Vt in the window
#             #     sorted_indices_by_row = []
#             #     for row in range(k):
#             #         # Sort window indices by descending absolute value for this row
#             #         sorted_indices = np.argsort(-np.abs(Vt[row, window_indices]))
#             #         sorted_indices_by_row.append(sorted_indices)
                
#             #     # Now fill the reservoir using the pre-sorted indices
#             #     for i in range(reservoir_size):
#             #         row_idx = i % k       # Cycle through rows of Vt
#             #         rank = i // k         # Which ranked element to select
                    
#             #         # Get the pre-sorted indices for this row
#             #         sorted_indices = sorted_indices_by_row[row_idx]
                    
#             #         # Select the element with the appropriate rank
#             #         if rank < len(sorted_indices):
#             #             temp = sorted_indices[rank]
#             #             reservoir_idx[i] = start_idx + temp
#             #             reservoir[i, :] = next_window[temp]
#             # else:
#             #     # For each eigenvector, sort both window elements and reservoir elements together
#             #     for row_idx in range(k):

#             #         # Get current elements in reservoir belonging to this row
#             #         row_reservoir_indices = [i for i in range(reservoir_size) if i % k == row_idx]
                    
#             #         # Combine window indices with reservoir indices for this row
#             #         combined_indices = np.concatenate([window_indices, row_permutation[reservoir_idx]])
                    
#             #         # Sort by magnitude (descending)
#             #         combined_magnitude = np.abs(Vt[row_idx, combined_indices])
#             #         sorted_indices = np.argsort(-combined_magnitude)
                    
#             #         # Keep track of which elements came from window vs reservoir
#             #         is_from_window = np.concatenate([
#             #             np.ones(len(window_indices), dtype=bool),
#             #             np.zeros(len(reservoir_idx), dtype=bool)
#             #         ])
                    
#             #          
#             #         # Fill this row's portion of the reservoir with top elements
#             #         for rank, i in enumerate(row_reservoir_indices):
#             #             if rank < len(sorted_indices):
#             #                 idx = sorted_indices[rank]
#             #                 if is_from_window[idx]:
#             #                     # Element from window
#             #                     window_idx = idx
#             #                     reservoir_idx[i] = start_idx + window_idx
#             #                     reservoir[i, :] = next_window[window_idx]
#             #                 else:
#             #                     # Element from reservoir
#             #                     res_idx = idx - len(window_indices)
#             #                     reservoir_idx[i] = reservoir_idx[res_idx]
#             #                     reservoir[i, :] = reservoir[res_idx, :]
#             if j == 0:
#                 # Pre-sort the absolute values for each row of Vt in the window
#                 sorted_indices_by_row = []
#                 for row in range(k):
#                     # Sort window indices by descending absolute value for this row
#                     sorted_indices = np.argsort(-np.abs(Vt[row, window_indices]))
#                     sorted_indices_by_row.append(sorted_indices)
                
#                 # Track which elements are already selected
#                 selected_elements = set()
                
#                 # Now fill the reservoir using the pre-sorted indices
#                 i = 0  # Reservoir position counter
#                 row_idx = 0  # Start with first row
#                 rank = 0  # Start with highest ranked element
                
#                 while i < reservoir_size and row_idx < k:
#                     # Get the pre-sorted indices for this row
#                     sorted_indices = sorted_indices_by_row[row_idx]
                    
#                     # Find next unselected element for this row
#                     while rank < len(sorted_indices):
#                         temp = sorted_indices[rank]
#                         element_idx = start_idx + temp
                        
#                         # Check if this element is already selected
#                         if element_idx not in selected_elements:
#                             # Add to reservoir
#                             reservoir_idx[i] = element_idx
#                             if isinstance(next_window, np.ndarray):
#                                 reservoir[i, :] = next_window[temp]
#                             else:
#                                 reservoir[i, :] = next_window[temp].toarray()
#                             selected_elements.add(element_idx)
#                             i += 1
#                             break
                        
#                         # Try next ranked element
#                         rank += 1
                    
#                     # Move to next row, reset rank if we've gone through all rows
#                     row_idx = (row_idx + 1) % k
#                     if row_idx == 0:
#                         rank += 1  # Move to next rank for all rows
#             else:
#                 # For subsequent iterations
#                 # Track which elements are already selected
#                 selected_elements = set()
                
#                 for row_idx in range(k):
#                     # Get current elements in reservoir belonging to this row
#                     row_reservoir_indices = [i for i in range(reservoir_size) if i % k == row_idx]
                    
#                     # Combine window indices with reservoir indices 
                     
#                     combined_indices = np.concatenate([window_indices, row_permutation[reservoir_idx]])
                    
#                     # Sort by magnitude (descending)
#                     combined_magnitude = np.abs(Vt[row_idx, combined_indices])
#                     sorted_indices = np.argsort(-combined_magnitude)
                    
#                     # Keep track of which elements came from window vs reservoir
#                     is_from_window = np.concatenate([
#                         np.ones(len(window_indices), dtype=bool),
#                         np.zeros(len(reservoir_idx), dtype=bool)
#                     ])
                    
#                     # Fill this row's portion of the reservoir with top elements
#                     rank = 0
#                     for i in row_reservoir_indices:
#                         # Find next unselected element
#                         while rank < len(sorted_indices):
#                             idx = sorted_indices[rank]
                            
#                             if is_from_window[idx]:
#                                 # Element from window
#                                 window_idx = idx
#                                 element_idx = start_idx + window_idx
                                
#                                 # Check if already selected
#                                 if element_idx not in selected_elements:
#                                     if isinstance(next_window, np.ndarray):
#                                         reservoir[i, :] = next_window[window_idx]
#                                     else:
#                                         reservoir[i, :] = next_window[window_idx].toarray()
#                                     selected_elements.add(element_idx)
#                                     break
#                             else:
#                                 # Element from reservoir
#                                 res_idx = idx - len(window_indices)
#                                 element_idx = reservoir_idx[res_idx]
                                
#                                 # Check if already selected
#                                 if element_idx not in selected_elements:
#                                     if isinstance(next_window, np.ndarray):
#                                         reservoir[i, :] = next_window[res_idx]
#                                     else:
#                                         reservoir[i, :] = next_window[res_idx].toarray()
#                                     selected_elements.add(element_idx)
#                                     break
                            
#                             # Try next ranked element
#                             rank += 1
                            
#                         # If we couldn't find an unselected element, leave this position unchanged
#                         if rank >= len(sorted_indices):
#                             # Could handle this case differently - e.g., by trying elements from other rows
#                             pass
                        
#                         rank += 1  # Move to next rank for next iteration
#             print("scores:")
#             for row_idx in range(k):
#                 print(np.abs(Vt[row_idx, row_permutation[reservoir_idx[[j for j in range(reservoir_size) if j % Vt.shape[0] == row_idx]]]]), end=", ")
#         elif reservoir_method == "current_window":
#             reservoir_idx = np.arange(start_idx, end_idx)
#             reservoir = next_window
#         else:
#             raise NotImplementedError
             
    Vt, S, reservoir, reservoir_idx = isvd_partial_step_(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, Vt_exact,
              col_permutation, track_U, 
              track_discarded, discarded_list,
              reservoir_size, reservoir_idx, reservoir, reservoir_method,
              Vt=Vt, S=S, reserved=reserved,
              use_soft_threshold=use_soft_threshold, use_Ghashami=use_Ghashami,)
    
    if reservoir.shape[0] >= Vt.shape[0]:
        # AV = reservoir @ Vt.T
        # S_quotient, V_coeffs = sp.linalg.eig(AV.T @ AV, AV.T @ Vt.T[row_permutation[reservoir_idx],:])
        
        print("Indexing temp array...")
        start_time = time.time()
        temp = Vt.T[row_permutation[reservoir_idx],:]
        indexing_time = time.time() - start_time
        print(f"Indexing completed in {indexing_time:.4f} seconds")
        
        print("QR decomposition...")
        start_time = time.time()
        Qw, Rw = np.linalg.qr(temp)
        qr_time = time.time() - start_time
        print(f"QR decomposition completed in {qr_time:.4f} seconds")
        
        # print("Pseudo-inverse...")
        # start_time = time.time()
        # Rw_pinv = np.linalg.pinv(Rw)
        # pinv_time = time.time() - start_time
        # print(f"Pseudo-inverse completed in {pinv_time:.4f} seconds")
        
        print("Matrix multiplication for A_new...")
        start_time = time.time()
        # A_new = Rw_pinv @ Qw.T @ reservoir @ Vt.T #1
        A_new = Qw.T @ reservoir @ Vt.T #2
        matmul_time = time.time() - start_time
        print(f"Matrix multiplication completed in {matmul_time:.4f} seconds")
        
        print("Eigenvalue decomposition...")
        start_time = time.time()
        S_quotient, V_coeffs = sp.linalg.eig(A_new, Rw)
        eig_time = time.time() - start_time
        print(f"Eigenvalue decomposition completed in {eig_time:.4f} seconds")
        
        # S_quotient, V_coeffs = sp.linalg.eig(Qw.T @ reservoir @ Vt.T, Rw) #2
        # S_quotient, V_coeffs = sp.linalg.eig((reservoir @ Vt.T).T @ reservoir @ Vt.T, (reservoir @ Vt.T).T @ Vt.T[row_permutation[reservoir_idx],:]) #3

    # elif reservoir.shape[0] == Vt.shape[0]:
    #     S_quotient, V_coeffs = sp.linalg.eig(reservoir @ Vt.T, Vt.T[row_permutation[reservoir_idx],:])
    else:
        raise Exception(f"Unforseen problem: reservoir shape {reservoir.shape}, Vt shape {Vt.shape}")

    
     
    # import scipy.io as sio
    # import os

    # # This script assumes you're running it from the pdb session or have access to these variables
    # # If running from pdb, you can use the following to execute this script:
    # # exec(open('export_to_matlab.py').read())
    # S_quotient_, V_coeffs_ = sp.linalg.eig((reservoir @ Vt.T).T @ reservoir @ Vt.T, (reservoir @ Vt.T).T @ Vt.T[row_permutation[reservoir_idx],:])

    # # Create a dictionary to store all the variables
    # matlab_vars = {}

    # # Store the main matrices
    # matlab_vars['reservoir'] = reservoir
    # matlab_vars['Vt'] = Vt
    # matlab_vars['row_permutation'] = row_permutation
    # matlab_vars['reservoir_idx'] = reservoir_idx

    # # Store the exact solutions if available
    # matlab_vars['S_exact'] = S_exact[:10]
    # matlab_vars['Vt_exact'] = Vt_exact[:10, :]

    # # Store the computed eigenvalues and eigenvectors
    # try:
    #     matlab_vars['S_quotient'] = S_quotient
    #     matlab_vars['V_coeffs'] = V_coeffs
    # except NameError:
    #     print("S_quotient or V_coeffs not defined")

    # try:
    #     matlab_vars['S_quotient_transformed'] = S_quotient_
    #     matlab_vars['V_coeffs_transformed'] = V_coeffs_
    # except NameError:
    #     print("S_quotient_ or V_coeffs_ not defined")

    # # Compute the matrices for the original generalized eigenvalue problem
    # A_original = reservoir @ Vt.T
    # B_original = Vt.T[row_permutation[reservoir_idx],:]
    # matlab_vars['A_original'] = A_original
    # matlab_vars['B_original'] = B_original

    # # Compute the matrices for the transformed generalized eigenvalue problem
    # A_transformed = (reservoir @ Vt.T).T @ reservoir @ Vt.T
    # B_transformed = (reservoir @ Vt.T).T @ Vt.T[row_permutation[reservoir_idx],:]
    # matlab_vars['A_transformed'] = A_transformed
    # matlab_vars['B_transformed'] = B_transformed

    # # Save to a .mat file
    # sio.savemat(f'matlab2/eigenproblem_data_it_{j}.mat', matlab_vars)
    # print(f"Variables exported to eigenproblem_data_it_{j}.mat")
    
    sorted_idx = np.argsort(S_quotient)[::-1]
    S_quotient = S_quotient[sorted_idx].real
    inf_mask = np.isinf(S_quotient)
    V_coeffs = V_coeffs[:, sorted_idx].real
    V_coeffs = V_coeffs[:, np.isfinite(S_quotient)]
    S_quotient = S_quotient[np.isfinite(S_quotient)]
    
    Vt_good_approx = V_coeffs.T @ Vt
    Vt_good_approx = np.linalg.qr(Vt_good_approx.T)[0].T

    print("Exact:", S_exact[:k])
    print("Approx:", S_quotient)
    print("Rela error:", np.abs(S_quotient - S_exact[:S_quotient.shape[0]]) / np.abs(S_exact[:S_quotient.shape[0]]))
    # sp.linalg.eig((reservoir[:10,:] @ Vt[:10,:].T).T @ reservoir[:10,:] @ Vt[:10,:].T, (reservoir[:10,:] @ Vt[:10,:].T).T @ Vt.T[row_permutation[reservoir_idx[:10]],:10])
    print("inner prod:", np.sum(Vt_good_approx * Vt_exact[:S_quotient.shape[0]], axis=1))
    # Vt_good_approx[0,:] @ Vt_exact[0, :]
    

    S_LS = []
    for i in range(k):
        reservoir_threshold = 1e-12
        mask = np.abs(reservoir @ Vt[i].T) > reservoir_threshold 
        S_truncated_Rayleigh = np.dot(Vt[i, row_permutation[reservoir_idx[mask]]].T, reservoir[mask] @ Vt[i].T)
        sq_norm_V = np.dot(Vt[i, row_permutation[reservoir_idx]].T, Vt[i, row_permutation[reservoir_idx]].T)
        if sq_norm_V < 1e-16:
            S_truncated_Rayleigh = S[i]
        else:
            S_truncated_Rayleigh /= sq_norm_V
        S_LS.append(S_truncated_Rayleigh)
    S_LS = np.array(S_LS)
    print("LS:", S_LS)
    print("LS relerr:", np.abs(S_LS - S_exact[:k]) / np.abs(S_exact[:k]))

    print("iSVD S:", S)
    print("iSVD S relerr:", np.abs(S - S_exact[:k]) / np.abs(S_exact[:k]))

    if adaptive_order_ours:
        Vt_adaptive = Vt_good_approx[~inf_mask].real
        S_adaptive = S_quotient[~inf_mask].real
    else:
        Vt_adaptive = Vt
        S_adaptive = S
    if num_Vs:
        if V_focus is None:
            if with_S:
                if reverse:
                    indices = np.argsort(np.sum((Vt_adaptive[:num_Vs, row_permutation[end_idx:]] * S_adaptive[:num_Vs].reshape(-1,1))**2, axis=0)).reshape(-1)[::1]
                else:
                    indices = np.argsort(np.sum((Vt_adaptive[:num_Vs, row_permutation[end_idx:]] * S_adaptive[:num_Vs].reshape(-1,1))**2, axis=0)).reshape(-1)[::-1]
            else:
                if reverse:
                    indices = np.argsort(np.sum((Vt_adaptive[:num_Vs, row_permutation[end_idx:]])**2, axis=0)).reshape(-1)[::1]
                else:
                    indices = np.argsort(np.sum((Vt_adaptive[:num_Vs, row_permutation[end_idx:]])**2, axis=0)).reshape(-1)[::-1]
#             print(indices)
        else:
            indices = np.argsort(np.sum((Vt[V_focus, row_permutation[end_idx:]])**2, axis=0)).reshape(-1)[::1]
        row_permutation[end_idx:] = row_permutation[end_idx:][indices]


    # Plot
    print("j:", j)
    save_spectrum_comparison(S+total_S_reduced, S_exact, 
                                A_norm, name, j, dir_path, S_quotient=S_quotient)
    save_residuals(A_csr, S+total_S_reduced, Vt, 
                    A_norm, name, j, dir_path, is_sym_psd,
                    row_permutation, start_idx, end_idx)
    if reservoir_size > 0:
        save_residuals_reservoir(reservoir, reservoir_idx, row_permutation,
                                    S, Vt_good_approx, A_norm, A_csr, S_quotient, 
                                    name, j, dir_path) 

    # temp = compute_eigenvector_error(A_csr, S_exact[0], Vt_exact[0,:], Vt[0,:])
    # print(temp['result_norm'])

    if not Vt_exact is None:
        print("Reconstruction quality:", np.linalg.norm(Vt_good_approx - Vt_exact[:Vt_good_approx.shape[0], :], 'fro'))
        save_canonical_angles(Vt_good_approx, Vt_exact, 
                                j, dir_path)
    if j == W - 1 and track_U and not U_exact is None and not is_sym_psd:
        save_canonical_angles(U.T, U_exact.T, 
                                j, dir_path, additional_label="_U")
     

    if not S_exact is None:
        print("Relative error in S:", np.linalg.norm(S - S_exact[:Vt.shape[0]]) / A_norm)
        print("Relative error in S_quotient:", np.linalg.norm(S_quotient - S_exact[:Vt_good_approx.shape[0]]) / A_norm)
    # X = np.linalg.pinv(Vt_exact[:Vt.shape[0],:].T) @ Vt.T 
    # Vt_reconstructed = Vt_exact[:Vt.shape[0],:].T @ X
    # print("Reconstruct Vt from Vt_exact:", np.linalg.norm(Vt.T - Vt_reconstructed, 'fro'))
    # print("Projection F-norm error:", np.linalg.norm(Vt.T @ Vt - Vt_exact[:Vt.shape[0], :].T @ Vt_exact[:Vt.shape[0], :], 'fro'))
    # print("Trace correlation", np.trace(Vt @ Vt_exact[:Vt.shape[0], :].T @ Vt_exact[:Vt.shape[0], :] @ Vt.T) / min(Vt.T.shape[1], Vt_exact[:Vt.shape[0], :].T.shape[1]))

    if return_ours:
        ret = [S_quotient[~inf_mask].real, Vt_good_approx[~inf_mask].real, reserved]
    else:
        ret = [S, Vt, reserved]

    if reservoir_size > 0:
        ret.append((reservoir_idx, reservoir))
    if track_U:
        ret.append(U)
    if track_discarded:
        ret.append(discarded_list)
    if return_row_order:
        ret.append(row_permutation)
    if total_S_reduced > 0:
        ret.append(total_S_reduced)
    return ret

def isvd_demix_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path, 
              col_permutation, track_U, 
              track_discarded, discarded_list,
              num_Vs, with_S, reverse, return_row_order,
              total_S_reduced,
              reservoir_size, reservoir_idx, reservoir, reservoir_method, 
              Vt=None, S=None,  V_focus=None, reserved=None, 
              use_soft_threshold=False, use_Ghashami=False,
    ):
    return isvd_demix_step_(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path, 
              col_permutation, track_U, 
              track_discarded, discarded_list,
              num_Vs, with_S, reverse, return_row_order,
              total_S_reduced,
              reservoir_size, reservoir_idx, reservoir, reservoir_method, 
              Vt=Vt, S=S,  V_focus=V_focus, reserved=reserved, 
              use_soft_threshold=use_soft_threshold, use_Ghashami=use_Ghashami,
        )

def isvd_demix_2_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path, 
              col_permutation, track_U, 
              track_discarded, discarded_list,
              num_Vs, with_S, reverse, return_row_order,
              total_S_reduced,
              reservoir_size, reservoir_idx, reservoir, reservoir_method, 
              Vt=None, S=None,  V_focus=None, reserved=None, 
    ):
    return isvd_demix_step_(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path, 
              col_permutation, track_U, 
              track_discarded, discarded_list,
              num_Vs, with_S, reverse, return_row_order,
              total_S_reduced,
              reservoir_size, reservoir_idx, reservoir, reservoir_method, 
              Vt=Vt, S=S,  V_focus=V_focus, reserved=reserved, adaptive_order_ours=True, return_ours=False,
        )

def isvd_demix_3_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path, 
              col_permutation, track_U, 
              track_discarded, discarded_list,
              num_Vs, with_S, reverse, return_row_order,
              total_S_reduced,
              reservoir_size, reservoir_idx, reservoir, reservoir_method, 
              Vt=None, S=None,  V_focus=None, reserved=None, 
    ):
    return isvd_demix_step_(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path, 
              col_permutation, track_U, 
              track_discarded, discarded_list,
              num_Vs, with_S, reverse, return_row_order,
              total_S_reduced,
              reservoir_size, reservoir_idx, reservoir, reservoir_method, 
              Vt=Vt, S=S,  V_focus=V_focus, reserved=reserved, adaptive_order_ours=True, return_ours=True,
        )





def kahan_sum(x):
    x = np.asarray(x).reshape(-1)
    s = 0.0
    c = 0.0
    for xi in x:
        y = float(xi) - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def project_feasible(x, Q, stats: Optional[TimeStats] = None):
    with timed(stats, "project_feasible"):
        x = np.asarray(x).reshape(-1)
        if Q is not None and Q.size:
            with timed(stats, "project_prev_basis"):
                x = x - Q @ (Q.T @ x)
        return np.ascontiguousarray(x)


def retract_feasible(x, Q, stats: Optional[TimeStats] = None):
    with timed(stats, "retract_feasible"):
        x = project_feasible(x, Q, stats=stats)
        nx = np.sqrt(kahan_sum(np.abs(x) ** 2))
        if nx <= 1e-14:
            return None
        return np.ascontiguousarray(x / nx, dtype=x.dtype)


def project_to_feasible_tangent(g, v, Q, stats: Optional[TimeStats] = None):
    with timed(stats, "project_tangent"):
        g_feas = np.asarray(g).reshape(-1)
        if Q is not None and Q.size:
            with timed(stats, "project_prev_basis"):
                g_feas = g_feas - Q @ (Q.T @ g_feas)
        with timed(stats, "project_current_vector"):
            g_feas = g_feas - v * float(v.T @ g_feas)
        return np.ascontiguousarray(g_feas)


def entropy_logscore_grad_rows(M, v, rows_total, ncols, stats: Optional[TimeStats] = None, mvstats: Optional[MatvecStats] = None):
    global _entropy_exact_operand_debug_printed
    with timed(stats, "entropy_logscore_grad_rows"):
        M_arr = np.asarray(M)
        M_dtype = M_arr.dtype
        v_arr = np.ascontiguousarray(np.asarray(v, dtype=M_dtype).reshape(-1))
        if not _entropy_exact_operand_debug_printed:
            v1 = np.asarray(v, dtype=M_arr.dtype).reshape(-1)
            t0 = time.perf_counter()
            out1 = M_arr @ v1
            t1 = time.perf_counter()
            u1 = np.asarray(out1, dtype=M_arr.dtype).reshape(-1)
            t2 = time.perf_counter()
            back1 = M_arr.T @ u1
            t3 = time.perf_counter()

            t4 = time.perf_counter()
            _ = M_arr @ v_arr
            t5 = time.perf_counter()
            _ = M_arr @ v_arr
            t6 = time.perf_counter()

            t7 = time.perf_counter()
            _ = M_arr.T @ u1
            t8 = time.perf_counter()
            _ = M_arr.T @ u1
            t9 = time.perf_counter()
            t10 = time.perf_counter()
            _ = M_arr @ v_arr
            t11 = time.perf_counter()
            t12 = time.perf_counter()
            _ = M_arr @ v1
            t13 = time.perf_counter()
        with timed(stats, "entropy_exact.forward_mv"):
            y_raw = tracked_matvec(M_arr, v_arr, mvstats)
            y = np.ascontiguousarray(np.asarray(y_raw, dtype=M_dtype).reshape(-1))
        with timed(stats, "entropy_exact.moments"):
            abs_y = np.abs(y)
            y2_sq = kahan_sum(abs_y ** 2)
            y4_4 = kahan_sum(abs_y ** 4)
        if y2_sq <= 1e-28 or y4_4 <= 1e-28 or np.any(~np.isfinite(y)):
            return -np.inf, np.zeros_like(v_arr), y2_sq, np.inf

        with timed(stats, "entropy_exact.scalar_terms"):
            c = 2.0 * np.log(rows_total / ncols) / np.log(rows_total)
            logf = (1.0 - c) * 0.5 * np.log(y2_sq) + c * 0.25 * np.log(y4_4)

        with timed(stats, "entropy_exact.reverse_mv_y"):
            My = np.ascontiguousarray(np.asarray(tracked_rmatvec(M_arr, y, mvstats), dtype=M_dtype).reshape(-1))
        with timed(stats, "entropy_exact.y_cube"):
            y3 = np.ascontiguousarray(y * y * y, dtype=M_dtype)
        if not _entropy_exact_operand_debug_printed:
            y32 = np.asarray(y, dtype=M_arr.dtype).reshape(-1)
            y332 = np.asarray(y3, dtype=M_arr.dtype).reshape(-1)
            t14 = time.perf_counter()
            _ = M_arr.T @ y
            t15 = time.perf_counter()
            t16 = time.perf_counter()
            _ = M_arr.T @ y32
            t17 = time.perf_counter()
            t18 = time.perf_counter()
            _ = M_arr.T @ y3
            t19 = time.perf_counter()
            t20 = time.perf_counter()
            _ = M_arr.T @ y332
            t21 = time.perf_counter()
            print({
                "objective_operand_layout": {
                    "M": arr_info("M", M_arr),
                    "v": arr_info("v", v_arr),
                    "y_raw": arr_info("y_raw", y_raw),
                    "y": arr_info("y", y),
                    "y3": arr_info("y3", y3),
                }
            })
            print({
                "forced_1d_objective_microbench": {
                    "forward": t1 - t0,
                    "reverse": t3 - t2,
                    "v_shape": v_arr.shape,
                    "v1_shape": v1.shape,
                    "u1_shape": u1.shape,
                    "back1_shape": np.asarray(back1).shape,
                }
            })
            print({
                "double_forward_same_operand": {
                    "first": t5 - t4,
                    "second": t6 - t5,
                }
            })
            print({
                "double_reverse_same_operand": {
                    "first": t8 - t7,
                    "second": t9 - t8,
                }
            })
            print({
                "forward_dtype_compare": {
                    "v_dtype": str(v_arr.dtype),
                    "v32_dtype": str(v1.dtype),
                    "slow_forward": t11 - t10,
                    "cast_forward": t13 - t12,
                }
            })
            print({
                "reverse_dtype_compare": {
                    "y_dtype": str(np.asarray(y).dtype),
                    "y32_dtype": str(y32.dtype),
                    "y3_dtype": str(np.asarray(y3).dtype),
                    "y332_dtype": str(y332.dtype),
                    "slow_reverse_y": t15 - t14,
                    "cast_reverse_y": t17 - t16,
                    "slow_reverse_y3": t19 - t18,
                    "cast_reverse_y3": t21 - t20,
                }
            })
            _entropy_exact_operand_debug_printed = True
        with timed(stats, "entropy_exact.reverse_mv_y3"):
            My3 = np.ascontiguousarray(np.asarray(tracked_rmatvec(M_arr, y3, mvstats), dtype=M_dtype).reshape(-1))
        with timed(stats, "entropy_exact.combine"):
            g = (1.0 - c) * (My / y2_sq) + c * (My3 / y4_4)
            H = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
        return logf, g, y2_sq, H


def entropy_streaming_logscore_grad(M_gain, A_block, V_old, s2_old, q_old, v, rows_total, ncols, stats: Optional[TimeStats] = None, mvstats: Optional[MatvecStats] = None):
    with timed(stats, "entropy_streaming_logscore_grad"):
        M_gain_arr = np.asarray(M_gain)
        A_block_arr = np.asarray(A_block)
        work_dtype = M_gain_arr.dtype
        v_work = np.ascontiguousarray(np.asarray(v, dtype=work_dtype).reshape(-1))
        V_old_work = np.ascontiguousarray(np.asarray(V_old, dtype=work_dtype))
        s2_old_work = np.asarray(s2_old, dtype=work_dtype)
        q_old_work = np.asarray(q_old, dtype=work_dtype)
        with timed(stats, "entropy_stream.forward_m_gain"):
            z = np.ascontiguousarray(np.asarray(tracked_matvec(M_gain_arr, v_work, mvstats), dtype=work_dtype).reshape(-1))
        with timed(stats, "entropy_stream.gain_moment"):
            gain2 = kahan_sum(np.abs(z) ** 2)
        if gain2 <= 1e-28 or np.any(~np.isfinite(z)):
            return -np.inf, np.zeros_like(v_work), 0.0, np.inf

        with timed(stats, "entropy_stream.lowrank_project"):
            a = np.ascontiguousarray(V_old_work.T @ v_work, dtype=work_dtype)
        with timed(stats, "entropy_stream.forward_a_block"):
            y = np.ascontiguousarray(np.asarray(tracked_matvec(A_block_arr, v_work, mvstats), dtype=work_dtype).reshape(-1))
        with timed(stats, "entropy_stream.block_moments"):
            y2_sq = kahan_sum(np.abs(y) ** 2)
            y4_4 = kahan_sum(np.abs(y) ** 4)

        with timed(stats, "entropy_stream.scalar_terms"):
            E_old = float(np.sum((a ** 2) * s2_old_work))
            Q_old = float(np.sum((a ** 4) * q_old_work))
            E = E_old + y2_sq
            Q = Q_old + y4_4
        if E <= 1e-28 or Q <= 1e-28 or np.any(~np.isfinite(y)):
            return -np.inf, np.zeros_like(v_work), E, np.inf

        with timed(stats, "entropy_stream.logf"):
            gamma = np.log(rows_total / ncols) / (2.0 * np.log(rows_total))
            logf = 0.5 * np.log(gain2) + gamma * np.log(Q) - 2.0 * gamma * np.log(E)

        with timed(stats, "entropy_stream.reverse_m_gain"):
            g_gain = np.ascontiguousarray(np.asarray(tracked_rmatvec(M_gain_arr, z, mvstats), dtype=work_dtype).reshape(-1)) / gain2
        with timed(stats, "entropy_stream.reverse_a_block_y"):
            aty = np.ascontiguousarray(np.asarray(tracked_rmatvec(A_block_arr, y, mvstats), dtype=work_dtype).reshape(-1))
        with timed(stats, "entropy_stream.y_cube"):
            a3 = np.ascontiguousarray(a * a * a, dtype=work_dtype)
            y3 = np.ascontiguousarray(y * y * y, dtype=work_dtype)
        with timed(stats, "entropy_stream.reverse_a_block_y3"):
            aty3 = np.ascontiguousarray(np.asarray(tracked_rmatvec(A_block_arr, y3, mvstats), dtype=work_dtype).reshape(-1))
        with timed(stats, "entropy_stream.lowrank_expand"):
            gE_lowrank = np.ascontiguousarray(V_old_work @ (s2_old_work * a), dtype=work_dtype)
            gQ_lowrank = np.ascontiguousarray(V_old_work @ (q_old_work * a3), dtype=work_dtype)
        with timed(stats, "entropy_stream.combine"):
            gE = 2.0 * gE_lowrank + 2.0 * aty
            gQ = 4.0 * gQ_lowrank + 4.0 * aty3
            g = g_gain + gamma * (gQ / Q) - 2.0 * gamma * (gE / E)
            Happrox = -np.log(Q / (E ** 2))
        return logf, g, E, Happrox


def build_entropy_fast_subspace(M_gain, active_r, q_subspace=None, method="setup_svd", q_oversample=16, dtype=np.float32):
    M_arr = np.asarray(M_gain, dtype=dtype)
    min_dim = min(M_arr.shape)
    if min_dim <= 1:
        raise ValueError("M_gain too small for reduced entropy subspace.")

    if q_subspace is None:
        q_subspace = min(max(2 * active_r, active_r + q_oversample, 64), min_dim - 1)
    q_subspace = max(active_r, min(int(q_subspace), min_dim - 1))

    if method == "setup_svd":
        _, s, vh = np.linalg.svd(M_arr, full_matrices=False)
        Vred = np.ascontiguousarray(vh[:q_subspace, :].T, dtype=dtype)
        sred = np.asarray(s[:q_subspace], dtype=float)
    elif method == "lanczos":
        u, s, vh = sp.sparse.linalg.svds(M_arr, k=q_subspace, which='LM')
        order = np.argsort(s)[::-1]
        vh = vh[order, :]
        s = s[order]
        Vred = np.ascontiguousarray(vh.T, dtype=dtype)
        sred = np.asarray(s, dtype=float)
    else:
        raise ValueError(f"Unknown subspace builder: {method}")

    Bred = np.ascontiguousarray(M_arr @ Vred, dtype=dtype)
    return Vred, Bred, sred


def project_reduced(x, Qz):
    x = np.asarray(x).reshape(-1)
    if Qz is None or Qz.size == 0:
        return np.ascontiguousarray(x)
    return np.ascontiguousarray(x - Qz @ (Qz.T @ x))


def retract_reduced(z, Qz, eps=1e-14):
    y = project_reduced(z, Qz)
    ny = np.linalg.norm(y)
    if ny <= eps:
        return None
    return np.ascontiguousarray(y / ny, dtype=y.dtype)


def append_unique_reduced_seed(starts, z, cos_tol=1e-10):
    if z is None:
        return False
    for prev in starts:
        if abs(float(prev @ z)) > 1.0 - cos_tol:
            return False
    starts.append(z)
    return True


def ordered_unused_indices(unused_mask, center_idx):
    order = np.flatnonzero(unused_mask)
    if order.size == 0:
        return order
    dist = np.abs(order - int(center_idx))
    return order[np.argsort(dist, kind="stable")]


def retire_closest_prior_index(prior_coeffs, unused_mask, z_best, Qz):
    if prior_coeffs is None or unused_mask is None:
        return None
    best_idx = None
    best_align = -np.inf
    for idx in np.flatnonzero(unused_mask):
        z_prior = retract_reduced(prior_coeffs[:, idx], Qz)
        if z_prior is None:
            continue
        align = abs(float(z_prior @ z_best))
        if align > best_align:
            best_align = align
            best_idx = int(idx)
    if best_idx is not None:
        unused_mask[best_idx] = False
    return best_idx


def entropy_logscore_grad_reduced(B, z, win, ncols):
    B_arr = np.asarray(B)
    work_dtype = B_arr.dtype
    z = np.ascontiguousarray(np.asarray(z, dtype=work_dtype).reshape(-1))
    c = 2.0 * np.log(win / ncols) / np.log(win)

    y = np.ascontiguousarray(B_arr @ z, dtype=work_dtype)
    y2_sq = max(float(np.dot(y, y)), 1e-30)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 1e-30)
    logf = (1.0 - c) * 0.5 * np.log(y2_sq) + c * 0.25 * np.log(y4_4)

    y3 = np.ascontiguousarray(y * y * y, dtype=work_dtype)
    g2 = np.ascontiguousarray(B_arr.T @ y, dtype=work_dtype) / y2_sq
    g4 = np.ascontiguousarray(B_arr.T @ y3, dtype=work_dtype) / y4_4
    grad = (1.0 - c) * g2 + c * g4
    H2 = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    s = float(np.sqrt(y2_sq))
    return logf, grad, s, H2


def entropy_streaming_logscore_grad_reduced(B_gain, B_block, C_prev, s2_old, q_old, z, rows_total, ncols):
    B_gain_arr = np.asarray(B_gain)
    B_block_arr = np.asarray(B_block)
    work_dtype = B_gain_arr.dtype
    z = np.ascontiguousarray(np.asarray(z, dtype=work_dtype).reshape(-1))
    C_prev_arr = np.ascontiguousarray(np.asarray(C_prev, dtype=work_dtype))
    s2_old_arr = np.asarray(s2_old, dtype=work_dtype)
    q_old_arr = np.asarray(q_old, dtype=work_dtype)

    gain_vec = np.ascontiguousarray(B_gain_arr @ z, dtype=work_dtype)
    gain2 = max(float(np.dot(gain_vec, gain_vec)), 1e-30)

    a = np.ascontiguousarray(C_prev_arr @ z, dtype=work_dtype)
    y = np.ascontiguousarray(B_block_arr @ z, dtype=work_dtype)
    y2_sq = float(np.dot(y, y))
    y4_4 = float(np.sum((y * y) * (y * y)))

    E_old = float(np.sum((a ** 2) * s2_old_arr))
    Q_old = float(np.sum((a ** 4) * q_old_arr))
    E = max(E_old + y2_sq, 1e-30)
    Q = max(Q_old + y4_4, 1e-30)

    gamma = np.log(rows_total / ncols) / (2.0 * np.log(rows_total))
    logf = 0.5 * np.log(gain2) + gamma * np.log(Q) - 2.0 * gamma * np.log(E)

    y3 = np.ascontiguousarray(y * y * y, dtype=work_dtype)
    a3 = np.ascontiguousarray(a * a * a, dtype=work_dtype)
    g_gain = np.ascontiguousarray(B_gain_arr.T @ gain_vec, dtype=work_dtype) / gain2
    aty = np.ascontiguousarray(B_block_arr.T @ y, dtype=work_dtype)
    aty3 = np.ascontiguousarray(B_block_arr.T @ y3, dtype=work_dtype)
    gE_lowrank = np.ascontiguousarray(C_prev_arr.T @ (s2_old_arr * a), dtype=work_dtype)
    gQ_lowrank = np.ascontiguousarray(C_prev_arr.T @ (q_old_arr * a3), dtype=work_dtype)
    gE = 2.0 * gE_lowrank + 2.0 * aty
    gQ = 4.0 * gQ_lowrank + 4.0 * aty3
    grad = g_gain + gamma * (gQ / Q) - 2.0 * gamma * (gE / E)
    H = -(np.log(Q / (E ** 2)))
    s = float(np.sqrt(gain2))
    return logf, grad, s, H


def entropyscore_forget_logscore_grad_rows(A_block, v, rows_ref):
    A_arr = np.asarray(A_block)
    work_dtype = A_arr.dtype
    v_work = np.ascontiguousarray(np.asarray(v, dtype=work_dtype).reshape(-1))
    y = np.ascontiguousarray(A_arr @ v_work, dtype=work_dtype)
    y2_sq = max(float(np.dot(y, y)), 1e-30)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 1e-30)
    rows_block = max(int(A_arr.shape[0]), 2)
    rows_ref = max(int(rows_ref), rows_block)
    c = np.log(rows_block / rows_ref) / np.log(rows_block)
    alpha = (rows_block / rows_ref) ** 0.25
    logpsi = np.log(alpha) + (1.0 - 0.5 * c) * np.log(y2_sq) + 0.25 * c * np.log(y4_4)
    y3 = np.ascontiguousarray(y * y * y, dtype=work_dtype)
    aty = np.ascontiguousarray(A_arr.T @ y, dtype=work_dtype)
    aty3 = np.ascontiguousarray(A_arr.T @ y3, dtype=work_dtype)
    grad = (2.0 - c) * (aty / y2_sq) + c * (aty3 / y4_4)
    H = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    s = float(np.sqrt(y2_sq))
    return logpsi, grad, s, H


def entropyscore_forget_streaming_logscore_grad(M_gain, A_block, V_old, s2_old, v, rows_ref):
    M_gain_arr = np.asarray(M_gain)
    A_block_arr = np.asarray(A_block)
    work_dtype = M_gain_arr.dtype
    v_work = np.ascontiguousarray(np.asarray(v, dtype=work_dtype).reshape(-1))
    V_old_work = np.ascontiguousarray(np.asarray(V_old, dtype=work_dtype))
    s2_old_work = np.asarray(s2_old, dtype=work_dtype)

    gain_vec = np.ascontiguousarray(M_gain_arr @ v_work, dtype=work_dtype)
    gain2 = max(float(np.dot(gain_vec, gain_vec)), 1e-30)

    a = np.ascontiguousarray(V_old_work.T @ v_work, dtype=work_dtype)
    y = np.ascontiguousarray(A_block_arr @ v_work, dtype=work_dtype)
    y2_sq = max(float(np.dot(y, y)), 1e-30)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 1e-30)
    rows_block = max(int(A_block_arr.shape[0]), 2)
    rows_ref = max(int(rows_ref), rows_block)
    c = np.log(rows_block / rows_ref) / np.log(rows_block)
    alpha = (rows_block / rows_ref) ** 0.25

    E_old = float(np.sum((a ** 2) * s2_old_work))
    logpsi = np.log(alpha) + (1.0 - 0.5 * c) * np.log(y2_sq) + 0.25 * c * np.log(y4_4)
    psi = float(np.exp(logpsi))
    score = max(E_old + psi, 1e-30)
    logf = np.log(score)

    y3 = np.ascontiguousarray(y * y * y, dtype=work_dtype)
    a_old = np.ascontiguousarray(V_old_work @ (s2_old_work * a), dtype=work_dtype)
    aty = np.ascontiguousarray(A_block_arr.T @ y, dtype=work_dtype)
    aty3 = np.ascontiguousarray(A_block_arr.T @ y3, dtype=work_dtype)
    gpsi = psi * ((2.0 - c) * (aty / y2_sq) + c * (aty3 / y4_4))
    grad_score = 2.0 * a_old + gpsi
    grad = grad_score / score

    H = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    s = float(np.sqrt(gain2))
    return logf, grad, s, H


def entropyscore_forget_logscore_grad_reduced(B, z, rows_block, rows_ref):
    B_arr = np.asarray(B)
    work_dtype = B_arr.dtype
    z = np.ascontiguousarray(np.asarray(z, dtype=work_dtype).reshape(-1))
    y = np.ascontiguousarray(B_arr @ z, dtype=work_dtype)
    y2_sq = max(float(np.dot(y, y)), 1e-30)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 1e-30)
    rows_block = max(int(rows_block), 2)
    rows_ref = max(int(rows_ref), rows_block)
    c = np.log(rows_block / rows_ref) / np.log(rows_block)
    alpha = (rows_block / rows_ref) ** 0.25
    logf = np.log(alpha) + (1.0 - 0.5 * c) * np.log(y2_sq) + 0.25 * c * np.log(y4_4)

    y3 = np.ascontiguousarray(y * y * y, dtype=work_dtype)
    g2 = np.ascontiguousarray(B_arr.T @ y, dtype=work_dtype) / y2_sq
    g4 = np.ascontiguousarray(B_arr.T @ y3, dtype=work_dtype) / y4_4
    grad = (2.0 - c) * g2 + c * g4
    H = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    s = float(np.sqrt(y2_sq))
    return logf, grad, s, H


def entropyscore_forget_streaming_logscore_grad_reduced(B_gain, B_block, C_prev, s2_old, z, rows_block, rows_ref):
    B_gain_arr = np.asarray(B_gain)
    B_block_arr = np.asarray(B_block)
    work_dtype = B_gain_arr.dtype
    z = np.ascontiguousarray(np.asarray(z, dtype=work_dtype).reshape(-1))
    C_prev_arr = np.ascontiguousarray(np.asarray(C_prev, dtype=work_dtype))
    s2_old_arr = np.asarray(s2_old, dtype=work_dtype)

    gain_vec = np.ascontiguousarray(B_gain_arr @ z, dtype=work_dtype)
    gain2 = max(float(np.dot(gain_vec, gain_vec)), 1e-30)

    a = np.ascontiguousarray(C_prev_arr @ z, dtype=work_dtype)
    y = np.ascontiguousarray(B_block_arr @ z, dtype=work_dtype)
    y2_sq = max(float(np.dot(y, y)), 1e-30)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 1e-30)
    rows_block = max(int(rows_block), 2)
    rows_ref = max(int(rows_ref), rows_block)
    c = np.log(rows_block / rows_ref) / np.log(rows_block)
    alpha = (rows_block / rows_ref) ** 0.25

    E_old = float(np.sum((a ** 2) * s2_old_arr))
    logpsi = np.log(alpha) + (1.0 - 0.5 * c) * np.log(y2_sq) + 0.25 * c * np.log(y4_4)
    psi = float(np.exp(logpsi))
    score = max(E_old + psi, 1e-30)
    logf = np.log(score)

    y3 = np.ascontiguousarray(y * y * y, dtype=work_dtype)
    aty = np.ascontiguousarray(B_block_arr.T @ y, dtype=work_dtype)
    aty3 = np.ascontiguousarray(B_block_arr.T @ y3, dtype=work_dtype)
    g_old = np.ascontiguousarray(C_prev_arr.T @ (s2_old_arr * a), dtype=work_dtype)
    grad_score = 2.0 * g_old + psi * ((2.0 - c) * (aty / y2_sq) + c * (aty3 / y4_4))
    grad = grad_score / score

    H = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    s = float(np.sqrt(gain2))
    return logf, grad, s, H


def entropyscore_forget_score_grad_reduced(B, z, rows_block, rows_ref):
    logf, grad_log, s, H = entropyscore_forget_logscore_grad_reduced(
        B, z, rows_block, rows_ref
    )
    score = float(np.exp(logf))
    grad = np.ascontiguousarray(score * grad_log, dtype=np.asarray(grad_log).dtype)
    return score, grad, s, H


def entropyscore_forget_streaming_score_grad_reduced(
    B_gain, B_block, C_prev, s2_old, z, rows_block, rows_ref
):
    logf, grad_log, s, H = entropyscore_forget_streaming_logscore_grad_reduced(
        B_gain, B_block, C_prev, s2_old, z, rows_block, rows_ref
    )
    score = float(np.exp(logf))
    grad = np.ascontiguousarray(score * grad_log, dtype=np.asarray(grad_log).dtype)
    return score, grad, s, H


def basic_projected_ascent_single_reduced_forget_cex(
    B, z0, Qz, rows_block, rows_ref, maxit=80, tol=1e-8,
    reuse_line_search_grad=True
):
    z = retract_reduced(z0, Qz)
    if z is None:
        raise RuntimeError("Initial reduced seed is infeasible.")

    score, grad, s, H = entropyscore_forget_score_grad_reduced(B, z, rows_block, rows_ref)
    stop = {"reason": "maxit", "iters": maxit, "grad_norm": np.nan}
    progress_f_tol = 1e-12
    progress_step_tol = 1e-10

    for it in range(maxit):
        gtan = project_reduced(grad - z * float(z @ grad), Qz)
        gnorm = float(np.linalg.norm(gtan))
        if gnorm <= tol:
            stop = {"reason": "grad_tol", "iters": it, "grad_norm": gnorm}
            break

        accepted = False
        accepted_eval = None
        alpha = 1.0
        score_old = score
        z_old = z

        for ls_iter in range(20):
            zt = retract_reduced(z + alpha * gtan, Qz)
            if zt is not None:
                score_t, grad_t, s_t, H_t = entropyscore_forget_score_grad_reduced(
                    B, zt, rows_block, rows_ref
                )
                rhs = score_old + 1e-4 * alpha * float(gtan @ gtan)
                if score_t >= rhs:
                    z = zt
                    accepted_eval = (score_t, grad_t, s_t, H_t)
                    accepted = True
                    break
            alpha *= 0.5

        if not accepted:
            stop = {
                "reason": "line_search_fail",
                "iters": it + 1,
                "grad_norm": gnorm,
                "line_search_steps": 20,
            }
            z = z_old
            break

        if reuse_line_search_grad and accepted_eval is not None:
            score, grad, s, H = accepted_eval
        else:
            score, grad, s, H = entropyscore_forget_score_grad_reduced(
                B, z, rows_block, rows_ref
            )
        step_norm = float(np.linalg.norm(z - z_old))
        f_change = abs(score - score_old)
        f_threshold = progress_f_tol * max(1.0, abs(score_old))
        stop = {
            "reason": "progress",
            "iters": it + 1,
            "grad_norm": gnorm,
            "step_norm": step_norm,
            "f_change": f_change,
            "f_threshold": f_threshold,
            "line_search_alpha": alpha,
            "line_search_steps": ls_iter + 1,
        }
        if f_change <= f_threshold:
            stop["reason"] = "f_change_tol"
            break
        if step_norm <= progress_step_tol:
            stop["reason"] = "step_tol"
            break
    else:
        gtan = project_reduced(grad - z * float(z @ grad), Qz)
        stop = {"reason": "maxit", "iters": maxit, "grad_norm": float(np.linalg.norm(gtan))}

    return z, score, s, H, stop


def basic_projected_ascent_single_reduced_streaming_forget_cex(
    B_gain, B_block, C_prev, s2_old, z0, Qz, rows_block, rows_ref,
    maxit=80, tol=1e-8, reuse_line_search_grad=True
):
    z = retract_reduced(z0, Qz)
    if z is None:
        raise RuntimeError("Initial reduced seed is infeasible.")

    score, grad, s, H = entropyscore_forget_streaming_score_grad_reduced(
        B_gain, B_block, C_prev, s2_old, z, rows_block, rows_ref
    )
    stop = {"reason": "maxit", "iters": maxit, "grad_norm": np.nan}
    progress_f_tol = 1e-12
    progress_step_tol = 1e-10

    for it in range(maxit):
        gtan = project_reduced(grad - z * float(z @ grad), Qz)
        gnorm = float(np.linalg.norm(gtan))
        if gnorm <= tol:
            stop = {"reason": "grad_tol", "iters": it, "grad_norm": gnorm}
            break

        accepted = False
        accepted_eval = None
        alpha = 1.0
        score_old = score
        z_old = z

        for ls_iter in range(20):
            zt = retract_reduced(z + alpha * gtan, Qz)
            if zt is not None:
                score_t, grad_t, s_t, H_t = entropyscore_forget_streaming_score_grad_reduced(
                    B_gain, B_block, C_prev, s2_old, zt, rows_block, rows_ref
                )
                rhs = score_old + 1e-4 * alpha * float(gtan @ gtan)
                if score_t >= rhs:
                    z = zt
                    accepted_eval = (score_t, grad_t, s_t, H_t)
                    accepted = True
                    break
            alpha *= 0.5

        if not accepted:
            stop = {
                "reason": "line_search_fail",
                "iters": it + 1,
                "grad_norm": gnorm,
                "line_search_steps": 20,
            }
            z = z_old
            break

        if reuse_line_search_grad and accepted_eval is not None:
            score, grad, s, H = accepted_eval
        else:
            score, grad, s, H = entropyscore_forget_streaming_score_grad_reduced(
                B_gain, B_block, C_prev, s2_old, z, rows_block, rows_ref
            )
        step_norm = float(np.linalg.norm(z - z_old))
        f_change = abs(score - score_old)
        f_threshold = progress_f_tol * max(1.0, abs(score_old))
        stop = {
            "reason": "progress",
            "iters": it + 1,
            "grad_norm": gnorm,
            "step_norm": step_norm,
            "f_change": f_change,
            "f_threshold": f_threshold,
            "line_search_alpha": alpha,
            "line_search_steps": ls_iter + 1,
        }
        if f_change <= f_threshold:
            stop["reason"] = "f_change_tol"
            break
        if step_norm <= progress_step_tol:
            stop["reason"] = "step_tol"
            break
    else:
        gtan = project_reduced(grad - z * float(z @ grad), Qz)
        stop = {"reason": "maxit", "iters": maxit, "grad_norm": float(np.linalg.norm(gtan))}

    return z, score, s, H, stop


def basic_projected_ascent_single_reduced_forget(B, z0, Qz, rows_block, rows_ref, maxit=80, tol=1e-8):
    z = retract_reduced(z0, Qz)
    if z is None:
        raise RuntimeError("Initial reduced seed is infeasible.")

    logf, grad, s, H = entropyscore_forget_logscore_grad_reduced(B, z, rows_block, rows_ref)
    stop = {"reason": "maxit", "iters": maxit, "grad_norm": np.nan}

    for it in range(maxit):
        gtan = project_reduced(grad - z * float(z @ grad), Qz)
        gnorm = float(np.linalg.norm(gtan))
        if gnorm <= tol:
            stop = {"reason": "grad_tol", "iters": it, "grad_norm": gnorm}
            break

        step = gtan / max(gnorm, 1e-30)
        alpha = 1.0
        improved = False

        for _ in range(20):
            zt = retract_reduced(z + alpha * step, Qz)
            if zt is None:
                alpha *= 0.5
                continue
            logf_t, grad_t, s_t, H_t = entropyscore_forget_logscore_grad_reduced(B, zt, rows_block, rows_ref)
            if logf_t > logf:
                z, logf, grad, s, H = zt, logf_t, grad_t, s_t, H_t
                improved = True
                break
            alpha *= 0.5

        if not improved:
            stop = {"reason": "line_search_fail", "iters": it + 1, "grad_norm": gnorm}
            break
    else:
        stop = {"reason": "maxit", "iters": maxit, "grad_norm": float(np.linalg.norm(project_reduced(grad - z * float(z @ grad), Qz)))}

    return z, logf, s, H, stop


def basic_projected_ascent_single_reduced_streaming_forget(B_gain, B_block, C_prev, s2_old, z0, Qz, rows_block, rows_ref,
                                                           maxit=80, tol=1e-8):
    z = retract_reduced(z0, Qz)
    if z is None:
        raise RuntimeError("Initial reduced seed is infeasible.")

    logf, grad, s, H = entropyscore_forget_streaming_logscore_grad_reduced(
        B_gain, B_block, C_prev, s2_old, z, rows_block, rows_ref
    )
    stop = {"reason": "maxit", "iters": maxit, "grad_norm": np.nan}

    for it in range(maxit):
        gtan = project_reduced(grad - z * float(z @ grad), Qz)
        gnorm = float(np.linalg.norm(gtan))
        if gnorm <= tol:
            stop = {"reason": "grad_tol", "iters": it, "grad_norm": gnorm}
            break

        step = gtan / max(gnorm, 1e-30)
        alpha = 1.0
        improved = False

        for _ in range(20):
            zt = retract_reduced(z + alpha * step, Qz)
            if zt is None:
                alpha *= 0.5
                continue
            logf_t, grad_t, s_t, H_t = entropyscore_forget_streaming_logscore_grad_reduced(
                B_gain, B_block, C_prev, s2_old, zt, rows_block, rows_ref
            )
            if logf_t > logf:
                z, logf, grad, s, H = zt, logf_t, grad_t, s_t, H_t
                improved = True
                break
            alpha *= 0.5

        if not improved:
            stop = {"reason": "line_search_fail", "iters": it + 1, "grad_norm": gnorm}
            break
    else:
        stop = {"reason": "maxit", "iters": maxit, "grad_norm": float(np.linalg.norm(project_reduced(grad - z * float(z @ grad), Qz)))}

    return z, logf, s, H, stop


def entropyscore_forget_full_gradient_residual(
    M_gain,
    A_block,
    v,
    Vred,
    state_prev,
    rows_ref,
    Q_prev=None,
    return_vector=False,
):
    if state_prev is None:
        _, grad, _, _ = entropyscore_forget_logscore_grad_rows(A_block, v, rows_ref)
    else:
        _, grad, _, _ = entropyscore_forget_streaming_logscore_grad(
            M_gain, A_block, state_prev["V"], state_prev["s2"], v, rows_ref
        )
    dtype = np.result_type(np.asarray(grad).dtype, np.asarray(v).dtype)
    grad = np.ascontiguousarray(np.asarray(grad, dtype=dtype).reshape(-1))
    v = np.ascontiguousarray(np.asarray(v, dtype=dtype).reshape(-1))

    g_tan = grad - v * float(v @ grad)
    if Q_prev is not None and np.asarray(Q_prev).size:
        Q_prev_arr = np.ascontiguousarray(np.asarray(Q_prev, dtype=dtype))
        g_tan = g_tan - Q_prev_arr @ (Q_prev_arr.T @ g_tan)

    Vred_arr = np.ascontiguousarray(np.asarray(Vred, dtype=dtype))
    r = g_tan - Vred_arr @ (Vred_arr.T @ g_tan)
    r = np.ascontiguousarray(r, dtype=dtype)
    r_norm = float(np.linalg.norm(r))
    g_norm = float(np.linalg.norm(g_tan))
    if return_vector:
        return r_norm, g_norm, r
    return r_norm, g_norm


def row_norm_seed(A_block, rank):
    """Top right singular vectors after normalizing each row to unit L2 norm."""
    A_arr = np.asarray(A_block)
    row_norms = np.linalg.norm(A_arr, axis=1, keepdims=True)
    safe = np.where(row_norms > 0, row_norms, 1.0)
    _, _, Vt = np.linalg.svd(A_arr / safe, full_matrices=False)
    return np.ascontiguousarray(Vt.T[:, : int(rank)])


# ===========================================================================
# Diagnostic helpers for score_variant='combined' (ported from
# test_matrices_fast/cex_restricted_space_probe.py). See
# test_matrices_fast/summary/diagnostic_dumps_howto.txt for semantics.
# ===========================================================================

def project_onto_span(X, Q):
    X_arr = np.asarray(X)
    if Q is None or not np.asarray(Q).size:
        return np.zeros_like(X_arr)
    Q_arr = np.ascontiguousarray(np.asarray(Q, dtype=np.result_type(X_arr.dtype, np.asarray(Q).dtype)))
    return np.ascontiguousarray(Q_arr @ (Q_arr.T @ X_arr), dtype=np.result_type(X_arr.dtype, Q_arr.dtype))


def right_singular_row_basis(M_gain, dtype=None, eps=1e-12):
    M_arr = np.asarray(M_gain)
    if dtype is None:
        dtype = M_arr.dtype
        if not np.issubdtype(dtype, np.floating):
            dtype = np.float64
    if M_arr.size == 0:
        n = M_arr.shape[1] if M_arr.ndim == 2 else 0
        return np.zeros((n, 0), dtype=dtype)
    _, s, Vh = np.linalg.svd(np.asarray(M_arr, dtype=np.float64), full_matrices=False)
    if s.size == 0:
        return np.zeros((M_arr.shape[1], 0), dtype=dtype)
    keep = s > eps
    if not np.any(keep):
        return np.zeros((M_arr.shape[1], 0), dtype=dtype)
    return np.ascontiguousarray(Vh[keep, :].T, dtype=dtype)


def subspace_principal_cosines(Q1, Q2):
    Q1_arr = orthonormalize_columns(Q1)
    Q2_arr = orthonormalize_columns(Q2)
    if Q1_arr.size == 0 or Q2_arr.size == 0:
        return np.zeros(0, dtype=float)
    s = np.linalg.svd(Q1_arr.T @ Q2_arr, compute_uv=False)
    return np.clip(np.asarray(s, dtype=float), 0.0, 1.0)


def normed_vector(v, dtype=np.float64):
    if v is None:
        return None
    out = np.asarray(v, dtype=dtype).reshape(-1)
    nrm = float(np.linalg.norm(out))
    if nrm <= 1e-30:
        return None
    return np.ascontiguousarray(out / nrm, dtype=dtype)


def orth_against(v, Q, dtype=np.float64):
    vv = normed_vector(v, dtype=dtype)
    if vv is None:
        return None
    Q_arr = np.asarray(Q, dtype=dtype)
    if Q_arr.size:
        vv = vv - Q_arr @ (Q_arr.T @ vv)
    return normed_vector(vv, dtype=dtype)


def block_svd_complement(A_block, Q, max_candidates=16):
    A_arr = np.asarray(A_block, dtype=np.float64)
    _, _, vh = np.linalg.svd(A_arr, full_matrices=False)
    best = None
    best_gain = -np.inf
    for row in vh[: min(int(max_candidates), vh.shape[0])]:
        cand = orth_against(row, Q, dtype=np.float64)
        if cand is None:
            continue
        gain = float(np.linalg.norm(A_arr @ cand) ** 2)
        if gain > best_gain:
            best_gain = gain
            best = cand
    return best


def svd_complement(M_gain, Q, max_candidates=16):
    M_arr = np.asarray(M_gain, dtype=np.float64)
    _, _, vh = np.linalg.svd(M_arr, full_matrices=False)
    best = None
    best_gain = -np.inf
    for row in vh[: min(int(max_candidates), vh.shape[0])]:
        cand = orth_against(row, Q, dtype=np.float64)
        if cand is None:
            continue
        gain = float(np.linalg.norm(M_arr @ cand) ** 2)
        if gain > best_gain:
            best_gain = gain
            best = cand
    return best


def response_shape(A_block, v):
    if v is None:
        return {"relH": np.nan, "max_frac": np.nan, "top4_frac": np.nan}
    y = np.asarray(A_block, dtype=np.float64) @ np.asarray(v, dtype=np.float64)
    e = y * y
    total = max(float(np.sum(e)), 1e-30)
    p = e / total
    p_pos = p[p > 0.0]
    H = -float(np.sum(p_pos * np.log(p_pos))) if p_pos.size else np.nan
    relH = H / np.log(max(len(e), 2)) if np.isfinite(H) else np.nan
    sorted_p = np.sort(p)[::-1]
    return {
        "relH": relH,
        "max_frac": float(sorted_p[0]) if sorted_p.size else np.nan,
        "top4_frac": float(np.sum(sorted_p[: min(4, sorted_p.size)])) if sorted_p.size else np.nan,
    }


def hmean_score(a, b, eps=1e-30):
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    a = max(float(a), 0.0)
    b = max(float(b), 0.0)
    return float((2.0 * a * b) / max(a + b, eps))


def rank2_svd_frame(v1, chosen_v2, M_gain, rank=2):
    v1 = np.asarray(v1, dtype=np.float64).reshape(-1)
    if chosen_v2 is None:
        return None
    v2 = np.asarray(chosen_v2, dtype=np.float64).reshape(-1)
    C = np.column_stack([v1, v2])
    Qc, Rc = np.linalg.qr(C)
    diag = np.abs(np.diag(Rc))
    if diag.size < 2 or diag[1] < 1e-10 * max(diag[0], 1e-30):
        return None
    Mc = np.asarray(M_gain, dtype=np.float64) @ Qc
    _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
    return np.ascontiguousarray(Qc @ Vt.T[:, :rank], dtype=np.float64)


_BENCHMARK_MODULE_CACHE = {}


def load_test_matrices_fast_module(module_name, filename):
    cache_key = (module_name, filename)
    if cache_key in _BENCHMARK_MODULE_CACHE:
        return _BENCHMARK_MODULE_CACHE[cache_key]
    module_path = os.path.join(os.path.dirname(__file__), "test_matrices_fast", filename)
    module_dir = os.path.dirname(module_path)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _BENCHMARK_MODULE_CACHE[cache_key] = module
    return module


def projected_true_span_oracle(M_gain, V_exact, r, dtype=None, eps=1e-12, row_samples=None):
    if V_exact is None:
        n = int(np.asarray(M_gain).shape[1])
        out_dtype = np.float64 if dtype is None else dtype
        return np.zeros((n, 0), dtype=out_dtype), np.zeros((n, 0), dtype=out_dtype)
    if dtype is None:
        dtype = np.result_type(np.asarray(M_gain).dtype, np.asarray(V_exact).dtype)
        if row_samples is not None:
            dtype = np.result_type(dtype, np.asarray(row_samples).dtype)
        if not np.issubdtype(dtype, np.floating):
            dtype = np.float64
    M_arr = np.ascontiguousarray(np.asarray(M_gain, dtype=dtype))
    V_target = orthonormalize_columns(np.asarray(V_exact, dtype=dtype)[:, : int(r)], dtype=dtype, eps=eps)
    row_basis_source = M_arr
    if row_samples is not None and np.asarray(row_samples).size:
        R_arr = np.asarray(row_samples, dtype=dtype)
        if R_arr.ndim == 1:
            R_arr = R_arr.reshape(1, -1)
        if R_arr.shape[1] != M_arr.shape[1]:
            raise ValueError("row_samples must have the same column count as M_gain.")
        row_basis_source = np.ascontiguousarray(np.vstack([M_arr, R_arr]), dtype=dtype)
    Q_row = right_singular_row_basis(row_basis_source, dtype=dtype, eps=eps)
    if Q_row.size == 0 or V_target.size == 0:
        return np.zeros((M_arr.shape[1], 0), dtype=dtype), Q_row
    V_proj = project_onto_span(V_target, Q_row)
    Q_oracle = orthonormalize_columns(V_proj, dtype=dtype, eps=eps)
    return Q_oracle, Q_row


def response_entropy_stats(response):
    y = np.asarray(response, dtype=float).reshape(-1)
    y2_sq = max(float(np.dot(y, y)), 1e-30)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 1e-30)
    H = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    rel_H = H / np.log(max(y.size, 2))
    return H, rel_H, y2_sq, y4_4


def combined_score_component_details(
    M_gain, A_block, v, rows_ref, state_prev=None, old_row_memory=None
):
    A_arr = np.asarray(A_block)
    work_dtype = np.result_type(A_arr.dtype, np.asarray(v).dtype)
    v_work = np.ascontiguousarray(np.asarray(v, dtype=work_dtype).reshape(-1))

    gain_vec = np.ascontiguousarray(np.asarray(M_gain, dtype=work_dtype) @ v_work, dtype=work_dtype)
    gain2 = max(float(np.dot(gain_vec, gain_vec)), 1e-30)

    y = np.ascontiguousarray(np.asarray(A_block, dtype=work_dtype) @ v_work, dtype=work_dtype)
    new_y2_sq = max(float(np.dot(y, y)), 0.0)
    new_y4_4 = max(float(np.sum((y * y) * (y * y))), 0.0)
    rows_block = max(int(A_arr.shape[0]), 0)
    rows_entropy = rows_block

    pooled_y2_sq = new_y2_sq
    pooled_y4_4 = new_y4_4
    old_y2_sq = np.nan
    old_y4_4 = np.nan
    old_H = np.nan
    old_rel_H = np.nan
    old_rows = 0

    R_old_arr = None if old_row_memory is None else np.asarray(old_row_memory, dtype=work_dtype)
    if state_prev is not None and R_old_arr is not None and R_old_arr.size:
        if R_old_arr.ndim == 1:
            R_old_arr = R_old_arr.reshape(1, -1)
        r = np.ascontiguousarray(R_old_arr @ v_work, dtype=work_dtype)
        old_y2_sq = max(float(np.dot(r, r)), 0.0)
        old_y4_4 = max(float(np.sum((r * r) * (r * r))), 0.0)
        pooled_y2_sq += old_y2_sq
        pooled_y4_4 += old_y4_4
        old_rows = int(R_old_arr.shape[0])
        rows_entropy += old_rows
        if old_y2_sq > 0.0 and old_y4_4 > 0.0:
            old_H = -(np.log(old_y4_4) - 2.0 * np.log(old_y2_sq))
            old_rel_H = old_H / np.log(max(old_rows, 2))

    pooled_y2_sq = max(pooled_y2_sq, 1e-30)
    pooled_y4_4 = max(pooled_y4_4, 1e-30)
    rows_entropy = max(rows_entropy, 2)
    rows_ref_eff = max(int(rows_ref), rows_entropy)
    n_old = 0 if state_prev is None else int(state_prev.get("rows_seen", 0))
    rows_seen = min(max(n_old + rows_block, 1), rows_ref_eff)
    c = np.log(rows_seen / rows_ref_eff) / (2.0 * np.log(rows_entropy))
    H_pooled = -(np.log(pooled_y4_4) - 2.0 * np.log(pooled_y2_sq))
    phi = float(np.exp(c * (np.log(pooled_y4_4) - 2.0 * np.log(pooled_y2_sq))))
    score_total = gain2 * phi
    new_H = np.nan
    new_rel_H = np.nan
    if new_y2_sq > 0.0 and new_y4_4 > 0.0:
        new_H = -(np.log(new_y4_4) - 2.0 * np.log(new_y2_sq))
        new_rel_H = new_H / np.log(max(rows_block, 2))

    return {
        "gain2": gain2,
        "score_total": score_total,
        "phi": phi,
        "pooled_y2_sq": pooled_y2_sq,
        "pooled_y4_4": pooled_y4_4,
        "pooled_H": H_pooled,
        "pooled_rel_H": H_pooled / np.log(rows_entropy) if rows_entropy > 1 else np.nan,
        "combined_c": float(c),
        "rows_entropy": int(rows_entropy),
        "rows_seen": int(rows_seen),
        "new_y2_sq": new_y2_sq,
        "new_y4_4": new_y4_4,
        "new_H": new_H,
        "new_rel_H": new_rel_H,
        "old_y2_sq": old_y2_sq,
        "old_y4_4": old_y4_4,
        "old_H": old_H,
        "old_rel_H": old_rel_H,
        "old_rows": old_rows,
    }


def score_full_vector_combined(M_gain, A_block, v, rows_ref, state_prev=None, old_row_memory=None):
    """Evaluate the combined-variant score at a full-space direction v."""
    A_arr = np.asarray(A_block)
    work_dtype = np.result_type(A_arr.dtype, np.asarray(v).dtype)
    v_work = np.ascontiguousarray(np.asarray(v, dtype=work_dtype).reshape(-1))
    if state_prev is None:
        score, _, s, H = combined_score_grad_reduced(A_arr, v_work, A_arr.shape[0], rows_ref)
    else:
        n_old = int(state_prev.get("rows_seen", 0))
        R_old = None if old_row_memory is None else np.asarray(old_row_memory, dtype=work_dtype)
        score, _, s, H = combined_streaming_score_grad_reduced(
            np.asarray(M_gain, dtype=work_dtype), A_arr, R_old, v_work,
            A_arr.shape[0], rows_ref, n_old,
        )
    return float(score), float(s), float(H)


def print_combined_score_component_dump(label, vectors, M_gain, A_block, rows_ref, state_prev, old_row_memory):
    """Print per-direction score breakdown (combined variant only)."""
    print(f"{label}:")
    for name, vec in vectors:
        comp = combined_score_component_details(
            M_gain, A_block, vec, rows_ref, state_prev=state_prev, old_row_memory=old_row_memory,
        )
        print(
            f"  {name}: total={comp['score_total']:.12f} gain2={comp['gain2']:.12f} "
            f"phi={comp['phi']:.12f} pooled_y2={comp['pooled_y2_sq']:.12f} "
            f"pooled_y4={comp['pooled_y4_4']:.12f} pooled_H={comp['pooled_H']:.12f} "
            f"pooled_rel_H={comp['pooled_rel_H']:.12f} combined_c={comp['combined_c']:.12f} "
            f"rows_entropy={comp['rows_entropy']} rows_seen={comp['rows_seen']} "
            f"new_y2={comp['new_y2_sq']:.12f} new_y4={comp['new_y4_4']:.12f} "
            f"new_H={comp['new_H']:.12f} new_rel_H={comp['new_rel_H']:.12f} "
            f"old_y2={comp['old_y2_sq']:.12f} old_y4={comp['old_y4_4']:.12f} "
            f"old_H={comp['old_H']:.12f} old_rel_H={comp['old_rel_H']:.12f} "
            f"old_rows={comp['old_rows']}"
        )


def oracle_projection_diagnostics_combined(
    M_gain, A_block, V_exact, V_opt, rank, rows_ref,
    state_prev=None, old_row_memory=None, oracle_projection_row_samples=None
):
    """Oracle projection sanity check: raw oracle scores, QR oracle scores,
    opt_proj_norms (how much of each oracle direction the optimizer captured),
    opt_vs_qoracle_cosines (principal cosines between V_opt and Q_oracle).
    """
    if V_exact is None or np.asarray(V_exact).size == 0:
        return None
    diag_dtype = np.result_type(np.asarray(M_gain).dtype, np.asarray(V_exact).dtype, np.float64)
    Q_oracle, Q_row = projected_true_span_oracle(
        np.asarray(M_gain, dtype=diag_dtype),
        np.asarray(V_exact, dtype=diag_dtype)[:, : int(rank)],
        int(rank), dtype=diag_dtype, row_samples=oracle_projection_row_samples,
    )
    raw_cols = []
    for j in range(min(int(rank), np.asarray(V_exact).shape[1])):
        v_proj = project_onto_span(np.asarray(V_exact, dtype=diag_dtype)[:, j], Q_row).reshape(-1)
        v_norm = float(np.linalg.norm(v_proj))
        if v_norm > 1e-30:
            raw_cols.append(np.ascontiguousarray(v_proj / v_norm, dtype=diag_dtype))
    if not raw_cols:
        return None
    raw_proj = np.column_stack(raw_cols)
    raw_scores = np.asarray([
        score_full_vector_combined(M_gain, A_block, raw_proj[:, j], rows_ref,
                                   state_prev=state_prev, old_row_memory=old_row_memory)[0]
        for j in range(raw_proj.shape[1])
    ], dtype=float)
    qr_scores = np.asarray([
        score_full_vector_combined(M_gain, A_block, Q_oracle[:, j], rows_ref,
                                   state_prev=state_prev, old_row_memory=old_row_memory)[0]
        for j in range(Q_oracle.shape[1])
    ], dtype=float)
    V_opt_arr = orthonormalize_columns(np.asarray(V_opt, dtype=diag_dtype), dtype=diag_dtype)
    projected_into_opt = V_opt_arr @ (V_opt_arr.T @ raw_proj) if V_opt_arr.size else np.zeros_like(raw_proj)
    opt_proj_norms = np.linalg.norm(projected_into_opt, axis=0)
    principal_cosines = subspace_principal_cosines(V_opt_arr, Q_oracle)
    raw_overlap = np.nan
    if raw_proj.shape[1] >= 2:
        raw_overlap = abs(float(raw_proj[:, 0] @ raw_proj[:, 1]))
    return {
        "raw_oracle_scores": raw_scores,
        "raw_oracle_score_sum": float(np.sum(raw_scores)),
        "qr_oracle_scores": qr_scores,
        "qr_oracle_score_sum": float(np.sum(qr_scores)),
        "opt_proj_norms": np.asarray(opt_proj_norms, dtype=float),
        "opt_vs_qoracle_cosines": np.asarray(principal_cosines, dtype=float),
        "raw_oracle_overlap": float(raw_overlap),
    }


def oracle_projection_candidate_combined(
    M_gain, A_block, V_exact, rank, rows_ref,
    state_prev=None, old_row_memory=None, oracle_projection_row_samples=None
):
    if V_exact is None or np.asarray(V_exact).size == 0:
        return None
    cand_dtype = np.result_type(np.asarray(M_gain).dtype, np.asarray(V_exact).dtype, np.float64)
    Q_oracle, _ = projected_true_span_oracle(
        np.asarray(M_gain, dtype=cand_dtype),
        np.asarray(V_exact, dtype=cand_dtype)[:, : int(rank)],
        int(rank), dtype=cand_dtype, row_samples=oracle_projection_row_samples,
    )
    if Q_oracle.shape[1] < int(rank):
        return None
    scores = np.zeros(int(rank), dtype=float)
    s_vals = np.zeros(int(rank), dtype=float)
    H_vals = np.zeros(int(rank), dtype=float)
    for j in range(int(rank)):
        scores[j], s_vals[j], H_vals[j] = score_full_vector_combined(
            M_gain, A_block, Q_oracle[:, j], rows_ref,
            state_prev=state_prev, old_row_memory=old_row_memory,
        )
    return {
        "V": np.ascontiguousarray(Q_oracle[:, : int(rank)], dtype=np.asarray(M_gain).dtype),
        "score": scores,
        "s": s_vals,
        "H": H_vals,
        "score_sum": float(np.sum(scores)),
    }


def oracle_old_row_responses_dump(
    M_gain, V_exact, rank, old_row_memory, old_row_memory_idx=None, label="",
    oracle_projection_row_samples=None
):
    if V_exact is None or np.asarray(V_exact).size == 0:
        print("oracle_old_row_response_dump: skipped no V_exact")
        return
    if old_row_memory is None or np.asarray(old_row_memory).size == 0:
        print("oracle_old_row_response_dump: skipped no old_row_memory")
        return
    dump_dtype = np.result_type(np.asarray(M_gain).dtype, np.asarray(V_exact).dtype, np.float64)
    _, Q_row = projected_true_span_oracle(
        np.asarray(M_gain, dtype=dump_dtype),
        np.asarray(V_exact, dtype=dump_dtype)[:, : int(rank)],
        int(rank), dtype=dump_dtype, row_samples=oracle_projection_row_samples,
    )
    R_old = np.asarray(old_row_memory, dtype=dump_dtype)
    print(f"oracle_old_row_response_dump{label}: rows={R_old.shape[0]}")
    if old_row_memory_idx is not None:
        print(
            "old_row_memory_indices_1based: "
            f"{np.array2string(np.asarray(old_row_memory_idx, dtype=int) + 1, max_line_width=120)}"
        )
    for j in range(min(int(rank), np.asarray(V_exact).shape[1])):
        v_proj = project_onto_span(np.asarray(V_exact, dtype=dump_dtype)[:, j], Q_row).reshape(-1)
        v_norm = float(np.linalg.norm(v_proj))
        if v_norm <= 1e-30:
            print(f"v{j + 1}_proj_old_row_response: skipped zero projection")
            continue
        v_proj = np.ascontiguousarray(v_proj / v_norm, dtype=dump_dtype)
        response = np.ascontiguousarray(R_old @ v_proj, dtype=dump_dtype)
        H, rel_H, y2_sq, y4_4 = response_entropy_stats(response)
        print(f"v{j + 1}_proj_old_row_response:")
        print(np.array2string(response, precision=8, suppress_small=False, max_line_width=120))
        print(
            f"v{j + 1}_proj_old_row_response_stats: "
            f"h={H:.12f} rel_h={rel_H:.12f} norm2_sq={y2_sq:.12f} norm4_4={y4_4:.12f}"
        )


def select_old_row_memory(seen_rows, V_r, max_rows, rng, return_indices=False):
    if max_rows is None or int(max_rows) <= 0:
        return (None, None) if return_indices else None
    A_seen = np.asarray(seen_rows)
    if A_seen.size == 0:
        return (None, None) if return_indices else None
    max_rows = int(max_rows)
    if A_seen.shape[0] <= max_rows:
        out = np.ascontiguousarray(A_seen.copy())
        idx = np.arange(A_seen.shape[0], dtype=int)
        return (out, idx) if return_indices else out

    uniform_count = max(1, max_rows // 2)
    high_count = max_rows - uniform_count
    uniform_idx = rng.choice(A_seen.shape[0], size=uniform_count, replace=False)

    if high_count > 0:
        if V_r is not None and np.asarray(V_r).size:
            responses = A_seen @ np.asarray(V_r)
            high_scores = np.sum(responses * responses, axis=1)
        else:
            high_scores = np.sum(A_seen * A_seen, axis=1)
        high_idx = np.argsort(high_scores)[-high_count:]
        idx = np.unique(np.concatenate([uniform_idx, high_idx]))
        if idx.size < max_rows:
            remaining = np.setdiff1d(np.arange(A_seen.shape[0]), idx, assume_unique=False)
            fill = rng.choice(remaining, size=max_rows - idx.size, replace=False)
            idx = np.concatenate([idx, fill])
    else:
        idx = uniform_idx
    idx = np.asarray(idx[:max_rows], dtype=int)
    out = np.ascontiguousarray(A_seen[idx, :].copy())
    return (out, idx) if return_indices else out


def combined_score_grad_reduced(B, z, rows_block, rows_ref):
    B_arr = np.asarray(B)
    work_dtype = B_arr.dtype
    z = np.ascontiguousarray(np.asarray(z, dtype=work_dtype).reshape(-1))
    y = np.ascontiguousarray(B_arr @ z, dtype=work_dtype)
    y2_sq = max(float(np.dot(y, y)), 1e-30)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 1e-30)
    rows_block = max(int(rows_block), 2)
    rows_ref = max(int(rows_ref), rows_block)
    c = np.log(rows_block / rows_ref) / (2.0 * np.log(rows_block))

    log_phi = c * (np.log(y4_4) - 2.0 * np.log(y2_sq))
    score = float(np.exp(np.log(y2_sq) + log_phi))
    y3 = np.ascontiguousarray(y * y * y, dtype=work_dtype)
    g2 = np.ascontiguousarray(B_arr.T @ y, dtype=work_dtype) / y2_sq
    g4 = np.ascontiguousarray(B_arr.T @ y3, dtype=work_dtype) / y4_4
    grad = np.ascontiguousarray(score * ((2.0 - 4.0 * c) * g2 + 4.0 * c * g4), dtype=work_dtype)
    H = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    s = float(np.sqrt(y2_sq))
    return score, grad, s, H


def combined_streaming_score_grad_reduced(
    B_gain, B_block, R_old_block, z, rows_block, rows_ref, n_old
):
    B_gain_arr = np.asarray(B_gain)
    B_block_arr = np.asarray(B_block)
    work_dtype = B_gain_arr.dtype
    z = np.ascontiguousarray(np.asarray(z, dtype=work_dtype).reshape(-1))

    gain_vec = np.ascontiguousarray(B_gain_arr @ z, dtype=work_dtype)
    gain2 = max(float(np.dot(gain_vec, gain_vec)), 1e-30)
    grad_energy = np.ascontiguousarray(2.0 * (B_gain_arr.T @ gain_vec), dtype=work_dtype)

    y = np.ascontiguousarray(B_block_arr @ z, dtype=work_dtype)
    y2_sq = max(float(np.dot(y, y)), 0.0)
    y4_4 = max(float(np.sum((y * y) * (y * y))), 0.0)
    cy = np.ascontiguousarray(B_block_arr.T @ y, dtype=work_dtype)
    y3 = np.ascontiguousarray(y * y * y, dtype=work_dtype)
    cy3 = np.ascontiguousarray(B_block_arr.T @ y3, dtype=work_dtype)
    rows_entropy = max(int(rows_block), 0)

    R_old_arr = None if R_old_block is None else np.asarray(R_old_block, dtype=work_dtype)
    if R_old_arr is not None and R_old_arr.size and R_old_arr.shape[0] > 0:
        r = np.ascontiguousarray(R_old_arr @ z, dtype=work_dtype)
        y2_sq += max(float(np.dot(r, r)), 0.0)
        y4_4 += max(float(np.sum((r * r) * (r * r))), 0.0)
        cy = np.ascontiguousarray(cy + R_old_arr.T @ r, dtype=work_dtype)
        r3 = np.ascontiguousarray(r * r * r, dtype=work_dtype)
        cy3 = np.ascontiguousarray(cy3 + R_old_arr.T @ r3, dtype=work_dtype)
        rows_entropy += int(R_old_arr.shape[0])

    y2_sq = max(y2_sq, 1e-30)
    y4_4 = max(y4_4, 1e-30)
    rows_entropy = max(rows_entropy, 2)
    rows_ref = max(int(rows_ref), rows_entropy)
    rows_seen = min(max(int(n_old) + int(rows_block), 1), rows_ref)
    c = np.log(rows_seen / rows_ref) / (2.0 * np.log(rows_entropy))

    log_phi = c * (np.log(y4_4) - 2.0 * np.log(y2_sq))
    phi = float(np.exp(log_phi))
    score = max(float(gain2 * phi), 1e-30)
    grad_log_phi = np.ascontiguousarray(4.0 * c * (cy3 / y4_4 - cy / y2_sq), dtype=work_dtype)
    grad = np.ascontiguousarray(phi * grad_energy + score * grad_log_phi, dtype=work_dtype)
    H = -(np.log(y4_4) - 2.0 * np.log(y2_sq))
    s = float(np.sqrt(gain2))
    return score, grad, s, H


def combined_full_gradient_residual(
    M_gain, A_block, v, Vred, state_prev, rows_ref,
    old_row_memory=None, Q_prev=None, return_vector=False
):
    rows_block = np.asarray(A_block).shape[0]
    if state_prev is None:
        _, grad, _, _ = combined_score_grad_reduced(A_block, v, rows_block, rows_ref)
    else:
        n_old = int(state_prev.get("rows_seen", 0))
        _, grad, _, _ = combined_streaming_score_grad_reduced(
            M_gain, A_block, old_row_memory, v, rows_block, rows_ref, n_old
        )
    dtype = np.result_type(np.asarray(grad).dtype, np.asarray(v).dtype)
    grad = np.ascontiguousarray(np.asarray(grad, dtype=dtype).reshape(-1))
    v = np.ascontiguousarray(np.asarray(v, dtype=dtype).reshape(-1))

    g_tan = grad - v * float(v @ grad)
    if Q_prev is not None and np.asarray(Q_prev).size:
        Q_prev_arr = np.ascontiguousarray(np.asarray(Q_prev, dtype=dtype))
        g_tan = g_tan - Q_prev_arr @ (Q_prev_arr.T @ g_tan)

    Vred_arr = np.ascontiguousarray(np.asarray(Vred, dtype=dtype))
    r = g_tan - Vred_arr @ (Vred_arr.T @ g_tan)
    r = np.ascontiguousarray(r, dtype=dtype)
    r_norm = float(np.linalg.norm(r))
    g_norm = float(np.linalg.norm(g_tan))
    if return_vector:
        return r_norm, g_norm, r
    return r_norm, g_norm


def basic_projected_ascent_single_reduced_combined_cex(
    B, z0, Qz, rows_block, rows_ref, maxit=80, tol=1e-8,
    reuse_line_search_grad=True
):
    z = retract_reduced(z0, Qz)
    if z is None:
        raise RuntimeError("Initial reduced seed is infeasible.")

    score, grad, s, H = combined_score_grad_reduced(B, z, rows_block, rows_ref)
    stop = {"reason": "maxit", "iters": maxit, "grad_norm": np.nan}
    progress_f_tol = 1e-12
    progress_step_tol = 1e-10

    for it in range(maxit):
        gtan = project_reduced(grad - z * float(z @ grad), Qz)
        gnorm = float(np.linalg.norm(gtan))
        if gnorm <= tol:
            stop = {"reason": "grad_tol", "iters": it, "grad_norm": gnorm}
            break

        accepted = False
        accepted_eval = None
        alpha = 1.0
        score_old = score
        z_old = z

        for ls_iter in range(20):
            zt = retract_reduced(z + alpha * gtan, Qz)
            if zt is not None:
                score_t, grad_t, s_t, H_t = combined_score_grad_reduced(
                    B, zt, rows_block, rows_ref
                )
                rhs = score_old + 1e-4 * alpha * float(gtan @ gtan)
                if score_t >= rhs:
                    z = zt
                    accepted_eval = (score_t, grad_t, s_t, H_t)
                    accepted = True
                    break
            alpha *= 0.5

        if not accepted:
            stop = {
                "reason": "line_search_fail",
                "iters": it + 1,
                "grad_norm": gnorm,
                "line_search_steps": 20,
            }
            z = z_old
            break

        if reuse_line_search_grad and accepted_eval is not None:
            score, grad, s, H = accepted_eval
        else:
            score, grad, s, H = combined_score_grad_reduced(B, z, rows_block, rows_ref)
        step_norm = float(np.linalg.norm(z - z_old))
        f_change = abs(score - score_old)
        f_threshold = progress_f_tol * max(1.0, abs(score_old))
        stop = {
            "reason": "progress",
            "iters": it + 1,
            "grad_norm": gnorm,
            "step_norm": step_norm,
            "f_change": f_change,
            "f_threshold": f_threshold,
            "line_search_alpha": alpha,
            "line_search_steps": ls_iter + 1,
        }
        if f_change <= f_threshold:
            stop["reason"] = "f_change_tol"
            break
        if step_norm <= progress_step_tol:
            stop["reason"] = "step_tol"
            break
    else:
        gtan = project_reduced(grad - z * float(z @ grad), Qz)
        stop = {"reason": "maxit", "iters": maxit, "grad_norm": float(np.linalg.norm(gtan))}

    return z, score, s, H, stop


def basic_projected_ascent_single_reduced_streaming_combined_cex(
    B_gain, B_block, R_old_block, z0, Qz, rows_block, rows_ref, n_old,
    maxit=80, tol=1e-8, reuse_line_search_grad=True
):
    z = retract_reduced(z0, Qz)
    if z is None:
        raise RuntimeError("Initial reduced seed is infeasible.")

    score, grad, s, H = combined_streaming_score_grad_reduced(
        B_gain, B_block, R_old_block, z, rows_block, rows_ref, n_old
    )
    stop = {"reason": "maxit", "iters": maxit, "grad_norm": np.nan}
    progress_f_tol = 1e-12
    progress_step_tol = 1e-10

    for it in range(maxit):
        gtan = project_reduced(grad - z * float(z @ grad), Qz)
        gnorm = float(np.linalg.norm(gtan))
        if gnorm <= tol:
            stop = {"reason": "grad_tol", "iters": it, "grad_norm": gnorm}
            break

        accepted = False
        accepted_eval = None
        alpha = 1.0
        score_old = score
        z_old = z

        for ls_iter in range(20):
            zt = retract_reduced(z + alpha * gtan, Qz)
            if zt is not None:
                score_t, grad_t, s_t, H_t = combined_streaming_score_grad_reduced(
                    B_gain, B_block, R_old_block, zt, rows_block, rows_ref, n_old
                )
                rhs = score_old + 1e-4 * alpha * float(gtan @ gtan)
                if score_t >= rhs:
                    z = zt
                    accepted_eval = (score_t, grad_t, s_t, H_t)
                    accepted = True
                    break
            alpha *= 0.5

        if not accepted:
            stop = {
                "reason": "line_search_fail",
                "iters": it + 1,
                "grad_norm": gnorm,
                "line_search_steps": 20,
            }
            z = z_old
            break

        if reuse_line_search_grad and accepted_eval is not None:
            score, grad, s, H = accepted_eval
        else:
            score, grad, s, H = combined_streaming_score_grad_reduced(
                B_gain, B_block, R_old_block, z, rows_block, rows_ref, n_old
            )
        step_norm = float(np.linalg.norm(z - z_old))
        f_change = abs(score - score_old)
        f_threshold = progress_f_tol * max(1.0, abs(score_old))
        stop = {
            "reason": "progress",
            "iters": it + 1,
            "grad_norm": gnorm,
            "step_norm": step_norm,
            "f_change": f_change,
            "f_threshold": f_threshold,
            "line_search_alpha": alpha,
            "line_search_steps": ls_iter + 1,
        }
        if f_change <= f_threshold:
            stop["reason"] = "f_change_tol"
            break
        if step_norm <= progress_step_tol:
            stop["reason"] = "step_tol"
            break
    else:
        gtan = project_reduced(grad - z * float(z @ grad), Qz)
        stop = {"reason": "maxit", "iters": maxit, "grad_norm": float(np.linalg.norm(gtan))}

    return z, score, s, H, stop


def basic_projected_ascent_single_reduced(B, z0, Qz, win, ncols, maxit=80, tol=1e-8):
    z = retract_reduced(z0, Qz)
    if z is None:
        raise RuntimeError("Initial reduced seed is infeasible.")

    logf, grad, s, H = entropy_logscore_grad_reduced(B, z, win, ncols)
    stop = {"reason": "maxit", "iters": maxit, "grad_norm": np.nan}

    for it in range(maxit):
        gtan = project_reduced(grad - z * float(z @ grad), Qz)
        gnorm = float(np.linalg.norm(gtan))
        if gnorm <= tol:
            stop = {"reason": "grad_tol", "iters": it, "grad_norm": gnorm}
            break

        step = gtan / max(gnorm, 1e-30)
        alpha = 1.0
        improved = False

        for _ in range(20):
            zt = retract_reduced(z + alpha * step, Qz)
            if zt is None:
                alpha *= 0.5
                continue
            logf_t, grad_t, s_t, H_t = entropy_logscore_grad_reduced(B, zt, win, ncols)
            if logf_t > logf:
                z, logf, grad, s, H = zt, logf_t, grad_t, s_t, H_t
                improved = True
                break
            alpha *= 0.5

        if not improved:
            stop = {"reason": "line_search_fail", "iters": it + 1, "grad_norm": gnorm}
            break
    else:
        stop = {"reason": "maxit", "iters": maxit, "grad_norm": float(np.linalg.norm(project_reduced(grad - z * float(z @ grad), Qz)))}

    return z, logf, s, H, stop


def basic_projected_ascent_single_reduced_streaming(B_gain, B_block, C_prev, s2_old, q_old, z0, Qz, rows_total, ncols,
                                                    maxit=80, tol=1e-8):
    z = retract_reduced(z0, Qz)
    if z is None:
        raise RuntimeError("Initial reduced seed is infeasible.")

    logf, grad, s, H = entropy_streaming_logscore_grad_reduced(
        B_gain, B_block, C_prev, s2_old, q_old, z, rows_total, ncols
    )
    stop = {"reason": "maxit", "iters": maxit, "grad_norm": np.nan}

    for it in range(maxit):
        gtan = project_reduced(grad - z * float(z @ grad), Qz)
        gnorm = float(np.linalg.norm(gtan))
        if gnorm <= tol:
            stop = {"reason": "grad_tol", "iters": it, "grad_norm": gnorm}
            break

        step = gtan / max(gnorm, 1e-30)
        alpha = 1.0
        improved = False

        for _ in range(20):
            zt = retract_reduced(z + alpha * step, Qz)
            if zt is None:
                alpha *= 0.5
                continue
            logf_t, grad_t, s_t, H_t = entropy_streaming_logscore_grad_reduced(
                B_gain, B_block, C_prev, s2_old, q_old, zt, rows_total, ncols
            )
            if logf_t > logf:
                z, logf, grad, s, H = zt, logf_t, grad_t, s_t, H_t
                improved = True
                break
            alpha *= 0.5

        if not improved:
            stop = {"reason": "line_search_fail", "iters": it + 1, "grad_norm": gnorm}
            break
    else:
        stop = {"reason": "maxit", "iters": maxit, "grad_norm": float(np.linalg.norm(project_reduced(grad - z * float(z @ grad), Qz)))}

    return z, logf, s, H, stop


def entropy_full_gradient_residual(M, v, Vred, win, ncols):
    _, grad, _, _ = entropy_logscore_grad_rows(M, v, win, ncols)
    proj = Vred @ (Vred.T @ grad)
    r = grad - proj
    return float(np.linalg.norm(r)), float(np.linalg.norm(grad))


def orthonormalize_columns(X, dtype=None, eps=1e-12):
    X_arr = np.asarray(X if X is not None else np.zeros((0, 0)))
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
    if X_arr.size == 0:
        rows = X_arr.shape[0] if X_arr.ndim == 2 else 0
        out_dtype = np.float32 if dtype is None else dtype
        return np.zeros((rows, 0), dtype=out_dtype)
    if dtype is None:
        dtype = X_arr.dtype
    Q, R = np.linalg.qr(np.asarray(X_arr, dtype=np.float64), mode="reduced")
    keep = np.abs(np.diag(R)) > eps
    if not np.any(keep):
        return np.zeros((X_arr.shape[0], 0), dtype=dtype)
    return np.ascontiguousarray(Q[:, keep], dtype=dtype)


def append_basis_columns(Vbasis, X_new, max_cols=None, eps=1e-12):
    X_arr = np.asarray(X_new if X_new is not None else np.zeros((Vbasis.shape[0], 0), dtype=Vbasis.dtype))
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
    if X_arr.size == 0:
        return Vbasis
    if max_cols is not None:
        remaining = max(0, int(max_cols) - Vbasis.shape[1])
        if remaining == 0:
            return np.ascontiguousarray(Vbasis, dtype=Vbasis.dtype)
        X_arr = X_arr[:, :remaining]
    combined = np.column_stack([Vbasis, X_arr]) if Vbasis.size else X_arr
    Q = orthonormalize_columns(combined, dtype=Vbasis.dtype, eps=eps)
    if max_cols is not None and Q.shape[1] > max_cols:
        Q = Q[:, :max_cols]
    return np.ascontiguousarray(Q, dtype=Vbasis.dtype)


_ENTROPYSCORE_COMBINED_HYBRID_METHODS = {
    "entropyscore_combined": {
        "aux_method": None,
        "rank_ratio": None,
        "score_variant": "combined",
    },
    "entropyscore_hybrid": {
        "aux_method": "deflated_svd",
        "rank_ratio": 0.5,
        "score_variant": "combined",
    },
}


def resolve_entropyscore_combined_hybrid(method):
    return _ENTROPYSCORE_COMBINED_HYBRID_METHODS.get(method)


def entropyscore_combined_hybrid_score_rank(method, rank):
    cfg = resolve_entropyscore_combined_hybrid(method)
    if cfg is None:
        return None
    ratio = cfg.get("rank_ratio")
    if ratio is None:
        return None
    return max(0, min(int(rank), int(round(float(ratio) * int(rank)))))


_FUTURE_HMEAN_ONLINE_HYBRID_METHODS = {
    "future_hmean_online_svd_aux": {
        "aux_method": "svd",
        "rank_ratio": None,
    },
    "future_hmean_online_deflated_svd_aux": {
        "aux_method": "deflated_svd",
        "rank_ratio": None,
    },
    "future_hmean_online_hybrid": {
        "aux_method": "deflated_svd",
        "rank_ratio": 0.2,
    },
}


def resolve_future_hmean_online_hybrid(method):
    return _FUTURE_HMEAN_ONLINE_HYBRID_METHODS.get(method)


def future_hmean_online_hybrid_score_rank(method, rank):
    cfg = resolve_future_hmean_online_hybrid(method)
    if cfg is None:
        return None
    ratio = cfg.get("rank_ratio")
    if ratio is None:
        return None
    return max(0, min(int(rank), int(round(float(ratio) * int(rank)))))


def is_future_hmean_online_method(method):
    return method == "future_hmean_online" or resolve_future_hmean_online_hybrid(method) is not None


def resolve_entropyscore_forget_aux_method(method):
    if method == "entropyscore_forget":
        return None
    if method == "entropyscore_forget_svd_aux":
        return "svd"
    if method == "entropyscore_forget_deflated_svd_aux":
        return "deflated_svd"
    cfg = resolve_entropyscore_combined_hybrid(method)
    if cfg is not None:
        return cfg["aux_method"]
    return None


def top_right_singular_vectors(M, rank, dtype=np.float32):
    M_arr = np.asarray(M, dtype=dtype)
    rank = int(min(max(rank, 0), min(M_arr.shape)))
    if rank <= 0:
        return np.zeros((M_arr.shape[1], 0), dtype=M_arr.dtype)
    _, _, vh = np.linalg.svd(M_arr, full_matrices=False)
    return np.ascontiguousarray(vh[:rank, :].T, dtype=M_arr.dtype)


def build_entropyscore_forget_direction_basis(M_gain, V_score, total_rank, aux_method=None):
    M_arr = np.asarray(M_gain, dtype=np.float32)
    total_rank = int(min(max(total_rank, 0), min(M_arr.shape)))
    V_score_orth = orthonormalize_columns(V_score, dtype=M_arr.dtype)
    if V_score_orth.shape[1] >= total_rank or aux_method is None:
        return np.ascontiguousarray(V_score_orth[:, :total_rank], dtype=M_arr.dtype)

    aux_rank = total_rank - V_score_orth.shape[1]
    if aux_method == "svd":
        M_aux = M_arr
    elif aux_method == "deflated_svd":
        if V_score_orth.size:
            M_aux = M_arr - (M_arr @ V_score_orth) @ V_score_orth.T
        else:
            M_aux = M_arr
    else:
        raise ValueError(f"Unknown EntropyScoreForget auxiliary method: {aux_method}")

    V_aux_candidates = top_right_singular_vectors(M_aux, min(M_aux.shape), dtype=M_arr.dtype)
    V_full = append_basis_columns(V_score_orth, V_aux_candidates, max_cols=total_rank)
    if V_full.shape[1] < total_rank and aux_method != "svd":
        V_fallback = top_right_singular_vectors(M_arr, min(M_arr.shape), dtype=M_arr.dtype)
        V_full = append_basis_columns(V_full, V_fallback, max_cols=total_rank)
    return np.ascontiguousarray(V_full[:, :total_rank], dtype=M_arr.dtype)


def entropy_iter_basis_expansion(M_gain, active_r, win, ncols, V_init=None, q0=5, qmax=None,
                                 krylov_depth=2, residual_tol=1e-2, expansion_maxit=8,
                                 num_restarts=3, maxit=40, tol=1e-8, rng=None, verbose=True,
                                 state_prev=None, A_block=None, rows_total=None):
    if rng is None:
        rng = np.random.default_rng(0)

    M_arr = np.asarray(M_gain, dtype=np.float32)
    is_initial_block = state_prev is None
    if not is_initial_block and (A_block is None or rows_total is None):
        raise ValueError("A_block and rows_total are required for expansion streaming entropy history.")

    if active_r <= 0:
        empty = np.zeros((M_arr.shape[1], 0), dtype=M_arr.dtype)
        return empty, np.zeros(0), np.zeros(0), np.zeros(0), {
            "seed_rank": 0,
            "max_rank": 0,
            "krylov_depth": int(krylov_depth),
            "residual_tol": float(residual_tol),
            "subspace_dims": [],
            "expansion_iters": [],
            "grad_perp_ratio": np.zeros(0),
        }

    min_dim = min(M_arr.shape)
    q0_eff = max(1, min(int(q0), M_arr.shape[1], min_dim))
    q0_eff = max(active_r, q0_eff)
    q0_eff = min(q0_eff, M_arr.shape[1], min_dim)
    if qmax is None:
        qmax = min(M_arr.shape[1], max(4 * q0_eff, q0_eff + 4 * max(1, krylov_depth) * active_r, 32))
    qmax = max(q0_eff, min(int(qmax), M_arr.shape[1]))

    t0 = time.time()
    if q0_eff >= min_dim:
        _, _, vh = np.linalg.svd(M_arr, full_matrices=False)
        Vseed = np.ascontiguousarray(vh[:q0_eff, :].T, dtype=M_arr.dtype)
    else:
        Vseed, _, _ = build_entropy_fast_subspace(
            M_arr, active_r=min(active_r, max(1, q0_eff)), q_subspace=q0_eff, method="lanczos", dtype=M_arr.dtype
        )
    Vbasis = np.ascontiguousarray(Vseed, dtype=M_arr.dtype)
    subspace_build_time = time.time() - t0

    prev_basis = None
    prev_s2 = None
    prev_q = None
    A_block_arr = None if A_block is None else np.asarray(A_block, dtype=M_arr.dtype)
    if not is_initial_block:
        prev_basis = np.ascontiguousarray(np.asarray(state_prev["V"], dtype=M_arr.dtype))
        prev_s2 = np.asarray(state_prev["s2"], dtype=M_arr.dtype)
        prev_q = np.asarray(state_prev["q"], dtype=M_arr.dtype)

    V_out = np.zeros((M_arr.shape[1], active_r), dtype=M_arr.dtype)
    s_out = np.zeros(active_r, dtype=float)
    H_out = np.zeros(active_r, dtype=float)
    score_out = np.zeros(active_r, dtype=float)
    grad_perp_ratio = np.zeros(active_r, dtype=float)
    subspace_dims = []
    expansion_iters = []
    timing_totals = {
        "reduced_setup": 0.0,
        "reduced_opt": 0.0,
        "full_gradient": 0.0,
        "expansion_matvec": 0.0,
        "expansion_append": 0.0,
    }
    timing_counts = {
        "basis_solves": 0,
        "restart_solves": 0,
        "full_gradient_evals": 0,
        "expansion_steps": 0,
    }

    if verbose:
        print({
            "EntropyScoreExpansion_setup": {
                "M_gain_shape": M_arr.shape,
                "active_rank": active_r,
                "seed_rank": q0_eff,
                "max_rank": qmax,
                "krylov_depth": int(krylov_depth),
                "residual_tol": float(residual_tol),
                "subspace_build_time": subspace_build_time,
            }
        })

    solve_t0 = time.time()
    V_init_work = None if V_init is None else np.ascontiguousarray(np.asarray(V_init, dtype=M_arr.dtype))
    prior_basis = orthonormalize_columns(V_init_work, dtype=M_arr.dtype) if V_init_work is not None and V_init_work.size else None

    for k_idx in range(active_r):
        basis_t0 = time.time()
        z_warm = None
        stop_reason = "max_expansion"
        best_restart = None
        best_stop = None
        expansion_count = 0
        v_best = None
        s_best = 0.0
        H_best = np.inf
        logf_best = -np.inf

        while True:
            t_stage = time.perf_counter()
            B_gain = np.ascontiguousarray(M_arr @ Vbasis, dtype=Vbasis.dtype)
            C_prev = None if is_initial_block else np.ascontiguousarray(prev_basis.T @ Vbasis, dtype=Vbasis.dtype)
            B_block = None if is_initial_block else np.ascontiguousarray(A_block_arr @ Vbasis, dtype=Vbasis.dtype)
            q = Vbasis.shape[1]
            Qz = np.ascontiguousarray(Vbasis.T @ V_out[:, :k_idx], dtype=Vbasis.dtype) if k_idx > 0 else np.zeros((q, 0), dtype=Vbasis.dtype)
            if k_idx > 0:
                Qz = orthonormalize_columns(Qz, dtype=Vbasis.dtype)

            starts = []
            if z_warm is not None:
                append_unique_reduced_seed(starts, retract_reduced(z_warm, Qz))

            if prior_basis is not None and prior_basis.shape[1] > k_idx:
                z_prior = np.ascontiguousarray(Vbasis.T @ prior_basis[:, k_idx], dtype=Vbasis.dtype)
                append_unique_reduced_seed(starts, retract_reduced(z_prior, Qz))

            if k_idx < q:
                e = np.zeros(q, dtype=Vbasis.dtype)
                e[k_idx] = 1.0
                append_unique_reduced_seed(starts, retract_reduced(e, Qz))

            while len(starts) < max(1, num_restarts):
                zrand = np.ascontiguousarray(rng.standard_normal(q), dtype=Vbasis.dtype)
                append_unique_reduced_seed(starts, retract_reduced(zrand, Qz))
                if len(starts) == 0 and q == 0:
                    raise RuntimeError("Expansion basis became empty.")
            timing_totals["reduced_setup"] += time.perf_counter() - t_stage

            t_stage = time.perf_counter()
            cand_results = []
            for z0 in starts:
                if is_initial_block:
                    cand = basic_projected_ascent_single_reduced(B_gain, z0, Qz, win, ncols, maxit=maxit, tol=tol)
                else:
                    cand = basic_projected_ascent_single_reduced_streaming(
                        B_gain, B_block, C_prev, prev_s2, prev_q, z0, Qz, rows_total, ncols, maxit=maxit, tol=tol
                    )
                cand_results.append(cand)
            timing_totals["reduced_opt"] += time.perf_counter() - t_stage
            timing_counts["restart_solves"] += len(starts)

            best = None
            for restart_idx, cand in enumerate(cand_results):
                if best is None or cand[1] > best[1]:
                    best = cand
                    best_restart = restart_idx + 1

            z_best, logf_best, s_best, H_best, best_stop = best
            v_best = np.ascontiguousarray(Vbasis @ z_best, dtype=Vbasis.dtype)
            v_best = np.ascontiguousarray(v_best / max(np.linalg.norm(v_best), 1e-30), dtype=Vbasis.dtype)

            t_stage = time.perf_counter()
            if is_initial_block:
                _, g_full_vec, _, _ = entropy_logscore_grad_rows(M_arr, v_best, win, ncols)
            else:
                _, g_full_vec, _, _ = entropy_streaming_logscore_grad(
                    M_arr, A_block_arr, prev_basis, prev_s2, prev_q, v_best, rows_total, ncols
                )
            proj = Vbasis @ (Vbasis.T @ g_full_vec)
            r_k = np.ascontiguousarray(g_full_vec - proj, dtype=Vbasis.dtype)
            g_full_norm = float(np.linalg.norm(g_full_vec))
            r_norm = float(np.linalg.norm(r_k))
            grad_ratio = r_norm / max(g_full_norm, 1e-30)
            timing_totals["full_gradient"] += time.perf_counter() - t_stage
            timing_counts["full_gradient_evals"] += 1

            if grad_ratio <= residual_tol:
                stop_reason = "subspace_grad_tol"
                break
            if Vbasis.shape[1] >= qmax:
                stop_reason = "subspace_rank_cap"
                break
            if expansion_count >= expansion_maxit:
                stop_reason = "expansion_maxit"
                break

            new_cols = [r_k]
            g_dir = np.ascontiguousarray(r_k / max(r_norm, 1e-30), dtype=Vbasis.dtype)
            t_stage = time.perf_counter()
            for _ in range(max(0, int(krylov_depth) - 1)):
                g_dir = np.ascontiguousarray(M_arr.T @ (M_arr @ g_dir), dtype=Vbasis.dtype)
                new_cols.append(g_dir)
            timing_totals["expansion_matvec"] += time.perf_counter() - t_stage

            prev_qdim = Vbasis.shape[1]
            t_stage = time.perf_counter()
            Vbasis = append_basis_columns(Vbasis, np.column_stack(new_cols), max_cols=qmax)
            timing_totals["expansion_append"] += time.perf_counter() - t_stage
            if Vbasis.shape[1] == prev_qdim:
                stop_reason = "no_expandable_direction"
                break

            z_warm = np.zeros(Vbasis.shape[1], dtype=Vbasis.dtype)
            z_warm[:z_best.shape[0]] = z_best
            expansion_count += 1
            timing_counts["expansion_steps"] += 1

        V_out[:, k_idx] = v_best
        s_out[k_idx] = s_best
        H_out[k_idx] = H_best
        score_out[k_idx] = float(np.exp(logf_best))
        grad_perp_ratio[k_idx] = grad_ratio
        subspace_dims.append(int(Vbasis.shape[1]))
        expansion_iters.append(int(expansion_count))
        timing_counts["basis_solves"] += 1

        if verbose and ((k_idx < 10) or ((k_idx + 1) % 25 == 0) or (k_idx + 1 == active_r)):
            print({
                "basis": k_idx + 1,
                "best_restart": best_restart,
                "stop_reason": stop_reason,
                "solver_stop_reason": None if best_stop is None else best_stop["reason"],
                "iters": None if best_stop is None else best_stop["iters"],
                "grad_norm": None if best_stop is None else best_stop["grad_norm"],
                "subspace_dim": int(Vbasis.shape[1]),
                "expansions": int(expansion_count),
                "s": float(s_best),
                "H": float(H_best),
                "time": time.time() - basis_t0,
                "grad_perp_ratio": float(grad_perp_ratio[k_idx]),
            })

    solve_time = time.time() - solve_t0
    diag = {
        "seed_rank": q0_eff,
        "max_rank": qmax,
        "krylov_depth": int(krylov_depth),
        "residual_tol": float(residual_tol),
        "subspace_build_time": subspace_build_time,
        "reduced_solve_time": solve_time,
        "grad_perp_ratio": grad_perp_ratio,
        "subspace_dims": np.asarray(subspace_dims, dtype=int),
        "expansion_iters": np.asarray(expansion_iters, dtype=int),
        "timing_totals": dict(timing_totals),
        "timing_counts": dict(timing_counts),
        "Vbasis_final": Vbasis,
    }
    return V_out, s_out, H_out, score_out, diag


def entropy_iter_basis_forget(M_gain, active_r, rows_ref, V_init=None, q0=5, qmax=None,
                              krylov_depth=2, residual_tol=1e-2, expansion_maxit=8,
                              num_restarts=3, maxit=40, tol=1e-8, rng=None, verbose=True,
                              state_prev=None, A_block=None, rows_total=None,
                              reduced_optimizer="legacy", work_dtype=np.float32,
                              expansion_direction="krylov_v",
                              reuse_line_search_grad=True,
                              expansion_warm_start=False,
                              post_expansion_maxit=None,
                              score_variant="forget",
                              old_row_memory=None):
    del rows_total
    if rng is None:
        rng = np.random.default_rng(0)

    if reduced_optimizer not in {"legacy", "cex"}:
        raise ValueError(f"Unknown reduced_optimizer: {reduced_optimizer}")
    if expansion_direction not in {"krylov_v", "residual"}:
        raise ValueError(f"Unknown expansion_direction: {expansion_direction}")
    if score_variant not in {"forget", "combined"}:
        raise ValueError(f"Unknown score_variant: {score_variant}")
    if score_variant == "combined" and reduced_optimizer != "cex":
        raise ValueError("score_variant='combined' requires reduced_optimizer='cex'.")
    work_dtype = np.dtype(work_dtype)

    M_arr = np.asarray(M_gain, dtype=work_dtype)
    A_block_arr = np.asarray(A_block, dtype=M_arr.dtype)
    is_initial_block = state_prev is None
    if A_block is None:
        raise ValueError("A_block is required for entropyscore_forget.")

    if active_r <= 0:
        empty = np.zeros((M_arr.shape[1], 0), dtype=M_arr.dtype)
        return empty, np.zeros(0), np.zeros(0), np.zeros(0), {
            "seed_rank": 0,
            "max_rank": 0,
            "krylov_depth": int(krylov_depth),
            "residual_tol": float(residual_tol),
            "reduced_optimizer": reduced_optimizer,
            "work_dtype": str(M_arr.dtype),
            "expansion_direction": expansion_direction,
            "reuse_line_search_grad": bool(reuse_line_search_grad),
            "expansion_warm_start": bool(expansion_warm_start),
            "post_expansion_maxit": None if post_expansion_maxit is None else int(post_expansion_maxit),
            "score_variant": score_variant,
            "old_row_memory_rows": 0,
            "subspace_build_time": 0.0,
            "reduced_solve_time": 0.0,
            "subspace_dims": np.zeros(0, dtype=int),
            "expansion_iters": np.zeros(0, dtype=int),
            "grad_perp_ratio": np.zeros(0),
            "timing_totals": {"reduced_setup": 0.0, "reduced_opt": 0.0, "full_gradient": 0.0,
                              "expansion_matvec": 0.0, "expansion_append": 0.0},
            "timing_counts": {"basis_solves": 0, "restart_solves": 0,
                              "full_gradient_evals": 0, "expansion_steps": 0},
            "Vbasis_final": np.zeros((M_arr.shape[1], 0), dtype=M_arr.dtype),
        }

    min_dim = min(M_arr.shape)
    q0_eff = max(1, min(int(q0), M_arr.shape[1], min_dim))
    q0_eff = max(active_r, q0_eff)
    q0_eff = min(q0_eff, M_arr.shape[1], min_dim)
    if qmax is None:
        qmax = min(M_arr.shape[1], max(4 * q0_eff, q0_eff + 4 * max(1, krylov_depth) * active_r, 32))
    qmax = max(q0_eff, min(int(qmax), M_arr.shape[1]))

    t0 = time.time()
    if q0_eff >= min_dim:
        _, _, vh = np.linalg.svd(M_arr, full_matrices=False)
        Vseed = np.ascontiguousarray(vh[:q0_eff, :].T, dtype=M_arr.dtype)
    else:
        Vseed, _, _ = build_entropy_fast_subspace(
            M_arr, active_r=min(active_r, max(1, q0_eff)), q_subspace=q0_eff, method="lanczos", dtype=M_arr.dtype
        )
    Vbasis = np.ascontiguousarray(Vseed, dtype=M_arr.dtype)
    subspace_build_time = time.time() - t0

    prev_basis = None
    prev_s2 = None
    if not is_initial_block:
        prev_basis = np.ascontiguousarray(np.asarray(state_prev["V"], dtype=M_arr.dtype))
        prev_s2 = np.asarray(state_prev["s2"], dtype=M_arr.dtype)

    old_row_memory_arr = None
    if old_row_memory is not None and np.asarray(old_row_memory).size:
        old_row_memory_arr = np.ascontiguousarray(np.asarray(old_row_memory, dtype=M_arr.dtype))
    n_old = 0 if is_initial_block else int(state_prev.get("rows_seen", 0))

    V_out = np.zeros((M_arr.shape[1], active_r), dtype=M_arr.dtype)
    s_out = np.zeros(active_r, dtype=float)
    H_out = np.zeros(active_r, dtype=float)
    score_out = np.zeros(active_r, dtype=float)
    grad_perp_ratio = np.zeros(active_r, dtype=float)
    subspace_dims = []
    expansion_iters = []
    timing_totals = {
        "reduced_setup": 0.0,
        "reduced_opt": 0.0,
        "full_gradient": 0.0,
        "expansion_matvec": 0.0,
        "expansion_append": 0.0,
    }
    timing_counts = {
        "basis_solves": 0,
        "restart_solves": 0,
        "full_gradient_evals": 0,
        "expansion_steps": 0,
    }

    if verbose:
        print({
            "EntropyScoreForget_setup": {
                "M_gain_shape": M_arr.shape,
                "A_block_shape": A_block_arr.shape,
                "active_rank": active_r,
                "seed_rank": q0_eff,
                "max_rank": qmax,
                "krylov_depth": int(krylov_depth),
                "residual_tol": float(residual_tol),
                "reduced_optimizer": reduced_optimizer,
                "work_dtype": str(M_arr.dtype),
                "expansion_direction": expansion_direction,
                "reuse_line_search_grad": bool(reuse_line_search_grad),
                "expansion_warm_start": bool(expansion_warm_start),
                "post_expansion_maxit": None if post_expansion_maxit is None else int(post_expansion_maxit),
                "subspace_build_time": subspace_build_time,
            }
        })

    solve_t0 = time.time()
    V_init_work = None if V_init is None else np.ascontiguousarray(np.asarray(V_init, dtype=M_arr.dtype))
    prior_basis = orthonormalize_columns(V_init_work, dtype=M_arr.dtype) if V_init_work is not None and V_init_work.size else None

    for k_idx in range(active_r):
        basis_t0 = time.time()
        z_warm = None
        stop_reason = "max_expansion"
        best_restart = None
        best_stop = None
        expansion_count = 0
        v_best = None
        s_best = 0.0
        H_best = np.inf
        logf_best = -np.inf

        while True:
            t_stage = time.perf_counter()
            B_gain = np.ascontiguousarray(M_arr @ Vbasis, dtype=Vbasis.dtype)
            B_block = np.ascontiguousarray(A_block_arr @ Vbasis, dtype=Vbasis.dtype)
            C_prev = None if is_initial_block else np.ascontiguousarray(prev_basis.T @ Vbasis, dtype=Vbasis.dtype)
            R_old_block = None if is_initial_block or old_row_memory_arr is None else (
                np.ascontiguousarray(old_row_memory_arr @ Vbasis, dtype=Vbasis.dtype)
            )
            q = Vbasis.shape[1]
            Qz = np.ascontiguousarray(Vbasis.T @ V_out[:, :k_idx], dtype=Vbasis.dtype) if k_idx > 0 else np.zeros((q, 0), dtype=Vbasis.dtype)
            if k_idx > 0:
                Qz = orthonormalize_columns(Qz, dtype=Vbasis.dtype)

            starts = []
            if reduced_optimizer == "cex":
                if expansion_warm_start and z_warm is not None:
                    append_unique_reduced_seed(starts, retract_reduced(z_warm, Qz))

                cex_restart_budget = max(0, max(1, num_restarts) - len(starts))
                if cex_restart_budget:
                    Q_full = np.ascontiguousarray(V_out[:, :k_idx], dtype=M_arr.dtype) if k_idx > 0 else np.zeros((M_arr.shape[1], 0), dtype=M_arr.dtype)
                    full_starts = make_basic_restart_seeds(
                        M_arr, Q_full, k_idx, V_init_work, cex_restart_budget
                    )
                    for v0 in full_starts:
                        z0 = np.ascontiguousarray(Vbasis.T @ np.asarray(v0, dtype=Vbasis.dtype), dtype=Vbasis.dtype)
                        append_unique_reduced_seed(starts, retract_reduced(z0, Qz))
            else:
                if z_warm is not None:
                    append_unique_reduced_seed(starts, retract_reduced(z_warm, Qz))

                if prior_basis is not None and prior_basis.shape[1] > k_idx:
                    z_prior = np.ascontiguousarray(Vbasis.T @ prior_basis[:, k_idx], dtype=Vbasis.dtype)
                    append_unique_reduced_seed(starts, retract_reduced(z_prior, Qz))

                if k_idx < q:
                    e = np.zeros(q, dtype=Vbasis.dtype)
                    e[k_idx] = 1.0
                    append_unique_reduced_seed(starts, retract_reduced(e, Qz))

            while len(starts) < max(1, num_restarts):
                zrand = np.ascontiguousarray(rng.standard_normal(q), dtype=Vbasis.dtype)
                append_unique_reduced_seed(starts, retract_reduced(zrand, Qz))
                if len(starts) == 0 and q == 0:
                    raise RuntimeError("Forget basis became empty.")
            timing_totals["reduced_setup"] += time.perf_counter() - t_stage

            t_stage = time.perf_counter()
            cand_results = []
            iter_budget = maxit
            if z_warm is not None and post_expansion_maxit is not None:
                iter_budget = max(1, min(int(maxit), int(post_expansion_maxit)))
            for z0 in starts:
                if is_initial_block:
                    if score_variant == "combined":
                        cand = basic_projected_ascent_single_reduced_combined_cex(
                            B_block, z0, Qz, A_block_arr.shape[0], rows_ref,
                            maxit=iter_budget, tol=tol,
                            reuse_line_search_grad=reuse_line_search_grad
                        )
                        cand = (cand[0], np.log(max(cand[1], 1e-300)), cand[2], cand[3], cand[4])
                    elif reduced_optimizer == "cex":
                        cand = basic_projected_ascent_single_reduced_forget_cex(
                            B_block, z0, Qz, A_block_arr.shape[0], rows_ref,
                            maxit=iter_budget, tol=tol,
                            reuse_line_search_grad=reuse_line_search_grad
                        )
                        cand = (cand[0], np.log(max(cand[1], 1e-300)), cand[2], cand[3], cand[4])
                    else:
                        cand = basic_projected_ascent_single_reduced_forget(
                            B_block, z0, Qz, A_block_arr.shape[0], rows_ref, maxit=iter_budget, tol=tol
                        )
                else:
                    if score_variant == "combined":
                        cand = basic_projected_ascent_single_reduced_streaming_combined_cex(
                            B_gain, B_block, R_old_block, z0, Qz,
                            A_block_arr.shape[0], rows_ref, n_old,
                            maxit=iter_budget, tol=tol,
                            reuse_line_search_grad=reuse_line_search_grad
                        )
                        cand = (cand[0], np.log(max(cand[1], 1e-300)), cand[2], cand[3], cand[4])
                    elif reduced_optimizer == "cex":
                        cand = basic_projected_ascent_single_reduced_streaming_forget_cex(
                            B_gain, B_block, C_prev, prev_s2, z0, Qz, A_block_arr.shape[0], rows_ref,
                            maxit=iter_budget, tol=tol,
                            reuse_line_search_grad=reuse_line_search_grad
                        )
                        cand = (cand[0], np.log(max(cand[1], 1e-300)), cand[2], cand[3], cand[4])
                    else:
                        cand = basic_projected_ascent_single_reduced_streaming_forget(
                            B_gain, B_block, C_prev, prev_s2, z0, Qz, A_block_arr.shape[0], rows_ref,
                            maxit=iter_budget, tol=tol
                        )
                cand_results.append(cand)
            timing_totals["reduced_opt"] += time.perf_counter() - t_stage
            timing_counts["restart_solves"] += len(starts)

            best = None
            for restart_idx, cand in enumerate(cand_results):
                if best is None or cand[1] > best[1]:
                    best = cand
                    best_restart = restart_idx + 1

            z_best, logf_best, s_best, H_best, best_stop = best
            v_best = np.ascontiguousarray(Vbasis @ z_best, dtype=Vbasis.dtype)
            v_best = np.ascontiguousarray(v_best / max(np.linalg.norm(v_best), 1e-30), dtype=Vbasis.dtype)

            t_stage = time.perf_counter()
            if score_variant == "combined":
                r_norm, g_full_norm, r_dir = combined_full_gradient_residual(
                    M_arr,
                    A_block_arr,
                    v_best,
                    Vbasis,
                    state_prev,
                    rows_ref,
                    old_row_memory=old_row_memory_arr,
                    Q_prev=V_out[:, :k_idx] if k_idx > 0 else None,
                    return_vector=True,
                )
            else:
                r_norm, g_full_norm, r_dir = entropyscore_forget_full_gradient_residual(
                    M_arr,
                    A_block_arr,
                    v_best,
                    Vbasis,
                    state_prev,
                    rows_ref,
                    Q_prev=V_out[:, :k_idx] if k_idx > 0 else None,
                    return_vector=True,
                )
            grad_ratio = r_norm / max(g_full_norm, 1e-30)
            timing_totals["full_gradient"] += time.perf_counter() - t_stage
            timing_counts["full_gradient_evals"] += 1

            if grad_ratio <= residual_tol:
                stop_reason = "subspace_grad_tol"
                break
            if Vbasis.shape[1] >= qmax:
                stop_reason = "subspace_rank_cap"
                break
            if expansion_count >= expansion_maxit:
                stop_reason = "expansion_maxit"
                break

            if expansion_direction == "residual":
                new_cols = [r_dir]
                g_dir = np.ascontiguousarray(r_dir, dtype=Vbasis.dtype)
            else:
                new_cols = [v_best]
                g_dir = np.ascontiguousarray(v_best, dtype=Vbasis.dtype)
            t_stage = time.perf_counter()
            for _ in range(max(0, int(krylov_depth) - 1)):
                g_dir = np.ascontiguousarray(M_arr.T @ (M_arr @ g_dir), dtype=Vbasis.dtype)
                new_cols.append(g_dir)
            timing_totals["expansion_matvec"] += time.perf_counter() - t_stage

            prev_qdim = Vbasis.shape[1]
            t_stage = time.perf_counter()
            Vbasis = append_basis_columns(Vbasis, np.column_stack(new_cols), max_cols=qmax)
            timing_totals["expansion_append"] += time.perf_counter() - t_stage
            if Vbasis.shape[1] == prev_qdim:
                stop_reason = "no_expandable_direction"
                break

            z_warm = np.zeros(Vbasis.shape[1], dtype=Vbasis.dtype)
            z_warm[:z_best.shape[0]] = z_best
            expansion_count += 1
            timing_counts["expansion_steps"] += 1

        V_out[:, k_idx] = v_best
        s_out[k_idx] = s_best
        H_out[k_idx] = H_best
        score_out[k_idx] = float(np.exp(logf_best))
        grad_perp_ratio[k_idx] = grad_ratio
        subspace_dims.append(int(Vbasis.shape[1]))
        expansion_iters.append(int(expansion_count))
        timing_counts["basis_solves"] += 1

        if verbose and ((k_idx < 10) or ((k_idx + 1) % 25 == 0) or (k_idx + 1 == active_r)):
            print({
                "basis": k_idx + 1,
                "best_restart": best_restart,
                "stop_reason": stop_reason,
                "solver_stop_reason": None if best_stop is None else best_stop["reason"],
                "iters": None if best_stop is None else best_stop["iters"],
                "grad_norm": None if best_stop is None else best_stop["grad_norm"],
                "subspace_dim": int(Vbasis.shape[1]),
                "expansions": int(expansion_count),
                "s": float(s_best),
                "H": float(H_best),
                "time": time.time() - basis_t0,
                "grad_perp_ratio": float(grad_perp_ratio[k_idx]),
            })

    solve_time = time.time() - solve_t0
    diag = {
        "seed_rank": q0_eff,
        "max_rank": qmax,
        "krylov_depth": int(krylov_depth),
        "residual_tol": float(residual_tol),
        "reduced_optimizer": reduced_optimizer,
        "work_dtype": str(M_arr.dtype),
        "expansion_direction": expansion_direction,
        "reuse_line_search_grad": bool(reuse_line_search_grad),
        "expansion_warm_start": bool(expansion_warm_start),
        "post_expansion_maxit": None if post_expansion_maxit is None else int(post_expansion_maxit),
        "score_variant": score_variant,
        "old_row_memory_rows": 0 if old_row_memory_arr is None else int(old_row_memory_arr.shape[0]),
        "subspace_build_time": subspace_build_time,
        "reduced_solve_time": solve_time,
        "grad_perp_ratio": grad_perp_ratio,
        "subspace_dims": np.asarray(subspace_dims, dtype=int),
        "expansion_iters": np.asarray(expansion_iters, dtype=int),
        "timing_totals": dict(timing_totals),
        "timing_counts": dict(timing_counts),
        "Vbasis_final": Vbasis,
    }
    return V_out, s_out, H_out, score_out, diag


def entropy_iter_basis_fast(M_gain, active_r, win, ncols, V_init=None, q_subspace=None,
                            subspace_method="lanczos", num_restarts=3, maxit=40, tol=1e-8,
                            rng=None, verbose=True, state_prev=None, A_block=None, rows_total=None):
    if rng is None:
        rng = np.random.default_rng(0)

    M_arr = np.asarray(M_gain, dtype=np.float32)
    t0 = time.time()
    Vred, Bred, sred = build_entropy_fast_subspace(
        M_arr, active_r=active_r, q_subspace=q_subspace, method=subspace_method, dtype=M_arr.dtype
    )
    is_initial_block = state_prev is None
    B_block_red = None
    C_prev = None
    prev_s2 = None
    prev_q = None
    if not is_initial_block:
        if A_block is None or rows_total is None:
            raise ValueError("A_block and rows_total are required for streaming entropy history.")
        B_block_red = np.ascontiguousarray(np.asarray(A_block, dtype=M_arr.dtype) @ Vred, dtype=Vred.dtype)
        prev_basis = np.ascontiguousarray(np.asarray(state_prev["V"], dtype=Vred.dtype))
        C_prev = np.ascontiguousarray(prev_basis.T @ Vred, dtype=Vred.dtype)
        prev_s2 = np.asarray(state_prev["s2"], dtype=Vred.dtype)
        prev_q = np.asarray(state_prev["q"], dtype=Vred.dtype)
    subspace_build_time = time.time() - t0
    q = Vred.shape[1]
    Qz = np.zeros((q, 0), dtype=Vred.dtype)

    V_out = np.zeros((M_arr.shape[1], active_r), dtype=Vred.dtype)
    s_out = np.zeros(active_r, dtype=float)
    H_out = np.zeros(active_r, dtype=float)
    score_out = np.zeros(active_r, dtype=float)
    grad_perp_ratio = np.zeros(active_r, dtype=float)

    if verbose:
        print({
            "EntropyScoreFast_setup": {
                "M_gain_shape": M_arr.shape,
                "active_rank": active_r,
                "subspace_method": subspace_method,
                "q_subspace": q,
                "subspace_build_time": subspace_build_time,
            }
        })

    solve_t0 = time.time()
    V_init_work = None if V_init is None else np.ascontiguousarray(np.asarray(V_init, dtype=Vred.dtype))
    prior_coeffs = None
    unused_prior_mask = None
    if V_init_work is not None and V_init_work.size:
        prior_coeffs = np.ascontiguousarray(Vred.T @ V_init_work, dtype=Vred.dtype)
        unused_prior_mask = np.ones(prior_coeffs.shape[1], dtype=bool)

    for k_idx in range(active_r):
        basis_t0 = time.time()
        starts = []
        if prior_coeffs is not None:
            for prior_idx in ordered_unused_indices(unused_prior_mask, k_idx):
                if len(starts) >= max(1, num_restarts):
                    break
                z0 = retract_reduced(prior_coeffs[:, prior_idx], Qz)
                append_unique_reduced_seed(starts, z0)

        if k_idx < q and len(starts) < max(1, num_restarts):
            e = np.zeros(q, dtype=Vred.dtype)
            e[k_idx] = 1.0
            z0 = retract_reduced(e, Qz)
            append_unique_reduced_seed(starts, z0)

        while len(starts) < max(1, num_restarts):
            zrand = np.ascontiguousarray(rng.standard_normal(q), dtype=Vred.dtype)
            zrand = retract_reduced(zrand, Qz)
            append_unique_reduced_seed(starts, zrand)

        best = None
        best_logf = -np.inf
        best_restart = None
        if len(starts) == 1:
            if is_initial_block:
                cand_results = [basic_projected_ascent_single_reduced(Bred, starts[0], Qz, win, ncols, maxit=maxit, tol=tol)]
            else:
                cand_results = [basic_projected_ascent_single_reduced_streaming(
                    Bred, B_block_red, C_prev, prev_s2, prev_q, starts[0], Qz, rows_total, ncols, maxit=maxit, tol=tol
                )]
        else:
            max_workers = min(len(starts), os.cpu_count() or 1)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                if is_initial_block:
                    cand_results = list(
                        executor.map(
                            lambda z0: basic_projected_ascent_single_reduced(Bred, z0, Qz, win, ncols, maxit=maxit, tol=tol),
                            starts,
                        )
                    )
                else:
                    cand_results = list(
                        executor.map(
                            lambda z0: basic_projected_ascent_single_reduced_streaming(
                                Bred, B_block_red, C_prev, prev_s2, prev_q, z0, Qz, rows_total, ncols, maxit=maxit, tol=tol
                            ),
                            starts,
                        )
                    )

        for restart_idx, cand in enumerate(cand_results):
            if cand[1] > best_logf:
                best = cand
                best_logf = cand[1]
                best_restart = restart_idx + 1

        z_best, logf_best, s_best, H_best, stop = best
        claimed_prior_idx = retire_closest_prior_index(prior_coeffs, unused_prior_mask, z_best, Qz)
        v_best = np.ascontiguousarray(Vred @ z_best, dtype=Vred.dtype)
        v_best = np.ascontiguousarray(v_best / max(np.linalg.norm(v_best), 1e-30), dtype=Vred.dtype)

        V_out[:, k_idx] = v_best
        s_out[k_idx] = s_best
        H_out[k_idx] = H_best
        score_out[k_idx] = float(np.exp(logf_best))
        Qz = np.column_stack([Qz, z_best]) if Qz.size else z_best[:, None]

        r_perp, g_full = entropy_full_gradient_residual(M_arr, v_best, Vred, win, ncols)
        grad_perp_ratio[k_idx] = r_perp / max(g_full, 1e-30)
        if verbose and ((k_idx < 10) or ((k_idx + 1) % 25 == 0) or (k_idx + 1 == active_r)):
            print({
                "basis": k_idx + 1,
                "best_restart": best_restart,
                "claimed_prior_idx": None if claimed_prior_idx is None else claimed_prior_idx + 1,
                "stop_reason": stop["reason"],
                "iters": stop["iters"],
                "grad_norm": stop["grad_norm"],
                "s": float(s_best),
                "H": float(H_best),
                "time": time.time() - basis_t0,
                "grad_perp_norm": r_perp,
                "grad_full_norm": g_full,
                "grad_perp_ratio": grad_perp_ratio[k_idx],
            })

    solve_time = time.time() - solve_t0
    diag = {
        "subspace_method": subspace_method,
        "q_subspace": q,
        "subspace_build_time": subspace_build_time,
        "reduced_solve_time": solve_time,
        "grad_perp_ratio": grad_perp_ratio,
        "Vred": Vred,
        "Bred": Bred,
        "sred": sred,
    }
    return V_out, s_out, H_out, score_out, diag


def basic_projected_ascent_single_exact(M, v0, Q, rows_total, ncols, maxit, tol, stats: Optional[TimeStats] = None, mvstats: Optional[MatvecStats] = None,
                                        log_runtime_state: bool = False):
    v = retract_feasible(v0, Q, stats=stats)
    if v is None:
        raise ValueError('Initial vector infeasible in exact optimizer.')

    if log_runtime_state:
        rng = np.random.default_rng(0)
        v_test = rng.standard_normal(M.shape[1]).astype(np.asarray(M).dtype, copy=False)
        u_test = rng.standard_normal(M.shape[0]).astype(np.asarray(M).dtype, copy=False)
        t0 = time.perf_counter()
        for _ in range(20):
            _ = M @ v_test
        t1 = time.perf_counter()
        t2 = time.perf_counter()
        for _ in range(20):
            _ = M.T @ u_test
        t3 = time.perf_counter()
        print({"before_first_restart_objective": runtime_state_snapshot()})
        print({
            "in_restart_scope_microbench": {
                "forward_avg": (t1 - t0) / 20.0,
                "reverse_avg": (t3 - t2) / 20.0,
            }
        })

    logf, gradE, s2, H2 = entropy_logscore_grad_rows(M, v, rows_total, ncols, stats=stats, mvstats=mvstats)
    if log_runtime_state:
        print({"after_first_restart_objective": runtime_state_snapshot()})
    progress_f_tol = 1e-12
    progress_step_tol = 1e-10
    prog = ProgressDiagnostics()
    prog.init_score(logf)
    stop = StopDiagnostics(solver="projected_ascent_exact", grad_tol=tol, step_tol=progress_step_tol)

    for it in range(maxit):
        g = project_to_feasible_tangent(gradE, v, Q, stats=stats)
        gnorm = np.sqrt(kahan_sum(np.abs(g) ** 2))
        if gnorm <= tol:
            stop.reason = "grad_tol"
            stop.iters = it
            stop.grad_norm = gnorm
            return v, logf, s2, H2, stop, prog

        accepted = False
        alpha = 1.0
        logf_old = logf
        v_old = v.copy()
        gg = float(np.real(g @ g))
        ls_steps = 0

        for _ in range(20):
            ls_steps += 1
            vt = retract_feasible(v + alpha * g, Q, stats=stats)
            if vt is not None:
                with timed(stats, "line_search_eval"):
                    logf_trial, _, _, _ = entropy_logscore_grad_rows(M, vt, rows_total, ncols, stats=stats, mvstats=mvstats)
                rhs = logf_old + 1e-4 * alpha * gg
                if logf_trial >= rhs:
                    accepted = True
                    v = vt
                    break
            alpha *= 0.5

        if not accepted:
            prog.no_update()
            stop.reason = "line_search_fail"
            stop.iters = it + 1
            stop.grad_norm = gnorm
            stop.line_search_steps = ls_steps
            return v_old, logf_old, s2, H2, stop, prog

        logf, gradE, s2, H2 = entropy_logscore_grad_rows(M, v, rows_total, ncols, stats=stats, mvstats=mvstats)
        step_norm = np.sqrt(kahan_sum(np.abs(v - v_old) ** 2))
        f_change = abs(logf - logf_old)
        f_threshold = progress_f_tol * max(1.0, abs(logf_old))
        prog.update(logf_old, logf, v_old, v)
        if f_change <= f_threshold or step_norm <= progress_step_tol:
            stop.reason = "progress_tol"
            stop.iters = it + 1
            stop.grad_norm = np.sqrt(kahan_sum(np.abs(project_to_feasible_tangent(gradE, v, Q, stats=stats)) ** 2))
            stop.step_norm = step_norm
            stop.f_change = f_change
            stop.f_threshold = f_threshold
            stop.line_search_alpha = alpha
            stop.line_search_steps = ls_steps
            stop.accepted = True
            return v, logf, s2, H2, stop, prog

    stop.reason = "maxit"
    stop.iters = maxit
    stop.grad_norm = np.sqrt(kahan_sum(np.abs(project_to_feasible_tangent(gradE, v, Q, stats=stats)) ** 2))
    return v, logf, s2, H2, stop, prog


def basic_projected_ascent_single_streaming(M_gain, A_block, V_old, s2_old, q_old, rows_total, ncols, v0, Q, maxit, tol, stats: Optional[TimeStats] = None, mvstats: Optional[MatvecStats] = None,
                                            log_runtime_state: bool = False):
    v = retract_feasible(v0, Q, stats=stats)
    if v is None:
        raise ValueError('Initial vector infeasible in streaming optimizer.')

    if log_runtime_state:
        rng = np.random.default_rng(0)
        v_test = rng.standard_normal(M_gain.shape[1]).astype(np.asarray(M_gain).dtype, copy=False)
        u_test = rng.standard_normal(M_gain.shape[0]).astype(np.asarray(M_gain).dtype, copy=False)
        t0 = time.perf_counter()
        for _ in range(20):
            _ = M_gain @ v_test
        t1 = time.perf_counter()
        t2 = time.perf_counter()
        for _ in range(20):
            _ = M_gain.T @ u_test
        t3 = time.perf_counter()
        print({"before_first_restart_objective": runtime_state_snapshot()})
        print({
            "in_restart_scope_microbench": {
                "forward_avg": (t1 - t0) / 20.0,
                "reverse_avg": (t3 - t2) / 20.0,
            }
        })

    logf, gradE, s2_total, H_approx = entropy_streaming_logscore_grad(
        M_gain, A_block, V_old, s2_old, q_old, v, rows_total, ncols, stats=stats, mvstats=mvstats
    )
    if log_runtime_state:
        print({"after_first_restart_objective": runtime_state_snapshot()})
    progress_f_tol = 1e-12
    progress_step_tol = 1e-10
    prog = ProgressDiagnostics()
    prog.init_score(logf)
    stop = StopDiagnostics(solver="projected_ascent_streaming", grad_tol=tol, step_tol=progress_step_tol)

    for it in range(maxit):
        g = project_to_feasible_tangent(gradE, v, Q, stats=stats)
        gnorm = np.sqrt(kahan_sum(np.abs(g) ** 2))
        if gnorm <= tol:
            stop.reason = "grad_tol"
            stop.iters = it
            stop.grad_norm = gnorm
            return v, logf, s2_total, H_approx, stop, prog

        accepted = False
        alpha = 1.0
        logf_old = logf
        v_old = v.copy()
        gg = float(np.real(g @ g))
        ls_steps = 0

        for _ in range(20):
            ls_steps += 1
            vt = retract_feasible(v + alpha * g, Q, stats=stats)
            if vt is not None:
                with timed(stats, "line_search_eval"):
                    logf_trial, _, _, _ = entropy_streaming_logscore_grad(
                        M_gain, A_block, V_old, s2_old, q_old, vt, rows_total, ncols, stats=stats, mvstats=mvstats
                    )
                rhs = logf_old + 1e-4 * alpha * gg
                if logf_trial >= rhs:
                    accepted = True
                    v = vt
                    break
            alpha *= 0.5

        if not accepted:
            prog.no_update()
            stop.reason = "line_search_fail"
            stop.iters = it + 1
            stop.grad_norm = gnorm
            stop.line_search_steps = ls_steps
            return v_old, logf_old, s2_total, H_approx, stop, prog

        logf, gradE, s2_total, H_approx = entropy_streaming_logscore_grad(
            M_gain, A_block, V_old, s2_old, q_old, v, rows_total, ncols, stats=stats, mvstats=mvstats
        )
        step_norm = np.sqrt(kahan_sum(np.abs(v - v_old) ** 2))
        f_change = abs(logf - logf_old)
        f_threshold = progress_f_tol * max(1.0, abs(logf_old))
        prog.update(logf_old, logf, v_old, v)
        if f_change <= f_threshold or step_norm <= progress_step_tol:
            stop.reason = "progress_tol"
            stop.iters = it + 1
            stop.grad_norm = np.sqrt(kahan_sum(np.abs(project_to_feasible_tangent(gradE, v, Q, stats=stats)) ** 2))
            stop.step_norm = step_norm
            stop.f_change = f_change
            stop.f_threshold = f_threshold
            stop.line_search_alpha = alpha
            stop.line_search_steps = ls_steps
            stop.accepted = True
            return v, logf, s2_total, H_approx, stop, prog

    stop.reason = "maxit"
    stop.iters = maxit
    stop.grad_norm = np.sqrt(kahan_sum(np.abs(project_to_feasible_tangent(gradE, v, Q, stats=stats)) ** 2))
    return v, logf, s2_total, H_approx, stop, prog


def make_basic_restart_seeds(M, Q, k_idx, V_init, num_restarts, Vsvd=None, stats: Optional[TimeStats] = None, verbose: bool = False):
    with timed(stats, "make_restart_seeds"):
        work_dtype = matrix_work_dtype(M)
        d = M.shape[1]
        if Vsvd is None:
            _, _, vh = np.linalg.svd(np.asarray(M), full_matrices=False)
            Vsvd = vh.T
        Vsvd = np.ascontiguousarray(np.asarray(Vsvd, dtype=work_dtype))
        V_init_work = None if V_init is None else np.ascontiguousarray(np.asarray(V_init, dtype=work_dtype))
        num_top = min(4, Vsvd.shape[1])
        alpha_grid = [0.98, 0.9, 0.75, 0.5, 0.25, 0.0]
        starts = []

        for restart in range(num_restarts):
            v_prev = V_init_work[:, k_idx] if V_init_work is not None and V_init_work.size and V_init_work.shape[1] > k_idx else None
            restart_type = (restart % 5) + 1
            restart_block = restart // 5

            if restart_type == 1:
                if v_prev is not None:
                    xi = np.random.standard_normal(d).astype(work_dtype, copy=False)
                    xi = project_feasible(xi, Q, stats=stats)
                    nxi = np.sqrt(kahan_sum(np.abs(xi) ** 2))
                    if nxi > 1e-14:
                        xi = np.ascontiguousarray(xi / nxi, dtype=work_dtype)
                    alpha = alpha_grid[restart_block % len(alpha_grid)]
                    v0 = np.ascontiguousarray(
                        alpha * v_prev + np.sqrt(max(0.0, 1.0 - alpha ** 2)) * xi,
                        dtype=work_dtype,
                    )
                else:
                    v0 = Vsvd[:, 0]
            elif restart_type == 2:
                j = restart_block % num_top
                v0 = Vsvd[:, j]
            elif restart_type == 3:
                j1 = restart_block % num_top
                j2 = (restart_block + 1) % num_top
                alpha = alpha_grid[restart_block % len(alpha_grid)]
                v0 = np.ascontiguousarray(
                    alpha * Vsvd[:, j1] + np.sqrt(max(0.0, 1.0 - alpha ** 2)) * Vsvd[:, j2],
                    dtype=work_dtype,
                )
            elif restart_type == 4:
                j = restart_block % num_top
                v0 = np.ascontiguousarray(
                    Vsvd[:, j] + np.asarray(1e-2 * np.random.standard_normal(d), dtype=work_dtype),
                    dtype=work_dtype,
                )
            else:
                v0 = np.random.standard_normal(d).astype(work_dtype, copy=False)

            v = retract_feasible(v0, Q, stats=stats)
            if v is None:
                v = retract_feasible(np.random.standard_normal(d).astype(work_dtype, copy=False), Q, stats=stats)
            if v is None:
                raise RuntimeError('Could not generate feasible restart seed.')
            starts.append(v)

        if verbose and len(starts) > 1:
            for a in range(len(starts)):
                for b in range(a + 1, len(starts)):
                    cosab = abs(float(starts[a] @ starts[b]))
                    if cosab > 1.0 - 1e-10:
                        print(f"[warn] duplicate restart seeds basis={k_idx + 1} seeds=({a + 1},{b + 1}) cos={cosab:.12f}")

        return starts


def entropy_iter_basis_streaming(A_block, r, ncols, state_prev, V_init, num_restarts=8, maxit=200, tol=1e-8,
                                 stats: Optional[TimeStats] = None, mvstats: Optional[MatvecStats] = None,
                                 verbose: bool = True, log_restarts: bool = True):
    with timed(stats, "entropy_iter_basis_streaming.total"):
        A_block = np.asarray(A_block)
        work_dtype = A_block.dtype
        d = A_block.shape[1]
        rows_new = A_block.shape[0]

        V_out = np.zeros((d, r), dtype=work_dtype)
        s_out = np.zeros(r)
        H_out = np.full(r, -np.inf)
        score_out = np.full(r, -np.inf)
        Q = np.zeros((d, 0), dtype=work_dtype)

        is_initial_block = state_prev is None
        if is_initial_block:
            rows_total = rows_new
            M_gain = A_block
            prev_basis = np.zeros((d, 0), dtype=work_dtype)
            prev_s2 = np.zeros(0, dtype=work_dtype)
            prev_q = np.zeros(0, dtype=work_dtype)
        else:
            rows_total = state_prev['rows_seen'] + rows_new
            B_top = np.asarray(np.diag(state_prev['s']) @ state_prev['V'].T, dtype=work_dtype)
            M_gain = np.vstack([B_top, A_block]).astype(work_dtype, copy=False)
            prev_basis = np.ascontiguousarray(np.asarray(state_prev['V'], dtype=work_dtype))
            prev_s2 = np.asarray(state_prev['s2'], dtype=work_dtype)
            prev_q = np.asarray(state_prev['q'], dtype=work_dtype)

        active_r = min(r, M_gain.shape[0], d)
        if active_r <= 0:
            state_out = {
                'V': V_out[:, :0],
                's': s_out[:0],
                's2': np.zeros(0),
                'H': H_out[:0],
                'q': np.zeros(0),
                'score': score_out[:0],
                'rows_seen': rows_total,
                'prev_basis': prev_basis,
                'prev_s2': prev_s2,
                'prev_q': prev_q,
            }
            return V_out[:, :0], s_out[:0], H_out[:0], score_out[:0], state_out

        if verbose:
            print(f'EntropyScore setup: M_gain shape={M_gain.shape}, active rank={active_r}, restarts={num_restarts}, maxit={maxit}')
            print({"before_setup_svd": runtime_state_snapshot()})

        svd_setup_start = time.time()
        setup_svd_mvstats = MatvecStats()
        setup_svd_summary = None
        first_restart_comparison_printed = False
        with timed(stats, "entropy_iter_basis_streaming.setup_svd"):
            M_gain_arr = np.asarray(M_gain)
            min_dim = min(M_gain_arr.shape)
            if active_r < min_dim:
                M_gain_op = make_tracked_linear_operator(M_gain_arr, mvstats=setup_svd_mvstats, dtype=M_gain_arr.dtype)
                _, s_setup, vh = sp.sparse.linalg.svds(M_gain_op, k=active_r, which='LM')
                order = np.argsort(s_setup)[::-1]
                s_setup = s_setup[order]
                vh = vh[order, :]
                setup_svd_summary = {
                    "matvec_calls": setup_svd_mvstats.matvec_calls,
                    "rmatvec_calls": setup_svd_mvstats.rmatvec_calls,
                    "matvec_rhs": setup_svd_mvstats.matvec_rhs,
                    "rmatvec_rhs": setup_svd_mvstats.rmatvec_rhs,
                    "matmat_calls": setup_svd_mvstats.matmat_calls,
                    "rmatmat_calls": setup_svd_mvstats.rmatmat_calls,
                    "matvec_rhs_sizes": dict(sorted(setup_svd_mvstats.matvec_rhs_sizes.items())),
                    "rmatvec_rhs_sizes": dict(sorted(setup_svd_mvstats.rmatvec_rhs_sizes.items())),
                    "avg_matvec_time": (
                        setup_svd_mvstats.matvec_time / setup_svd_mvstats.matvec_calls
                        if setup_svd_mvstats.matvec_calls > 0 else None
                    ),
                    "avg_rmatvec_time": (
                        setup_svd_mvstats.rmatvec_time / setup_svd_mvstats.rmatvec_calls
                        if setup_svd_mvstats.rmatvec_calls > 0 else None
                    ),
                }
            else:
                _, s_setup, vh = np.linalg.svd(M_gain_arr, full_matrices=False)
            Vsvd = np.ascontiguousarray(np.asarray(vh.T, dtype=work_dtype))
        if verbose:
            print(f'EntropyScore setup SVD time: {time.time() - svd_setup_start:.2f}s')
            print({
                "entropy_layout": {
                    "A_block_flags": {
                        "c_contiguous": bool(A_block.flags.c_contiguous),
                        "f_contiguous": bool(A_block.flags.f_contiguous),
                        "owndata": bool(A_block.flags.owndata),
                    },
                    "M_gain_flags": {
                        "c_contiguous": bool(M_gain_arr.flags.c_contiguous),
                        "f_contiguous": bool(M_gain_arr.flags.f_contiguous),
                        "owndata": bool(M_gain_arr.flags.owndata),
                    },
                    "A_block_strides": tuple(int(x) for x in A_block.strides),
                    "M_gain_strides": tuple(int(x) for x in M_gain_arr.strides),
                    "dtype": str(M_gain_arr.dtype),
                    "blas_env": {
                        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                    },
                }
            })
            print({"before_setup_microbench": runtime_state_snapshot()})
            print({"entropy_dense_microbench": dense_mv_microbench(A_block, num_trials=10, seed=0)})
            if setup_svd_summary is not None:
                print({"setup_svd_matvec_stats": setup_svd_summary})

        energy = np.cumsum(s_setup ** 2)
        target_energy = 0.95 if active_r > 128 else 0.99
        if energy.size and energy[-1] > 0:
            energy /= energy[-1]
            optimize_r = min(active_r, max(16, int(np.searchsorted(energy, target_energy)) + 1))
        else:
            optimize_r = min(active_r, 16)
        optimize_r = min(optimize_r, active_r)
        if verbose:
            print(f'EntropyScore optimized rank: {optimize_r}, tail rank: {active_r - optimize_r}, target energy={target_energy}')

        for k_idx in range(active_r):
            with timed(stats, f"entropy_iter_basis_streaming.vector_{k_idx + 1}"):
                basis_start_time = time.time()
                basis_totals_before, basis_counts_before = stats.snapshot() if stats is not None else ({}, {})
                basis_mv_before = mvstats.as_dict() if mvstats is not None else {}
                should_log_basis = verbose and ((k_idx < 10) or ((k_idx + 1) % 25 == 0) or (k_idx + 1 == active_r))
                if should_log_basis:
                    print(f'EntropyScore basis {k_idx + 1}/{active_r}')

                best = None
                best_logf = -np.inf
                best_restart_idx = None
                best_stop = None
                best_prog = None

                if k_idx < optimize_r:
                    if active_r > 128:
                        restart_budget = min(num_restarts, 2)
                        iter_budget = min(maxit, 30 if is_initial_block else 20)
                    else:
                        restart_budget = min(num_restarts, 4 if is_initial_block else 3)
                        iter_budget = min(maxit, 80 if is_initial_block else 40)

                    starts = make_basic_restart_seeds(
                        M_gain, Q, k_idx, V_init, restart_budget, Vsvd=Vsvd, stats=stats, verbose=log_restarts
                    )

                    for restart_idx, v0 in enumerate(starts):
                        restart_totals_before, restart_counts_before = stats.snapshot() if stats is not None else ({}, {})
                        restart_mv_before = mvstats.as_dict() if mvstats is not None else {}
                        with timed(stats, "entropy_iter_basis_streaming.restart"):
                            if is_initial_block:
                                cand = basic_projected_ascent_single_exact(
                                    A_block, v0, Q, rows_total, ncols, iter_budget, tol, stats=stats, mvstats=mvstats,
                                    log_runtime_state=(k_idx == 0 and restart_idx == 0)
                                )
                            else:
                                cand = basic_projected_ascent_single_streaming(
                                    M_gain, A_block, prev_basis, prev_s2, prev_q, rows_total, ncols, v0, Q, iter_budget, tol,
                                    stats=stats, mvstats=mvstats, log_runtime_state=(k_idx == 0 and restart_idx == 0)
                                )

                        if log_restarts:
                            stop = cand[4]
                            prog = cand[5]
                            objective_key = "entropy_logscore_grad_rows" if is_initial_block else "entropy_streaming_logscore_grad"
                            restart_timing = stats_delta_summary(
                                stats,
                                restart_totals_before,
                                restart_counts_before,
                                [
                                    objective_key,
                                    "line_search_eval",
                                    "project_prev_basis",
                                    "project_tangent",
                                    "project_current_vector",
                                    "retract_feasible",
                                    "entropy_exact.forward_mv",
                                    "entropy_exact.moments",
                                    "entropy_exact.scalar_terms",
                                    "entropy_exact.reverse_mv_y",
                                    "entropy_exact.y_cube",
                                    "entropy_exact.reverse_mv_y3",
                                    "entropy_exact.combine",
                                    "entropy_stream.forward_m_gain",
                                    "entropy_stream.gain_moment",
                                    "entropy_stream.lowrank_project",
                                    "entropy_stream.forward_a_block",
                                    "entropy_stream.block_moments",
                                    "entropy_stream.scalar_terms",
                                    "entropy_stream.logf",
                                    "entropy_stream.reverse_m_gain",
                                    "entropy_stream.reverse_a_block_y",
                                    "entropy_stream.y_cube",
                                    "entropy_stream.reverse_a_block_y3",
                                    "entropy_stream.lowrank_expand",
                                    "entropy_stream.combine",
                                ],
                            )
                            obj_calls = restart_timing.get(objective_key, {}).get("count", 0)
                            obj_avg = restart_timing.get(objective_key, {}).get("avg")
                            mv_delta = None
                            if mvstats is not None:
                                mv_after = mvstats.as_dict()
                                mv_delta = {
                                    "matvec_calls": mv_after["MATVEC_CALLS"] - restart_mv_before.get("MATVEC_CALLS", 0),
                                    "rmatvec_calls": mv_after["RMATVEC_CALLS"] - restart_mv_before.get("RMATVEC_CALLS", 0),
                                    "matvec_rhs": mv_after["MATVEC_RHS"] - restart_mv_before.get("MATVEC_RHS", 0),
                                    "rmatvec_rhs": mv_after["RMATVEC_RHS"] - restart_mv_before.get("RMATVEC_RHS", 0),
                                    "matmat_calls": mv_after["MATMAT_CALLS"] - restart_mv_before.get("MATMAT_CALLS", 0),
                                    "rmatmat_calls": mv_after["RMATMAT_CALLS"] - restart_mv_before.get("RMATMAT_CALLS", 0),
                                    "matvec_time": mv_after["matvec_time"] - restart_mv_before.get("matvec_time", 0.0),
                                    "rmatvec_time": mv_after["rmatvec_time"] - restart_mv_before.get("rmatvec_time", 0.0),
                                }
                            print({
                                "basis": k_idx + 1,
                                "restart": restart_idx + 1,
                                "reason": stop.reason,
                                "iters": stop.iters,
                                "grad_norm": stop.grad_norm,
                                "line_search_steps": stop.line_search_steps,
                                "objective_calls": obj_calls,
                                "avg_objective_time": obj_avg,
                                "gain": prog.as_dict()["total_gain"],
                                "last_step_norm": prog.last_step_norm,
                                "timing": restart_timing,
                                "matvec_delta": mv_delta,
                            })
                            if (not first_restart_comparison_printed) and restart_idx == 0 and setup_svd_summary is not None:
                                first_restart_summary = {
                                    "avg_matvec_time": (
                                        mv_delta["matvec_time"] / mv_delta["matvec_calls"]
                                        if mv_delta is not None and mv_delta["matvec_calls"] > 0 else None
                                    ),
                                    "avg_rmatvec_time": (
                                        mv_delta["rmatvec_time"] / mv_delta["rmatvec_calls"]
                                        if mv_delta is not None and mv_delta["rmatvec_calls"] > 0 else None
                                    ),
                                    "matvec_calls": None if mv_delta is None else mv_delta["matvec_calls"],
                                    "rmatvec_calls": None if mv_delta is None else mv_delta["rmatvec_calls"],
                                    "matvec_rhs": None if mv_delta is None else mv_delta["matvec_rhs"],
                                    "rmatvec_rhs": None if mv_delta is None else mv_delta["rmatvec_rhs"],
                                    "matmat_calls": None if mv_delta is None else mv_delta["matmat_calls"],
                                    "rmatmat_calls": None if mv_delta is None else mv_delta["rmatmat_calls"],
                                    "avg_objective_time": obj_avg,
                                    "objective_calls": obj_calls,
                                }
                                print({
                                    "setup_vs_first_restart": {
                                        "setup_svd": setup_svd_summary,
                                        "first_restart": first_restart_summary,
                                    }
                                })
                                first_restart_comparison_printed = True

                        if cand[1] > best_logf:
                            best = cand
                            best_logf = cand[1]
                            best_restart_idx = restart_idx
                            best_stop = cand[4]
                            best_prog = cand[5]
                else:
                    seed_idx = min(k_idx, Vsvd.shape[1] - 1)
                    seed = retract_feasible(Vsvd[:, seed_idx], Q, stats=stats)
                    if seed is None:
                        seed = retract_feasible(np.random.randn(d), Q, stats=stats)
                    if seed is None:
                        raise RuntimeError(f'Could not generate feasible tail seed for k={k_idx + 1}.')

                    if is_initial_block:
                        logf_seed, _, s2_seed, H_seed = entropy_logscore_grad_rows(
                            A_block, seed, rows_total, ncols, stats=stats, mvstats=mvstats
                        )
                    else:
                        logf_seed, _, s2_seed, H_seed = entropy_streaming_logscore_grad(
                            M_gain, A_block, prev_basis, prev_s2, prev_q, seed, rows_total, ncols, stats=stats, mvstats=mvstats
                        )

                    best = (seed, logf_seed, s2_seed, H_seed)
                    best_logf = logf_seed
                    best_restart_idx = 0
                    best_stop = StopDiagnostics(reason="tail_seed", iters=0, grad_norm=np.nan, solver="tail_seed")
                    best_prog = ProgressDiagnostics(first_score=logf_seed, last_score=logf_seed, best_score=logf_seed)

                if best is None:
                    raise RuntimeError(f'All restarts failed for k={k_idx + 1}.')

                best_v, _, best_s2, best_H = best[:4]
                best_v = np.ascontiguousarray(np.asarray(best_v, dtype=work_dtype).reshape(-1))
                Q = np.column_stack([Q, best_v]) if Q.size else best_v[:, None]
                V_out[:, k_idx] = best_v
                s_out[k_idx] = np.sqrt(max(best_s2, 0.0))
                H_out[k_idx] = best_H
                score_out[k_idx] = np.exp(best_logf)

                if should_log_basis:
                    basis_keys = [
                        "make_restart_seeds",
                        "entropy_iter_basis_streaming.restart",
                        "entropy_streaming_logscore_grad",
                        "entropy_logscore_grad_rows",
                        "line_search_eval",
                        "project_prev_basis",
                        "project_tangent",
                        "project_current_vector",
                        "retract_feasible",
                    ]
                    basis_timing = stats_delta_summary(
                        stats,
                        basis_totals_before,
                        basis_counts_before,
                        basis_keys,
                    )
                    basis_mv_delta = None
                    if mvstats is not None:
                        basis_mv_after = mvstats.as_dict()
                        basis_mv_delta = {
                            "matvec_calls": basis_mv_after["MATVEC_CALLS"] - basis_mv_before.get("MATVEC_CALLS", 0),
                            "rmatvec_calls": basis_mv_after["RMATVEC_CALLS"] - basis_mv_before.get("RMATVEC_CALLS", 0),
                            "matvec_rhs": basis_mv_after["MATVEC_RHS"] - basis_mv_before.get("MATVEC_RHS", 0),
                            "rmatvec_rhs": basis_mv_after["RMATVEC_RHS"] - basis_mv_before.get("RMATVEC_RHS", 0),
                            "matmat_calls": basis_mv_after["MATMAT_CALLS"] - basis_mv_before.get("MATMAT_CALLS", 0),
                            "rmatmat_calls": basis_mv_after["RMATMAT_CALLS"] - basis_mv_before.get("RMATMAT_CALLS", 0),
                            "matvec_time": basis_mv_after["matvec_time"] - basis_mv_before.get("matvec_time", 0.0),
                            "rmatvec_time": basis_mv_after["rmatvec_time"] - basis_mv_before.get("rmatvec_time", 0.0),
                        }
                    print({
                        "basis": k_idx + 1,
                        "best_restart": None if best_restart_idx is None else best_restart_idx + 1,
                        "stop_reason": None if best_stop is None else best_stop.reason,
                        "iters": None if best_stop is None else best_stop.iters,
                        "grad_norm": None if best_stop is None else best_stop.grad_norm,
                        "total_gain": None if best_prog is None else best_prog.as_dict()["total_gain"],
                        "time": time.time() - basis_start_time,
                        "s": float(s_out[k_idx]),
                        "H": float(H_out[k_idx]),
                        "timing": basis_timing,
                        "matvec_delta": basis_mv_delta,
                    })

        V_out = V_out[:, :active_r]
        s_out = s_out[:active_r]
        H_out = H_out[:active_r]
        score_out = score_out[:active_r]
        s2_out = s_out ** 2
        q_out = (s_out ** 4) * np.exp(-H_out)
        state_out = {
            'V': V_out,
            's': s_out,
            's2': s2_out,
            'H': H_out,
            'q': q_out,
            'score': score_out,
            'rows_seen': rows_total,
            'prev_basis': prev_basis,
            'prev_s2': prev_s2,
            'prev_q': prev_q,
        }

        if verbose:
            with timed(stats, "entropy_iter_basis_streaming.total_report"):
                if stats is not None:
                    print(stats.report(sort_by="time"))
                if mvstats is not None:
                    print(mvstats.as_dict())

        return V_out, s_out, H_out, score_out, state_out


def entropyscore_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path,
              col_permutation, track_U,
              track_discarded, discarded_list,
              num_Vs, with_S, reverse, return_row_order,
              total_S_reduced,
              reservoir_size, reservoir_idx, reservoir, reservoir_method,
              Vt=None, S=None, V_focus=None, reserved=None, save_in_text=True):
    del first_window_size, U_exact, reservoir_method, S
    if col_permutation is not None:
        next_window = next_window[:, col_permutation]
    if sparse.issparse(next_window):
        A_block = next_window.toarray()
    else:
        A_block = np.asarray(next_window)

    state_prev = reserved
    V_init = None if Vt is None else np.asarray(Vt).T
    stats = TimeStats()
    mvstats = MatvecStats()
    step_start_time = time.time()
    V_new, s_new, H_new, score_new, state_out = entropy_iter_basis_streaming(
        A_block, k, A_block.shape[1], state_prev, V_init, num_restarts=8, maxit=200, tol=1e-8,
        stats=stats, mvstats=mvstats, verbose=True, log_restarts=True
    )
    Vt = V_new.T
    S = s_new
    print(f'EntropyScore active rank: {len(S)} / requested {k}, block rows={A_block.shape[0]}, total rows={state_out["rows_seen"]}')
    print(f'EntropyScore basis solve time: {time.time() - step_start_time:.2f}s')
    print('EntropyScore s:', S[:min(10, len(S))])
    print('EntropyScore H:', H_new[:min(10, len(H_new))])
    print('EntropyScore scores:', score_new[:min(10, len(score_new))])

    S_quotient = S.copy()
    if num_Vs:
        if V_focus is None:
            block = Vt[:num_Vs, row_permutation[end_idx:]]
            weights = S_quotient[:num_Vs].reshape(-1, 1) if with_S else 1.0
            order_scores = np.sum((block * weights) ** 2, axis=0)
            indices = np.argsort(order_scores).reshape(-1)
            if not reverse:
                indices = indices[::-1]
        else:
            indices = np.argsort(np.sum((Vt[V_focus, row_permutation[end_idx:]]) ** 2, axis=0)).reshape(-1)
        row_permutation[end_idx:] = row_permutation[end_idx:][indices]

    num_save_files = 50
    should_save = (j == 0 or (j * (num_save_files - 1)) // W != ((j - 1) * (num_save_files - 1)) // W)
    if should_save:
        save_spectrum_comparison(S + total_S_reduced, S_exact, A_norm, name, j, dir_path,
                                 S_quotient=S_quotient, score_history=score_new, save_in_text=save_in_text)
        save_residuals(A_csr, S + total_S_reduced, Vt, S_exact, A_norm, name, j, dir_path, is_sym_psd,
                       row_permutation, start_idx, end_idx, save_in_text=save_in_text)
        if Vt_exact is not None:
            save_canonical_angles(Vt, Vt_exact, j, dir_path, save_in_text=save_in_text)

    ret = [S, Vt, state_out]
    if reservoir_size > 0:
        ret.append((reservoir_idx, reservoir))
    if track_U:
        ret.append(None)
    if track_discarded:
        ret.append(discarded_list)
    if return_row_order:
        ret.append(row_permutation)
    if total_S_reduced > 0:
        ret.append(total_S_reduced)
    return ret


def entropyscore_fast_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path,
              col_permutation, track_U,
              track_discarded, discarded_list,
              num_Vs, with_S, reverse, return_row_order,
              total_S_reduced,
              reservoir_size, reservoir_idx, reservoir, reservoir_method,
              Vt=None, S=None, V_focus=None, reserved=None, save_in_text=True,
              subspace_method="lanczos", q_subspace=None, num_restarts=3, maxit=40, tol=1e-8):
    del first_window_size, U_exact, reservoir_method
    if col_permutation is not None:
        next_window = next_window[:, col_permutation]
    if sparse.issparse(next_window):
        A_block = next_window.toarray().astype(np.float32, copy=False)
    else:
        A_block = np.asarray(next_window, dtype=np.float32)

    step_start_time = time.time()
    state_prev = reserved
    if Vt is None or S is None:
        M_gain = A_block
        V_init = None
        rows_seen = A_block.shape[0]
    else:
        S_prev = np.asarray(S, dtype=np.float32).reshape(-1)
        Vt_prev = np.asarray(Vt, dtype=np.float32)
        B_top = (S_prev[:, None] * Vt_prev).astype(np.float32, copy=False)
        M_gain = np.vstack([B_top, A_block]).astype(np.float32, copy=False)
        V_init = Vt_prev.T
        prev_rows_seen = 0 if state_prev is None else int(state_prev.get("rows_seen", 0))
        rows_seen = prev_rows_seen + A_block.shape[0]

    active_r = min(k, M_gain.shape[0], M_gain.shape[1])
    V_new, s_new, H_new, score_new, diag = entropy_iter_basis_fast(
        M_gain=M_gain,
        active_r=active_r,
        win=max(rows_seen, 2),
        ncols=A_block.shape[1],
        V_init=V_init,
        q_subspace=q_subspace,
        subspace_method=subspace_method,
        num_restarts=num_restarts,
        maxit=maxit,
        tol=tol,
        rng=np.random.default_rng(0),
        verbose=True,
        state_prev=state_prev,
        A_block=A_block,
        rows_total=rows_seen,
    )

    Vt = V_new.T
    S = s_new
    print(f'EntropyScoreFast active rank: {len(S)} / requested {k}, block rows={A_block.shape[0]}, total rows={rows_seen}')
    print(f'EntropyScoreFast basis solve time: {time.time() - step_start_time:.2f}s')
    print('EntropyScoreFast s:', S[:min(10, len(S))])
    print('EntropyScoreFast H:', H_new[:min(10, len(H_new))])
    print('EntropyScoreFast scores:', score_new[:min(10, len(score_new))])
    print({
        "EntropyScoreFast_diag": {
            "subspace_method": diag["subspace_method"],
            "q_subspace": diag["q_subspace"],
            "subspace_build_time": diag["subspace_build_time"],
            "reduced_solve_time": diag["reduced_solve_time"],
            "grad_perp_ratio_head": diag["grad_perp_ratio"][:min(10, len(diag["grad_perp_ratio"]))],
        }
    })

    S_quotient = S.copy()
    if num_Vs:
        if V_focus is None:
            block = Vt[:num_Vs, row_permutation[end_idx:]]
            weights = S_quotient[:num_Vs].reshape(-1, 1) if with_S else 1.0
            order_scores = np.sum((block * weights) ** 2, axis=0)
            indices = np.argsort(order_scores).reshape(-1)
            if not reverse:
                indices = indices[::-1]
        else:
            indices = np.argsort(np.sum((Vt[V_focus, row_permutation[end_idx:]]) ** 2, axis=0)).reshape(-1)
        row_permutation[end_idx:] = row_permutation[end_idx:][indices]

    num_save_files = 50
    should_save = (j == 0 or (j * (num_save_files - 1)) // W != ((j - 1) * (num_save_files - 1)) // W)
    if should_save:
        save_spectrum_comparison(S + total_S_reduced, S_exact, A_norm, name, j, dir_path,
                                 S_quotient=S_quotient, score_history=score_new, save_in_text=save_in_text)
        save_residuals(A_csr, S + total_S_reduced, Vt, S_exact, A_norm, name, j, dir_path, is_sym_psd,
                       row_permutation, start_idx, end_idx, save_in_text=save_in_text)
        if Vt_exact is not None:
            save_canonical_angles(Vt, Vt_exact, j, dir_path, save_in_text=save_in_text)
            save_leftout_projected(Vt, Vt_exact, M_gain, j, dir_path, save_in_text=save_in_text)

    state_out = {
        "V": V_new,
        "s": S,
        "s2": S ** 2,
        "H": H_new,
        "q": (S ** 4) * np.exp(-H_new),
        "score": score_new,
        "rows_seen": rows_seen,
        "diag": diag,
    }

    ret = [S, Vt, state_out]
    if reservoir_size > 0:
        ret.append((reservoir_idx, reservoir))
    if track_U:
        ret.append(None)
    if track_discarded:
        ret.append(discarded_list)
    if return_row_order:
        ret.append(row_permutation)
    if total_S_reduced > 0:
        ret.append(total_S_reduced)
    return ret


def entropyscore_expansion_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path,
              col_permutation, track_U,
              track_discarded, discarded_list,
              num_Vs, with_S, reverse, return_row_order,
              total_S_reduced,
              reservoir_size, reservoir_idx, reservoir, reservoir_method,
              Vt=None, S=None, V_focus=None, reserved=None, save_in_text=True,
              q0=5, qmax=None, krylov_depth=2, residual_tol=1e-2, expansion_maxit=8,
              num_restarts=3, maxit=40, tol=1e-8):
    del first_window_size, U_exact, reservoir_method
    if col_permutation is not None:
        next_window = next_window[:, col_permutation]
    if sparse.issparse(next_window):
        A_block = next_window.toarray().astype(np.float32, copy=False)
    else:
        A_block = np.asarray(next_window, dtype=np.float32)

    step_start_time = time.time()
    state_prev = reserved
    if Vt is None or S is None:
        M_gain = A_block
        V_init = None
        rows_seen = A_block.shape[0]
    else:
        S_prev = np.asarray(S, dtype=np.float32).reshape(-1)
        Vt_prev = np.asarray(Vt, dtype=np.float32)
        B_top = (S_prev[:, None] * Vt_prev).astype(np.float32, copy=False)
        M_gain = np.vstack([B_top, A_block]).astype(np.float32, copy=False)
        V_init = Vt_prev.T
        prev_rows_seen = 0 if state_prev is None else int(state_prev.get("rows_seen", 0))
        rows_seen = prev_rows_seen + A_block.shape[0]

    active_r = min(k, M_gain.shape[0], M_gain.shape[1])
    V_new, s_new, H_new, score_new, diag = entropy_iter_basis_expansion(
        M_gain=M_gain,
        active_r=active_r,
        win=max(rows_seen, 2),
        ncols=A_block.shape[1],
        V_init=V_init,
        q0=q0,
        qmax=qmax,
        krylov_depth=krylov_depth,
        residual_tol=residual_tol,
        expansion_maxit=expansion_maxit,
        num_restarts=num_restarts,
        maxit=maxit,
        tol=tol,
        rng=np.random.default_rng(0),
        verbose=True,
        state_prev=state_prev,
        A_block=A_block,
        rows_total=rows_seen,
    )

    Vt = V_new.T
    S = s_new
    print(f'EntropyScoreExpansion active rank: {len(S)} / requested {k}, block rows={A_block.shape[0]}, total rows={rows_seen}')
    print(f'EntropyScoreExpansion basis solve time: {time.time() - step_start_time:.2f}s')
    print('EntropyScoreExpansion s:', S[:min(10, len(S))])
    print('EntropyScoreExpansion H:', H_new[:min(10, len(H_new))])
    print('EntropyScoreExpansion scores:', score_new[:min(10, len(score_new))])
    print({
        "EntropyScoreExpansion_diag": {
            "seed_rank": diag["seed_rank"],
            "max_rank": diag["max_rank"],
            "krylov_depth": diag["krylov_depth"],
            "residual_tol": diag["residual_tol"],
            "subspace_build_time": diag["subspace_build_time"],
            "reduced_solve_time": diag["reduced_solve_time"],
            "timing_totals": diag["timing_totals"],
            "timing_counts": diag["timing_counts"],
            "grad_perp_ratio_head": diag["grad_perp_ratio"][:min(10, len(diag["grad_perp_ratio"]))],
            "subspace_dims_head": diag["subspace_dims"][:min(10, len(diag["subspace_dims"]))],
            "expansion_iters_head": diag["expansion_iters"][:min(10, len(diag["expansion_iters"]))],
        }
    })

    S_quotient = S.copy()
    if num_Vs:
        if V_focus is None:
            block = Vt[:num_Vs, row_permutation[end_idx:]]
            weights = S_quotient[:num_Vs].reshape(-1, 1) if with_S else 1.0
            order_scores = np.sum((block * weights) ** 2, axis=0)
            indices = np.argsort(order_scores).reshape(-1)
            if not reverse:
                indices = indices[::-1]
        else:
            indices = np.argsort(np.sum((Vt[V_focus, row_permutation[end_idx:]]) ** 2, axis=0)).reshape(-1)
        row_permutation[end_idx:] = row_permutation[end_idx:][indices]

    num_save_files = 50
    should_save = (j == 0 or (j * (num_save_files - 1)) // W != ((j - 1) * (num_save_files - 1)) // W)
    if should_save:
        save_spectrum_comparison(S + total_S_reduced, S_exact, A_norm, name, j, dir_path,
                                 S_quotient=S_quotient, score_history=score_new, save_in_text=save_in_text)
        save_residuals(A_csr, S + total_S_reduced, Vt, S_exact, A_norm, name, j, dir_path, is_sym_psd,
                       row_permutation, start_idx, end_idx, save_in_text=save_in_text)
        if Vt_exact is not None:
            save_canonical_angles(Vt, Vt_exact, j, dir_path, save_in_text=save_in_text)
            save_leftout_projected(Vt, Vt_exact, M_gain, j, dir_path, save_in_text=save_in_text)

    state_out = {
        "V": V_new,
        "s": S,
        "s2": S ** 2,
        "H": H_new,
        "q": (S ** 4) * np.exp(-H_new),
        "score": score_new,
        "rows_seen": rows_seen,
        "diag": diag,
    }

    ret = [S, Vt, state_out]
    if reservoir_size > 0:
        ret.append((reservoir_idx, reservoir))
    if track_U:
        ret.append(None)
    if track_discarded:
        ret.append(discarded_list)
    if return_row_order:
        ret.append(row_permutation)
    if total_S_reduced > 0:
        ret.append(total_S_reduced)
    return ret


def entropyscore_forget_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path,
              col_permutation, track_U,
              track_discarded, discarded_list,
              num_Vs, with_S, reverse, return_row_order,
              total_S_reduced,
              reservoir_size, reservoir_idx, reservoir, reservoir_method,
              Vt=None, S=None, V_focus=None, reserved=None, save_in_text=True,
              score_rank=None,
              aux_direction_method=None,
              q0=5, qmax=None, krylov_depth=2, residual_tol=1e-2, expansion_maxit=8,
              num_restarts=3, maxit=40, tol=1e-8,
              reduced_optimizer="cex", work_dtype=np.float32,
              expansion_direction="krylov_v",
              reuse_line_search_grad=True,
              expansion_warm_start=False,
              post_expansion_maxit=None,
              score_variant="forget",
              old_memory_size=None,
              rownorm_seed_first_block=False,
              rownorm_seed_all_blocks=False,
              dump_score_components=False,
              dump_oracle_old_row_responses=False,
              dump_oracle_old_row_response_block=0,
              oracle_candidate_check=False,
              debug_mode="off",
              seed=0):
    del first_window_size, U_exact, reservoir_method, window_indices
    if debug_mode in {"combined", "summary"}:
        dump_score_components = True
        dump_oracle_old_row_responses = True
        oracle_candidate_check = True
    if score_variant == "combined":
        np.random.seed(int(seed) + int(j))
    if col_permutation is not None:
        next_window = next_window[:, col_permutation]
    if sparse.issparse(next_window):
        A_block = next_window.toarray().astype(work_dtype, copy=False)
    else:
        A_block = np.asarray(next_window, dtype=work_dtype)
    rows_ref = max(int(A_csr.shape[0]), int(A_block.shape[0]), 2)

    step_start_time = time.time()
    state_prev = reserved
    if Vt is None or S is None:
        M_gain = A_block
        if rownorm_seed_first_block or rownorm_seed_all_blocks:
            V_init = np.asarray(row_norm_seed(A_block, k), dtype=work_dtype)
        else:
            V_init = None
        rows_seen = A_block.shape[0]
    else:
        S_prev = np.asarray(S, dtype=work_dtype).reshape(-1)
        Vt_prev = np.asarray(Vt, dtype=work_dtype)
        B_top = (S_prev[:, None] * Vt_prev).astype(work_dtype, copy=False)
        M_gain = np.vstack([B_top, A_block]).astype(work_dtype, copy=False)
        if rownorm_seed_all_blocks:
            V_init = np.asarray(row_norm_seed(A_block, k), dtype=work_dtype)
        else:
            V_init = Vt_prev.T
        prev_rows_seen = 0 if state_prev is None else int(state_prev.get("rows_seen", 0))
        rows_seen = prev_rows_seen + A_block.shape[0]

    active_r = min(k, M_gain.shape[0], M_gain.shape[1])
    if score_rank is None:
        optimize_r = min(active_r, A_block.shape[0])
    else:
        optimize_r = min(max(0, int(score_rank)), active_r)
    old_row_memory_in = None
    if state_prev is not None:
        old_row_memory_in = state_prev.get("old_row_memory")
    V_score, s_score, H_score, score_score, diag = entropy_iter_basis_forget(
        M_gain=M_gain,
        active_r=optimize_r,
        rows_ref=rows_ref,
        V_init=V_init,
        q0=q0,
        qmax=qmax,
        krylov_depth=krylov_depth,
        residual_tol=residual_tol,
        expansion_maxit=expansion_maxit,
        num_restarts=num_restarts,
        maxit=maxit,
        tol=tol,
        rng=np.random.default_rng(0),
        verbose=True,
        state_prev=state_prev,
        A_block=A_block,
        rows_total=rows_seen,
        reduced_optimizer=reduced_optimizer,
        work_dtype=work_dtype,
        expansion_direction=expansion_direction,
        reuse_line_search_grad=reuse_line_search_grad,
        expansion_warm_start=expansion_warm_start,
        post_expansion_maxit=post_expansion_maxit,
        score_variant=score_variant,
        old_row_memory=old_row_memory_in,
    )

    # --- Diagnostic dumps (combined variant only) -----------------------------
    V_exact = None if Vt_exact is None else np.asarray(Vt_exact[:k, :]).T
    oracle_candidate_status = None
    if score_variant == "combined" and V_exact is not None:
        if oracle_candidate_check:
            cand = oracle_projection_candidate_combined(
                M_gain, A_block, V_exact, optimize_r, rows_ref,
                state_prev=state_prev, old_row_memory=old_row_memory_in,
            )
            if cand is not None and V_score.shape[1] == cand["V"].shape[1]:
                optimizer_sum = float(np.sum(score_score[:optimize_r]))
                candidate_sum = float(cand["score_sum"])
                accepted = candidate_sum > optimizer_sum + 1e-10
                oracle_candidate_status = {
                    "accepted": bool(accepted),
                    "optimizer_sum": optimizer_sum,
                    "candidate_sum": candidate_sum,
                }
                if accepted:
                    V_score = cand["V"][:, :optimize_r]
                    score_score = cand["score"][:optimize_r]
                    H_score = cand["H"][:optimize_r]
                    s_score = cand["s"][:optimize_r]
        if dump_score_components and V_score.shape[1] > 0:
            # Build oracle_raw_v{j} (normalized P_row(M_gain) V_exact[:, j]) and
            # optimizer_v{j} (greedy V_score columns) vector lists for the dump.
            _, Q_row_dump = projected_true_span_oracle(
                M_gain, V_exact, optimize_r,
            )
            vectors = []
            for jj in range(min(optimize_r, V_exact.shape[1])):
                v_proj = project_onto_span(V_exact[:, jj], Q_row_dump).reshape(-1)
                v_norm = float(np.linalg.norm(v_proj))
                if v_norm > 1e-30:
                    vectors.append((f"oracle_raw_v{jj+1}", v_proj / v_norm))
            for jj in range(V_score.shape[1]):
                vectors.append((f"optimizer_v{jj+1}", V_score[:, jj]))
            print_combined_score_component_dump(
                f"score_components_block_{int(j)+1}",
                vectors, M_gain, A_block, rows_ref,
                state_prev=state_prev, old_row_memory=old_row_memory_in,
            )
        oracle_diag = oracle_projection_diagnostics_combined(
            M_gain, A_block, V_exact, V_score, optimize_r, rows_ref,
            state_prev=state_prev, old_row_memory=old_row_memory_in,
        )
        if oracle_diag is not None:
            print({
                "oracle_projection_diag": {
                    "raw_oracle_score_sum": oracle_diag["raw_oracle_score_sum"],
                    "qr_oracle_score_sum": oracle_diag["qr_oracle_score_sum"],
                    "opt_proj_norms": np.asarray(oracle_diag["opt_proj_norms"]).tolist(),
                    "opt_vs_qoracle_cosines": np.asarray(oracle_diag["opt_vs_qoracle_cosines"]).tolist(),
                    "raw_oracle_overlap": oracle_diag["raw_oracle_overlap"],
                }
            })
        if dump_oracle_old_row_responses and (
            dump_oracle_old_row_response_block == 0
            or int(dump_oracle_old_row_response_block) == int(j) + 1
        ):
            oracle_old_row_responses_dump(
                M_gain, V_exact, optimize_r, old_row_memory_in,
                label=f" block={int(j)+1}",
            )
        if oracle_candidate_status is not None:
            verdict = "accepted" if oracle_candidate_status["accepted"] else "rejected"
            print(
                "oracle_candidate_check: "
                f"{verdict} optimizer_sum={oracle_candidate_status['optimizer_sum']:.12f} "
                f"candidate_sum={oracle_candidate_status['candidate_sum']:.12f}"
            )
    # -------------------------------------------------------------------------

    V_new = build_entropyscore_forget_direction_basis(
        M_gain,
        V_score,
        active_r,
        aux_method=aux_direction_method,
    )
    aux_r = V_new.shape[1] - V_score.shape[1]
    s_aux = np.zeros(aux_r, dtype=np.float32)
    H_aux = np.full(aux_r, np.inf, dtype=np.float32)
    score_aux = np.zeros(aux_r, dtype=np.float32)
    s_new = np.concatenate([np.asarray(s_score, dtype=np.float32), s_aux])
    H_new = np.concatenate([np.asarray(H_score, dtype=np.float32), H_aux])
    score_new = np.concatenate([np.asarray(score_score, dtype=np.float32), score_aux])

    Vt = V_new.T
    S = s_new
    Ut_carry, S_carry, Vt_carry, U_hat = left_projected_operator_svd_factors(Vt, M_gain)
    Vt = Vt_carry
    S = S_carry
    H_state = np.asarray(H_new[:len(S)], dtype=np.float32)
    score_state = np.asarray(score_new[:len(S)], dtype=np.float32)
    print(f'EntropyScoreForget active rank: {len(S)} / requested {k}, block rows={A_block.shape[0]}, total rows={rows_seen}')
    print(f'EntropyScoreForget row reference: {rows_ref}')
    print(f'EntropyScoreForget score rank: {optimize_r}, auxiliary rank: {aux_r}, auxiliary method: {aux_direction_method or "none"}')
    print(f'EntropyScoreForget basis solve time: {time.time() - step_start_time:.2f}s')
    print('EntropyScoreForget s_score:', s_score[:min(10, len(s_score))])
    print('EntropyScoreForget H_score:', H_score[:min(10, len(H_score))])
    print('EntropyScoreForget score_score:', score_score[:min(10, len(score_score))])
    print('EntropyScoreForget S_carry:', S[:min(10, len(S))])
    print({
        "EntropyScoreForget_diag": {
            "seed_rank": diag["seed_rank"],
            "max_rank": diag["max_rank"],
            "krylov_depth": diag["krylov_depth"],
            "residual_tol": diag["residual_tol"],
            "rows_ref": rows_ref,
            "score_rank": optimize_r,
            "aux_rank": aux_r,
            "auxiliary_method": aux_direction_method or "none",
            "reduced_optimizer": reduced_optimizer,
            "work_dtype": str(np.dtype(work_dtype)),
            "expansion_direction": diag["expansion_direction"],
            "reuse_line_search_grad": diag["reuse_line_search_grad"],
            "expansion_warm_start": diag["expansion_warm_start"],
            "post_expansion_maxit": diag["post_expansion_maxit"],
            "subspace_build_time": diag["subspace_build_time"],
            "reduced_solve_time": diag["reduced_solve_time"],
            "timing_totals": diag["timing_totals"],
            "timing_counts": diag["timing_counts"],
            "grad_perp_ratio_head": diag["grad_perp_ratio"][:min(10, len(diag["grad_perp_ratio"]))],
            "subspace_dims_head": diag["subspace_dims"][:min(10, len(diag["subspace_dims"]))],
            "expansion_iters_head": diag["expansion_iters"][:min(10, len(diag["expansion_iters"]))],
        }
    })

    S_quotient = S.copy()
    if num_Vs:
        if V_focus is None:
            block = Vt[:num_Vs, row_permutation[end_idx:]]
            weights = S_quotient[:num_Vs].reshape(-1, 1) if with_S else 1.0
            order_scores = np.sum((block * weights) ** 2, axis=0)
            indices = np.argsort(order_scores).reshape(-1)
            if not reverse:
                indices = indices[::-1]
        else:
            indices = np.argsort(np.sum((Vt[V_focus, row_permutation[end_idx:]]) ** 2, axis=0)).reshape(-1)
        row_permutation[end_idx:] = row_permutation[end_idx:][indices]

    num_save_files = 50
    should_save = (j == 0 or (j * (num_save_files - 1)) // W != ((j - 1) * (num_save_files - 1)) // W)
    if should_save:
        save_spectrum_comparison(S + total_S_reduced, S_exact, A_norm, name, j, dir_path,
                                 S_quotient=S_quotient, score_history=score_new, save_in_text=save_in_text,
                                 extra_fields={
                                     "s_score": np.asarray(s_score, dtype=np.float32),
                                     "H_score": np.asarray(H_score, dtype=np.float32),
                                     "score_score": np.asarray(score_score, dtype=np.float32),
                                     "S_carry": np.asarray(S, dtype=np.float32),
                                 })
        save_residuals(A_csr, S + total_S_reduced, Vt, S_exact, A_norm, name, j, dir_path, is_sym_psd,
                       row_permutation, start_idx, end_idx, save_in_text=save_in_text)
        if Vt_exact is not None:
            save_canonical_angles(Vt, Vt_exact, j, dir_path, save_in_text=save_in_text)
            exact_vectors = Vt_exact[:len(Vt), :].T
            current_total = np.linalg.norm(M_gain @ exact_vectors, axis=0)
            keep_projected_true = np.linalg.norm((S[:, None] * Vt) @ exact_vectors, axis=0)
            keep_left_projector_direct = np.linalg.norm(U_hat @ (U_hat.T @ (M_gain @ exact_vectors)), axis=0)
            save_leftout(
                Vt,
                S,
                Vt_exact,
                M_gain,
                j,
                dir_path,
                save_in_text=save_in_text,
                extra_fields={
                    "leftout_mode": "left_projected_operator_svd",
                    "keep_left_projected_true": keep_projected_true,
                    "keep_left_projector_direct": keep_left_projector_direct,
                    "keep_left_projector_vs_factor_gap": keep_left_projector_direct - keep_projected_true,
                    "throw_left_projected_true": current_total - keep_projected_true,
                    "left_projector_rank": np.int64(U_hat.shape[1]),
                },
            )

    old_row_memory_out = None
    if score_variant == "combined" and old_memory_size is not None and int(old_memory_size) > 0:
        seen_indices = row_permutation[:end_idx]
        seen_slice = A_csr[seen_indices, :]
        if sparse.issparse(seen_slice):
            seen_dense = seen_slice.toarray()
        else:
            seen_dense = np.asarray(seen_slice)
        if S_exact is not None and S_exact[0] != 0:
            seen_dense = seen_dense / S_exact[0]
        if col_permutation is not None:
            seen_dense = seen_dense[:, col_permutation]
        seen_dense = seen_dense.astype(work_dtype, copy=False)
        old_row_memory_out = select_old_row_memory(
            seen_dense,
            Vt.T,
            int(old_memory_size),
            np.random.default_rng(int(seed) + int(end_idx)),
        )

    state_out = {
        "V": Vt.T,
        "s": S,
        "s2": S ** 2,
        "H": H_state,
        "q": (S ** 4) * np.exp(-H_state),
        "score": score_state,
        "s_score": np.asarray(s_score, dtype=np.float32),
        "H_score": np.asarray(H_score, dtype=np.float32),
        "score_score": np.asarray(score_score, dtype=np.float32),
        "S_carry": np.asarray(S, dtype=np.float32),
        "rows_seen": rows_seen,
        "old_row_memory": old_row_memory_out,
        "diag": diag,
    }

    ret = [S, Vt, state_out]
    if reservoir_size > 0:
        ret.append((reservoir_idx, reservoir))
    if track_U:
        ret.append(None)
    if track_discarded:
        ret.append(discarded_list)
    if return_row_order:
        ret.append(row_permutation)
    if total_S_reduced > 0:
        ret.append(total_S_reduced)
    return ret


def future_hmean_online_step(next_window, future_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
              name, dir_path,
              col_permutation, track_U,
              track_discarded, discarded_list,
              num_Vs, with_S, reverse, return_row_order,
              total_S_reduced,
              reservoir_size, reservoir_idx, reservoir, reservoir_method,
              Vt=None, S=None, V_focus=None, reserved=None, save_in_text=True,
              score_rank=None,
              aux_direction_method=None,
              q0=5, qmax=None, krylov_depth=2, residual_tol=1e-2, expansion_maxit=8,
              num_restarts=3, maxit=40, tol=1e-8,
              reduced_optimizer="cex", work_dtype=np.float32,
              expansion_direction="residual",
              reuse_line_search_grad=True,
              expansion_warm_start=True,
              post_expansion_maxit=None,
              old_memory_size=None,
              rownorm_seed_first_block=True,
              rownorm_seed_all_blocks=True,
              seed=0):
    del first_window_size, U_exact, reservoir_method, window_indices
    if col_permutation is not None:
        next_window = next_window[:, col_permutation]
        if future_window is not None:
            future_window = future_window[:, col_permutation]
    if sparse.issparse(next_window):
        A_block = next_window.toarray().astype(work_dtype, copy=False)
    else:
        A_block = np.asarray(next_window, dtype=work_dtype)
    if future_window is None:
        A_future = np.zeros((0, A_block.shape[1]), dtype=work_dtype)
    elif sparse.issparse(future_window):
        A_future = future_window.toarray().astype(work_dtype, copy=False)
    else:
        A_future = np.asarray(future_window, dtype=work_dtype)

    rows_ref = max(int(A_csr.shape[0]), int(A_block.shape[0]), 2)
    step_start_time = time.time()
    state_prev = reserved
    if Vt is None or S is None:
        M_gain = A_block
        if rownorm_seed_first_block or rownorm_seed_all_blocks:
            V_init = np.asarray(row_norm_seed(A_block, k), dtype=work_dtype)
        else:
            V_init = None
        rows_seen = A_block.shape[0]
    else:
        S_prev = np.asarray(S, dtype=work_dtype).reshape(-1)
        Vt_prev = np.asarray(Vt, dtype=work_dtype)
        B_top = (S_prev[:, None] * Vt_prev).astype(work_dtype, copy=False)
        M_gain = np.vstack([B_top, A_block]).astype(work_dtype, copy=False)
        if rownorm_seed_all_blocks:
            V_init = np.asarray(row_norm_seed(A_block, k), dtype=work_dtype)
        else:
            V_init = Vt_prev.T
        prev_rows_seen = 0 if state_prev is None else int(state_prev.get("rows_seen", 0))
        rows_seen = prev_rows_seen + A_block.shape[0]

    active_r = min(k, M_gain.shape[0], M_gain.shape[1])
    if score_rank is None:
        optimize_r = active_r
    else:
        optimize_r = min(max(0, int(score_rank)), active_r)
    old_row_memory_in = None if state_prev is None else state_prev.get("old_row_memory")
    V_score, s_score, H_score, score_score, diag = entropy_iter_basis_forget(
        M_gain=M_gain,
        active_r=optimize_r,
        rows_ref=rows_ref,
        V_init=V_init,
        q0=q0,
        qmax=qmax,
        krylov_depth=krylov_depth,
        residual_tol=residual_tol,
        expansion_maxit=expansion_maxit,
        num_restarts=num_restarts,
        maxit=maxit,
        tol=tol,
        rng=np.random.default_rng(0),
        verbose=True,
        state_prev=state_prev,
        A_block=A_block,
        rows_total=rows_seen,
        reduced_optimizer=reduced_optimizer,
        work_dtype=work_dtype,
        expansion_direction=expansion_direction,
        reuse_line_search_grad=reuse_line_search_grad,
        expansion_warm_start=expansion_warm_start,
        post_expansion_maxit=post_expansion_maxit,
        score_variant="combined",
        old_row_memory=old_row_memory_in,
    )

    V_default = np.ascontiguousarray(np.asarray(V_score[:, :active_r], dtype=np.float64))
    chosen_label = "opt2"
    chosen_score = np.nan
    svd_frame_used = 0
    prev_opt2 = None if state_prev is None else normed_vector(state_prev.get("prev_opt2"), dtype=np.float64)
    if active_r >= 2 and V_default.shape[1] >= 2:
        v1 = np.asarray(V_default[:, :1], dtype=np.float64)
        candidates = {
            "opt2": normed_vector(V_default[:, 1], dtype=np.float64),
            "mgain_deflated_svd": svd_complement(M_gain, v1),
            "block_complement": block_svd_complement(A_block, v1),
            "prev_opt2": orth_against(prev_opt2, v1, dtype=np.float64) if prev_opt2 is not None else None,
        }
        finite_records = {}
        gain1_max = 0.0
        gain2_max = 0.0
        for label, cand in candidates.items():
            if cand is None:
                continue
            gain1 = float(np.linalg.norm(np.asarray(A_block, dtype=np.float64) @ cand) ** 2)
            gain2 = float(np.linalg.norm(np.asarray(A_future, dtype=np.float64) @ cand) ** 2) if A_future.size else np.nan
            shape = response_shape(A_block, cand)
            finite_records[label] = {
                "vec": cand,
                "gain1": gain1,
                "gain2": gain2,
                "relH1": shape["relH"],
            }
            gain1_max = max(gain1_max, gain1)
            if np.isfinite(gain2):
                gain2_max = max(gain2_max, gain2)
        best_value = -np.inf
        for label, rec in finite_records.items():
            g1_share = rec["gain1"] / max(gain1_max, 1e-30)
            if np.isfinite(rec["gain2"]) and gain2_max > 0.0:
                g2_share = rec["gain2"] / gain2_max
            else:
                g2_share = np.nan
            relH1 = max(float(rec["relH1"]), 0.0) if np.isfinite(rec["relH1"]) else 0.0
            obj = hmean_score(g1_share, g2_share) * relH1
            rec["obj_future_online"] = obj
            if np.isfinite(obj) and obj > best_value:
                best_value = obj
                chosen_label = label
                chosen_score = obj
        chosen_v2 = finite_records.get(chosen_label, {}).get("vec")
        if chosen_v2 is not None:
            V_svd = rank2_svd_frame(V_default[:, 0], chosen_v2, M_gain, rank=active_r)
            if V_svd is not None and V_svd.shape[1] >= active_r:
                V_new = np.ascontiguousarray(V_svd[:, :active_r], dtype=np.float64)
                svd_frame_used = 1
            else:
                V_new = np.ascontiguousarray(V_default.copy(), dtype=np.float64)
                V_new[:, 1] = chosen_v2
                V_new = orthonormalize_columns(V_new[:, :active_r], dtype=np.float64)[:, :active_r]
        else:
            V_new = np.ascontiguousarray(V_default, dtype=np.float64)
    else:
        V_new = np.ascontiguousarray(V_default, dtype=np.float64)

    aux_r = 0
    if aux_direction_method is not None and V_new.shape[1] < active_r:
        V_padded = build_entropyscore_forget_direction_basis(
            M_gain, V_new, active_r, aux_method=aux_direction_method
        )
        aux_r = int(V_padded.shape[1]) - int(V_new.shape[1])
        V_new = np.ascontiguousarray(np.asarray(V_padded, dtype=np.float64))

    score_new = np.zeros(V_new.shape[1], dtype=np.float32)
    H_new = np.zeros(V_new.shape[1], dtype=np.float32)
    s_new = np.zeros(V_new.shape[1], dtype=np.float32)
    for col_idx in range(V_new.shape[1]):
        score_val, s_val, H_val = score_full_vector_combined(
            M_gain,
            A_block,
            V_new[:, col_idx],
            rows_ref,
            state_prev=state_prev,
            old_row_memory=old_row_memory_in,
        )
        score_new[col_idx] = float(score_val)
        s_new[col_idx] = float(s_val)
        H_new[col_idx] = float(H_val)

    Vt = np.ascontiguousarray(V_new.T, dtype=np.float64)
    Ut_carry, S_carry, Vt_carry, U_hat = left_projected_operator_svd_factors(Vt, M_gain)
    Vt = Vt_carry
    S = S_carry
    H_state = np.asarray(H_new[:len(S)], dtype=np.float32)
    score_state = np.asarray(score_new[:len(S)], dtype=np.float32)
    print(f'FutureHMeanOnline active rank: {len(S)} / requested {k}, block rows={A_block.shape[0]}, total rows={rows_seen}')
    print(f'FutureHMeanOnline score rank: {optimize_r}, future rows: {A_future.shape[0]}, aux rank: {aux_r}, aux method: {aux_direction_method or "none"}')
    print(f'FutureHMeanOnline chosen label: {chosen_label}, score={chosen_score}')
    print(f'FutureHMeanOnline basis solve time: {time.time() - step_start_time:.2f}s')
    print({
        "FutureHMeanOnline_diag": {
            "score_rank": optimize_r,
            "aux_rank": aux_r,
            "auxiliary_method": aux_direction_method or "none",
            "selected_label": chosen_label,
            "selected_score": chosen_score,
            "svd_frame_used": int(svd_frame_used),
            "rows_ref": rows_ref,
            "future_rows": int(A_future.shape[0]),
            "subspace_build_time": diag["subspace_build_time"],
            "reduced_solve_time": diag["reduced_solve_time"],
            "timing_totals": diag["timing_totals"],
            "timing_counts": diag["timing_counts"],
        }
    })

    S_quotient = S.copy()
    if num_Vs:
        if V_focus is None:
            block = Vt[:num_Vs, row_permutation[end_idx:]]
            weights = S_quotient[:num_Vs].reshape(-1, 1) if with_S else 1.0
            order_scores = np.sum((block * weights) ** 2, axis=0)
            indices = np.argsort(order_scores).reshape(-1)
            if not reverse:
                indices = indices[::-1]
        else:
            indices = np.argsort(np.sum((Vt[V_focus, row_permutation[end_idx:]]) ** 2, axis=0)).reshape(-1)
        row_permutation[end_idx:] = row_permutation[end_idx:][indices]

    num_save_files = 50
    should_save = (j == 0 or (j * (num_save_files - 1)) // W != ((j - 1) * (num_save_files - 1)) // W)
    if should_save:
        save_spectrum_comparison(S + total_S_reduced, S_exact, A_norm, name, j, dir_path,
                                 S_quotient=S_quotient, score_history=score_new, save_in_text=save_in_text,
                                 extra_fields={
                                     "selected_label": chosen_label,
                                     "selected_score": np.float64(chosen_score) if np.isfinite(chosen_score) else np.nan,
                                     "svd_frame_used": np.int64(svd_frame_used),
                                 })
        save_residuals(A_csr, S + total_S_reduced, Vt, S_exact, A_norm, name, j, dir_path, is_sym_psd,
                       row_permutation, start_idx, end_idx, save_in_text=save_in_text)
        if Vt_exact is not None:
            save_canonical_angles(Vt, Vt_exact, j, dir_path, save_in_text=save_in_text)
            exact_vectors = Vt_exact[:len(Vt), :].T
            current_total = np.linalg.norm(M_gain @ exact_vectors, axis=0)
            keep_projected_true = np.linalg.norm((S[:, None] * Vt) @ exact_vectors, axis=0)
            keep_left_projector_direct = np.linalg.norm(U_hat @ (U_hat.T @ (M_gain @ exact_vectors)), axis=0)
            save_leftout(
                Vt,
                S,
                Vt_exact,
                M_gain,
                j,
                dir_path,
                save_in_text=save_in_text,
                extra_fields={
                    "leftout_mode": "left_projected_operator_svd",
                    "keep_left_projected_true": keep_projected_true,
                    "keep_left_projector_direct": keep_left_projector_direct,
                    "keep_left_projector_vs_factor_gap": keep_left_projector_direct - keep_projected_true,
                    "throw_left_projected_true": current_total - keep_projected_true,
                    "left_projector_rank": np.int64(U_hat.shape[1]),
                },
            )

    old_row_memory_out = None
    if old_memory_size is not None and int(old_memory_size) > 0:
        seen_indices = row_permutation[:end_idx]
        seen_slice = A_csr[seen_indices, :]
        if sparse.issparse(seen_slice):
            seen_dense = seen_slice.toarray()
        else:
            seen_dense = np.asarray(seen_slice)
        if S_exact is not None and S_exact[0] != 0:
            seen_dense = seen_dense / S_exact[0]
        if col_permutation is not None:
            seen_dense = seen_dense[:, col_permutation]
        seen_dense = seen_dense.astype(work_dtype, copy=False)
        old_row_memory_out = select_old_row_memory(
            seen_dense,
            Vt.T,
            int(old_memory_size),
            np.random.default_rng(int(seed) + int(end_idx)),
        )

    state_out = {
        "V": Vt.T,
        "s": S,
        "s2": S ** 2,
        "H": H_state,
        "q": (S ** 4) * np.exp(-H_state),
        "score": score_state,
        "s_score": np.asarray(s_new, dtype=np.float32),
        "H_score": np.asarray(H_new, dtype=np.float32),
        "score_score": np.asarray(score_new, dtype=np.float32),
        "S_carry": np.asarray(S, dtype=np.float32),
        "rows_seen": rows_seen,
        "old_row_memory": old_row_memory_out,
        "prev_opt2": None if V_new.shape[1] < 2 else np.asarray(V_new[:, 1], dtype=np.float32),
        "diag": diag,
    }

    ret = [S, Vt, state_out]
    if reservoir_size > 0:
        ret.append((reservoir_idx, reservoir))
    if track_U:
        ret.append(None)
    if track_discarded:
        ret.append(discarded_list)
    if return_row_order:
        ret.append(row_permutation)
    if total_S_reduced > 0:
        ret.append(total_S_reduced)
    return ret


def run_future_hmean_online_reference_experiment(
    matrix_name,
    method,
    stream_size,
    k,
    score_rank,
    output_dir,
    run_name,
    forget_params=None,
    save_in_text=True,
):
    if matrix_name != "mixed-tail-sharp":
        raise ValueError("Reference future_hmean_online path is only wired for mixed-tail-sharp.")
    if method not in {"future_hmean_online", "isvd"}:
        raise ValueError(f"Unsupported reference policy: {method}")

    half_window_mod = load_test_matrices_fast_module(
        "half_window_sliding_hmean_experiment_local",
        "half_window_sliding_hmean_experiment.py",
    )
    probe = half_window_mod.probe

    forget_params = dict(forget_params or {})
    args = type("Args", (), {})()
    args.matrix = matrix_name
    args.half_win = int(stream_size)
    args.rank = int(k)
    args.n = 1024
    args.preset = "fast"
    args.seed = 0
    args.shuffle_rows = True
    args.row_shuffle_seed = 0
    args.old_memory_size = int(stream_size)
    work_dtype = forget_params.get("work_dtype", np.float32)
    args.dtype = "float64" if np.dtype(work_dtype) == np.float64 else "float32"
    args.q0 = int(forget_params.get("q0", 8))
    args.qmax = int(forget_params.get("qmax", 48))
    args.krylov_depth = int(forget_params.get("krylov_depth", 2))
    args.residual_tol = float(forget_params.get("residual_tol", 0.01))
    args.expansion_maxit = int(forget_params.get("expansion_maxit", 8))
    args.num_restarts = int(forget_params.get("num_restarts", 3))
    args.maxit = int(forget_params.get("maxit", 120))
    args.tol = float(forget_params.get("tol", 1e-8))
    args.post_expansion_maxit = int(forget_params.get("post_expansion_maxit", 80))
    args.patience = int(forget_params.get("patience", 5))
    args.patience_rel_tol = float(forget_params.get("patience_rel_tol", 1e-5))
    args.r_sig = 2
    args.alpha_sig = 0.003
    args.alpha_tail = 0.0145
    args.tail_scale = 0.99
    args.sigma1 = 0.991
    args.v_type = "rand"

    np.random.seed(args.seed)
    A, V_exact, _, sigma1 = probe.generate_matrix_input(
        matrix=matrix_name,
        n=args.n,
        preset=args.preset,
        seed=args.seed,
        r_sig=args.r_sig,
        alpha_sig=args.alpha_sig,
        alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale,
        sigma1=args.sigma1,
        v_type=args.v_type,
        shuffle_rows=args.shuffle_rows,
        row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, dtype=np.float64)
    V_exact = np.asarray(V_exact, dtype=np.float64)
    _ = half_window_mod.run_pair_stream(A, V_exact, sigma1, args, method, args.half_win, sliding=False)
    result = half_window_mod.run_pair_stream(A, V_exact, sigma1, args, method, args.half_win, sliding=True)
    summary = half_window_mod.summarize_result(result)

    dir_path = os.path.join(output_dir, run_name)
    os.makedirs(dir_path, exist_ok=True)
    rows = result["rows"]
    final_row = rows[-1] if rows else {}
    if save_in_text:
        save_txt(
            os.path.join(dir_path, "other_info.txt"),
            time_elapsed=np.float64(result.get("elapsed", np.nan)),
            reference_policy=method,
            reference_mode=result.get("mode"),
            reference_summary=summary,
        )
        save_txt(
            os.path.join(dir_path, "reference_summary.txt"),
            summary=summary,
            final_row=final_row,
            rows=rows,
        )
    else:
        np.savez(
            os.path.join(dir_path, "other_info.npz"),
            time_elapsed=np.float64(result.get("elapsed", np.nan)),
            reference_policy=method,
            reference_mode=result.get("mode"),
            reference_summary=summary,
            rows=rows,
            allow_pickle=True,
        )
    print({"reference_future_hmean_online_summary": summary})
    return result, summary

def isvd(A_csr, S_exact=None, Vt_exact=None, U_exact=None, 
         first_window_size=100, k=None,
         num_windows=None, row_permutation=None, name="temp", output_dir="figures", is_sym_psd=False,
         num_Vs=None, track_U=False, track_discarded=False, with_S=False, V_focus=None, reverse=False,
         return_row_order=False, stream_size=None, col_permutation=None, reservoir_size=0, reservoir_method="uniform",
         method="isvd", use_true_matrix=False, track_reconstruction_error=False, threshold_factor=100,# nystrom 
         score_rank=None,
         forget_params=None,
         save_in_text=True,
         ):
    global Vt

    import time
    start_time = time.time()
    U = None
    m, n = A_csr.shape
    
    # W = num_windows  # number of windows (columns in this case)
    # l = m // W  # window size
    # k = k if k and k < l else l-1 # Number of singular values/vectors to compute
    # r = min(k, m, l)
    k = first_window_size if k is None else k # TODO
    stream_size = first_window_size if stream_size is None else stream_size
    W = (m - first_window_size) // stream_size + 1

    # Create the directory if it doesn't exist
    dir_path = f"{output_dir}/{name}/"
    directory = os.path.dirname(dir_path) 
    if directory and not os.path.exists(directory):
        print("Making directory:", directory)
        os.makedirs(directory)
    
    # Create a permutation of row indices
    row_permutation = row_permutation if row_permutation is not None else np.arange(m)

    sp_norm = sparse.linalg.norm if isinstance(A_csr, csr_matrix) else np.linalg.norm

    A_norm = sp_norm(A_csr) if A_csr.shape[1] < 5e4 else -1  # TODO
    inverse_perm = None

    total_S_reduced = 0
    # if track_discarded:
    discarded_list = []
    # if reservoir_size > 0:
    reservoir = np.zeros((reservoir_size, n), dtype=float)
    reservoir_idx = np.zeros((reservoir_size,), dtype=int)
    # if track_reconstruction_error:
    reconstruction_errors = []
    Vt, S = None, None
    B = None
    forget_params = dict(forget_params or {})

    forget_aux_method = resolve_entropyscore_forget_aux_method(method)
    forget_combined_hybrid = resolve_entropyscore_combined_hybrid(method)
    if method == "nystrom" or method == "isvd" or method == "entropyscore" or method == "entropyscore_fast" or method == "entropyscore_expansion" or method == "entropyscore_forget" or is_future_hmean_online_method(method) or forget_aux_method is not None or forget_combined_hybrid is not None or "isvd1by1" in method or "demix" in method or "isvdls" in method or "isvdst" in method:
        removed_rows = None
        method1_num_windows = None
    else:   
        temp = method.split("_")
        if len(temp) == 3:
            removed_rows = "window"
            method1, method1_num_windows, method2 = temp
        elif len(temp) == 4:
            method1, method1_num_windows, method2, removed_rows = temp
            removed_rows = int(removed_rows)
        else: 
            raise NotImplementedError
        
        if method1 == "isvdnew":
            raise Exception("Not possible")

        if method1 == "nystrom" and method2 == "isvdnew":
            removed_rows = None
        if (method1 == "isvd") and method2 == "nystrom":
            removed_rows = None

        method1_num_windows = int(method1_num_windows)


    reserved = None
    current_eigenvector_idx = 0
    print("Num windows:", W)
    for j in range(W+1):
        # All eigenvectors converged
        if current_eigenvector_idx == k:
            ret = [S, Vt]
            np.savez(os.path.join(dir_path, f'row_order_final.npz'),
                    row_permutation=row_permutation,)
            if track_U:
                ret.append(U)
            if track_discarded:
                ret.append(discarded_list)
            if return_row_order:
                ret.append(row_permutation)
            if total_S_reduced > 0:
                ret.append(total_S_reduced)
            return ret

        print("Window:", j+1)        
        
        # Calculate the start and end indices for this window
        if j == 0:
            start_idx = 0
            end_idx = min(first_window_size, m)
        else:
            end_idx_window_1 = min(first_window_size, m)
            start_idx = end_idx_window_1 + (j-1)*stream_size
            end_idx = min(end_idx_window_1 + j*stream_size, m)
        if end_idx <= start_idx:
            break

        if method1_num_windows and j < method1_num_windows:
            current_method = method1
        elif method1_num_windows:
            current_method = method2
        else:
            current_method = method
        
        # Extract the next window
        window_indices = row_permutation[start_idx:end_idx]
        next_window = A_csr[window_indices, :] / S_exact[0] if S_exact is not None and S_exact[0] != 0 else A_csr[window_indices, :]
        future_window = None
        future_start_idx = end_idx
        future_end_idx = min(end_idx + stream_size, m)
        if future_end_idx > future_start_idx:
            future_indices = row_permutation[future_start_idx:future_end_idx]
            future_window = A_csr[future_indices, :] / S_exact[0] if S_exact is not None and S_exact[0] != 0 else A_csr[future_indices, :]
        if S_exact is None:
            print("No S_exact provided, skipping normalization of input matrix")
        
        def step_function(method):
            if method == "nystrom":
                print("Doing Nystrom..")
                ret = nystrom_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
                window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
                name, dir_path, 
                col_permutation, track_U, reservoir_size, 
                num_Vs, with_S, reverse,
                threshold_factor, track_reconstruction_error, reconstruction_errors,
                use_true_matrix, m, return_row_order,
                total_S_reduced, 
                Vt=Vt, S=S,inverse_perm=inverse_perm, V_focus=V_focus,)
            elif method == "isvd":
                print("Doing iSVD..")
                ret = isvd_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
                window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
                name, dir_path, 
                col_permutation, track_U, 
                track_discarded, discarded_list,
                num_Vs, with_S, reverse, return_row_order,
                total_S_reduced,
                reservoir_size, reservoir_idx, reservoir, reservoir_method,
                Vt=Vt, S=S, V_focus=V_focus, reserved=reserved)
            elif method == "entropyscore":
                print("Doing EntropyScore..")
                ret = entropyscore_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
                window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
                name, dir_path,
                col_permutation, track_U,
                track_discarded, discarded_list,
                num_Vs, with_S, reverse, return_row_order,
                total_S_reduced,
                reservoir_size, reservoir_idx, reservoir, reservoir_method,
                Vt=Vt, S=S, V_focus=V_focus, reserved=reserved, save_in_text=save_in_text)
            elif method == "entropyscore_fast":
                print("Doing EntropyScoreFast..")
                ret = entropyscore_fast_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
                window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
                name, dir_path,
                col_permutation, track_U,
                track_discarded, discarded_list,
                num_Vs, with_S, reverse, return_row_order,
                total_S_reduced,
                reservoir_size, reservoir_idx, reservoir, reservoir_method,
                Vt=Vt, S=S, V_focus=V_focus, reserved=reserved, save_in_text=save_in_text)
            elif method == "entropyscore_expansion":
                print("Doing EntropyScoreExpansion..")
                ret = entropyscore_expansion_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
                window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
                name, dir_path,
                col_permutation, track_U,
                track_discarded, discarded_list,
                num_Vs, with_S, reverse, return_row_order,
                total_S_reduced,
                reservoir_size, reservoir_idx, reservoir, reservoir_method,
                Vt=Vt, S=S, V_focus=V_focus, reserved=reserved, save_in_text=save_in_text)
            elif (method == "entropyscore_forget"
                  or resolve_entropyscore_forget_aux_method(method) is not None
                  or resolve_entropyscore_combined_hybrid(method) is not None):
                print("Doing EntropyScoreForget..")
                if score_rank is not None:
                    effective_score_rank = max(0, min(int(k), int(score_rank)))
                else:
                    effective_score_rank = entropyscore_combined_hybrid_score_rank(method, k)
                effective_forget_params = dict(forget_params)
                combined_hybrid_cfg = resolve_entropyscore_combined_hybrid(method)
                if combined_hybrid_cfg is not None:
                    effective_forget_params.setdefault(
                        "score_variant", combined_hybrid_cfg.get("score_variant", "forget")
                    )
                    if effective_forget_params.get("score_variant") == "combined":
                        effective_forget_params.setdefault(
                            "old_memory_size",
                            stream_size if stream_size is not None else first_window_size,
                        )
                        effective_forget_params.setdefault("rownorm_seed_first_block", True)
                        effective_forget_params.setdefault("rownorm_seed_all_blocks", True)
                ret = entropyscore_forget_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
                window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
                name, dir_path,
                col_permutation, track_U,
                track_discarded, discarded_list,
                num_Vs, with_S, reverse, return_row_order,
                total_S_reduced,
                reservoir_size, reservoir_idx, reservoir, reservoir_method,
                Vt=Vt, S=S, V_focus=V_focus, reserved=reserved, save_in_text=save_in_text,
                score_rank=effective_score_rank,
                aux_direction_method=resolve_entropyscore_forget_aux_method(method),
                **effective_forget_params)
            elif is_future_hmean_online_method(method):
                print("Doing FutureHMeanOnline..")
                future_hybrid_cfg = resolve_future_hmean_online_hybrid(method)
                if score_rank is not None:
                    effective_score_rank = max(0, min(int(k), int(score_rank)))
                else:
                    hybrid_score_rank = future_hmean_online_hybrid_score_rank(method, k)
                    if hybrid_score_rank is not None:
                        effective_score_rank = hybrid_score_rank
                    else:
                        effective_score_rank = min(2, int(k))
                aux_direction_method = None
                if future_hybrid_cfg is not None:
                    aux_direction_method = future_hybrid_cfg.get("aux_method")
                effective_forget_params = dict(forget_params)
                effective_forget_params.setdefault("old_memory_size", stream_size if stream_size is not None else first_window_size)
                effective_forget_params.setdefault("rownorm_seed_first_block", True)
                effective_forget_params.setdefault("rownorm_seed_all_blocks", True)
                ret = future_hmean_online_step(next_window, future_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
                window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
                name, dir_path,
                col_permutation, track_U,
                track_discarded, discarded_list,
                num_Vs, with_S, reverse, return_row_order,
                total_S_reduced,
                reservoir_size, reservoir_idx, reservoir, reservoir_method,
                Vt=Vt, S=S, V_focus=V_focus, reserved=reserved, save_in_text=save_in_text,
                score_rank=effective_score_rank,
                aux_direction_method=aux_direction_method,
                **effective_forget_params)
            elif method == "isvdls":
                print("Doing iSVD..")
                ret = isvd_ls_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
                window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
                name, dir_path, 
                col_permutation, track_U, 
                track_discarded, discarded_list,
                num_Vs, with_S, reverse, return_row_order,
                total_S_reduced,
                reservoir_size, reservoir_idx, reservoir, reservoir_method,
                Vt=Vt, S=S, V_focus=V_focus, reserved=reserved)
            elif method == "isvdls2":
                print("Doing iSVD..")
                ret = isvd_ls_2_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
                window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
                name, dir_path, 
                col_permutation, track_U, 
                track_discarded, discarded_list,
                num_Vs, with_S, reverse, return_row_order,
                total_S_reduced,
                reservoir_size, reservoir_idx, reservoir, reservoir_method,
                Vt=Vt, S=S, V_focus=V_focus, reserved=reserved)
            elif method == "isvdnew":
                print("Doing iSVD..")
                ret = isvd_new_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
                window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
                name, dir_path, 
                col_permutation, track_U, 
                track_discarded, discarded_list,
                num_Vs, with_S, reverse, return_row_order,
                total_S_reduced,
                reservoir_size, reservoir_idx, reservoir, inverse_perm,
                Vt=Vt, S=S, V_focus=V_focus, )
            elif method == "isvd1by1":
                print("Doing iSVD..")
                ret = isvd_1by1_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
                window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
                name, dir_path, 
                col_permutation, track_U, 
                track_discarded, discarded_list,
                num_Vs, with_S, reverse, return_row_order,
                total_S_reduced,
                reservoir_size, reservoir_idx, reservoir, current_eigenvector_idx,
                Vt=Vt, S=S, V_focus=V_focus, reserved=reserved)
            elif method == "isvd1by1new":
                print("Doing iSVD..")
                ret = isvd_1by1_new_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
                window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
                name, dir_path, 
                col_permutation, track_U, 
                track_discarded, discarded_list,
                num_Vs, with_S, reverse, return_row_order,
                total_S_reduced,
                reservoir_size, reservoir_idx, reservoir, current_eigenvector_idx,
                Vt=Vt, S=S, V_focus=V_focus, reserved=reserved)
            elif method == "isvddemix":
                print("Doing iSVD..")
                ret = isvd_demix_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
                window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
                name, dir_path, 
                col_permutation, track_U, 
                track_discarded, discarded_list,
                num_Vs, with_S, reverse, return_row_order,
                total_S_reduced,
                reservoir_size, reservoir_idx, reservoir, reservoir_method,
                Vt=Vt, S=S, V_focus=V_focus, reserved=reserved)
            elif method == "isvddemix2":
                print("Doing iSVD..")
                ret = isvd_demix_2_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
                window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
                name, dir_path, 
                col_permutation, track_U, 
                track_discarded, discarded_list,
                num_Vs, with_S, reverse, return_row_order,
                total_S_reduced,
                reservoir_size, reservoir_idx, reservoir, reservoir_method,
                Vt=Vt, S=S, V_focus=V_focus, reserved=reserved)
            elif method == "isvddemix3":
                print("Doing iSVD..")
                ret = isvd_demix_3_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
                window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
                name, dir_path, 
                col_permutation, track_U, 
                track_discarded, discarded_list,
                num_Vs, with_S, reverse, return_row_order,
                total_S_reduced,
                reservoir_size, reservoir_idx, reservoir, reservoir_method,
                Vt=Vt, S=S, V_focus=V_focus, reserved=reserved)
            elif method == "isvdst":
                print("Doing iSVD..")
                ret = isvd_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
                window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
                name, dir_path, 
                col_permutation, track_U, 
                track_discarded, discarded_list,
                num_Vs, with_S, reverse, return_row_order,
                total_S_reduced,
                reservoir_size, reservoir_idx, reservoir, reservoir_method,
                Vt=Vt, S=S, V_focus=V_focus, reserved=reserved,
                use_soft_threshold=True, use_Ghashami=False, save_in_text=save_in_text)
            elif method == "isvdstG":
                print("Doing iSVD..")
                ret = isvd_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
                window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
                name, dir_path, 
                col_permutation, track_U, 
                track_discarded, discarded_list,
                num_Vs, with_S, reverse, return_row_order,
                total_S_reduced,
                reservoir_size, reservoir_idx, reservoir, reservoir_method,
                Vt=Vt, S=S, V_focus=V_focus, reserved=reserved,
                use_soft_threshold=True, use_Ghashami=False, save_in_text=save_in_text)
            elif method == "isvddemixst":
                print("Doing iSVD..")
                ret = isvd_demix_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
                window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
                name, dir_path, 
                col_permutation, track_U, 
                track_discarded, discarded_list,
                num_Vs, with_S, reverse, return_row_order,
                total_S_reduced,
                reservoir_size, reservoir_idx, reservoir, reservoir_method,
                Vt=Vt, S=S, V_focus=V_focus, reserved=reserved,
                use_soft_threshold=True, use_Ghashami=False)
            elif method == "isvddemixstG":
                print("Doing iSVD..")
                ret = isvd_demix_step(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
                window_indices, A_csr, S_exact, Vt_exact, U_exact, A_norm, is_sym_psd,
                name, dir_path, 
                col_permutation, track_U, 
                track_discarded, discarded_list,
                num_Vs, with_S, reverse, return_row_order,
                total_S_reduced,
                reservoir_size, reservoir_idx, reservoir, reservoir_method,
                Vt=Vt, S=S, V_focus=V_focus, reserved=reserved,
                use_soft_threshold=True, use_Ghashami=True)
            return ret

        # if method == "nystrom" or method == "isvd":
        #     ret = step_function(method)
        #     removed_rows = None
        # else:
        #     temp = method.split("_")
        #     if len(temp) == 3:
        #         removed_rows = "window"
        #         method1, method1_num_windows, method2 = temp
        #     elif len(temp) == 4:
        #         method1, method1_num_windows, method2, removed_rows = temp
        #         removed_rows = int(removed_rows)
        #     else: 
        #         raise NotImplementedError
            
        #     if method1 == "isvd" and method2 == "nystrom":
        #         removed_rows = None

        #     method1_num_windows = int(method1_num_windows)
        #     if j < method1_num_windows:
        #         ret = step_function(method1)
        #     else:
                # ret = step_function(method2)
        ret = step_function(current_method)

         
        S, Vt = ret[:2]


        entropies = []
        for i in range(Vt.shape[0]):
            col = Vt[i, :] 
            prob_dist = col**2 
            try:
                col_entropy = sp.stats.entropy(prob_dist, base=2)  # higher entropy means more spread out, lower means more concentrated)
            except:
                print(traceback.format_exc())
                raise
            entropies.append(col_entropy)
        print("Entropies:", [float(f"{entropies[i]:.3f}") for i in range(Vt.shape[0])])
        print("Normalized Entropies:", [float(f"{e:.3f}") for e in (np.array(entropies) / np.log2(Vt_exact.shape[1]))[:Vt.shape[0]]])
        normalized_entropies = np.array(entropies) / np.log2(Vt_exact.shape[1])
        if save_in_text:
            save_txt(
                os.path.join(dir_path, f'approx_entropy_{j}.txt'),
                entropies=np.array(entropies),
                normalized_entropies=normalized_entropies
            )
        else:
            np.savez(
                os.path.join(dir_path, f'approx_entropy_{j}.npz'),
                entropies=entropies,
                normalized_entropies=normalized_entropies,
                allow_pickle=True
            )
        
        # Save other per-window info
        if save_in_text:
            save_txt(
                os.path.join(dir_path, f'window_info_{j}.txt'),
                start_idx=start_idx,
                end_idx=end_idx,
            )
        else:
            np.savez(
                os.path.join(dir_path, f'window_info_{j}.npz'),
                start_idx=start_idx,
                end_idx=end_idx,
                allow_pickle=True
            )

        i = 2
        if current_method == "nystrom" or current_method == "isvdnew":
            inverse_perm = ret[i]
            i += 1
        elif (
            current_method == "isvd"
            or current_method == "entropyscore"
            or current_method == "entropyscore_fast"
            or current_method == "entropyscore_expansion"
            or current_method == "entropyscore_forget"
            or is_future_hmean_online_method(current_method)
            or resolve_entropyscore_forget_aux_method(current_method) is not None
            or resolve_entropyscore_combined_hybrid(current_method) is not None
            or "demix" in current_method
            or "isvdls" in current_method
            or "isvdst" in current_method
        ):
            reserved = ret[i]
            i += 1
            if reservoir_size > 0:
                reservoir_idx, reservoir = ret[i]
                i += 1
        elif current_method == "isvd1by1":
            current_eigenvector_idx = ret[i]
            i += 1
        elif current_method == "isvd1by1new":
            current_eigenvector_idx = ret[i]
            i += 1

        if track_U:
            U = ret[i]
            i += 1
        if track_discarded:
            discarded_list = ret[i]
            i += 1
        if return_row_order:
            row_permutation = ret[i]
            i += 1
        if total_S_reduced:
            total_S_reduced = ret[i]
            i += 1

        if removed_rows is None:
            pass 
        elif removed_rows == "window":
            # All window after switching
            if j >= method1_num_windows - 1 and end_idx < m:
                S = np.sqrt(S**2 * (1 - np.linalg.norm(Vt[:, window_indices], axis=1)**2))
        elif removed_rows == -1:
            if j == method1_num_windows - 1:
                S = S * np.linalg.norm(Vt[:, window_indices], axis=1)
        elif removed_rows == -2:
            if j == method1_num_windows - 1:
                 
                S = np.linalg.norm(next_window @ Vt.T, axis=0)  
        elif removed_rows == -3:
            if j == method1_num_windows - 1:
                S = S_exact[:k] * np.linalg.norm(Vt_exact[:k, window_indices], axis=1) 
                Vt = Vt_exact[:k,:]
        elif removed_rows == -4:
            if j == method1_num_windows - 1:
                S = np.linalg.norm(next_window @ Vt_exact[:k, :].T, axis=0)  
                Vt = Vt_exact[:k,:]
        elif removed_rows == -5:
            if j >= method1_num_windows - 1 and end_idx < m:
                S = np.sqrt(S_exact[:k]**2 * (1 - np.linalg.norm(Vt_exact[:k, window_indices], axis=1)**2))
        else:
            # Only the last window before switching
            if j == method1_num_windows - 1:
                # Remove last k rows of the current permutation
                S = np.sqrt(S**2 * (1 - np.linalg.norm(Vt[:, row_permutation[-removed_rows:]], axis=1)**2))
                num_Vs = None # Don't change permutations anymore (may allow changing in future)
        del next_window
        gc.collect()
        print_memory_usage(f"End of iSVD loop, window {j+1}")
    end_time = time.time()

    ret = [S, Vt]
    final_diag = {
        "method": method,
        "name": name,
        "elapsed_s": float(end_time - start_time),
        "active_rank": int(0 if S is None else len(S)),
        "requested_rank": int(k),
    }
    if S is not None and S_exact is not None:
        S_arr = np.asarray(S, dtype=np.float64)
        S_ex_arr = np.asarray(S_exact, dtype=np.float64)
        S_ex_topk = S_ex_arr[: len(S_arr)]
        scale = float(S_ex_arr[0]) if S_ex_arr.size and S_ex_arr[0] != 0.0 else 1.0
        S_recovered = S_arr * scale
        tr_S = float(np.sum(S_recovered))
        tr_S_ex = float(np.sum(S_ex_topk))
        final_diag["trace_S_recovered"] = tr_S
        final_diag["trace_S_exact_topk"] = tr_S_ex
        if tr_S_ex != 0.0:
            final_diag["trace_relerr"] = (tr_S - tr_S_ex) / tr_S_ex
        if A_norm and A_norm > 0:
            final_diag["S_relerr_l2"] = float(np.linalg.norm(S_recovered - S_ex_topk)) / float(A_norm)
        if len(S_arr):
            final_diag["top_sval_recovered"] = float(S_recovered[0])
            final_diag["top_sval_exact"] = float(S_ex_topk[0])
            if S_ex_topk[0] != 0.0:
                final_diag["top_sval_relerr"] = float(abs(S_recovered[0] - S_ex_topk[0]) / abs(S_ex_topk[0]))
    print({"isvd_final_diag": final_diag})

    if save_in_text:
        save_txt(
            os.path.join(dir_path, 'row_order_final.txt'),
            row_permutation=row_permutation
        )
    else:
        np.savez(
            os.path.join(dir_path, 'row_order_final.npz'),
            row_permutation=row_permutation
        )
    
    entropies = []
    for i in range(Vt_exact.shape[0]):
        col = Vt_exact[i, :] 
        prob_dist = col**2 
        col_entropy = sp.stats.entropy(prob_dist, base=2)  # higher entropy means more spread out, lower means more concentrated)
        entropies.append(col_entropy)
    true_normalized_entropy = np.array(entropies) / np.log2(Vt_exact.shape[1])

    if save_in_text:
        save_txt(
            os.path.join(dir_path, 'other_info.txt'),
            time_elapsed=end_time - start_time,
            true_normalized_entropy=true_normalized_entropy
        )
    else:
        np.savez(
            os.path.join(dir_path, 'other_info.npz'),
            other_info={
                "time_elapsed": end_time - start_time,
                "true_normalized_entropy": true_normalized_entropy,
            },
            allow_pickle=True
        )
    if track_U:
        ret.append(U)
    if track_discarded:
        ret.append(discarded_list)
    if return_row_order:
        ret.append(row_permutation)
    if total_S_reduced > 0:
        ret.append(total_S_reduced)
    return ret
