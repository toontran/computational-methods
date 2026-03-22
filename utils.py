import requests
import io
import tarfile
import os
import sys
import time
import requests
import re
import time
import gc

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

def save_txt(filename, **kwargs):
    data = {}

    for key, value in kwargs.items():
        if isinstance(value, np.ndarray):
            data[key] = {
                "type": "ndarray",
                "value": value.tolist()
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
        return os.path.join(self.cache_dir, f"rbf_n{self.n}_ls{self.lengthscale:g}_{self._cache_key()}")

    def _meta_path(self):
        return os.path.join(self._resolved_cache_dir(), "meta.json")

    def _block_path(self, i0, i1):
        return os.path.join(self._resolved_cache_dir(), f"block_{i0}_{i1}.npy")

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
        total_blocks = (self.n + self.block_size - 1) // self.block_size

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

        if self.verbose:
            print(f"[cache] done in {time.time() - t0:.2f}s -> {cache_root}")

    def _load_block(self, i0, i1):
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

        self._matvec_calls += 1
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

        self._matmat_calls += 1
        out = np.zeros((self.n, V.shape[1]), dtype=self.dtype)

        for i0 in range(0, self.n, self.block_size):
            i1 = min(i0 + self.block_size, self.n)
            K_block = self._load_block(i0, i1)
            out[i0:i1, :] = K_block @ V

        return out

    def rmatmat(self, V):
        return self.matmat(V)

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

def save_spectrum_comparison(S, S_exact, A_norm, name, iteration, dir_path, S_quotient=None, save_in_text=True):
    os.makedirs(dir_path, exist_ok=True)

    ext = "txt" if save_in_text else "npz"

    # First plot data
    filepath = os.path.join(dir_path, f"spectrum_data_{iteration}.{ext}")
    if save_in_text:
        save_txt(filepath, S=S, S_exact=S_exact, S_quotient=S_quotient, iteration=iteration)
    else:
        np.savez(
            filepath,
            S=S, S_exact=S_exact, S_quotient=S_quotient, iteration=iteration,
            allow_pickle=True
        )

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
                   A_norm, name, iteration, dir_path, is_sym_psd,
                   row_permutation, start_idx, end_idx, save_in_text=True):
    os.makedirs(dir_path, exist_ok=True)

    ext = "txt" if save_in_text else "npz"

    if is_sym_psd:
        approx_residuals_sym = []
        if A_csr.shape[1] < 5e4:
            for i in range(len(S)):
                approx_res = (A_csr @ Vt[i].T) - S[i] * Vt[i].T
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
                approx_res = (A_csr[window_indices, :] @ Vt[i].T) - S[i] * Vt[i, window_indices].T
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
                    A_csr[window_indices, :] @ Vt[i].T
                )
                sq_norm_V = np.dot(Vt[i, window_indices].T, Vt[i, window_indices].T)

                S_truncated_Rayleigh_full = np.dot(
                    Vt[i, row_permutation[:end_idx]].T,
                    A_csr[row_permutation[:end_idx], :] @ Vt[i].T
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
                    A_csr[window_indices, :] @ Vt[i].T
                ) - S_truncated_Rayleigh * Vt[i, window_indices].T

                approx_res_full = (
                    A_csr[row_permutation[:end_idx], :] @ Vt[i].T
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
                u = A_csr @ Vt[i].T
                u = u / np.linalg.norm(u)
                approx_res = (A_csr.T @ u) - S[i] * Vt[i].T
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
                             S, Vt, A_norm, A_csr, S_quotient,
                             name, iteration, dir_path, save_in_text=True):
    os.makedirs(dir_path, exist_ok=True)

    ext = "txt" if save_in_text else "npz"

    print_memory_usage(f"Before residual reservoir, window {iteration+1}")
    reservoir_Vt = reservoir @ Vt.T
    regular_Vt = A_csr @ Vt.T

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


def save_leftout(Vt, S, Vt_exact, combined, iteration, dir_path, additional_label="", save_in_text=True):
    current_total = np.linalg.norm(combined @ Vt_exact[:len(Vt), :].T, axis=0)
    keep = np.linalg.norm((S[:, None] * Vt) @ Vt_exact[:len(Vt), :].T, axis=0)
    throw = current_total - keep

    os.makedirs(dir_path, exist_ok=True)

    ext = "txt" if save_in_text else "npz"
    filepath = os.path.join(dir_path, f"leftout{additional_label}_data_{iteration}.{ext}")

    if save_in_text:
        save_txt(filepath, iteration=iteration, current_total=current_total, throw=throw)
    else:
        np.savez(
            filepath,
            iteration=iteration,
            current_total=current_total,
            throw=throw,
            allow_pickle=True
        )

    print(f"Leftout data saved successfully for iteration {iteration}")

    
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
    
def compute_svd(A, k, is_sparse=True):
    if not is_sparse:
        print("Small matrix")
        return sp.linalg.svd(A, lapack_driver="gesdd", full_matrices=False)
    else:
        print("Large matrix")
        u, s, vt = sp.sparse.linalg.svds(A, k=min(k+k//5, min(A.shape)-1))
        s = s[::-1]
        vt = vt[::-1, :]
        u = u[:, ::-1]
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
                    A_norm, name, j, dir_path, is_sym_psd,
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
        U_sketch, S, Vt = compute_svd(combined, k, is_sparse=is_sparse)
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
        del S, Vt
        gc.collect()
        print_memory_usage(f"Before, window {j+1}")
        U_sketch, S, Vt = compute_svd(combined, k, is_sparse=is_sparse)
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
    Vt, S, reservoir, reservoir_idx = isvd_partial_step_(next_window, row_permutation, j, start_idx, end_idx, first_window_size, k, W,
              window_indices, A_csr, Vt_exact,
              col_permutation, track_U, 
              track_discarded, discarded_list,
              reservoir_size, reservoir_idx, reservoir, reservoir_method,
              Vt=Vt, S=S, reserved=reserved,
              use_soft_threshold=use_soft_threshold, use_Ghashami=use_Ghashami,
              dir_path=dir_path, save_in_text=save_in_text)

    print("Vt shape:", Vt.shape)

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
    print("Vt shape:", Vt.shape)
    if j == 0 or (j * (num_save_files- 1)) // W != ((j - 1) * (num_save_files - 1)) // W:
        save_spectrum_comparison(S+total_S_reduced, S_exact, 
                                    A_norm, name, j, dir_path, S_quotient=S_quotient, save_in_text=save_in_text)
        save_residuals(A_csr, S+total_S_reduced, Vt, 
                        A_norm, name, j, dir_path, is_sym_psd,
                        row_permutation, start_idx, end_idx, save_in_text=save_in_text)
        if reservoir_size > 0:
            print("Vt shape:", Vt.shape)
            save_residuals_reservoir(reservoir, reservoir_idx, row_permutation,
                                        S, Vt, A_norm, A_csr, S_quotient, 
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



def isvd(A_csr, S_exact=None, Vt_exact=None, U_exact=None, 
         first_window_size=100, k=None,
         num_windows=None, row_permutation=None, name="temp", figure_dir="figures", is_sym_psd=False,
         num_Vs=None, track_U=False, track_discarded=False, with_S=False, V_focus=None, reverse=False,
         return_row_order=False, stream_size=None, col_permutation=None, reservoir_size=0, reservoir_method="uniform",
         method="isvd", use_true_matrix=False, track_reconstruction_error=False, threshold_factor=100,# nystrom 
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
    dir_path = f"{figure_dir}/{name}/"
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

    if method == "nystrom" or method == "isvd" or "isvd1by1" in method or "demix" in method or "isvdls" in method or "isvdst" in method:
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
                Vt=Vt, S=S, V_focus=V_focus, reserved=reserved, save_in_text=save_in_text)
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
        elif current_method == "isvd" or "demix" in current_method or "isvdls" in current_method or "isvdst" in current_method:
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