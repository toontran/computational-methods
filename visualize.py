import requests
import io
import tarfile
import os
import sys 
import time
import requests
import re

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from bs4 import BeautifulSoup
import matspy
import seaborn as sns

from scipy.sparse.linalg import svds
from scipy.io import mmread
import scipy.sparse as sparse
# import scipy.sparse.linalg.norm
from scipy.linalg import orthogonal_procrustes, subspace_angles, matrix_balance
from scipy.sparse._csr import csr_matrix
from scipy.optimize import linear_sum_assignment

from scipy import stats
from scipy.spatial.distance import pdist, squareform

import matplotlib
matplotlib.use('Agg')

def sample_4d_hyperboloid(num_points, a, b, c, d):
    np.random.seed(42)
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

import numpy as np

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

class StreamingRBFKernel:
    def __init__(self, points, gamma=1.0):
        self.points = points
        self.gamma = gamma
        self.n = len(points)
        
    def calculate_row(self, i):
        diff = self.points - self.points[i]
        sq_dists = np.sum(diff**2, axis=1)
        return np.exp(-sq_dists / (2*self.gamma**2))
    
    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_idx, col_idx = key
        else:
            row_idx, col_idx = key, slice(None)
        
        if isinstance(row_idx, int):
            row = self.calculate_row(row_idx)
            return row[col_idx]
        elif isinstance(row_idx, slice):
            start, stop, step = row_idx.indices(self.n)
            rows = np.array([self.calculate_row(i) for i in range(start, stop, step)])
            return rows[:, col_idx]
        elif isinstance(row_idx, (list, np.ndarray)):
            rows = np.array([self.calculate_row(i) for i in row_idx])
            return rows[:, col_idx]
        else:
            raise IndexError("Invalid index type")

    def __len__(self):
        return self.n

    @property
    def shape(self):
        return (self.n, self.n)

class StreamingKroneckerGraph:
    def __init__(self, SCALE, edgefactor):
        self.SCALE = SCALE
        self.N = 2**SCALE
        self.M = int(edgefactor * self.N)
        self.A, self.B, self.C = 0.57, 0.19, 0.19
        self.ab = self.A + self.B
        self.c_norm = self.C / (1 - (self.A + self.B))
        self.a_norm = self.A / (self.A + self.B)
        
        # Generate and store the edge list
        self.ijw = self._generate_edge_list()
        
    def _generate_edge_list(self):
        ijw = np.ones((3, self.M))
        for ib in range(1, self.SCALE + 1):
            ii_bit = np.random.uniform(0, 1, size=(1, self.M)) > self.ab
            jj_bit = np.random.uniform(0, 1, size=(1, self.M)) > (self.c_norm * ii_bit + self.a_norm * (~ii_bit))
            ijw[0:2] += 2**(ib - 1) * np.append(ii_bit, jj_bit, axis=0)
        
        ijw[2] = np.random.uniform(0, 1, size=(1, self.M))
        ijw[0] = np.random.permutation(ijw[0])
        ijw[1] = np.random.permutation(ijw[1])
        ijw[0:2] -= 1
        return ijw
    
    def _calculate_row(self, i):
        row = np.zeros(self.N)
        mask = (self.ijw[0] == i) | (self.ijw[1] == i)
        for j in range(self.M):
            if mask[j]:
                other_vertex = int(self.ijw[1][j] if self.ijw[0][j] == i else self.ijw[0][j])
                weight = self.ijw[2][j]
                if row[other_vertex] == 0 or weight < row[other_vertex]:
                    row[other_vertex] = weight
        return row
    
    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_idx, col_idx = key
        else:
            row_idx, col_idx = key, slice(None)
        
        if isinstance(row_idx, int):
            row = self._calculate_row(row_idx)
            return row[col_idx]
        elif isinstance(row_idx, slice):
            start, stop, step = row_idx.indices(self.N)
            rows = np.array([self._calculate_row(i) for i in range(start, stop, step)])
            return rows[:, col_idx]
        elif isinstance(row_idx, (list, np.ndarray)):
            rows = np.array([self._calculate_row(i) for i in row_idx])
            return rows[:, col_idx]
        else:
            raise IndexError("Invalid index type")

    def __len__(self):
        return self.N

    @property
    def shape(self):
        return (self.N, self.N)

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
    np.save('some_measure.npy', some_measure)
    
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

def save_spectrum_comparison(S, S_exact, A_norm, name, iteration, dir_path):
    # Create directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)

    # Save data for the first plot
    np.savez(os.path.join(dir_path, f'spectrum_data_{iteration}.npz'),
             S=S, S_exact=S_exact, iteration=iteration)

    # Save data for the second plot
    if not S_exact is None:
        np.savez(os.path.join(dir_path, f'diffspec_relA_data_{iteration}.npz'),
                 diff=np.abs(S - S_exact[:len(S)]) / A_norm, iteration=iteration)
    
        # Save data for the third plot
        np.savez(os.path.join(dir_path, f'diffspec_relS_data_{iteration}.npz'),
                 diff=np.abs(S - S_exact[:len(S)]) / np.abs(S_exact[:len(S)]), iteration=iteration)

    print(f"Data saved successfully for iteration {iteration}")

def save_canonical_angles(Vt, Vt_exact, iteration, dir_path, additional_label=""):
    # Compute the singular values of Q1.T @ Q2
    # print(Vt.shape, Vt_exact.shape)
    # print(Vt.shape, Vt_exact.shape
    C = Vt @ Vt_exact[:Vt.shape[0], :].T
    s = np.linalg.svd(C, compute_uv=False)
    
    # Compute the angles in radians
    eps = 1e-6
    assert np.all(s > -1.0 - eps) and np.all(s < 1.0 + eps), "Invalid canonical correlation found" 
    angles = np.arccos(np.clip(s, -1.0, 1.0))
    print("Subspace angle 2:", max(angles), np.mean(angles))
    
    epsilon = 1e-4
    s = -np.log(np.maximum(1 - s, epsilon))

    # Create directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)

    # Save data
    np.savez(os.path.join(dir_path, f'canonical_angles{additional_label}_data_{iteration}.npz'),
             s=s, iteration=iteration, C=C)

    print(f"Canonical angles data saved successfully for iteration {iteration}")

def save_residuals(A_csr, S, Vt,  
                   A_norm, name, iteration, dir_path, is_sym_psd, 
                   row_permutation, start_idx, end_idx,):
    # Create directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)

    if is_sym_psd:
        approx_residuals_sym = []
        for i in range(len(S)):
            approx_res = (A_csr @ Vt[i].T) - (S[i]) * Vt[i].T
            approx_residuals_sym.append(np.linalg.norm(approx_res) / A_norm)
        
        approx_residuals_sym = np.array(approx_residuals_sym)
        
        # Save symmetric PSD data
        np.savez(os.path.join(dir_path, f'residuals_sym_psd_data_{iteration}.npz'),
                 approx_residuals=approx_residuals_sym,
                 iteration=iteration,
                 A_norm=A_norm)
        
        window_indices = row_permutation[start_idx:end_idx]
        approx_residuals_sym = []
        for i in range(len(S)):
            approx_res = (A_csr[window_indices, :] @ Vt[i].T) - (S[i]) * Vt[i, window_indices].T
            approx_residuals_sym.append(np.linalg.norm(approx_res) / A_norm)
        
        approx_residuals_sym = np.array(approx_residuals_sym)
        
        # Save symmetric PSD data
        np.savez(os.path.join(dir_path, f'residuals_sym_psd_data_truncated_{iteration}.npz'),
                 approx_residuals=approx_residuals_sym,
                 iteration=iteration,
                 A_norm=A_norm)
        
        approx_residuals_sym = []
        approx_residuals_sym_full = []
        S_truncated_Rayleigh_list = []
        S_truncated_Rayleigh_full_list = []
        
        for i in range(len(S)):
            S_truncated_Rayleigh = np.dot(Vt[i, window_indices].T, A_csr[window_indices, :] @ Vt[i].T)
            sq_norm_V = np.dot(Vt[i, window_indices].T, Vt[i, window_indices].T)
            S_truncated_Rayleigh_full = np.dot(Vt[i, row_permutation[:end_idx]].T, A_csr[row_permutation[:end_idx], :] @ Vt[i].T)
            sq_norm_V_full = np.dot(Vt[i, row_permutatoin[:end_idx]].T, Vt[i, row_permutation[:end_idx]].T)
            if sq_norm_V == 0:
                S_truncated_Rayleigh = np.nan
            else:
                S_truncated_Rayleigh /= sq_norm_V
            if sq_norm_V_full == 0:
                S_truncated_Rayleigh_full = np.nan
            else:
                S_truncated_Rayleigh_full /= sq_norm_V_full
            approx_res = (A_csr[window_indices, :] @ Vt[i].T) - (S_truncated_Rayleigh) * Vt[i, window_indices].T
            approx_res_full = (A_csr[row_permutation[:end_idx], :] @ Vt[i].T) - (S_truncated_Rayleigh) * Vt[i, row_permutation[:end_idx]].T
            approx_residuals_sym.append(np.linalg.norm(approx_res) / A_norm)
            approx_residuals_sym_full.append(np.linalg.norm(approx_res) / A_norm)
            S_truncated_Rayleigh_list.append(S_truncated_Rayleigh)
            S_truncated_Rayleigh_full_list.append(S_truncated_Rayleigh_full)
        
        approx_residuals_sym = np.array(approx_residuals_sym)
        S_truncated_Rayleigh_list = np.array(S_truncated_Rayleigh_list)
        approx_residuals_sym_full = np.array(approx_residuals_sym_full)
        S_truncated_Rayleigh_full_list = np.array(S_truncated_Rayleigh_full_list)
        
        # Save symmetric PSD data
        np.savez(os.path.join(dir_path, f'residuals_sym_psd_data_truncated_Rayleigh_{iteration}.npz'),
                 approx_residuals=approx_residuals_sym,
                 approx_residuals_full=approx_residuals_sym_full,
                 iteration=iteration,
                 S_truncated_Rayleigh_list=S_truncated_Rayleigh_list,
                 S_truncated_Rayleigh_full_list=S_truncated_Rayleigh_full_list,
                 A_norm=A_norm,
                 )
    else:
        approx_residuals = []
        for i in range(len(S)):
            u = A_csr @ Vt[i].T
            u = u / np.linalg.norm(u)
            approx_res = (A_csr.T @ u) - (S[i]) * Vt[i].T
            approx_residuals.append(np.linalg.norm(approx_res) / A_norm)
        
        approx_residuals = np.array(approx_residuals)
        
        # Save non-symmetric PSD data
        np.savez(os.path.join(dir_path, f'residuals_data_{iteration}.npz'),
                 approx_residuals=approx_residuals,
                 iteration=iteration)

    print(f"Residuals data saved successfully for iteration {iteration}")


def isvd(A_csr, S_exact=None, Vt_exact=None, U_exact=None, 
         window_size=100, k=None,
         num_windows=None, row_permutation=None, name="temp", figure_dir="figures", is_sym_psd=False,
         num_Vs=None, track_U=False, track_discarded=False, with_S=False, reverse=False,
         return_row_order=False, stream_size=None, col_permutation=None):
    global Vt
    U = None
    m, n = A_csr.shape
    
    # W = num_windows  # number of windows (columns in this case)
    # l = m // W  # window size
    # k = k if k and k < l else l-1 # Number of singular values/vectors to compute
    # r = min(k, m, l)
    k = window_size if k is None else k 
    stream_size = window_size if stream_size is None else stream_size
    W = (m - window_size) // stream_size + 1

    # Create the directory if it doesn't exist
    dir_path = f"{figure_dir}/{name}/"
    directory = os.path.dirname(dir_path) 
    if directory and not os.path.exists(directory):
        print("Making directory:", directory)
        os.makedirs(directory)
    
    # Create a permutation of row indices
    row_permutation = row_permutation if row_permutation is not None else np.arange(m)

    sp_norm = sparse.linalg.norm if isinstance(A_csr, csr_matrix) else np.linalg.norm
    A_norm = sp_norm(A_csr) 

    total_S_reduced = 0
    if track_discarded:
        discarded_list = []
    print("Num windows:", W)
    for j in range(W+1):
        print("Window:", j+1)        
        
        # Calculate the start and end indices for this window
        if j == 0:
            start_idx = j * window_size
            end_idx = min((j + 1) * window_size, m)
        else:
            start_idx = j * stream_size
            end_idx = min((j + 1) * stream_size, m)
        if end_idx <= start_idx:
            break
        
        # Extract the next window
        window_indices = row_permutation[start_idx:end_idx]
#         print("Index:", end_idx, len(row_permutation))
        next_window = A_csr[window_indices, :]
        if not col_permutation is None:
            next_window = next_window[:, col_permutation]
        if isinstance(A_csr, csr_matrix):
            next_window = next_window.toarray()
        
        # print(next_window.shape)

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
            S = S[:window_size]
            Vt = Vt[:window_size, :]
            
            B = S.reshape(-1, 1) * Vt

            if track_U:
                U = U_sketch

        else:
        
            # Concatenate B[j-1] and the next window
            combined = np.concatenate((B, next_window), axis=0)
            
            # Perform SVD on the combined matrix
            # Reverse the order to get largest singular values first
            # _, S, Vt = svds(combined, k=r)
            # S = S[::-1]
            # Vt = Vt[::-1, :]
            
            U_sketch, S, Vt = sp.linalg.svd(combined, lapack_driver="gesdd", full_matrices=False)
            if track_discarded:
                print(f"Discarding: {S[window_size:].shape}/{S.shape}")
                discarded_list.append([S[window_size:], Vt[window_size:, :]])
            
            # Optional: Apply soft thresholding to singular values
            # S = soft_thresholding(S)
            # total_S_reduced += S[-1]
#             S = soft_thresholding_Ghashami(S)
#             S = soft_thresholding_SS(S)

            S = S[:window_size]
            Vt = Vt[:window_size, :]

            # Update B
            B = S.reshape(-1, 1) * Vt
#             print("B", B[0,:10])

            if track_U:
                 # Update U
                U_new = np.zeros((U.shape[0] + len(window_indices), U.shape[1] + len(window_indices)))
                U_new[:U.shape[0], :U.shape[1]] = U
                U_new[U.shape[0]:, U.shape[1]:] = np.eye(len(window_indices))
                U = U_new
                U = U @ U_sketch
#                 print("U", U.shape, U_sketch.shape)
                U = U[:, :window_size]
    
        # Plot
        # plot_spectrum_comparison(S, S_exact, 
        #                          A_norm, name, j, dir_path)
        # plot_residuals(A_csr, S, Vt, S_exact, Vt_exact, U_exact, 
        #                A_norm, name, j, dir_path, is_sym_psd) 
        # plot_canonical_angles(Vt, Vt_exact, 
        #                       j, dir_path)

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
            save_canonical_angles(U.T, U_exact.T, 
                                  j, dir_path, additional_label="_U")
        

        if not S_exact is None:
            print("Relative error in S:", np.linalg.norm(S - S_exact[:Vt.shape[0]]) / A_norm)
        # X = np.linalg.pinv(Vt_exact[:Vt.shape[0],:].T) @ Vt.T 
        # Vt_reconstructed = Vt_exact[:Vt.shape[0],:].T @ X
        # print("Reconstruct Vt from Vt_exact:", np.linalg.norm(Vt.T - Vt_reconstructed, 'fro'))
        # print("Projection F-norm error:", np.linalg.norm(Vt.T @ Vt - Vt_exact[:Vt.shape[0], :].T @ Vt_exact[:Vt.shape[0], :], 'fro'))
        # print("Trace correlation", np.trace(Vt @ Vt_exact[:Vt.shape[0], :].T @ Vt_exact[:Vt.shape[0], :] @ Vt.T) / min(Vt.T.shape[1], Vt_exact[:Vt.shape[0], :].T.shape[1]))
    
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

# def isvd_order_by_approx_V(A_csr, S_exact, Vt_exact, U_exact, 
#          num_windows=10, row_permutation=None, k=None, name="temp", figure_dir="figures", is_sym_psd=False,
#                           num_Vs=10):
#     global Vt
#     m, n = A_csr.shape
#     W = num_windows  # number of windows (columns in this case)
#     l = m // W  # window size
#     k = k if k and k < l else l-1 # Number of singular values/vectors to compute
#     r = min(k, m, l)

#     # Create the directory if it doesn't exist
#     name += f"_approxV_{num_Vs}"
#     dir_path = f"{figure_dir}/{name}/"
#     directory = os.path.dirname(dir_path) 
#     if directory and not os.path.exists(directory):
#         print("Making directory:", directory)
#         os.makedirs(directory)
    
#     # Create a permutation of row indices
#     row_permutation = row_permutation if row_permutation is not None else np.arange(m)

#     sp_norm = sparse.linalg.norm if isinstance(A_csr, csr_matrix) else np.linalg.norm
#     A_norm = sp_norm(A_csr)

#     total_S_reduced = 0
#     for j in range(W):
#         print("Window:", j+1)
        
#         # Calculate the start and end indices for this window
#         start_idx = j * l
#         end_idx = min((j + 1) * l, m)
        
#         # Extract the next window
#         window_indices = row_permutation[start_idx:end_idx]
#         print("Index:", end_idx, len(row_permutation))
#         next_window = A_csr[window_indices, :]
#         if isinstance(A_csr, csr_matrix):
#             next_window = next_window.toarray()
        
#         # print(next_window.shape)

#         if j == 0:
#              # Initial SVD for the first window
            
#             # Reverse the order to get largest singular values first
#             # _, S, Vt = svds(next_window, k=r)
#             # S = S[::-1]
#             # Vt = Vt[::-1, :]
            
#             _, S, Vt = sp.linalg.svd(next_window, lapack_driver="gesdd")
#             print(len(S))
#             S = S[:r]
#             Vt = Vt[:r, :]           
            
#             B = S.reshape(-1, 1) * Vt

#         else:
        
#             # Concatenate B[j-1] and the next window
#             combined = np.concatenate((B, next_window), axis=0)
            
#             # Perform SVD on the combined matrix
#             # Reverse the order to get largest singular values first
#             # _, S, Vt = svds(combined, k=r)
#             # S = S[::-1]
#             # Vt = Vt[::-1, :]
            
#             _, S, Vt = sp.linalg.svd(combined, lapack_driver="gesdd")
#             print(len(S))
#             S = S[:r]
#             Vt = Vt[:r, :]
            
#             # Optional: Apply soft thresholding to singular values
#             # S = soft_thresholding(S)
#             # total_S_reduced += S[-1]
#             # S = soft_thresholding_Ghashami(S)

#             # Update B
#             B = S.reshape(-1, 1) * Vt
    
#         # Plot
#         # plot_spectrum_comparison(S, S_exact, 
#         #                          A_norm, name, j, dir_path)
#         # plot_residuals(A_csr, S, Vt, S_exact, Vt_exact, U_exact, 
#         #                A_norm, name, j, dir_path, is_sym_psd) 
#         # plot_canonical_angles(Vt, Vt_exact, 
#         #                       j, dir_path)

#         row_permutation[end_idx:] = np.argsort(np.sum(Vt[:num_Vs, row_permutation[end_idx:]]**2, axis=0)).reshape(-1)[::-1]
        
#         # Plot
#         save_spectrum_comparison(S, S_exact, 
#                                  A_norm, name, j, dir_path)
#         save_residuals(A_csr, S, Vt, S_exact, Vt_exact, U_exact, 
#                        A_norm, name, j, dir_path, is_sym_psd) 
#         save_canonical_angles(Vt, Vt_exact, 
#                               j, dir_path)
#         print("Reconstruction quality:", np.linalg.norm(Vt - Vt_exact[:Vt.shape[0], :], 'fro'))
#         print("Relative error in S:", np.linalg.norm(S - S_exact[:Vt.shape[0]]) / A_norm)
#         # X = np.linalg.pinv(Vt_exact[:Vt.shape[0],:].T) @ Vt.T 
#         # Vt_reconstructed = Vt_exact[:Vt.shape[0],:].T @ X
#         # print("Reconstruct Vt from Vt_exact:", np.linalg.norm(Vt.T - Vt_reconstructed, 'fro'))
#         # print("Projection F-norm error:", np.linalg.norm(Vt.T @ Vt - Vt_exact[:Vt.shape[0], :].T @ Vt_exact[:Vt.shape[0], :], 'fro'))
#         # print("Trace correlation", np.trace(Vt @ Vt_exact[:Vt.shape[0], :].T @ Vt_exact[:Vt.shape[0], :] @ Vt.T) / min(Vt.T.shape[1], Vt_exact[:Vt.shape[0], :].T.shape[1]))
#     return S, Vt

def normalize_csr_matrix_rows(csr_matrix):
    # Calculate the square root of sum of squares for each row
    row_sums = np.array(csr_matrix.power(2).sum(axis=1)).flatten()
    row_norms = np.sqrt(row_sums)
    
    # Avoid division by zero
    row_norms[row_norms == 0] = 1
    
    # Create a diagonal matrix with the reciprocals of the norms
    row_normalizer = sparse.diags(1 / row_norms)
    
    # Multiply the original matrix by the normalizer
    normalized_matrix = row_normalizer @ csr_matrix
    
    return normalized_matrix, row_norms

def get_matrix_properties(matrix_name):
    # Construct the URL
    url = f"https://sparse.tamu.edu/{matrix_name}"
    
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code != 200:
        return f"Failed to retrieve data. Status code: {response.status_code}"
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    def find_property(property_name):
        element = soup.find(string=lambda text: text and property_name.lower() == text.lower().strip())
        if element:
            grandparent = element.find_parent().find_parent()
            value = grandparent.find(string=lambda text: text and text.strip() and not property_name.lower() in text.lower())
            if value:
                value = value.strip().lower()
                # Try to convert to float if it's a number
                try:
                    return float(value)
                except ValueError:
                    if value == "yes" or value == "no":
                        return value == "yes"
                    else:
                        return value
            return "Unknown"
        return "Not found"

    properties = ["symmetric", "positive definite", "condition number","Minimum Singular Value", 
                  "matrix norm", "type", "kind", "rank"]
    d = {}
    for p in properties:
        d[p] = find_property(p)
    return d

# Function to pad and concatenate arrays
def pad_and_concatenate(arrays, axis=0):
    if not arrays:
        return np.array([])
    
    # Find the maximum size across all arrays for the non-concatenation axis
    if axis == 0:
        max_size = max(arr.shape[1] for arr in arrays)
        pad_axis = 1
    else:
        max_size = max(arr.shape[0] for arr in arrays)
        pad_axis = 0
    
    # Pad each array to the maximum size with NaN (only on the right/bottom)
    padded_arrays = []
    for arr in arrays:
        current_size = arr.shape[pad_axis]
        if current_size < max_size:
            if pad_axis == 1:  # padding columns (right side)
                pad_width = ((0, 0), (0, max_size - current_size))
            else:  # padding rows (bottom)
                pad_width = ((0, max_size - current_size), (0, 0))
            padded_arr = np.pad(arr, pad_width, constant_values=np.nan)
        else:
            padded_arr = arr
        padded_arrays.append(padded_arr)
    
    return np.concatenate(padded_arrays, axis=axis)


import glob

#  "hyperboloid_10000_2.5"
#     "hyperboloid_10000_1.5"
#     "hyperboloid_10000_0.5"
#     "hyperboloid_10000_-0.5"
#     "hyperboloid_10000_-0.25"
#     "hyperboloid_10000_-0.75"
#     "hyperboloid_10000_-1.5"
#     "hyperboloid_10000_-2.5"
# #     "hyperboloid_10000_-5.0"
# #     "hyperboloid_10000_-4.0"
#     "hyperboloid_10000_-3.0"
#     "hyperboloid_10000_-2.0"
#     "hyperboloid_10000_-1.0"
#     "hyperboloid_10000_0.0"
#     "hyperboloid_10000_1.0"
#     "hyperboloid_10000_2.0"
#     "hyperboloid_10000_3.0"

# Plot e_i = trace(S_exact) - trace(S_i) w.r.t iteration
# want to check if e_i = e_\infty + e_1 (1-k/n)?
# Plot e_i / e_1 vs. iteration

import warnings
warnings.filterwarnings("ignore")


# for scaling_factor in [-3.0,-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:

# plot_spectrum = True 
# plot_jer_residual = False #
# plot_trace_error = True # 
# plot_eig_err_heatmap = False
# plot_trace_error_only_log = True
# plot_tr_angles = False
# plot_str_angles_only_log = True
# plot_detailed_iterations = False
# plot_ev_change = True #
# plot_time_elapsed = False 
# plot_regular_residual = True
# plot_reservoir_residual = True
# plot_angles = False
# plot_angles_indi = False #
# plot_entropy = True 
# plot_wholespace_residual = False
# plot_ws_reg = True
# plot_leftout = True
plot_spectrum = False 
plot_jer_residual = False #
plot_trace_error = False # 
plot_eig_err_heatmap = False
plot_trace_error_only_log = False
plot_tr_angles = False
plot_str_angles_only_log = False
plot_detailed_iterations = False
plot_ev_change = False #
plot_time_elapsed = False 
plot_regular_residual = False
plot_reservoir_residual = False
plot_angles = False
plot_angles_indi = False #
plot_entropy = False 
plot_wholespace_residual = False
plot_ws_reg = False
plot_leftout = True

missing_data = []
incomplete_data = []
entropy_d = {}
# for matrix_name_prefix in ["kernel_swissroll", "kernel_torus", "kernel_gaussianmixture"]:
# for matrix_size in [20000]:
from itertools import product

matrix_size_default = 20000

matrix_sizes = [matrix_size_default]

# for matrix_name_prefix in ["olafu"]:
matrix_name_prefixes = ["kernel_stocks"]
# matrix_name_prefixes = ["parabolic_fem", "thermomech_dM", "G2_circuit"]
# matrix_name_prefixes = ["hyperboloid",]
# matrix_name_prefixes = ["kernel_swissroll", "kernel_random"]
# matrix_name_prefixes = ["hyperboloid", "kernel_stocks", "kernel_swissroll",  "kernel_torus"]
# matrix_name_prefixes = ["pdb1HYS"]
# matrix_name_prefixes = ["msc10848", "bcsstk36", "bcsstk17", "crystm02", "olafu", "bodyy4"]
# matrix_name_prefixes = ["crystm02", "olafu", "bodyy4", "pdb1HYS", "Queen_4147", "Flan_1565", "parabolic_fem", "thermomech_dM", "G2_circuit"]
# matrix_name_prefixes = ["Queen_4147", "Flan_1565"]

random_seeds = ["", "_2", "_3"]
# random_seeds = [""]

# for name_postfix in ["", "new"]:
# for name_postfix in ["nystrom_1_isvd_-1", "nystrom_1_isvd_-2", "nystrom_1_isvd", 
#                      "nystrom_5_isvd", "nystrom", "isvd", "nystrom_1_isvd_100", "nystrom_1_isvd_500"]:
# for name_postfix in ["nystrom_1_isvd_-3", "nystrom_1_isvd_-4", "nystrom_1_isvd_-5"]:
# for name_postfix in ["isvd", "nystrom"]:
# for name_postfix in ["nystrom_1_isvdnew"]:
# for name_postfix in ["isvd1by1"]:
# for name_postfix in ["isvd1by1new", "isvd1by1", "isvd"]:
# for name_postfix in ["isvddemix"]:
name_postfixes = ["isvd", "isvdstG"]
# name_postfixes = ["isvd", "isvddemix", "isvdst", "isvdstG", "isvddemixst", "isvddemixstG"]
# name_postfixes = ["isvd", "isvdls", "isvdls2", "isvddemix", "isvddemix2", "isvddemix3"]
# name_postfixes = ["isvddemix2"]

# for noise in [0.0, 0.01, 0.05, 0.1]:
# for noise in [0.001, 0.01, 0.1]:
# for noise in [0.5]:
noises = [0.0]

is_reversed_opts = [False]
use_true_matrix_opts = [False]

# for reservoir_size in [10, 50, 100, 200]:
reservoir_sizes_placeholder = ["default"]
# reservoir_sizes = [100]

# for threshold_factor in [1e1, 1e2, 1e3, 1e4]:
threshold_factors = [1e2]

# for error_kind in ["both", "kernel", "point"]:
error_kinds = ["both"]

# for reservoir_method in ["greedy", "current_window"]:
reservoir_methods = ["greedy"]

plot_S_quotient_opts = [True, False]


for (
    matrix_size,
    matrix_name_prefix,
    random_seed,
    name_postfix,
    noise,
    is_reversed,
    use_true_matrix,
    reservoir_size_flag,
    threshold_factor,
    error_kind,
    reservoir_method,
    plot_S_quotient,
) in product(
    matrix_sizes,
    matrix_name_prefixes,
    random_seeds,
    name_postfixes,
    noises,
    is_reversed_opts,
    use_true_matrix_opts,
    reservoir_sizes_placeholder,
    threshold_factors,
    error_kinds,
    reservoir_methods,
    plot_S_quotient_opts,
):

    # --- defaults depending on matrix ---
    if matrix_name_prefix in ["Queen_4147", "Flan_1565", "parabolic_fem", "thermomech_dM", "G2_circuit"]:
        k_default = 100
        reservoir_size_default = 100
    else:
        k_default = 10
        reservoir_size_default = 10

    matrix_size = matrix_size_default

    default_scaling_factors = [1.0, 10.0]
    if matrix_name_prefix == "kernel_stocks":
        default_scaling_factors = [10.0, 2.2361, 0.7071, 0.2236]

    for scaling_factor in default_scaling_factors:

        if matrix_name_prefix == "kernel_stocks":
            if default_scaling_factors not in [1.0, 10.0]:
                # matrix_size = 10000
                # matrix_size = 100000
                matrix_size = 5000
                k_default = 10
            elif default_scaling_factors in [1.0, 10.0]:
                # matrix_size = 10000
                # matrix_size = 100000
                matrix_size = 5000
                k_default = 10
            reservoir_size_default = k_default

        for k in [k_default]:
        # for k in [10, 20, 50, 100, 200, 400, 600, 800, 1000]:

            if matrix_name_prefix == "pdb1HYS" and scaling_factor != 1.0:
                continue

            if not ("kernel" in matrix_name_prefix or "hyperboloid" in matrix_name_prefix) and scaling_factor != 1.0:
                continue

            ev_change_figures = []
            trace_error_figures = []
            regular_residuals_figures = []
            ws_residual_figures = []
            method_d_trace_error = {}
            method_d_residuals = {}

            if name_postfix:
                name_postfix = "_" + name_postfix

            reservoir_size = reservoir_size_default
            if not "demix" in name_postfix and reservoir_method == "current_window":
                continue
            if noise == 0.0 and error_kind != "both":
                continue
            if not "isvd" in name_postfix and reservoir_size != 10:
                continue
            # if name_postfix == "" and (k != 10 or is_reversed or use_true_matrix):
            #     continue
            
            print(k , random_seed, name_postfix)
            print("Scaling factor:", scaling_factor)
            print("Is reversed:", is_reversed)
            print("use_true_matrix:", use_true_matrix)
            print("reservoir size", reservoir_size)

            kernel_error_only = error_kind == "kernel"
            point_error_only = error_kind == "point"
            plot_S_quotient = plot_S_quotient # trace_error graph, residual
            plot_ws_quotient = plot_S_quotient

            # plot_ev_change = True #
            # plot_trace_error = True #
            # plot_wholespace_residual = True
            # plot_jer_residual = True #
            # plot_time_elapsed = True
            # plot_entropy = True 
            # plot_spectrum = True 
            
            # plot_tr_angles = True
            # plot_detailed_iterations = False
            # plot_angles = True
            # plot_angles_indi = True #

            #postfix = "_quotient"
            postfix = f"_k_{k}"
            if name_postfix == "":
                postfix += f"_rs_{reservoir_size}"
            postfix += f"_factor_{threshold_factor}" if name_postfix == "_new" and threshold_factor != 1e2 else ""
            postfix += f"_reservoir_{reservoir_method}" if reservoir_method != "uniform" else ""
            if "isvddemix" in name_postfix and reservoir_size != 10 and reservoir_method == "greedy":
                postfix += f"{reservoir_size}"
            
            size = 100 if k <= 100 else k #k #100
            stream_size = 1 #size TODO: change to 1 for FD vs iSVD, size for window vs sketch sizes
            if matrix_name_prefix in ["Queen_4147"]:
                size = 414711
                # k = 100
            elif matrix_name_prefix in ["Flan_1565"]:
                size = 156479
                # k = 100
            elif "parabolic_fem" in matrix_name_prefix:
                size = 26291
            elif "circuit" in matrix_name_prefix:
                size = 7505
            elif "thermomech" in matrix_name_prefix:
                size = 10215
            elif matrix_name_prefix in ["kernel_stocks"]:
                size = 100 #5000
            # is_reversed = False
            # matrix_name = "bodyy4"
            # matrix_name = "kronecker_graph_13_0.3"
            if not ("kernel" in matrix_name_prefix or "hyperboloid" in matrix_name_prefix):#in ["msc10848", "bcsstk36", "bcsstk17", "crystm02", "olafu", "bodyy4", "pdb1HYS", "Queen_4147", "Flan_1565"]:
                matrix_name = matrix_name_prefix
            else:
                matrix_name = f"{matrix_name_prefix}_{matrix_size}_{scaling_factor}"
            
            # matrix_name = f"kernel_random_1000_{scaling_factor}"
            # matrix_name = "synthetic_1000_20"
            # import pdb;pdb.set_trace()
            og_matrix_name = matrix_name
            # for pre_postfix in [
            #                     "_low_low",
            #                     "_med_low",
            #                     "_high_low",
            #                     # "_low_med",
            #                     # "_med_med",
            #                     # "_high_med",
            #                     # "_low_high",
            #                     # "_med_high",
            #                     # "_high_high",
                                # ]:
            # for pre_postfix in ["_high", "_low"]:
            # for pre_postfix in ["_low_high"]:
            for pre_postfix in [""]:
                matrix_name = og_matrix_name
                if (kernel_error_only or point_error_only) and noise == 0.0:
                    continue
                if noise > 0.0:
                    if kernel_error_only:
                        matrix_name = matrix_name + f"_{noise}_0.0"
                    elif point_error_only:
                        matrix_name = matrix_name + f"_0.0_{noise}"
                    else:
                        matrix_name += f"_{noise}"
                
                matrix_name = f"{matrix_name}{pre_postfix}{name_postfix}"
                
                print(matrix_name)
                # raise
                figure_dir = 'output'
                # matrix_type2 = '_original'
                matrix_type = f'_random_uniform{random_seed}'
                # matrix_type = "_manual_perm"
                # matrix_type = "_manual_perm_reverse"
                # matrix_type = '_original'
                # matrices_random = [f'_random_uniform_{x}' for x in range(1,6)]
                # matrices_random[0] = '_random_uniform'
                # matrix_type = '_random_uniform_col_perm'
                colors = plt.cm.rainbow(np.linspace(0, 1, 6))
                # matrix_postfix + "_new_" + f"Vapprox_withS_{num_Vs}_" + row_permutation + "_" + f"size_{size}_k_{k}" 
                
                k_vals = np.array([1, 2, 4, 8, 16, 32, 64, 102, 106, 108, 109])
                mem_size = 110
                stream_sizes = mem_size - k_vals
                ks_and_stream_sizes = [(k, stream_size)] # TODO: FD vs iSVD
                # ks_and_stream_sizes = [(k, stream_size) for k, stream_size in zip(k_vals, stream_sizes)] # TODO: stream vs window sizes
                for k, stream_size in ks_and_stream_sizes:
                    postfix = f"_ssize_{stream_size}" + postfix if stream_size != size else postfix
                    matrices = {
                        f'{matrix_name}{matrix_type}{"_true" if use_true_matrix else ""}_size_{size}{postfix}': [colors[0], '-', '.'],
                    #     f'{matrix_name}{matrix_type2}_size_{size}{postfix}': [colors[0], ':', 'o'],
                    #     f'{matrix_name}{matrix_type3}': [colors[0], '-.'],
                    # f'{matrix_name}_decreasing_norm{"_true" if use_true_matrix else ""}_size_{size}{postfix}': [colors[1], '-.', '^'],
                    # f'{matrix_name}_increasing_norm{"_true" if use_true_matrix else ""}_size_{size}{postfix}': [colors[2], '-.', '^'],
                    # f'{matrix_name}_decreasing_V2{"_true" if use_true_matrix else ""}_size_{size}{postfix}': [colors[2], '-.', '^'],
                        # f'{matrix_name}_Vapprox_withS_{num_Vs}{matrix_type}': [colors[2], '-'],
                    #     f'{matrix_name}_Vapprox_{num_Vs}{matrix_type}': [colors[3], '-'],
                    #     f'{matrix_name}_Vapprox_reversed_{num_Vs}{matrix_type}': [colors[4], '-']
                    }
                    # num_Vs_list = [1,10,100]
                    # num_Vs_list = [1,5,10]
                    num_Vs_list = [10]
                    if "_isvd1by1" in name_postfix:
                        num_Vs_list = []
                    # import pdb;pdb.set_trace()
                    Vs_list = np.unique([min(x, k) for x in num_Vs_list])
                    # import pdb;pdb.set_trace()
                    matrices.update({
                    f'{matrix_name}_Vapprox_withS_{num_Vs}{matrix_type}{"_reversed" if is_reversed else ""}{"_true" if use_true_matrix else ""}_size_{size}{postfix}': [colors[3+i], '-', 's'] for i, num_Vs in enumerate(Vs_list)
                    })
                    matrices.update({
                    f'{matrix_name}_Vapprox_{num_Vs}{matrix_type}{"_reversed" if is_reversed else ""}{"_true" if use_true_matrix else ""}_size_{size}{postfix}': [colors[len(matrices)+i], '-', 's'] for i, num_Vs in enumerate(Vs_list)
                    })
                    # matrices.update({
                    #     f'{matrix_name}_Vapprox_withS_{num_Vs}{matrix_type2}_size_{size}{postfix}': [colors[2+i], ':', '*'] for i, num_Vs in enumerate([1,10,100])
                    # })

                    print("Matrices:", matrices.keys())

                    # matrix_name = 'temp'
                    # figure_dir = 'figures'
                    # matrices = [f'{matrix_name}',]
                    # colors = plt.cm.rainbow(np.linspace(0, 1, 5))
                    # matrices = {f'{matrix_name}': [colors[0], '-']}

                    dir_paths = [f"{figure_dir}/{matrix_postfix}/" for matrix_postfix in matrices]
                    # dir_path = dir_paths[1]
                    # print(dir_paths)
                    # raise

                    labels = [' '.join(s.split('_')[1:]) for s in matrices]
                    label_colors = {k:v[0] for k,v in matrices.items()}
                    label_linestyles = {k:v[1] for k,v in matrices.items()}
                    label_markers = {k:v[2] for k,v in matrices.items()}
                    label_colors = label_colors.values()
                    label_linestyles = label_linestyles.values()
                    label_markers = label_markers.values()

                    last_available_file_number = np.inf
                    for dir_path in dir_paths:
                        # Use glob to find files matching the pattern
                        files = glob.glob(os.path.join(dir_path, f'spectrum_data_*.npz'))

                        # Extract the numeric part from each file and convert to an integer
                        file_numbers = sorted([int(os.path.splitext(file)[0].split('_')[-1]) for file in files])
                    #     print(dir_path, files)
                        if len(file_numbers) == 0:
                            print("Path", dir_path, "not available")
                            continue
                        last_consecutive = -1
                        current = 0
                        
                        while current in file_numbers:
                            last_consecutive = current
                            current += 1

                        last_file_number = last_consecutive#file_numbers[-1]
                        if last_file_number < last_available_file_number:
                            last_available_file_number = last_file_number

                    if last_available_file_number == np.inf:
                        print("No available file found for", dir_paths[0])
                        continue

                    # Load data first
                    data_list = []
                    smallest_ei = np.inf
                    
                    is_incomplete = False
                    is_missing = False
                    for dir_path, label in zip(dir_paths, labels):
                        tr_S = []
                        tr_S_quotient = []
                        Ss = []
                        Ss_quotient = []
                        err_mat = []
                        
                        for iteration in range(last_available_file_number+1):
                            file_path = os.path.join(dir_path, f'spectrum_data_{iteration}.npz')
                            try:
                                data = np.load(file_path, allow_pickle=True)
                                Ss.append(data['S'].reshape(1,-1))
                                trace = np.sum(data['S'])
                                tr_S.append(trace)
                                # import pdb;pdb.set_trace()
                                if not data['S_quotient'] is None:
                                    Ss_quotient.append(data['S_quotient'].reshape(1,-1))
                                    trace_quotient = np.sum(data['S_quotient'])
                                    tr_S_quotient.append(trace_quotient)
                                # print(data['S'], data['S_exact'][:len(data['S'])])
                    #             raise
                            except FileNotFoundError:
                                is_incomplete = True
                                print(f"File not found: {file_path}")
                        
                        if len(Ss) == 0:
                            is_missing = True
                            missing_data.append(dir_path)
                            continue
                        elif is_incomplete:
                            incomplete_data.append(dir_path)

                        limit_S = min(10, len(data['S']))
                        # limit_S = len(data['S'])
                        # e_i = np.abs(data['S_exact'][:len(data['S'])].sum() - tr_S) / data['S_exact'][:len(data['S'])].sum()
                        Ss = np.concatenate(Ss, axis=0)
                        tr_S = np.sum(Ss[:, :limit_S], axis=1)
                        if not data['S_quotient'] is None: #name_postfix == "" or name_postfix ==:
                            # Ss_quotient = np.concatenate(Ss_quotient, axis=0)
                            Ss_quotient = pad_and_concatenate(Ss_quotient, axis=0)
                            tr_S_quotient = np.sum(Ss_quotient[:, :limit_S], axis=1)
                            if plot_S_quotient:
                                tr_S = tr_S_quotient
                        try:
                            e_i = np.abs(data['S_exact'][:limit_S].sum() - tr_S) / data['S_exact'][:limit_S].sum()
                        except:
                            import pdb;pdb.set_trace()
                        e_i = np.clip(e_i, np.finfo(float).eps, None)
                        exact_temp = data['S_exact'][:len(data['S'])]
                        # print(Ss[-1])
                        
                        
                        data_list.append((e_i, label))
                        if smallest_ei > min(e_i):
                            smallest_ei = min(e_i)

                        if plot_eig_err_heatmap:
                            if plot_S_quotient:
                                Ss = Ss_quotient
                            err_mat = np.abs(data['S_exact'][:len(data['S'])].reshape(-1,1) - Ss)/data['S_exact'][:len(data['S'])]
                            err_mat = err_mat.real
                            # import pdb;pdb.set_trace()
                            # err_mat = np.concatenate(err_mat, axis=1).real
                            # import pdb;pdb.set_trace()
                            fig, ax = plt.subplots(figsize=(12, 8))
                            sns.heatmap(np.log10(err_mat), 
                #                 cmap='viridis',  # Color scheme
                                cbar_kws={'label': 'Value'},  # Colorbar label
                                # xticklabels=False,  # Hide x-axis labels for cleaner look
                                # yticklabels=False,
                                ax=ax)
                            
                            plt.savefig(f"figures/{matrix_name}_heatmap_error_over_time_{'_'.join(label.split(' '))}.png")
                            plt.close()

                        if plot_ev_change:
                            if plot_S_quotient:
                                Ss = Ss_quotient
                            num_evs = 10
                            try:
                                err_mat = np.abs(data['S_exact'][:num_evs].reshape(1,-1) - Ss[:,:num_evs])/data['S_exact'][:num_evs]
                            except:
                                import pdb;pdb.set_trace()
                            err_mat = err_mat.real
                            # import pdb;pdb.set_trace()
                            # err_mat = np.concatenate(err_mat, axis=1).real
                            # import pdb;pdb.set_trace()
                            fig, ax = plt.subplots(figsize=(12, 8))
                            num_sv = err_mat.shape[1]
                            color_range = np.linspace(0, 1.0, num_sv)
                            color_range[2] = (color_range[-1] + color_range[-2]) / 2
                            color_range = np.sort(color_range)
                            colors = plt.cm.jet(color_range)
                            # Plot each window's data
                            for i in range(num_sv):
                                plt.semilogy(np.arange(err_mat.shape[0]), err_mat[:,i], color=colors[i], label=f'Eigenvalue #{i}', marker='o')
                            #     plt.scatter(np.arange(err_mat.shape[0]), err_mat[:,i], color=colors[i], label=f'Eigenvalue #{i}')
                            # plt.yscale('log')
                            plt.legend()
                            # if scaling_factor == 2.0:
                            #     plt.ylim(1e-7, 1e2)
                            # elif scaling_factor == 1.0:
                            #     plt.ylim(1e-7, 10**(0.5))
                            # elif scaling_factor == 5.0:
                            #     plt.ylim(1e-9,1e1)
                            # elif scaling_factor == 10.0:
                            #     plt.ylim(1e-11,1e3)
                            # elif scaling_factor == 20.0:
                            #     plt.ylim(10**(-14.5), 1e2)
                            plt.ylabel("Relative eigenvalue difference")
                            plt.xlabel("Window")
                            plt.xticks(fontsize=15)
                            plt.yticks(fontsize=15)
                            plt.title(f"{label}_{'quotient' if plot_S_quotient else ''}")
                            plt.grid()
                            filename = f"figures/{matrix_name}_{'quotient_' if plot_S_quotient else ''}sv_error_over_time_{'_'.join(label.split(' '))}.png"
                            # plt.savefig()
                            current_fig = plt.gcf()
                            ev_change_figures.append([current_fig, filename])
                            plt.close()

                    if is_missing:
                        continue

                    if plot_spectrum:
                        plt.figure(figsize=(12, 6))
                        # plt.plot(np.arange(data["S_exact"].shape[0]), np.log10(data["S_exact"]), label=matrix_name)
                        plt.semilogy(np.arange(data["S_exact"].shape[0]), data["S_exact"], label=matrix_name)
                        plt.ylabel("Eigenvalue", fontsize=16)
                        plt.xlabel("Index", fontsize=16)
                        plt.title(f"Spectrum")
                        plt.legend()
                        plt.xticks(fontsize=15)
                        plt.yticks(fontsize=15)
                        plt.grid(True, which='both', linestyle='--', alpha=0.5)
                        plt.tight_layout()
                        plt.savefig(f"figures/{matrix_name}_spectrum.png")
                        plt.close()

                        # if scaling_factor == 2.0:
                        #     viz_rank = 70
                        # elif scaling_factor == 1.0:
                        #     viz_rank = 300
                        # else:
                        #     viz_rank = None

                        viz_rank = 50

                        if viz_rank is not None:
                            plt.figure(figsize=(12, 6))
                            # plt.plot(np.arange(data["S_exact"].shape[0]), np.log10(data["S_exact"]), label=matrix_name)
                            plt.semilogy(np.arange(viz_rank), data["S_exact"][:viz_rank], label=matrix_name)
                            plt.ylabel("Eigenvalue")
                            plt.xlabel("Index")
                            plt.title(f"Spectrum")
                            plt.legend()
                            plt.grid(True, which='both', linestyle='--', alpha=0.5)
                            plt.xticks(fontsize=15)
                            plt.yticks(fontsize=15)
                            plt.tight_layout()
                            plt.savefig(f"figures/{matrix_name}_spectrum_zoomed.png")
                            plt.close()
                    # matrix_size = len(data['S_exact'])
                    
                    # plt.semilogy(data['S_exact'][:100])
                    # plt.title(f"Lengthscale  {scaling_factor}")
                    # plt.ylabel("Eigenvalue")
                    # plt.xlabel("Index")
                    # plt.savefig(f"figures/{matrix_name}_{scaling_factor}_spectrum.png")
                    
                    # plt.close()
                    # continue

                    if plot_trace_error:
                        if not plot_trace_error_only_log:
                            plt.figure(figsize=(12, 6))
                            i = 0
                            for (e_i, label), color, linestyle, marker in zip(data_list, label_colors, label_linestyles, label_markers):
                                i += 1
                                if e_i[0] == 0:
                                    print("Initial Error is already 0!")
                                    break
                                # import pdb;pdb.set_trace()
                                # plt.plot(np.arange(last_available_file_number+1), ((e_i)/e_i[0]), label=f'{label}, init err: {e_i[0]}', linestyle=linestyle, marker=marker,
                                #         color=color, alpha=0.7, markevery=i, markersize=12)
                                plt.plot(np.arange(last_available_file_number+1), ((e_i)), label=f'{label}, init err: {e_i[0]}', linestyle=linestyle, marker=marker,
                                        color=color, alpha=0.7, markevery=i, markersize=12)

                            # plt.plot(np.arange(last_available_file_number+1), (1-np.arange(last_available_file_number+1)/(last_available_file_number+1)),
                            #         label='1-k/n', linestyle='--', alpha=0.7)

                        
                            plt.ylabel("$e_k\ /\ e_0$", fontsize=16)
                            plt.xlabel("Iteration", fontsize=16)
                            plt.title(f"Error $e_k\ /\ e_0$ over Iterations, Length Scale: 1e{(scaling_factor):.1f}")
                            plt.legend()
                            plt.grid(True, which='both', linestyle='--', alpha=0.5)
                            plt.tight_layout()
                            plt.xticks(fontsize=15)
                            plt.yticks(fontsize=15)
                            # plt.ylim([-0.01, 1.01])
                            #plt.show()
                            plt.savefig(f"figures/{matrix_name}_{'quotient_' if plot_S_quotient else ''}error_over_time_{'_'.join(labels[-1].split(' '))}.png")
                            plt.close()
                            

                        plt.figure(figsize=(12, 6))
                        i = 0
                        for (e_i, label), color, linestyle, marker in zip(data_list, label_colors, label_linestyles, label_markers):
                            i += 1
                            if e_i[0] == 0:
                                print("Initial Error is already 0!")
                                break
                            if "Vapprox" not in label:
                                method_d_trace_error[name_postfix+"_quotient" if plot_S_quotient else name_postfix] = e_i
                            # plt.plot(np.arange(last_available_file_number+1), np.log10(np.abs((e_i)/e_i[0])), label=f'{label}', linestyle=linestyle, marker=marker,
                            #         color=color, alpha=0.7, markevery=i, markersize=12)
                            plt.semilogy(np.arange(last_available_file_number+1), np.abs((e_i)), label=f'{label}', linestyle=linestyle, marker=marker,
                                    color=color, alpha=0.7, markevery=i, markersize=12)
                            # import pdb;pdb.set_trace()
                        # plt.plot(np.arange(last_available_file_number+1), np.log10(1-np.arange(last_available_file_number)/last_available_file_number),
                        #          label='1-k/n', linestyle='--', alpha=0.7)
                        # if scaling_factor == 2.0:
                        #     plt.ylim(-7,1)
                        # elif scaling_factor == 5.0:
                        #     plt.ylim(-7,1)
                        # elif scaling_factor == 10.0:
                        #     plt.ylim(-6,0)
                        # elif scaling_factor == 20.0:
                        #     plt.ylim(-8,0)
                        plt.ylabel("e_k", fontsize=16)
                        plt.xlabel("Iteration", fontsize=16)
                        plt.title(f"Relative Eigenvalue Error over Iterations, Length Scale: {(scaling_factor):.1f}")
                        plt.legend()
                        plt.xticks(fontsize=15)
                        plt.yticks(fontsize=15)
                        plt.grid(True, which='both', linestyle='--', alpha=0.5)
                        plt.tight_layout()
                        #plt.show()
                        # plt.savefig(f"figures/{matrix_name}_{'quotient_' if plot_S_quotient else ''}error_over_time_log_{'_'.join(labels[-1].split(' '))}.png")
                        current_figure = plt.gcf()
                        filename = f"figures/{matrix_name}_{'quotient_' if plot_S_quotient else ''}error_over_time_log_{'_'.join(labels[-1].split(' '))}.png"
                        trace_error_figures.append([current_figure, filename])
                        plt.close()

                    import glob

                    S_exact = None

                    def plot_multiple_graphs_single_iteration(dir_paths, iteration, labels, is_sym_psd=False, save_path=None,
                                                            colors=None, linestyles=None, with_S_exact=True, rel_A_norm=False,
                                                            with_angles=False, markers=None):
                        # Create a figure with 4 subplots in a row
                        num_plots = 4 if with_angles else 3
                        fig, axs = plt.subplots(1, num_plots, figsize=(36, 8))  # 4 * (12, 8) for width    

                        # 1. Spectrum comparison plot
                        load_and_plot_spectrum_comparison(dir_paths, iteration, labels, colors=colors, linestyles=linestyles, markers=markers,
                                                        with_S_exact=with_S_exact, rel_A_norm=False, axs=[axs[0], axs[1]])
                        
                        # 2. Residuals plot
                        load_and_plot_same_iteration_residuals(dir_paths, iteration, labels, is_sym_psd, colors=colors, 
                                                            linestyles=linestyles, markers=markers, ax=axs[2])
                        
                        
                        
                        # # 3. Relative difference w.r.t A_norm plot
                        # if rel_A_norm:
                        #     load_and_plot_spectrum_comparison(dir_paths, iteration, labels, colors=colors, linestyles=linestyles, 
                        #                                       with_S_exact=with_S_exact, rel_A_norm=True, ax=axs[3])
                        # else:
                        #     axs[3].axis('off')  # Turn off the axis if rel_A_norm is False
                        
                        # 4. Canonical angles plot
                        if with_angles:
                            load_and_plot_multiple_canonical_angles(dir_paths, iteration, labels, ax=axs[3])
                        
                        plt.tight_layout()
                        
                        if save_path:
                            plt.savefig(save_path, bbox_inches='tight', dpi=300)
                        
                        #plt.show()

                        plt.savefig(f"figures/{matrix_name}_it_{iteration}_{'_'.join(labels[-1].split(' '))}.png")
                        plt.close()

                    # Modified functions to work with the new combined plot

                    def load_and_plot_same_iteration_residuals(dir_paths, iteration, labels, is_sym_psd=False, save_path=None,
                                                            colors=None, linestyles=None, ax=None, markers=None):
                        if ax is None:
                            fig, ax = plt.subplots(figsize=(12, 8))
                        
                        colors = plt.cm.rainbow(np.linspace(0, 1, len(dir_paths))) if colors is None else colors
                        linestyles = ['-' for _ in colors] if linestyles is None else linestyles

                        i = 0
                        for dir_path, label, color, linestyle, marker in zip(dir_paths, labels, colors, linestyles, markers):
                            i += 1
                            if is_sym_psd:
                                file_path = os.path.join(dir_path, f'residuals_sym_psd_data_{iteration}.npz')
                    #             file_path = os.path.join(dir_path, f'residuals_sym_psd_data_truncated_{iteration}.npz')
                    #             file_path = os.path.join(dir_path, f'residuals_sym_psd_data_truncated_Rayleigh_{iteration}.npz')
                            else:
                                file_path = os.path.join(dir_path, f'residuals_data_{iteration}.npz')
                    #         print(file_path); raise
                            try:
                                data = np.load(file_path)
                                approx_residuals = data['approx_residuals']
                                ax.semilogy(approx_residuals, label=f'{label}', color=color, linestyle=linestyle, marker=marker, alpha=0.7, markevery=i, markersize=12)
                            except FileNotFoundError:
                                print(f"File not found: {file_path}")

                        if is_sym_psd:
                            ax.set_ylabel('Residual Norm (sym pd)')
                            ax.set_title(f'Residuals Comparison - Iteration {iteration}')
                        else:
                            ax.set_ylabel('Residual Norm (not sym pd)')
                            ax.set_title(f'Residuals Comparison - Iteration {iteration}')

                        ax.set_xlabel('Index')
                        ax.legend()
                        ax.grid(True)

                    def load_and_plot_spectrum_comparison(dir_paths, iteration, labels, save_dir=None,
                                                        colors=None, linestyles=None, with_S_exact=True, 
                                                        rel_A_norm=False, axs=None, markers=None):
                        global S_exact
                        if axs is None:
                            raise
                            fig, ax = plt.subplots(figsize=(12, 8))

                        ax = axs[0]
                        colors = plt.cm.rainbow(np.linspace(0, 1, len(dir_paths))) if colors is None else colors
                        linestyles = ['-' for _ in colors] if linestyles is None else linestyles
                        
                        # Load exact spectrum from the first directory
                        exact_file_path = os.path.join(dir_paths[0], f'spectrum_data_{iteration}.npz')
                        exact_data = np.load(exact_file_path)

                        if with_S_exact:
                            S_exact = exact_data['S_exact']
                        else:
                            exact_file_path = os.path.join(dir_paths[-1], f'spectrum_data_{2000}.npz')
                            exact_data = np.load(exact_file_path)
                            S_exact = exact_data['S']
                        
                        i = 0
                        for dir_path, label, color, linestyle, marker in zip(dir_paths, labels, colors, linestyles, markers):
                            i += 1
                            file_path = os.path.join(dir_path, f'spectrum_data_{iteration}.npz')
                            
                            try:
                                data = np.load(file_path)
                                S = data['S']
                                if rel_A_norm:
                                    file_path = os.path.join(dir_path, f'diffspec_relA_data_{iteration}.npz')
                                    data = np.load(file_path)            
                                    rel_diff = data['diff']
                                    ax.semilogy(rel_diff, label=f'{label}', color=color, linestyle=linestyle, marker=marker, alpha=0.7, markevery=i, markersize=12)
                                else:
                                    ax.semilogy(S, label=f'{label}', color=color, linestyle=linestyle, marker=marker, alpha=0.7, markevery=i, markersize=12)
                            except FileNotFoundError:
                                print(f"File not found: {file_path}")

                        if not S_exact is None and not rel_A_norm:
                            ax.semilogy(S_exact[:len(S)], label='Exact', color='black', linestyle='--')

                        ax.set_xlabel('Index')
                        # else:
                        ax.set_ylabel('Singular Value')
                        ax.set_title(f'Spectrum Comparison - Iteration {iteration}')
                        ax.legend()
                        ax.grid(True)

                        # Plot relative difference to S_exact
                        i = 0
                        for dir_path, label, color, linestyle, marker in zip(dir_paths, labels, colors, linestyles, markers):
                            # file_path = os.path.join(dir_path, f'spectrum_data_{iteration}.npz')
                            i += 1
                            file_path = os.path.join(dir_path, f'diffspec_relS_data_{iteration}.npz')
                    #         print(file_path); raise
                            try:
                                if with_S_exact:
                                    data = np.load(file_path)
                                    rel_diff = data['diff']
                                    axs[1].semilogy(rel_diff, label=label, color=color, linestyle=linestyle, marker=marker, alpha=0.7,
                                                    markevery=i, markersize=12)
                                    
                                    if iteration == 7:
                                        file_path = os.path.join(dir_path, f'spectrum_data_{iteration}.npz')
                                        data = np.load(file_path)
                                        S = data['S']
                                        rel_diff_2 = np.abs(S-S_exact[:len(S)]) / S_exact[:len(S)]
                                        # import pdb;pdb.set_trace()
                                else:
                                    file_path = os.path.join(dir_path, f'spectrum_data_{iteration}.npz')
                                    data = np.load(file_path)
                                    S = data['S']
                                    rel_diff = np.abs(S-S_exact) / S_exact
                                    axs[1].semilogy(rel_diff, label=label, color=color, linestyle=linestyle, marker=marker, alpha=0.7,
                                                    markevery=i, markersize=12)
                    #             if "exact" in label or "random" in label:
                    #                 plt.semilogy(rel_diff, label=label, color=color, linestyle='--', alpha=0.7)
                    #             else:
                    #                 plt.semilogy(rel_diff, label=label, color=color, alpha=0.7)
                            except FileNotFoundError:
                                print(f"File not found: {file_path}")

                        axs[1].set_xlabel('Index')
                        axs[1].set_ylabel('Relative Difference')
                        axs[1].set_title(f'Relative Difference w.r.t S_exact - Iteration {iteration}')
                        axs[1].legend()
                        axs[1].grid(True)

                    def load_and_plot_multiple_canonical_angles(dir_paths, iteration, labels, save_path=None, additional_labels="", ax=None):
                        if ax is None:
                            fig, ax = plt.subplots(figsize=(12, 8))
                        
                        num_experiments = len(dir_paths)
                        positions = np.arange(1, num_experiments + 1)
                        width = 0.2
                        
                        all_data = []
                        all_bp = []
                        
                        for i, (dir_path, label) in enumerate(zip(dir_paths, labels)):
                            file_path = os.path.join(dir_path, f'canonical_angles{additional_labels}_data_{iteration}.npz')
                            
                            try:
                                data = np.load(file_path)
                                s = data['s']
                                C = np.log10(np.clip(np.abs(data['C']), 1e-32,1)-np.eye(len(s)))
                    #             C = np.log10(np.clip(1-np.abs(data['C']), 1e-32,1))
                                sns.heatmap(C, 
                    #                 cmap='viridis',  # Color scheme
                                    cbar_kws={'label': 'Value'},  # Colorbar label
                                    xticklabels=False,  # Hide x-axis labels for cleaner look
                                    yticklabels=False,
                                    ax=ax,
                                    vmin=-12)
                    #             print((1-np.abs(data['C']))[0,0])
                                ax.set_title('$\log(|V_{approx}^T\ x\ V| - I)$ (element-wise)')
                    #             plt.imshow(C)
                    #             plt.show()
                                return
                                all_data.append(s)
                                
                                bp = ax.boxplot(s, positions=[positions[i]], widths=width, patch_artist=True)
                                all_bp.append(bp)
                                
                                # Add annotations (simplified for space)
                                x_pos = positions[i]
                                min_val, max_val = np.min(s), np.max(s)
                                q1, median_val, q3 = np.percentile(s, [25, 50, 75])
                                ax.annotate(f'Med: {1-np.exp(-median_val):.4f}', (x_pos, median_val), xytext=(5, 0), 
                                            textcoords='offset points', ha='left', va='center', fontsize=8)
                                
                            except FileNotFoundError:
                                print(f"File not found: {file_path}")
                        
                        ax.set_title(f'Canonical Angles Comparison - Iteration {iteration}')
                        ax.set_ylabel('Values')
                        ax.set_ylim(0, -np.log10(1e-4)+0.1)
                        
                        y_ticks = np.array([0, 0.5, 0.9, 0.99, 0.999, 0.9999, 1])
                        ax.set_yticks(-np.log10(1 - y_ticks + 1e-4))
                        ax.set_yticklabels([f"{y}" for y in y_ticks])
                        
                        ax.set_xticks(positions)
                        ax.set_xticklabels(labels, rotation=45, ha='right')
                        
                        ax.grid(axis='y')
                        #plt.show()

                        # plt.savefig(f"figures/it_{iteration}.png")
                        # plt.close()
                        
                        # plt.plot(np.arange(len(s)), s)
                        # plt.show()

                    # dir_paths, iteration, labels, is_sym_psd=False, save_path=None,
                    # colors=None, linestyles=None, with_S_exact=True, rel_A_norm=False
                    # for i in range(10):
                        
                    # import pdb;pdb.set_trace()
                    start = 3
                    end = last_available_file_number
                    n_steps = 3

                    # Using linspace to include both start and end with n_steps intervals
                    the_rest = np.linspace(start, end, n_steps+1, dtype=int)[1:] 
                    the_rest = np.minimum(np.ones(the_rest.shape)*end, the_rest)
                    start = min(start, end)

                    iterations = np.concatenate([np.array(range(start)), np.minimum(np.ones(the_rest.shape)*end, the_rest)])
                    iterations = np.unique(iterations)
                    # for i in [20,40,80,320,1280,2000]:
                    # for i in range(10):
                    # for i in the_rest:
                    # if end == 1:
                    #     import pdb;pdb.set_trace()
                    if plot_detailed_iterations:
                        print("Generating per iteration plots")
                        for i in iterations:
                            print(i, start, end, the_rest)
                            i = int(i)
                            plot_multiple_graphs_single_iteration(
                                dir_paths, 
                                i,  
                                labels, 
                                is_sym_psd=True, 
                                colors=label_colors,
                                linestyles=label_linestyles,
                                markers=label_markers,
                                with_S_exact=True,
                                with_angles=True,
                            )

                    # import pdb;pdb.set_trace()
                    reservoir_size = int(reservoir_size)
                    if (name_postfix == "_isvd" or "_isvd1by1" in name_postfix or "demix" in name_postfix or "isvdst" in name_postfix) and plot_jer_residual:
                        for dir_path, label in zip(dir_paths, labels):
                            reservoir_residuals = []
                            regular_residuals = []
                            reservoir_residuals_quotient = []
                            regular_residuals_quotient = []
                            num_ev = 10
                            fig, ax = plt.subplots(figsize=(8, 6))
                            for iteration in range(last_available_file_number+1):
                                # Load data
                                file_path = os.path.join(dir_path, f'reservoir_residuals_data_{iteration}.npz')
                                try:
                                    data = np.load(file_path)
                                    res_residuals = data['reservoir_residuals'].reshape(1,-1)[:, :num_ev]
                                    res_residuals *= np.sqrt(matrix_size / reservoir_size)
                                    reg_residuals = data['regular_residuals'].reshape(1,-1)[:, :num_ev]
                                    reservoir_residuals.append(res_residuals)
                                    regular_residuals.append(reg_residuals)
                                    res_residuals_quotient = data['reservoir_residuals_quotient'].reshape(1,-1)[:, :num_ev]
                                    res_residuals_quotient *= np.sqrt(matrix_size / reservoir_size)
                                    reg_residuals_quotient = data['regular_residuals_quotient'].reshape(1,-1)[:, :num_ev]
                                    reservoir_residuals_quotient.append(res_residuals_quotient)
                                    regular_residuals_quotient.append(reg_residuals_quotient)

                                    # trace = np.sum(data['S'])
                                    # tr_S.append(trace)
                                    # ax.semilogy(approx_residuals, label=f'{label}', color=color, linestyle=linestyle, marker=marker, alpha=0.7, markevery=i, markersize=12)
                                except FileNotFoundError:
                                    print(f"File not found: {file_path}")
                            
                            reservoir_residuals = pad_and_concatenate(reservoir_residuals, axis=0)
                            regular_residuals = pad_and_concatenate(regular_residuals, axis=0)
                            reservoir_residuals_quotient = pad_and_concatenate(reservoir_residuals_quotient, axis=0)
                            regular_residuals_quotient = pad_and_concatenate(regular_residuals_quotient, axis=0)

                            # reservoir_residuals = np.concatenate(reservoir_residuals, axis=0)
                            # regular_residuals = np.concatenate(regular_residuals, axis=0)
                            # reservoir_residuals_quotient = np.concatenate(reservoir_residuals_quotient, axis=0)
                            # regular_residuals_quotient = np.concatenate(regular_residuals_quotient, axis=0)
                            # regular_residuals = np.concatenate(regular_residuals, axis=0)
                            # print(regular_residuals_quotient)
                            # import pdb;pdb.set_trace()
                            # e_i = np.abs(data['S_exact'][:len(data['S'])].sum() - data['S'].sum()) / data['S_exact'][:len(data['S'])].sum()                                                                    
                            # cmap = plt.cm.plasma # viridis
                            color_range = np.linspace(0, 1.0, reservoir_residuals.shape[1])
                            color_range[2] = (color_range[-1] + color_range[-2]) / 2
                            color_range = np.sort(color_range)
                            colors = plt.cm.jet(color_range)
                            if not plot_S_quotient:
                                if plot_reservoir_residual:
                                    fig, ax = plt.subplots(figsize=(12, 8))
                                    
                                    for i in range(reservoir_residuals.shape[1]):
                                        ax.semilogy(range(reservoir_residuals.shape[0]), reservoir_residuals[:,i], color=colors[i], marker='o')
                                        # ax.scatter(range(reservoir_residuals.shape[0]), reservoir_residuals[:,i], color=colors[i])
                                    ax.set_yscale('log')
                                    ax.grid(True)
                                    ax.set_xlabel('window index', fontsize=16)
                                    ax.set_ylabel('residual', fontsize=16)
                                    # ax.set_ylim(1e-2, 1e2)
                                    
                                    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, reservoir_residuals.shape[1]))
                                    # cbar = fig.colorbar(sm, ax=ax, label='pair index', ticks=range(1, reservoir_residuals.shape[1]+1))
                                    # cbar.ax.set_yticklabels(range(1, reservoir_residuals.shape[1]+1))
                                    # plt.rc(('xtick.major', 'ytick.major'), width=2.5, size=20)
                                    plt.xticks(fontsize=15)
                                    plt.yticks(fontsize=15)
                                    plt.tight_layout()
                                    plt.legend()
                                    plt.title(f"reservoir_residuals_{'_'.join(label.split(' '))}")
                                    plt.savefig(f"figures/{matrix_name}_reservoir_residuals_{'_'.join(label.split(' '))}.png")
                                    plt.close()
                                    # import pdb;pdb.set_trace()
                                
                                if plot_regular_residual:
                                    fig, ax = plt.subplots(figsize=(12, 8))
                                    for i in range(regular_residuals.shape[1]):
                                        ax.semilogy(range(regular_residuals.shape[0]), regular_residuals[:,i], color=colors[i], marker='o')
                                        # ax.scatter(range(regular_residuals.shape[0]), regular_residuals[:,i], color=colors[i])
                                    ax.set_yscale('log')
                                    ax.grid(True)
                                    ax.set_xlabel('window index', fontsize=16)
                                    ax.set_ylabel('residual', fontsize=16)
                                    # ax.set_ylim(1e-2, 1e2)
                                    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, regular_residuals.shape[1]))
                                    # cbar = fig.colorbar(sm, ax=ax, label='pair index', ticks=range(1, regular_residuals.shape[1]+1))
                                    # cbar.ax.set_yticklabels(range(1, regular_residuals.shape[1]+1))
                                    plt.xticks(fontsize=15)
                                    plt.yticks(fontsize=15)
                                    plt.tight_layout()
                                    plt.legend()
                                    plt.title(f"regular_residuals_{'_'.join(label.split(' '))}")
                                    # plt.savefig(f"figures/{matrix_name}_regular_residuals_{'_'.join(label.split(' '))}.png")
                                    current_figure = plt.gcf()
                                    filename = f"figures/{matrix_name}_regular_residuals_{'_'.join(label.split(' '))}.png"
                                    regular_residuals_figures.append([current_figure, filename])
                                    plt.close()


                            # ==== Quotient ====
                            if plot_S_quotient:
                                if plot_reservoir_residual:
                                    fig, ax = plt.subplots(figsize=(12, 8))
                                    for i in range(reservoir_residuals_quotient.shape[1]):
                                        ax.semilogy(range(reservoir_residuals_quotient.shape[0]), reservoir_residuals_quotient[:,i], color=colors[i], marker='o')
                                        # ax.scatter(range(reservoir_residuals_quotient.shape[0]), reservoir_residuals_quotient[:,i], color=colors[i])
                                    ax.set_yscale('log')
                                    ax.grid(True)
                                    ax.set_xlabel('window index', fontsize=16)
                                    ax.set_ylabel('residual', fontsize=16)
                                    # ax.set_ylim(1e-2, 1e2)
                                    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, reservoir_residuals_quotient.shape[1]))
                                    # cbar = fig.colorbar(sm, ax=ax, label='pair index', ticks=range(1, reservoir_residuals_quotient.shape[1]+1))
                                    # cbar.ax.set_yticklabels(range(1, reservoir_residuals_quotient.shape[1]+1))
                                    plt.xticks(fontsize=15)
                                    plt.yticks(fontsize=15)
                                    plt.tight_layout()
                                    plt.legend()
                                    plt.title(f"reservoir_residuals_quotient_{'_'.join(label.split(' '))}")
                                    plt.savefig(f"figures/{matrix_name}_reservoir_residuals_quotient_{'_'.join(label.split(' '))}.png")
                                    plt.close()

                                if plot_regular_residual:
                                    fig, ax = plt.subplots(figsize=(12, 8))
                                    for i in range(regular_residuals_quotient.shape[1]):
                                        ax.semilogy(range(regular_residuals_quotient.shape[0]), regular_residuals_quotient[:,i], color=colors[i], marker='o')
                                        # ax.scatter(range(regular_residuals_quotient.shape[0]), regular_residuals_quotient[:,i], color=colors[i])
                                    # ax.set_yscale('log')
                                    ax.grid(True)
                                    ax.set_xlabel('window index', fontsize=16)
                                    ax.set_ylabel('residual', fontsize=16)
                                    # if scaling_factor == 2.0:
                                    #     ax.set_ylim(1e-13, 1e4)
                                    # elif scaling_factor == 0.0:
                                    #     ax.set_ylim(1e-4, 1e2)
                                    # if scaling_factor == 2.0:
                                    #     ax.set_ylim(1e-2, 1e4)
                                    # elif scaling_factor == 5.0:
                                    #     ax.set_ylim(1e-4,1e6)
                                    # elif scaling_factor == 10.0:
                                    #     ax.set_ylim(1e-6,10**(4.5))
                                    # elif scaling_factor == 20.0:
                                    #     ax.set_ylim(10**(-9), 10**(4.5))
                                    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, regular_residuals_quotient.shape[1]))
                                    # cbar = fig.colorbar(sm, ax=ax, label='pair index', ticks=range(1, regular_residuals_quotient.shape[1]+1))
                                    # cbar.ax.set_yticklabels(range(1, regular_residuals_quotient.shape[1]+1))
                                    plt.xticks(fontsize=15)
                                    plt.yticks(fontsize=15)
                                    plt.tight_layout()
                                    plt.legend()
                                    plt.title(f"regular_residuals_quotient_{'_'.join(label.split(' '))}")
                                    # plt.savefig()
                                    current_figure = plt.gcf()
                                    filename = f"figures/{matrix_name}_regular_residuals_quotient_{'_'.join(label.split(' '))}.png"
                                    regular_residuals_figures.append([current_figure, filename])
                                    plt.close()
                        
                    if plot_leftout:
                        data_list = []
                        for dir_path, label in zip(dir_paths, labels):
                            current_totals = []
                            current_throws = []
                            current_throw = 0
                            for iteration in range(last_available_file_number+1):
                                file_path = os.path.join(dir_path, f'leftout_data_{iteration}.npz')
                                try:
                                    data = np.load(file_path, allow_pickle=True)
                                    current_total = data['current_total']
                                    throw = data['throw']
                                    current_throw += throw
                                    iter_num = data['iteration']
                                    current_totals.append(current_total)
                                    current_throws.append(current_throw)
                                    # print(f"Iteration {iter_num}: current_total={current_total}, throw={throw}, label={label}")
                                except FileNotFoundError:
                                    print(f"File not found: {file_path}")
                            data_list.append([current_totals, current_throws, label]) 
                        
                        combined_log_figures = []

                        for current_totals, current_throws, label in data_list:
                            current_totals = np.asarray(current_totals)
                            current_throws = np.asarray(current_throws)

                            fig, ax = plt.subplots(figsize=(12, 8))

                            num_sv = current_totals.shape[1]
                            windows = np.arange(current_totals.shape[0])

                            color_range = np.linspace(0, 1.0, num_sv)
                            if num_sv > 2:
                                color_range[2] = (color_range[-1] + color_range[-2]) / 2
                            color_range = np.sort(color_range)
                            colors = plt.cm.jet(color_range)

                            for i in range(num_sv):
                                ax.semilogy(
                                    windows,
                                    np.abs(current_totals[:, i]),
                                    color=colors[i],
                                    linestyle='-',
                                    marker='o',
                                    label=f'Total #{i}'
                                )
                                ax.semilogy(
                                    windows,
                                    np.abs(current_throws[:, i]),
                                    color=colors[i],
                                    linestyle='--',
                                    marker='x',
                                    label=f'Throw #{i}'
                                )

                            ax.legend()
                            ax.set_ylabel("Energy (log scale)")
                            ax.set_xlabel("Window")
                            ax.tick_params(axis='x', labelrotation=45)
                            ax.set_title(f"{label}_total_and_throw_log")
                            ax.grid(True)

                            filename = f"figures/{matrix_name}_total_throw_log_{'_'.join(label.split(' '))}.png"
                            current_fig = plt.gcf()
                            combined_log_figures.append([current_fig, filename])
                            plt.close()

                        combined_linear_figures = []

                        for current_totals, current_throws, label in data_list:
                            current_totals = np.asarray(current_totals)
                            current_throws = np.asarray(current_throws)

                            fig, ax = plt.subplots(figsize=(12, 8))

                            num_sv = current_totals.shape[1]
                            windows = np.arange(current_totals.shape[0])

                            color_range = np.linspace(0, 1.0, num_sv)
                            if num_sv > 2:
                                color_range[2] = (color_range[-1] + color_range[-2]) / 2
                            color_range = np.sort(color_range)
                            colors = plt.cm.jet(color_range)

                            for i in range(num_sv):
                                ax.plot(
                                    windows,
                                    current_totals[:, i],
                                    color=colors[i],
                                    linestyle='-',
                                    marker='o',
                                    label=f'Total #{i}'
                                )
                                ax.plot(
                                    windows,
                                    current_throws[:, i],
                                    color=colors[i],
                                    linestyle='--',
                                    marker='x',
                                    label=f'Throw #{i}'
                                )

                            ax.legend()
                            ax.set_ylabel("Energy")
                            ax.set_xlabel("Window")
                            ax.tick_params(axis='x', labelrotation=45)
                            ax.set_title(f"{label}_total_and_throw_linear")
                            ax.grid(True)

                            filename = f"figures/{matrix_name}_total_throw_linear_{'_'.join(label.split(' '))}.png"
                            current_fig = plt.gcf()
                            combined_linear_figures.append([current_fig, filename])
                            plt.close()

                        total_figures = []
                        throw_figures = []

                        for current_totals, current_throws, label in data_list:

                            current_totals = np.asarray(current_totals)
                            current_throws = np.asarray(current_throws)

                            num_sv = current_totals.shape[1]
                            windows = np.arange(current_totals.shape[0])

                            color_range = np.linspace(0, 1.0, num_sv)
                            if num_sv > 2:
                                color_range[2] = (color_range[-1] + color_range[-2]) / 2
                            color_range = np.sort(color_range)
                            colors = plt.cm.jet(color_range)

                            # -------- totals plot --------
                            fig, ax = plt.subplots(figsize=(12, 8))

                            for i in range(num_sv):
                                ax.plot(
                                    windows,
                                    current_totals[:, i],
                                    color=colors[i],
                                    marker='o',
                                    label=f'Eigenvalue #{i}'
                                )

                            ax.legend()
                            ax.set_ylabel("Current total")
                            ax.set_xlabel("Window")
                            ax.set_xticks(windows)
                            ax.tick_params(axis='x', labelsize=15, labelrotation=45)
                            ax.tick_params(axis='y', labelsize=15)
                            ax.set_title(f"{label}_current_total")
                            ax.grid(True)

                            filename = f"figures/{matrix_name}_current_total_{'_'.join(label.split(' '))}.png"
                            current_fig = plt.gcf()
                            total_figures.append([current_fig, filename])
                            plt.close()

                            # -------- throws plot --------
                            fig, ax = plt.subplots(figsize=(12, 8))

                            for i in range(num_sv):
                                ax.plot(
                                    windows,
                                    current_throws[:, i],
                                    color=colors[i],
                                    marker='o',
                                    label=f'Eigenvalue #{i}'
                                )

                            ax.legend()
                            ax.set_ylabel("Cumulative throw")
                            ax.set_xlabel("Window")
                            ax.set_xticks(windows)
                            ax.tick_params(axis='x', labelsize=15, labelrotation=45)
                            ax.tick_params(axis='y', labelsize=15)
                            ax.set_title(f"{label}_cumulative_throw")
                            ax.grid(True)

                            filename = f"figures/{matrix_name}_cumulative_throw_{'_'.join(label.split(' '))}.png"
                            current_fig = plt.gcf()
                            throw_figures.append([current_fig, filename])
                            plt.close()
                        

                    if plot_tr_angles:
                        data_list = []
                        for dir_path, label in zip(dir_paths, labels):
                            tr_angles = []
                            s_list = []
                            additional_labels = ""
                            for iteration in range(last_available_file_number+1):
                                file_path = os.path.join(dir_path, f'canonical_angles{additional_labels}_data_{iteration}.npz')
                                
                                try:
                                    data = np.load(file_path)
                                    S = data['C']
                                    # S = 
                                    # S = np.sqrt(1-np.clip(S**2, 0,1)) 
                                    s = np.linalg.svd(S, compute_uv=False)
                                    s = np.sqrt(1-np.clip(s**2, 0,1))
                                    s_list.append(s.reshape(1,-1))

                                except FileNotFoundError:
                                    print(f"File not found: {file_path}")
                            
                            limit_S = min(10, s_list[0].shape[-1])
                            # limit_S = len(data['S'])
                            # e_i = np.abs(data['S_exact'][:len(data['S'])].sum() - tr_S) / data['S_exact'][:len(data['S'])].sum()
                            s_list = np.concatenate(s_list, axis=0)
                            e_i = np.sum(s_list[:, :limit_S], axis=1)
                            e_i = e_i / limit_S
                            # print(Ss[-1])
                            # print(e_i)
                            # import pdb;pdb.set_trace()
                            
                            data_list.append((e_i, label))
                            if smallest_ei > min(e_i):
                                smallest_ei = min(e_i)

                        # if not plot_tr_angles_only_log:
                        #     plt.figure(figsize=(12, 6))
                        #     i = 0
                        #     for (e_i, label), color, linestyle, marker in zip(data_list, label_colors, label_linestyles, label_markers):
                        #         i += 1
                        #         if e_i[0] == 0:
                        #             print("Initial Error is already 0!")
                        #             break
                        #         # import pdb;pdb.set_trace()
                        #         # plt.plot(np.arange(last_available_file_number+1), ((e_i)/e_i[0]), label=f'{label}, init err: {e_i[0]}', linestyle=linestyle, marker=marker,
                        #         #         color=color, alpha=0.7, markevery=i, markersize=12)
                        #         plt.plot(np.arange(last_available_file_number+1), ((e_i)), label=f'{label}, init err: {e_i[0]}', linestyle=linestyle, marker=marker,
                        #                 color=color, alpha=0.7, markevery=i, markersize=12)

                        #     # plt.plot(np.arange(last_available_file_number+1), (1-np.arange(last_available_file_number+1)/(last_available_file_number+1)),
                        #     #         label='1-k/n', linestyle='--', alpha=0.7)

                        #     plt.ylabel("$e_k\ /\ e_0$")
                        #     plt.xlabel("Iteration")
                        #     plt.title(f"Tr( sin $\\theta$ ) over Iterations, Length Scale: 1e{(scaling_factor):.1f}")
                        #     plt.legend()
                        #     plt.grid(True, which='both', linestyle='--', alpha=0.5)
                        #     plt.tight_layout()
                        #     # plt.ylim([-0.01, 1.01])
                        #     #plt.show()
                        #     plt.savefig(f"figures/{matrix_name}_{'quotient_' if plot_S_quotient else ''}tr_angles_over_time_{'_'.join(labels[-1].split(' '))}.png")
                        #     plt.close()
                        
                        # import pdb;pdb.set_trace()
                        plt.figure(figsize=(12, 6))
                        i = 0
                        for (e_i, label), color, linestyle, marker in zip(data_list, label_colors, label_linestyles, label_markers):
                            i += 1
                            if e_i[0] == 0:
                                print("Initial Error is already 0!")
                                break
                            # plt.plot(np.arange(last_available_file_number+1), np.log10(np.abs((e_i)/e_i[0])), label=f'{label}', linestyle=linestyle, marker=marker,
                            #         color=color, alpha=0.7, markevery=i, markersize=12)
                            plt.plot(np.arange(last_available_file_number+1), np.log10(np.abs((e_i))), label=f'{label}', linestyle=linestyle, marker=marker,
                                    color=color, alpha=0.7, markevery=i, markersize=12)
                        # plt.plot(np.arange(last_available_file_number+1), np.log10(1-np.arange(last_available_file_number)/last_available_file_number),
                        #          label='1-k/n', linestyle='--', alpha=0.7)

                        plt.ylabel("log $e_k\ /\ e_0$")
                        plt.xlabel("Iteration")
                        plt.title(f"(Log) Tr( sin $\\theta$ ) over Iterations, Length Scale: 1e{(scaling_factor):.1f}")
                        plt.legend()
                        plt.grid(True, which='both', linestyle='--', alpha=0.5)
                        plt.tight_layout()
                        #plt.show()
                        plt.savefig(f"figures/{matrix_name}_{'quotient_' if plot_S_quotient else ''}tr_angles_over_time_log_{'_'.join(labels[-1].split(' '))}.png")
                        plt.close()

                    if plot_angles:
                        data_list = []
                        for dir_path, label in zip(dir_paths, labels):
                            tr_angles = []
                            s_list = []
                            additional_labels = ""
                            for iteration in range(last_available_file_number+1):
                                file_path = os.path.join(dir_path, f'canonical_angles{additional_labels}_data_{iteration}.npz')
                                
                                try:
                                    data = np.load(file_path)
                                    S = data['C']
                                    # S = 
                                    # S = np.sqrt(1-np.clip(S**2, 0,1)) 
                                    try:
                                        s = np.linalg.svd(S, compute_uv=False)
                                    except:
                                        import pdb;pdb.set_trace()
                                    s = np.sqrt(1-np.clip(s**2, 0,1))
                                    s_list.append(s.reshape(1,-1))
                                    # import pdb;pdb.set_trace()

                                except FileNotFoundError:
                                    print(f"File not found: {file_path}")
                            
                            limit_S = min(10, s_list[0].shape[-1])
                            # limit_S = len(data['S'])
                            # e_i = np.abs(data['S_exact'][:len(data['S'])].sum() - tr_S) / data['S_exact'][:len(data['S'])].sum()
                            s_list = np.concatenate(s_list, axis=0)
                            e_i = np.sum(s_list[:, :limit_S], axis=1)
                            e_i = e_i / limit_S
                            # print(Ss[-1])
                            # print(e_i)
                            # import pdb;pdb.set_trace()
                            
                            data_list.append((e_i, label))
                            if smallest_ei > min(e_i):
                                smallest_ei = min(e_i)

                            # if "demix" in label and scaling_factor == 2.0:
                            #     import pdb;pdb.set_trace()

                        # import pdb;pdb.set_trace()
                            fig, ax = plt.subplots(figsize=(12, 8))
                            color_range = np.linspace(0, 1.0, s_list.shape[1])
                            color_range[2] = (color_range[-1] + color_range[-2]) / 2
                            color_range = np.sort(color_range)
                            colors = plt.cm.jet(color_range)
                            # Plot each window's data
                            for i in range(s_list.shape[1]):
                                try:
                                    plt.semilogy(np.arange(s_list.shape[0]), s_list[:,i], color=colors[i], label=f'Principle Angle #{i}', marker='o', alpha=0.7)
                                except:
                                    # import pdb;pdb.set_trace()
                                    raise
                            # Find min and max non-zero values for better y-limit setting
                            all_values = s_list.flatten()
                            non_zero_values = all_values[all_values > 0]
                            if len(non_zero_values) > 0:
                                y_min = np.min(non_zero_values) * 0.8  # Padding below
                                y_max = np.max(all_values) * 1.2       # Padding above
                                plt.ylim(y_min, y_max)
                            plt.ylim([1e-8,1])
                            plt.legend()
                            # plt.yscale('log')
                            plt.ylabel("Relative eigenvalue difference")
                            plt.xlabel("Window")
                            plt.xticks(fontsize=15)
                            plt.yticks(fontsize=15)
                            plt.title(f"{label}_{'quotient' if plot_S_quotient else ''}")
                            plt.grid()
                            plt.savefig(f"figures/{matrix_name}_{'quotient_' if plot_S_quotient else ''}angles_over_time_{'_'.join(label.split(' '))}.png")
                            plt.close()

                    if plot_angles_indi:
                        data_list = []
                        for dir_path, label in zip(dir_paths, labels):
                            tr_angles = []
                            s_list = []
                            additional_labels = ""
                            for iteration in range(last_available_file_number+1):
                                file_path = os.path.join(dir_path, f'canonical_angles{additional_labels}_data_{iteration}.npz')
                                
                                try:
                                    data = np.load(file_path)
                                    S = data['C']
                                    s = np.linalg.norm(S,axis=0)
                                    # S = 
                                    # S = np.sqrt(1-np.clip(S**2, 0,1)) 
                                    # try:
                                    #     s = np.linalg.svd(S, compute_uv=False)
                                    # except:
                                    #     import pdb;pdb.set_trace()
                                    # s = np.sqrt(1-np.clip(s**2, 0,1))
                                    s = np.sqrt(1-np.clip(s**2,0,1))
                                    s_list.append(s.reshape(1,-1))
                                    # import pdb;pdb.set_trace()

                                except FileNotFoundError:
                                    print(f"File not found: {file_path}")
                            
                            limit_S = min(10, s_list[0].shape[-1])
                            # limit_S = len(data['S'])
                            # e_i = np.abs(data['S_exact'][:len(data['S'])].sum() - tr_S) / data['S_exact'][:len(data['S'])].sum()
                            s_list = np.concatenate(s_list, axis=0)
                            e_i = np.sum(s_list[:, :limit_S], axis=1)
                            e_i = e_i / limit_S
                            # print(Ss[-1])
                            # print(e_i)
                            # import pdb;pdb.set_trace()
                            
                            data_list.append((e_i, label))
                            if smallest_ei > min(e_i):
                                smallest_ei = min(e_i)

                            # if "demix" in label and scaling_factor == 2.0:
                            #     import pdb;pdb.set_trace()

                        # import pdb;pdb.set_trace()
                            fig, ax = plt.subplots(figsize=(12, 8))
                            color_range = np.linspace(0, 1.0, s_list.shape[1])
                            color_range[2] = (color_range[-1] + color_range[-2]) / 2
                            color_range = np.sort(color_range)
                            colors = plt.cm.jet(color_range)
                            # Plot each window's data
                            for i in range(s_list.shape[1]):
                                try:
                                    plt.semilogy(np.arange(s_list.shape[0]), s_list[:,i], color=colors[i], label=f'Principle Angle #{i}', marker='o', alpha=0.7)
                                except:
                                    # import pdb;pdb.set_trace()
                                    raise
                            # Find min and max non-zero values for better y-limit setting
                            all_values = s_list.flatten()
                            non_zero_values = all_values[all_values > 0]
                            if len(non_zero_values) > 0:
                                y_min = np.min(non_zero_values) * 0.8  # Padding below
                                y_max = np.max(all_values) * 1.2       # Padding above
                                plt.ylim(y_min, y_max)
                            plt.ylim([1e-8,1])
                            plt.legend()
                            # plt.yscale('log')
                            plt.ylabel("Relative eigenvalue difference")
                            plt.xlabel("Window")
                            plt.xticks(fontsize=15)
                            plt.yticks(fontsize=15)
                            plt.title(f"{label}_{'quotient' if plot_S_quotient else ''}")
                            plt.grid()
                            plt.savefig(f"figures/{matrix_name}_{'quotient_' if plot_S_quotient else ''}angles_indi_over_time_{'_'.join(label.split(' '))}.png")
                            plt.close()
                        # plt.figure(figsize=(12, 6))
                        # i = 0
                        # for (e_i, label), color, linestyle, marker in zip(data_list, label_colors, label_linestyles, label_markers):
                        #     i += 1
                        #     if e_i[0] == 0:
                        #         print("Initial Error is already 0!")
                        #         break
                        #     # plt.plot(np.arange(last_available_file_number+1), np.log10(np.abs((e_i)/e_i[0])), label=f'{label}', linestyle=linestyle, marker=marker,
                        #     #         color=color, alpha=0.7, markevery=i, markersize=12)
                        #     plt.plot(np.arange(last_available_file_number+1), np.log10(np.abs((e_i))), label=f'{label}', linestyle=linestyle, marker=marker,
                        #             color=color, alpha=0.7, markevery=i, markersize=12)
                        # # plt.plot(np.arange(last_available_file_number+1), np.log10(1-np.arange(last_available_file_number)/last_available_file_number),
                        # #          label='1-k/n', linestyle='--', alpha=0.7)

                        # plt.ylabel("log $e_k\ /\ e_0$")
                        # plt.xlabel("Iteration")
                        # plt.title(f"(Log) Tr( sin $\\theta$ ) over Iterations, Length Scale: 1e{(scaling_factor):.1f}")
                        # plt.legend()
                        # plt.grid(True, which='both', linestyle='--', alpha=0.5)
                        # plt.tight_layout()
                        # #plt.show()
                        # plt.savefig(f"figures/{matrix_name}_{'quotient_' if plot_S_quotient else ''}tr_angles_over_time_log_{'_'.join(labels[-1].split(' '))}.png")
                        # plt.close()
                    
                    if plot_time_elapsed:
                        from matplotlib import cm

                        # Create figure and axis
                        plt.figure(figsize=(12, 6))
                        time_elapsed = []
                        for dir_path, label in zip(dir_paths, labels):
                            file_path = os.path.join(dir_path, f'other_info.npz')
                            try:
                                data = np.load(file_path, allow_pickle=True)
                                # import pdb;pdb.set_trace()
                                other_info = data['other_info'].item()
                                time_elapsed.append(other_info["time_elapsed"])
                            except FileNotFoundError:
                                print(f"File not found: {file_path}")
                        time_elapsed = np.array(time_elapsed)

                        # Generate a color map based on number of categories
                        num_categories = len(time_elapsed)
                        categories = np.arange(num_categories)
                        cmap = cm.get_cmap('viridis', num_categories)  # 'tab10', 'tab20', 'viridis', etc.

                        # Plot each bar individually with its own label and color
                        for i, (category, value, label) in enumerate(zip(categories, time_elapsed, labels)):
                            plt.bar(i, np.log10(value), color=cmap(i), label=label)

                        plt.ylabel("10^{x} Time Elapsed", fontsize=16)
                        plt.xlabel("", fontsize=16)
                        plt.title(f"")
                        plt.legend()
                        plt.grid(True, which='both', linestyle='--', alpha=0.5)
                        plt.tight_layout()
                        plt.xticks(fontsize=15)
                        plt.yticks(fontsize=15)
                        # plt.ylim([-0.01, 1.01])
                        #plt.show()
                        plt.savefig(f"figures/{matrix_name}_time_elapsed_{'_'.join(labels[-1].split(' '))}.png")
                        plt.close()

                    if plot_entropy and matrix_name not in entropy_d:
                        # Create figure and axis
                        plt.figure(figsize=(12, 6))
                        entropy = []
                        dir_path = dir_paths[0]
                        label = labels[0]
                        file_path = os.path.join(dir_path, f'other_info.npz')
                        try:
                            data = np.load(file_path, allow_pickle=True)
                            # import pdb;pdb.set_trace()
                            other_info = data['other_info'].item()
                            entropy.append(other_info["true_normalized_entropy"])

                            data_S = np.load(os.path.join(dir_path, "spectrum_data_0.npz"), allow_pickle=True)
                            S_exact = data_S["S_exact"]

                            p = S_exact / np.sum(S_exact)
                            ent = sp.stats.entropy(p, base=2)
                            # calculate 
                            erank = 2**(ent)

                            data_row_perm = np.load(os.path.join(dir_path, f'row_order_final.npz'), allow_pickle=True)
                            matrix_size = data_row_perm["row_permutation"].shape[0]

                            # low_S_exact = np.pad(S_exact, (0, matrix_size - len(S_exact)), 'constant')


                            
                        except FileNotFoundError:
                            print(f"File not found: {file_path}")
                            # raise
                        entropy = np.array(entropy).squeeze()
                        entropy_d["_".join(matrix_name.split("_")[:-1])] = [entropy, erank, matrix_size, len(S_exact)]

                    if plot_wholespace_residual:
                        data_list = []
                        for dir_path, label in zip(dir_paths, labels):
                            reservoir_residuals = []
                            regular_residuals = []
                            reservoir_residuals_quotient = []
                            regular_residuals_quotient = []
                            ws_reg_res_2norms = []
                            ws_reg_res_fros = []
                            ws_res_res_2norms = []
                            ws_res_res_fros = []
                            ws_reg_res_quotient_2norms = []
                            ws_reg_res_quotient_fros = []
                            ws_res_res_quotient_2norms = []
                            ws_res_res_quotient_fros = []
                            fig, ax = plt.subplots(figsize=(8, 6))
                            for iteration in range(last_available_file_number+1):
                                # Load data
                                file_path = os.path.join(dir_path, f'reservoir_residuals_data_{iteration}.npz')
                                try:
                                    data = np.load(file_path)
                                    # res_residuals = data['reservoir_residuals'].reshape(1,-1)
                                    # res_residuals *= np.sqrt(matrix_size / reservoir_size)
                                    # reg_residuals = data['regular_residuals'].reshape(1,-1)
                                    # reservoir_residuals.append(res_residuals)
                                    # regular_residuals.append(reg_residuals)
                                    # res_residuals_quotient = data['reservoir_residuals_quotient'].reshape(1,-1) 
                                    # res_residuals_quotient *= np.sqrt(matrix_size / reservoir_size)
                                    # reg_residuals_quotient = data['regular_residuals_quotient'].reshape(1,-1) 
                                    # reservoir_residuals_quotient.append(res_residuals_quotient)
                                    # regular_residuals_quotient.append(reg_residuals_quotient)
                                    ws_reg_res_2norms.append(data["whole_space_regular_residuals_2norm"])
                                    ws_reg_res_fros.append(data["whole_space_regular_residuals_fro"])
                                    ws_res_res_2norms.append(data["whole_space_reservoir_residuals_2norm"])
                                    ws_res_res_fros.append(data["whole_space_reservoir_residuals_fro"])
                                    ws_reg_res_quotient_2norms.append(data["whole_space_regular_residuals_quotient_2norm"])
                                    ws_reg_res_quotient_fros.append(data["whole_space_regular_residuals_quotient_fro"])
                                    ws_res_res_quotient_2norms.append(data["whole_space_reservoir_residuals_quotient_2norm"])
                                    ws_res_res_quotient_fros.append(data["whole_space_reservoir_residuals_quotient_fro"])

                                    # trace = np.sum(data['S'])
                                    # tr_S.append(trace)
                                    # ax.semilogy(approx_residuals, label=f'{label}', color=color, linestyle=linestyle, marker=marker, alpha=0.7, markevery=i, markersize=12)
                                except FileNotFoundError:
                                    print(f"File not found: {file_path}")
                            if plot_ws_reg: 
                                if plot_ws_quotient:
                                    data_list.append([ws_reg_res_quotient_fros, label])
                                else:
                                    data_list.append([ws_reg_res_fros, label])
                            else:
                                if plot_ws_quotient:
                                    data_list.append([ws_res_res_quotient_fros, label])
                                else:
                                    data_list.append([ws_res_res_fros, label])

                        plt.figure(figsize=(12, 6))
                        i = 0
                        for (ws_res, label), color, linestyle, marker in zip(data_list, label_colors, label_linestyles, label_markers):
                            i += 1
                            if e_i[0] == 0:
                                print("Initial Error is already 0!")
                                break
                            if "Vapprox" not in label:
                                method_d_residuals[name_postfix+"_quotient" if plot_S_quotient else name_postfix] = ws_res
                            # plt.plot(np.arange(last_available_file_number+1), np.log10(np.abs((e_i)/e_i[0])), label=f'{label}', linestyle=linestyle, marker=marker,
                            #         color=color, alpha=0.7, markevery=i, markersize=12)
                            plt.semilogy(np.arange(last_available_file_number+1), ws_res, label=f'{label}', linestyle=linestyle, marker=marker,
                                    color=color, alpha=0.7, markevery=i, markersize=12)
                            # import pdb;pdb.set_trace()
                        # plt.plot(np.arange(last_available_file_number+1), np.log10(1-np.arange(last_available_file_number)/last_available_file_number),
                        #          label='1-k/n', linestyle='--', alpha=0.7)
                        # if scaling_factor == 2.0:
                        #     plt.ylim(-7,1)
                        # elif scaling_factor == 5.0:
                        #     plt.ylim(-7,1)
                        # elif scaling_factor == 10.0:
                        #     plt.ylim(-6,0)
                        # elif scaling_factor == 20.0:
                        #     plt.ylim(-8,0)
                        plt.ylabel("Residual", fontsize=16)
                        plt.xlabel("Iteration", fontsize=16)
                        plt.title(f"Whole space residual over Windows, Length Scale: {(scaling_factor):.1f}")
                        plt.legend()
                        plt.xticks(fontsize=15)
                        plt.yticks(fontsize=15)
                        plt.grid(True, which='both', linestyle='--', alpha=0.5)
                        plt.tight_layout()
                        #plt.show()
                        # plt.savefig(f"figures/{matrix_name}_{'quotient_' if plot_S_quotient else ''}error_over_time_log_{'_'.join(labels[-1].split(' '))}.png")
                        current_figure = plt.gcf()
                        filename = f"figures/{matrix_name}_{'quotient_' if plot_ws_quotient else ''}ws_residuals_over_time_log_{'_'.join(labels[-1].split(' '))}.png"
                        ws_residual_figures.append([current_figure, filename])
                        plt.close()

                        
                    # Plot and change y_axis
                    # ev_change_figures = []
                    # trace_error_figures = []
                    # regular_residuals_figures = []
                    # Find the global y-limits across all figures
                    def set_figures_same_ylim(figures):
                        all_ylims = []
                        for fig, _ in figures:
                            current_ylim = fig.axes[0].get_ylim()
                            all_ylims.extend(current_ylim)

                        # Set common ylim based on global range
                        if len(all_ylims) == 0:
                            # skip
                            return
                        global_ymin = min(all_ylims)
                        global_ymax = max(all_ylims)

                        # Apply to all figures
                        for fig, _ in figures:
                            fig.axes[0].set_ylim(global_ymin, global_ymax)

                        for fig, file_path in figures:
                            # for quality in range(100, 5, -5):
                            quality = 80
                            fig.savefig(file_path, format='jpg', dpi=100,
                                        bbox_inches='tight', pad_inches=0,
                                        pil_kwargs={'quality': quality, 'optimize': True})
                            #     print(f"Compressed to ~{os.path.getsize(file_path)/1024:.1f}KB with quality {quality}")
                            # raise
                            # print(file_path)
                        # print(all_ylims)
                        # import pdb;pdb.set_trace()

                    if plot_ev_change:
                        set_figures_same_ylim(ev_change_figures)
                    if plot_trace_error:
                        set_figures_same_ylim(trace_error_figures)
                    if plot_jer_residual:
                        set_figures_same_ylim(regular_residuals_figures)
                        # import pdb;pdb.set_trace()
                    if plot_wholespace_residual:
                        set_figures_same_ylim(ws_residual_figures)
                    if plot_leftout:
                        set_figures_same_ylim(combined_log_figures)
                        set_figures_same_ylim(combined_linear_figures)
                        set_figures_same_ylim(total_figures)
                        set_figures_same_ylim(throw_figures)


                    if method_d_trace_error and method_d_residuals:
                        # Create a list to control legend order: iSVD, Quotient, then demix
                        method_order = []
                        for n in method_d_trace_error.keys():
                            if "isvddemix" in n and not "quotient" in n:
                                continue
                            method_order.append(n)

                        # Sort methods: iSVD first, then quotient, then demix
                        def sort_key(method_name):
                            if "isvd" in method_name.lower() and "demix" not in method_name.lower():
                                return (0, method_name)  # iSVD methods first
                            elif "quotient" in method_name.lower():
                                return (1, method_name)  # Quotient methods second
                            elif "demix" in method_name.lower():
                                return (2, method_name)  # Demix methods last
                            else:
                                return (3, method_name)  # Other methods at the end

                        method_order.sort(key=sort_key)

                        color_range = np.linspace(0, 1.0, len(method_order))
                        # color_range[2] = (color_range[-1] + color_range[-2]) / 2
                        color_range = np.sort(color_range)
                        colors = plt.cm.jet(color_range)
                        plt.figure(figsize=(12, 6))
                        # print(method_d_trace_error.keys())
                        # import pdb;pdb.set_trace()
                        for n, color in zip(method_order, colors):
                            if "isvddemix" in n and not "quotient" in n:
                                continue
                            # print(f"Label: '{n}'") 
                            # plt.plot(np.arange(last_available_file_number+1), np.log10(np.abs((e_i)/e_i[0])), label=f'{label}', linestyle=linestyle, marker=marker,
                            #         color=color, alpha=0.7, markevery=i, markersize=12)
                            if "isvddemix" in n and "quotient" in n:
                                name = "Demix"
                            elif "quotient" in n:
                                name = "Least Squares"
                            else:
                                name = "iSVD"
                            plt.semilogy(np.arange(len(method_d_trace_error[n])), np.abs(method_d_trace_error[n]), label=f'{name}', marker='o',
                                    color=color, alpha=0.7, markersize=12)
                            # import pdb;pdb.set_trace()
                        # plt.plot(np.arange(last_available_file_number+1), np.log10(1-np.arange(last_available_file_number)/last_available_file_number),
                        #          label='1-k/n', linestyle='--', alpha=0.7)
                        plt.ylabel("Relative trace error", fontsize=16)
                        plt.xlabel("Window index", fontsize=16)
                        plt.title(f"Relative Trace Error over Windows, Length Scale: {(scaling_factor)}")
                        plt.legend()
                        # legend = plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
                        # print(legend)
                        plt.yticks(fontsize=15)
                        # Set custom x-axis ticks to show even numbers (2, 4, 6, 8, ...)
                        max_x = max([len(method_d_trace_error[n]) for n in method_order]) - 1
                        tick_positions = np.arange(0, max_x + 1, 2)  # Start from 2, step by 2
                        plt.xticks(tick_positions, fontsize=15)
                        plt.grid(True, which='both', linestyle='--', alpha=0.5)
                        plt.tight_layout()
                        #plt.show()
                        plt.savefig(f"figures/{matrix_name}_method_error_over_time_log.png", bbox_inches='tight')
                        plt.close()

                        # color_range = np.linspace(0, 1.0, len(method_d_residuals))
                        # # color_range[2] = (color_range[-1] + color_range[-2]) / 2
                        # color_range = np.sort(color_range)
                        # colors = plt.cm.jet(color_range)
                        plt.figure(figsize=(12, 6))
                        # print(method_d_residuals.keys())
                        # import pdb;pdb.set_trace()
                        for n, color in zip(method_order, colors):
                            if "isvddemix" in n and not "quotient" in n:
                                continue
                            if "isvddemix" in n and "quotient" in n:
                                name = "Demix"
                            elif "quotient" in n:
                                name = "Least Squares"
                            else:
                                name = "iSVD"
                            # print(f"Label: '{n}'") 
                            # plt.plot(np.arange(last_available_file_number+1), np.log10(np.abs((e_i)/e_i[0])), label=f'{label}', linestyle=linestyle, marker=marker,
                            #         color=color, alpha=0.7, markevery=i, markersize=12)
                            plt.semilogy(np.arange(len(method_d_residuals[n])), np.abs(method_d_residuals[n]), label=f'{name}', marker='o',
                                    color=color, alpha=0.7, markersize=12)
                            # import pdb;pdb.set_trace()
                        # plt.plot(np.arange(last_available_file_number+1), np.log10(1-np.arange(last_available_file_number)/last_available_file_number),
                        #          label='1-k/n', linestyle='--', alpha=0.7)
                        plt.ylabel("Whole space residual", fontsize=16)
                        plt.xlabel("Window index", fontsize=16)
                        plt.title(f"Whole Space Residual over Windows, Length Scale: {(scaling_factor)}")
                        plt.legend()

                        # Set custom x-axis ticks to show even numbers (2, 4, 6, 8, ...)
                        max_x = max([len(method_d_trace_error[n]) for n in method_order]) - 1
                        tick_positions = np.arange(0, max_x + 1, 2)  # Start from 2, step by 2
                        plt.xticks(tick_positions, fontsize=15)
                        # legend = plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
                        # print(legend)
                        plt.yticks(fontsize=15)
                        plt.grid(True, which='both', linestyle='--', alpha=0.5)
                        plt.tight_layout()
                        #plt.show()
                        plt.savefig(f"figures/{matrix_name}_method_residuals_ws_over_time_log.png", bbox_inches='tight')
                        plt.close()

if plot_entropy:
    # print(entropy_d)
    # import pdb;pdb.set_trace()
    color_range = np.linspace(0, 1.0, len(entropy_d))
    if color_range.shape[0] > 2:
        color_range[2] = (color_range[-1] + color_range[-2]) / 2
        color_range = np.sort(color_range)
    colors = plt.cm.jet(color_range)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, matrix_name in enumerate(entropy_d):
        ax.plot(range(entropy_d[matrix_name][0].shape[0]), entropy_d[matrix_name][0], color=colors[i], label=matrix_name + f", erank{'>=' if entropy_d[matrix_name][3] < entropy_d[matrix_name][2] else '='}{entropy_d[matrix_name][1]:.1f},matrix size={entropy_d[matrix_name][2]}")
    ax.grid(True)
    # ax.set_yscale('log')
    ax.set_xlabel('Index', fontsize=16)
    ax.set_ylabel('Normalized Entropy', fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.title(f"entropy")
    plt.legend()
    plt.savefig(f"figures/entropy_{'_'.join([n[:3] for n in entropy_d])}.png")
    plt.close()


print("Missing data:")
print(missing_data)

print("Incomplete data:")
print(incomplete_data)

    # Reconstruction Error
    # print("Visualizing reconstruction error")
    # for k in [10]:
    #     for random_seed in [""]:
    #         for name_postfix in ["_new"]:
    #             for scaling_factor in [2.0, 1.5, 1.0, 0.5, 0.0]:
    #                 for is_reversed in [False]:
    #                     for use_true_matrix in [True]:
    #                         if name_postfix == "" and (k != 10 or is_reversed or use_true_matrix):
    #                             continue
    #                         print(k , random_seed, name_postfix)
    #                         print("Scaling factor:", scaling_factor)
    #                         print("Is reversed:", is_reversed)
    #                         print("use_true_matrix:", use_true_matrix)
                            
    #                         #postfix = "_quotient"
    #                         postfix = f"_k_{k}"
    #                         size = 100
    #                         # is_reversed = False
    #                         # matrix_name = "bodyy4"
    #                         # matrix_name = "kronecker_graph_13_0.3"
    #                         matrix_name = f"hyperboloid_1000_{scaling_factor}{name_postfix}"
    #                         figure_dir = 'output'
    #                         # matrix_type2 = '_original'
    #                         matrix_type = '_random_uniform'
    #                         # matrix_type = '_original'
    #                         # matrices_random = [f'_random_uniform_{x}' for x in range(1,6)]
    #                         # matrices_random[0] = '_random_uniform'
    #                         # matrix_type = '_random_uniform_col_perm'
    #                         colors = plt.cm.rainbow(np.linspace(0, 1, 5))
    #                         # matrix_postfix + "_new_" + f"Vapprox_withS_{num_Vs}_" + row_permutation + "_" + f"size_{size}_k_{k}"
    #                         matrices = {
    #                             f'{matrix_name}{matrix_type}{"_true" if use_true_matrix else ""}_size_{size}{postfix}': [colors[0], '-', '.'],
    #                         #     f'{matrix_name}{matrix_type2}_size_{size}{postfix}': [colors[0], ':', 'o'],
    #                         #     f'{matArix_name}{matrix_type3}': [colors[0], '-.'],
    #                         # f'{matrix_name}_decreasing_norm_size_{size}{postfix}': [colors[1], '-.', '^'],
    #                         # f'{matrix_name}_increasing_norm_size_{size}{postfix}': [colors[1], '-.', '^'],
    #                             # f'{matrix_name}_Vapprox_withS_{num_Vs}{matrix_type}': [colors[2], '-'],
    #                         #     f'{matrix_name}_Vapprox_{num_Vs}{matrix_type}': [colors[3], '-'],
    #                         #     f'{matrix_name}_Vapprox_reversed_{num_Vs}{matrix_type}': [colors[4], '-']
    #                         }
    #                         matrices.update({
    #                         f'{matrix_name}_Vapprox_withS_{num_Vs}{matrix_type}{"_reversed" if is_reversed else ""}{"_true" if use_true_matrix else ""}_size_{size}{postfix}': [colors[2+i], '-', 's'] for i, num_Vs in enumerate([1,10,100])
    #                         })
    #                         # matrices.update({
    #                         #     f'{matrix_name}_Vapprox_withS_{num_Vs}{matrix_type2}_size_{size}{postfix}': [colors[2+i], ':', '*'] for i, num_Vs in enumerate([1,10,100])
    #                         # })

    #                         print("Matrices:", matrices.keys())

    #                         # matrix_name = 'temp'
    #                         # figure_dir = 'figures'
    #                         # matrices = [f'{matrix_name}',]
    #                         # colors = plt.cm.rainbow(np.linspace(0, 1, 5))
    #                         # matrices = {f'{matrix_name}': [colors[0], '-']}

    #                         dir_paths = [f"{figure_dir}/{matrix_postfix}/" for matrix_postfix in matrices]
    #                         # dir_path = dir_paths[1]
    #                         # print("dir_paths:", dir_paths)
    #                         # raise

    #                         labels = [' '.join(s.split('_')[1:]) for s in matrices]
    #                         label_colors = {k:v[0] for k,v in matrices.items()}
    #                         label_linestyles = {k:v[1] for k,v in matrices.items()}
    #                         label_markers = {k:v[2] for k,v in matrices.items()}
    #                         label_colors = label_colors.values()
    #                         label_linestyles = label_linestyles.values()
    #                         label_markers = label_markers.values()

    #                         plt.figure(figsize=(12, 6))
    #                         i = 0
    #                         for dir_path, label, color, linestyle, marker in zip(dir_paths, labels, label_colors, label_linestyles, label_markers):
    #                             file_path = os.path.join(dir_path, f'reconstruction_error.npz')
    #                             try:
    #                                 data = np.load(file_path)
    #                                 reconstruction_errors = data['reconstruction_errors']
    #                             except FileNotFoundError:
    #                                 print(f"File not found: {file_path}")
                                
    #                             i += 1
    #                             # import pdb;pdb.set_trace()
    #                             plt.plot(np.arange(len(reconstruction_errors)), reconstruction_errors, label=f'{label}', linestyle=linestyle, marker=marker,
    #                                     color=color, alpha=0.7, markevery=i, markersize=12)

    #                         plt.ylabel("$e_k\ /\ e_0$")
    #                         plt.xlabel("Iteration")
    #                         plt.title(f"Reconstruction Error $\|M_{{approx}}-M\| / \|M\|$ over Iterations, Length Scale: 1e{(scaling_factor):.1f}")
    #                         plt.legend()
    #                         plt.grid(True, which='both', linestyle='--', alpha=0.5)
    #                         plt.tight_layout()
    #                         # plt.ylim([-0.01, 1.01])
    #                         #plt.show()
    #                         plt.savefig(f"figures/reconstruction_err_{'_'.join(labels[-1].split(' '))}.png")
    #                         plt.close()


# # for matrix_name_prefix in ["kernel_swissroll", "kernel_torus", "kernel_gaussianmixture"]:
# for matrix_name_prefix in ["hyperboloid"]:
# # for k in [10, 20, 50, 100]:
#     for k in [10]:
#     # for k in [10, 20, 50, 100, 200, 400, 600, 800, 1000]:
#         # for random_seed in ["", "_2", "_3"]:
#         for random_seed in [""]:
            
#             # for scaling_factor in [2.0, 1.5, 1.0, 0.5, 0.0]:
#             # for scaling_factor in [2.0, 1.0, 0.0]:
#             for scaling_factor in [2.0]:
#                 # for noise in [0.0, 0.01, 0.05, 0.1]:
#                 # for noise in [0.001, 0.01, 0.1]:
#                 # for noise in [0.5]:
#                 for noise in [0.0]:
#                     for is_reversed in [False]:
#                         for use_true_matrix in [False]:
#                             # for reservoir_size in [10, 50, 100, 200]:
#                             for reservoir_size in [10]:
#                                 # for threshold_factor in [1e1, 1e2, 1e3, 1e4]:
#                                 for threshold_factor in [1e2]:
#                                     # for error_kind in ["both", "kernel", "point"]:
#                                     for error_kind in ["both"]:
                                            
#                                         if noise == 0.0 and error_kind != "both":
#                                             continue
#                                         if name_postfix != "" and reservoir_size != 10:
#                                             continue
#                                         # if name_postfix == "" and (k != 10 or is_reversed or use_true_matrix):
#                                         #     continue
                                        
#                                         print(k , random_seed, name_postfix)
#                                         print("Scaling factor:", scaling_factor)
#                                         print("Is reversed:", is_reversed)
#                                         print("use_true_matrix:", use_true_matrix)
#                                         print("reservoir size", reservoir_size)

#                                         plot_spectrum = False
#                                         kernel_error_only = error_kind == "kernel"
#                                         point_error_only = error_kind == "point"
#                                         plot_jer_residual = False
#                                         plot_trace_error = True
#                                         plot_eig_err_heatmap = False
#                                         plot_trace_error_only_log = True
#                                         plot_S_quotient = True # just for the trace_error graph
#                                         plot_tr_angles = False
#                                         plot_tr_angles_only_log = True
#                                         plot_detailed_iterations = False

#                                         #postfix = "_quotient"
#                                         postfix = f"_k_{k}"
#                                         if name_postfix == "":
#                                             postfix += f"_rs_{reservoir_size}"
#                                         postfix += f"_factor_{threshold_factor}" if name_postfix == "_new" and threshold_factor != 1e2 else ""
#                                         size = 100 if k <= 100 else k #k #100
#                                         # is_reversed = False
#                                         # matrix_name = "bodyy4"
#                                         # matrix_name = "kronecker_graph_13_0.3"
#                                         matrix_name = f"{matrix_name_prefix}_1000_{scaling_factor}"
#                                         # matrix_name = f"kernel_random_1000_{scaling_factor}"
#                                         # matrix_name = "synthetic_1000_20"
#                                         # import pdb;pdb.set_trace()

#                                         if (kernel_error_only or point_error_only) and noise == 0.0:
#                                             continue
#                                         if noise > 0.0:
#                                             if kernel_error_only:
#                                                 matrix_name = matrix_name + f"_{noise}_0.0"
#                                             elif point_error_only:
#                                                 matrix_name = matrix_name + f"_0.0_{noise}"
#                                             else:
#                                                 matrix_name += f"_{noise}"

#                                         data_list = []
#                                         smallest_ei = np.inf
#                                         og_matrix_name = matrix_name
#                                         # for name_postfix in ["", "new"]:
#                                         # for name_postfix in ["nystrom_1_isvd_-1", "nystrom_1_isvd_-2", "nystrom_1_isvd", 
#                                         #                      "nystrom_5_isvd", "nystrom", "isvd", "nystrom_1_isvd_100", "nystrom_1_isvd_500"]:
#                                         for name_postfix in ["nystrom_1_isvd_-1", "nystrom_1_isvd_-2", "nystrom_1_isvd", 
#                                                              "nystrom_1_isvd_-3", "nystrom_1_isvd_-4", "nystrom_1_isvd_-5"]:
#                                             matrix_name = og_matrix_name
#                                             if name_postfix:
#                                                 name_postfix = "_" + name_postfix
#                                             # for name_postfix in [""]:
#                                             matrix_name = f"{matrix_name}{name_postfix}"
#                                             figure_dir = 'output'
#                                             # matrix_type2 = '_original'
#                                             matrix_type = f'_random_uniform{random_seed}'
#                                             # matrix_type = "_manual_perm"
#                                             # matrix_type = "_manual_perm_reverse"
#                                             # matrix_type = '_original'
#                                             # matrices_random = [f'_random_uniform_{x}' for x in range(1,6)]
#                                             # matrices_random[0] = '_random_uniform'
#                                             # matrix_type = '_random_uniform_col_perm'
#                                             colors = plt.cm.rainbow(np.linspace(0, 1, 6))
#                                             # matrix_postfix + "_new_" + f"Vapprox_withS_{num_Vs}_" + row_permutation + "_" + f"size_{size}_k_{k}"
#                                             matrices = {
#                                                 f'{matrix_name}{matrix_type}{"_true" if use_true_matrix else ""}_size_{size}{postfix}': [colors[0], '-', '.'],
#                                             #     f'{matrix_name}{matrix_type2}_size_{size}{postfix}': [colors[0], ':', 'o'],
#                                             #     f'{matArix_name}{matrix_type3}': [colors[0], '-.'],
#                                             # f'{matrix_name}_decreasing_norm{"_true" if use_true_matrix else ""}_size_{size}{postfix}': [colors[1], '-.', '^'],
#                                             # f'{matrix_name}_increasing_norm{"_true" if use_true_matrix else ""}_size_{size}{postfix}': [colors[2], '-.', '^'],
#                                                 # f'{matrix_name}_Vapprox_withS_{num_Vs}{matrix_type}': [colors[2], '-'],
#                                             #     f'{matrix_name}_Vapprox_{num_Vs}{matrix_type}': [colors[3], '-'],
#                                             #     f'{matrix_name}_Vapprox_reversed_{num_Vs}{matrix_type}': [colors[4], '-']
#                                             }
#                                             Vs_list = np.unique([min(x, k) for x in [1,10,100]])
#                                             # import pdb;pdb.set_trace()
#                                             # matrices.update({
#                                             # f'{matrix_name}_Vapprox_withS_{num_Vs}{matrix_type}{"_reversed" if is_reversed else ""}{"_true" if use_true_matrix else ""}_size_{size}{postfix}': [colors[3+i], '-', 's'] for i, num_Vs in enumerate(Vs_list)
#                                             # })
#                                             # matrices.update({
#                                             #     f'{matrix_name}_Vapprox_withS_{num_Vs}{matrix_type2}_size_{size}{postfix}': [colors[2+i], ':', '*'] for i, num_Vs in enumerate([1,10,100])
#                                             # })

#                                             print("Matrices:", matrices.keys())

#                                             # matrix_name = 'temp'
#                                             # figure_dir = 'figures'
#                                             # matrices = [f'{matrix_name}',]
#                                             # colors = plt.cm.rainbow(np.linspace(0, 1, 5))
#                                             # matrices = {f'{matrix_name}': [colors[0], '-']}

#                                             dir_paths = [f"{figure_dir}/{matrix_postfix}/" for matrix_postfix in matrices]
#                                             # dir_path = dir_paths[1]
#                                             # print(dir_paths)
#                                             # raise

#                                             labels = [' '.join(s.split('_')[1:]) for s in matrices]
#                                             label_colors = {k:v[0] for k,v in matrices.items()}
#                                             label_linestyles = {k:v[1] for k,v in matrices.items()}
#                                             label_markers = {k:v[2] for k,v in matrices.items()}
#                                             label_colors = label_colors.values()
#                                             label_linestyles = label_linestyles.values()
#                                             label_markers = label_markers.values()

#                                             last_available_file_number = np.inf
#                                             for dir_path in dir_paths:
#                                                 # Use glob to find files matching the pattern
#                                                 files = glob.glob(os.path.join(dir_path, f'spectrum_data_*.npz'))

#                                                 # Extract the numeric part from each file and convert to an integer
#                                                 file_numbers = sorted([int(os.path.splitext(file)[0].split('_')[-1]) for file in files])
#                                             #     print(dir_path, files)
#                                                 if len(file_numbers) == 0:
#                                                     print("Path", dir_path, "not available")
#                                                     continue
#                                                 last_file_number = file_numbers[-1]
#                                                 if last_file_number < last_available_file_number:
#                                                     last_available_file_number = last_file_number

#                                             if last_available_file_number == np.inf:
#                                                 print("No available file found for", dir_paths[0])
#                                                 continue

#                                             dir_path, label = dir_paths[0], labels[0]
#                                             # Load data first                           
#                                             tr_S = []
#                                             tr_S_quotient = []
#                                             Ss = []
#                                             Ss_quotient = []
#                                             err_mat = []
                                            
#                                             for iteration in range(last_available_file_number+1):
#                                                 file_path = os.path.join(dir_path, f'spectrum_data_{iteration}.npz')
#                                                 try:
#                                                     data = np.load(file_path)
#                                                     Ss.append(data['S'].reshape(1,-1))
#                                                     trace = np.sum(data['S'])
#                                                     tr_S.append(trace)
#                                                     if name_postfix == "":
#                                                         Ss_quotient.append(data['S_quotient'].reshape(1,-1))
#                                                         trace_quotient = np.sum(data['S_quotient'])
#                                                         tr_S_quotient.append(trace_quotient)
#                                                     # print(data['S'], data['S_exact'][:len(data['S'])])
#                                         #             raise
#                                                 except FileNotFoundError:
#                                                     print(f"File not found: {file_path}")
                                            
#                                             limit_S = min(10, len(data['S']))
#                                             # limit_S = len(data['S'])
#                                             # e_i = np.abs(data['S_exact'][:len(data['S'])].sum() - tr_S) / data['S_exact'][:len(data['S'])].sum()
#                                             Ss = np.concatenate(Ss, axis=0)
#                                             tr_S = np.sum(Ss[:, :limit_S], axis=1)
#                                             if name_postfix == "":
#                                                 Ss_quotient = np.concatenate(Ss_quotient, axis=0)
#                                                 tr_S_quotient = np.sum(Ss_quotient[:, :limit_S], axis=1)
#                                                 if plot_S_quotient:
#                                                     tr_S = tr_S_quotient
#                                             e_i = np.abs(data['S_exact'][:limit_S].sum() - tr_S) / data['S_exact'][:limit_S].sum()
#                                             e_i = np.clip(e_i, np.finfo(float).eps, None)
#                                             exact_temp = data['S_exact'][:len(data['S'])]
#                                             # print(Ss[-1])
#                                             print(e_i)
                                            
#                                             data_list.append((e_i, label))
#                                             if smallest_ei > min(e_i):
#                                                 smallest_ei = min(e_i)
                                            
#                                         matrix_size = len(data['S_exact'])

#                                         # plt.semilogy(data['S_exact'][:100])
#                                         # plt.title(f"Lengthscale  {scaling_factor}")
#                                         # plt.ylabel("Eigenvalue")
#                                         # plt.xlabel("Index")
#                                         # plt.savefig(f"figures/{matrix_name}_{scaling_factor}_spectrum.png")

#                                         # plt.close()
#                                         # continue

#                                         if plot_trace_error:
#                                             if not plot_trace_error_only_log:
#                                                 plt.figure(figsize=(12, 6))
#                                                 i = 0
#                                                 for (e_i, label),in data_list:
#                                                     i += 1
#                                                     if e_i[0] == 0:
#                                                         print("Initial Error is already 0!")
#                                                         break
#                                                     # import pdb;pdb.set_trace()
#                                                     # plt.plot(np.arange(last_available_file_number+1), ((e_i)/e_i[0]), label=f'{label}, init err: {e_i[0]}', linestyle=linestyle, marker=marker,
#                                                     #         color=color, alpha=0.7, markevery=i, markersize=12)
#                                                     plt.plot(np.arange(last_available_file_number+1), ((e_i)), label=f'{label}, init err: {e_i[0]}', linestyle=linestyle, marker=marker,
#                                                             color=color, alpha=0.7, markevery=i, markersize=12)

#                                                 # plt.plot(np.arange(last_available_file_number+1), (1-np.arange(last_available_file_number+1)/(last_available_file_number+1)),
#                                                 #         label='1-k/n', linestyle='--', alpha=0.7)

#                                                 plt.ylabel("$e_k\ /\ e_0$")
#                                                 plt.xlabel("Iteration")
#                                                 plt.title(f"Error $e_k\ /\ e_0$ over Iterations, Length Scale: 1e{(scaling_factor):.1f}")
#                                                 plt.legend()
#                                                 plt.grid(True, which='both', linestyle='--', alpha=0.5)
#                                                 plt.tight_layout()
#                                                 # plt.ylim([-0.01, 1.01])
#                                                 #plt.show()
#                                                 plt.savefig(f"figures/err_comparison.png")
#                                                 plt.close()

#                                             plt.figure(figsize=(12, 6))
#                                             i = 0
#                                             markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X']
#                                             for (e_i, label) in data_list:
#                                                 i += 1
#                                                 if e_i[0] == 0:
#                                                     print("Initial Error is already 0!")
#                                                     break
#                                                 # plt.plot(np.arange(last_available_file_number+1), np.log10(np.abs((e_i)/e_i[0])), label=f'{label}', linestyle=linestyle, marker=marker,
#                                                 #         color=color, alpha=0.7, markevery=i, markersize=12)
#                                                 plt.plot(np.arange(last_available_file_number+1), np.log10(np.abs((e_i))), label=f'{label}', markevery=i, marker=markers[i], markersize=12,
#                                                         )
#                                                 # import pdb;pdb.set_trace()
#                                             # plt.plot(np.arange(last_available_file_number+1), np.log10(1-np.arange(last_available_file_number)/last_available_file_number),
#                                             #          label='1-k/n', linestyle='--', alpha=0.7)

#                                             plt.ylabel("log $e_k\ /\ e_0$")
#                                             plt.xlabel("Iteration")
#                                             plt.title(f"(Log) Error $e_k\ /\ e_0$ over Iterations, Length Scale: 1e{(scaling_factor):.1f}")
#                                             plt.legend()
#                                             plt.grid(True, which='both', linestyle='--', alpha=0.5)
#                                             plt.tight_layout()
#                                             #plt.show()

#                                             plt.savefig(f"figures/err_comparison_log.png")
#                                             plt.close()

#                                             # print([x[1] for x in data_list])

