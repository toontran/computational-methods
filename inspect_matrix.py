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
    # assert np.all(s > -1.0 - eps) and np.all(s < 1.0 + eps), "Invalid canonical correlation found" 
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
        
        # TODO: Choose randomly vs. choose 
        for i in range(len(S)):
            S_truncated_Rayleigh = np.dot(Vt[i, window_indices].T, A_csr[window_indices, :] @ Vt[i].T)
            sq_norm_V = np.dot(Vt[i, window_indices].T, Vt[i, window_indices].T)
            S_truncated_Rayleigh_full = np.dot(Vt[i, row_permutation[:end_idx]].T, A_csr[row_permutation[:end_idx], :] @ Vt[i].T)
            sq_norm_V_full = np.dot(Vt[i, row_permutation[:end_idx]].T, Vt[i, row_permutation[:end_idx]].T)
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
            import pdb;pdb.set_trace()
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
    return [p1[p2[i]] for i in range(len(p1))]


def isvd(A_csr, S_exact=None, Vt_exact=None, U_exact=None, 
         window_size=100, k=None,
         num_windows=None, row_permutation=None, name="temp", figure_dir="figures", is_sym_psd=False,
         num_Vs=None, track_U=False, track_discarded=False, with_S=False, reverse=False,
         return_row_order=False, stream_size=None, col_permutation=None, use_true_matrix=False,
         track_reconstruction_error=False):
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
    A_norm = sp_norm(A_csr) # TODO
    inverse_perm = None

    total_S_reduced = 0
    if track_discarded:
        discarded_list = []
    if track_reconstruction_error:
        reconstruction_errors = []
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
        # print("Index:", end_idx, len(row_permutation))
#         import pdb;pdb.set_trace()
        next_window = A_csr[window_indices, :]
        next_window = next_window[:, row_permutation]
        if isinstance(A_csr, csr_matrix):
            next_window = next_window.toarray()
        
        # print(next_window.shape)
        if j == 0:
             # Initial SVD for the first window
            w = window_size
            
            S11, V11 = sp.linalg.eig(next_window[:, :window_size])
            idx = np.argsort(S11)[::-1]
            S11, V11 = S11[idx], V11[:, idx]

            # Initial SVD
            S11_sqrt = np.sqrt(np.abs(S11))
            S11_invsqrt = np.zeros_like(S11)
            mask = S11 > 1e-10 
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
            # import pdb;pdb.set_trace()

            n = j + 1  # Current number of windows being processed

            # Correct eigenvectors for current row perm
            # Inverts the previous permutation, then applies current row permutation
            # print(V.shape, len(row_permutation), len(inverse_perm))
            # import pdb;pdb.set_trace()
            V = V[compose_permutations(inverse_perm, row_permutation), :]  

            # Get new window
            current_window = next_window
            w = window_size
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

            # print("Window after:", prev_window[:, row_permutation][-3:,-5:], (V[(j-1)*window_size:(j)*window_size,:] @ np.diag(S) @ V[:,:].T)[-3:,-5:])
            # print("Norm after:", np.linalg.norm(prev_window[:, row_permutation] - V[(j-1)*window_size:(j)*window_size,:] @ np.diag(S) @ V[:,:].T))
            
            # import pdb;pdb.set_trace()

            # Create operator for eigendecomposition
            if use_true_matrix:
                A11_op = make_operator_true(n, w, A_csr, N)
            else:
                A11_op = make_operator(current_window, N, n, w, V, S)
            keep = len(idx) # for now
            S11, V11 = sp.sparse.linalg.eigs(A11_op, k=keep, which='LM', ncv=n*w - 1)

            if end_idx == m:
                # If last window, stop
                V = V11
                S = S11
#                 print(f"Eigenvalues: {S}")
#                 print("Exact:", S_exact)
#                 print(f"V shape: {V.shape}")
                
            else:

                # Prepare for SVD
                S11_sqrt = np.sqrt(np.abs(S11))
                S11_invsqrt = np.zeros_like(S11)
                mask = np.abs(S11) > 1e-10
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
            # import pdb;pdb.set_trace()

        print(S[:10])
        print(S_exact[:10])

        print("V.shape, S.shape:", V.shape, S.shape)

        # invert row perm, compose with new row perm
        inverse_perm = inverse_permutation(row_permutation)
        temp = compose_permutations(row_permutation, inverse_permutation(row_permutation))
        # print("Inverse is correct:", [i for i in range(len(temp)) if i != temp[i]])
        print("Window before:", next_window[-3:,-5:], (V[j*window_size:(j+1)*window_size,:] @ np.diag(S) @ V[:,:].T)[-3:,-5:])
        print("Norm before:", np.linalg.norm(next_window - V[j*window_size:(j+1)*window_size,:] @ np.diag(S) @ V[:,:].T))
        # prev_window = A_csr[window_indices, :]
        # prev_permutation = row_permutation
        # import pdb;pdb.set_trace()
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
    
    ret = [S, Vt]
    np.savez(os.path.join(dir_path, f'row_order_final.npz'),
             row_permutation=row_permutation,)
#     if track_U:
#         ret.append(U)
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

import seaborn as sns

if __name__ == "__main__":
    # Get the name
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <matrix_name>")
        print("Example: python script_name.py HB/west0067")
        sys.exit(1)
    
    matrix_name = sys.argv[1]
    matrix_postfix = matrix_name.split('/')[-1]
    figure_dir = "output"
    url = f'https://suitesparse-collection-website.herokuapp.com/MM/{matrix_name}.tar.gz'

    if figure_dir and not os.path.exists(figure_dir):
        print("Making directory:", figure_dir)
        os.makedirs(figure_dir)

    if "hyperboloid" in matrix_name:
        _, num_points, gamma = matrix_name.split("_")
        num_points, gamma = int(num_points), float(gamma)
        # num_points = 10000
        gamma = 10**gamma
        a, b, c, d = 1, 2, 3, 4
        points = sample_4d_hyperboloid(num_points, a, b, c, d)
        kernel = StreamingRBFKernel(points, gamma=gamma)
        A_csr = kernel[:,:]
        title = ""
        A_is_sym_psd = True
    elif "kronecker_graph" in matrix_name:
        _, _, scale, edgefactor = matrix_name.split("_")
        scale, edgefactor = int(scale), float(edgefactor)
        kernel = StreamingKroneckerGraph(scale, edgefactor=edgefactor)
        A_csr = kernel[:,:]
        title = ""
        A_is_sym_psd = True
    elif "bad_matrix" in matrix_name:
        _, _, scale = matrix_name.split("_")
        scale = int(scale)
        n = 2**scale
        start=10**5 
        end=0.1
        rate = -np.log(end / start) / (n - 1)

        # Generate the singular values
        singular_values = start * np.exp(-rate * np.arange(n))
        kernel = StreamingMatrix(n, singular_values)
        A_csr = kernel[:,:]
        title = ""
        A_is_sym_psd = True
    elif "synthetic" in matrix_name:
        np.random.seed(10)
        _, N, rank = matrix_name.split("_")  
        N, rank = int(N), int(rank)

        # Generate test matrix
        Q = np.random.randn(N, N)
        Q, _ = np.linalg.qr(Q)
        S = np.linspace(1, 15, N)**2
        Q = Q[:, :rank]
        S = S[:rank]
        title = ""
        A_csr = Q @ np.diag(S) @ Q.T
        A_is_sym_psd = True
#         A_csr = (A_csr + A_csr.T) / 2
    elif "kernel_random" in matrix_name:
        np.random.seed(10)

        num_points, gamma = matrix_name.split("_")[-2:]
        num_points, gamma = int(num_points), float(gamma)
        # num_points = 10000
        gamma = 10**gamma
        a, b, c, d = 1, 2, 3, 4
        points = sample_4d_hyperboloid(num_points, a, b, c, d)
        kernel = StreamingRBFKernel(points, gamma=gamma)
        A_csr = kernel[:,:]
        start_time = time.time()
        print("Computing exact spectrum to generate random matrix...")
        S_exact = sp.linalg.svd(A_csr, lapack_driver="gesdd", compute_uv=False)
        exact_time = time.time() - start_time
        print("Exact:", exact_time)

        N = num_points
        # Generate test matrix
        Q = np.random.randn(N, N)
        Q, _ = np.linalg.qr(Q)
        S = S_exact
        title = ""
        A_csr = Q @ np.diag(S) @ Q.T
        A_is_sym_psd = True
    else:
        # Download and read the matrix
        A = download_and_read_matrix(url)
        print(f"A's shape: {A.shape}, nonzeros: {A.nnz}")
        
        # Convert to CSR format for efficient operations
        A_csr = sparse.csr_matrix(A)
    
        properties = get_matrix_properties(matrix_name)
        print(properties)
        
        A_is_sym_psd = properties['symmetric'] and properties['positive definite']
        
        title = f"\n{matrix_name}: {properties['kind']}"
        title += f"\nSymmetric: {properties['symmetric']}, PD: {properties['positive definite']}"
        if isinstance(properties['Minimum Singular Value'], float) or (properties['Minimum Singular Value'].replace('.','',1).isdigit() and\
            properties['Minimum Singular Value'].count('.') < 2):
            title += f"\nMinimum Singular Value: {properties['Minimum Singular Value']:.2e}"
    
        if isinstance(properties['condition number'], float) or (properties['condition number'].replace('.','',1).isdigit() and\
            properties['condition number'].count('.') < 2):
            title += f"\nCondition number: {properties['condition number']:.2e}"
        
    
    plt.figure(figsize=(12, 8))
    fig, ax = matspy.spy_to_mpl(A_csr)
    title = ax.set_title(title,
         loc='center', wrap=True)
    fig.tight_layout()
    #plt.show()
    fig.savefig(f'{figure_dir}/{matrix_postfix}.png', dpi=100)
    plt.close(fig) 
    
    if "hyperboloid" in matrix_name:
        plt.figure(figsize=(10, 8))
        im = plt.imshow(A_csr, 
                   cmap='viridis',
                   aspect='equal',
                   origin='upper',  # to match the orientation in your image
                   interpolation='nearest')
        plt.title(matrix_name)
        plt.xlabel("Samples")
        plt.ylabel("Samples")
        plt.savefig(f'{figure_dir}/{matrix_postfix}_heatmap.png', dpi=100)
#     raise

    if "kernel_random" in matrix_name:
        U_exact, Vt_exact = Q, Q.T
        np.savez(f'{figure_dir}/US_exact1000_{matrix_postfix}.npz', U=U_exact[:,:1000], S=S_exact[:1000])
    elif A_csr.shape[1] < 5e4:
        # Compute exact SVD (full
        # start_time = time.time()
        # U_exact, S_exact, Vt_exact = sp.linalg.svd(A_csr.todense(), lapack_driver="gesdd")
        # exact_time = time.time() - start_time
        # Compute exact SVD (full
        start_time = time.time()
        print("Computing exact SVD...")
        if isinstance(A_csr, csr_matrix):
            U_exact, S_exact, Vt_exact = sp.linalg.svd(A_csr.todense(), lapack_driver="gesdd")
        elif isinstance(A_csr, StreamingRBFKernel) or isinstance(A_csr, StreamingKroneckerGraph):
            U_exact, S_exact, Vt_exact = sp.linalg.svd(A_csr[:,:], lapack_driver="gesdd")
        else: 
             U_exact, S_exact, Vt_exact = sp.linalg.svd(A_csr, lapack_driver="gesdd")
        np.savez(f'{figure_dir}/US_exact1000_{matrix_postfix}.npz', U=U_exact[:,:1000], S=S_exact[:1000])
        exact_time = time.time() - start_time
        print("Exact:", exact_time)
    else:
        U_exact, S_exact, Vt_exact = None, None, None
#     raise

    if isinstance(A_csr, csr_matrix):
        A_squared = A_csr.copy()
        A_squared.data **= 2
    else:
        A_squared = A_csr ** 2
    weights = np.asarray(np.sqrt(np.sum(A_squared, axis=1))).reshape(-1)
    weights = np.array(weights) / np.sum(weights)
    
    np.random.seed(42)
#     np.random.permutation(len(weights)) # since we're commenting random_uniform(_1)
    permutations = {
        "original": None,
        # "reversed_original": np.arange(len(weights))[::-1],
        "decreasing_norm": np.argsort(weights)[::-1],
        # "increasing_norm": np.argsort(weights),
        "random_uniform": np.random.permutation(len(weights)),
        "random_uniform_2": np.random.permutation(len(weights)),
        "random_uniform_3": np.random.permutation(len(weights)),
#         "random_uniform_4": np.random.permutation(len(weights)),
#         "random_uniform_5": np.random.permutation(len(weights)),
        # "random_weighted_norm": np.random.choice(len(weights), size=len(weights), replace=False, p=weights),
        # "decreasing_exactV_norm": np.argsort(np.sum(Vt_exact**2, axis=0)).reshape(-1)[::-1],
        # "increasing_exactV_norm": np.argsort(np.sum(Vt_exact**2, axis=0)).reshape(-1),
        # "decreasing_exactV_norm_100": np.argsort(np.sum(Vt_exact[:100, :]**2, axis=0)).reshape(-1)[::-1],
        # "increasing_exactV_norm_100": np.argsort(np.sum(Vt_exact[:100, :]**2, axis=0)).reshape(-1),
        # "decreasing_exactV_norm_10": np.argsort(np.sum(Vt_exact[:10, :]**2, axis=0)).reshape(-1)[::-1],
        # "increasing_exactV_norm_10": np.argsort(np.sum(Vt_exact[:10, :]**2, axis=0)).reshape(-1),
        # "decreasing_exactV_norm_1": np.argsort(np.sum(Vt_exact[:1, :]**2, axis=0)).reshape(-1)[::-1],
        # "increasing_exactV_norm_1": np.argsort(np.sum(Vt_exact[:1, :]**2, axis=0)).reshape(-1),
    }
#     import pdb;pdb.set_trace()
    if "hyperboloid" in matrix_name or "synthetic" in matrix_name or "kernel" in matrix_name:
        np.random.seed(42)
        data = np.load("perms/row_order_final.npz")
        perm = data["row_permutation"]
        
        random_uniform_perms = [np.random.permutation(len(weights)) for i in range(5)]
        permutations = {
            # "original": None,
            # "decreasing_norm": np.argsort(weights)[::-1],
#             "increasing_norm": np.argsort(weights)[::1],
            "random_uniform": random_uniform_perms[0],
            # "manual_perm": perm,
            # "manual_perm_reverse": perm[::-1],
#             "random_uniform_2": random_uniform_perms[1],
            # "random_uniform_3": random_uniform_perms[2],
#             "random_uniform_4": random_uniform_perms[3],
#             "random_uniform_5": random_uniform_perms[4],
        }
        
#         Need to plot eigenvectors of matrix decreasing norm
        import scipy.io
        # for perm_name in permutations:
        #     perm = permutations[perm_name]
        #     permuted_matrix = A_csr[perm, :]
        #     permuted_matrix = permuted_matrix[:, perm]
        #     filename = f'perms/matrix_data_{matrix_name}_{perm_name}.mat'
        #     scipy.io.savemat(filename, {'matrix': permuted_matrix})
        #     print(f"Matrix saved in", filename)

        # filename = f'perms/matrix_data_{matrix_name}.mat'
        # scipy.io.savemat(filename, {'matrix': A_csr})
        # print(f"Matrix saved in", filename)
        filename = f'perms/matrix_data_{matrix_name}.mtx'
        scipy.io.mmwrite(filename, A_csr)
        print(f"Matrix saved in", filename)
        if len(S_exact) < 100:
            print(S_exact)

        # filename = f'spectrums/S_exact_{matrix_postfix}.npz'
        # np.savez(filename, S=S_exact)
        # print("Saved,", filename)
        
        
