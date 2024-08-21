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
    #plt.show()
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

def soft_thresholding_Ghashami(S):
    # Assuming S is descending order
    return np.sqrt(S**2 - S[-1]**2) 

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
    #plt.show()
    plt.close('all')


    plt.figure(figsize=(10, 6))
    plt.plot(np.abs(S - S_exact[:len(S)]) / A_norm, label='$\\left\\|\\frac{S-S_\\text{exact}}{\|A\|_F}\\right\\|$')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.title(f'Difference between Approximated and Exact Spectra iteration {iteration}')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(dir_path + f'/diffspec_window_{iteration+1}.png', bbox_inches='tight', pad_inches=0.5)
    #plt.show()
    plt.close('all')

def plot_residuals(A_csr, S, Vt, S_exact, Vt_exact, U_exact, 
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
        #plt.show()
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
    #plt.show()
    plt.close('all')

    # plt.figure(figsize=(10, 6))
    # plt.semilogy(list(matches.keys()), [matches[i][3] for i in matches])
    # plt.xlabel('Index')
    # plt.ylabel('Angles')
    # plt.title('')
    # plt.legend()
    # plt.grid(True)
    # #plt.show()
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
    

def plot_canonical_angles(Vt, Vt_exact, iteration, dir_path):
    # Compute the singular values of Q1.T @ Q2
    s = np.linalg.svd(Vt @ Vt_exact[:Vt.shape[0], :].T, compute_uv=False)
    
    # Compute the angles in radians
    eps = 1e-6
    assert np.all(s > -1.0 - eps) and np.all(s < 1.0 + eps), "Invalid canonical correlation found" 
    angles = np.arccos(np.clip(s, -1.0, 1.0))
    print("Subspace angle 2:", max(angles), np.mean(angles))
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
    
    # Add annotations with adjusted positions
    add_annotation(0.6, min_val+0.1, f'Min: {min_val:.2f}', offset=5, va='bottom')
    add_annotation(0.6, max_val-0.1, f'Max: {max_val:.2f}', offset=5, va='top')
    add_annotation(1.1, q1, f'Q1: {q1:.2f}', offset=5)
    add_annotation(1.1+0.1, median_val, f'Median: {median_val:.2f}', offset=5)
    add_annotation(1.1, q3, f'Q3: {q3:.2f}', offset=5)
    add_annotation(1.1+0.1, lower_whisker, f'Lower whisker: {lower_whisker:.2f}', offset=5, va='bottom')
    add_annotation(1.1+0.1, upper_whisker, f'Upper whisker: {upper_whisker:.2f}', offset=5, va='top')
    
    # Add outlier information
    if len(fliers) > 0:
        outlier_min, outlier_max = np.min(fliers), np.max(fliers)
        add_annotation(0.78, outlier_min, f'Min outlier: {outlier_min:.2f}', offset=5, va='bottom', color='red')
        add_annotation(0.78, outlier_max, f'Max outlier: {outlier_max:.2f}', offset=5, va='top', color='red')
        add_annotation(0.78, (outlier_min+outlier_max)/2, f'Number of outliers: {len(fliers)}', offset=5, va='center', color='red')
    
    # Customize the plot
    ax.set_title('Boxplot with Non-Overlapping Annotations')
    ax.set_ylabel('Values')
    ax.set_xlim(0.5, 1.5)  # Adjust x-axis limits to make room for annotations
    plt.grid()
    plt.tight_layout()
    plt.savefig(dir_path + f'/angles_window_{iteration+1}.png', bbox_inches='tight', pad_inches=0.5)
    #plt.show()
    plt.close('all')


def isvd(A_csr, S_exact, Vt_exact, U_exact, 
         num_windows=10, row_permutation=None, k=None, name="temp", figure_dir="figures", is_sym_psd=False):
    global Vt
    m, n = A_csr.shape
    W = num_windows  # number of windows (columns in this case)
    l = m // W  # window size
    k = k if k and k < l else l-1 # Number of singular values/vectors to compute
    r = min(k, m, l)

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
    for j in range(W):
        print("Window:", j+1)
        
        # Calculate the start and end indices for this window
        start_idx = j * l
        end_idx = min((j + 1) * l, m)
        
        # Extract the next window
        window_indices = row_permutation[start_idx:end_idx]
        print("Index:", end_idx, len(row_permutation))
        next_window = A_csr[window_indices, :]
        if isinstance(A_csr, csr_matrix):
            next_window = next_window.toarray()
        
        # print(next_window.shape)

        if j == 0:
             # Initial SVD for the first window
            
            # Reverse the order to get largest singular values first
            # _, S, Vt = svds(next_window, k=r)
            # S = S[::-1]
            # Vt = Vt[::-1, :]
            
            _, S, Vt = sp.linalg.svd(next_window, lapack_driver="gesdd")
            print(len(S))
            S = S[:r]
            Vt = Vt[:r, :]           
            
            B = S.reshape(-1, 1) * Vt

        else:
        
            # Concatenate B[j-1] and the next window
            combined = np.concatenate((B, next_window), axis=0)
            
            # Perform SVD on the combined matrix
            # Reverse the order to get largest singular values first
            # _, S, Vt = svds(combined, k=r)
            # S = S[::-1]
            # Vt = Vt[::-1, :]
            
            _, S, Vt = sp.linalg.svd(combined, lapack_driver="gesdd")
            print(len(S))
            S = S[:r]
            Vt = Vt[:r, :]
            
            # Optional: Apply soft thresholding to singular values
            # S = soft_thresholding(S)
            # total_S_reduced += S[-1]
            # S = soft_thresholding_Ghashami(S)

            # Update B
            B = S.reshape(-1, 1) * Vt
    
        # Plot
        plot_spectrum_comparison(S, S_exact, 
                                 A_norm, name, j, dir_path)
        plot_residuals(A_csr, S, Vt, S_exact, Vt_exact, U_exact, 
                       A_norm, name, j, dir_path, is_sym_psd) 
        plot_canonical_angles(Vt, Vt_exact, 
                              j, dir_path)
        print("Reconstruction quality:", np.linalg.norm(Vt - Vt_exact[:Vt.shape[0], :], 'fro'))
        print("Relative error in S:", np.linalg.norm(S - S_exact[:Vt.shape[0]]) / A_norm)
        # X = np.linalg.pinv(Vt_exact[:Vt.shape[0],:].T) @ Vt.T 
        # Vt_reconstructed = Vt_exact[:Vt.shape[0],:].T @ X
        # print("Reconstruct Vt from Vt_exact:", np.linalg.norm(Vt.T - Vt_reconstructed, 'fro'))
        # print("Projection F-norm error:", np.linalg.norm(Vt.T @ Vt - Vt_exact[:Vt.shape[0], :].T @ Vt_exact[:Vt.shape[0], :], 'fro'))
        # print("Trace correlation", np.trace(Vt @ Vt_exact[:Vt.shape[0], :].T @ Vt_exact[:Vt.shape[0], :] @ Vt.T) / min(Vt.T.shape[1], Vt_exact[:Vt.shape[0], :].T.shape[1]))
    return S, Vt

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
    
    return normalized_matrix

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
                  "matrix norm", "type", "kind"]
    d = {}
    for p in properties:
        d[p] = find_property(p)
    return d

if __name__ == "__main__":
    # Get the name
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <matrix_name>")
        print("Example: python script_name.py HB/west0067")
        sys.exit(1)
    
    matrix_name = sys.argv[1]
    matrix_postfix = matrix_name.split('/')[-1]
    figure_dir = "figures"
    url = f'https://suitesparse-collection-website.herokuapp.com/MM/{matrix_name}.tar.gz'

    if figure_dir and not os.path.exists(figure_dir):
        print("Making directory:", figure_dir)
        os.makedirs(figure_dir)
    
    # Download and read the matrix
    A = download_and_read_matrix(url)
    print(f"A's shape: {A.shape}, nonzeros: {A.nnz}")
    
    # Convert to CSR format for efficient operations
    A_csr = sparse.csr_matrix(A)

    properties = get_matrix_properties(matrix_name)
    A_is_sym_psd = properties['symmetric'] and properties['positive definite']

    plt.figure(figsize=(12, 8))
    fig, ax = matspy.spy_to_mpl(A_csr)
    title = ax.set_title(f"\n{matrix_name}: {properties['kind']}\nSymmetric: {properties['symmetric']}, PD: {properties['positive definite']}\nMinimum Singular Value: {properties['Minimum Singular Value']:.2e}\nCondition number: {properties['condition number']:.2e}",
         loc='center', wrap=True)
    fig.tight_layout()
    #plt.show()
    fig.savefig(f'{figure_dir}/{matrix_postfix}.png', dpi=100)
    plt.close(fig) 

    # Compute exact SVD (full
    start_time = time.time()
    U_exact, S_exact, Vt_exact = sp.linalg.svd(A_csr.todense(), lapack_driver="gesdd")
    exact_time = time.time() - start_time
    print("Exact:", exact_time)

    A_squared = A_csr.copy()
    A_squared.data **= 2
    weights = np.asarray(np.sqrt(np.sum(A_squared, axis=1))).reshape(-1)
    weights = np.array(weights) / np.sum(weights)
    permutations = {
        "original": None,
        "reversed_original": np.arange(len(weights)),
        "decreasing_norm": np.argsort(weights)[::-1],
        "increasing_norm": np.argsort(weights)[::-1],
        "random_uniform": np.random.permutation(len(weights)),
        "random_weighted_norm": np.random.choice(len(weights), size=len(weights), replace=False, p=weights),
        "decreasing_exactV_norm": np.argsort(np.sum(Vt_exact**2, axis=0)).reshape(-1)[::-1],
        "increasing_exactV_norm": np.argsort(np.sum(Vt_exact**2, axis=0)).reshape(-1),
    }
    for row_permutation in permutations:
        S, Vt = isvd(A_csr, 
                     S_exact, Vt_exact, U_exact,
                     row_permutation=permutations[row_permutation], 
                     name=matrix_postfix + "_" + row_permutation,
                     figure_dir=figure_dir,
                     is_sym_psd=A_is_sym_psd)

    
        
    A_normalized = normalize_csr_matrix_rows(A_csr)
    start_time = time.time()
    U_exact, S_exact, Vt_exact = sp.linalg.svd(A_normalized.todense(), lapack_driver="gesdd")
    exact_time = time.time() - start_time
    print("Exact:", exact_time)

    A_squared = A_normalized.copy()
    A_squared.data **= 2
    weights = np.asarray(np.sqrt(np.sum(A_squared, axis=1))).reshape(-1)
    permutations = {
        "original": None,
        "reversed_original": np.arange(len(weights)),
        "random_uniform": np.random.permutation(len(weights)),
        "decreasing_exactV_norm": np.argsort(np.sum(Vt_exact**2, axis=0)).reshape(-1)[::-1],
        "increasing_exactV_norm": np.argsort(np.sum(Vt_exact**2, axis=0)).reshape(-1),
    }
    for row_permutation in permutations:
        S, Vt = isvd(A_normalized, 
                     S_exact, Vt_exact, U_exact,
                     row_permutation=permutations[row_permutation], 
                     name=matrix_postfix + "_" + "balance" + "_" + row_permutation,
                     figure_dir=figure_dir,
                     is_sym_psd=A_is_sym_psd)



