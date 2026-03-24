import requests
import io
import tarfile
import os
import sys
import time
import re

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from bs4 import BeautifulSoup
import matspy

# from scipy.sparse.linalg import svds
import scipy.sparse as sparse
# from scipy.linalg import orthogonal_procrustes, subspace_angles, matrix_balance
from scipy.sparse._csr import csr_matrix


import matplotlib
matplotlib.use('Agg')

from utils import *

import faulthandler
import signal
import sys

faulthandler.enable(file=sys.stderr, all_threads=True)
faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)


def find_sparsity_thresholds(matrix, target_sparsities=[0.90, 0.95, 0.99, 0.999], 
                           return_sparse_matrices=False, plot=False):
    """
    Find thresholds to achieve target sparsity levels in a matrix.
    
    Parameters:
    -----------
    matrix : numpy.ndarray or scipy.sparse matrix
        Input matrix
    target_sparsities : list of float
        Target sparsity levels (e.g., [0.90, 0.95, 0.99, 0.999])
    return_sparse_matrices : bool
        If True, return the sparsified matrices as well
    plot : bool
        If True, plot the sparsity vs threshold relationship
    
    Returns:
    --------
    dict : Dictionary with target sparsities as keys and threshold info as values
    """
    
    # Convert to dense numpy array if sparse
    if sp.sparse.issparse(matrix):
        dense_matrix = matrix.toarray()
    else:
        dense_matrix = matrix.copy()
    
    # Get absolute values for threshold calculation
    abs_matrix = np.abs(dense_matrix)
    
    # Flatten and sort values (excluding zeros)
    nonzero_values = abs_matrix[abs_matrix != 0]
    sorted_values = np.sort(nonzero_values)
    
    total_elements = matrix.size
    current_nonzeros = len(nonzero_values)
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Total elements: {total_elements}")
    print(f"Current non-zero elements: {current_nonzeros}")
    print(f"Current sparsity: {1 - current_nonzeros/total_elements:.4f}")
    print("-" * 50)
    
    results = {}
    sparse_matrices = {}
    
    for target_sparsity in target_sparsities:
        # Calculate how many elements should remain non-zero
        target_nonzeros = int(total_elements * (1 - target_sparsity))
        
        if target_nonzeros >= current_nonzeros:
            # Target sparsity is already achieved or impossible
            threshold = 0.0
            actual_sparsity = 1 - current_nonzeros/total_elements
            sparsified = dense_matrix.copy()
        else:
            # Find threshold: keep only the largest target_nonzeros elements
            if target_nonzeros == 0:
                threshold = np.inf
                actual_sparsity = 1.0
                sparsified = np.zeros_like(dense_matrix)
            else:
                # Get the threshold as the (target_nonzeros)th largest value
                threshold_idx = len(sorted_values) - target_nonzeros
                threshold = sorted_values[threshold_idx] if threshold_idx < len(sorted_values) else sorted_values[-1]
                
                # Apply threshold
                sparsified = dense_matrix.copy()
                sparsified[abs_matrix < threshold] = 0
                
                # Calculate actual sparsity achieved
                actual_nonzeros = np.count_nonzero(sparsified)
                actual_sparsity = 1 - actual_nonzeros/total_elements
        
        results[target_sparsity] = {
            'threshold': threshold,
            'actual_sparsity': actual_sparsity,
            'target_nonzeros': target_nonzeros,
            'actual_nonzeros': np.count_nonzero(sparsified),
            'sparsity_error': abs(actual_sparsity - target_sparsity)
        }
        
        if return_sparse_matrices:
            sparse_matrices[target_sparsity] = sp.sparse.csr_matrix(sparsified)
        
        print(f"Target sparsity: {target_sparsity:.1%}")
        print(f"  Threshold: {threshold:.6f}")
        print(f"  Actual sparsity: {actual_sparsity:.4f}")
        print(f"  Non-zero elements: {np.count_nonzero(sparsified)}")
        print()
    
    if plot:
        plot_sparsity_analysis(dense_matrix, results)
    
    if return_sparse_matrices:
        return results, sparse_matrices
    else:
        return results

def plot_sparsity_analysis(matrix, results):
    """Plot sparsity vs threshold relationship"""
    abs_matrix = np.abs(matrix)
    nonzero_values = abs_matrix[abs_matrix != 0]
    sorted_values = np.sort(nonzero_values)
    
    # Create threshold range
    thresholds = np.logspace(np.log10(sorted_values[0]), np.log10(sorted_values[-1]), 100)
    sparsities = []
    
    total_elements = matrix.size
    
    for thresh in thresholds:
        remaining = np.sum(abs_matrix >= thresh)
        sparsity = 1 - remaining / total_elements
        sparsities.append(sparsity)
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(thresholds, sparsities, 'b-', linewidth=2, label='Sparsity curve')
    
    # Plot target points
    for target_sparsity, info in results.items():
        if info['threshold'] > 0 and info['threshold'] != np.inf:
            plt.semilogx(info['threshold'], info['actual_sparsity'], 'ro', 
                        markersize=8, label=f"{target_sparsity:.1%} target")
    
    plt.xlabel('Threshold Value')
    plt.ylabel('Sparsity')
    plt.title('Sparsity vs Threshold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def apply_threshold(matrix, threshold):
    """Apply threshold to create sparse matrix"""
    if sp.sparse.issparse(matrix):
        matrix = matrix.toarray()
    
    sparsified = matrix.copy()
    sparsified[np.abs(sparsified) < threshold] = 0
    return sp.sparse.csr_matrix(sparsified)



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


if __name__ == "__main__":
    # Require at least 2 arguments, allow an optional 3rd along with 4th
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print("Usage: python main.py <matrix_name> <method> [stream_size] [k]")
        print("Example: python main.py HB/west0067 isvd 500 100")
        sys.exit(1)

    matrix_name = sys.argv[1]
    method = sys.argv[2]

    # Optional parameter
    stream_size = int(sys.argv[3]) if len(sys.argv) >= 4 and sys.argv[3].isdigit() else None
    k = int(sys.argv[4]) if len(sys.argv) == 5 and sys.argv[4].isdigit() else None

    matrix_postfix = matrix_name.split('/')[-1]
    output_dir = "output"
    cache_dir = "cache"
    url = f'https://suitesparse-collection-website.herokuapp.com/MM/{matrix_name}.tar.gz'
    save_in_text = True
    fixed_rank = True

    if output_dir and not os.path.exists(output_dir):
        print("Making directory:", output_dir)
        os.makedirs(output_dir)

    if "hyperboloid" in matrix_name and (not "sparsify" in matrix_name):
        temp = matrix_name.split("_")
        if len(temp) == 3:
            _, num_points, lengthscale = temp
            kernel_noise_std, point_noise_std = 0.0, 0.0
        elif len(temp) == 4:
            _, num_points, lengthscale, noise = temp
            noise = float(noise)
            kernel_noise_std, point_noise_std = noise, noise
        elif len(temp) == 5:
            _, num_points, lengthscale, kernel_noise_std, point_noise_std = temp
            kernel_noise_std, point_noise_std = float(kernel_noise_std), float(point_noise_std)
        else:
            raise Exception("Input incorrect")
        num_points, lengthscale = int(num_points), float(lengthscale)
        
        # num_points = 10000
        lengthscale = lengthscale
        a, b, c, d = 1, 2, 3, 4
        print("Sampling")
        points = sample_4d_hyperboloid(num_points, a, b, c, d)
        print("Making kernel")
        kernel = StreamingRBFKernel(points, lengthscale=lengthscale, 
                                    kernel_noise_std=kernel_noise_std,
                                    point_noise_std=point_noise_std,
                                    cache_dir=os.path.join("cache", matrix_name, "kernel_cache"),
                                    verbose=True,)
        if num_points < 5e4:
            A_csr = kernel[:,:]
        else:
            kernel.precompute_blocks(overwrite=False)
            A_csr = kernel
        A_csr = kernel.to_linear_operator()
        title = ""
        A_is_sym_psd = True
        print("Kernel matrix done.")
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
        # input (you can change this)
        _, N, rank = matrix_name.split("_")  
        N, rank = int(N), int(rank)

        # Generate test matrix
        Q = np.random.randn(N, N)
        Q, _ = np.linalg.qr(Q)
        S = np.linspace(1, 15, N)**2
        Q = Q[:, :rank]
        S = S[:rank]

        # Must be defined
        title = ""
        A_csr = Q @ np.diag(S) @ Q.T
        A_is_sym_psd = True
    elif "bad_case1" in matrix_name:
        np.random.seed(10)

        _, _, N = matrix_name.split("_")
        N, rank = int(N), k if k is not None else 8
        if fixed_rank:
            rank = 8

        # Top r vectors: spread
        U_top = np.random.randn(N, rank)
        U_top, _ = np.linalg.qr(U_top)

        # Tail: identity-like directions
        tail_dim = N - rank
        U_tail_init = np.eye(N)[:, rank:]

        # Remove projection onto top space
        U_tail = U_tail_init - U_top @ (U_top.T @ U_tail_init)
        U_tail, _ = np.linalg.qr(U_tail)

        U = np.concatenate([U_top, U_tail], axis=1)

        # spectrum
        beta_top = 0.5
        alpha_tail = 0.0145

        sigma_top = np.arange(1, rank + 1)**(-beta_top)
        sigma_tail = np.arange(1, tail_dim + 1)**(-alpha_tail)
        sigma_tail = sigma_tail / sigma_tail[0]

        S = np.concatenate([sigma_top, sigma_tail])**2

        title = ""
        A_csr = U @ np.diag(S) @ U.T
        A_is_sym_psd = True
    elif "bad_case2" in matrix_name:
        np.random.seed(10)

        _, _, N = matrix_name.split("_")
        N, rank = int(N), k if k is not None else 8
        if fixed_rank:
            rank = 8

        # Concentrated top block (Hadamard-style)
        if rank == 1:
            Hdim = 2
        else:
            Hdim = rank

        from scipy.linalg import hadamard
        H = hadamard(Hdim).astype(float) / np.sqrt(Hdim)

        U_top = np.zeros((N, rank))
        U_top[:Hdim, :rank] = H[:, :rank]

        # Tail: spread random vectors
        tail_dim = N - rank
        Q = np.random.randn(N, tail_dim)
        Q = Q - U_top @ (U_top.T @ Q)
        U_tail, _ = np.linalg.qr(Q)

        U = np.concatenate([U_top, U_tail], axis=1)

        # spectrum
        beta_top = 0.5
        alpha_tail = 0.0145

        sigma_top = np.arange(1, rank + 1)**(-beta_top)
        sigma_tail = np.arange(1, tail_dim + 1)**(-alpha_tail)
        sigma_tail = sigma_tail / sigma_tail[0]

        S = np.concatenate([sigma_top, sigma_tail])**2

        title = ""
        A_csr = U @ np.diag(S) @ U.T
        A_is_sym_psd = True
    elif "bad_case3" in matrix_name:
        np.random.seed(10)

        _, _, N = matrix_name.split("_")
        N, rank = int(N), k 
        if fixed_rank:
            rank = 8

        # Concentrated top block (Hadamard-style)
        if rank == 1:
            Hdim = 2
        else:
            Hdim = rank

        from scipy.linalg import hadamard
        H = hadamard(Hdim).astype(float) / np.sqrt(Hdim)

        U_top = np.zeros((N, rank))
        U_top[:Hdim, :rank] = H[:, :rank]

        # Tail: identity-like
        tail_dim = N - rank
        U_tail_init = np.eye(N)[:, rank:]

        U_tail = U_tail_init - U_top @ (U_top.T @ U_tail_init)
        U_tail, _ = np.linalg.qr(U_tail)

        U = np.concatenate([U_top, U_tail], axis=1)

        # spectrum
        beta_top = 0.5
        alpha_tail = 0.0145

        sigma_top = np.arange(1, rank + 1)**(-beta_top)
        sigma_tail = np.arange(1, tail_dim + 1)**(-alpha_tail)
        sigma_tail = sigma_tail / sigma_tail[0]

        S = np.concatenate([sigma_top, sigma_tail])**2

        title = ""
        A_csr = U @ np.diag(S) @ U.T
        A_is_sym_psd = True
    elif "test_coherence" in matrix_name:
        np.random.seed(10)
        _, _, num_points, lengthscale = matrix_name.split("_")[:4]  
        num_points, lengthscale = int(num_points), float(lengthscale)
        first_few_ev_concentration = matrix_name.split("_")[4:]  

        from scipy.stats import entropy
        def create_orthonormal_matrix(n, high_conc=3, med_conc=3):
            # Total special columns
            special_cols = high_conc + med_conc
            
            # Much more distinct concentration levels
            high_density = max(1, n // 500)     # ~2% non-zero elements
            med_density = max(1, n // 50)       # ~12.5% non-zero elements

            # high_density = max(1, n // 50)     # ~2% non-zero elements
            # med_density = max(1, n // 4)       # ~12.5% non-zero elements
            
            # Initialize matrix
            concentrated = np.zeros((n, n))
            
            # Generate high concentration columns (very sparse)
            for i in range(high_conc):
                v = np.zeros(n)
                positions = np.random.choice(n, high_density, replace=False)
                v[positions] = np.random.normal(0, 1, high_density)
                concentrated[:, i] = v
            
            # Generate medium concentration columns
            for i in range(high_conc, high_conc + med_conc):
                v = np.zeros(n)
                positions = np.random.choice(n, med_density, replace=False)
                v[positions] = np.random.normal(0, 1, med_density)
                concentrated[:, i] = v
            
            # The rest with low concentration
            remaining = np.random.normal(0, 1, (n, n-special_cols))
            concentrated[:, special_cols:] = remaining
            Q, _ = np.linalg.qr(concentrated[:, :], mode='complete')    
            return Q

        num_points = min(10000, num_points)
        n = num_points
        high_num, med_num = 3, 3
        Q = create_orthonormal_matrix(n, high_conc=high_num, med_conc=med_num)

        # Calculate entropy for each column
        entropies = []
        for i in range(n):
            col = Q[:, i]
            prob_dist = np.abs(col)
            col_entropy = entropy(prob_dist)  # higher entropy means more spread out, lower means more concentrated)
            entropies.append(col_entropy)

        # Reorder Q by entropy
        new_order = np.argsort(entropies)
        sorted_entropies = [entropies[i] for i in new_order]
        print(sorted_entropies[:10], sorted_entropies[-1])

        # TODO: logic here first
        # if ev something, permute upfront, else put to the bottom
        temp_order = new_order.copy()
        high = 0
        med = high_num 
        low = med + med_num 
        for i, c in enumerate(first_few_ev_concentration):
            if c == "high":
                temp_order[i] = new_order[high]
                high += 1
            elif c == "med":
                temp_order[i] = new_order[med]
                med += 1
            elif c == "low":
                temp_order[i] = new_order[low]
                low += 1
        # import pdb;pdb.set_trace()
        temp_order[len(first_few_ev_concentration):num_points-low+len(first_few_ev_concentration)] = new_order[low:] 
        i = 0
        num_high_left = high_num - high
        num_med_left = high_num + med_num - med 
        # print(num_high_left, num_med_left)
        # import pdb;pdb.set_trace()
        temp_order[-num_high_left:] = new_order[high:high_num] 
        temp_order[-num_med_left-num_high_left:-num_high_left] = new_order[med:high_num+med_num]
        # new_order = temp_order
        print(new_order[:high_num+med_num+3], new_order[-high_num-med_num-3:])
        print(temp_order[:high_num+med_num+3], temp_order[-high_num-med_num-3:])
        Q = Q[:, temp_order]

        # Get the spectrum
        lengthscale = lengthscale
        a, b, c, d = 1, 2, 3, 4
        points = sample_4d_hyperboloid(num_points, a, b, c, d)
        kernel = StreamingRBFKernel(points, lengthscale=lengthscale)
        A_csr = kernel[:,:]
        start_time = time.time()
        print("Computing exact spectrum to generate random matrix...")
        S = sp.linalg.svd(A_csr, lapack_driver="gesdd", compute_uv=False)
        exact_time = time.time() - start_time
        print("Exact:", exact_time)

        S = np.concatenate([S, np.ones(num_points-S.shape[0]) * S[-1]])

        # import pdb;pdb.set_trace()
        title = ""
        A_csr = Q @ np.diag(S) @ Q.T
        A_is_sym_psd = True
    elif "test_cve" in matrix_name:
        np.random.seed(10)
        from scipy.stats import entropy
        _, _, num_points, lengthscale, ent = matrix_name.split("_")[:5]  
        num_points, lengthscale = int(num_points), float(lengthscale)
        n = num_points

        n = 1000
        eps = 1e-2
        a = 1/3

        if ent == "low":
            k = 100
        elif ent == "high":
            k = 999
        else:
            raise NotImplementedError
        b = 2/3 - eps

        norm_rest_sq = (1-a-b)

        u = np.zeros(n)
        u[0] = np.sqrt(a)
        # u[1] = b
        u[1:k] = np.random.randn(k-1)
        u[1:k] /= np.linalg.norm(u[1:k])
        u[1:k] *= np.sqrt(b) # norm sqrt(b)
        u[k:] = np.random.randn(n-k)
        u[k:] /= np.linalg.norm(u[k:])
        u[k:] *= np.sqrt(norm_rest_sq) 
        # u[k:] = np.ones(n-k) * c #np.random.randn(n-2)
        print(np.linalg.norm(u))
        # u /= np.linalg.norm(u)


        Q = np.random.randn(n, n)
        Q[:,0] = u
        Q, _ = np.linalg.qr(Q)
        
        print(max(np.abs(Q[:,0])))
        print([entropy(Q[:,i]**2) for i in range(10)])
        # fig, ax = plt.subplots(figsize=(12, 8))
        # num_sv = 10
        # colors = plt.cm.jet(np.linspace(0, 1, num_sv))
        # # Plot each window's data
        # for i in range(num_sv):
        #     plt.semilogy(Q[:,i]**2, 'o-', color=colors[i], label=f'Singular vector #{i}')
        # # import pdb;pdb.set_trace()
        # plt.legend()
        # plt.ylabel("")
        # plt.xlabel("Index")
        # plt.title(f"Eigenvector Distribution")
        # plt.grid()
        # plt.savefig(f"figures/{matrix_name}_singular_vector_dist.png")
        # plt.close()
        # raise 

        # eps = 1e-2
        # a = 1/np.sqrt(3)
        # k = 100
        # b = 10*a/(k)
        # norm_b = np.sqrt((k-1)*b**2)
        # norm_rest = np.sqrt((1-a**2-(k-1)*b**2))

        # u = np.zeros(n)
        # u[0] = a
        # # u[1] = b
        # u[1:k] = np.random.randn(k-1)
        # u[1:k] /= np.linalg.norm(u[1:k])
        # u[1:k] *= norm_b
        # u[k:] = norm_rest * np.random.randn(n-k)
        # u[k:] /= np.linalg.norm(u[k:])
        # u[k:] *= norm_rest
        # # u[k:] = np.ones(n-k) * c #np.random.randn(n-2)
        # print(np.linalg.norm(u))
        # # u /= np.linalg.norm(u)

        # v = np.zeros(n)
        # v[0] = a
        # # v[1:] = np.ones(n-1) * 10 * eps
        # v[1:] = np.random.randn(n-1)

        # # Get v perp to u[1:] first, then do the rest
        # v_perp = v[1:] - np.dot(u[1:], v[1:]) * u[1:]  / np.linalg.norm(u[1:])**2
        # v_perp = v_perp * np.sqrt(1-(a**2 / np.linalg.norm(u[1:]))**2-a**2) / np.linalg.norm(v_perp)
        # v_ = np.zeros(n)
        # v_[0] = a
        # v_[1:] = -a**2 * u[1:] / np.linalg.norm(u[1:])**2 + v_perp
        # print(np.dot(u,v_), np.linalg.norm(v_))

        # Q = np.random.randn(n, n)
        # Q[:,0] = v_
        # Q[:,1] = u
        # Q, _ = np.linalg.qr(Q)
        
        # if ent == "high":
        #     # swap lower entropy column down
        #     temp = Q[:,1].copy()
        #     Q[:,1] = Q[:, -1]
        #     Q[:,-1] = temp
        # elif ent == "low":
        #     # swap lower one up, higher one to bottom
        #     temp = Q[:,0].copy()
        #     Q[:,0] = Q[:, 1]
        #     Q[:,1] = temp
        #     temp = Q[:,1].copy()
        #     Q[:,1] = Q[:, -1]
        #     Q[:,-1] = temp
        # else:
        #     raise NotImplementedError
        
        # Get the spectrum
        lengthscale = lengthscale
        a, b, c, d = 1, 2, 3, 4
        points = sample_4d_hyperboloid(num_points, a, b, c, d)
        kernel = StreamingRBFKernel(points, lengthscale=lengthscale)
        A_csr = kernel[:,:]
        start_time = time.time()
        print("Computing exact spectrum to generate random matrix...")
        S = sp.linalg.svd(A_csr, lapack_driver="gesdd", compute_uv=False)
        exact_time = time.time() - start_time
        print("Exact:", exact_time)

        # import pdb;pdb.set_trace()
        title = ""
        A_csr = Q @ np.diag(S) @ Q.T
        A_is_sym_psd = True
        
    elif "kernel_random" in matrix_name:
        np.random.seed(10)

        num_points, lengthscale = matrix_name.split("_")[-2:]
        num_points, lengthscale = int(num_points), float(lengthscale)
        # num_points = 10000
        lengthscale = lengthscale
        a, b, c, d = 1, 2, 3, 4
        file_path = f'{output_dir}/USVt_exact_{matrix_postfix.replace("kernel_random", "hyperboloid")}.npz'
            
        # Check if file exists
        if False and os.path.exists(file_path):
            print(f"File found: {file_path}")
            # Load the file
            data = np.load(file_path)
            U_exact, S_exact, Vt_exact = data['U'], data['S'], data['Vt_exact']
        else:
            points = sample_4d_hyperboloid(num_points, a, b, c, d)
            kernel = StreamingRBFKernel(points, lengthscale=lengthscale)
            A_csr = kernel[:,:]
            start_time = time.time()
            print("Computing exact spectrum to generate random matrix...")
            S = sp.linalg.svd(A_csr, lapack_driver="gesdd", compute_uv=False)
            exact_time = time.time() - start_time
            print("Exact:", exact_time)
            start_time = time.time()
            print("Computing exact SVD...")
            U_exact, S_exact, Vt_exact = sp.linalg.svd(A_csr, lapack_driver="gesdd")
            if not os.path.exists(file_path):
                np.savez(file_path, U=None, S=S_exact[:1000], Vt_exact=Vt_exact[:1000, :])
            exact_time = time.time() - start_time
            print("Exact:", exact_time)

        N = num_points
        S = S_exact
        # Generate test matrix
        Q = np.random.randn(N, N)
        Q, _ = np.linalg.qr(Q)
        title = ""
        A_csr = Q @ np.diag(S) @ Q.T
        A_is_sym_psd = True
    elif "kernel_" in matrix_name:
        np.random.seed(10)
        name, num_points, lengthscale = matrix_name.split("_")[-3:]
        num_points, lengthscale = int(num_points), float(lengthscale)
        # num_points = 10000
        lengthscale = lengthscale

        if name == "swissroll":
            points = sample_swiss_roll(num_points)
        elif name == "torus":
            points = sample_torus(num_points)
        elif name == "gaussianmixture":
            points = sample_gaussian_mixture(num_points)
        elif name == "stocks":
            home_dir = os.path.expanduser("~")
            filepath = os.path.join(home_dir, f'data/data_2m.mtx')
            points = sp.io.mmread(filepath)
            points = points[:num_points, :]
        else:
            raise Exception("Shape not supported")
        
        kernel = StreamingRBFKernel(points, lengthscale=lengthscale, 
            cache_dir=os.path.join("cache", matrix_name, "kernel_cache"),
            verbose=True,)
        if num_points < 5e4:
            A_csr = kernel[:,:]
        else:
            kernel.precompute_blocks(overwrite=False)
            A_csr = kernel
        title = ""
        A_is_sym_psd = True
    elif "sparsify_" in matrix_name:
        np.random.seed(10)
        name, num_points, lengthscale, sparsity = matrix_name.split("_")[-4:]
        num_points, lengthscale, sparsity = int(num_points), float(lengthscale), float(sparsity)
        # num_points = 10000
        lengthscale = lengthscale

        if name == "swissroll":
            points = sample_swiss_roll(num_points)
        elif name == "torus":
            points = sample_torus(num_points)
        elif name == "gaussianmixture":
            points = sample_gaussian_mixture(num_points)
        elif name == "stocks":
            home_dir = os.path.expanduser("~")
            filepath = os.path.join(home_dir, f'data/data_2m.mtx')
            points = sp.io.mmread(filepath)
            points = points[:num_points, :]
        elif name == "hyperboloid":
            a, b, c, d = 1, 2, 3, 4
            points = sample_4d_hyperboloid(num_points, a, b, c, d)
        else:
            raise Exception("Shape not supported")
        
        kernel = StreamingRBFKernel(points, lengthscale=lengthscale)
        A_csr = kernel[:,:]
        results3, sparse_mats = find_sparsity_thresholds(
            A_csr, 
            target_sparsities=[sparsity], 
            return_sparse_matrices=True 
        )
        
        print("Sparse matrix formats:")
        for s, mat in sparse_mats.items():
            print(f"  {s:.1%}: {mat.format} matrix with {mat.nnz} non-zeros")

        A_csr = sparse_mats[sparsity]
        title = ""
        A_is_sym_psd = True
    else:
        # Download and read the matrix
        # A = download_and_read_matrix(url)
        A = download_and_read_matrix_cached(url, cache_dir+"/"+matrix_name+".tar.gz")
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
        

    
    # import pdb;pdb.set_trace()
    # plt.figure(figsize=(20, 15))
    # plt.hist(A_csr.flatten(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    # plt.xlabel('Element Values')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Matrix Elements')
    # plt.grid(True, alpha=0.3)
    # plt.savefig(f'hist.png', dpi=100)
    # plt.close()

    # plt.figure(figsize=(20, 15))
    # plt.hist(A_csr.flatten(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    # plt.xlabel('Element Values')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Matrix Elements')
    # plt.grid(True, alpha=0.3)
    # plt.yscale('log', base=10)
    # plt.savefig(f'hist_log.png', dpi=100)
    # plt.close()

    # plt.figure(figsize=(20, 15))
    # plt.hist(np.log10(A_csr.flatten()), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    # plt.xlabel('Element Values')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Matrix Elements')
    # plt.grid(True, alpha=0.3)
    # plt.yscale('log', base=10)
    # plt.savefig(f'hist_logxy.png', dpi=100)
    # plt.close()
    # import pdb;pdb.set_trace()
    # raise


    # Plot matrix
    # if A_csr.shape[1] < 5e4:
    #     plt.figure(figsize=(12, 8)) 
    #     fig, ax = matspy.spy_to_mpl(A_csr) 
    #     title = ax.set_title(title, 
    #         loc='center', wrap=True) 
    #     fig.tight_layout() 
    #     #plt.show() 
    #     fig.savefig(f'{output_dir}/{matrix_postfix}.png', dpi=100) 
    #     plt.close(fig) 
    
    #     if "hyperboloid" in matrix_name:
    #         plt.figure(figsize=(10, 8))
    #         im = plt.imshow(A_csr, 
    #                 cmap='viridis',
    #                 aspect='equal',
    #                 origin='upper',  # to match the orientation in your image
    #                 interpolation='nearest')
    #         plt.title(matrix_name)
    #         plt.xlabel("Samples")
    #         plt.ylabel("Samples")
    #         plt.savefig(f'{output_dir}/{matrix_postfix}_heatmap.png', dpi=100)
    # else:
    #     save_in_text = False # False
    # raise

    if A_csr.shape[1] < 5e4:
        # Compute exact SVD (full
        # start_time = time.time()
        # U_exact, S_exact, Vt_exact = sp.linalg.svd(A_csr.todense(), lapack_driver="gesdd")
        # exact_time = time.time() - start_time
        # Compute exact SVD (full
        
        if "kernel_random" in matrix_name:
            U_exact, S_exact, Vt_exact = Q, S, Q.T 
            # np.savez(f'{output_dir}/US_exact1000_{matrix_postfix}.npz', U=U_exact[:,:1000], S=S_exact[:1000]) 
        elif "test_coherence" in matrix_name:
            U_exact, S_exact, Vt_exact = Q, S, Q.T 
        elif "hyperboloid" in matrix_name and \
                (kernel_noise_std > 0.0 or point_noise_std > 0.0):
            kernel_true = StreamingRBFKernel(points, lengthscale=lengthscale)
            kernel_true = kernel_true[:, :]

            file_path = f'{output_dir}/USVt_exact_{matrix_postfix}.npz'
            cache_path = f'{cache_dir}/USVt_exact_{matrix_postfix}.npz'

            if os.path.exists(file_path):
                print(f"File found: {file_path}")
                data = np.load(file_path)
                U_exact, S_exact, Vt_exact = None, data['S'], data['Vt_exact']
            elif os.path.exists(cache_path):
                print(f"File found in cache: {cache_path}")
                data = np.load(cache_path)
                U_exact, S_exact, Vt_exact = None, data['S'], data['Vt_exact']
            else:
                start_time = time.time()
                print("Computing exact SVD for noisy kernel...")
                if isinstance(kernel_true, csr_matrix):
                    U_exact, S_exact, Vt_exact = sp.linalg.svd(
                        kernel_true.todense(), lapack_driver="gesdd"
                    )
                elif isinstance(kernel_true, StreamingRBFKernel) or isinstance(kernel_true, StreamingKroneckerGraph):
                    U_exact, S_exact, Vt_exact = sp.linalg.svd(
                        kernel_true[:, :], lapack_driver="gesdd"
                    )
                else:
                    U_exact, S_exact, Vt_exact = sp.linalg.svd(
                        kernel_true, lapack_driver="gesdd"
                    )

                np.savez(
                    file_path,
                    U=None,
                    S=S_exact[:1000],
                    Vt_exact=Vt_exact[:1000, :]
                )
                exact_time = time.time() - start_time
                print("Exact:", exact_time)

        else:
            file_path = f'{output_dir}/USVt_exact_{matrix_postfix}.npz'
            cache_path = f'{cache_dir}/USVt_exact_{matrix_postfix}.npz'

            if os.path.exists(file_path):
                print(f"File found: {file_path}")
                data = np.load(file_path)
                U_exact, S_exact, Vt_exact = None, data['S'], data['Vt_exact']
            elif os.path.exists(cache_path):
                print(f"File found in cache: {cache_path}")
                data = np.load(cache_path)
                U_exact, S_exact, Vt_exact = None, data['S'], data['Vt_exact']
            else:
                start_time = time.time()
                print("Computing exact SVD...")
                if isinstance(A_csr, csr_matrix):
                    U_exact, S_exact, Vt_exact = sp.sparse.linalg.svds(A_csr, k=500)
                    S_exact = S_exact[::-1]
                    Vt_exact = Vt_exact[::-1, :]
                    U_exact = U_exact[:, ::-1]
                elif isinstance(A_csr, StreamingRBFKernel) or isinstance(A_csr, StreamingKroneckerGraph):
                    U_exact, S_exact, Vt_exact = sp.linalg.svd(
                        A_csr[:, :], lapack_driver="gesdd"
                    )
                else:
                    U_exact, S_exact, Vt_exact = sp.linalg.svd(
                        A_csr, lapack_driver="gesdd"
                    )

                np.savez(
                    file_path,
                    U=None,
                    S=S_exact[:1000],
                    Vt_exact=Vt_exact[:1000, :]
                )
                exact_time = time.time() - start_time
                print("Exact:", exact_time)
    else:
        start_time = time.time()
        file_path = f'{output_dir}/USVt_exact_{matrix_postfix}.npz'
        cache_path = f'{cache_dir}/USVt_exact_{matrix_postfix}.npz'
        print("Large matrix, computing svds")

        if os.path.exists(file_path):
            print(f"File found: {file_path}")
            data = np.load(file_path)
            U_exact, S_exact, Vt_exact = None, data['S'], data['Vt_exact']
        elif os.path.exists(cache_path):
            print(f"File found in cache: {cache_path}")
            data = np.load(cache_path)
            U_exact, S_exact, Vt_exact = None, data['S'], data['Vt_exact']
        else:
            if True and "kernel" in matrix_name:
                print("Using LinearOperator for kernel SVD")
                v = np.random.randn(kernel.shape[1])
                t0 = time.time()
                y = kernel.matvec(v)
                print("single matvec time:", time.time() - t0)
                print("output norm:", np.linalg.norm(y))

                V = np.random.randn(kernel.shape[1], 8)
                t0 = time.time()
                Y = kernel.matmat(V)
                print("single matmat(8) time:", time.time() - t0)

                U_exact, S_exact, Vt_exact = sp.sparse.linalg.svds(
                    A_csr.to_linear_operator(), k=200
                )
            else:
                U_exact, S_exact, Vt_exact = sp.sparse.linalg.svds(A_csr, k=200)

            S_exact = S_exact[::-1]
            Vt_exact = Vt_exact[::-1, :]
            U_exact = U_exact[:, ::-1]

            np.savez(
                file_path,
                U=None,
                S=S_exact[:1000],
                Vt_exact=Vt_exact[:1000, :]
            )
            exact_time = time.time() - start_time
            print("Exact:", exact_time)
    # raise
    # Calculate entropy for each column
    # from scipy.stats import entropy
    if not Vt_exact is None:
        entropies = []
        for i in range(Vt_exact.shape[0]):
            col = Vt_exact[i, :] 
            prob_dist = col**2 
            col_entropy = sp.stats.entropy(prob_dist, base=2)  # higher entropy means more spread out, lower means more concentrated)
            entropies.append(col_entropy)

        print("Entropies:", [float(f"{entropies[i]:.3f}") for i in range(50)])
        print("Normalized Entropies:", [float(f"{e:.3f}") for e in (np.array(entropies) / np.log2(Vt_exact.shape[1]))[:50]])
    
    # sp.io.savemat("matlab2/eigenvectors5.mat", {"eigenvectors":Vt_exact[:5,:].T})
    # raise


    # if not Vt_exact is None:
    #     if isinstance(A_csr, csr_matrix):
    #         A_squared = A_csr.copy()
    #         A_squared.data **= 2
    #     else:
    #         A_squared = A_csr ** 2
    #     weights = np.asarray(np.sqrt(np.sum(A_squared, axis=1))).reshape(-1)
    #     weights = np.array(weights) / np.sum(weights)
    
    # if "hyperboloid" in matrix_name or "synthetic" in matrix_name or "kernel" in matrix_name or "test" in matrix_name:
    if "olafu" in matrix_name:
        np.random.seed(52) # Just changing up and see for crystm
    else:
        np.random.seed(42) #42
    # data = np.load("perms/row_order_final.npz")
    # perm = data["row_permutation"]
    # import scipy.io
    # permuted_matrix = A_csr[perm, :]
    # permuted_matrix = permuted_matrix[:, perm]
    # scipy.io.savemat('perms/matrix_data.mat', {'matrix': permuted_matrix})
    # raise
    random_uniform_perms = [np.random.permutation(A_csr.shape[0]) for i in range(5)]
    random_uniform_streams = [np.random.randint(0, A_csr.shape[0], size=A_csr.shape[0]) for _ in range(5)]
    permutations = {
        # "original": np.arange(A_csr.shape[0]),
        "random_uniform": random_uniform_perms[0],
        # "decreasing_norm": np.argsort(weights)[::-1],
        # "increasing_norm": np.argsort(weights)[::1],
        # "decreasing_V2": np.argsort(np.abs(Vt_exact[1,:]))[::-1],
        # "manual_perm": perm,
        # "manual_perm_reverse": perm[::-1],
        "random_uniform_2": random_uniform_perms[1],
        "random_uniform_3": random_uniform_perms[2],
        "random_uniform_4": random_uniform_perms[3],
        "random_uniform_5": random_uniform_perms[4],
        # "random_wr": random_uniform_streams[0],
    }

    # print(Vt_exact[0, permutations["decreasing_V2"][:10]])
    # print(np.linalg.norm(A_csr[permutations["decreasing_V2"][:10], :], axis=1))
    # print(Vt_exact[0, permutations["decreasing_norm"][:10]])
    # print(np.linalg.norm(A_csr[permutations["decreasing_norm"][:10], :], axis=1))
    # for k in [10, 20, 50, 100]:
    # print(random_uniform_perms[:10])
    # raise
    save_mat = False
    no_og = False
    no_adaptive = True 
    if save_mat:
        import scipy
    
    if A_csr.shape[1] < 5e4:
        k_default = 10
        first_window_size = k_default*10
    # elif "stocks" in matrix_name:
    #     k_default = 10
    #     first_window_size = A_csr.shape[0] // 20
    else:
        print("Large matrix, using larger k")
        k_default = 100 
        # if "stocks" in matrix_name:
        #     if np.abs(lengthscale - 2.2361) < 1e-3:
        #         k_default = 10
        #     elif np.abs(lengthscale - 0.7071) < 1e-3:
        #         k_default = 20
        #     elif np.abs(lengthscale - 0.2236) < 1e-3:
        #         k_default = 50
        first_window_size = A_csr.shape[0] // 20
    # For comparison with FD
    
    k_list = [k] if k else [k_default]
    first_window_size = stream_size + k if stream_size else first_window_size
    for k in k_list:
    # for k in [200, 400, 600, 800, 1000]:
        for size in [first_window_size]:
            # stream_size = size
            stream_size = size if not stream_size else stream_size
            # stream_size = mem_size - k if mem_size else size
            # first_window_size = mem_size

            # for threshold_factor in [1e1, 1e2, 1e3, 1e4]:
            for threshold_factor in [1e2]:
                for row_permutation in permutations:
                    # for reservoir_size in [10, 50, 100]:
                    for reservoir_size in [k_default]:
                        for use_true_matrix in [False]:
                            for reservoir_method in ["greedy"]:
                            # for reservoir_method in ["greedy", "current_window"]:
                                if reservoir_method == "current_window" and method != "isvddemix":
                                    continue
                                if save_mat:
                                    filename = f'perms/matrix_data_{matrix_name}_{row_permutation}.mat'
                                    perm = permutations[row_permutation]
                                    permuted_matrix = A_csr[perm, :]
                                    # permuted_matrix = permuted_matrix[:, perm]
                                    # scipy.io.savemat(filename, {'matrix': permuted_matrix})
                                    scipy.io.savemat(filename, {'matrix': A_csr, 'perm':perm})
                                    # filename = f'perms/matrix_data_{matrix_name}_{row_permutation}.mtx'
                                    # scipy.io.mmwrite(filename, A_csr)
                                    print(f"Matrix saved in", filename)
                                    if len(S_exact) < 100:
                                        print(S_exact)
                                elif not no_og:
                                    print(f"{row_permutation}: OG")
                                    name = matrix_postfix + f"_{method}_" + row_permutation + "_"  
                                    name += "true_" if use_true_matrix else "" 
                                    name += f"size_{size}"
                                    name += f"_ssize_{stream_size}" if stream_size != size else ""
                                    name += f"_k_{k}"
                                    name += f"_factor_{threshold_factor}" if threshold_factor != 1e2 else ""
                                    name += f"_reservoir_{reservoir_method}" if reservoir_method != "uniform" else ""
                                    if method == "isvddemix" and reservoir_size != 10 and reservoir_method == "greedy":
                                        name += f"{reservoir_size}"
                                    print("Name:", name)
                                    S, Vt = isvd(A_csr, 
                                                S_exact, Vt_exact, U_exact, 
                                                row_permutation=permutations[row_permutation].copy(),
                                                name=name,
                                                output_dir=output_dir,
                                                is_sym_psd=A_is_sym_psd,
                                                stream_size=stream_size,
                                                first_window_size=size,
                                                col_permutation=None,
                                                track_reconstruction_error=True,
                                                k=k,
                                                use_true_matrix=use_true_matrix,
                                                threshold_factor=threshold_factor,
                                                reservoir_size=reservoir_size,
                                                reservoir_method=reservoir_method,
                                                method=method,
                                                save_in_text=save_in_text,
                                                )
                                if row_permutation != "original" and not "random" in row_permutation and \
                                    row_permutation != "manual_perm":
                                    continue

                                # num_Vs_list = [1, 5, 10]
                                num_Vs_list = [10]
                                num_Vs_list = [min(x, k) for x in num_Vs_list]
                                num_Vs_list = list(set(num_Vs_list))
                                if no_adaptive:
                                    num_Vs_list = []
                                
                                if "isvd1by1" in method:
                                    num_Vs_list = [] # Skip

                                for num_Vs in num_Vs_list:
                                    for is_reversed in [False]:
                                        for with_S in [True, False]:
                                            if num_Vs > k:
                                                num_Vs = min(k, num_Vs)
                                            print(f"{row_permutation}: num_V={num_Vs}")
                                            name = matrix_postfix + f"_{method}_" + f"Vapprox{'_withS' if with_S else ''}_{num_Vs}_" + row_permutation + "_" 
                                            name += "reversed_" if is_reversed else "" 
                                            name += "true_" if use_true_matrix else ""
                                            name += f"size_{size}"
                                            name += f"_ssize_{stream_size}" if stream_size != size else ""
                                            name += f"_k_{k}"
                                            name += f"_factor_{threshold_factor}" if threshold_factor != 1e2 else ""
                                            name += f"_reservoir_{reservoir_method}" if reservoir_method != "uniform" else ""
                                            if method == "isvddemix" and reservoir_size != 10 and reservoir_method == "greedy":
                                                name += f"{reservoir_size}"
                                            print("Name:", name)
                                            S, Vt, perm = isvd(A_csr, 
                                                        S_exact, Vt_exact, U_exact,
                                                        row_permutation=permutations[row_permutation].copy(), 
                                                        #  name=matrix_postfix + "_new_" + f"Vapprox_withS_{num_Vs}_" + row_permutation + "_reversed_" + f"size_{size}_k_{k}",
                                                        name=name, # row perm only initially
                                                        output_dir=output_dir,
                                                        is_sym_psd=A_is_sym_psd,
                                                        num_Vs=num_Vs,
                                                        with_S=with_S,
                                                        stream_size=stream_size,
                                                        first_window_size=size,
                                                        k=k,
                                                        track_reconstruction_error=True,
                                                        reverse=is_reversed, # False default, for decreasing norm approximations
                                                        use_true_matrix=use_true_matrix,
                                                        threshold_factor=threshold_factor,
                                                        reservoir_size=reservoir_size,
                                                        reservoir_method=reservoir_method,
                                                        method=method,
                                                        return_row_order=True,
                                                        save_in_text=save_in_text,
                                                        )
                                        if save_mat:
                                            filename = f'perms/matrix_data_{matrix_name}_{row_permutation}_Vapprox_withS_{num_Vs}.mat'
                                            permuted_matrix = A_csr[perm, :]
                                            # permuted_matrix = permuted_matrix[:, perm]
                                            # scipy.io.savemat(filename, {'matrix': permuted_matrix})
                                            scipy.io.savemat(filename, {'matrix': A_csr, 'perm':perm})
                                            # filename = f'perms/matrix_data_{matrix_name}_{row_permutation}.mtx'
                                            # scipy.io.mmwrite(filename, A_csr)
                                            print(f"Matrix saved in", filename)
                                            if len(S_exact) < 100:
                                                print(S_exact)

                                print("Done:", name)
                            # import seaborn as sns
                            # save_dir = "figures"
                            # # Compute the overlap matrix
                            # C = Vt @ Vt_exact[:Vt.shape[0], :].T
                            
                            # # Compute singular values of the overlap matrix
                            # s = np.linalg.svd(C, compute_uv=False)
                            
                            # # Compute angles in radians
                            # angles = np.arccos(np.clip(s, -1.0, 1.0))
                            # print("Subspace angle 2:", max(angles), np.mean(angles))
                            
                            # # Create figure and axis
                            # fig, ax = plt.subplots(figsize=(10, 8))
                            
                            # # Calculate heatmap data
                            # C_log = np.log10(np.clip(np.abs(C), 1e-32, 1) - np.eye(len(s)))
                            
                            # # Create heatmap
                            # sns.heatmap(C_log, 
                            #             cbar_kws={'label': 'Log10 Overlap'},
                            #             xticklabels=False,
                            #             yticklabels=False,
                            #             ax=ax,
                            #             vmin=-12)
                            
                            # ax.set_title('$\log(|V_{approx}^T \cdot V| - I)$ (element-wise)')
                            
                            # # Create directory if it doesn't exist
                            # os.makedirs(save_dir, exist_ok=True)
                            
                            # # Save figure
                            # plt.savefig(os.path.join(save_dir, f'singular_vectors_overlap_{matrix_name}_{row_permutation}.png'))
                            # plt.close()

    # else:
    #     np.random.seed(42)
    #     # data = np.load("perms/row_order_final.npz")
    #     # perm = data["row_permutation"]
    #     # import scipy.io
    #     # permuted_matrix = A_csr[perm, :]
    #     # permuted_matrix = permuted_matrix[:, perm]
    #     # scipy.io.savemat('perms/matrix_data.mat', {'matrix': permuted_matrix})
    #     # raise
    #     random_uniform_perms = [np.random.permutation(A_csr.shape[0]) for i in range(5)]
    #     permutations = {
    #         # "original": np.arange(A_csr.shape[0]),
    #         "random_uniform": random_uniform_perms[0],
    #         # "decreasing_norm": np.argsort(weights)[::-1],
    #         # "increasing_norm": np.argsort(weights)[::1],
    #         # "decreasing_V2": np.argsort(np.abs(Vt_exact[1,:]))[::-1],
    #         # "manual_perm": perm,
    #         # "manual_perm_reverse": perm[::-1],
    #         # "random_uniform_2": random_uniform_perms[1],
    #         # "random_uniform_3": random_uniform_perms[2],
    #         # "random_uniform_4": random_uniform_perms[3],
    #         # "random_uniform_5": random_uniform_perms[4],
    #     }
    #     # print(Vt_exact[0, permutations["decreasing_V2"][:10]])
    #     # print(np.linalg.norm(A_csr[permutations["decreasing_V2"][:10], :], axis=1))
    #     # print(Vt_exact[0, permutations["decreasing_norm"][:10]])
    #     # print(np.linalg.norm(A_csr[permutations["decreasing_norm"][:10], :], axis=1))
    #     # for k in [10, 20, 50, 100]:
    #     # raise
    #     save_mat = False
    #     if save_mat:
    #         import scipy
    #     for k in [10]:
    #     # for k in [200, 400, 600, 800, 1000]:
    #         for size in [100]:
    #             # for threshold_factor in [1e1, 1e2, 1e3, 1e4]:
    #             for threshold_factor in [1e2]:
    #                 for row_permutation in permutations:
    #                     # for reservoir_size in [10, 50, 100]:
    #                     for reservoir_size in [10]:
    #                         for use_true_matrix in [False]:
    #                             for reservoir_method in ["greedy"]:
    #                             # for reservoir_method in ["greedy", "current_window"]:
    #                                 if reservoir_method == "current_window" and method != "isvddemix":
    #                                     continue
    #                                 if save_mat:
    #                                     filename = f'perms/matrix_data_{matrix_name}_{row_permutation}.mat'
    #                                     perm = permutations[row_permutation]
    #                                     permuted_matrix = A_csr[perm, :]
    #                                     # permuted_matrix = permuted_matrix[:, perm]
    #                                     # scipy.io.savemat(filename, {'matrix': permuted_matrix})
    #                                     scipy.io.savemat(filename, {'matrix': A_csr, 'perm':perm})
    #                                     # filename = f'perms/matrix_data_{matrix_name}_{row_permutation}.mtx'
    #                                     # scipy.io.mmwrite(filename, A_csr)
    #                                     print(f"Matrix saved in", filename)
    #                                     if len(S_exact) < 100:
    #                                         print(S_exact)
    #                                 else:
    #                                     print(f"{row_permutation}: OG")
    #                                     name = matrix_postfix + f"_{method}_" + row_permutation + "_"  
    #                                     name += "true_" if use_true_matrix else "" 
    #                                     name += f"size_{size}_k_{k}"
    #                                     name += f"_factor_{threshold_factor}" if threshold_factor != 1e2 else ""
    #                                     name += f"_reservoir_{reservoir_method}" if reservoir_method != "uniform" else ""
    #                                     if method == "isvddemix" and reservoir_size != 10 and reservoir_method == "greedy":
    #                                         name += f"{reservoir_size}"
    #                                     print("Name:", name)
    #                                     S, Vt = isvd(A_csr, 
    #                                                 S_exact, Vt_exact, U_exact, 
    #                                                 row_permutation=permutations[row_permutation].copy(),
    #                                                 name=name,
    #                                                 output_dir=output_dir,
    #                                                 is_sym_psd=A_is_sym_psd,
    #                                                 stream_size=size,
    #                                                 first_window_size=size,
    #                                                 col_permutation=None,
    #                                                 track_reconstruction_error=True,
    #                                                 k=k,
    #                                                 use_true_matrix=use_true_matrix,
    #                                                 threshold_factor=threshold_factor,
    #                                                 reservoir_size=reservoir_size,
    #                                                 reservoir_method=reservoir_method,
    #                                                 method=method,
    #                                                 )
    #                                 if row_permutation != "original" and not "random" in row_permutation and \
    #                                     row_permutation != "manual_perm":
    #                                     continue

    #                                 # num_Vs_list = [1, 5, 10]
    #                                 num_Vs_list = [10]
    #                                 num_Vs_list = [min(x, k) for x in num_Vs_list]
    #                                 num_Vs_list = list(set(num_Vs_list))
                                    
    #                                 if "isvd1by1" in method:
    #                                     num_Vs_list = [] # Skip

    #                                 for num_Vs in num_Vs_list:
    #                                     for is_reversed in [False]:
    #                                         if num_Vs > k:
    #                                             num_Vs = min(k, num_Vs)
    #                                         print(f"{row_permutation}: num_V={num_Vs}")
    #                                         name = matrix_postfix + f"_{method}_" + f"Vapprox_withS_{num_Vs}_" + row_permutation + "_" 
    #                                         name += "reversed_" if is_reversed else "" 
    #                                         name += "true_" if use_true_matrix else ""
    #                                         name += f"size_{size}_k_{k}"
    #                                         name += f"_factor_{threshold_factor}" if threshold_factor != 1e2 else ""
    #                                         name += f"_reservoir_{reservoir_method}" if reservoir_method != "uniform" else ""
    #                                         if method == "isvddemix" and reservoir_size != 10 and reservoir_method == "greedy":
    #                                             name += f"{reservoir_size}"
    #                                         print("Name:", name)
    #                                         S, Vt, perm = isvd(A_csr, 
    #                                                     S_exact, Vt_exact, U_exact,
    #                                                     row_permutation=permutations[row_permutation].copy(), 
    #                                                     #  name=matrix_postfix + "_new_" + f"Vapprox_withS_{num_Vs}_" + row_permutation + "_reversed_" + f"size_{size}_k_{k}",
    #                                                     name=name, # row perm only initially
    #                                                     output_dir=output_dir,
    #                                                     is_sym_psd=A_is_sym_psd,
    #                                                     num_Vs=num_Vs,
    #                                                     with_S=True,
    #                                                     stream_size=size,
    #                                                     first_window_size=size,
    #                                                     k=k,
    #                                                     track_reconstruction_error=True,
    #                                                     reverse=is_reversed, # False default, for decreasing norm approximations
    #                                                     use_true_matrix=use_true_matrix,
    #                                                     threshold_factor=threshold_factor,
    #                                                     reservoir_size=reservoir_size,
    #                                                     reservoir_method=reservoir_method,
    #                                                     method=method,
    #                                                     return_row_order=True,
    #                                                     )
    #                                     if save_mat:
    #                                         filename = f'perms/matrix_data_{matrix_name}_{row_permutation}_Vapprox_withS_{num_Vs}.mat'
    #                                         permuted_matrix = A_csr[perm, :]
    #                                         # permuted_matrix = permuted_matrix[:, perm]
    #                                         # scipy.io.savemat(filename, {'matrix': permuted_matrix})
    #                                         scipy.io.savemat(filename, {'matrix': A_csr, 'perm':perm})
    #                                         # filename = f'perms/matrix_data_{matrix_name}_{row_permutation}.mtx'
    #                                         # scipy.io.mmwrite(filename, A_csr)
    #                                         print(f"Matrix saved in", filename)
    #                                         if len(S_exact) < 100:
    #                                             print(S_exact)

                                        
    #                             # import seaborn as sns
    #                             # save_dir = "figures"
    #                             # # Compute the overlap matrix
    #                             # C = Vt @ Vt_exact[:Vt.shape[0], :].T
                                
    #                             # # Compute singular values of the overlap matrix
    #                             # s = np.linalg.svd(C, compute_uv=False)
                                
    #                             # # Compute angles in radians
    #                             # angles = np.arccos(np.clip(s, -1.0, 1.0))
    #                             # print("Subspace angle 2:", max(angles), np.mean(angles))
                                
    #                             # # Create figure and axis
    #                             # fig, ax = plt.subplots(figsize=(10, 8))
                                
    #                             # # Calculate heatmap data
    #                             # C_log = np.log10(np.clip(np.abs(C), 1e-32, 1) - np.eye(len(s)))
                                
    #                             # # Create heatmap
    #                             # sns.heatmap(C_log, 
    #                             #             cbar_kws={'label': 'Log10 Overlap'},
    #                             #             xticklabels=False,
    #                             #             yticklabels=False,
    #                             #             ax=ax,
    #                             #             vmin=-12)
                                
    #                             # ax.set_title('$\log(|V_{approx}^T \cdot V| - I)$ (element-wise)')
                                
    #                             # # Create directory if it doesn't exist
    #                             # os.makedirs(save_dir, exist_ok=True)
                                
    #                             # # Save figure
    #                             # plt.savefig(os.path.join(save_dir, f'singular_vectors_overlap_{matrix_name}_{row_permutation}.png'))
    #                             # plt.close()
