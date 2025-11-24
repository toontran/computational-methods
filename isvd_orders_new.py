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

# def isvd(A_csr, S_exact=None, Vt_exact=None, U_exact=None, 
#          window_size=100, k=None,
#          num_windows=None, row_permutation=None, name="temp", figure_dir="figures", is_sym_psd=False,
#          num_Vs=None, track_U=False, track_discarded=False, with_S=False, reverse=False,
#          return_row_order=False, stream_size=None, col_permutation=None, use_true_matrix=False,
#          track_reconstruction_error=False, threshold_factor=100, nystrom=True):
#     global Vt
#     U = None
#     m, n = A_csr.shape
    
#     # W = num_windows  # number of windows (columns in this case) 
#     # l = m // W  # window size 
#     # k = k if k and k < l else l-1 # Number of singular values/vectors to compute
#     # r = min(k, m, l) 
#     k = window_size if k is None else k 
#     stream_size = window_size if stream_size is None else stream_size
#     W = (m - window_size) // stream_size + 1

#     # Create the directory if it doesn't exist
#     dir_path = f"{figure_dir}/{name}/"
#     directory = os.path.dirname(dir_path) 
#     if directory and not os.path.exists(directory):
#         print("Making directory:", directory)
#         os.makedirs(directory)
    
#     # Create a permutation of row indices
#     row_permutation = row_permutation if row_permutation is not None else np.arange(m)

#     sp_norm = sparse.linalg.norm if isinstance(A_csr, csr_matrix) else np.linalg.norm
#     A_norm = sp_norm(A_csr) # TODO
#     inverse_perm = None

#     total_S_reduced = 0
#     if track_discarded:
#         discarded_list = []
#     if track_reconstruction_error:
#         reconstruction_errors = []
#     print("Num windows:", W)
#     for j in range(W+1):
#         print("Window:", j+1)        
        
#         # Calculate the start and end indices for this window
#         if j == 0:
#             start_idx = j * window_size
#             end_idx = min((j + 1) * window_size, m)
#         else:
#             start_idx = j * stream_size
#             end_idx = min((j + 1) * stream_size, m)
#         if end_idx <= start_idx:
#             break
        
#         # Extract the next window
#         window_indices = row_permutation[start_idx:end_idx]
#         # print("Index:", end_idx, len(row_permutation))
# #         import pdb;pdb.set_trace()
#         next_window = A_csr[window_indices, :]

#         next_window = next_window[:, row_permutation]
#         if isinstance(A_csr, csr_matrix):
#             next_window = next_window.toarray()
        
#         # print(next_window.shape)
#         if j == 0:
#              # Initial SVD for the first window
#             w = window_size
            
#             S11, V11 = sp.linalg.eig(next_window[:, :window_size])
#             idx = np.argsort(S11)[::-1]
#             S11, V11 = S11[idx], V11[:, idx]

#             # Initial SVD
#             S11_sqrt = np.sqrt(np.abs(S11))
#             S11_invsqrt = np.zeros_like(S11)
#             mask = S11 > threshold_factor * w * np.finfo(float).eps * np.max(np.abs(S11))
#             S11_invsqrt[mask] = 1 / np.sqrt(S11[mask])

#             upper = V11 @ np.diag(S11_sqrt) @ V11.T
#             lower = next_window[:, w:].T @ V11 @ np.diag(S11_invsqrt) @ V11.T
#             aug_matrix = np.vstack([upper, lower])

#             U, S, Vt = sp.linalg.svd(aug_matrix, full_matrices=False)
#             idx = np.argsort(S)[::-1]
#             # keep = len(idx) # for now
#             V = U[:, idx[:k]]
#             S = S[idx[:k]]**2

#         else:
#             # import pdb;pdb.set_trace()

#             n = j + 1  # Current number of windows being processed

#             # Correct eigenvectors for current row perm
#             # Inverts the previous permutation, then applies current row permutation
#             # print(V.shape, len(row_permutation), len(inverse_perm))
#             # import pdb;pdb.set_trace()
#             V = V[compose_permutations(inverse_perm, row_permutation), :]  
#             print("Before recalculating:", (V[j*window_size:(j+1)*window_size,:] @ np.diag(S) @ V[:,:].T)[-3:,-5:])

#             # Get new window
#             current_window = next_window
#             w = window_size
#             N = A_csr.shape[0]

#             if track_reconstruction_error:
#                 # TODO: check reconstruction of part of current_window
#                 reconstructed = V[:(n-1)*w, :] @ np.diag(S) @ V[(n-1)*w:n*w, :].T
#                 print(n, w, "Shapes:", reconstructed.shape)
#                 reconstruction_error = np.linalg.norm(current_window[:, :(n-1)*w] - reconstructed.T) / np.linalg.norm(current_window[:, :(n-1)*w])
#                 reconstruction_errors.append(reconstruction_error)
#                 print("Reconstruction error P_C:", reconstruction_error)
#                 np.savez(os.path.join(dir_path, f'reconstruction_error.npz'),
#                         reconstruction_errors=np.array(reconstruction_errors),)

#             # print("Window after:", prev_window[:, row_permutation][-3:,-5:], (V[(j-1)*window_size:(j)*window_size,:] @ np.diag(S) @ V[:,:].T)[-3:,-5:])
#             # print("Norm after:", np.linalg.norm(prev_window[:, row_permutation] - V[(j-1)*window_size:(j)*window_size,:] @ np.diag(S) @ V[:,:].T))
            
#             # import pdb;pdb.set_trace()

#             # Create operator for eigendecomposition
#             if use_true_matrix:
#                 A11_op = make_operator_true(n, w, A_csr, N)
#             else:
#                 A11_op = make_operator(current_window, N, n, w, V, S)
#             print(A11_op.shape, n*w-1)
#             keep = len(S) # for now
#             S11, V11 = sp.sparse.linalg.eigs(A11_op, k=keep, which='LM', ncv=n*w - 1)

#             if end_idx == m:
#                 # If last window, stop
#                 V = V11
#                 S = S11
# #                 print(f"Eigenvalues: {S}")
# #                 print("Exact:", S_exact)
# #                 print(f"V shape: {V.shape}")
                
#             else:

#                 # Prepare for SVD
#                 S11_sqrt = np.sqrt(np.abs(S11))
#                 S11_invsqrt = np.zeros_like(S11)
#                 mask = S11 > threshold_factor * A11_op.shape[0] * np.finfo(float).eps * np.max(np.abs(S11))
#                 S11_invsqrt[mask] = 1 / np.sqrt(np.abs(S11[mask]))

#                 # Create SVD operator and compute decomposition
# #                 print("V11.shape, S11.shape, S11_sqrt.shape, current_window.shape, V.shape, S.shape, n, w, N")
# #                 print(V11.shape, S11.shape, S11_sqrt.shape, current_window.shape, V.shape, S.shape, n, w, N)

#                 if use_true_matrix:
#                     svd_op = create_svd_operators_true(V11, S11, S11_sqrt, S11_invsqrt, 
#                                              current_window, A_csr, n, w, N)
#                 else:
#                     svd_op = create_svd_operators(V11, S11, S11_sqrt, S11_invsqrt, 
#                                                 current_window, V, S, n, w, N)
#                 U, S, Vt = sp.sparse.linalg.svds(svd_op, k=k, ncv=n*w - 1, which="LM")

#                 # Update V and S for next iteration
#                 idx = np.argsort(S)[::-1]
#                 S, V = S[idx]**2, U[:, idx]

#             # if j == 4:
#             #     temp = A11_op(np.ones(n*w)) - A_csr[:n*w, :n*w] @ np.ones(n*w)
#             #     print("A11_op:", A11_op(np.ones(n*w))[:10], (A_csr[:n*w, :n*w] @ np.ones(n*w))[:10])
#             #     G = np.vstack([(V11 @ np.diag(S11_sqrt) @ V11.T),
#             #                    (np.vstack([V[:(n-1)*w, :] @ (np.diag(S) @ V[n*w:, :].T), 
#             #                                current_window[:, n*w:]]).T @ V11 @ np.diag(S11_invsqrt) @ V11.T)])
#             #     print("G:", (G@G.T)[:5, :5], A_csr[:5,:5])
#             #     svd_G = np.linalg.svd(G.real, full_matrices=False)
#             #     print(svd_G.S[:5]**2, S[:5])
#             # import pdb;pdb.set_trace()

#         print(S[:10])
#         print(S_exact[:10])

#         print("V.shape, S.shape:", V.shape, S.shape)

#         # invert row perm, compose with new row perm
#         inverse_perm = inverse_permutation(row_permutation)
#         temp = compose_permutations(row_permutation, inverse_permutation(row_permutation))
#         # print("Inverse is correct:", [i for i in range(len(temp)) if i != temp[i]])
#         print("Window before:", next_window[-3:,-5:], (V[j*window_size:(j+1)*window_size,:] @ np.diag(S) @ V[:,:].T)[-3:,-5:])
#         print("Norm before:", np.linalg.norm(next_window - V[j*window_size:(j+1)*window_size,:] @ np.diag(S) @ V[:,:].T))
#         # prev_window = A_csr[window_indices, :]
#         # prev_permutation = row_permutation
#         # import pdb;pdb.set_trace()
# #                 print(f"Eigenvalues: {S}")
# #                 print(f"V shape: {V.shape}")
    
#         # Plot
#         # plot_spectrum_comparison(S, S_exact, 
#         #                          A_norm, name, j, dir_path)
#         # plot_residuals(A_csr, S, Vt, S_exact, Vt_exact, U_exact, 
#         #                A_norm, name, j, dir_path, is_sym_psd) 
#         # plot_canonical_angles(Vt, Vt_exact, 
#         #                       j, dir_path)

#         Vt = V[inverse_perm, :].T # VSV^T ~ A
#         if num_Vs:
#             if with_S:
#                 if reverse:
#                     indices = np.argsort(np.sum((Vt[:num_Vs, row_permutation[end_idx:]] * S[:num_Vs].reshape(-1,1))**2, axis=0)).reshape(-1)[::1]
#                 else:
#                     indices = np.argsort(np.sum((Vt[:num_Vs, row_permutation[end_idx:]] * S[:num_Vs].reshape(-1,1))**2, axis=0)).reshape(-1)[::-1]
#             else:
#                 if reverse:
#                     indices = np.argsort(np.sum((Vt[:num_Vs, row_permutation[end_idx:]])**2, axis=0)).reshape(-1)[::1]
#                 else:
#                     indices = np.argsort(np.sum((Vt[:num_Vs, row_permutation[end_idx:]])**2, axis=0)).reshape(-1)[::-1]
# #             print(indices)
#             row_permutation[end_idx:] = row_permutation[end_idx:][indices] 

#         # Plot
#         save_spectrum_comparison(S+total_S_reduced, S_exact, 
#                                  A_norm, name, j, dir_path)
#         save_residuals(A_csr, S+total_S_reduced, Vt, 
#                        A_norm, name, j, dir_path, is_sym_psd,
#                        row_permutation, start_idx, end_idx) 

#         if not Vt_exact is None:
#             print("Reconstruction quality:", np.linalg.norm(Vt - Vt_exact[:Vt.shape[0], :], 'fro'))
#             save_canonical_angles(Vt, Vt_exact, 
#                                   j, dir_path)
#         if j == W - 1 and track_U and not U_exact is None and not is_sym_psd:
#             print("No tracking U option available")
# #             save_canonical_angles(U.T, U_exact.T, 
# #                                   j, dir_path, additional_label="_U")
        

#         if not S_exact is None:
#             print("Relative error in S:", np.linalg.norm(S - S_exact[:Vt.shape[0]]) / A_norm)
#         # X = np.linalg.pinv(Vt_exact[:Vt.shape[0],:].T) @ Vt.T 
#         # Vt_reconstructed = Vt_exact[:Vt.shape[0],:].T @ X
#         # print("Reconstruct Vt from Vt_exact:", np.linalg.norm(Vt.T - Vt_reconstructed, 'fro'))
#         # print("Projection F-norm error:", np.linalg.norm(Vt.T @ Vt - Vt_exact[:Vt.shape[0], :].T @ Vt_exact[:Vt.shape[0], :], 'fro'))
#         # print("Trace correlation", np.trace(Vt @ Vt_exact[:Vt.shape[0], :].T @ Vt_exact[:Vt.shape[0], :] @ Vt.T) / min(Vt.T.shape[1], Vt_exact[:Vt.shape[0], :].T.shape[1]))
    
#     ret = [S, Vt]
#     np.savez(os.path.join(dir_path, f'row_order_final.npz'),
#              row_permutation=row_permutation,)
# #     if track_U:
# #         ret.append(U)
#     if track_discarded:
#         ret.append(discarded_list)
#     if return_row_order:
#         ret.append(row_permutation)
#     if total_S_reduced > 0:
#         ret.append(total_S_reduced)
#     return ret


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
        temp = matrix_name.split("_")
        if len(temp) == 3:
            _, num_points, gamma = temp
            kernel_noise_std, point_noise_std = 0.0, 0.0
        elif len(temp) == 4:
            _, num_points, gamma, noise = temp
            noise = float(noise)
            kernel_noise_std, point_noise_std = noise, noise
        elif len(temp) == 5:
            _, num_points, gamma, kernel_noise_std, point_noise_std = temp
            kernel_noise_std, point_noise_std = float(kernel_noise_std), float(point_noise_std)
        else:
            raise Exception("Input incorrect")
        num_points, gamma = int(num_points), float(gamma)
        
        # num_points = 10000
        gamma = 10**gamma
        a, b, c, d = 1, 2, 3, 4
        points = sample_4d_hyperboloid(num_points, a, b, c, d)
        kernel = StreamingRBFKernel(points, gamma=gamma, 
                                    kernel_noise_std=kernel_noise_std,
                                    point_noise_std=point_noise_std)
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
    elif "kernel_" in matrix_name:
        np.random.seed(10)
        name, num_points, gamma = matrix_name.split("_")[-3:]
        num_points, gamma = int(num_points), float(gamma)
        # num_points = 10000
        gamma = 10**gamma

        if name == "swissroll":
            points = sample_swiss_roll(num_points)
        elif name == "torus":
            points = sample_torus(num_points)
        elif name == "gaussianmixture":
            points = sample_gaussian_mixture(num_points)
        else:
            raise Exception("Shape not supported")
        
        kernel = StreamingRBFKernel(points, gamma=gamma)
        A_csr = kernel[:,:]
        title = ""
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
    elif "hyperboloid" in matrix_name and \
            (kernel_noise_std > 0.0 or point_noise_std > 0.0):
        kernel_true = StreamingRBFKernel(points, gamma=gamma)
        kernel_true = kernel_true[:,:]
        start_time = time.time()
        print("Computing exact SVD for noisy kernel...")
        if isinstance(kernel_true, csr_matrix):
            U_exact, S_exact, Vt_exact = sp.linalg.svd(kernel_true.todense(), lapack_driver="gesdd")
        elif isinstance(kernel_true, StreamingRBFKernel) or isinstance(kernel_true, StreamingKroneckerGraph):
            U_exact, S_exact, Vt_exact = sp.linalg.svd(kernel_true[:,:], lapack_driver="gesdd")
        else: 
             U_exact, S_exact, Vt_exact = sp.linalg.svd(kernel_true, lapack_driver="gesdd")
        np.savez(f'{figure_dir}/US_exact1000_{matrix_postfix}.npz', U=U_exact[:,:1000], S=S_exact[:1000])
        exact_time = time.time() - start_time
        print("Exact:", exact_time) 
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
        # import scipy.io
        # permuted_matrix = A_csr[perm, :]
        # permuted_matrix = permuted_matrix[:, perm]
        # scipy.io.savemat('perms/matrix_data.mat', {'matrix': permuted_matrix})
        # raise
        random_uniform_perms = [np.random.permutation(len(weights)) for i in range(5)]
        permutations = {
            # "original": None,
            # "decreasing_norm": np.argsort(weights)[::-1],
            # "increasing_norm": np.argsort(weights)[::1],
            "random_uniform": random_uniform_perms[0],
            # "manual_perm": perm,
            # "manual_perm_reverse": perm[::-1],
            # "random_uniform_2": random_uniform_perms[1],
            # "random_uniform_3": random_uniform_perms[2],
            # "random_uniform_4": random_uniform_perms[3],
            # "random_uniform_5": random_uniform_perms[4],
        }
#         import pdb;pdb.set_trace()
        # for k in [10, 20, 50, 100, 200, 400, 600, 800, 1000]:
        # k = 10
        # for k in [10, 20, 50, 100]:
        for k in [10]:
        # for k in [200, 400, 600, 800, 1000]:
            for size in [100]:
                # for threshold_factor in [1e1, 1e2, 1e3, 1e4]:
                for threshold_factor in [1e2]:
                    for row_permutation in permutations:
                        for use_true_matrix in [False]:
                            print(f"{row_permutation}: OG")
                            name = matrix_postfix + "_new_" + row_permutation + "_"  
                            name += "true_" if use_true_matrix else "" 
                            name += f"size_{size}_k_{k}"
                            name += f"_factor_{threshold_factor}" if threshold_factor != 1e2 else ""
                            print("Name:", name)
                            S, Vt = isvd(A_csr, 
                                        S_exact, Vt_exact, U_exact, 
                                        row_permutation=permutations[row_permutation].copy(),
                                        name=name,
                                        # name=matrix_postfix + "_new_" + row_permutation + "_" + f"size_{size}_k_{k}",
                                        # name=matrix_postfix + "_new_" + row_permutation + "_" + f"size_{size}_k_{k}",
            #                              name=matrix_postfix + "_" + row_permutation + "_" + f"size_{size}",
                                        figure_dir=figure_dir,
                                        is_sym_psd=A_is_sym_psd,
                                        stream_size=size,
                                        window_size=size,
                                        col_permutation=None,
                                        track_reconstruction_error=True,
                                        k=k,
                                        use_true_matrix=use_true_matrix,
                                        threshold_factor=threshold_factor,
                                        method="nystrom",
                                        )
            #                 S, Vt = isvd(A_csr, 
            #                              S_exact, Vt_exact, U_exact, 
            #                              row_permutation=permutations[row_permutation], 
            #                              name=matrix_postfix + "_" + row_permutation + "_" + "col_perm" + "_" + f"size_{size}_old",
            # #                              name=matrix_postfix + "_" + row_permutation + "_" + f"size_{size}",
            #                              figure_dir=figure_dir,
            #                              is_sym_psd=A_is_sym_psd,
            #                              stream_size=size,
            #                              window_size=size,
            #                              col_permutation=permutations[row_permutation])
                    #         for num_Vs in [1, 2, 4, 10, 20, 50, 100]:
                            if row_permutation != "original" and not "random" in row_permutation and \
                                row_permutation != "manual_perm":
                                continue

                            # num_Vs_list = [1, 10, 100]
                            # # num_Vs_list = [10, 100]
                            # num_Vs_list = [min(x, k) for x in num_Vs_list]
                            # num_Vs_list = list(set(num_Vs_list))
                            # for num_Vs in num_Vs_list:
                            #     for is_reversed in [False]:
                            #         if num_Vs > k:
                            #             num_Vs = min(k, num_Vs)
                            #         print(f"{row_permutation}: num_V={num_Vs}")
                            #         name = matrix_postfix + "_new_" + f"Vapprox_withS_{num_Vs}_" + row_permutation + "_" 
                            #         name += "reversed_" if is_reversed else "" 
                            #         name += "true_" if use_true_matrix else ""
                            #         name += f"size_{size}_k_{k}"
                            #         name += f"_factor_{threshold_factor}" if threshold_factor != 1e2 else ""
                            #         print("Name:", name)
                            #         S, Vt = isvd(A_csr, 
                            #                     S_exact, Vt_exact, U_exact,
                            #                     row_permutation=permutations[row_permutation].copy(), 
                            #                     #  name=matrix_postfix + "_new_" + f"Vapprox_withS_{num_Vs}_" + row_permutation + "_reversed_" + f"size_{size}_k_{k}",
                            #                     name=name, # row perm only initially
                            #                     figure_dir=figure_dir,
                            #                     is_sym_psd=A_is_sym_psd,
                            #                     num_Vs=num_Vs,
                            #                     with_S=True,
                            #                     stream_size=size,
                            #                     window_size=size,
                            #                     k=k,
                            #                     track_reconstruction_error=True,
                            #                     reverse=is_reversed, # False default, for decreasing norm approximations
                            #                     use_true_matrix=use_true_matrix,
                            #                     threshold_factor=threshold_factor,
                            #                     method="nystrom",
                            #                     )
        
    #     permutations = {
    #         "decreasing_norm": np.argsort(weights)[::-1],
    #         "random_uniform": np.random.permutation(len(weights)),
    #     }
    #     use_true_matrix = False
    #     for k in [10]:
    #     # for k in [10]:
    #         for size in [100]:
    #             for row_permutation in permutations:
    #                 print(f"{row_permutation}: OG")
    #                 name = matrix_postfix + "_new_" + row_permutation + "_" 
    #                 name += "true_" if use_true_matrix else "" 
    #                 name += f"size_{size}_k_{k}"
    #                 S, Vt = isvd(A_csr, 
    #                             S_exact, Vt_exact, U_exact, 
    #                             row_permutation=permutations[row_permutation], 
    #                             name=name, 
    #                             # name=matrix_postfix + "_new_" + row_permutation + "_" + f"size_{size}_k_{k}",
    #                             # name=matrix_postfix + "_new_" + row_permutation + "_" + f"size_{size}_k_{k}",
    # #                              name=matrix_postfix + "_" + row_permutation + "_" + f"size_{size}",
    #                             figure_dir=figure_dir,
    #                             is_sym_psd=A_is_sym_psd,
    #                             stream_size=size,
    #                             window_size=size,
    #                             col_permutation=None,
    #                             k=k,
    #                             track_reconstruction_error=True,
    #                             use_true_matrix=use_true_matrix,
    #                             )
    # #                 S, Vt = isvd(A_csr, 
    # #                              S_exact, Vt_exact, U_exact, 
    # #                              row_permutation=permutations[row_permutation], 
    # #                              name=matrix_postfix + "_" + row_permutation + "_" + "col_perm" + "_" + f"size_{size}_old",
    # # #                              name=matrix_postfix + "_" + row_permutation + "_" + f"size_{size}",
    # #                              figure_dir=figure_dir,
    # #                              is_sym_psd=A_is_sym_psd,
    # #                              stream_size=size,
    # #                              window_size=size,
    # #                              col_permutation=permutations[row_permutation])
    #         #         for num_Vs in [1, 2, 4, 10, 20, 50, 100]:
    #                 if row_permutation != "original" and not "random" in row_permutation:
    #                     continue

    #                 for num_Vs in [1, 10, 100]:
    #                 # for num_Vs in [10]:
    #                     for is_reversed in [False]:
    #                         print(f"{row_permutation}: num_V={num_Vs}")
    #                         name = matrix_postfix + "_new_" + f"Vapprox_withS_{num_Vs}_" + row_permutation  + "_"
    #                         name += "reversed_" if is_reversed else "" 
    #                         name += "true_" if use_true_matrix else ""
    #                         name += f"size_{size}_k_{k}"
    #                         S, Vt = isvd(A_csr, 
    #                                     S_exact, Vt_exact, U_exact,
    #                                     row_permutation=permutations[row_permutation], 
    #                                     #  name=matrix_postfix + "_new_" + f"Vapprox_withS_{num_Vs}_" + row_permutation + "_reversed_" + f"size_{size}_k_{k}",
    #                                     name=name, # row perm only initially
    #                                     figure_dir=figure_dir,
    #                                     is_sym_psd=A_is_sym_psd,
    #                                     num_Vs=num_Vs,
    #                                     with_S=True,
    #                                     stream_size=size,
    #                                     window_size=size,
    #                                     k=k,
    #                                     reverse=is_reversed, # False default, for decreasing norm approximations
    #                                     track_reconstruction_error=True,
    #                                     use_true_matrix=use_true_matrix,
    #                                     )
    
    else:
        size = 100
        for row_permutation in permutations:
            if matrix_name == "hyperboloid" or "kronecker" in matrix_name or A_csr.shape[1] < 5e5:
                S, Vt = isvd(A_csr, 
                             S_exact, Vt_exact, U_exact,
                             row_permutation=permutations[row_permutation], 
                             name=matrix_postfix + "_" + row_permutation + "_" + f"size_{size}",
                             figure_dir=figure_dir,
                             is_sym_psd=A_is_sym_psd,
                             stream_size=size,
                             window_size=size)
    #         for num_Vs in [1, 2, 4, 10, 20, 50, 100]:
            if not (matrix_name == "hyperboloid" or "kronecker" in matrix_name or A_csr.shape[1] < 5e5):
                num_Vs = 1
                S, Vt = isvd(A_csr, 
                             S_exact, Vt_exact, U_exact,
                             row_permutation=permutations[row_permutation], 
                             name=matrix_postfix + "_" + f"Vapprox_withS_{num_Vs}_" + row_permutation, # row perm only initially
                             figure_dir=figure_dir,
                             is_sym_psd=A_is_sym_psd,
                             num_Vs=num_Vs,
                             with_S=True)
                continue
            for num_Vs in [1, 10, 100]:
                S, Vt = isvd(A_csr, 
                             S_exact, Vt_exact, U_exact,
                             row_permutation=permutations[row_permutation], 
                             name=matrix_postfix + "_" + f"Vapprox_withS_{num_Vs}_" + row_permutation + "_" + f"size_{size}", # row perm only initially
                             figure_dir=figure_dir,
                             is_sym_psd=A_is_sym_psd,
                             num_Vs=num_Vs,
                             with_S=True,
                             stream_size=size,
                             window_size=size)
                # S, Vt = isvd(A_csr, 
                #              S_exact, Vt_exact, U_exact,
                #              row_permutation=permutations[row_permutation], 
                #              name=matrix_postfix + "_" + f"Vapprox_withS_reversed_{num_Vs}_" + row_permutation, # row perm only initially
                #              figure_dir=figure_dir,
                #              is_sym_psd=A_is_sym_psd,
                #              num_Vs=num_Vs,
                #              with_S=True,
                #              reverse=True)
                # S, Vt = isvd(A_csr, 
                #              S_exact, Vt_exact, U_exact,
                #              row_permutation=permutations[row_permutation], 
                #              name=matrix_postfix + "_" + f"Vapprox_{num_Vs}_" + row_permutation, # row perm only initially
                #              figure_dir=figure_dir,
                #              is_sym_psd=A_is_sym_psd,
                #              num_Vs=num_Vs,
                #              with_S=False,
                #              reverse=False)
                # S, Vt = isvd(A_csr, 
                #              S_exact, Vt_exact, U_exact,
                #              row_permutation=permutations[row_permutation], 
                #              name=matrix_postfix + "_" + f"Vapprox_reversed_{num_Vs}_" + row_permutation, # row perm only initially
                #              figure_dir=figure_dir,
                #              is_sym_psd=A_is_sym_psd,
                #              num_Vs=num_Vs,
                #              with_S=False,
                #              reverse=True)

    
    ## Balanced ##
    # A_normalized = normalize_csr_matrix_rows(A_csr)

    # if A_csr.shape[1] < 5e4:
        # start_time = time.time()
        # U_exact, S_exact, Vt_exact = sp.linalg.svd(A_normalized.todense(), lapack_driver="gesdd")
        # exact_time = time.time() - start_time
        # print("Exact:", exact_time)

    # A_squared = A_normalized.copy()
    # A_squared.data **= 2
    # weights = np.asarray(np.sqrt(np.sum(A_squared, axis=1))).reshape(-1)
    # permutations = {
    #     "original": None,
    #     "reversed_original": np.arange(len(weights))[::-1],
    #     "random_uniform": np.random.permutation(len(weights)),
    #     # "decreasing_exactV_norm": np.argsort(np.sum(Vt_exact**2, axis=0)).reshape(-1)[::-1],
    #     # "increasing_exactV_norm": np.argsort(np.sum(Vt_exact**2, axis=0)).reshape(-1),
    #     "decreasing_exactV_norm_100": np.argsort(np.sum(Vt_exact[:100, :]**2, axis=0)).reshape(-1)[::-1],
    #     "increasing_exactV_norm_100": np.argsort(np.sum(Vt_exact[:100, :]**2, axis=0)).reshape(-1),
    #     "decreasing_exactV_norm_10": np.argsort(np.sum(Vt_exact[:10, :]**2, axis=0)).reshape(-1)[::-1],
    #     "increasing_exactV_norm_10": np.argsort(np.sum(Vt_exact[:10, :]**2, axis=0)).reshape(-1),
    #     "decreasing_exactV_norm_1": np.argsort(np.sum(Vt_exact[:1, :]**2, axis=0)).reshape(-1)[::-1],
    #     "increasing_exactV_norm_1": np.argsort(np.sum(Vt_exact[:1, :]**2, axis=0)).reshape(-1),
    # }
    # for row_permutation in permutations:
    #     S, Vt = isvd(A_normalized, 
    #                  S_exact, Vt_exact, U_exact,
    #                  row_permutation=permutations[row_permutation], 
    #                  name=matrix_postfix + "_" + "balance" + "_" + row_permutation,
    #                  figure_dir=figure_dir,
    #                  is_sym_psd=A_is_sym_psd)




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