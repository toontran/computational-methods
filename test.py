import os
import numpy as np

# for iteration in range(last_available_file_number+1):
iteration = 0
dir_path = "~/pj/computational-methods/output/kernel_stocks_5000_10.0_isvd_random_uniform_size_100_k_10_reservoir_greedy/"
file_path = os.path.join(dir_path, f'spectrum_data_{iteration}.npz')
# try:
data = np.load(file_path, allow_pickle=True)
print(data['current_total'], data['throw'], data['iteration'])

iteration = 49
dir_path = "~/pj/computational-methods/output/kernel_stocks_5000_10.0_isvd_random_uniform_size_100_k_10_reservoir_greedy/"
file_path = os.path.join(dir_path, f'spectrum_data_{iteration}.npz')
# try:
data = np.load(file_path, allow_pickle=True)
print(data['current_total'], data['throw'], data['iteration'])
    