import os
import numpy as np
import matplotlib.pyplot as plt


def plot_spectrum_from_file(file_path, save_path=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = np.load(file_path)
    if 'S' not in data:
        raise KeyError("Expected key 'S' in .npz file")

    S = data['S']

    # sort descending just in case
    S = np.sort(S)[::-1]

    plt.figure()
    plt.semilogy(S)
    plt.xlabel("Index")
    plt.ylabel("Singular value (log scale)")
    plt.title("Spectrum")
    plt.grid(True)

    if save_path is None:
        base = os.path.splitext(os.path.basename(file_path))[0]
        save_path = f"{base}_spectrum.png"

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Saved figure to: {save_path}")


# Example usage
file_path = "output/USVt_exact_finan512.npz"  # replace with your manual path
plot_spectrum_from_file(file_path)

