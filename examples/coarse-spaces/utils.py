import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib import cm
from scipy.sparse import load_npz
from context import msfem


def parse_args(example_description):
    parser = argparse.ArgumentParser(description=example_description)
    parser.add_argument(
        "-N",
        type=int,
        help="The number of coarse cells on each direction.",
        required=True,
    )
    parser.add_argument(
        "-n",
        type=int,
        help="The number of fine cells on each direction within each coarse cell.",
        required=True,
    )
    parser.add_argument(
        "--coarse-space",
        type=str,
        help="The coarse space used to compute the basis functions.",
        choices=["msfem", "q1", "rgdsw-opt-1", "rgdsw-opt-2-2", "ams"],
        default="msfem",
        required=True,
    )
    args = parser.parse_args()
    return args


def sort_ext_indices(p, xs, ys):
    p_sorted = np.vstack((xs, ys)).T
    sorted_idx = np.zeros(len(p[0]), dtype=int)
    for i, (nx, ny) in enumerate(p_sorted):
        sorted_idx[i] = np.where((p[0] == nx) & (p[1] == ny))[0][0]
    return sorted_idx


def run_example(input_dir, N, n, c, coarse_space):
    """Runs one of the examples in the `examples` directory.

    Args:
        input_dir (string): The name of the directory containing the input files.
            Must be a subdirectory of `data`.
        N (int): The number of coarse cells on each direction.
        n (int): The number of fine cells on each direction within a coarse cell.
        c (function): A function that computes the scalar coefficient of the
            elliptic problem.
        coarse_space (string): The coarse space used to compute the basis functions.
    """
    m = (N - 1) * (n - 1) + 1

    A = load_npz(f"../../data/{input_dir}/lhs_{m - 1}x{m - 1}.npz")
    xs, ys = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))
    fine_grid = np.load(f"../../data/{input_dir}/grid_{m - 1}x{m - 1}.npy")
    idx = sort_ext_indices(fine_grid, xs.flatten(), ys.flatten())
    A = A[:, idx]
    A = A[idx, :]

    if coarse_space == "msfem":
        cs = msfem.MsFEMCoarseSpace(N, n, A, c)
    elif coarse_space == "q1":
        cs = msfem.Q1CoarseSpace(N, n, A)
    elif coarse_space == "rgdsw-opt-1":
        cs = msfem.RGDSWConstantCoarseSpace(N, n, A)
    elif coarse_space == "rgdsw-opt-2-2":
        cs = msfem.RGDSWInverseDistanceCoarseSpace(N, n, A)
    elif coarse_space == "ams":
        cs = msfem.AMSCoarseSpace(N, n, A)
    else:
        raise ValueError("Invalid coarse space choice.")

    Phi = cs.assemble_operator()

    # BASIS FUNCTION PLOT
    N_c = N - 2
    i = (N_c + 1) * (N_c // 2)
    zs = Phi[i, :].A.reshape((m, m))
    fig, ax = plt.subplots()
    ax.set_title("An example of basis function")
    surf = ax.pcolormesh(xs, ys, zs)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
