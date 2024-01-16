import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib import cm
from scipy.sparse.linalg import spsolve
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
        choices=["msfem", "q1"],
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


def compute_l2_rel_error(u, u_ref):
    return (np.sum((u - u_ref) ** 2) / np.sum(u_ref**2)) ** 0.5


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

    if coarse_space == "msfem":
        msfem_bf = msfem.MsFEMBasisFunction(N, n, c)
    elif coarse_space == "q1":
        msfem_bf = msfem.Q1BasisFunction(N, n)
    else:
        raise ValueError('The coarse space must be one of "msfem" or "q1".')

    print("--------# 1. Computing the basis functions.")
    Phi = msfem_bf.assemble_operator()

    A = load_npz(f"../../data/{input_dir}/lhs_{m - 1}x{m - 1}.npz")
    b = np.load(f"../../data/{input_dir}/rhs_{m - 1}x{m - 1}.npy")

    fine_grid = np.load(f"../../data/{input_dir}/grid_{m - 1}x{m - 1}.npy")
    xs, ys = np.meshgrid(np.linspace(0, 1, msfem_bf.m), np.linspace(0, 1, msfem_bf.m))

    idx = sort_ext_indices(fine_grid, xs.flatten(), ys.flatten())
    b = b[idx]
    A = A[:, idx]
    A = A[idx, :]

    print("--------# 2. Solving the fine-scale system.")
    u_ref = spsolve(A, b)

    print("--------# 3. Solving the multiscale system.")
    A_c = Phi @ (A @ Phi.transpose())
    b_c = Phi @ b
    x_c = spsolve(A_c, b_c)
    u_ms = Phi.transpose() @ x_c

    # MsFEM SOLUTION PLOT
    u_ms_grid = u_ms.reshape((m, m))
    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax1.plot_surface(
        xs, ys, u_ms_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    ax1.set_title("MsFEM solution")
    fig1.colorbar(surf, shrink=0.5, aspect=5)

    # BASIS FUNCTION PLOT
    bf = Phi[6, :].A.reshape((msfem_bf.m, msfem_bf.m))
    fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    ax2.set_title("An example of basis function")
    surf = ax2.plot_surface(
        xs, ys, bf, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    fig2.colorbar(surf, shrink=0.5, aspect=5)

    # FINE-SCALE SOLUTION PLOT
    u_ref_grid = u_ref.reshape((m, m))
    fig3, ax3 = plt.subplots(subplot_kw={"projection": "3d"})
    ax3.set_title("FEM solution")
    surf = ax3.plot_surface(
        xs, ys, u_ref_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    fig3.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
