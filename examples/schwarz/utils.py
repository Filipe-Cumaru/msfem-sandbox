import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib import cm
from scipy.sparse.linalg import gmres
from scipy.sparse import load_npz
from context import schwarz


class IterationsCounter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print("iter %3i\trk = %s" % (self.niter, str(rk)))


def parse_args(example_description):
    parser = argparse.ArgumentParser(description=example_description)
    parser.add_argument(
        "-N", type=int, help="The number of subdomains on each direction."
    )
    parser.add_argument(
        "-n",
        type=int,
        help="The number of cells on each direction within each subdomain.",
    )
    parser.add_argument(
        "-k", type=int, help="The number of overlapping layers for each subdomain."
    )
    parser.add_argument(
        "--precond",
        type=str,
        help="The preconditioning method to be used.",
        choices=["AS", "RAS"],
    )
    args = parser.parse_args()
    return args


def sort_ext_indices(p, xs, ys):
    p_sorted = np.vstack((xs, ys)).T
    sorted_idx = np.zeros(len(p[0]), dtype=int)
    for i, (nx, ny) in enumerate(p_sorted):
        sorted_idx[i] = np.where((p[0] == nx) & (p[1] == ny))[0][0]
    return sorted_idx


def run_example(input_dir, N, n, k, precond_type):
    """Runs one of the examples in the `examples` directory.

    Args:
        input_dir (string): The name of the directory containing the input files.
            Must be a subdirectory of `data`.
        N (int): The number of subdomains on each direction.
        n (int): The number of cells on each direction within each subdomain.
        k (int): The number of overlapping layers.
        precond_type (string): The preconditioning method. Possible values: `AS` (Additive Schwarz)
            or `RAS` (Restricted Additive Schwarz).
    """
    m = N * n + 1

    A = load_npz(f"../../data/{input_dir}/lhs_{m - 1}x{m - 1}.npz")
    b = np.load(f"../../data/{input_dir}/rhs_{m - 1}x{m - 1}.npy")
    Phi = load_npz(f"../../data/{input_dir}/phi_{N}x{N}_{m - 1}x{m - 1}.npz")
    grid = np.load(f"../../data/{input_dir}/grid_{m - 1}x{m - 1}.npy")

    xs, ys = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))
    idx = sort_ext_indices(grid, xs.flatten(), ys.flatten())

    A = A[:, idx]
    A = A[idx, :]
    b = b[idx]

    print("------- 1. Initialization")
    if precond_type == "AS":
        precond = schwarz.TwoLevelASPreconditioner(A, Phi, N, n, k)
    elif precond_type == "RAS":
        precond = schwarz.TwoLevelRASPreconditioner(A, Phi, N, n, k)
    else:
        raise ValueError("The preconditioner type must one of AS or RAS.")
    print("Done!")

    print("------- 2. Preconditioner assembly")
    M_as = precond.assemble()
    print("Done!")

    print("------- 3. Solution without a preconditioner")
    u_ref, _ = gmres(A, b, callback=IterationsCounter())
    print("Done!")

    print("------- 4. Solution with a preconditioner")
    u_precond, _ = gmres(A, b, M=M_as, callback=IterationsCounter())
    print("Done!")

    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    surf1 = ax1.plot_surface(
        xs, ys, u_ref.reshape((m, m)), cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    ax1.set_title("Reference solution")
    fig1.colorbar(surf1, shrink=0.5, aspect=5)

    fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    surf2 = ax2.plot_surface(
        xs,
        ys,
        u_precond.reshape((m, m)),
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    ax2.set_title("Preconditioned solution")
    fig2.colorbar(surf2, shrink=0.5, aspect=5)

    plt.show()
