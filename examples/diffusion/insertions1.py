import numpy as np
from dolfinx import fem, mesh
from ..utils import run_example, parse_args, msfem

N, n = 0, 0


def find_insertions(x):
    global N, n
    m = N * n
    h = 1 / m
    margin = 2 * h
    xcs, ycs = np.meshgrid(np.linspace(0, 1, N + 1), np.linspace(0, 1, N + 1))
    xcs, ycs = xcs.flatten(), ycs.flatten()
    mask = (xcs > 0) & (xcs < 1) & (ycs > 0) & (ycs < 1)
    xcs, ycs = xcs[mask], ycs[mask]
    xs, ys = x[0], x[1]

    in_mask = np.zeros(len(xs), dtype=bool)
    for xc, yc in zip(xcs, ycs):
        coarse_node_check = (
            np.isclose(np.abs(xc - xs), margin) | (np.abs(xc - xs) < margin)
        ) & (np.isclose(np.abs(yc - ys), margin) | (np.abs(yc - ys) < margin))
        in_mask = in_mask | coarse_node_check

    return in_mask


def coeff_eval(x, y):
    global N, n
    m = N * n
    h = 1 / m
    margin = 2 * h
    xcs, ycs = np.meshgrid(np.linspace(0, 1, N + 1), np.linspace(0, 1, N + 1))
    xcs, ycs = xcs.flatten(), ycs.flatten()
    mask = (xcs > 0) & (xcs < 1) & (ycs > 0) & (ycs < 1)
    xcs, ycs = xcs[mask], ycs[mask]

    coarse_node_check = (
        np.isclose(np.abs(xcs - x), margin) | (np.abs(xcs - x) < margin)
    ) & (np.isclose(np.abs(ycs - y), margin) | (np.abs(ycs - y) < margin))

    return 1e8 if np.any(coarse_node_check) else 1


def coeff_fem(msh):
    global N, n

    Omega_in = mesh.locate_entities(msh, msh.topology.dim, find_insertions)

    V = fem.functionspace(msh, ("DG", 0))
    c = fem.Function(V)
    c.x.array[:] = 1
    c.x.array[Omega_in] = 1e4

    return c


def main(args):
    global N, n
    N = args.N
    n = args.n
    k = args.k
    precond = args.precond
    coarse_space = args.coarse_space
    slab_size = args.slab_size
    run_example(
        N,
        n,
        k,
        precond,
        coarse_space,
        slab_size,
        coeff_fem,
        coeff_eval,
        problem_type=msfem.NullSpaceType.DIFFUSION,
    )


if __name__ == "__main__":
    args = parse_args(
        "High coefficient inclusions around the coarse nodes."
    )
    main(args)
