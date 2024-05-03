import numpy as np
from dolfinx import fem, mesh
from ..utils import run_example, parse_args, msfem

N, n = 0, 0


def find_insertions(x, horizontal=False, vertical=False):
    global N, n
    m = N * n
    h = 1 / m
    margin = h
    xcs, ycs = np.meshgrid(np.linspace(0, 1, N + 1), np.linspace(0, 1, N + 1))
    xcs, ycs = xcs.flatten(), ycs.flatten()
    mask = (xcs > 0) & (xcs < 1) & (ycs > 0) & (ycs < 1)
    xcs, ycs = xcs[mask], ycs[mask]
    xs, ys = x[0], x[1]

    in_mask = np.zeros(len(xs), dtype=bool)
    for xc, yc in zip(xcs, ycs):
        if horizontal:
            horizontal_check = (
                np.isclose(np.abs(xc - xs), (2 * margin))
                | (np.abs(xc - xs) < (2 * margin))
            ) & (np.isclose(np.abs(yc - ys), margin) | (np.abs(yc - ys) < margin))
            in_mask = in_mask | horizontal_check
        if vertical:
            vertical_check = (
                np.isclose(np.abs(xc - xs), margin) | (np.abs(xc - xs) < margin)
            ) & (
                np.isclose(np.abs(yc - ys), (2 * margin))
                | (np.abs(yc - ys) < (2 * margin))
            )
            in_mask = in_mask | vertical_check

    return in_mask


def coeff_eval(x, y):
    global N, n
    m = N * n
    h = 1 / m
    margin = h
    xcs, ycs = np.meshgrid(np.linspace(0, 1, N + 1), np.linspace(0, 1, N + 1))
    xcs, ycs = xcs.flatten(), ycs.flatten()
    mask = (xcs > 0) & (xcs < 1) & (ycs > 0) & (ycs < 1)
    xcs, ycs = xcs[mask], ycs[mask]

    horizontal_check = (
        np.isclose(np.abs(xcs - x), (2 * margin)) | (np.abs(xcs - x) < (2 * margin))
    ) & (np.isclose(np.abs(ycs - y), margin) | (np.abs(ycs - y) < margin))
    vertical_check = (
        np.isclose(np.abs(xcs - x), margin) | (np.abs(xcs - x) < margin)
    ) & (np.isclose(np.abs(ycs - y), (2 * margin)) | (np.abs(ycs - y) < (2 * margin)))

    E1, E2, Nu = 1e4, 1, 0.3
    Lambda1 = E1 * Nu / ((1 + Nu) * (1 - 2 * Nu))
    Lambda2 = E2 * Nu / ((1 + Nu) * (1 - 2 * Nu))

    if np.any(horizontal_check & vertical_check):
        coeff = 2 * Lambda1
    elif np.any(horizontal_check | vertical_check):
        coeff = Lambda1
    else:
        coeff = Lambda2

    return coeff


def coeff_fem(msh):
    global N, n

    Omega_h = mesh.locate_entities(
        msh, msh.topology.dim, lambda x: find_insertions(x, horizontal=True)
    )
    Omega_v = mesh.locate_entities(
        msh, msh.topology.dim, lambda x: find_insertions(x, vertical=True)
    )
    Omega_int = np.intersect1d(Omega_h, Omega_v)

    V = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim,)))
    E = fem.Function(V, name="E")
    E.x.array[:] = 1
    E.x.array[2 * Omega_h] = 1e4
    E.x.array[2 * Omega_v + 1] = 1e4
    E.x.array[2 * Omega_int] = E.x.array[2 * Omega_int + 1] = 1e4
    Nu = 0.3

    return E, Nu


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
        problem_type=msfem.NullSpaceType.LINEAR_ELASTICITY,
        output=args.output,
    )


if __name__ == "__main__":
    args = parse_args(
        "Orthotropic problem with high coefficient inclusions around the coarse nodes."
    )
    main(args)
