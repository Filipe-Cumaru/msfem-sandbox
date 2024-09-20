import numpy as np
import ngsolve as ngs
from ..utils import run_example, parse_args, msfem

N, n = 0, 0


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

    E1, E2, Nu = 1e4, 1, 0.3
    Lambda1 = E1 * Nu / ((1 + Nu) * (1 - 2 * Nu))
    Lambda2 = E2 * Nu / ((1 + Nu) * (1 - 2 * Nu))

    return Lambda1 if np.any(coarse_node_check) else Lambda2


def coeff_fem():
    global N, n
    E, Nu = 0, 0.3
    H, h = 1 / N, 1 / (N * n)
    l = 4 * h
    for i in range(1, N):
        for j in range(1, N):
            ox, oy = i * H, j * H
            E += ngs.IfPos(
                ngs.sqrt((ngs.x - ox + ngs.y - oy) ** 2)
                + ngs.sqrt((ngs.x - ox - ngs.y + oy) ** 2)
                - l,
                (N - 1) ** (-2),
                1e4,
            )
    return E, E, Nu


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
        "Isotropic problem with high coefficient inclusions around the coarse nodes."
    )
    main(args)
