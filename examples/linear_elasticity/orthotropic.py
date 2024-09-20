import numpy as np
import ngsolve as ngs
from ..utils import run_example, parse_args, msfem

N, n = 0, 0


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


def coeff_fem():
    global N, n
    Ex, Ey, Nu = 0, 0, 0.3
    H, h = 1 / N, 1 / (N * n)
    l, w = 4 * h, 2 * h
    for i in range(1, N):
        for j in range(1, N):
            ox, oy = i * H, j * H
            Ex += ngs.IfPos(
                ngs.sqrt(((ngs.x - ox) / l + (ngs.y - oy) / w) ** 2)
                + ngs.sqrt(((ngs.x - ox) / l - (ngs.y - oy) / w) ** 2)
                - 1,
                (N - 1) ** (-2),
                1e4,
            )
            Ey += ngs.IfPos(
                ngs.sqrt(((ngs.x - ox) / w + (ngs.y - oy) / l) ** 2)
                + ngs.sqrt(((ngs.x - ox) / w - (ngs.y - oy) / l) ** 2)
                - 1,
                (N - 1) ** (-2),
                1e4,
            )
    return Ex, Ey, Nu


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
