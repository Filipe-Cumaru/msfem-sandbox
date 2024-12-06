import numpy as np
import ngsolve as ngs
from ..utils import run_example, parse_args, msfem


def coeff_eval(x, y):
    sx, sy = np.sin(25 * np.pi * x), np.sin(25 * np.pi * y)
    cy = np.cos(25 * np.pi * y)
    return ((2 + 1.99 * sx) / (2 + 1.99 * cy)) + ((2 + sy) / (2 + 1.99 * sx))


def coeff_fem():
    sx, sy = ngs.sin(25 * ngs.pi * ngs.x), ngs.sin(25 * ngs.pi * ngs.y)
    cy = ngs.cos(25 * ngs.pi * ngs.y)
    c = ((2 + 1.99 * sx) / (2 + 1.99 * cy)) + ((2 + sy) / (2 + 1.99 * sx))
    return c


def main(args):
    Nx = args.Nx
    Ny = args.Ny
    nx = args.nx
    ny = args.ny
    k = args.k
    precond = args.precond
    coarse_space = args.coarse_space
    slab_size = args.slab_size
    run_example(
        Nx,
        Ny,
        nx,
        ny,
        k,
        precond,
        coarse_space,
        slab_size,
        coeff_fem,
        coeff_eval,
        problem_type=msfem.NullSpaceType.DIFFUSION,
        output=args.output,
    )


if __name__ == "__main__":
    args = parse_args(
        "Runs an example based on the problem proposed in eq. 4.36 from Heinlein (2016)."
    )
    main(args)
