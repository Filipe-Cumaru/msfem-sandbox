import numpy as np
import ufl
from ..utils import run_example, parse_args, msfem


def coeff_eval(x, y):
    return 1 / (1.2 + np.cos(32 * np.pi * x * (1 - x) * y * (1 - y)))


def coeff_fem(msh):
    x = ufl.SpatialCoordinate(msh)
    c = (1.2 + ufl.cos(32 * ufl.pi * x[0] * (1 - x[0]) * x[1] * (1 - x[1]))) ** -1
    return c


def main(args):
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
        "Runs an example based on the problem proposed in eq. 5.6 from Hetmaniuk & Lehoucq (2010)."
    )
    main(args)
