import numpy as np
import ngsolve as ngs
from ..utils import run_example, parse_args, msfem


def coeff_eval(x, y):
    return 1 / (1.2 + np.cos(32 * np.pi * x * (1 - x) * y * (1 - y)))


def coeff_fem(*_):
    c = (1.2 + ngs.cos(32 * ngs.pi * ngs.x * (1 - ngs.x) * ngs.y * (1 - ngs.y))) ** -1
    return c


if __name__ == "__main__":
    args = parse_args(
        "Runs an example based on the problem proposed in eq. 5.6 from Hetmaniuk & Lehoucq (2010, DOI 10.1051/m2an/2010007)."
    )
    run_example(args, coeff_fem, coeff_eval, msfem.NullSpaceType.DIFFUSION)
