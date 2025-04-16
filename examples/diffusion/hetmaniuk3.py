import numpy as np
import ngsolve as ngs
from ..utils import run_example, parse_args, msfem


def coeff_eval(x, y):
    sx, sy = np.sin(25 * np.pi * x), np.sin(25 * np.pi * y)
    cy = np.cos(25 * np.pi * y)
    return ((2 + 1.8 * sx) / (2 + 1.8 * cy)) + ((2 + sy) / (2 + 1.8 * sx))


def coeff_fem(*_):
    sx, sy = ngs.sin(25 * ngs.pi * ngs.x), ngs.sin(25 * ngs.pi * ngs.y)
    cy = ngs.cos(25 * ngs.pi * ngs.y)
    c = ((2 + 1.8 * sx) / (2 + 1.8 * cy)) + ((2 + sy) / (2 + 1.8 * sx))
    return c


if __name__ == "__main__":
    args = parse_args(
        "Runs an example based on the problem proposed in eq. 5.8 from Hetmaniuk & Lehoucq (2010, DOI 10.1051/m2an/2010007)."
    )
    run_example(args, coeff_fem, coeff_eval, msfem.NullSpaceType.DIFFUSION)
