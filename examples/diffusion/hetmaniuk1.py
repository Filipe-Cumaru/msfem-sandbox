from ..utils import run_example, parse_args, msfem


def coeff_eval(x, y):
    return 1


def coeff_fem(*_):
    return 1


if __name__ == "__main__":
    args = parse_args(
        "Runs an example based on the problem proposed in eq. 5.5 from Hetmaniuk & Lehoucq (2010, DOI 10.1051/m2an/2010007)."
    )
    run_example(args, coeff_fem, coeff_eval, msfem.NullSpaceType.DIFFUSION)
