from ..utils import run_example, parse_args, msfem


def coeff_eval(*_):
    E, Nu = 1, 0.3
    Lambda = E * Nu / ((1 + Nu) * (1 - 2 * Nu))
    return Lambda


def coeff_fem(*_):
    return 1.0, 0.3


if __name__ == "__main__":
    args = parse_args(
        "Runs an example based on Dohrmann & Widlund (2017, DOI 10.1137/17M1114272), Table 1."
    )
    run_example(args, coeff_fem, coeff_eval, msfem.NullSpaceType.LINEAR_ELASTICITY)
