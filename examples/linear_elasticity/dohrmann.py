from ..utils import run_example, parse_args, msfem


def coeff_eval(x, y):
    E, Nu = 1, 0.3
    Lambda = E * Nu / ((1 + Nu) * (1 - 2 * Nu))
    return Lambda


def coeff_fem(_):
    return [1.0, 1.0], 0.3


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
        problem_type=msfem.NullSpaceType.LINEAR_ELASTICITY,
        output=args.output
    )


if __name__ == "__main__":
    args = parse_args(
        "Runs an example based on Dohrmann & Widlund (2017), Table 1."
    )
    main(args)
