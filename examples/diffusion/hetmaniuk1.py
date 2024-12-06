from ..utils import run_example, parse_args, msfem


def coeff_eval(x, y):
    return 1


def coeff_fem():
    return 1


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
        output=args.output
    )


if __name__ == "__main__":
    args = parse_args(
        "Runs an example based on the problem proposed in eq. 5.5 from Hetmaniuk & Lehoucq (2010)."
    )
    main(args)
