from utils import run_example, parse_args


def main(args):
    input_dir = "laplace"
    N = args.N
    n = args.n
    k = args.k
    precond_type = args.precond
    run_example(input_dir, N, n, k, precond_type)


if __name__ == "__main__":
    args = parse_args(
        "Solves the Laplace equation in a unit square domain with zero Dirichlet boundary condition."
    )
    main(args)
