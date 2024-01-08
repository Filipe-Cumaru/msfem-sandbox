from utils import run_example, parse_args


def main(args):
    input_dir = "hetmaniuk-5-5"
    N = args.N
    n = args.n
    k = args.k
    precond_type = args.precond
    run_example(input_dir, N, n, k, precond_type)


if __name__ == "__main__":
    args = parse_args(
        "Runs an example based on the problem proposed in eq. 5.5 from Hetmaniuk and Lehoucq (2010)."
    )
    main(args)
