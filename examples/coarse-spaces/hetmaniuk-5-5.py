from utils import run_example, parse_args


def c(x, y):
    return 1


def main(args):
    input_dir = "hetmaniuk-5-5"
    N = args.N + 1
    n = args.n + 1
    coarse_space = args.coarse_space
    k = args.k
    run_example(input_dir, N, n, c, coarse_space, k)


if __name__ == "__main__":
    args = parse_args(
        "Runs an example based on the problem proposed in eq. 5.5 from Hetmaniuk and Lehoucq (2010)."
    )
    main(args)
