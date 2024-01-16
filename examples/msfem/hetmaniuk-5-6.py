import numpy as np
from utils import run_example, parse_args


def c(x, y):
    """Hetmaniuk and Lehoucq (2010), Eq. 5.6."""
    return 1 / (1.2 + np.cos(32 * np.pi * x * (1 - x) * y * (1 - y)))


def main(args):
    input_dir = "hetmaniuk-5-6"
    N = args.N + 1
    n = args.n + 1
    coarse_space = args.coarse_space
    run_example(input_dir, N, n, c, coarse_space)


if __name__ == "__main__":
    args = parse_args(
        "Runs an example based on the problem proposed in eq. 5.6 from Hetmaniuk and Lehoucq (2010)."
    )
    main(args)
