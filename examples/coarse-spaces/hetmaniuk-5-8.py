import numpy as np
from utils import run_example, parse_args


def c(x, y):
    """Hetmaniuk and Lehoucq (2010), Eq. 5.8."""
    sx, sy = np.sin(25 * np.pi * x), np.sin(25 * np.pi * y)
    cy = np.cos(25 * np.pi * y)
    return ((2 + 1.8 * sx) / (2 + 1.8 * cy)) + ((2 + sy) / (2 + 1.8 * sx))


def main(args):
    input_dir = "hetmaniuk-5-8"
    N = args.N + 1
    n = args.n + 1
    coarse_space = args.coarse_space
    k = args.k
    run_example(input_dir, N, n, c, coarse_space, k)


if __name__ == "__main__":
    args = parse_args(
        "Runs an example based on the problem proposed in eq. 5.8 from Hetmaniuk and Lehoucq (2010)."
    )
    main(args)
