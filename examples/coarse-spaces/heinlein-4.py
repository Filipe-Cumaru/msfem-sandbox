import numpy as np
from utils import run_example, parse_args


def c(x, y):
    """Heinlein (2016), Problem 4, Eq. 4.36"""
    sx, sy = np.sin(25 * np.pi * x), np.sin(25 * np.pi * y)
    cy = np.cos(25 * np.pi * y)
    return ((2 + 1.99 * sx) / (2 + 1.99 * cy)) + ((2 + sy) / (2 + 1.99 * sx))


def main(args):
    input_dir = "heinlein-4"
    N = args.N + 1
    n = args.n + 1
    coarse_space = args.coarse_space
    run_example(input_dir, N, n, c, coarse_space)


if __name__ == "__main__":
    args = parse_args(
        "Runs an example based on the problem proposed in Problem 4, Eq. 4.36 from Heinlein (2016)."
    )
    main(args)
