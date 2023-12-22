import numpy as np
from utils import run_example


def c(x, y):
    """Hetmaniuk and Lehoucq (2010), Eq. 5.6."""
    return 1 / (1.2 + np.cos(32 * np.pi * x * (1 - x) * y * (1 - y)))


def main():
    N = 5
    m = 257
    input_dir = "hetmaniuk-5-6"
    run_example(input_dir, N, m, c)


if __name__ == "__main__":
    main()
