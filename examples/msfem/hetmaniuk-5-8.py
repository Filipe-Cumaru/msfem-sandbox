import numpy as np
from utils import run_example


def c(x, y):
    """Hetmaniuk and Lehoucq (2010), Eq. 5.8."""
    sx, sy = np.sin(25 * np.pi * x), np.sin(25 * np.pi * y)
    cy = np.cos(25 * np.pi * y)
    return ((2 + 1.8 * sx) / (2 + 1.8 * cy)) + ((2 + sy) / (2 + 1.8 * sx))


def main():
    N = 5
    m = 257
    input_dir = "hetmaniuk-5-8"
    run_example(input_dir, N, m, c)


if __name__ == "__main__":
    main()
