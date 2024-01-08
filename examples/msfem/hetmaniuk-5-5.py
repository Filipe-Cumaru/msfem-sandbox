from utils import run_example


def c(x, y):
    return 1


def main():
    N = 5
    m = 257
    input_dir = "hetmaniuk-5-5"
    run_example(input_dir, N, m, c)


if __name__ == "__main__":
    main()
