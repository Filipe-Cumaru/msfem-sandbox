import numpy as np


def cg(A, b, tol=1e-5, maxiter=1000, x0=None):
    """A conjugate gradiente (CG) linear solver based on
    Algorithm 6.18 from Saad (2003).

    Args:
        A (sparse matrix): The matrix of the linear system.
        b (numpy.ndarray): The RHS vector.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-5.
        maxiter (int, optional): Maximum number of iterations. Defaults to 1000.
        x0 (numpy.ndarray, optional): The initial guess for the solution.
        If not provided, it is initialized to a vector of zeros. Defaults to None.

    Returns:
        numpy.ndarray: The converged solution.
    """
    # Initialization of the iterative variables.
    r_curr = b - A @ x0 if x0 is not None else b[:]
    p_curr = r_curr[:]
    x_curr = x0 if x0 is not None else np.zeros(len(b))

    for _ in range(maxiter):
        alpha = np.dot(r_curr, r_curr) / np.dot(A @ p_curr, p_curr)
        x_curr += alpha * p_curr
        r_next = r_curr - alpha * A @ p_curr
        beta = np.dot(r_next, r_next) / np.dot(r_curr, r_curr)
        p_curr = r_next + beta * p_curr
        r_curr = r_next
        if np.linalg.norm(r_curr) < tol:  # type: ignore
            break

    return x_curr
