import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spilu, splu, bicgstab, LinearOperator


def cg(
    A, b, tol=1e-5, maxiter=1000, x0=None, return_lanczos=False, M=None, callback=None
):
    """A conjugate gradiente (CG) linear solver based on
    algorithms 6.18 and 9.1 from Saad (2003).

    Args:
        A (sparse matrix): The matrix of the linear system.
        b (numpy.ndarray): The RHS vector.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-5.
        maxiter (int, optional): Maximum number of iterations. Defaults to 1000.
        x0 (numpy.ndarray, optional): The initial guess for the solution.
            If not provided, it is initialized to a vector of zeros. Defaults to None.
        return_lanczos (bool, optional): Compute and return Lanczo's tridiagonal
            matrix, used for a condition number estimate. Defaults to False.
        M (sparse matrix or SciPy LinearOperator, optional): A preconditioner to applied to
            the system matrix. The object should approximate the inverse of `A` and
            implement a matrix-vector product through the method `matvec`.
        callback (callable, optional): User-supplied function to call after each iteration.
            It is called as callback(xk), where xk is the current iterate of the solution.

    Returns:
        numpy.ndarray: The converged solution. If `return_lanczos` is set to `True`, then
        the tridiagonal matrix is also returned.
    """
    # Initialization of the iterative variables.
    r_curr = b - A @ x0 if x0 is not None else b.copy()
    z_curr = M.matvec(r_curr) if M is not None else r_curr.copy()
    p_curr = z_curr.copy() if M is not None else r_curr.copy()
    x_curr = x0.copy() if x0 is not None else np.zeros(len(b))

    # An array that stores the values of the CG coefficients.
    # This will be used to assemble Lanczo's tridiagonal matrix.
    alpha = np.zeros(maxiter)
    beta = np.zeros(maxiter)

    for i in range(maxiter):
        alpha[i] = np.dot(r_curr, z_curr) / np.dot(A @ p_curr, p_curr)
        x_curr += alpha[i] * p_curr
        r_next = r_curr - alpha[i] * A @ p_curr
        z_next = M.matvec(r_next) if M is not None else r_next[:]
        beta[i] = np.dot(r_next, z_next) / np.dot(r_curr, z_curr)
        p_curr = z_next + beta[i] * p_curr
        r_curr = r_next
        z_curr = z_next

        if callback is not None:
            callback(x_curr)

        if np.linalg.norm(r_curr) < tol:
            alpha = alpha[: i + 1]
            beta = beta[: i + 1]
            break

    if return_lanczos:
        T_m_main_diag = 1 / alpha
        T_m_main_diag[1:] += beta[1:] / alpha[1:]
        T_m_lower_diag = T_m_upper_diag = np.sqrt(beta[1:]) / alpha[:-1]
        T_m = diags(
            [T_m_lower_diag, T_m_main_diag, T_m_upper_diag],
            offsets=[-1, 0, 1],
            format="csc",
        )
        return x_curr, T_m

    return x_curr


def two_stage_ms_solver(
    A, b, Phi, tol=1e-5, maxiter=1000, x0=None, callback=None, n_s=10
):
    """A two-stage multiscale solver based on the iterative procedure introduced in
     Wang, Hajibeygi and Tchelepi (2014) (https://doi.org/10.1016/j.jcp.2013.11.024).

    Args:
        A (sparse matrix): The matrix of the linear system.
        b (numpy.ndarray): The RHS vector.
        Phi (sparse matrix): The multiscale basis functions as an operator.
        tol (float, optional): Residual tolerance for convergence. Defaults to 1e-5.
        maxiter (int, optional): Maximum number of iterations. Defaults to 1000.
        x0 (numpy.ndarray, optional): Initial solution guess.
            If not provided, the prolonged multiscale solution is used. Defaults to None.
        callback (callable, optional): User-supplied function to call after each iteration.
            It is called as callback(xk), where xk is the current iterate of the solution. Defaults to None.
        n_s (int, optional): Number of smoothing steps used at the local stage. Defaults to 10.

    Returns:
        numpy.ndarray: The converged solution.
    """
    M_ilu = spilu(A)
    M_ilu_op = LinearOperator(A.shape, lambda x: M_ilu.solve(x))
    A_c_lu = splu(Phi @ (A @ Phi.T))

    x_curr = x0.copy() if x0 is not None else Phi.T @ (A_c_lu.solve(Phi @ b))
    r_curr = b - A @ x_curr

    for _ in range(maxiter):
        # Global stage
        dx_global = Phi.T @ (A_c_lu.solve(Phi @ r_curr))
        r_global = r_curr - A @ dx_global

        # Local stage
        dx_local, _ = bicgstab(A, r_global, M=M_ilu_op, maxiter=n_s)

        x_curr += dx_global + dx_local
        r_curr = b - A @ x_curr

        if callback is not None:
            callback(x_curr)

        if np.linalg.norm(r_curr) < tol:
            break

    return x_curr
