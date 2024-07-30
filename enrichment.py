from scipy.sparse import csc_matrix, vstack
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.csgraph import connected_components
from schwarz import TwoLevelASPreconditioner
from solvers import cg, IterationsCounter
import numpy as np
import warnings


def compute_convergence_rate(r0, rk, n):
    return np.log10(np.linalg.norm(r0) / np.linalg.norm(rk)) / n


def global_coarse_space_enrichment(
    A,
    Phi,
    nu,
    eps,
    alpha,
    Gamma,
    Lambda,
    max_enrich_rounds,
    N,
    n,
    overlap,
):
    """Implementation of the coarse space enrichment procedure proposed by Manea et al. (2016)
    DOI 10.3997/2214-4609.201601894.

    Args:
        A (sparse matrix): Linear system matrix.
        Phi (sparse matrix): Prolongation operator, i.e., the initial set of coarse basis functions.
        nu (int): Maximum number of CG iterations performed to find the error modes.
        eps (float): Tolerance used in the CG iterations to find the error modes.
        alpha (float): Scaling factor used in the cut-off term for the relevant error modes.
        Gamma (float): Target convergence rate.
        Lambda (int): Maximum dimension of the coarse space, i.e., the maximum number of coarse basis functions.
        max_enrich_rounds (int): Maximum number of enrichment rounds.
        N (int): Number of subdomains on each direction.
        n (int): Number of nodes on each direction within a subdomain.
        overlap (int): Number of layers of overlap for the one-level Schwarz preconditioner.

    Returns:
        sparse matrix: The enriched version of the prolongation operator.
    """
    if max_enrich_rounds <= 0:
        raise ValueError("max_enrich_rounds must be greater than 0.")

    # Setup the two-level OAS preconditioner using the initial
    # coarse basis functions.
    precond = TwoLevelASPreconditioner(A, Phi, N, n, k=overlap)
    M = LinearOperator(A.shape, lambda x: precond.apply(x))

    # Variables used for the CG iterations run to detect error modes.
    num_dofs = A.shape[0]
    it_counter = IterationsCounter(disp=False)
    b0 = np.zeros(num_dofs)

    # Initialize the convergence rates at each enrichment round and
    # the initial random guesses used to compute the new coarse
    # basis functions.
    conv_rates = []
    init_guesses = []
    coarse_space_dim = Phi.shape[0]

    # Phi_enriched stores the result of each enrichment round and
    # Phi_prev is an auxiliary variable that stores the result of the
    # last successful enrichment round. Hence, if a round fails, we can
    # rollback to the last sucessful one.
    Phi_prev, Phi_enriched = Phi.copy(), None

    for _ in range(max_enrich_rounds):
        # Run the CG iterations to detect the error modes.
        it_counter.niter = 0
        x0 = np.random.random(num_dofs)
        xk, _ = cg(A, b0, tol=eps, maxiter=nu, x0=x0, M=M, callback=it_counter)

        # Filter the relevant error components.
        agg_mask = np.abs(xk) > (alpha * np.linalg.norm(xk, ord=np.inf))
        agg_idx = agg_mask.nonzero()[0]

        # Aggregate the connected error components by checking their
        # connectivity in the system matrix.
        A_agg = A[agg_idx[:, None], agg_idx]
        n_agg, agg_labels = connected_components(A_agg)

        # Update the current set of coarse basis functions with the
        # new components.
        B_row_idx, B_col_idx, B_values = agg_labels[:], agg_idx[:], xk[agg_mask]
        B = csc_matrix((B_values, (B_row_idx, B_col_idx)), shape=(n_agg, num_dofs))
        Phi_curr = vstack((Phi_prev, B), format="csc")

        # Update the preconditioner with the new coarse basis functions.
        precond.update_coarse_level(Phi_curr)

        # Check if the new set of basis functions is an actual improvement
        # compared to the previous rounds.
        # The new coarse operator is considered better if its solution
        # using the initial guesses from previous rounds gives a better
        # convergence rate each time.
        for j, x0_j in enumerate(init_guesses):
            it_counter.niter = 0
            xj_mod, _ = cg(
                A, b0, tol=eps, maxiter=nu, x0=x0_j, M=M, callback=it_counter
            )
            r0_j, rk_j_mod = A @ x0_j, A @ xj_mod
            gamma_j_mod = compute_convergence_rate(r0_j, rk_j_mod, it_counter.niter)

            # If the current enrichment round is not an improvement compared
            # to a previous one, then rollback.
            if gamma_j_mod < conv_rates[j]:
                Phi_curr = Phi_prev.copy()
                precond.update_coarse_level(Phi_curr)
                break
        else:
            # If the current enrichment round is an improvement, then store
            # its parameters (initial random guess and convergence rate).
            it_counter.niter = 0
            xk, _ = cg(A, b0, tol=eps, maxiter=nu, x0=x0, M=M, callback=it_counter)
            r0, rk = A @ x0, A @ xk
            gamma_k = compute_convergence_rate(r0, rk, it_counter.niter)

            init_guesses.append(x0)
            conv_rates.append(gamma_k)
            Phi_prev = Phi_curr.copy()
            coarse_space_dim = Phi_curr.shape[0]

            # Check if the enrichment satisfies the desirable parameters.
            if conv_rates[-1] >= Gamma or coarse_space_dim >= Lambda:
                Phi_enriched = Phi_curr
                break
    else:
        warnings.warn("Maximum number of enrichment rounds reached.")
        Phi_enriched = Phi_curr

    return Phi_enriched
