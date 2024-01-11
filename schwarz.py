import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import inv


class BaseTwoLevelASPreconditioner(object):
    """Base class used to implement the variants of the
    two-level Additive Schwarz preconditioner.
    """

    def __init__(self, A, Phi, N, n, k) -> None:
        """Constructor

        Args:
            A (sparse matrix): The matrix of the linear system.
            Phi (sparse matrix): The MsFEM prolongation operator,
                i.e., the assembled basis functions.
            N (int): Number of subdomains on each direction.
            n (int): Number of cells per subdomain on each direction.
            k (int): Number of layers for the overlap.
        """
        self.A = A
        self.Phi = Phi
        self.N = N
        self.n = n
        self.k = k

        # Number of nodes on each direction for the global domain.
        self.m = self.N * self.n + 1

        # Coarse grid size.
        self.H = 1 / self.N

        # Fine grid size.
        self.h = 1 / self.n

        # A connectivity matrix indicating which nodes are neighbors.
        self.M = (self.A != 0).astype(int)

        # A partition vector that splits the nodes into each subdmain.
        self.P = self._compute_partitions()

        # A partition vector that splits the nodes into the overlapping subdmains.
        self.P_extended = self._compute_overlapping_partitions()

    def _compute_partitions(self):
        """Sets `P`, a matrix that indicates the partition of the domain's
        nodes into the non-overlapping subdomains.
        """
        # Since each subdomain is a square, the partition is computed by
        # moving a "window" across the domain and assigning the nodes within
        # the window to the subdomain.
        ref_idx = np.arange(self.n + 1, dtype=int)
        ref_window = np.concatenate([ref_idx + j * self.m for j in range(self.n + 1)])

        row_idx = []
        col_idx = []
        P_values = []

        for i in range(self.N**2):
            # Horizontal and vertical displacement of the reference window.
            displ_horiz, displ_vert = i % self.N, i // self.N

            # The nodes in the subdomain \Omega_i.
            Omega_i = (
                ref_window + (displ_horiz * self.n) + (displ_vert * self.n * self.m)
            )

            row_idx.extend(Omega_i)
            col_idx.extend(i * np.ones(len(Omega_i), dtype=int))
            P_values.extend(np.ones(len(Omega_i)))

        return csc_matrix(
            (P_values, (row_idx, col_idx)), shape=(self.m**2, self.N**2)
        )

    def _compute_overlapping_partitions(self):
        row_idx = []
        col_idx = []

        for i in range(self.N**2):
            # First, retrieve the nodes in the subdomain.
            Omega_i = self.P[:, i].nonzero()[0]

            # Compute the overlapping subdomain.
            Omega_i_extended = self._compute_overlap(Omega_i, self.k)

            row_idx.extend(Omega_i_extended)
            col_idx.extend(i * np.ones(len(Omega_i_extended), dtype=int))

        return csc_matrix(
            (np.ones(len(row_idx)), (row_idx, col_idx)),
            shape=(self.m**2, self.N**2),
        )

    def _compute_overlap(self, Omega_i, l):
        """Computes the overlapping extension of subdomain \Omega_i
        into l levels.

        Args:
            Omega_i (numpy.ndarray): The non-overlapping subdomain.
            l (int): The number of overlapping layers.

        Returns:
            numpy.ndarray: The set of nodes comprising the extended subdomain.
        """
        if l == 1:
            Omega_i_neighbors = self.M[Omega_i, :].nonzero()[1]
            Omega_i_extended = np.union1d(Omega_i, Omega_i_neighbors)
        else:
            Omega_i_prev = self._compute_overlap(Omega_i, l - 1)
            Omega_i_prev_neighbors = self.M[Omega_i_prev, :].nonzero()[1]
            Omega_i_extended = np.union1d(Omega_i_prev, Omega_i_prev_neighbors)
        return Omega_i_extended


class TwoLevelASPreconditioner(BaseTwoLevelASPreconditioner):
    """An implementation of the two-level Additive Schwarz (AS)
    preconditioner using MsFEM basis functions as a coarse space.
    """

    def __init__(self, A, Phi, N, n, k) -> None:
        super().__init__(A, Phi, N, n, k)

    def assemble(self):
        """Assembles the two-level AS preconditioner.

        Returns:
            scipy.sparse.csc_matrix: The inverse of the two-level AS preconditioner.
        """
        # The first level AS preconditioner.
        M_as_1 = csc_matrix((self.m**2, self.m**2))
        for i in range(self.N**2):
            # First, retrieve the nodes in the subdomain.
            Omega_i = self.P[:, i].nonzero()[0]

            # Compute the overlapping subdomain.
            Omega_i_extended = self._compute_overlap(Omega_i, self.k)

            # Extract and invert the matrix block corresponding to the
            # extended subdomain.
            A_i = self.A[Omega_i_extended, Omega_i_extended[:, None]]
            A_i_inv = inv(A_i)

            # Project the local matrix back to the global domain and
            # add the result to first level preconditioner.
            row_idx, col_idx = A_i_inv.nonzero()
            A_i_prolonged = csc_matrix(
                (A_i_inv.data, (Omega_i_extended[row_idx], Omega_i_extended[col_idx])),
                shape=(self.m**2, self.m**2),
            )
            M_as_1 += A_i_prolonged

        # The second level preconditioner.
        A_0 = self.Phi @ (self.A @ self.Phi.transpose())
        M_as_2 = self.Phi.transpose() @ (inv(A_0) @ self.Phi)

        # The additive two-level preconditioner.
        M_as = M_as_1 + M_as_2

        return M_as


class TwoLevelRASPreconditioner(BaseTwoLevelASPreconditioner):
    """An implementation of the two-level Restricted Additive Schwarz
    (RAS) preconditioner using MsFEM basis functions as a coarse space.
    """

    def __init__(self, A, Phi, N, n, k) -> None:
        super().__init__(A, Phi, N, n, k)
        self.P_unique = self._compute_unique_partition()

    def _compute_unique_partition(self):
        row_idx = -np.ones(self.m**2, dtype=int)
        col_idx = -np.ones(self.m**2, dtype=int)
        it = 0

        for j in range(self.N**2):
            Omega_i = self.P[:, j].nonzero()[0]
            Omega_i_interior = Omega_i[self.P[Omega_i, :].sum(axis=1).A.flatten() == 1]
            row_idx[it : (it + len(Omega_i_interior))] = Omega_i_interior[:]
            col_idx[it : (it + len(Omega_i_interior))] = j
            it += len(Omega_i_interior)

        Gamma = np.where(self.P.sum(axis=1).A.flatten() > 1)[0]
        for xk in Gamma:
            subdomains_sharing_xk = self.P[xk, :].nonzero()[1]
            j = np.argmin([np.sum(col_idx == i) for i in subdomains_sharing_xk])
            row_idx[it] = xk
            col_idx[it] = subdomains_sharing_xk[j]
            it += 1

        return csc_matrix(
            (np.ones(self.m**2), (row_idx, col_idx)),
            shape=(self.m**2, self.N**2),
        )

    def assemble(self):
        """Assembles the two-level RAS preconditioner.

        Returns:
            scipy.sparse.csc_matrix: The inverse of the two-level RAS preconditioner.
        """
        # The first level RAS preconditioner.
        M_ras_1 = csc_matrix((self.m**2, self.m**2))
        for i in range(self.N**2):
            # First, retrieve the nodes in the subdomain.
            Omega_i = self.P[:, i].nonzero()[0]

            # Retrieve the nodes in the unique partition.
            Omega_i_unique = self.P_unique[:, i].nonzero()[0]

            # Compute the overlapping extension of the subdomain.
            Omega_i_extended = self._compute_overlap(Omega_i, self.k)

            # Extract and invert the matrix block corresponding to the
            # extended subdomain.
            A_i = self.A[Omega_i_extended, Omega_i_extended[:, None]]
            A_i_inv = inv(A_i)

            # Project the local inverse to the global domain.
            row_idx, col_idx = A_i_inv.nonzero()
            A_i_prolonged_ext = csc_matrix(
                (A_i_inv.data, (Omega_i_extended[row_idx], Omega_i_extended[col_idx])),
                shape=(self.m**2, self.m**2),
            )

            # Add to the total preconditioner the contribution of the nodes
            # within the non-overlapping subdomain.
            M_ras_1[Omega_i_unique, Omega_i_unique[:, None]] += A_i_prolonged_ext[
                Omega_i_unique, Omega_i_unique[:, None]
            ]

        # The second level preconditioner.
        A_0 = self.Phi @ (self.A @ self.Phi.transpose())
        M_ras_2 = self.Phi.transpose() @ (inv(A_0) @ self.Phi)

        # The additive two-level preconditioner.
        M_ras = M_ras_1 + M_ras_2

        return M_ras


class TwoLevelSASPreconditioner(BaseTwoLevelASPreconditioner):
    """An implementation of the two-level Scaled Additive Schwarz
    (SAS) preconditioner using MsFEM basis functions as a coarse space.
    """

    def __init__(self, A, Phi, N, n, k) -> None:
        super().__init__(A, Phi, N, n, k)

    def assemble(self):
        """Assembles the two-level SAS preconditioner.

        Returns:
            scipy.sparse.csc_matrix: The inverse of the two-level SAS preconditioner.
        """
        # The first level SAS preconditioner.
        M_sas_1 = csc_matrix((self.m**2, self.m**2))
        for i in range(self.N**2):
            # First, retrieve the nodes in the subdomain.
            Omega_i = self.P[:, i].nonzero()[0]

            # Compute the overlapping extension of the subdomain.
            Omega_i_extended = self._compute_overlap(Omega_i, self.k)

            # Extract and invert the matrix block corresponding to the
            # extended subdomain.
            A_i = self.A[Omega_i_extended, Omega_i_extended[:, None]]
            A_i_inv = inv(A_i)

            # Project the local inverse to the global domain.
            row_idx, col_idx = A_i_inv.nonzero()
            A_i_prolonged_ext = csc_matrix(
                (A_i_inv.data, (Omega_i_extended[row_idx], Omega_i_extended[col_idx])),
                shape=(self.m**2, self.m**2),
            )

            # Scale the local contribution of each node in the non-overlapping
            # subdomain by their inverse multiplicity.
            node_multiplicity = np.asarray(self.P[Omega_i, :].sum(axis=1)).flatten()
            W = diags(node_multiplicity**-1)
            A_i_scaled = W @ A_i_prolonged_ext[Omega_i, Omega_i[:, None]]

            # Add to the total preconditioner the contribution of the nodes
            # within the non-overlapping subdomain.
            M_sas_1[Omega_i, Omega_i[:, None]] += A_i_scaled

        # The second level preconditioner.
        A_0 = self.Phi @ (self.A @ self.Phi.transpose())
        M_sas_2 = self.Phi.transpose() @ (inv(A_0) @ self.Phi)

        # The additive two-level preconditioner.
        M_sas = M_sas_1 + M_sas_2

        return M_sas
