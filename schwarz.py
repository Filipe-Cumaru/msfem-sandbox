import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv


class BaseTwoLevelASPreconditioner(object):
    def __init__(self, A, Phi, N, n, k) -> None:
        self.A = A
        self.Phi = Phi
        self.N = N
        self.n = n
        self.k = k

        self.m = self.N * self.n + 1
        self.H = 1 / self.N
        self.h = 1 / self.n
        self.P = self._compute_partitions()
        self.M = (self.A != 0).astype(int)

    def assemble(self):
        raise NotImplementedError()

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
    def __init__(self, A, Phi, N, n, k) -> None:
        super().__init__(A, Phi, N, n, k)

    def assemble(self):
        # The first level AS preconditioner.
        M_as_1 = csc_matrix((self.m**2, self.m**2))
        for i in range(self.N**2):
            Omega_i = self.P[:, i].nonzero()[0]
            Omega_i_extended = self._compute_overlap(Omega_i, self.k)
            A_i = self.A[Omega_i_extended, Omega_i_extended[:, None]]
            A_i_inv = inv(A_i)
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
    def __init__(self, A, Phi, N, n, k) -> None:
        super().__init__(A, Phi, N, n, k)

    def assemble(self):
        # The first level RAS preconditioner.
        M_ras_1 = csc_matrix((self.m**2, self.m**2))
        for i in range(self.N**2):
            Omega_i = self.P[:, i].nonzero()[0]
            Omega_i_extended = self._compute_overlap(Omega_i, self.k)

            A_i = self.A[Omega_i_extended, Omega_i_extended[:, None]]
            A_i_inv = inv(A_i)

            row_idx, col_idx = A_i_inv.nonzero()
            A_i_prolonged_ext = csc_matrix(
                (A_i_inv.data, (Omega_i_extended[row_idx], Omega_i_extended[col_idx])),
                shape=(self.m**2, self.m**2),
            )
            M_ras_1[Omega_i, Omega_i[:, None]] += A_i_prolonged_ext[
                Omega_i, Omega_i[:, None]
            ]

        # The second level preconditioner.
        A_0 = self.Phi @ (self.A @ self.Phi.transpose())
        M_ras_2 = self.Phi.transpose() @ (inv(A_0) @ self.Phi)

        # The additive two-level preconditioner.
        M_ras = M_ras_1 + M_ras_2

        return M_ras
