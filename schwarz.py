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
        self.P = None
        self.M = None

    def assemble(self):
        raise NotImplementedError()

    def _set_connectivity_matrix(self):
        """Sets `M`, the connectivity matrix indicating which nodes are
        neighbors, i.e., M_{ij} = 1 if i and j are neighbors.
        """
        # Internal nodes
        inodes = np.array(
            [
                i
                for i in range(self.m**2)
                if i % self.m not in (0, self.m - 1)
                and i // self.m not in (0, self.m - 1)
            ]
        )

        # Boundary nodes
        bnodes = np.setdiff1d(np.arange(self.m**2), inodes)

        corner_nodes = np.array([0, self.m - 1, self.m * (self.m - 1), self.m**2 - 1])

        # Boundary nodes split into each facet of the domain.
        left_bnodes = np.array(
            [i for i in bnodes if i not in corner_nodes and i % self.m == 0]
        )
        right_bnodes = np.array(
            [i for i in bnodes if i not in corner_nodes and i % self.m == self.m - 1]
        )
        bottom_bnodes = np.array(
            [i for i in bnodes if i not in corner_nodes and i // self.m == 0]
        )
        top_bnodes = np.array(
            [i for i in bnodes if i not in corner_nodes and i // self.m == self.m - 1]
        )

        row_idx = []
        col_idx = []

        # Internal nodes neighbors
        row_idx.extend(np.tile(inodes, 4))
        col_idx.extend(
            np.concatenate((inodes - 1, inodes + 1, inodes - self.m, inodes + self.m))
        )

        # Left boundary nodes neighbors (except corner nodes)
        row_idx.extend(np.tile(left_bnodes, 3))
        col_idx.extend(
            np.concatenate(
                (left_bnodes + 1, left_bnodes - self.m, left_bnodes + self.m)
            )
        )

        # Right boundary nodes neighbors (except corner nodes)
        row_idx.extend(np.tile(right_bnodes, 3))
        col_idx.extend(
            np.concatenate(
                (right_bnodes - 1, right_bnodes - self.m, right_bnodes + self.m)
            )
        )

        # Bottom boundary nodes neighbors (except corner nodes)
        row_idx.extend(np.tile(bottom_bnodes, 3))
        col_idx.extend(
            np.concatenate(
                (bottom_bnodes + 1, bottom_bnodes - 1, bottom_bnodes + self.m)
            )
        )

        # Top boundary nodes neighbors (except corner nodes)
        row_idx.extend(np.tile(top_bnodes, 3))
        col_idx.extend(
            np.concatenate((top_bnodes + 1, top_bnodes - 1, top_bnodes - self.m))
        )

        # Bottom left corner
        row_idx.extend([0] * 2)
        col_idx.extend([1, self.m])

        # Bottom right corner
        row_idx.extend([self.m - 1] * 2)
        col_idx.extend([self.m - 2, 2 * self.m - 1])

        # Top left corner
        row_idx.extend([self.m * (self.m - 1)] * 2)
        col_idx.extend([self.m * (self.m - 2), self.m * (self.m - 1) + 1])

        # Top right corner
        row_idx.extend([self.m**2 - 1] * 2)
        col_idx.extend([self.m * (self.m - 1) - 1, self.m**2 - 2])

        M_values = np.ones(len(row_idx), dtype=int)

        self.M = csc_matrix(
            (M_values, (row_idx, col_idx)), shape=(self.m**2, self.m**2)
        )

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

        self.P = csc_matrix(
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
