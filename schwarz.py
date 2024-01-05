import numpy as np
from scipy.sparse import csc_matrix


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
                ref_window + (displ_horiz * self.N) + (displ_vert * self.n * self.m)
            )

            row_idx.extend(Omega_i)
            col_idx.extend(i * np.ones(len(Omega_i), dtype=int))
            P_values.extend(np.ones(len(Omega_i)))

        self.P = csc_matrix(
            (P_values, (row_idx, col_idx)), shape=(self.m**2, self.N**2)
        )


class TwoLevelASPreconditioner(BaseTwoLevelASPreconditioner):
    def __init__(self, A, Phi, N, n, k) -> None:
        super().__init__(A, Phi, N, n, k)

    def assemble(self):
        return super().assemble()


class TwoLevelRASPreconditioner(BaseTwoLevelASPreconditioner):
    def __init__(self, A, Phi, N, n, k) -> None:
        super().__init__(A, Phi, N, n, k)
