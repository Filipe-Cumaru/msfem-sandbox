import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from msfem import NullSpaceType


class BaseTwoLevelASPreconditioner(object):
    """Base class used to implement the variants of the
    two-level Additive Schwarz preconditioner.
    """

    def __init__(
        self, A, Phi, N, n, k, null_space_type=NullSpaceType.DIFFUSION, dofs_map=None
    ) -> None:
        """Constructor

        Args:
            A (sparse matrix): The matrix of the linear system.
            Phi (sparse matrix): The MsFEM prolongation operator,
                i.e., the assembled basis functions.
            N (int): Number of subdomains on each direction.
            n (int): Number of cells per subdomain on each direction.
            k (int): Number of layers for the overlap.
            null_space_type(NullSpaceType, optional): The kind of problem solved.
            Optional, defaults to NullSpaceType.DIFFUSION.
            dofs_map(numpy.ndarray, optional): A map from the grid nodes to the DoFs of the problem
            as a # nodes X # dofs numpy array. Optional, defaults to None.
        """
        self.A = A
        self.Phi = Phi
        self.N = N
        self.n = n
        self.k = k

        # Number of nodes on each direction for the global domain.
        self.m = self.N * self.n + 1

        if not isinstance(null_space_type, NullSpaceType):
            raise ValueError(
                "Invalid null space type. Must be taken from the enumeration NullSpaceType."
            )
        self.null_space_type = null_space_type

        if self.null_space_type is NullSpaceType.DIFFUSION:
            self.dofs_map = (
                dofs_map
                if dofs_map is not None
                else np.arange(self.m**2).reshape((self.m**2, 1))
            )
            self.num_dofs = 1
        elif self.null_space_type is NullSpaceType.LINEAR_ELASTICITY:
            self.dofs_map = (
                dofs_map
                if dofs_map is not None
                else np.array([[2 * i, 2 * i + 1] for i in range(self.m**2)])
            )
            self.num_dofs = 2

        # A connectivity matrix indicating which nodes are neighbors.
        dofs = self.dofs_map[:, 0].flatten()
        self.M = (self.A[dofs[:, None], dofs] != 0).astype(int)

        # The boundary nodes extracted from the system matrix.
        self.boundary_nodes = np.where(self.M.sum(axis=1).A.flatten() == 1)[0]

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

        return csc_matrix((P_values, (row_idx, col_idx)), shape=(self.m**2, self.N**2))

    def _compute_overlapping_partitions(self):
        row_idx = []
        col_idx = []

        for i in range(self.N**2):
            # First, retrieve the nodes in the subdomain.
            Omega_i = self.P[:, i].nonzero()[0]

            # Compute the overlapping subdomain.
            Omega_i_extended = self._compute_overlap(Omega_i, self.k)

            # Filter the boundary nodes in the subdomain so their value is preserved.
            Omega_i_extended = np.setdiff1d(Omega_i_extended, self.boundary_nodes)

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

    def __init__(
        self, A, Phi, N, n, k, null_space_type=NullSpaceType.DIFFUSION, dofs_map=None
    ) -> None:
        super().__init__(
            A, Phi, N, n, k, null_space_type=null_space_type, dofs_map=dofs_map
        )
        self.A_i_lu_decompostions = self._compute_local_lu_decompositions()
        self.A_0_lu = splu(self.Phi @ (self.A @ self.Phi.transpose()))

    def apply(self, x):
        """Applies the two-level AS preconditioner to the current iterate
        of a solver.

        Args:
            x (numpy.ndarray): The current iterate of the solver.

        Returns:
            numpy.ndarray: The preconditioned iterate.
        """
        y = np.zeros(self.num_dofs * (self.m**2))

        # First level.
        for i in range(self.N**2):
            Omega_i_extended = self.P_extended[:, i].nonzero()[0]
            Omega_i_extended_dofs = self.dofs_map[Omega_i_extended, :].flatten()
            A_i_lu = self.A_i_lu_decompostions[i]
            x_i = x[Omega_i_extended_dofs]
            y_i = A_i_lu.solve(x_i)
            y[Omega_i_extended_dofs] += y_i

        # Second level.
        x_0 = self.Phi @ x
        y_0 = self.A_0_lu.solve(x_0)
        y += self.Phi.transpose() @ y_0

        return y

    def _compute_local_lu_decompositions(self):
        A_lu_decompositions = []
        for i in range(self.N**2):
            Omega_i_extended = self.P_extended[:, i].nonzero()[0]
            Omega_i_extended_dofs = self.dofs_map[Omega_i_extended, :].flatten()
            A_i = self.A[Omega_i_extended_dofs, Omega_i_extended_dofs[:, None]]
            A_i_lu = splu(A_i)
            A_lu_decompositions.append(A_i_lu)
        return A_lu_decompositions
