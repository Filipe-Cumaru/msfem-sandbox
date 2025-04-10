import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from msfem import NullSpaceType


class OneLevelOASPreconditioner(object):
    """An implementation of the one-level overlapping additive Schwarz (OAS)
    preconditioner.
    """

    def __init__(
        self,
        A,
        Nx,
        Ny,
        nx,
        ny,
        k,
        P,
        null_space_type=NullSpaceType.DIFFUSION,
        dofs_map=None,
    ) -> None:
        """Constructor

        Args:
            A (sparse matrix): The matrix of the linear system.
            Nx (int): The number of subdomains on the x-axis direction.
            Ny (int): The number of subdomains on the y-axis direction.
            nx (int): The number of cells on the x-axis direction within each subdomain.
            ny (int): The number of cells on the y-axis direction within each subdomain.
            k (int): Number of layers for the overlap.
            null_space_type(NullSpaceType, optional): The kind of problem solved.
            Optional, defaults to NullSpaceType.DIFFUSION.
            dofs_map(numpy.ndarray, optional): A map from the grid nodes to the DoFs of the problem
            as a # nodes X # dofs numpy array. Optional, defaults to None.
        """
        self.A = A
        self.Nx = Nx
        self.Ny = Ny
        self.nx = nx
        self.ny = ny
        self.k = k

        # Number of nodes on each direction for the global domain.
        self.mx = self.Nx * self.nx + 1
        self.my = self.Ny * self.ny + 1

        if not isinstance(null_space_type, NullSpaceType):
            raise ValueError(
                "Invalid null space type. Must be taken from the enumeration NullSpaceType."
            )
        self.null_space_type = null_space_type

        if self.null_space_type is NullSpaceType.DIFFUSION:
            self.dofs_map = (
                dofs_map
                if dofs_map is not None
                else np.arange(self.mx * self.my).reshape((self.mx * self.my, 1))
            )
            self.num_dofs = 1
        elif self.null_space_type is NullSpaceType.LINEAR_ELASTICITY:
            self.dofs_map = (
                dofs_map
                if dofs_map is not None
                else np.array([[2 * i, 2 * i + 1] for i in range(self.mx * self.my)])
            )
            self.num_dofs = 2

        # A connectivity matrix indicating which nodes are neighbors.
        dofs = self.dofs_map[:, 0].flatten()
        self.M = (self.A[dofs[:, None], dofs] != 0).astype(int)

        # The boundary nodes extracted from the system matrix.
        self.boundary_nodes = np.where(self.M.sum(axis=1).A.flatten() == 1)[0]

        # A partition vector that splits the nodes into each subdmain.
        self.P = P

        # A partition vector that splits the nodes into the overlapping subdmains.
        self.P_extended = self._compute_overlapping_partitions()

        # The LU decompositions of each submatrix.
        self.A_i_lu_decompostions = self._compute_local_lu_decompositions()

    def apply(self, x):
        """Applies the single-level additive Schwarz preconditioner to the current iterate
        of a solver.

        Args:
            x (numpy.ndarray): The current iterate of the solver.

        Returns:
            numpy.ndarray: The preconditioned iterate.
        """
        y = np.zeros(x.shape[0])

        for i in range(self.P.shape[1]):  # type: ignore
            Omega_i_extended = self.P_extended[:, i].nonzero()[0]
            Omega_i_extended_dofs = self.dofs_map[Omega_i_extended, :].flatten()
            A_i_lu = self.A_i_lu_decompostions[i]
            x_i = x[Omega_i_extended_dofs]
            y_i = A_i_lu.solve(x_i)
            y[Omega_i_extended_dofs] += y_i

        return y

    def _compute_overlapping_partitions(self):
        row_idx = []
        col_idx = []

        for i in range(self.Nx * self.Ny):
            # First, retrieve the nodes in the subdomain.
            Omega_i = self.P[:, i].nonzero()[0]

            # Compute the overlapping subdomain.
            Omega_i_extended = self._compute_overlap(Omega_i, self.k)

            row_idx.extend(Omega_i_extended)
            col_idx.extend(i * np.ones(len(Omega_i_extended), dtype=int))

        return csc_matrix(
            (np.ones(len(row_idx)), (row_idx, col_idx)),
            shape=(self.mx * self.my, self.Nx * self.Ny),
        )

    def _compute_overlap(self, Omega_i, l):
        """Computes the overlapping extension of subdomain \\Omega_i
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

    def _compute_local_lu_decompositions(self):
        A_lu_decompositions = []
        for i in range(self.Nx * self.Ny):
            Omega_i_extended = self.P_extended[:, i].nonzero()[0]
            Omega_i_extended_dofs = self.dofs_map[Omega_i_extended, :].flatten()
            A_i = self.A[Omega_i_extended_dofs, Omega_i_extended_dofs[:, None]]
            A_i_lu = splu(A_i)
            A_lu_decompositions.append(A_i_lu)
        return A_lu_decompositions


class TwoLevelOASPreconditioner(OneLevelOASPreconditioner):
    """An implementation of the two-level overlapping additive Schwarz (OAS)
    preconditioner.
    """

    def __init__(
        self,
        A,
        Phi,
        Nx,
        Ny,
        nx,
        ny,
        k,
        P,
        null_space_type=NullSpaceType.DIFFUSION,
        dofs_map=None,
    ) -> None:
        super().__init__(
            A,
            Nx,
            Ny,
            nx,
            ny,
            k,
            P,
            null_space_type=null_space_type,
            dofs_map=dofs_map,
        )
        self.Phi = Phi
        self.A_0_lu = splu(self.Phi @ (self.A @ self.Phi.transpose()))

    def apply(self, x):
        """Applies the two-level AS preconditioner to the current iterate
        of a solver.

        Args:
            x (numpy.ndarray): The current iterate of the solver.

        Returns:
            numpy.ndarray: The preconditioned iterate.
        """
        # First level.
        y = super().apply(x)

        # Second level.
        x_0 = self.Phi @ x
        y_0 = self.A_0_lu.solve(x_0)
        y += self.Phi.transpose() @ y_0

        return y

    def update_coarse_level(self, Phi_new):
        self.Phi = Phi_new
        self.A_0_lu = splu(self.Phi @ (self.A @ self.Phi.transpose()))
