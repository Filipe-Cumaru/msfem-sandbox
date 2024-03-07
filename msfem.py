from scipy.sparse import lil_matrix, csc_matrix, eye, vstack, diags
from scipy.integrate import quad
from scipy.sparse.linalg import spsolve
from enum import Enum
import numpy as np


def set_sparse_matrix_rows_to_value(M, rows, value):
    for row in rows:
        M.data[M.indptr[row] : M.indptr[row + 1]] = value


class NullSpaceType(Enum):
    DIFFUSION = 1
    LINEAR_ELASTICITY = 2


class MSBasisFunction(object):
    def __init__(self, N, n, c):
        """Constructor method.

        Args:
            N (int): Number of coarse nodes on each direction.
            n (int): Number of nodes on each direction within each subdomain.
            c (function): A function that computes the coefficient of the Laplace
            equation.
        """
        if n < 3:
            raise ValueError("n must be greater than 3")

        self.N = N
        self.n = n
        self.m = (self.N - 1) * (self.n - 1) + 1
        self.c = c
        self.H = 1 / (self.N - 1)
        self.h = 1 / (self.m - 1)
        self.boundary_fine_nodes = [
            i
            for i in range(self.m**2)
            if i % self.m in (0, self.m - 1) or i < self.m or i >= self.m * (self.m - 1)
        ]
        self.coarse_edges = [
            (v, v + 1) for v in range(self.N**2) if v % self.N != self.N - 1
        ] + [(v, v + self.N) for v in range(self.N**2) if v // self.N < self.N - 1]

    def assemble_operator(self):
        raise NotImplementedError()


class RGDSWCoarseSpace(MSBasisFunction):
    def __init__(
        self, N, n, A, c, null_space_type=NullSpaceType.DIFFUSION, dofs_map=None
    ):
        super().__init__(N, n, c)
        self.A = A
        self.P, self.P_I, self.P_B = self._compute_partitions()

        self.coarse_nodes = np.array(
            [
                i
                for i in range(self.N**2)
                if i % self.N not in (0, self.N - 1)
                and i >= self.N
                and i < self.N * (self.N - 1)
            ]
        )
        self.coarse_nodes_global_idx = (self.coarse_nodes % self.N) * (self.n - 1) + (
            self.coarse_nodes // self.N
        ) * (self.n - 1) * self.m

        self.xs, self.ys = np.meshgrid(
            np.linspace(0, 1, self.m), np.linspace(0, 1, self.m)
        )
        self.xs, self.ys = self.xs.flatten(), self.ys.flatten()

        self.D = self._assemble_inverse_distance_matrix()

        if not isinstance(null_space_type, NullSpaceType):
            raise ValueError(
                "Invalid null space type. Must be taken from the enumeration NullSpaceType."
            )
        self.null_space_type = null_space_type

        if dofs_map is not None:
            self.dofs_map = dofs_map
        elif self.null_space_type is NullSpaceType.DIFFUSION:
            self.dofs_map = np.arange(self.m**2).reshape((self.m**2, 1))
        elif self.null_space_type is NullSpaceType.LINEAR_ELASTICITY:
            self.dofs_map = np.arange(3 * self.m**2).reshape((self.m**2, 3))

    def assemble_operator(self):
        interface_pou = self._compute_interface_pou()
        Phi = lil_matrix(interface_pou)

        for i in range((self.N - 1) ** 2):
            # The set of fine nodes in the subdomain.
            Omega_i = self.P[:, i].nonzero()[0]

            # The set of fine nodes on the subdomain's boundary.
            Omega_i_boundary = self.P_B[:, i].nonzero()[0]

            # The set of interior nodes of the subdomain.
            Omega_i_interior = self.P_I[:, i].nonzero()[0]

            # The vertices of the subdomain on the coarse scale, i.e.,
            # the corners that form the square subdomain.
            Omega_i_coarse_vertices = np.array(
                [
                    i % (self.N - 1) + (i // (self.N - 1)) * self.N,
                    i % (self.N - 1) + (i // (self.N - 1)) * self.N + 1,
                    i % (self.N - 1) + (i // (self.N - 1)) * self.N + self.N,
                    i % (self.N - 1) + (i // (self.N - 1)) * self.N + self.N + 1,
                ]
            )

            # The vertices that are actually coarse nodes according to the
            # definition in Dohrmann and Windlund (2017).
            Omega_i_coarse_nodes = np.intersect1d(
                Omega_i_coarse_vertices, self.coarse_nodes
            )

            # The portion of the system matrix corresponding to the interior nodes
            # of \Omega_i.
            Ai_II = self.A[Omega_i_interior, Omega_i_interior[:, None]]

            # The portion of the system matrix corresponding to the interaction
            # between interior and boundary nodes of \Omega_i.
            Ai_IB = self.A[Omega_i_boundary, Omega_i_interior[:, None]]

            # The contribution of the interface POU to the subdomain
            # \Omega_i (c.f. Eq. 3 from Dohrmann and Windlund (2017)).
            Psi_i = interface_pou[Omega_i_coarse_nodes, Omega_i[:, None]]

            # From the slice of the interface prolongation operator, we extract
            # the contribution of the nodes on the boundary of \Omega_i.
            boundary_mask = np.isin(Omega_i, Omega_i_boundary, assume_unique=True)
            Psi_i_B = Psi_i[boundary_mask, :]

            # Finally, the increment for the coarse nodes in \Omega_i is computed
            # as described in Dohrmann and Windlund (2017) (c.f. \Phi_{ic} in the paper).
            Phi_i_IB_inc = spsolve(Ai_II, Ai_IB @ Psi_i_B).reshape(
                (len(Omega_i_interior), len(Omega_i_coarse_nodes))
            )
            Phi[Omega_i_coarse_nodes, Omega_i_interior[:, None]] -= Phi_i_IB_inc

        # Restrict the prolongation operator to the coarse nodes.
        Phi = Phi.tocsc()
        Phi = Phi[self.coarse_nodes, :]

        return Phi

    def _compute_interface_pou(self):
        """Computes a partition of unit (POU) over the interface of the subdomains.

        Returns:
            csc_matrix: A sparse matrix representing the interface POU.
        """
        Phi_row_idx = []
        Phi_col_idx = []
        Phi_values = []

        for nc in self.coarse_nodes:
            inv_dist_to_nc = self.D[:, nc].A.flatten()
            supp_nodes = np.where(inv_dist_to_nc != 0)[0]
            inv_dist_sum = self.D[supp_nodes, :].sum(axis=1).A.flatten()
            Phi_values.extend(inv_dist_to_nc[supp_nodes] / inv_dist_sum)
            Phi_row_idx.extend(len(inv_dist_sum) * [nc])
            Phi_col_idx.extend(supp_nodes)

        Phi_values.extend(np.ones(len(self.coarse_nodes)))
        Phi_row_idx.extend(self.coarse_nodes)
        Phi_col_idx.extend(self.coarse_nodes_global_idx)

        Phi = csc_matrix(
            (Phi_values, (Phi_row_idx, Phi_col_idx)), shape=(self.N**2, self.m**2)
        )

        return Phi

    def _assemble_inverse_distance_matrix(self):
        interface_nodes = np.array(
            [
                ni
                for ni in range(self.m**2)
                if ((ni % self.m) % (self.n - 1)) == 0
                or ((ni // self.m) % (self.n - 1)) == 0
            ]
        )
        gamma = np.setdiff1d(interface_nodes, self.boundary_fine_nodes)  # type: ignore
        gamma = np.setdiff1d(gamma, self.coarse_nodes_global_idx)

        D_row_idx = []
        D_col_idx = []
        D_values = []

        for nc in self.coarse_nodes:
            Ni, Nj = nc % self.N, nc // self.N
            x_nc, y_nc = Ni * self.H, Nj * self.H

            nc_global_idx = self.coarse_nodes_global_idx[self.coarse_nodes == nc]
            supp_subdomains = self.P[nc_global_idx, :].nonzero()[1]
            supp_nodes = self.P[:, supp_subdomains].nonzero()[0]
            gamma_mask = (
                self.P_B[supp_nodes[:, None], supp_subdomains].sum(axis=1).A.flatten()
                > 1
            )
            local_interface = supp_nodes[gamma_mask]
            gamma_nodes = np.intersect1d(local_interface, gamma)

            inv_dist = self._compute_inv_distances(gamma_nodes, x_nc, y_nc, nc)

            D_values.extend(inv_dist)
            D_row_idx.extend(gamma_nodes)
            D_col_idx.extend(len(inv_dist) * [nc])

        D = csc_matrix((D_values, (D_row_idx, D_col_idx)), shape=(self.m**2, self.N**2))

        return D

    def _compute_inv_distances(self, fine_nodes, x_coarse, y_coarse, nc):
        """Computes the inverse distance from the nodes in `fine_nodes`
         to the coarse node with coordinates (x_coarse, y_coarse).

        Args:
            fine_nodes (numpy.ndarray): The nodes for which the inverse distances will be computed.
            x_coarse (float): The x coordinate of the coarse node of interest.
            y_coarse (float): The y coordinate of the coarse node of interest.
            nc (int): The local index of the coarse node.
        """
        raise NotImplementedError()

    def _compute_partitions(self):
        # Since each subdomain is a square, the partition is computed by
        # moving a "window" across the domain and assigning the nodes within
        # the window to the subdomain.
        ref_idx = np.arange(self.n, dtype=int)
        ref_window = np.concatenate([ref_idx + j * self.m for j in range(self.n)])

        row_idx = []
        col_idx = []
        R_values = []

        N_cells = self.N - 1
        n_cells = self.n - 1
        for i in range(N_cells**2):
            # Horizontal and vertical displacement of the reference window.
            displ_horiz, displ_vert = i % N_cells, i // N_cells

            # The nodes in the subdomain \Omega_i.
            Omega_i = (
                ref_window + (displ_horiz * n_cells) + (displ_vert * n_cells * self.m)
            )

            row_idx.extend(Omega_i)
            col_idx.extend(i * np.ones(len(Omega_i), dtype=int))
            R_values.extend(np.ones(len(Omega_i)))

        # The partition of the nodes into each subdomain.
        P = csc_matrix((R_values, (row_idx, col_idx)), shape=(self.m**2, N_cells**2))

        # The partition of the interior nodes for each subdomain.
        global_boundary_mask = np.ones((self.m**2, 1)).astype(bool)
        global_boundary_mask[self.boundary_fine_nodes] = False
        P_I = P.multiply((P.sum(axis=1) == 1) & global_boundary_mask).tocsc()

        # The partition of the nodes on the boundary for each subdomain.
        P_B = P - P_I

        return P, P_I, P_B


class RGDSWConstantCoarseSpace(RGDSWCoarseSpace):
    """A RGDSW coarse space as proposed by Dohrmann and Windlund (2017) in
    Eq. 1 (option 1).
    """

    def __init__(self, N, n, A):
        super().__init__(N, n, A, None)

    def _compute_inv_distances(self, fine_nodes, x_coarse, y_coarse, nc):
        return np.ones(len(fine_nodes))


class RGDSWInverseDistanceCoarseSpace(RGDSWCoarseSpace):
    """A RGDSW coarse space as proposed by Dohrmann and Windlund (2017) in
    Eq. 5 (option 2.2).
    """

    def __init__(self, N, n, A):
        super().__init__(N, n, A, None)

    def _compute_inv_distances(self, fine_nodes, x_coarse, y_coarse, nc):
        return (
            np.sqrt(
                (self.xs[fine_nodes] - x_coarse) ** 2
                + (self.ys[fine_nodes] - y_coarse) ** 2
            )
            ** -1
        )


class MsFEMCoarseSpace(RGDSWCoarseSpace):
    def __init__(self, N, n, A, c):
        super().__init__(N, n, A, c)

    def _compute_inv_distances(self, fine_nodes, x_coarse, y_coarse, nc):
        inv_dist = np.zeros(len(fine_nodes))
        cx = lambda x: 1 / self.c(x, y_coarse)
        cy = lambda y: 1 / self.c(x_coarse, y)
        for i, n in enumerate(fine_nodes):
            xn, yn = self.xs[n], self.ys[n]
            if np.isclose(xn, x_coarse):
                yL = y_coarse - self.H if yn < y_coarse else y_coarse + self.H
                inv_dist[i] = abs(quad(cy, yn, yL)[0])
            else:
                xL = x_coarse - self.H if xn < x_coarse else x_coarse + self.H
                inv_dist[i] = abs(quad(cx, xn, xL)[0])
        return inv_dist


class Q1CoarseSpace(RGDSWCoarseSpace):
    def __init__(self, N, n, A):
        super().__init__(N, n, A, None)

    def _compute_inv_distances(self, fine_nodes, x_coarse, y_coarse, nc):
        inv_dist = np.zeros(len(fine_nodes))
        for i, n in enumerate(fine_nodes):
            xn, yn = self.xs[n], self.ys[n]
            if np.isclose(xn, x_coarse):
                yL = y_coarse - self.H if yn < y_coarse else y_coarse + self.H
                inv_dist[i] = abs(yn - yL)
            else:
                xL = x_coarse - self.H if xn < x_coarse else x_coarse + self.H
                inv_dist[i] = abs(xn - xL)
        return inv_dist


class AMSCoarseSpace(RGDSWCoarseSpace):
    def __init__(self, N, n, A):
        super().__init__(N, n, A, None)
        self.vertex_nodes, self.edge_nodes, self.interior_nodes = (
            self._group_nodes_into_ams_classes()
        )
        self.G = np.hstack(
            (self.interior_nodes, self.edge_nodes, self.vertex_nodes)
        ).argsort()

    def assemble_operator(self):
        # Split the system matrix into each corresponding block.
        I_vv = eye(len(self.coarse_nodes), format="csc")

        A_ii = self.A[self.interior_nodes[:, None], self.interior_nodes]
        A_ie = self.A[self.interior_nodes[:, None], self.edge_nodes]
        A_iv = self.A[self.interior_nodes[:, None], self.vertex_nodes]

        A_ee = self.A[self.edge_nodes[:, None], self.edge_nodes]
        A_ev = self.A[self.edge_nodes[:, None], self.vertex_nodes]
        A_ei = self.A[self.edge_nodes[:, None], self.interior_nodes]

        A_ee = A_ee + diags(A_ei.sum(axis=1).A.flatten(), format="csr")

        # Compute the initial value of the basis functions on the edges.
        Phi_e = -spsolve(A_ee, A_ev)

        # Since the FEM stencil may contain nodes that do not share an
        # edge, the basis function on the edge nodes must be modified
        # to prevent growth outside the support region and preserve the
        # partition of unit.
        # First, for each coarse node, the edge nodes that are not inside
        # the support region are filtered.
        Phi_e_in_edges_mask = self.D[self.edge_nodes, :] > 0
        Phi_e = Phi_e.multiply(Phi_e_in_edges_mask)

        # Next, the partition of unit is reinforced by normalization.
        Phi_e_row_sum = 1 / Phi_e.sum(axis=1).A.flatten()
        N_ee = diags(Phi_e_row_sum, format="csc")
        Phi_e = N_ee @ Phi_e

        Phi_i = -spsolve(A_ii, A_ie @ Phi_e + A_iv)

        # Assemble all blocks and sort the operator to the natural order.
        Phi_wirebasket = vstack((Phi_i, Phi_e, I_vv), format="csc").T
        Phi = Phi_wirebasket[:, self.G]

        return Phi

    def _assemble_inverse_distance_matrix(self):
        D = super()._assemble_inverse_distance_matrix()
        D = D[:, self.coarse_nodes]
        return D

    def _compute_inv_distances(self, fine_nodes, x_coarse, y_coarse, nc):
        return np.ones(len(fine_nodes))

    def _group_nodes_into_ams_classes(self):
        vertex_nodes = self.coarse_nodes_global_idx[:]
        interface_nodes = np.array(
            [
                ni
                for ni in range(self.m**2)
                if ((ni % self.m) % (self.n - 1)) == 0
                or ((ni // self.m) % (self.n - 1)) == 0
            ]
        )
        edge_nodes = np.setdiff1d(
            np.setdiff1d(interface_nodes, self.boundary_fine_nodes), vertex_nodes
        )
        interior_nodes = np.setdiff1d(
            np.arange(self.m**2, dtype=int),
            np.union1d(edge_nodes, vertex_nodes),
        )
        return vertex_nodes, edge_nodes, interior_nodes

    def _compute_scaling_matrix(self):
        max_edge_entries = self.A[self.edge_nodes, :].max(axis=1).A.flatten()
        min_edge_entries = self.A[self.edge_nodes, :].min(axis=1).A.flatten()
        A_edge_contrast = min_edge_entries / max_edge_entries
        E_diag = np.ones(self.m**2)
        E_diag[self.edge_nodes] = A_edge_contrast
        E = diags(E_diag, format="csr")
        return E


class MsFEMSlabCoarseSpace(MsFEMCoarseSpace):
    def __init__(self, N, n, A, c, k):
        # The slab size in terms of the number of layers of elements.
        self.k = k

        # A connectivity matrix indicating which nodes are neighbors.
        self.M = (A != 0).astype(int)

        super().__init__(N, n, A, c)

    def _compute_inv_distances(self, fine_nodes, x_coarse, y_coarse, nc):
        # Get the global indices for the coarse node and its neighbors.
        _, _, nc_neighbors_idx = np.intersect1d(
            [nc - 1, nc + 1, nc + self.N, nc - self.N],
            self.coarse_nodes,
            return_indices=True,
        )
        nc_neighbors_global_idx = self.coarse_nodes_global_idx[nc_neighbors_idx]
        nc_global_idx = self.coarse_nodes_global_idx[self.coarse_nodes == nc]

        # Retrieve the nodes in the support of the basis function.
        supp_subdomains = self.P[nc_global_idx, :].nonzero()[1]
        supp_nodes = self.P[:, supp_subdomains].nonzero()[0]

        # Extend the interface nodes to a slab around the coarse edges.
        slab = self._extend_edge_to_slab(fine_nodes, supp_nodes)
        nc_slab_idx = np.where(slab == nc_global_idx)[0]
        _, _, nc_neighbors_slab_idx = np.intersect1d(
            slab, nc_neighbors_global_idx, return_indices=True
        )

        # Solve the Neumann problem on the slab.
        A_slab = self.A[slab, slab[:, None]]
        set_sparse_matrix_rows_to_value(A_slab, nc_neighbors_slab_idx, 0)
        set_sparse_matrix_rows_to_value(A_slab, nc_slab_idx, 0)
        A_slab[nc_neighbors_slab_idx, nc_neighbors_slab_idx] = 1
        A_slab[nc_slab_idx, nc_slab_idx] = 1

        b_slab = np.zeros(len(slab))
        b_slab[nc_slab_idx] = 1
        slab_inv_dist = spsolve(A_slab, b_slab)

        # Restrict the solution on the slab to the edges.
        edge_inv_dist = slab_inv_dist[np.isin(slab, fine_nodes, assume_unique=True)]

        return edge_inv_dist

    def _extend_edge_to_slab(self, edge_nodes, supp_nodes):
        slab = edge_nodes[:]
        for _ in range(self.k):
            slab_neighbors = self.M[slab, :].nonzero()[1]
            slab = np.union1d(slab, slab_neighbors)
        slab = np.intersect1d(slab, supp_nodes)
        return slab
