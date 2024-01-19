from scipy.sparse import lil_matrix, csc_matrix
from scipy.integrate import quad
from scipy.sparse.linalg import spsolve
import numpy as np


class MSBasisFunction(object):
    def __init__(self, N, n, c=None):
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


class Q1BasisFunction(MSBasisFunction):
    def __init__(self, N, n):
        super().__init__(N, n)

    def assemble_operator(self):
        xs, ys = np.meshgrid(np.linspace(0, 1, self.m), np.linspace(0, 1, self.m))
        xs, ys = xs.flatten(), ys.flatten()

        Phi_row_idx = []
        Phi_col_idx = []
        Phi_values = []

        for P in range(self.N**2):
            Ni, Nj = P % self.N, P // self.N
            xP, yP = Ni * self.H, Nj * self.H

            # Filter the fine nodes in the neighborhood of P.
            loc_mask = (np.abs(xs - xP) < self.H) & (np.abs(ys - yP) < self.H)
            xs_loc = xs[loc_mask]
            ys_loc = ys[loc_mask]

            # Get the indices of the internal nodes.
            xs_idx = (xs_loc / self.h).astype(int)
            ys_idx = (ys_loc / self.h).astype(int)
            global_idx = xs_idx + ys_idx * self.m
            in_mask = ~np.isin(global_idx, self.boundary_fine_nodes, assume_unique=True)
            xs_in, ys_in = xs_loc[in_mask], ys_loc[in_mask]

            # Compute the values of the basis function for the internal nodes.
            Phi_P = np.abs(
                (self.H - np.abs(xs_in - xP)) * (self.H - np.abs(ys_in - yP))
            ) / (self.H**2)

            Phi_row_idx.extend(len(Phi_P) * [P])
            Phi_col_idx.extend(global_idx[in_mask])
            Phi_values.extend(Phi_P)

        Phi = csc_matrix(
            (Phi_values, (Phi_row_idx, Phi_col_idx)), shape=(self.N**2, self.m**2)
        )

        return Phi


class RGDSWCoarseSpace(MSBasisFunction):
    def __init__(self, N, n, A):
        super().__init__(N, n)
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

    def assemble_operator(self):
        Phi_interface = self._compute_interface_basis_function()
        Phi = csc_matrix(Phi_interface)

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
            Omega_i_coarse_nodes = np.intersect1d(Omega_i_coarse_vertices, self.coarse_nodes)

            # The portion of the system matrix corresponding to the interior nodes
            # of \Omega_i.
            Ai_II = self.A[Omega_i_interior, Omega_i_interior[:, None]]

            # The portion of the system matrix corresponding to the interaction 
            # between interior and boundary nodes of \Omega_i.
            Ai_IB = self.A[Omega_i_boundary, Omega_i_interior[:, None]]

            # The contribution of the interface basis function for the subdomain
            # \Omega_i (c.f. Eq. 3 from Dohrmann and Windlund (2017)).
            Psi_i = Phi_interface[Omega_i_coarse_nodes, Omega_i[:, None]]

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
        Phi = Phi[self.coarse_nodes, :]

        return Phi

    def _compute_interface_basis_function(self):
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
        P = csc_matrix(
            (R_values, (row_idx, col_idx)), shape=(self.m**2, N_cells**2)
        )

        # The partition of the interior nodes for each subdomain.
        P_I = P.multiply(P.sum(axis=1) == 1).tocsc()

        # The partition of the nodes on the boundary for each subdomain.
        P_B = P - P_I

        return P, P_I, P_B


class RGDSWConstantCoarseSpace(RGDSWCoarseSpace):
    """Implementation of option 1 for the RGDSW coarse space described in Dohrmann and Windlund (2017).
    """
    def __init__(self, N, n, A):
        super().__init__(N, n, A)

    def _compute_interface_basis_function(self):
        xs, ys = np.meshgrid(np.linspace(0, 1, self.m), np.linspace(0, 1, self.m))
        xs, ys = xs.flatten(), ys.flatten()

        Phi_row_idx = []
        Phi_col_idx = []
        Phi_values = []

        for nc in self.coarse_nodes:
            Ni, Nj = nc % self.N, nc // self.N
            x_nc, y_nc = Ni * self.H, Nj * self.H

            # Retrieve the subdomains that contain the coarse node nc.
            Ni_fine, Nj_fine = Ni * (self.n - 1), Nj * (self.n - 1)
            nc_fine_idx = Ni_fine + Nj_fine * self.m

            # Filter the fine nodes in the neighborhood of nc.
            supp_mask = (np.abs(xs - x_nc) <= self.H) & (np.abs(ys - y_nc) <= self.H)
            xs_supp = xs[supp_mask]
            ys_supp = ys[supp_mask]
            xs_interface = xs_supp[(xs_supp == x_nc) | (ys_supp == y_nc)]
            ys_interface = ys_supp[(xs_supp == x_nc) | (ys_supp == y_nc)]

            # Get the indices of the internal nodes.
            xs_idx = (xs_interface / self.h).astype(int)
            ys_idx = (ys_interface / self.h).astype(int)
            global_idx = xs_idx + ys_idx * self.m
            in_nodes = np.setdiff1d(global_idx, self.boundary_fine_nodes)  # type: ignore

            # Compute the values of the basis function for the internal nodes.
            Phi_nc = 0.5 * np.ones(len(in_nodes))
            Phi_nc[in_nodes == nc_fine_idx] = 1

            Phi_row_idx.extend(len(Phi_nc) * [nc])
            Phi_col_idx.extend(in_nodes)
            Phi_values.extend(Phi_nc)

        Phi = csc_matrix(
            (Phi_values, (Phi_row_idx, Phi_col_idx)), shape=(self.N**2, self.m**2)
        )

        return Phi


class MsFEMBasisFunction(object):
    """An implementation of the Multiscale Finite Element Method (MsFEM)
    basis functions applied to the linear second order Laplace equation
    using oscillatory boundary conditions.
    """

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
        self.coarse_edges_integrals = {}

    def assemble_operator(self):
        Phi = lil_matrix((self.N**2, self.m**2))
        self._compute_coarse_edges_integrals()
        for P in range(self.N**2):
            Ni, Nj = P % self.N, P // self.N
            xP, yP = Ni * self.H, Nj * self.H
            for p in range(self.m**2):
                ni, nj = p % self.m, p // self.m
                xp, yp = ni * self.h, nj * self.h
                if p not in self.boundary_fine_nodes:
                    if xp == xP and yp == yP:
                        Phi[P, p] = 1
                    elif abs(xp - xP) < self.H and abs(yp - yP) < self.H:
                        Phi[P, p] = self._compute_basis_function(ni, nj, Ni, Nj)
        Phi = Phi.tocsc()
        return Phi

    def _compute_basis_function(self, ni, nj, Ni, Nj):
        """Computes the basis function value via the boundary values.

        Args:
            Ni (int): The coarse node index on the x-axis.
            Nj (int): The coarse node index on the y-axis.
            ni (int): The fine node index on the x-axis.
            nj (int): The fine node index on the x-axis.

        Returns:
            float: The basis function value at the fine node (ni, nj).
        """
        # The coarse node index.
        P = Ni + self.N * Nj

        # Coordinates of the coarse node.
        X, Y = Ni * self.H, Nj * self.H

        # Coordinates of the fine-scale node.
        x, y = ni * self.h, nj * self.h

        # Find the coarse edge on which the projection of the fine node
        # lies on each direction.
        Ec_horizontal = (P - 1, P) if x < X else (P, P + 1)
        Ec_vertical = (P - self.N, P) if y < Y else (P, P + self.N)

        # The limits of the denominator integral.
        xL = X - self.H if x < X else X + self.H
        yL = Y - self.H if y < Y else Y + self.H

        # The limits of the numerator integral.
        x_start, x_end = xL, x
        y_start, y_end = yL, y

        # The coefficient evaluated on each edge.
        cx = lambda x: 1 / self.c(x, Y)
        cy = lambda y: 1 / self.c(X, y)

        sign_x = 1 if x < X else -1
        sign_y = 1 if y < Y else -1

        # The value of the boundary basis function on each direction.
        mu_x = (
            sign_x
            * quad(cx, x_start, x_end)[0]
            / self.coarse_edges_integrals[Ec_horizontal]
        )
        mu_y = (
            sign_y
            * quad(cy, y_start, y_end)[0]
            / self.coarse_edges_integrals[Ec_vertical]
        )

        return mu_x * mu_y

    def _compute_coarse_edges_integrals(self):
        for Ec in self.coarse_edges:
            nc_1, nc_2 = Ec
            xc_1, yc_1 = (nc_1 % self.N) * self.H, (nc_1 // self.N) * self.H
            xc_2, yc_2 = (nc_2 % self.N) * self.H, (nc_2 // self.N) * self.H

            # The coefficient function restricted to a single direction.
            cx = lambda x: 1 / self.c(x, yc_1)
            cy = lambda y: 1 / self.c(xc_1, y)

            # Horizontal edge
            if nc_2 - nc_1 == 1:
                self.coarse_edges_integrals[Ec] = quad(cx, xc_1, xc_2)[0]
            # Vertical edge
            else:
                self.coarse_edges_integrals[Ec] = quad(cy, yc_1, yc_2)[0]
