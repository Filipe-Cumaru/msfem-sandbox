from scipy.sparse import lil_matrix
from scipy.integrate import quad
from skfem import MeshQuad, ElementQuad1, Basis, BilinearForm, LinearForm, asm, enforce
from skfem.helpers import dot, grad


class MsFEMBasisFunction(object):
    """An implementation of the Multiscale Finite Element Method (MsFEM)
    basis functions applied to the linear second order Laplace equation
    using oscillatory boundary conditions.
    """

    def __init__(self, N, n, c):
        if N > n:
            raise ValueError("n must be greater than N")

        # Number of coarse cells on each direction.
        self.N = N

        # Number of fine cells on each direction.
        self.n = n

        # Coarse and fine grid cell sizes.
        self.H = 1 / self.N
        self.h = 1 / self.n

        self.c = c

        coarse_grid_refine_ratio = np.log2(self.N)
        if coarse_grid_refine_ratio != int(coarse_grid_refine_ratio):
            raise ValueError("N must be a power of 2")
        self.coarse_grid = MeshQuad().refined(int(coarse_grid_refine_ratio))

        fine_grid_refine_ratio = np.log2(self.n)
        if fine_grid_refine_ratio != int(fine_grid_refine_ratio):
            raise ValueError("n must be a power of 2")
        self.fine_grid = MeshQuad().refined(int(fine_grid_refine_ratio))

        self.coarse_edge_trace = {}

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

    def _compute_fine_edges_trace(self):
        num_fine_edges_in_coarse_edge = int(self.H / self.h)
        lnodes, rnodes = self.coarse_grid.facets  # type: ignore

        for nc_1, nc_2 in zip(lnodes, rnodes):
            xc_1, yc_1 = self.coarse_grid.p[:, nc_1]
            xc_2, yc_2 = self.coarse_grid.p[:, nc_2]

            if xc_1 > xc_2 or yc_1 > yc_2:
                xc_1, xc_2 = xc_2, xc_1
                yc_1, yc_2 = yc_2, yc_1

            # The coefficient function restricted to a single direction.
            cx = lambda x: self.c(x, yc_1)
            cy = lambda y: self.c(xc_1, y)

            self.fine_edge_trace[(nc_1, nc_2)] = np.zeros(num_fine_edges_in_coarse_edge)

            for i in range(num_fine_edges_in_coarse_edge):
                # Horizontal edge
                if yc_1 == yc_2:
                    xf_1, xf_2 = xc_1 + i * self.h, xc_1 + (i + 1) * self.h
                    self.fine_edge_trace[(nc_1, nc_2)][i] = quad(cx, xf_1, xf_2)[0]
                # Vertical edge
                else:
                    yf_1, yf_2 = yc_1 + i * self.h, yc_1 + (i + 1) * self.h
                    self.fine_edge_trace[(nc_1, nc_2)][i] = quad(cy, yf_1, yf_2)[0]

    def _compute_coarse_edges_trace(self):
        lnodes, rnodes = self.coarse_grid.facets  # type: ignore

        for nc_1, nc_2 in zip(lnodes, rnodes):
            xc_1, yc_1 = self.coarse_grid.p[:, nc_1]
            xc_2, yc_2 = self.coarse_grid.p[:, nc_2]

            # The coefficient function restricted to a single direction.
            cx = lambda x: 1 / self.c(x, yc_1)
            cy = lambda y: 1 / self.c(xc_1, y)

            # Horizontal edge
            if yc_1 == yc_2:
                self.coarse_edge_trace[(nc_1, nc_2)] = quad(cx, xc_1, xc_2)[0]
            # Vertical edge
            else:
                self.coarse_edge_trace[(nc_1, nc_2)] = quad(cy, yc_1, yc_2)[0]
