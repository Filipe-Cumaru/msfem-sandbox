from scipy.sparse import lil_matrix
from scipy.integrate import quad
from skfem import (
    MeshQuad,
    ElementQuad1,
    Basis,
    BilinearForm,
    LinearForm,
    asm,
    enforce,
    solve,
)
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

        self.coarse_edges_trace = {}
        self.fine_edges_trace = {}

    def assemble_operator(self):
        Phi = lil_matrix(((self.N + 1) ** 2, (self.n + 1) ** 2))
        fem_basis = Basis(self.fine_grid, ElementQuad1())

        n_coarse_edges = self.coarse_grid.facets.shape[1]
        in_coarse_edges_idx = np.setdiff1d(
            np.arange(n_coarse_edges), self.coarse_grid.boundary_facets()
        )
        in_coarse_edges = self.coarse_grid.facets[:, in_coarse_edges_idx]
        num_fine_edges_in_coarse_edge = int(self.H / self.h)

        self._compute_coarse_edges_trace()
        self._compute_fine_edges_trace()

        A_base = self._assemble_local_problem_lhs(fem_basis)
        b_base = self._assemble_local_problem_rhs(fem_basis)
        A_base, b_base = enforce(A_base, b_base, D=self.fine_grid.boundary_nodes())

        for P in range((self.N + 1) ** 2):
            A_P, b_P = A_base.copy(), b_base[:]
            coarse_edges_around_P = in_coarse_edges[
                :, (in_coarse_edges[0, :] == P) | (in_coarse_edges[1, :] == P)
            ]
            P_x, P_y = self.coarse_grid.p[:, P]
            P_fine_idx = fem_basis.dofs.nodal_dofs[
                0, (self.fine_grid.p[0, :] == P_x) & (self.fine_grid.p[1, :] == P_y)
            ]
            fine_nodes_on_gamma = []
            fine_nodes_trace = np.zeros(b_P.shape[0])

            for nc_1, nc_2 in coarse_edges_around_P.T:
                xc_1, yc_1 = self.coarse_grid.p[0, nc_1], self.coarse_grid.p[1, nc_1]
                xc_2, yc_2 = self.coarse_grid.p[0, nc_2], self.coarse_grid.p[1, nc_2]
                coarse_edge_trace = self.coarse_edges_trace[(nc_1, nc_2)]

                if xc_1 > xc_2 or yc_1 > yc_2:
                    xc_1, xc_2 = xc_2, xc_1
                    yc_1, yc_2 = yc_2, yc_1

                for i in range(1, num_fine_edges_in_coarse_edge):
                    # Horizontal edge
                    if yc_1 == yc_2:
                        xf = xc_1 + i * self.h
                        nf = fem_basis.nodal_dofs[
                            0,
                            (self.fine_grid.p[0] == xf) & (self.fine_grid.p[1] == yc_1),
                        ][0]
                    # Vertical edge
                    else:
                        yf = yc_1 + i * self.h
                        nf = fem_basis.nodal_dofs[
                            0,
                            (self.fine_grid.p[0] == xc_1) & (self.fine_grid.p[1] == yf),
                        ][0]

                    fine_nodes_on_gamma.append(nf)
                    fine_nodes_trace[nf] = (
                        self.fine_edges_trace[(nc_1, nc_2)][:i].sum()
                        / coarse_edge_trace
                    )

            if len(fine_nodes_on_gamma) > 0:
                fine_nodes_on_gamma = np.array(fine_nodes_on_gamma)
                A_P, b_P = enforce(A_P, b_P, D=fine_nodes_on_gamma, x=fine_nodes_trace)
            if P_fine_idx not in self.fine_grid.boundary_nodes():
                kronecker_delta = np.zeros(self.fine_grid.p.shape[1])
                kronecker_delta[P_fine_idx] = 1
                A_P, b_P = enforce(A_P, b_P, D=P_fine_idx, x=kronecker_delta)

            Phi[P, :] = solve(A_P, b_P)

        Phi = Phi.tocsc()
        return Phi

    def _assemble_local_problem_lhs(self, basis):
        @BilinearForm
        def bilinear_form(u, v, w):
            return dot(self.c(w.x[0], w.x[1]) * grad(u), grad(v))

        return asm(bilinear_form, basis)

    def _assemble_local_problem_rhs(self, basis):
        @LinearForm
        def linear_form(v, _):
            return 0

        return asm(linear_form, basis)

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
            cx = lambda x: 1 / self.c(x, yc_1)
            cy = lambda y: 1 / self.c(xc_1, y)

            self.fine_edges_trace[(nc_1, nc_2)] = np.zeros(
                num_fine_edges_in_coarse_edge
            )

            for i in range(num_fine_edges_in_coarse_edge):
                # Horizontal edge
                if yc_1 == yc_2:
                    xf_1, xf_2 = xc_1 + i * self.h, xc_1 + (i + 1) * self.h
                    self.fine_edges_trace[(nc_1, nc_2)][i] = quad(cx, xf_1, xf_2)[0]
                # Vertical edge
                else:
                    yf_1, yf_2 = yc_1 + i * self.h, yc_1 + (i + 1) * self.h
                    self.fine_edges_trace[(nc_1, nc_2)][i] = quad(cy, yf_1, yf_2)[0]

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
                self.coarse_edges_trace[(nc_1, nc_2)] = quad(cx, xc_1, xc_2)[0]
            # Vertical edge
            else:
                self.coarse_edges_trace[(nc_1, nc_2)] = quad(cy, yc_1, yc_2)[0]
