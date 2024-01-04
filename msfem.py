from scipy.sparse import lil_matrix
from scipy.integrate import quad


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
