from typing import Callable, Any
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
from dolfinx.fem import petsc
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator

import numpy as np
import ufl
import argparse

import msfem
import schwarz
import solvers

np.random.seed(42)


class IterationsCounter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print("iter %3i\trk = %s" % (self.niter, str(rk)))


class FEMProblem(object):
    def __init__(self, n: int, coeff: Callable[..., Any], num_dofs: int) -> None:
        self.n = n
        self.coeff = coeff
        self.num_dofs = num_dofs
        self.msh = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [np.zeros(2), np.ones(2)],
            [self.n, self.n],
            mesh.CellType.quadrilateral,
            ghost_mode=mesh.GhostMode.shared_facet,
        )

    def _build_bilinear_form(self, u, v):
        raise NotImplementedError()

    def assemble(self):
        """Assembles the FEM system of equations."""
        # Definition of the function space and FEM forms.
        V = fem.functionspace(self.msh, ("Lagrange", 1, (self.num_dofs,)))
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        a = self._build_bilinear_form(u, v)
        f = fem.Function(V)
        f.x.array[:] = np.random.rand(len(f.x.array))
        L = fem.form(ufl.inner(f, v) * ufl.dx)

        # Definition of the boundary conditions.
        facets = mesh.locate_entities_boundary(
            self.msh,
            dim=1,
            marker=lambda x: np.logical_or(
                np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0)),
                np.logical_or(np.isclose(x[0], 1.0), np.isclose(x[1], 1.0)),
            ),
        )
        bc = fem.dirichletbc(
            np.zeros(self.num_dofs, dtype=PETSc.ScalarType),
            fem.locate_dofs_topological(V, entity_dim=1, entities=facets),
            V=V,
        )

        # Assemble the LHS and the RHS.
        A = petsc.assemble_matrix(a, bcs=[bc])
        A.assemble()
        b = petsc.assemble_vector(L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        # Exports the LHS and RHS to a SciPy friendly format.
        A_out = csc_matrix(A.getValuesCSR()[::-1], shape=A.size)
        b_out = b.array[:]
        grid = self.msh.geometry.x[:, 0:2].T

        return A_out, b_out, grid


class DiffusionFEMProblem(FEMProblem):
    def __init__(self, n: int, coeff: Callable[..., Any]) -> None:
        super().__init__(n, coeff, num_dofs=1)

    def _build_bilinear_form(self, u, v):
        C = self.coeff(self.msh)
        return fem.form(ufl.inner(C * ufl.grad(u), ufl.grad(v)) * ufl.dx)


class LinearElasticityFEMProblem(FEMProblem):
    def __init__(self, n: int, coeff: Callable[..., Any]) -> None:
        super().__init__(n, coeff, num_dofs=2)

    def _build_bilinear_form(self, u, v):
        return fem.form(ufl.inner(self._sigma(u), self._sym_grad(v)) * ufl.dx)

    def _sym_grad(self, v):
        return ufl.sym(ufl.grad(v))

    def _strain2voigt(self, e):
        return ufl.as_vector([e[0, 0], e[1, 1], 2 * e[0, 1]])

    def _voigt2stress(self, s):
        return ufl.as_tensor([[s[0], s[2]], [s[2], s[1]]])

    def _sigma(self, u):
        # Young's moduli and Poisson ratio
        E, Nu = self.coeff(self.msh)
        E_x, E_y = E[0], E[1]

        # Shear modulus
        G_xy = E_x * E_y / (E_x + E_y + 2 * E_y * Nu)

        # Stiffness tensor
        C = ufl.as_matrix(
            [
                [E_x / (1 - Nu**2), Nu * E_x / (1 - Nu**2), 0],
                [Nu * E_x / (1 - Nu**2), E_y / (1 - Nu**2), 0],
                [0, 0, G_xy],
            ]
        )

        return self._voigt2stress(ufl.dot(C, self._strain2voigt(self._sym_grad(u))))


def sort_ext_indices(p, xs, ys):
    p_sorted = np.vstack((xs, ys)).T
    sorted_idx = np.zeros(len(p[0]), dtype=int)
    for i, (nx, ny) in enumerate(p_sorted):
        sorted_idx[i] = np.where(np.isclose(p[0], nx) & np.isclose(p[1], ny))[0][0]
    return sorted_idx


def parse_args(example_description):
    parser = argparse.ArgumentParser(description=example_description)
    parser.add_argument(
        "-N",
        type=int,
        help="The number of subdomains on each direction.",
        required=True,
    )
    parser.add_argument(
        "-n",
        type=int,
        help="The number of cells on each direction within each subdomain.",
        required=True,
    )
    parser.add_argument(
        "-k",
        type=int,
        help="The number of overlapping layers for each subdomain, i.e., the overlap size.",
        required=True,
    )
    parser.add_argument(
        "--precond",
        type=str,
        help="The preconditioning method to be used.",
        choices=["single-level", "two-level"],
        required=True,
    )
    parser.add_argument(
        "--coarse-space",
        type=str,
        help="The coarse space used to compute the basis functions. Required if using a two-level preconditioner.",
        choices=["msfem", "q1", "rgdsw-opt-1", "rgdsw-opt-2-2", "ams", "slab-msfem"],
        default=None,
        required=False,
    )
    parser.add_argument(
        "--slab-size",
        type=int,
        help="The number of layers of nodes used in the edge slab. Required if using the coarse space slab-msfem.",
        default=None,
        required=False,
    )

    args = parser.parse_args()
    if args.precond == "two-level" and args.coarse_space is None:
        parser.error(
            "The coarse space must be specified via --coarse-space if the two-level preconditioner is used."
        )
    if (
        args.precond == "two-level"
        and args.coarse_space == "slab-msfem"
        and args.slab_size is None
    ):
        parser.error(
            "Using the coarse space slab-msfem requires --slab-size to be specified."
        )

    return args


def run_example(
    N: int,
    n: int,
    k: int,
    precond: str,
    coarse_space: str,
    slab_size: int,
    coeff_fem: Callable,
    coeff_eval: Callable,
    problem_type: msfem.NullSpaceType,
):
    """Runs an example from the `examples` folder.

    Args:
        N (int): The number of subdomains on each direction.
        n (int): The number of cells on each direction within each subdomain.
        k (int): The number of overlapping layers for each subdomain, i.e., the overlap size.
        precond (str): The preconditioning method to be used.
        coarse_space (str): The coarse space used to compute the basis functions.
        slab_size (int): The number of layers of nodes used in the edge slab. Required if using the coarse space slab-msfem.
        coeff_fem (Callable): A callable object that computes the coefficient function using DOLFINx. Required if using a MsFEM coarse space.
        coeff_eval (Callable): A callable object that evaluates the coefficient function nodally. Required if using a MsFEM coarse space.
        problem_type (msfem.NullSpaceType): The problem type to be run (diffusion or linear elasticity).
    """
    # Number of nodes on each direction (m x m grid).
    m = N * n + 1

    # Initialization of the local dof map according to the type of problem.
    if problem_type is msfem.NullSpaceType.DIFFUSION:
        fem_problem = DiffusionFEMProblem(m-1, coeff_fem)
        dofs_map = np.arange(m**2, dtype=int).reshape((m**2, 1))
    elif problem_type is msfem.NullSpaceType.LINEAR_ELASTICITY:
        fem_problem = LinearElasticityFEMProblem(m-1, coeff_fem)
        ns = np.arange(m**2, dtype=int)
        dofs_map = np.zeros((m**2, 2), dtype=int)
        dofs_map[:, 0] = 2 * ns
        dofs_map[:, 1] = 2 * ns + 1
    else:
        raise ValueError(
            "The problem type must be either diffusion or linear elasticity."
        )

    # Assembly of the FE system of equations.
    A, b, grid = fem_problem.assemble()

    # Reordering of the system so it is consistent with the
    # definition adopted in the coarse space.
    xs, ys = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))
    idx = sort_ext_indices(grid, xs.flatten(), ys.flatten())
    for i in range(dofs_map.shape[1]):
        dofs_map[:, i] = dofs_map[idx, i]
    dofs_idx = dofs_map.flatten()
    A = A[:, dofs_idx]
    A = A[dofs_idx, :]
    b = b[dofs_idx]

    # Initialization of the coarse space.
    if coarse_space == "msfem":
        cs = msfem.MsFEMCoarseSpace(N + 1, n + 1, A, coeff_eval, problem_type)
    elif coarse_space == "q1":
        cs = msfem.Q1CoarseSpace(N + 1, n + 1, A, problem_type)
    elif coarse_space == "rgdsw-opt-1":
        cs = msfem.RGDSWConstantCoarseSpace(N + 1, n + 1, A, problem_type)
    elif coarse_space == "rgdsw-opt-2-2":
        cs = msfem.RGDSWInverseDistanceCoarseSpace(N + 1, n + 1, A, problem_type)
    elif coarse_space == "slab-msfem":
        cs = msfem.MsFEMSlabCoarseSpace(N + 1, n + 1, A, coeff_eval, slab_size, problem_type)
    else:
        raise ValueError("Invalid coarse space.")

    # Computes the coarse interpolation operator equiv. to
    # the multiscale prolongation operator.
    Phi = cs.assemble_operator()

    # Solution of the system of equations using the Schwarz preconditioner.
    precond_op = schwarz.TwoLevelASPreconditioner(
        A, Phi, N, n, k, problem_type, dofs_map
    )
    M_as = LinearOperator(A.shape, lambda x: precond_op.apply(x))
    x = solvers.cg(A, b, M=M_as, callback=IterationsCounter())
