from typing import Callable, Any
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
from netgen.meshing import Mesh, MeshPoint, Pnt, Element1D, Element2D

import numpy as np
import ngsolve as ngs
import argparse
import os

import msfem
import schwarz
import solvers


class FEMProblem(object):
    def __init__(self, n: int, coeff: Any, num_dofs: int) -> None:
        self.n = n
        self.coeff = coeff
        self.num_dofs = num_dofs
        self.mesh = self._init_mesh()

    def _init_mesh(self):
        ngmesh = Mesh(dim=2)
        point_ids = []
        for i in range(self.n + 1):
            for j in range(self.n + 1):
                point_ids.append(ngmesh.Add(MeshPoint(Pnt(i / self.n, j / self.n, 0))))

        idx_dom = ngmesh.AddRegion("mat", dim=2)
        for j in range(self.n):
            for i in range(self.n):
                ngmesh.Add(
                    Element2D(
                        idx_dom,
                        [
                            point_ids[i + j * (self.n + 1)],
                            point_ids[i + (j + 1) * (self.n + 1)],
                            point_ids[i + 1 + (j + 1) * (self.n + 1)],
                            point_ids[i + 1 + j * (self.n + 1)],
                        ],
                    )
                )

        for i in range(self.n):
            ngmesh.Add(
                Element1D(
                    [
                        point_ids[self.n + i * (self.n + 1)],
                        point_ids[self.n + (i + 1) * (self.n + 1)],
                    ],
                    index=1,
                )
            )
            ngmesh.Add(
                Element1D(
                    [point_ids[i * (self.n + 1)], point_ids[(i + 1) * (self.n + 1)]],
                    index=1,
                )
            )

        for i in range(self.n):
            ngmesh.Add(Element1D([point_ids[i], point_ids[i + 1]], index=2))
            ngmesh.Add(
                Element1D(
                    [
                        point_ids[i + self.n * (self.n + 1)],
                        point_ids[i + 1 + self.n * (self.n + 1)],
                    ],
                    index=2,
                )
            )

        return ngs.Mesh(ngmesh)

    def _build_bilinear_form(self, u, v):
        raise NotImplementedError()

    def assemble(self):
        """Assembles the FEM system of equations."""
        # Function space and the trial and test functions.
        fes = ngs.H1(self.mesh, dirichlet=".*")
        u, v = fes.TrialFunction(), fes.TestFunction()

        # Assemble the weak forms.
        a = self._build_bilinear_form(u, v)
        f = ngs.LinearForm(v * ngs.dx).Assemble()

        # Export the assembled system to NumPy/SciPy format.
        rows, cols, vals = a.mat.COO()
        A = csr_matrix((vals, (rows, cols)))
        b = f.vec.FV().NumPy()

        # Set boundary conditions.
        boundary_dofs = np.nonzero(~fes.FreeDofs())[0]
        msfem.set_sparse_matrix_rows_to_value(A, boundary_dofs, 0)
        A[boundary_dofs, boundary_dofs] = 1
        A.eliminate_zeros()
        b[boundary_dofs] = 0

        return A, b


class DiffusionFEMProblem(FEMProblem):
    def __init__(self, n: int, coeff: Any) -> None:
        super().__init__(n, coeff, num_dofs=1)

    def _build_bilinear_form(self, u, v):
        return ngs.BilinearForm(
            self.coeff * ngs.grad(u) * ngs.grad(v) * ngs.dx
        ).Assemble()


class LinearElasticityFEMProblem(FEMProblem):
    def __init__(self, n: int, coeff: Any) -> None:
        super().__init__(n, coeff, num_dofs=2)

    def _init_fes(self):
        return ngs.VectorH1(self.mesh, dirichlet=".*")

    def _build_bilinear_form(self, u, v):
        eps_u, eps_v = self._strain(u), self._strain(v)
        sigma = self._stress(eps_u)
        return ngs.BilinearForm(
            ngs.InnerProduct(sigma, eps_v).Compile() * ngs.dx
        ).Assemble()
    
    def _build_linear_form(self, v):
        force = ngs.CF((1, 1))
        return ngs.LinearForm(force * v * ngs.dx).Assemble()

    def _strain(self, u):
        return ngs.Sym(ngs.Grad(u))

    def _stress(self, strain):
        E, nu = self.coeff(self.mesh)
        mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
        return 2 * mu * strain + lam * ngs.Trace(strain) * ngs.Id(strain.shape[0])


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
        choices=["none", "single-level", "two-level"],
        required=True,
    )
    parser.add_argument(
        "--coarse-space",
        type=str,
        help="The coarse space used to compute the basis functions. Required if using a two-level preconditioner.",
        choices=[
            "msfem",
            "q1",
            "rgdsw-opt-1",
            "rgdsw-opt-2-2",
            "ams",
            "slab-msfem",
            "gdsw",
        ],
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
    parser.add_argument(
        "--output",
        action="store_true",
        help="Enables the output. The solution will be saved in the `output` directory as a .npy file.",
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
    output: bool,
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
    print("Assembling the FE problem.")
    if problem_type is msfem.NullSpaceType.DIFFUSION:
        fem_problem = DiffusionFEMProblem(m - 1, coeff_fem)
        dofs_map = np.arange(m**2, dtype=int).reshape((m**2, 1))
    elif problem_type is msfem.NullSpaceType.LINEAR_ELASTICITY:
        fem_problem = LinearElasticityFEMProblem(m - 1, coeff_fem)
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
    print("===> Done ✔.")

    # Reordering of the system so it is consistent with the
    # definition adopted in the coarse space.
    idx = np.lexsort((grid[0], grid[1]))
    for i in range(dofs_map.shape[1]):
        dofs_map[:, i] = dofs_map[idx, i]
    dofs_idx = dofs_map.flatten()
    A = A[:, dofs_idx]
    A = A[dofs_idx, :]
    b = b[dofs_idx]

    if precond == "single-level":
        print("Initializing the preconditioner.")
        precond_op = schwarz.SingleLevelASPreconditioner(A, N, n, k, problem_type)
        print("===> Done ✔.")
    elif precond == "two-level":
        # Initialization of the coarse space.
        print("Initializing the coarse space.")
        if coarse_space == "msfem":
            cs = msfem.MsFEMCoarseSpace(N + 1, n + 1, A, coeff_eval, problem_type)
        elif coarse_space == "q1":
            cs = msfem.Q1CoarseSpace(N + 1, n + 1, A, problem_type)
        elif coarse_space == "rgdsw-opt-1":
            cs = msfem.RGDSWConstantCoarseSpace(N + 1, n + 1, A, problem_type)
        elif coarse_space == "rgdsw-opt-2-2":
            cs = msfem.RGDSWInverseDistanceCoarseSpace(N + 1, n + 1, A, problem_type)
        elif coarse_space == "slab-msfem":
            cs = msfem.MsFEMSlabCoarseSpace(
                N + 1, n + 1, A, coeff_eval, slab_size, problem_type
            )
        elif coarse_space == "ams":
            cs = msfem.AMSCoarseSpace(N + 1, n + 1, A, problem_type)
        elif coarse_space == "gdsw":
            if problem_type is not msfem.NullSpaceType.DIFFUSION:
                raise ValueError(
                    "The GDSW coarse space is currently only available for the diffusion problem."
                )
            cs = msfem.GDSWCoarseSpace(N + 1, n + 1, A, problem_type)
        else:
            raise ValueError("Invalid coarse space.")

        # Computes the coarse interpolation operator equiv. to
        # the multiscale prolongation operator.
        Phi = cs.assemble_operator()
        print("===> Done ✔.")

        print("Initializing the preconditioner.")
        precond_op = schwarz.TwoLevelASPreconditioner(A, Phi, N, n, k, problem_type)
        print("===> Done ✔.")

    # Solution of the system of equations using the Schwarz preconditioner.
    print("Solving the system of equations.")
    M_as = (
        LinearOperator(A.shape, lambda x: precond_op.apply(x))
        if precond != "none"
        else None
    )
    it_counter = solvers.IterationsCounter(disp=False)
    x = solvers.cg(A, b, M=M_as, callback=it_counter)

    if output:
        if not os.path.exists("./output"):
            os.mkdir("./output")
        fname = (
            f"solution_{m-1}x{m-1}_{N}x{N}_{precond}"
            + (f"_{coarse_space}" if precond == "two-level" else "")
            + ".npy"
        )
        np.save(f"./output/{fname}", x)

    print(f"Number of iterations: {it_counter.niter}")
    print("===> Done ✔.")
