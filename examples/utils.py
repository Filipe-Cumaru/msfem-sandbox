from typing import Callable, Any
from scipy.io import savemat
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
from ngsolve.meshes import MakeQuadMesh

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
        self.mesh = MakeQuadMesh(nx=n, ny=n)

    def _init_fes(self):
        raise NotImplementedError()

    def _build_bilinear_form(self, u, v):
        raise NotImplementedError()

    def _build_linear_form(self, v):
        raise NotImplementedError()

    def assemble(self):
        """Assembles the FEM system of equations."""
        # Function space and the trial and test functions.
        fes = self._init_fes()
        u, v = fes.TrialFunction(), fes.TestFunction()

        # Assemble the weak forms.
        a = self._build_bilinear_form(u, v)
        f = self._build_linear_form(v)

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

    def _init_fes(self):
        return ngs.H1(self.mesh, dirichlet=".*")

    def _build_bilinear_form(self, u, v):
        c = self.coeff(self.mesh)
        return ngs.BilinearForm(c * ngs.grad(u) * ngs.grad(v) * ngs.dx).Assemble()

    def _build_linear_form(self, v):
        return ngs.LinearForm(v * ngs.dx).Assemble()


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

    print("Assembling the FE problem.")

    # Initialization of an instance of a FEM problem.
    match problem_type:
        case msfem.NullSpaceType.DIFFUSION:
            fem_problem = DiffusionFEMProblem(m - 1, coeff_fem)
            num_dofs_per_node = 1
        case msfem.NullSpaceType.LINEAR_ELASTICITY:
            fem_problem = LinearElasticityFEMProblem(m - 1, coeff_fem)
            num_dofs_per_node = 2
        case _:
            raise ValueError(
                "The problem type must be either diffusion or linear elasticity."
            )

    # Assemble the FE system of equations.
    A, b = fem_problem.assemble()

    # A mapping of the grid nodes to their respective dofs.
    node_ids = np.arange(m**2)
    ngs_dofs_map = np.array([node_ids + i * m**2 for i in range(num_dofs_per_node)]).T
    A = A[ngs_dofs_map.flatten(), :]
    A = A[:, ngs_dofs_map.flatten()]  # type: ignore
    b = b[ngs_dofs_map.flatten()]

    print("===> Done ✔.")

    if precond == "single-level":
        print("Initializing the preconditioner.")
        precond_op = schwarz.SingleLevelASPreconditioner(A, N, n, k, problem_type)
        print("===> Done ✔.")
    elif precond == "two-level":
        # Initialization of the coarse space.
        print("Initializing the coarse space.")
        match coarse_space:
            case "msfem":
                cs = msfem.MsFEMCoarseSpace(N + 1, n + 1, A, coeff_eval, problem_type)
            case "q1":
                cs = msfem.Q1CoarseSpace(N + 1, n + 1, A, problem_type)
            case "rgdsw-opt-1":
                cs = msfem.RGDSWConstantCoarseSpace(N + 1, n + 1, A, problem_type)
            case "rgdsw-opt-2-2":
                cs = msfem.RGDSWInverseDistanceCoarseSpace(
                    N + 1, n + 1, A, problem_type
                )
            case "slab-msfem":
                cs = msfem.MsFEMSlabCoarseSpace(
                    N + 1, n + 1, A, coeff_eval, slab_size, problem_type
                )
            case "ams":
                cs = msfem.AMSCoarseSpace(N + 1, n + 1, A, problem_type)
            case "gdsw":
                if problem_type is not msfem.NullSpaceType.DIFFUSION:
                    raise ValueError(
                        "The GDSW coarse space is currently only available for the diffusion problem."
                    )
                cs = msfem.GDSWCoarseSpace(N + 1, n + 1, A, problem_type)
            case _:
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
    x, Tn = solvers.cg(
        A, b, M=M_as, tol=1e-8, maxiter=1000, callback=it_counter, return_lanczos=True
    )

    if output:
        if not os.path.exists("./output"):
            os.mkdir("./output")
        fname = (
            f"output_{m-1}x{m-1}_{N}x{N}_{precond}"
            + (f"_{coarse_space}" if precond == "two-level" else "")
            + ".mat"
        )
        out_dict = {"A": A, "b": b, "Tn": Tn, "x": x}

        if precond == "two-level":
            out_dict["Phi"] = Phi

        savemat(f"./output/{fname}", out_dict)

    print(f"Number of iterations: {it_counter.niter}")
    print("===> Done ✔.")
