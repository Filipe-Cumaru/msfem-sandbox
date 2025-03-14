from typing import Callable, Any
from scipy.io import savemat
from scipy.sparse import coo_matrix
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
    def __init__(self, nx: int, ny: int, coeff: Any, num_dofs: int) -> None:
        self.nx = nx
        self.ny = ny
        self.coeff = coeff
        self.num_dofs = num_dofs
        self.mesh = MakeQuadMesh(nx=nx, ny=ny)

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
        A = coo_matrix((vals, (rows, cols))).tocsc()
        b = f.vec.FV().NumPy()

        # BUG: This is a workaround due to some errors in SciPy
        # related to the size of the integer used for the indices.
        # It has been reported in some issues:
        # https://github.com/scipy/scipy/issues/13155
        # https://github.com/scipy/scipy/issues/16774
        A.indptr = A.indptr.astype(np.int64)
        A.indices = A.indices.astype(np.int64)

        # Set boundary conditions.
        boundary_dofs = np.nonzero(~fes.FreeDofs())[0]
        A[boundary_dofs, :] *= 0  # type: ignore
        A[:, boundary_dofs] *= 0  # type: ignore
        A[boundary_dofs, boundary_dofs] = 1
        A.eliminate_zeros()
        b[boundary_dofs] = 0

        return A, b


class DiffusionFEMProblem(FEMProblem):
    def __init__(self, nx: int, ny: int, coeff: Any) -> None:
        super().__init__(nx, ny, coeff, num_dofs=1)

    def _init_fes(self):
        return ngs.H1(self.mesh, dirichlet=".*")

    def _build_bilinear_form(self, u, v):
        c = self.coeff(self.mesh)
        return ngs.BilinearForm(c * ngs.grad(u) * ngs.grad(v) * ngs.dx).Assemble()

    def _build_linear_form(self, v):
        return ngs.LinearForm(v * ngs.dx).Assemble()


class LinearElasticityFEMProblem(FEMProblem):
    def __init__(self, nx: int, ny: int, coeff: Any) -> None:
        super().__init__(nx, ny, coeff, num_dofs=2)

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
        # Ex, Ey, nu = self.coeff()
        # Gxy = Ex * Ey / (Ex + Ey + 2 * Ey * nu)
        # C = ngs.CF(
        #     (
        #         (Ex / (1 - nu**2), nu * Ex / (1 - nu**2), 0),
        #         (nu * Ex / (1 - nu**2), Ey / (1 - nu**2), 0),
        #         (0, 0, Gxy),
        #     ),
        #     dims=(3, 3),
        # )
        # return self._voigt2stress(C * self._strain2voigt(strain))
        E, _, nu = self.coeff()
        Lambda, Mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
        return Lambda * ngs.Trace(strain) * ngs.Id(2) + 2 * Mu * strain

    def _strain2voigt(self, e):
        return ngs.CF((e[0, 0], e[1, 1], 2 * e[0, 1]))

    def _voigt2stress(self, s):
        return ngs.CF(((s[0], s[2]), (s[2], s[1])), dims=(2, 2))


def parse_args(example_description):
    parser = argparse.ArgumentParser(description=example_description)
    parser.add_argument(
        "-Nx",
        type=int,
        help="The number of subdomains on the x-axis direction.",
        required=True,
    )
    parser.add_argument(
        "-Ny",
        type=int,
        help="The number of subdomains on the y-axis direction.",
        required=True,
    )
    parser.add_argument(
        "-nx",
        type=int,
        help="The number of cells on the x-axis direction within each subdomain.",
        required=True,
    )
    parser.add_argument(
        "-ny",
        type=int,
        help="The number of cells on the y-axis direction within each subdomain.",
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
        choices=["none", "one-level", "two-level"],
        required=True,
    )
    parser.add_argument(
        "--coarse-space",
        type=str,
        help="The coarse space used to compute the basis functions. Required if using a two-level preconditioner.",
        choices=[
            "msfem",
            "rgdsw-opt-1",
            "rgdsw-opt-2-2",
            "ams",
            "slab-msfem",
            "gdsw",
            "spectral-ams"
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
        "--enrichment-tol",
        type=float,
        help="The tolerance value used to select the eigenvalues related to the eigenmodes used in the spectral enrichment of AMS. Required if using the coarse space spectral-ams",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--output",
        action="store_true",
        help="Enables the output. The solution will be saved in the `output` directory as a .mat file.",
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
    if (
        args.precond == "two-level"
        and args.coarse_space == "spectral-ams"
        and args.enrichment_tol is None
    ):
        parser.error(
            "Using the coarse space spectral-ams requires --enrichment-tol to be specified."
        )

    return args


def run_example(
    Nx: int,
    Ny: int,
    nx: int,
    ny: int,
    k: int,
    precond: str,
    coarse_space: str,
    slab_size: int,
    coeff_fem: Callable,
    coeff_eval: Callable,
    problem_type: msfem.NullSpaceType,
    output: bool,
    enrichment_tol: float
):
    """Runs an example from the `examples` folder.

    Args:
        Nx (int): The number of subdomains on the x-axis direction.
        Ny (int): The number of subdomains on the y-axis direction.
        nx (int): The number of cells on the x-axis direction within each subdomain.
        ny (int): The number of cells on the y-axis direction within each subdomain.
        k (int): The number of overlapping layers for each subdomain, i.e., the overlap size.
        precond (str): The preconditioning method to be used.
        coarse_space (str): The coarse space used to compute the basis functions.
        slab_size (int): The number of layers of nodes used in the edge slab. Required if using the coarse space slab-msfem.
        coeff_fem (Callable): A callable object that computes the coefficient function using DOLFINx. Required if using a MsFEM coarse space.
        coeff_eval (Callable): A callable object that evaluates the coefficient function nodally. Required if using a MsFEM coarse space.
        problem_type (msfem.NullSpaceType): The problem type to be run (diffusion or linear elasticity).
    """
    # Number of nodes on each direction (mx x my grid).
    mx = Nx * nx + 1
    my = Ny * ny + 1

    print("Assembling the FE problem.")

    # Initialization of an instance of a FEM problem.
    match problem_type:
        case msfem.NullSpaceType.DIFFUSION:
            fem_problem = DiffusionFEMProblem(mx - 1, my - 1, coeff_fem)
            num_dofs_per_node = 1
        case msfem.NullSpaceType.LINEAR_ELASTICITY:
            fem_problem = LinearElasticityFEMProblem(mx - 1, my - 1, coeff_fem)
            num_dofs_per_node = 2
        case _:
            raise ValueError(
                "The problem type must be either diffusion or linear elasticity."
            )

    # Assemble the FE system of equations.
    A, b = fem_problem.assemble()

    # A mapping of the grid nodes to their respective dofs.
    node_ids = np.arange(mx * my)
    ngs_dofs_map = np.array(
        [node_ids + i * mx * my for i in range(num_dofs_per_node)]
    ).T
    A = A[ngs_dofs_map.flatten(), :]
    A = A[:, ngs_dofs_map.flatten()]  # type: ignore
    b = b[ngs_dofs_map.flatten()]

    print("===> Done ✔.")

    if precond == "one-level":
        print("Initializing the preconditioner.")
        precond_op = schwarz.OneLevelOASPreconditioner(
            A, Nx, Ny, nx, ny, k, problem_type
        )
        print("===> Done ✔.")
    elif precond == "two-level":
        # Initialization of the coarse space.
        print("Initializing the coarse space.")
        match coarse_space:
            case "msfem":
                cs = msfem.MsFEMCoarseSpace(
                    Nx + 1, Ny + 1, nx + 1, ny + 1, A, coeff_eval, problem_type
                )
            case "rgdsw-opt-1":
                cs = msfem.RGDSWConstantCoarseSpace(
                    Nx + 1, Ny + 1, nx + 1, ny + 1, A, problem_type
                )
            case "rgdsw-opt-2-2":
                cs = msfem.RGDSWInverseDistanceCoarseSpace(
                    Nx + 1, Ny + 1, nx + 1, ny + 1, A, problem_type
                )
            case "slab-msfem":
                cs = msfem.MsFEMSlabCoarseSpace(
                    Nx + 1,
                    Ny + 1,
                    nx + 1,
                    ny + 1,
                    A,
                    coeff_eval,
                    slab_size,
                    problem_type,
                )
            case "ams":
                cs = msfem.AMSCoarseSpace(
                    Nx + 1, Ny + 1, nx + 1, ny + 1, A, problem_type
                )
            case "gdsw":
                if problem_type is not msfem.NullSpaceType.DIFFUSION:
                    raise ValueError(
                        "The GDSW coarse space is currently only available for the diffusion problem."
                    )
                cs = msfem.GDSWCoarseSpace(
                    Nx + 1, Ny + 1, nx + 1, ny + 1, A, problem_type
                )
            case "spectral-ams":
                cs = msfem.SpectralAMSCoarseSpace(
                    Nx + 1, Ny + 1, nx + 1, ny + 1, A, problem_type, tol=enrichment_tol
                )
            case _:
                raise ValueError("Invalid coarse space.")

        # Computes the coarse interpolation operator equiv. to
        # the multiscale prolongation operator.
        Phi = cs.assemble_operator()
        print("===> Done ✔.")

        print("Initializing the preconditioner.")
        precond_op = schwarz.TwoLevelOASPreconditioner(
            A, Phi, Nx, Ny, nx, ny, k, problem_type
        )
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
        A, b, M=M_as, tol=1e-8, maxiter=int(1e5), callback=it_counter, return_lanczos=True
    )

    if output:
        if not os.path.exists("./output"):
            os.mkdir("./output")
        fname = (
            f"output_{mx - 1}x{my - 1}_{Nx}x{Ny}_{precond}"
            + (f"_{coarse_space}" if precond == "two-level" else "")
            + ".mat"
        )
        out_dict = {"A": A, "b": b, "Tn": Tn, "x": x}

        if precond == "two-level":
            out_dict["Phi"] = Phi

        savemat(f"./output/{fname}", out_dict)

    print(f"Number of iterations: {it_counter.niter}")
    print("===> Done ✔.")
