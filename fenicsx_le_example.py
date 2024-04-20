from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.mesh import (
    create_rectangle,
    CellType,
    GhostMode,
    locate_entities_boundary,
    locate_entities,
)
from dolfinx.fem import (
    Function,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc
from dolfinx.io import XDMFFile
from ufl import (
    dx,
    sym,
    grad,
    as_tensor,
    as_matrix,
    as_vector,
    dot,
    TestFunction,
    TrialFunction,
    inner,
    tr,
    Identity
)
from scipy.sparse import csr_matrix, save_npz
import numpy as np

dtype = PETSc.ScalarType  # type: ignore


def eps(v):
    return sym(grad(v))


def strain2voigt(e):
    """e is a 2nd-order tensor, returns its Voigt vectorial representation"""
    return as_vector([e[0, 0], e[1, 1], 2 * e[0, 1]])


def voigt2stress(s):
    """
    s is a stress-like vector (no 2 factor on last component)
    returns its tensorial representation
    """
    return as_tensor([[s[0], s[2]], [s[2], s[1]]])


def sigma(mesh, u, V, coeff):
    # Young's moduli and Poisson ratio
    E, Nu = coeff(mesh, V)
    E_x, E_y = E[0], E[1]

    # Shear modulus
    G_xy = E_x * E_y / (E_x + E_y + 2 * E_y * Nu)

    # Stiffness tensor
    C = as_matrix(
        [
            [E_x / (1 - Nu**2), Nu * E_x / (1 - Nu**2), 0],
            [Nu * E_x / (1 - Nu**2), E_y / (1 - Nu**2), 0],
            [0, 0, G_xy],
        ]
    )

    return voigt2stress(dot(C, strain2voigt(eps(u))))


def sigma_iso(mesh, u, V, coeff):
    E, Nu = coeff(mesh, V)
    Lambda, Mu = E[0] * Nu / (1 - Nu**2), E[0] / (2 * (1 + Nu))
    return 2.0 * Mu * sym(grad(u)) + Lambda * tr(sym(grad(u))) * Identity(len(u))


def coeff_ex_1(mesh, V):
    """Isotropic, constant coefficients."""
    return [100, 100], 0.3


def coeff_ex_2(mesh, V):
    """Isotropic, high coefficient inclusions at the coarse nodes."""
    n = 64

    def find_insertions(x):
        N = 8
        h = 1 / n
        margin = 2 * h
        xcs, ycs = np.meshgrid(np.linspace(0, 1, N + 1), np.linspace(0, 1, N + 1))
        xcs, ycs = xcs.flatten(), ycs.flatten()
        mask = (xcs > 0) & (xcs < 1) & (ycs > 0) & (ycs < 1)
        xcs, ycs = xcs[mask], ycs[mask]
        xs, ys = x[0], x[1]

        in_mask = np.zeros(len(xs), dtype=bool)
        for xc, yc in zip(xcs, ycs):
            coarse_node_check = (
                np.isclose(np.abs(xc - xs), margin) | (np.abs(xc - xs) < margin)
            ) & (np.isclose(np.abs(yc - ys), margin) | (np.abs(yc - ys) < margin))
            in_mask = in_mask | coarse_node_check

        return in_mask

    Omega_in = locate_entities(mesh, 0, find_insertions)

    E = Function(V, name="E")
    E.x.array[:] = 1
    E.x.array[2 * Omega_in] = E.x.array[2 * Omega_in + 1] = 1e4
    Nu = 0.3

    return E, Nu


def coeff_ex_3(mesh, V):
    """Orthotropic, high coefficient inclusions at the coarse nodes with
    different values on each direction."""
    n = 256

    def find_insertions(x, horizontal=False, vertical=False):
        N = 32
        h = 1 / n
        margin = h
        xcs, ycs = np.meshgrid(np.linspace(0, 1, N + 1), np.linspace(0, 1, N + 1))
        xcs, ycs = xcs.flatten(), ycs.flatten()
        mask = (xcs > 0) & (xcs < 1) & (ycs > 0) & (ycs < 1)
        xcs, ycs = xcs[mask], ycs[mask]
        xs, ys = x[0], x[1]

        in_mask = np.zeros(len(xs), dtype=bool)
        for xc, yc in zip(xcs, ycs):
            if horizontal:
                horizontal_check = (
                    np.isclose(np.abs(xc - xs), (2 * margin))
                    | (np.abs(xc - xs) < (2 * margin))
                ) & (np.isclose(np.abs(yc - ys), margin) | (np.abs(yc - ys) < margin))
                in_mask = in_mask | horizontal_check
            if vertical:
                vertical_check = (
                    np.isclose(np.abs(xc - xs), margin) | (np.abs(xc - xs) < margin)
                ) & (
                    np.isclose(np.abs(yc - ys), (2 * margin))
                    | (np.abs(yc - ys) < (2 * margin))
                )
                in_mask = in_mask | vertical_check

        return in_mask

    Omega_h = locate_entities(mesh, 0, lambda x: find_insertions(x, horizontal=True))
    Omega_v = locate_entities(mesh, 0, lambda x: find_insertions(x, vertical=True))
    Omega_int = np.intersect1d(Omega_h, Omega_v)

    E = Function(V, name="E")
    E.x.array[:] = 1
    E.x.array[2 * Omega_h] = 1e4
    E.x.array[2 * Omega_v + 1] = 1e4
    E.x.array[2 * Omega_int] = E.x.array[2 * Omega_int + 1] = 1e4
    Nu = 0.3

    return E, Nu


def main():
    global dtype
    np.random.seed(42)

    # ----
    # Grid generation
    # ----
    n = 64
    mesh = create_rectangle(
        MPI.COMM_WORLD,
        [np.zeros(2), np.ones(2)],
        [n, n],
        CellType.quadrilateral,
        ghost_mode=GhostMode.shared_facet,
    )


    # ----
    # Definition of the weak form
    # ----
    V = functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
    u, v = TrialFunction(V), TestFunction(V)
    # a = form(inner(sigma(mesh, u, V, coeff_ex_2), eps(v)) * dx)
    a = form(inner(sigma_iso(mesh, u, V, coeff_ex_2), eps(v)) * dx)

    # ----
    # Source term
    # ----
    # f = as_vector((1, 1))
    f = Function(V)
    f.x.array[:] = 5 * np.random.rand(len(f.x.array))
    L = form(inner(f, v) * dx)

    # Boundary conditions
    facets = locate_entities_boundary(
        mesh,
        dim=1,
        marker=lambda x: np.logical_or(
            np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0)),
            np.logical_or(np.isclose(x[0], 1.0), np.isclose(x[1], 1.0)),
        ),
    )
    bc = dirichletbc(
        np.zeros(2, dtype=dtype),
        locate_dofs_topological(V, entity_dim=1, entities=facets),
        V=V,
    )

    # ----
    # Assembly
    # ----
    A = assemble_matrix(a, bcs=[bc])
    A.assemble()

    b = assemble_vector(L)
    # apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
    set_bc(b, [bc])

    # ----
    # Export the system of equations to SciPy's format.
    # ----
    out_dir = "vertex-insertions"
    A_out = csr_matrix(A.getValuesCSR()[::-1], shape=A.size)
    save_npz(f"./output/linear-elasticity/{out_dir}/lhs_{n}x{n}.npz", A_out)
    np.save(f"./output/linear-elasticity/{out_dir}/rhs_{n}x{n}.npy", b.array)
    np.save(
        f"./output/linear-elasticity/{out_dir}/grid_{n}x{n}",
        V.tabulate_dof_coordinates()[:, 0:2].T,
    )

    # ----
    # Solution of the system of equations
    # ----
    opts = PETSc.Options()  # type: ignore
    opts["ksp_type"] = "cg"
    opts["ksp_rtol"] = 1.0e-5
    solver = PETSc.KSP().create(mesh.comm)  # type: ignore
    solver.setFromOptions()
    solver.setOperators(A)

    uh = Function(V)

    # Set a monitor, solve linear system, and display the solver
    # configuration
    solver.setMonitor(
        lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}")
    )
    solver.solve(b, uh.vector)

    # Scatter forward the solution vector to update ghost values
    uh.x.scatter_forward()

    # Translate the mesh nodes according to the displacement computed previously.
    idx = np.arange((n + 1) ** 2)
    mesh.geometry.x[:, 0] += uh.x.array[2 * idx]
    mesh.geometry.x[:, 1] += uh.x.array[2 * idx + 1]

    # E, _ = coeff_ex_2(mesh, V)

    # ----
    # Output
    # ----
    with XDMFFile(mesh.comm, "./displacements_coeff_3.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(uh)
        # file.write_function(E)


if __name__ == "__main__":
    main()
