import numpy as np
import ngsolve as ngs
from ..utils import run_example, parse_args, msfem

Nx, Ny = 0, 0
nx, ny = 0, 0


def coeff_eval(*_):
    pass


def coeff_fem(mesh, P):
    global Nx, Ny, nx, ny
    hx, hy = 1 / (Nx * nx), 1 / (Ny * ny)

    lx_center, ly_center = 2 * hx, 2 * hy
    lx_side, ly_side = hx, 3 * hy

    material_fes = ngs.L2(mesh, order=0)
    material_gf = ngs.GridFunction(material_fes)

    xs, ys = np.meshgrid(np.linspace(0, 1, Nx * nx + 1), np.linspace(0, 1, Ny * ny + 1))
    xs, ys = xs.flatten(), ys.flatten()
    coarse_nodes = np.where(P.sum(axis=1) > 2)[0]
    Xs, Ys = xs[coarse_nodes], ys[coarse_nodes]

    elem_vertices = np.array(
        [[mesh[vID].point for vID in el.vertices] for el in mesh.Elements()]
    )
    mask = np.zeros(elem_vertices.shape[0], dtype=bool)

    for ox, oy in zip(Xs.flatten(), Ys.flatten()):
        ox_r, ox_l = ox + 6 * hx, ox - 6 * hx

        dx = np.abs(elem_vertices[:, :, 0] - ox)
        dy = np.abs(elem_vertices[:, :, 1] - oy)

        dx_r = np.abs(elem_vertices[:, :, 0] - ox_r)
        dx_l = np.abs(elem_vertices[:, :, 0] - ox_l)

        mask = (
            mask
            | np.all((dx <= lx_center) & (dy <= ly_center), axis=1)
            | np.all((dx_r <= lx_side) & (dy <= ly_side), axis=1)
            | np.all((dx_l <= lx_side) & (dy <= ly_side), axis=1)
        )

    material_gf.vec.FV().NumPy()[:] = 1
    material_gf.vec.FV().NumPy()[mask] = 1e8
    return ngs.CoefficientFunction(material_gf).Compile()


if __name__ == "__main__":
    args = parse_args(
        "High coefficient inclusions crossing the horizontal subdomain edges. NOTE: this example is only consistent with structured domain decompositions."
    )
    Nx = args.Nx
    Ny = args.Ny
    nx = args.nx
    ny = args.ny
    run_example(args, coeff_fem, coeff_eval, msfem.NullSpaceType.DIFFUSION)
