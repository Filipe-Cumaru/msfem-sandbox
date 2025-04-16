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
    Hx = 1 / Nx

    lx_in, ly_in = 2 * hx, 2 * hy
    lx_out, ly_out = 5 * hx, 5 * hy

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
        ox_r = ox + 8 * hx
        if ox_r <= 1 - Hx and ox_r >= Hx:
            dx = np.abs(elem_vertices[:, :, 0] - ox_r)
            dy = np.abs(elem_vertices[:, :, 1] - oy)
            mask = mask | (
                np.all((dx <= lx_out) & (dy <= ly_out), axis=1)
                ^ np.all((dx <= lx_in) & (dy <= ly_in), axis=1)
            )

    material_gf.vec.FV().NumPy()[:] = 1
    material_gf.vec.FV().NumPy()[mask] = 1e8
    return ngs.CoefficientFunction(material_gf).Compile()


if __name__ == "__main__":
    args = parse_args("High coefficient inclusions crossing the subdomain interfaces.")
    Nx = args.Nx
    Ny = args.Ny
    nx = args.nx
    ny = args.ny
    run_example(args, coeff_fem, coeff_eval, msfem.NullSpaceType.DIFFUSION)
