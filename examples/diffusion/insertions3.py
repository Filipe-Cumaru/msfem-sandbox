import numpy as np
import ngsolve as ngs
from ..utils import run_example, parse_args, msfem

Nx, Ny = 0, 0
nx, ny = 0, 0


def coeff_eval(x, y):
    pass


def coeff_fem(mesh):
    global Nx, Ny, nx, ny
    mx, my = Nx * nx + 1, Ny * ny + 1
    hx, hy = 1 / (Nx * nx), 1 / (Ny * ny)

    lx_in, ly_in = 2 * hx, 2 * hy
    lx_out, ly_out = 5 * hx, 5 * hy

    material_fes = ngs.L2(mesh, order=0)
    material_gf = ngs.GridFunction(material_fes)

    Xs, Ys = np.meshgrid(np.linspace(0, 1, Nx + 1)[1:-1], 
                         np.linspace(0, 1, Ny + 1)[1:-1])
    mask = np.zeros(mx * my, dtype=bool)

    elem_vertices = np.array([
        [mesh[vID].point for vID in el.vertices] for el in mesh.Elements()])

    for ox, oy in zip(Xs.flatten(), Ys.flatten()):
        dx = np.abs(elem_vertices[:, :, 0] - ox)
        dy = np.abs(elem_vertices[:, :, 1] - oy)

        mask = mask | \
            (np.all((dx <= lx_out) & (dy <= ly_out), axis=1) \
             ^ np.all((dx <= lx_in) & (dy <= ly_in), axis=1))

    material_gf.vec.FV().NumPy()[:] = 1
    material_gf.vec.FV().NumPy()[mask] = 1e8
    return ngs.CoefficientFunction(material_gf).Compile()


def main(args):
    global Nx, Ny, nx, ny
    Nx = args.Nx
    Ny = args.Ny
    nx = args.nx
    ny = args.ny
    k = args.k
    precond = args.precond
    coarse_space = args.coarse_space
    slab_size = args.slab_size
    run_example(
        Nx,
        Ny,
        nx,
        ny,
        k,
        precond,
        coarse_space,
        slab_size,
        coeff_fem,
        coeff_eval,
        problem_type=msfem.NullSpaceType.DIFFUSION,
        output=args.output,
        enrichment_tol=args.enrichment_tol
    )


if __name__ == "__main__":
    args = parse_args("High coefficient inclusions around the coarse nodes.")
    main(args)
