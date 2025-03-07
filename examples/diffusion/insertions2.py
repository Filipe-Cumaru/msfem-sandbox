import numpy as np
import ngsolve as ngs
from ..utils import run_example, parse_args, msfem

Nx, Ny = 0, 0
nx, ny = 0, 0


def coeff_eval(x, y):
    pass


def coeff_fem():
    global Nx, Ny, nx, ny
    Hx, Hy = 1 / Nx, 1 / Ny
    hx, hy = 1 / (Nx * nx), 1 / (Ny * ny)
    lx_center, ly_center = 2 * hx, 2 * hy
    lx_side, ly_side = hx, 3 * hy
    c0, c1, c2 = 0, 0, 0
    for i in range(1, Nx):
        for j in range(1, Ny):
            ox, oy = i * Hx, j * Hy
            ox_r, ox_l = i * Hx + 6 * hx, i * Hx - 6 * hx
            c0 += ngs.IfPos(
                ngs.sqrt(((ngs.x - ox) / lx_center + (ngs.y - oy) / ly_center) ** 2)
                + ngs.sqrt(((ngs.x - ox) / lx_center - (ngs.y - oy) / ly_center) ** 2)
                - 2,
                0,
                1e8,
            )
            c1 += ngs.IfPos(
                ngs.sqrt(((ngs.x - ox_l) / lx_side + (ngs.y - oy) / ly_side) ** 2)
                + ngs.sqrt(((ngs.x - ox_l) / lx_side - (ngs.y - oy) / ly_side) ** 2)
                - 2,
                0,
                1e8,
            )
            c2 += ngs.IfPos(
                ngs.sqrt(((ngs.x - ox_r) / lx_side + (ngs.y - oy) / ly_side) ** 2)
                + ngs.sqrt(((ngs.x - ox_r) / lx_side - (ngs.y - oy) / ly_side) ** 2)
                - 2,
                0,
                1e8,
            )
    c3 = ngs.IfPos(c0 + c1 + c2, 0, 1)
    c = c0 + c1 + c2 + c3
    return c.Compile()


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
    args = parse_args("High coefficient inclusions on the coarse nodes and crossing the interface of the subdomains.")
    main(args)
