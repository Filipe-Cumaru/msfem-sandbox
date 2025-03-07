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
    lx, Lx = 2 * hx, 5 * hx
    ly, Ly = 2 * hy, 5 * hy
    c0, c1 = 0, 0
    for i in range(1, Nx):
        for j in range(1, Ny):
            ox, oy = i * Hx, j * Hy
            c0 += ngs.IfPos(
                ngs.sqrt(((ngs.x - ox) / lx + (ngs.y - oy) / ly) ** 2)
                + ngs.sqrt(((ngs.x - ox) / lx - (ngs.y - oy) / ly) ** 2)
                - 2,
                0,
                1e8,
            )
            c1 += ngs.IfPos(
                ngs.sqrt(((ngs.x - ox) / Lx + (ngs.y - oy) / Ly) ** 2)
                + ngs.sqrt(((ngs.x - ox) / Lx - (ngs.y - oy) / Ly) ** 2)
                - 2,
                0,
                1e8,
            )
    c2 = ngs.IfPos(c1 - c0, 0, 1)
    c = (c1 - c0) + c2
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
    args = parse_args("High coefficient inclusions around the coarse nodes.")
    main(args)
