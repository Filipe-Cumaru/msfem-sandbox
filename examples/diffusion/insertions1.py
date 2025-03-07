import numpy as np
import ngsolve as ngs
from ..utils import run_example, parse_args, msfem

Nx, Ny = 0, 0
nx, ny = 0, 0


def coeff_eval(x, y):
    global Nx, Ny, nx, ny
    mx, my = Nx * nx, Ny * ny
    hx, hy = 1 / mx, 1 / my
    lx, ly = 2 * hx, 2 * hy
    xcs, ycs = np.meshgrid(np.linspace(0, 1, Nx + 1), np.linspace(0, 1, Ny + 1))
    xcs, ycs = xcs.flatten(), ycs.flatten()
    mask = (xcs > 0) & (xcs < 1) & (ycs > 0) & (ycs < 1)
    xcs, ycs = xcs[mask], ycs[mask]

    coarse_node_check = (np.isclose(np.abs(xcs - x), lx) | (np.abs(xcs - x) < lx)) & (
        np.isclose(np.abs(ycs - y), ly) | (np.abs(ycs - y) < ly)
    )

    return 1e8 if np.any(coarse_node_check) else 1


def coeff_fem():
    global Nx, Ny, nx, ny
    Hx, Hy = 1 / Nx, 1 / Ny
    hx, hy = 1 / (Nx * nx), 1 / (Ny * ny)
    lx, ly = 2 * hx, 2 * hy
    c = 0
    for i in range(1, Nx):
        for j in range(1, Ny):
            ox, oy = i * Hx, j * Hy
            c += ngs.IfPos(
                ngs.sqrt(((ngs.x - ox) / lx + (ngs.y - oy) / ly) ** 2)
                + ngs.sqrt(((ngs.x - ox) / lx - (ngs.y - oy) / ly) ** 2)
                - 2,
                1 / ((Nx - 1) * (Ny - 1)),
                1e8,
            )
    return c


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
