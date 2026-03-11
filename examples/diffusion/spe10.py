import numpy as np
import ngsolve as ngs
from ..utils import run_example, parse_args, msfem

Nx, Ny = 0, 0
nx, ny = 0, 0


def coeff_eval(x, y):
    pass


def coeff_fem(mesh, *_):
    material_fes = ngs.L2(mesh, order=0)
    material_gf = ngs.GridFunction(material_fes)
    material_gf.vec.FV().NumPy()[:] = np.load("./output/spe10/spe_perm_kxx.npy")[:, :, 72].flatten(order="F")
    return ngs.CoefficientFunction(material_gf).Compile()


def main(args):
    global Nx, Ny, nx, ny
    Nx = args.Nx
    Ny = args.Ny
    nx = args.nx
    ny = args.ny
    run_example(args, coeff_fem, coeff_eval, msfem.NullSpaceType.DIFFUSION)


if __name__ == "__main__":
    args = parse_args("A layer of the SPE10 benchmark.")
    main(args)
