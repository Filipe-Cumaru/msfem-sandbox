# msfem-sandbox

A collection of coarse spaces for the two-level additive Schwarz preconditioner implemented in Python for 2-D elliptic problems.

## Getting Started

### Dependencies

This project uses [Netgen/NGSolve](https://ngsolve.org/) in order to assemble the finite-element problem. It and all the remaining Python dependencies can be installed via:
```
pip install -r requirements.txt
```

### Running the examples

In the `examples` directory there are a number of examples of applications of the Schwarz preconditioner for the scalar diffusion problem and also the linear elasticity problem.

To check the execution options of an example, run:
```
python3 -m examples.{problem-type}.{example-name} --help
```
where `problem-type` is either `diffusion` or `linear_elasticity`, and `example-name` is the name of the example that you wish to run. The example name corresponds to the name of the Python file in `examples/{problem-type}` without the `.py` extension. For instance:
```
python3 -m examples.linear_elasticity.dohrmann --help
```

## Authors

[Filipe Cumaru](https://filipe-cumaru.github.io/)

## References

Much of the work done in this repository is based on the references below.

```
@book{Toselli2010,
   author = {Andrea Toselli and Olof Widlund},
   publisher = {Springer},
   title = {Domain decomposition methods - algorithms and theory},
   year = {2010},
}

@article{Dohrmann2017,
   author = {Clark R Dohrmann and Olof B Widlund},
   doi = {10.1137/17M1114272},
   issue = {4},
   journal = {SIAM Journal on Scientific Computing},
   pages = {A1466-A1488},
   title = {On the Design of Small Coarse Spaces for Domain Decomposition Algorithms},
   volume = {39},
   url = {https://doi.org/10.1137/17M1114272},
   year = {2017},
}

@article{Heinlein2018,
   author = {Alexander Heinlein and Axel Klawonn and Jascha Knepper and Oliver Rheinbach},
   doi = {10.1553/etna_vol48s156},
   issn = {1068-9613},
   journal = {ETNA - Electronic Transactions on Numerical Analysis},
   pages = {156-182},
   title = {Multiscale coarse spaces for overlapping Schwarz methods based on the ACMS space in 2D},
   volume = {48},
   year = {2018},
}

@article{Wang2014,
   author = {Yixuan Wang and Hadi Hajibeygi and Hamdi A Tchelepi},
   doi = {https://doi.org/10.1016/j.jcp.2013.11.024},
   issn = {0021-9991},
   journal = {Journal of Computational Physics},
   keywords = {Algebraic multiscale solver,Iterative multiscale methods,Multiscale methods,Scalable linear solvers},
   pages = {284-303},
   title = {Algebraic multiscale solver for flow in heterogeneous porous media},
   volume = {259},
   url = {https://www.sciencedirect.com/science/article/pii/S0021999113007869},
   year = {2014},
}

@article{Souza2022,
   title = {An algebraic multiscale solver for the simulation of two-phase flow in heterogeneous and anisotropic porous media using general unstructured grids (AMS-U)},
   journal = {Applied Mathematical Modelling},
   volume = {103},
   pages = {792-823},
   year = {2022},
   issn = {0307-904X},
   doi = {https://doi.org/10.1016/j.apm.2021.11.017},
   url = {https://www.sciencedirect.com/science/article/pii/S0307904X21005552},
   author = {Artur Castiel Reis {de Souza} and Darlan Karlo Elisiário {de Carvalho} and José Cícero Araujo {dos Santos} and Ramiro Brito Willmersdorf and Paulo Roberto Maciel Lyra and Michael G. Edwards},
   keywords = {MsFV, AMS, MPFA-D, Unstructured grids, Background grid, Reservoir simulation}
}

@misc{Alves2024,
   title={A computational study of algebraic coarse spaces for two-level overlapping additive Schwarz preconditioners}, 
   author={Filipe A. C. S. Alves and Alexander Heinlein and Hadi Hajibeygi},
   year={2024},
   eprint={2408.08187},
   archivePrefix={arXiv},
   primaryClass={math.NA},
   url={https://arxiv.org/abs/2408.08187}, 
}
```

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
