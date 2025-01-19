# ATSC_project
Project of the course Advanced Topics in Scientific Computing
Bulk-surface multiphysiscs coupling: a CutFEM approach with boundary conditions à la Nitsche.

In this project we develop a deal.II code in which a coupled bulk-interface partial differential equation system s solved through the CutFEM method.
The Robin boundary conditions are imposed weakly with Nitsche method.

Some possible future extensions:
stabilization of the surface solution;
parallelization;
explicit imposition of the constarint to guarantee uniqueness.
