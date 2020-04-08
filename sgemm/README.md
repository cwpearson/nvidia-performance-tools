# Matrix-Multiplication Profiling Examples

This code times the execution of a C = A x B matrix multiplication.
C and A are column-major, B is row-major.
A is MxK, B is KxN, C is MxN.

The first multiplication product is checked for correctness on the host.

There are three programs:
* `sgemm-basic` (`basic.cu`): A global-memory multiplication
* `sgemm-tiled` (`tiled.cu`): a shared-memory tiled multiplication
* `sgemm-regtiled-coarsened` (`regtiled_coarsened.cu`): a register-tiled and coarsened multiplication

All programs share the same basic options:

* Three optional positional arguments to set M, N, and K.
* `--iters <int>` the number of measured iterations (default `5`)
* `--warmup <int>` the number of warmup iterations (default `5`)
* `--check`: check correctness (default `false`). Only use for small multiplications

## Optimizing regtiled_coarsened

regtiled_coarsened.cu: 

theoretical occupancy is 75% instead of 100%.
limited to 12 blocks per SM by registers

Achieved occupancy is only 6.25%