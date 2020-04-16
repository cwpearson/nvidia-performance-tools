# Matrix-Multiplication Profiling Examples

This code contains a global memory, shared-memory tiled, and joint shared-memory and register-tiled matrix matrix multiplications.


## Module 1: Nvidia Nsight Compute

Examples for using Nsight Compute to compare kernel performance.

* `1-1-pinned-basic`: (`1_1_pinned_basic.cu`)
* `1-2-pinned-tiled`: (`1_1_pinned_tiled.cu`)
* `1-3-pinned-joint`: (`1_1_pinned_joint.cu`)

## Module 2: Nvidia Nsight Systems

Examples for using Nsight Systems to compare data transfer, and relationship between data transfer and end-to-end time.

* `2-1-pageable-basic`: (`2_1_pageable_basic.cu`)
* `2-2-pinned-basic`: (`2_2_pinned_basic.cu`)
* `2-3-pinned-tiled`: (`2_3_pinned_tiled.cu`)
* `2-4-pinned-tiled-overlap`: (`2_4_pinned_tiled_overlap.cu`)
* `2-5-pinned-joint`: (`2_5_pinned_joint.cu`)
* `2-6-pinned-joint-overlap`: (`2_6_pinned_joint_overlap.cu`)

All programs share the same basic options:

* Three optional positional arguments to set M, N, and K.
* `--iters <int>` the number of measured iterations (default `5`)
* `--warmup <int>` the number of warmup iterations (default `5`)
* `--check`: check correctness (default `false`). Only use for small multiplications
