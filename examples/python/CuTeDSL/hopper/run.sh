#!/bin/bash

# export DEBUG_MODE=1

python dense_gemm.py                                   \
--mnkl 2048,4096,1024,1 --tile_shape_mn 128,256                      \
--cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16           \
--c_dtype Float16 --acc_dtype Float32                                \
--a_major k --b_major k --c_major n > dense_gemm.log 2>&1