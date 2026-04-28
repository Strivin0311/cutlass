#!/bin/bash

# Uncomment the following line to enable debug mode 
# with classic configuration with verbose logging, good for learning and debugging
export DEBUG_MODE=1

if [[ $DEBUG_MODE -eq 1 ]]; then
    M=2048
    K=4096
    N=1024
else
    M=8192
    K=8192
    N=8192
fi

python dense_gemm.py                                   \
--mnkl $M,$K,$N,1 --tile_shape_mn 128,256                      \
--cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16           \
--c_dtype Float16 --acc_dtype Float32                                \
--a_major k --b_major k --c_major n > dense_gemm.log 2>&1