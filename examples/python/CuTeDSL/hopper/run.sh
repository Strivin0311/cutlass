#!/bin/bash

# Uncomment the following line to enable debug mode 
# with classic configuration with verbose logging, good for learning and debugging
export DEBUG_MODE=1

export PROFILE_MODE=0 # set to 1 to enable profiling with either Nsight Systems (nsys) or Nsight Compute (ncu)
export PROFILE_TYPE="nsys" # choose from "nsys" or "ncu" when enabling PROFILE_MODE

if [[ $DEBUG_MODE -eq 1 ]]; then
    M=2048
    K=4096
    N=1024
    PROFILE_MODE=0 # disable profiling when in debug mode to avoid conflicts with verbose logging
elif [[ $PROFILE_MODE -eq 1 ]]; then
    M=6144
    K=2048
    N=8192
else
    M=8192
    K=8192
    N=8192
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

if [[ $PROFILE_MODE -eq 1 ]]; then
    echo "Profiling configuration: M=$M, K=$K, N=$N, Profile Type=$PROFILE_TYPE"

    if [[ $PROFILE_TYPE == "nsys" ]]; then
        mkdir -p nsys_reps
        PROFILE_CMD="nsys profile -o nsys_reps/dense_gemm_$TIMESTAMP -f true --capture-range=cudaProfilerApi "
    elif [[ $PROFILE_TYPE == "ncu" ]]; then
        mkdir -p ncu_reps
        PROFILE_CMD="ncu --set full --kernel-name regex:kernel_cutlass -f -o ncu_reps/dense_gemm_$TIMESTAMP "
    else
        echo "Unsupported PROFILE_TYPE: $PROFILE_TYPE"
        exit 1
    fi
fi


SCRIPT_CMD="python dense_gemm.py                                   \
--mnkl $M,$K,$N,1 --tile_shape_mn 128,256                      \
--cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16           \
--c_dtype Float16 --acc_dtype Float32                                \
--a_major k --b_major k --c_major n "


mkdir -p logs

if [[ $PROFILE_MODE -eq 1 ]]; then
    echo "Running in profile mode with $PROFILE_TYPE and logging to logs/prof_dense_gemm.log ..."
    eval $PROFILE_CMD $SCRIPT_CMD > logs/prof_dense_gemm.log 2>&1
elif [[ $DEBUG_MODE -eq 1 ]]; then
    echo "Running in debug mode and logging to logs/debug_dense_gemm.log ..."
    eval $SCRIPT_CMD > logs/debug_dense_gemm.log 2>&1
else
    echo "Running in test mode and logging to logs/test_dense_gemm.log ..."
    eval $SCRIPT_CMD > logs/test_dense_gemm.log 2>&1
fi