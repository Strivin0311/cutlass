#!/bin/bash

# Uncomment the following line to enable debug mode 
# with classic configuration with verbose logging, good for learning and debugging
# export DEBUG_MODE=1

export PROFILE_MODE=1 # set to 1 to enable profiling with either Nsight Systems (nsys) or Nsight Compute (ncu)
export PROFILE_TYPE="nsys" # choose from "nsys" or "ncu" when enabling PROFILE_MODE

# TEST_SCRIPT="dense_gemm"
# TEST_SCRIPT="dense_gemm_software_pipeline"
# TEST_SCRIPT="dense_gemm_persistent"
# TEST_SCRIPT="dense_blockscaled_gemm_persistent"
TEST_SCRIPT="grouped_gemm"
# TEST_SCRIPT="fmha"

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


if [[ $TEST_SCRIPT == "dense_gemm" ]]; then
    # PFLOPS: 1.288 for fp16
    SCRIPT_CMD="
    python dense_gemm.py                                     \
    --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                  \
    --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                             \
    --mnkl $M,$K,$N,1                                                   \
    --use_tma_store --use_2cta_instrs
    "
elif [[ $TEST_SCRIPT == "dense_gemm_software_pipeline" ]]; then
    # PFLOPS: 1.338 for fp16
    SCRIPT_CMD="
    python dense_gemm_software_pipeline.py                   \
    --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                  \
    --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                             \
    --mnkl $M,$K,$N,1                                                   \
    --use_tma_store --use_2cta_instrs
    "
elif [[ $TEST_SCRIPT == "dense_gemm_persistent" ]]; then
    # PFLOPS: 1.431 for fp16
    SCRIPT_CMD="
    python dense_gemm_persistent.py                          \
    --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                  \
    --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                             \
    --mnkl $M,$K,$N,1                                                   \
    --use_tma_store --use_2cta_instrs
    "
elif [[ $TEST_SCRIPT == "dense_blockscaled_gemm_persistent" ]]; then
    # PFLOPS: 10.8 for fp4
    SCRIPT_CMD="
    python dense_blockscaled_gemm_persistent.py            \
    --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16        \
    --c_dtype Float16                                                        \
    --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
    --mnkl $M,$K,$N,1
    "
elif [[ $TEST_SCRIPT == "grouped_gemm" ]]; then
    SCRIPT_CMD="
    python grouped_gemm.py                                                 \
    --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                                \
    --mma_tiler_mn 128,64 --cluster_shape_mn 1,1                                            \
    --problem_sizes_mnkl \"(8192,1280,32,1),(16,384,1536,1),(640,1280,16,1),(640,160,16,1)\"  \
    --num_groups 4  --tensormap_update_mode SMEM
    "
elif [[ $TEST_SCRIPT == "fmha" ]]; then
    SCRIPT_CMD="
    python fmha.py                                     \
    --qk_acc_dtype Float32 --pv_acc_dtype Float32                       \
    --mma_tiler_mn 128,128                                              \
    --q_shape 4,1024,8,64 --k_shape 4,1024,8,64                         \
    --is_persistent
    "
else
    echo "Unsupported TEST_SCRIPT: $TEST_SCRIPT"
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

if [[ $PROFILE_MODE -eq 1 ]]; then
    echo "Profiling configuration: M=$M, K=$K, N=$N, Profile Type=$PROFILE_TYPE"

    if [[ $PROFILE_TYPE == "nsys" ]]; then
        mkdir -p nsys_reps
        PROFILE_CMD="nsys profile -o nsys_reps/${TEST_SCRIPT}_$TIMESTAMP -f true --capture-range=cudaProfilerApi "
    elif [[ $PROFILE_TYPE == "ncu" ]]; then
        mkdir -p ncu_reps
        PROFILE_CMD="ncu --set full --kernel-name regex:kernel_cutlass -f -o ncu_reps/${TEST_SCRIPT}_$TIMESTAMP "
    else
        echo "Unsupported PROFILE_TYPE: $PROFILE_TYPE"
        exit 1
    fi
fi

mkdir -p logs

if [[ $PROFILE_MODE -eq 1 ]]; then
    echo "Running in profile mode with $PROFILE_TYPE and logging to logs/prof_${TEST_SCRIPT}.log ..."
    eval $PROFILE_CMD $SCRIPT_CMD > logs/prof_${TEST_SCRIPT}.log 2>&1
elif [[ $DEBUG_MODE -eq 1 ]]; then
    echo "Running in debug mode and logging to logs/debug_${TEST_SCRIPT}.log ..."
    eval $SCRIPT_CMD > logs/debug_${TEST_SCRIPT}.log 2>&1
else
    echo "Running in test mode and logging to logs/test_${TEST_SCRIPT}.log ..."
    eval $SCRIPT_CMD > logs/test_${TEST_SCRIPT}.log 2>&1
fi