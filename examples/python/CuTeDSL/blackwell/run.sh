#!/bin/bash

# Uncomment the following line to enable debug mode 
# with classic configuration with verbose logging, good for learning and debugging
export DEBUG_MODE=1

export PROFILE_MODE=0 # set to 1 to enable profiling with either Nsight Systems (nsys) or Nsight Compute (ncu)
export PROFILE_TYPE="nsys" # choose from "nsys" or "ncu" when enabling PROFILE_MODE

TEST_SCRIPT="dense_gemm"
# TEST_SCRIPT="dense_gemm_software_pipeline"
# TEST_SCRIPT="dense_gemm_persistent"
# TEST_SCRIPT="dense_blockscaled_gemm_persistent"
# TEST_SCRIPT="grouped_gemm"
# TEST_SCRIPT="fmha"


if [[ $TEST_SCRIPT == "dense_gemm" ]]; then
    SCRIPT_CMD="
    python dense_gemm.py                                     \
    --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                  \
    --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                             \
    --mnkl 8192,8192,8192,1                                                   \
    --use_tma_store --use_2cta_instrs
    "
elif [[ $TEST_SCRIPT == "dense_gemm_software_pipeline" ]]; then
    SCRIPT_CMD="
    python dense_gemm_software_pipeline.py                   \
    --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                  \
    --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                             \
    --mnkl 8192,8192,8192,1                                                   \
    --use_tma_store --use_2cta_instrs
    "
elif [[ $TEST_SCRIPT == "dense_gemm_persistent" ]]; then
    SCRIPT_CMD="
    python dense_gemm_persistent.py                          \
    --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                  \
    --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                             \
    --mnkl 8192,8192,8192,1                                                   \
    --use_tma_store --use_2cta_instrs
    "
elif [[ $TEST_SCRIPT == "dense_blockscaled_gemm_persistent" ]]; then
    SCRIPT_CMD="
    python dense_blockscaled_gemm_persistent.py            \
    --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16        \
    --c_dtype Float16                                                        \
    --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
    --mnkl 8192,8192,1024,1
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