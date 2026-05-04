#!/bin/bash

# Uncomment the following line to enable debug mode
# with classic configuration with verbose logging, good for learning and debugging
export DEBUG_MODE=1

export PROFILE_MODE=0 # set to 1 to enable profiling with either Nsight Systems (nsys) or Nsight Compute (ncu)
export PROFILE_TYPE="nsys" # choose from "nsys" or "ncu" when enabling PROFILE_MODE

# TEST_SCRIPT="elementwise_add"
# TEST_SCRIPT="elementwise_apply"
# TEST_SCRIPT="sgemm"
TEST_SCRIPT="tensorop_gemm"
# TEST_SCRIPT="flash_attention_v2"
# TEST_SCRIPT="smem_allocator"
# TEST_SCRIPT="dynamic_smem_size"
# TEST_SCRIPT="call_bypass_dlpack"
# TEST_SCRIPT="call_from_jit"

if [[ $DEBUG_MODE -eq 1 ]]; then
    M=2048
    K=4096
    N=1024
    SEQLEN_Q=2048
    SEQLEN_K=4096
    BATCH_SIZE=1
    PROFILE_MODE=0 # disable profiling when in debug mode to avoid conflicts with verbose logging
elif [[ $PROFILE_MODE -eq 1 ]]; then
    M=6144
    K=2048
    N=8192
    SEQLEN_Q=4096
    SEQLEN_K=4096
    BATCH_SIZE=1
else
    M=8192
    K=8192
    N=8192
    SEQLEN_Q=1280
    SEQLEN_K=1536
    BATCH_SIZE=1
fi


if [[ $TEST_SCRIPT == "elementwise_add" ]]; then
    # Bandwidth: 2.6TB/s for fp32
    SCRIPT_CMD="
    python elementwise_add.py                                             \
    --M $M --N $N    \
    --benchmark                                                     
    "
elif [[ $TEST_SCRIPT == "elementwise_apply" ]]; then
    SCRIPT_CMD="
    python elementwise_apply.py                                           \
    --M $M --N $N   \
    --op mul \
    --benchmark                                                                
    "
elif [[ $TEST_SCRIPT == "sgemm" ]]; then
    # TFLOPS: 14 for fp32
    SCRIPT_CMD="
    python sgemm.py                                                       \
    --mnk $M,$K,$N                                                        \
    --a_major k --b_major k --c_major n
    "
elif [[ $TEST_SCRIPT == "tensorop_gemm" ]]; then
    # TFLOPS: 353 for fp16
    SCRIPT_CMD="
    python tensorop_gemm.py                                               \
    --mnkl $M,$K,$N,1 --atom_layout_mnk 2,2,1                            \
    --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32          \
    --a_major k --b_major k --c_major n
    "
elif [[ $TEST_SCRIPT == "flash_attention_v2" ]]; then
    SCRIPT_CMD="
    python flash_attention_v2.py                                          \
    --dtype Float16 --head_dim 128 --m_block_size 128 --n_block_size 128 \
    --num_threads 128 --batch_size $BATCH_SIZE                            \
    --seqlen_q $SEQLEN_Q --seqlen_k $SEQLEN_K                            \
    --num_head 16 --softmax_scale 1.0 --is_causal
    "
elif [[ $TEST_SCRIPT == "smem_allocator" ]]; then
    SCRIPT_CMD="python smem_allocator.py"
elif [[ $TEST_SCRIPT == "dynamic_smem_size" ]]; then
    SCRIPT_CMD="python dynamic_smem_size.py"
elif [[ $TEST_SCRIPT == "call_bypass_dlpack" ]]; then
    SCRIPT_CMD="python call_bypass_dlpack.py"
elif [[ $TEST_SCRIPT == "call_from_jit" ]]; then
    SCRIPT_CMD="python call_from_jit.py"
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