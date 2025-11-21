#!/bin/bash

# python dense_gemm.py                                     \
#     --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                  \
#     --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                             \
#     --mnkl 8192,8192,8192,1                                                   \
#     --use_tma_store --use_2cta_instrs

# python dense_gemm_software_pipeline.py                   \
#       --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                  \
#       --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                             \
#       --mnkl 8192,8192,8192,1                                                   \
#       --use_tma_store --use_2cta_instrs

# python dense_gemm_persistent.py                          \
#       --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                  \
#       --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                             \
#       --mnkl 8192,8192,8192,1                                                   \
#       --use_tma_store --use_2cta_instrs

# python dense_blockscaled_gemm_persistent.py            \
#       --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16        \
#       --c_dtype Float16                                                        \
#       --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
#       --mnkl 8192,8192,1024,1

# python grouped_gemm.py                                                 \
#     --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                                \
#     --mma_tiler_mn 128,64 --cluster_shape_mn 1,1                                            \
#     --problem_sizes_mnkl "(8192,1280,32,1),(16,384,1536,1),(640,1280,16,1),(640,160,16,1)"  \
#     --num_groups 4  --tensormap_update_mode SMEM

python fmha.py                                     \
      --qk_acc_dtype Float32 --pv_acc_dtype Float32                       \
      --mma_tiler_mn 128,128                                              \
      --q_shape 4,1024,8,64 --k_shape 4,1024,8,64                         \
      --is_persistent