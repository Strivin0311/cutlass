#!/bin/bash

# call_bypass_dlpack
# python call_bypass_dlpack.py; exit 0

# call_from_jit.py
# python call_from_jit.py; exit 0

# dynamic_smem_size
# python dynamic_smem_size.py; exit 0

# elementwise_add
# python elementwise_add.py --M 1024 --N 1024 --benchmark --warmup_iterations 2 --iterations 1000; exit 0

# elementwise_apply
# python elementwise_apply.py --M 2048 --N 2048 --op add --benchmark --warmup_iterations 2 --iterations 10; exit 0

# smem_allocator
# python smem_allocator.py; exit 0

# sgemm
# python sgemm.py                       \
# --mnk 8192,8192,8192                                \
# --a_major m --b_major n --c_major n; exit 0

# tensorop_gemm
# python tensorop_gemm.py                                  \
# --mnkl 8192,8192,8192,1 --atom_layout_mnk 2,2,1                        \
# --ab_dtype Float16                                                     \
# --c_dtype Float16 --acc_dtype Float32                                  \
# --a_major m --b_major n --c_major n; exit 0

# flash_attention_v2
python flash_attention_v2.py                                            \
--dtype Float16 --head_dim 128 --m_block_size 128 --n_block_size 128                  \
--num_threads 128 --batch_size 1 --seqlen_q 1280 --seqlen_k 1536                      \
--num_head 16 --softmax_scale 1.0 --is_causal; exit 0