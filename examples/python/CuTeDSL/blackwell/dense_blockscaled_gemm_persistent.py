# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import os
from typing import Optional, Type, Tuple, Union

import cuda.bindings.driver as cuda
import torch

import cutlass
from cutlass import const_expr
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import from_dlpack

"""
This example provides an experimental implementation of the SM100 batched dense blockscaled GEMM kernel, please note that the APIs and implementation details related to this kernel may change in future releases.

A high-performance persistent batched dense blockscaled GEMM example for the NVIDIA Blackwell SM100 architecture
using CUTE DSL.
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M") for MXF8 input type and can only be row-major("K") for MXF4/NVF4 input type
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K") for MXF8 input type and can only be row-major("K") for MXF4/NVF4 input type
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")
- Matrix SFA layout is filled internally according to A shape and BlockScaledBasicChunk, which has Mxceil_div(K, sf_vec_size)xL elements respectively
- Matrix SFB layout is filled internally according to B shape and BlockScaledBasicChunk, which has Nxceil_div(K, sf_vec_size)xL elements respectively

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Blackwell's tcgen05.mma for matrix multiply-accumulate (MMA) operations (including 2cta mma instructions)
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Support persistent tile scheduling to better overlap memory load/store with mma between tiles
    - Support warp specialization to avoid explicit pipelining between mainloop load and mma

This GEMM works as follows:
1. DMA warp: Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. MMA warp:
    - Load scale factor A/B from shared memory (SMEM) to tensor memory (TMEM) using tcgen05.cp instruction.
    - Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
3. EPILOGUE warp:
    - Load completed accumulator from tensor memory (TMEM) to registers (RMEM) using tcgen05.ld.
    - Type convert C matrix to output type.
    - Optionally store C matrix from registers (RMEM) to shared memory (SMEM) to global memory (GMEM) with TMA operations,
      or directly store C matrix from registers (RMEM) to global memory (GMEM) without TMA operations.
    - Optionally accept an elementwise lambda function epilogue_op to apply to the output tensor:
      e.g., relu can set epilogue_op = lambda x: cute.where(x > 0, x, cute.full_like(x, 0))

SM100 tcgen05.mma.kind.block_scale instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Read scalefactor A from TMEM
- Read scalefactor B from TMEM
- Write accumulator to TMEM
The accumulator in TMEM must then be loaded to registers before writing back to GMEM.

Input arguments to this example is shown below:

.. code-block:: bash

    python examples/blackwell/dense_blockscaled_gemm_persistent.py            \
      --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16        \
      --c_dtype Float16                                                        \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
      --mnkl 8192,8192,1024,1

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/blackwell/dense_blockscaled_gemm_persistent.py        \
      --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16        \
      --c_dtype Float16                                                        \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
      --mnkl 8192,8192,1024,1                                                  \
      --warmup_iterations 1 --iterations 10 --skip_ref_check


Constraints:
* Supported input data types: mxf8, mxf4, nvf4
  see detailed valid dtype combinations in below BlockScaledDenseGemmPersistentKernelSm100 class documentation
* A/B tensor must have the same data type, mixed data type is not supported (e.g., mxf8 x mxf4)
* Mma tiler M must be 128 or 256(use_2cta_instrs)
* Mma tiler N must be 128 or 256
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if Mma tiler M is 256(use_2cta_instrs)
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 16 and 32 for Float8 and Float4, respectively.
"""

DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"
PROFILE_MODE = os.environ.get("PROFILE_MODE", "0") == "1"


class BlockScaledDenseGemmPersistentKernelSm100:
    """This class implements batched matrix multiplication (C = A x SFA x B x SFB) with support for various data types
    and architectural features specific to Blackwell GPUs with persistent tile scheduling and warp specialization.

    :param sf_vec_size: Scalefactor vector size.
    :type sf_vec_size: int
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: In current version, A and B tensor must have the same data type
        - i.e., Float8E4M3FN for A and Float8E5M2 for B is not supported

    :note: Supported combinations of A/B data types, SF data typs and SF vector size:
        - MXF8: A/B: Float8E5M2/Float8E4M3FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - MXF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - NVF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU/Float8E4M3FN + sf_vec_size: 16

    :note: Supported accumulator data types:
        - Float32

    :note: Supported C data types:
        - Float32
        - Float16/BFloat16
        - Float8E4M3FN/Float8E5M2
    :note: Constraints:
        - MMA tiler M must be 128 or 256 (use_2cta_instrs)
        - MMA tiler N must be 128/256
        - Cluster shape M must be multiple of 2 if Mma tiler M is 256
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Also, Cluster shape M/N must be <= 4 for scale factor multicasts due to limited size of scale factors

    Example:
        >>> gemm = BlockScaledDenseGemmPersistentKernelSm100(
        ...     sf_vec_size=16,
        ...     mma_tiler_mn=(256, 128),
        ...     cluster_shape_mn=(2, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor, max_active_clusters, stream)
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        debug_print: bool = False,
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator, always set to Float32
            - sf_vec_size: Scalefactor A/B vector size.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        :param sf_vec_size: Scalefactor vector size.
        :type sf_vec_size: int
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        """

        self.acc_dtype = cutlass.Float32 # must be fp32
        self.sf_vec_size = sf_vec_size # 16
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn # (CGA_M2, CGA_N1)
        
        # K dimension is deferred in _setup_attributes
        self.mma_tiler_mnk = (*mma_tiler_mn, 1) # (tileM256, tileN128)

        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1 # we only want one CTA to reside on one SM
        
        self.buffer_align_bytes = 1024
        
        # Set specialized warp ids
        self.epilog_warp_id = (0, 1, 2, 3) # the first warp group forms the epilogue consumer warps
        self.mma_warp_id = 4 # a single warp for umma consumer / acc producer
        self.tma_warp_id = 5 # a single warp for tma producer
        
        self.epilogue_threads = 32 * len(self.epilog_warp_id)
        self.tmem_ptr_read_threads = 32 + self.epilogue_threads  # all threads in mma warp and epilogue warps can read tmem ptr from shared memory
        self.threads_per_cta = 32 + self.tmem_ptr_read_threads
        
        # Set barrier id for cta sync, epilogue sync and tmem ptr sync
        self.cta_sync_bar_id = 0
        self.epilog_sync_bar_id = 1
        self.tmem_ptr_sync_bar_id = 2
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.debug_print = debug_print

        if const_expr(self.debug_print):
            print()
            print(f"Initialized BlockScaledDenseGemmPersistentKernelSm100 with configurations:")
            print(f"  acc_dtype: {self.acc_dtype=}")
            print(f"  sf_vec_size: {self.sf_vec_size=}")
            print(f"  use_2cta_instrs: {self.use_2cta_instrs=}")
            print(f"  mma_tiler: {self.mma_tiler_mnk=}")
            print(f"  cluster_shape_mn: {self.cluster_shape_mn=}")
            print(f"  CTA group for MMA: {self.cta_group=}")
            print(f"  warp ids: {self.epilog_warp_id=}, {self.mma_warp_id=}, {self.tma_warp_id=}")
            print(f"  barrier ids: {self.cta_sync_bar_id=}, {self.epilog_sync_bar_id=}, {self.tmem_ptr_sync_bar_id=}")
            print(f"  epilogue_threads: {self.epilogue_threads=}")
            print(f"  tmem_ptr_read_threads: {self.tmem_ptr_read_threads=}")
            print(f"  threads_per_cta: {self.threads_per_cta=}")
            print(f"  smem_capacity: {self.smem_capacity=} bytes")
            print(f"  num_tmem_alloc_cols: {self.num_tmem_alloc_cols=}")

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B/SFA/SFB
        - Computing epilogue subtile
        - Setting up A/B/SFA/SFB/C stage counts in shared memory
        - Computing A/B/SFA/SFB/C shared memory layout
        - Computing tensor memory allocation columns
        """
        # Compute mma instruction shapes
        mma_inst_bits_k = 256 # 32B
        self.mma_inst_shape_mnk = ( # (m256, n128, k64)
            self.mma_tiler_mnk[0],
            self.mma_tiler_mnk[1],
            mma_inst_bits_k // self.a_dtype.width, # 256 / sizeof(fp4e2m1) = 64
        )
        self.mma_inst_shape_mnk_sfb = ( # (CTA_m128, n128, k64)
            self.mma_inst_shape_mnk[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mnk[1], 128),
            self.mma_inst_shape_mnk[2],
        )

        # Make tiled mma
        # ThrID:           2:1
        # Shape MNK:       (256,128,64)
        # TV Layout A:     (2,(128,64)):(128,(1,256))
        # TV Layout B:     (2,(64,64)):(64,(1,128))
        # TV Layout C:     (2,(128,128)):(128,(1,256))
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            ab_dtype=self.a_dtype,
            a_leading_mode=self.a_major_mode,
            b_leading_mode=self.b_major_mode,
            sf_dtype=self.sf_dtype,
            sf_vec_size=self.sf_vec_size,
            cta_group=self.cta_group,
            mma_tiler_mn=self.mma_inst_shape_mnk[:2],
        )

        # Make tiled mma for SFB
        # ThrID:           1:0
        # Shape MNK:       (128,128,64)
        # TV Layout A:     (1,(128,64)):(128,(1,128))
        # TV Layout B:     (1,(128,64)):(128,(1,128))
        # TV Layout C:     (1,(128,128)):(128,(1,128)) 
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            ab_dtype=self.a_dtype,
            a_leading_mode=self.a_major_mode,
            b_leading_mode=self.b_major_mode,
            sf_dtype=self.sf_dtype,
            sf_vec_size=self.sf_vec_size,
            cta_group=tcgen05.CtaGroup.ONE,
            mma_tiler_mn=self.mma_inst_shape_mnk_sfb[:2],
        )
        
        self.tiled_mma = tiled_mma
        self.tiled_mma_sfb = tiled_mma_sfb
        self.atom_thr_id = tiled_mma.thr_id
        self.atom_thr_shape = self.atom_thr_id.shape
        self.atom_thr_size = cute.size(self.atom_thr_shape)

        # Compute mma/cluster/tile shapes
        mma_inst_tile_k = 4
        self.mma_tiler_mnk = ( # (tileM256, tileN128, tileK256)
            self.mma_inst_shape_mnk[0],
            self.mma_inst_shape_mnk[1],
            self.mma_inst_shape_mnk[2] * mma_inst_tile_k, # 64 x 4 = 256
        )
        self.mma_tiler_sfb = ( # (CTA_tileM128, tileN128, tileK256)
            self.mma_inst_shape_mnk_sfb[0],
            self.mma_inst_shape_mnk_sfb[1],
            self.mma_inst_shape_mnk_sfb[2] * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = ( # (CTA_tileM256, tileN128, tileK256)
            self.mma_tiler_mnk[0] // self.atom_thr_size,
            self.mma_tiler_mnk[1],
            self.mma_tiler_mnk[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide( # ((2),1,1,1):((1),0,0,0)
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (self.atom_thr_shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide( # ((1),2,1,1):((0),1,0,0)
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2]) # 1, along N dim
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1]) # 1, along M dim
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1]) # 2, along M dim for SFB
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Compute epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape( # (epi_tileM128:1, epi_tileN32:1)
            cta_tile_shape=self.cta_tile_shape_mnk,
            use_2cta_instrs=self.use_2cta_instrs,
            layout_d=self.c_layout,
            elem_ty_d=self.c_dtype,
        )

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler_mnk,
            self.a_dtype,
            self.a_major_mode,
            self.b_dtype,
            self.b_major_mode,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
            debug_print=self.debug_print,
        )

        # Compute A/B/SFA/SFB/C shared memory layout
        # sA: S<3,4,3> o 0 o (MMA=(128,64),MMA_M=1,MMA_K=4,MMA_STAGES=7):((256,1),0,64,32768)
        # sB: S<3,4,3> o 0 o (MMA=(64,64),MMA_N=1,MMA_K=4,MMA_STAGES=7):((256,1),0,64,16384)
        # sSFA: ((Atom_M=((32,4),1), Atom_K=(16,4)),MMA_M=1,MMA_K=4,MMA_STAGES=7):((((16,4),0),(0,1)),0,512,2048)
        # sSFB: (Atom_N=(((32,4),1), Atom_K=(16,4)),MMA_M=1,MMA_K=4,MMA_STAGES=7):((((16,4),0),(0,1)),0,512,2048)
        #   NOTE: sSFB has a double size of the sharded sB, i.e. it is replicated within the CTA-pair, to allow TMA multicasting
        # sC: S<2,4,3> o 0 o (epi_tileM=(8,16), epi_tileN=(32,1), epi_stages=(1,3)):((32,256),(1,0),(0,4096))
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler_mnk,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler_mnk,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler_mnk,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler_mnk,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )

        if const_expr(self.debug_print):
            print()
            print(f"Setup attributes dependent on GEMM inputs:")
            print(f"  MMA tiler (M, N, K): {self.mma_tiler_mnk=}")
            print(f"  MMA inst shape for SFB (M, N, K): {self.mma_inst_shape_mnk_sfb=}")
            print(f"  MMA tiler SFB (M, N, K): {self.mma_tiler_sfb=}")
            print(f"  CTA tile shape (M, N, K): {self.cta_tile_shape_mnk=}")
            print(f"  Cluster layout: {self.cluster_layout_vmnk=}")
            print(f"  Cluster layout SFB: {self.cluster_layout_sfb_vmnk=}")
            print(f"  Number of multicast CTAs for A: {self.num_mcast_ctas_a=}")
            print(f"  Number of multicast CTAs for B: {self.num_mcast_ctas_b=}")
            print(f"  Number of multicast CTAs for SFB: {self.num_mcast_ctas_sfb=}")
            print(f"  Epilogue tile shape: {self.epi_tile=}")
            print(f"  Number of accumulator stages: {self.num_acc_stage=}")
            print(f"  Number of A/B stages: {self.num_ab_stage=}")
            print(f"  Number of C stages: {self.num_c_stage=}")
            print()

            print()
            print(f"A SMEM layout (a_smem_layout_staged) (MMA,MMA_M,MMA_K,STAGE): {self.a_smem_layout_staged}")
            print(f"B SMEM layout (b_smem_layout_staged) (MMA,MMA_N,MMA_K,STAGE): {self.b_smem_layout_staged}")
            print(f"SFA SMEM layout (sfa_smem_layout_staged) (MMA,MMA_M,MMA_K,STAGE): {self.sfa_smem_layout_staged}")
            print(f"SFB SMEM layout (sfb_smem_layout_staged) (MMA,MMA_N,MMA_K,STAGE): {self.sfb_smem_layout_staged}")
            print(f"C SMEM layout (c_smem_layout_staged) (EPI_M,EPI_N,STAGE): {self.c_smem_layout_staged}")
            print()

            print()
            print("tiled_mma: ", tiled_mma, f"\n\nshape_mnk: {tiled_mma.shape_mnk}", f"thr_id.shape: {self.atom_thr_shape}")
            print("tiled_mma_sfb: ", tiled_mma_sfb, f"\n\nshape_mnk: {tiled_mma_sfb.shape_mnk}", f"thr_id.shape: {tiled_mma_sfb.thr_id.shape}")
            print()

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a_tensor: Input tensor A
        :type a_tensor: cute.Tensor
        :param b_tensor: Input tensor B
        :type b_tensor: cute.Tensor
        :param sfa_tensor: Scale factor tensor A
        :type sfa_tensor: cute.Tensor
        :param sfb_tensor: Scale factor tensor B
        :type sfb_tensor: cute.Tensor
        :param c_tensor: Output tensor C
        :type c_tensor: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type # fp4e2m1
        self.b_dtype: Type[cutlass.Numeric] = b_tensor.element_type # fp4e2m1
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type # fp8e8m0
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type # fp16
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M,Rest_M)=((32,4),16),(Atom_K,Rest_K)=((16,4),16),RestL=(1,1)):(((16,4),8192),((0,1),512),(0,131072))
        # 
        # NOTE: from the layout above, we can figure out that:
        #   1. the mimimum block is a (M4,K4):(4,1) row-major block, 
        #       where the K4 will expand 16 times to (SFV16,K4):(0,1), 
        #       and the M4 will repeat 32 times to (Row32,M4):(16,4)
        #       to form a large expanded SF tile atom as (Atom_M(32,4), Atom_K(16,4)):((16,4), (0,1))
        #   2. then the mimimum M/N size for tiled MMA should be 32x4 = 128,
        #       while the mimimum K size for tiled MMA should be 16x4 = 64
        #       and that's probably why our chosen CTA tiled shape for a single mma is (M128, N128, K64)
        #   3. see https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
        #       for more details abouot how scale factors are laid out in memory and accessed by tcgen05.mma instructions
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)

        # ((Atom_N,Rest_N)=((32,4),32),(Atom_K,Rest_K)=((16,4),16),RestL=(1,1)):(((16,4),8192),((0,1),512),(0,262144))
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)

        tiled_mma = self.tiled_mma
        tiled_mma_sfb = self.tiled_mma_sfb
        atom_thr_size = self.atom_thr_size

        # /////////////////////////////////////////////////////////////////////////////
        #  Setup TMA load for A
        # /////////////////////////////////////////////////////////////////////////////
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, self.atom_thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        
        # tma_atom_a: Src: (2,32k):(32k,1) | Dst: (2,32k):(32k,1), where CTA_tileM128 x tileK256 = 32k
        # tma_tensor_a: (pM=2048,pK=1024,1):(1@1,1@0,1@2)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler_mnk,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Setup TMA load for B
        # /////////////////////////////////////////////////////////////////////////////
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, self.atom_thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        
        # tma_atom_b: Src: (2,16k):(16k,1) | Dst: (2,16k):(16k,1), where CTA_tileN64 x tileK256 = 16k
        # tma_tensor_b: (pN=4096, pK=1024,1):(1@1,1@0,1@2)
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor,
            b_smem_layout,
            self.mma_tiler_mnk,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Setup TMA load for SFA
        # /////////////////////////////////////////////////////////////////////////////
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, self.atom_thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        
        # tma_atom_sfa: Src: (2,2048):(2048,1) | Dst: (2,2048):(2048,1), where CTA_tileM128 x tileK256 / SFV16 = 2k
        # tma_tensor_sfa: ((Atom_M, Rest_M)=((32,4),16),(Atom_K, Rest_K)=((16,4),16),RestL=(1,1)):(((8@0,2@0),1@2),((0,1/2@0),1@1),(0,1@3))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler_mnk,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            # NOTE: we use int16 to access the SF factors, 
            # so if it's fp8e8m0, we will have a factorial stride of 1/2
            internal_type=cutlass.Int16,
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Setup TMA load for SFB
        # /////////////////////////////////////////////////////////////////////////////
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, self.atom_thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        
        # tma_atom_sfb: Src: (2,2048):(2048,1) | Dst: (2,2048):(2048,1)
        # tma_tensor_sfb: ((Atom_N, Rest_N)=((32,4),32),(Atom_K, Rest_K)=((16,4),16),RestL=(1,1)):(((8@0,2@0),1@2),((0,1/2@0),1@1),(0,1@3))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            # NOTE: we use int16 to access the SF factors, 
            # so if it's fp8e8m0, we will have a factorial stride of 1/2
            internal_type=cutlass.Int16,
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Setup TMA store for C
        # /////////////////////////////////////////////////////////////////////////////
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        
        # tma_atom_c: Src: (1,4096):(0,1) | Dst: (1,4096):(0,1), where epi_tileM128 x epi_tileN32 = 4096
        # tma_tensor_c: (pM2048, pN4096,1):(1@1,1@0,1@2)
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c_tensor,
            epi_smem_layout,
            self.epi_tile,
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Compute grid size
        # /////////////////////////////////////////////////////////////////////////////
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size
        
        self.tile_sched_params, grid = self._compute_grid( # (CGA_M2, 1, num_persist_clusters=74)
            c_tensor,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
            debug_print=self.debug_print,
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Define shared storage for kernel
        # /////////////////////////////////////////////////////////////////////////////
        @cute.struct
        class SharedStorage:
            # mainloop full/empty mbar array ptrs for each ab stage
            ab_full_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            
            # tmem accumulation full/empty mbar for each acc stage
            acc_full_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            
            # the mbar ptr to synchronize all threads in two CTAs before issuing tmem deallocation
            tmem_dealloc_mbar_ptr: cutlass.Int64
            
            # the smem buffer to hold the allocated tmem address
            tmem_holding_smem_buf: cutlass.Int32
            
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        if const_expr(self.debug_print):
            print()
            print(f"{self.a_dtype=}, {self.b_dtype=}, {self.sf_dtype=}, {self.c_dtype=}, {self.a_major_mode=}, {self.b_major_mode=}, {self.c_layout=}")
            print(f"{a_copy_size=}, {b_copy_size=}, {sfa_copy_size=}, {sfb_copy_size=}, {self.num_tma_load_bytes=}")
            print(f"{self.tile_sched_params=}")
            print()

            print()
            print("TMA A: a_op: ", a_op, "\ntma_atom_a: ", tma_atom_a)
            print()
            print("TMA B: b_op: ", b_op, "\ntma_atom_b: ", tma_atom_b)
            print()
            print("TMA SFA: sfa_op: ", sfa_op, "\ntma_atom_sfa: ", tma_atom_sfa)
            print()
            print("TMA SFB: sfb_op: ", sfb_op, "\ntma_atom_sfb: ", tma_atom_sfb)
            print()
            print("TMA C: tma_atom_c: ", tma_atom_c)
            print()

            cute.printf("")
            cute.printf("sfa_tensor.layout: {}", sfa_layout)
            cute.printf("")
            cute.printf("sfb_tensor.layout: {}", sfb_layout)
            cute.printf("")
            cute.printf("tma_tensor_a: {}", tma_tensor_a)
            cute.printf("")
            cute.printf("tma_tensor_b: {}", tma_tensor_b)
            cute.printf("")
            cute.printf("tma_tensor_sfa: {}", tma_tensor_sfa)
            cute.printf("")
            cute.printf("tma_tensor_sfb: {}", tma_tensor_sfb)
            cute.printf("")
            cute.printf("tma_tensor_c: {}", tma_tensor_c)
            cute.printf("")
            cute.printf("grid: {}", grid)

        # /////////////////////////////////////////////////////////////////////////////
        #  Launch the kernel
        # /////////////////////////////////////////////////////////////////////////////
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = self.use_2cta_instrs
        
        # used only for debug print
        is_print_block = (bidx == 0) and (bidy == 0) and (bidz == 0)  # pick a leader CTA
        is_print_thread = (tidx == 127) and is_print_block

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch tma descriptor
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)

        # /////////////////////////////////////////////////////////////////////////////
        #  Setup cta/thread coordinates
        # /////////////////////////////////////////////////////////////////////////////
        # Coords inside cluster
        
        mma_tile_coord_v = bidx % self.atom_thr_size # CTA idx in the CTA-pair
        is_leader_cta = mma_tile_coord_v == 0 # leader CTA in the CTA-pair
        cta_rank_in_cluster = cute.arch.make_warp_uniform( # CTA idx in the cluster, which might be different from mma_tile_coord_v if cluster size > 2
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord( # CTA (CGA_V2, CGA_M1, CGA_N1, CGA_K1) coord in the cluster
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord( # CTA (CGA_V1, CGA_M2, CGA_N1, CGA_K1) coord in the cluster for SFB multicast
            cta_rank_in_cluster
        )        

        if const_expr(self.debug_print):
            if is_print_thread:
                cute.printf("")
                cute.printf("tidx: {}, warp_idx: {}, block_idx: ({}, {}, {})", tidx, warp_idx, bidx, bidy, bidz)
                cute.printf("mma_tile_coord_v: {}, is_leader_cta: {}, cta_rank_in_cluster: {}", mma_tile_coord_v, is_leader_cta, cta_rank_in_cluster)
                cute.printf("block_in_cluster_coord_vmnk: {}", block_in_cluster_coord_vmnk)
                cute.printf("block_in_cluster_coord_sfb_vmnk: {}", block_in_cluster_coord_sfb_vmnk)

        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        # /////////////////////////////////////////////////////////////////////////////
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_smem_buf = storage.tmem_holding_smem_buf
        ab_full_empty_mbar_ptr = storage.ab_full_empty_mbar_ptr.data_ptr()
        acc_full_empty_mbar_ptr = storage.acc_full_empty_mbar_ptr.data_ptr()

        # Initialize mainloop ab_pipeline (barrier) and states
        num_tma_producer = 1
        ab_pipeline_producer_group = pipeline.CooperativeGroup(
            # full mbar of this CTA will be arrived by this CTA's TMA producer warp once
            # if this CTA is the leader CTA, otherwise no need to arrive since only the mma warp on the leader CTA will wait for it
            pipeline.Agent.Thread, size=num_tma_producer
        )
        num_tma_consumer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            # empty mbar of this CTA will be arrived by all the multicasted CTAs' UMMA consumer warps
            pipeline.Agent.Thread, size=num_tma_consumer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=ab_full_empty_mbar_ptr,
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            # Total sA/sB/sSFA/sSFB size per stage x atom_thr_size,
            # Although only TMA producer warp from the leader CTA will arrive the full mbar and set this tx_count
            # TMA hardware will handle the CTA-pair thing and reduce the transaction count correctly on leader CTA's waiting for the full mbar
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Initialize acc_pipeline (barrier) and states
        num_acc_producer = 1
        acc_pipeline_producer_group = pipeline.CooperativeGroup(
            # only the mma warp in the leader CTA will arrive the acc full mbar
            # using `tcgen05.commit.mbarrier::arrive::one` to auto-multicast to each CTA's epilogue warps
            pipeline.Agent.Thread, size=num_acc_producer
        )
        # all the epilogue warps in the CTA-pair are the acc consumers
        num_acc_consumer = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            # empty mbar of this CTA will be arrived by all epilogue warps in the CTA-pair if this is a leader CTA,
            # where the epilogue warps in non-leader CTA will mapa the acc empty mbar of the corr. leader CTA and arrive it remotely
            pipeline.Agent.Thread, size=num_acc_consumer
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=acc_full_empty_mbar_ptr,
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Tensor memory dealloc barrier init
        if use_2cta_instrs:
            if warp_idx == self.tma_warp_id:
                # NOTE: all the threads in the first epilogue warp of the peer CTA-pair
                # will arrive at self's mbar to tell remote tmem is ready to deallocate
                # and this CTA will do the same to arrive at peer's mbar and then wait for self's mbar to be arrived
                num_tmem_dealloc_threads = 32
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(
                        tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads
                    )
                    
        # Fence all the mbars initialized above before any thread (in the CTA) can access them
        cute.arch.mbarrier_init_fence()

        # Cluster arrive after barrier init
        # for later waiting all CTAs in the cluster to finish all the mbars
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        # /////////////////////////////////////////////////////////////////////////////
        #  Setup smem tensor A/B/SFA/SFB/C
        # /////////////////////////////////////////////////////////////////////////////
        
        # (MMA=(128,64), MMA_M=1, MMA_K=4, STAGE=7) => M-sliced within CTA-pair
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA=(64,64), MMA_N=1, MMA_K=4, STAGE=7) => N-shared within CTA-pair
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        # (MMA=((32,4),1),(16,4)), MMA_M=1, MMA_K=4, STAGE=7) => M-sliced within CTA-pair
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (MMA=((32,4),1),(16,4)), MMA_N=1, MMA_K=4, STAGE=7) => N-replicated within CTA-pair
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        # (EPI_M=(8,16), EPI_N=(32,1), EPI_STAGE=(1,3))
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )

        if const_expr(self.debug_print):
            if is_print_thread:
                # FIXME: when printing tensor for this block-scaled script, 
                # it will stuck at the compiling, but fine with other scripts, need to investigate later
                cute.printf("")
                cute.printf("sA.layout: {}", sA.layout)
                cute.printf("sB.layout: {}", sB.layout)
                cute.printf("sSFA.layout: {}", sSFA.layout)
                cute.printf("sSFB.layout: {}", sSFB.layout)
                cute.printf("sC.layout: {}", sC.layout)
                cute.printf("")

        # /////////////////////////////////////////////////////////////////////////////
        #  Compute multicast mask for A/B/SFA/SFB buffer full
        # /////////////////////////////////////////////////////////////////////////////
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask( # 0b(01), only for itself
                cluster_layout_vmnk, 
                block_in_cluster_coord_vmnk, 
                mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask( # 0b(01), only for itself
                cluster_layout_vmnk, 
                block_in_cluster_coord_vmnk,
                mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask( # 0b(01), only for itself
                cluster_layout_vmnk, 
                block_in_cluster_coord_vmnk, 
                mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask( # 0b(11), multicast to self and peer CTAs
                cluster_layout_sfb_vmnk,
                block_in_cluster_coord_sfb_vmnk, 
                mcast_mode=1
            )

        if const_expr(self.debug_print):
            if is_print_thread:
                cute.printf("")
                cute.printf("a_full_mcast_mask: {}, b_full_mcast_mask: {}", a_full_mcast_mask, b_full_mcast_mask)
                cute.printf("sfa_full_mcast_mask: {}, sfb_full_mcast_mask: {}", sfa_full_mcast_mask, sfb_full_mcast_mask)
                cute.printf("")

        # /////////////////////////////////////////////////////////////////////////////
        #  Local_tile partition global tensors
        # /////////////////////////////////////////////////////////////////////////////
        
        # (tileM=256, tileK=256, restM=8, restK=4, restL=1)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler_mnk, (None, 0, None)), (None, None, None)
        )
        # (tileN=128, tileK=256, restN=32, restK=4, restL=1)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler_mnk, (0, None, None)), (None, None, None)
        )
        # ((SF_atomM,SF_restM)=(32,4,2),(SF_atomK,SF_restK)=(16,4,4), restM=8, restK=4, restL=1)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler_mnk, (None, 0, None)), (None, None, None)
        )
        # ((SF_atomN,SF_restN)=(32,4,1),(SF_atomK,SF_restK)=(16,4,4), restN=32, restK=4, restL=1)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.mma_tiler_mnk, (0, None, None)), (None, None, None)
        )
        # (tileM=256, tileN=128, restM=8, restN=32, restL=1)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler_mnk, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3]) # restK=4 dim for the iterations in the mainloop

        if const_expr(self.debug_print):
            if is_print_thread:
                cute.printf("")
                cute.printf("k_tile_cnt: {}", k_tile_cnt)
                cute.printf("mA_mkl.layout: {}", mA_mkl.layout)
                cute.printf("mB_nkl.layout: {}", mB_nkl.layout)
                cute.printf("mSFA_mkl.layout: {}", mSFA_mkl.layout)
                cute.printf("mSFB_nkl.layout: {}", mSFB_nkl.layout)
                cute.printf("mC_mnl.layout: {}", mC_mnl.layout)
                cute.printf("")
                cute.printf("gA_mkl.layout: {}", gA_mkl.layout)
                cute.printf("gB_nkl.layout: {}", gB_nkl.layout)
                cute.printf("gSFA_mkl.layout: {}", gSFA_mkl.layout)
                cute.printf("gSFB_nkl.layout: {}", gSFB_nkl.layout)
                cute.printf("gC_mnl.layout: {}", gC_mnl.layout)
                cute.printf("")

        # /////////////////////////////////////////////////////////////////////////////
        #  Partition global tensor for TiledMMA_A/B/C
        # /////////////////////////////////////////////////////////////////////////////
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v) # slice with CTA-pair idx
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v) # slice with CTA-pair idx
        
        # (MMA=(128,64), MMA_M=1, MMA_K=4, RestM=8, RestK=4, RestL=1)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA=(64,64), MMA_N=1, MMA_K=4, RestN=32, RestK=4, RestL=1)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (SF_atom=((32,4),(16,4)), SF_restM=1, SF_restK=4, RestM=8, RestK=4, RestL=(1,1))
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (SF_atom=((32,4),(16,4)), SF_restN=1, SF_restK=4, RestN=32, RestK=4, RestL=(1,1))
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA=(128,128), MMA_M=1, MMA_N=1, RestM=8, RestN=32, RestL=1)
        tCgC = thr_mma.partition_C(gC_mnl)

        if const_expr(self.debug_print):
            if is_print_thread:
                cute.printf("")
                cute.printf("tCgA.layout: {}", tCgA.layout)
                cute.printf("tCgB.layout: {}", tCgB.layout)
                cute.printf("tCgSFA.layout: {}", tCgSFA.layout)
                cute.printf("tCgSFB.layout: {}", tCgSFB.layout)
                cute.printf("tCgC.layout: {}", tCgC.layout)
                cute.printf("")

        # /////////////////////////////////////////////////////////////////////////////
        #  Partition global/shared tensor for TMA load A/B/SFA/SFB
        # /////////////////////////////////////////////////////////////////////////////
        
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout( # (1):(0)
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        
        # tAsA: ((TMA_atom_v, rest_v)=(32768,1), PIPE=7)
        # tAgA: ((TMA_atom_v, rest_v)=((256,128),1), RestM=8, RestK=4, RestL=1)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout( # (1):(0)
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        
        # tBsB: ((TMA_atom_v, rest_v)=(16384,1), PIPE=7)
        # tBgB: ((TMA_atom_v, rest_v)=(((256,64),1), RestN=32, RestK=4, RestL=1)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #  TMA load SFA partition_S/D
        sfa_cta_layout = a_cta_layout # (1):(0)
        
        # tAsSFA: ((TMA_atom_v, rest_v)=(2048, SFV=1), PIPE=7)
        # tAgSFA: ((TMA_atom_v, rest_v)=((512,4), SFV=1), RestM=8, RestK=4, RestL=1):(((1/2@0,1@1),0),2@2,4@1,(0,1@3))
        #   where 512 = 32rows x M4 x K4, with SF_atomK4
        #   NOTE: the first mode0 has a stride of 1/2, since we use int16 (internal dtype) to access the SF factors
        #   so each 1/2 step we have a fp8e8m0 SF factor embedded in an int16
        tAsSFA_, tAgSFA_ = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA_) # filter out the zero-strided dims like SFV dim 16
        tAgSFA = cute.filter_zeros(tAgSFA_) # filter out the zero-strided dims like SFV dim 16

        # TMA load SFB partition_S/D
        sfb_cta_layout = cute.make_layout( # (2):(1) => SFB needs to be multicast to 2 CTAs in the CTA-pair
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        
        # tAsSFB: ((TMA_atom_v, rest_v)=(2048, SFV=1), PIPE=7)
        # tAgSFB: ((TMA_atom_v, rest_v)=((512,4), SFV=1), RestN=32, RestK=4, RestL=1):(((1/2@0,1@1),0),1@2,4@1,(0,1@3))
        #   where 512 = 32rows x M4 x K4, with SF_atomK4
        #   NOTE: the first mode0 has a stride of 1/2, since we use int16 (internal dtype) to access the SF factors
        #   so each 1/2 step we have a fp8e8m0 SF factor embedded in an int16
        tBsSFB_, tBgSFB_ = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB_) # filter out the zero-strided dims like SFV dim 16
        tBgSFB = cute.filter_zeros(tBgSFB_) # filter out the zero-strided dims like SFV dim 16

        if const_expr(self.debug_print):
            if is_print_thread:
                cute.printf("")
                cute.printf("a_cta_layout: {}", a_cta_layout)
                cute.printf("tAsA.layout: {}", tAsA.layout)
                cute.printf("tAgA.layout: {}", tAgA.layout)
                cute.printf("")
                cute.printf("b_cta_layout: {}", b_cta_layout)
                cute.printf("tBsB.layout: {}", tBsB.layout)
                cute.printf("tBgB.layout: {}", tBgB.layout)
                cute.printf("")
                cute.printf("sfa_cta_layout: {}", sfa_cta_layout)
                cute.printf("tAsSFA_.layout: {}", tAsSFA_.layout)
                cute.printf("tAsSFA.layout: {}", tAsSFA.layout)
                cute.printf("tAgSFA_.layout: {}", tAgSFA_.layout)
                cute.printf("tAgSFA.layout: {}", tAgSFA.layout)
                cute.printf("")
                cute.printf("sfb_cta_layout: {}", sfb_cta_layout)
                cute.printf("tBsSFB_.layout: {}", tBsSFB_.layout)
                cute.printf("tBsSFB.layout: {}", tBsSFB.layout)
                cute.printf("tBgSFB_.layout: {}", tBgSFB_.layout)
                cute.printf("tBgSFB.layout: {}", tBgSFB.layout)
                cute.printf("")

        # /////////////////////////////////////////////////////////////////////////////
        #  Partition shared/tensor memory tensor for TiledMMA_A/B/C
        # /////////////////////////////////////////////////////////////////////////////
        
        # (MMA=1, MMA_M=1, MMA_K=4, STAGE=7):(0,0,2,1024)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA=1, MMA_N=1, MMA_K=4, STAGE=7):(0,0,2,512)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA=(128,128), MMA_M=1, MMA_N=1, ACC_STAGE=2):((65536,1),0,0,128)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler_mnk[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        if const_expr(self.debug_print):
            if is_print_thread:
                cute.printf("")
                cute.printf("tCrA.layout: {}", tCrA.layout)
                cute.printf("tCrB.layout: {}", tCrB.layout)
                cute.printf("tCtAcc_fake.layout: {}", tCtAcc_fake.layout)
                cute.printf("")

        # /////////////////////////////////////////////////////////////////////////////
        #  Create static persistent tile scheduler
        # /////////////////////////////////////////////////////////////////////////////
        
        tile_sched = utils.StaticPersistentTileScheduler.create(
            params=tile_sched_params,
            block_idx=cute.arch.block_idx(),
            grid_dim=cute.arch.grid_dim()
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Cluster wait before tensor memory alloc
        # /////////////////////////////////////////////////////////////////////////////
        if cute.size(self.cluster_shape_mn) > 1:
            # wait all CTAs in the cluster to finish all the mbars
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier(
                barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta
            )

        # /////////////////////////////////////////////////////////////////////////////
        #  Specialized TMA load producer
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.tma_warp_id:
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            # /////////////////////////////////////////////////////////////////////////////
            #  Persistent tile scheduling loop
            # /////////////////////////////////////////////////////////////////////////////
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx # block coord
                mma_tile_coord_mnl = ( # cluster coord
                    cur_tile_coord[0] // self.atom_thr_size,
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # /////////////////////////////////////////////////////////////////////////////
                #  Slice to per mma tile index
                # /////////////////////////////////////////////////////////////////////////////
                
                # ((atom_v, rest_v)=((256,128),1), RestK4)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2]) # slice RestM8 and RestL1 idx
                ]
                # ((atom_v, rest_v)=((256,64),1), RestK4)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2]) # slice RestN32 and RestL1 idx
                ]
                # ((atom_v, rest_v)=((512,4),1), Rest4)
                tAgSFA_slice = tAgSFA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2]) # slice RestM8 and RestL1 idx
                ]
                # ((atom_v, rest_v)=((512,4),1), RestK4)
                tBgSFB_slice = tBgSFB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2]) # slice RestN32 and RestL1 idx
                ]

                if const_expr(self.debug_print):
                    is_first_work_tile = (cur_tile_coord[0] == 0) and (cur_tile_coord[1] == 0) and (cur_tile_coord[2] == 0)
                    if (tidx == 32 * self.tma_warp_id) and is_print_block and is_first_work_tile:
                        cute.printf("")
                        cute.printf("[TMA warp] mma_tile_coord_mnl: ({}, {}, {})", mma_tile_coord_mnl[0], mma_tile_coord_mnl[1], mma_tile_coord_mnl[2])
                        cute.printf("[TMA warp] tAgA_slice.layout: {}", tAgA_slice.layout)
                        cute.printf("[TMA warp] tBgB_slice.layout: {}", tBgB_slice.layout)
                        cute.printf("[TMA warp] tAgSFA_slice.layout: {}", tAgSFA_slice.layout)
                        cute.printf("[TMA warp] tBgSFB_slice.layout: {}", tBgSFB_slice.layout)
                        cute.printf("")

                # Peek for the first ab empty mbar to be arrived by the consumer w/o blocking
                ab_producer_state.reset_count() # NOTE: persistent kernel needs to reset count for each tile
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )
                
                # /////////////////////////////////////////////////////////////////////////////
                #  Tma load loop
                # /////////////////////////////////////////////////////////////////////////////
                for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                    # Wait for current ab empty mbar to be arrived by the consumer
                    # and then arrive the ab full mbar to notify the consumer the data is ready after the TMA load
                    # NOTE: it is only arrived by the leader CTA (inside logics), since only leader CTA waits the ab empty mbar,
                    # and accordingly, only the leader CTA's TMA producer warp needs to arrive the ab full mbar as well
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )

                    # TMA load A/B/SFA/SFB
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, ab_producer_state.count)],
                        tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, ab_producer_state.count)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    # Peek for the next ab empty mbar to be arrived by the consumer w/o blocking
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Wait for the last ab empty mbar to avoid dangling signals
            ab_pipeline.producer_tail(ab_producer_state)

        # /////////////////////////////////////////////////////////////////////////////
        #  Specialized MMA consumer / ACC producer
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id: # umma consumer warp / epilogue acc producer warp, if on the leader CTA
            # Bar sync for retrieve tensor memory ptr from shared mem
            # NOTE: both umma consumer warp and epilogue warps need to sync here before retriving the tmem ptr,
            # to ensure its visibility, allocated by the first epilogue warp and stored in shared mem
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=self.tmem_ptr_read_threads,
            )

            # /////////////////////////////////////////////////////////////////////////////
            #  Retrieving tensor memory ptr and make tensor for ACC/SFA/SFB
            # /////////////////////////////////////////////////////////////////////////////

            # Make accumulator tmem tensor
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_smem_buf,
            )
            # (MMA=(128,128), MMA_M=1, MMA_N=1, ACC_STAGE=2):((65536,1),0,0,128)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base), # offset tmem after the accumulator tensor
                dtype=self.sf_dtype,
            )
            # ((((TMEM_lanes32, M4), RestK4),(SFV16, K4)), RestM1, MK4):((((262144,4),8388608),(0,1)),0,16)
            #   where MK4 means 4 (M4,K4) blocks share the same tmem col,
            #   and 262144 = 65536 x 4B/sizeof(fp8) = 65536 x 4, is the tmem row stride for fp8 elems
            #   and 8388608 = 262144 X TMEM_lanes32, is the stride for next k tile in the main loop along the RestK4 dim
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler_mnk,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA), # offset tmem after the SFA tensor
                dtype=self.sf_dtype,
            )
            # ((((TMEM_lanes32, M4), RestK4),(SFV16, K4)), RestM1, MK4):((((262144,4),8388608),(0,1)),0,16)
            #   where MK4 means 4 (M4,K4) blocks share the same tmem col,
            #   and 262144 = 65536 x 4B/sizeof(fp8) = 65536 x 4, is the tmem row stride for fp8 elems
            #   and 8388608 = 262144 X TMEM_lanes32, is the stride for next k tile in the main loop along the RestK4 dim
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler_mnk,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            if const_expr(self.debug_print):
                if (tidx == 32 * self.mma_warp_id) and is_print_block:
                    cute.printf("")
                    cute.printf("[MMA warp] tCtAcc_base.layout: {}", tCtAcc_base.layout)
                    cute.printf("[MMA warp] tCtSFA.layout: {}", tCtSFA.layout)
                    cute.printf("[MMA warp] tCtSFB.layout: {}", tCtSFB.layout)
                    cute.printf("")

            # /////////////////////////////////////////////////////////////////////////////
            #  Partition for S2T copy of SFA/SFB
            # /////////////////////////////////////////////////////////////////////////////
            tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            )
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
            )

            # /////////////////////////////////////////////////////////////////////////////
            #  Persistent tile scheduling loop
            # /////////////////////////////////////////////////////////////////////////////
            work_tile = tile_sched.initial_work_tile_info()

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // self.atom_thr_size,
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                # if const_expr(self.debug_print):
                #     is_first_work_tile = (cur_tile_coord[0] == 0) and (cur_tile_coord[1] == 0) and (cur_tile_coord[2] == 0)
                #     if (tidx == 32 * self.mma_warp_id) and is_print_block and is_first_work_tile:
                #         cute.printf("")
                #         cute.printf("[MMA warp] mma_tile_coord_mnl: ({}, {}, {})", mma_tile_coord_mnl[0], mma_tile_coord_mnl[1], mma_tile_coord_mnl[2])
                #         cute.printf("[MMA warp] tCtAcc: {}", tCtAcc)
                #         cute.printf("")

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                # /////////////////////////////////////////////////////////////////////////////
                #  Wait for accumulator buffer empty
                # /////////////////////////////////////////////////////////////////////////////
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                # /////////////////////////////////////////////////////////////////////////////
                #  Reset the ACCUMULATE field for each tile
                # /////////////////////////////////////////////////////////////////////////////
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                # /////////////////////////////////////////////////////////////////////////////
                #  Mma mainloop
                # /////////////////////////////////////////////////////////////////////////////
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                        #  Copy SFA/SFB from smem to tmem
                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
                        )

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )

                            # Set SFA/SFB tensor to tiled_mma
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kblock_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB[sf_kblock_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )

                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                # /////////////////////////////////////////////////////////////////////////////
                #  Async arrive accumulator buffer full
                # /////////////////////////////////////////////////////////////////////////////
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                # /////////////////////////////////////////////////////////////////////////////
                #  Advance to next tile
                # /////////////////////////////////////////////////////////////////////////////
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # /////////////////////////////////////////////////////////////////////////////
            #  Wait for accumulator buffer empty
            # /////////////////////////////////////////////////////////////////////////////
            acc_pipeline.producer_tail(acc_producer_state)
        
        # /////////////////////////////////////////////////////////////////////////////
        #  Specialized epilogue consumer
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.mma_warp_id:
            # /////////////////////////////////////////////////////////////////////////////
            #  Alloc tensor memory buffer
            # /////////////////////////////////////////////////////////////////////////////
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols,
                    tmem_holding_smem_buf,
                    is_two_cta=use_2cta_instrs,
                )

            # /////////////////////////////////////////////////////////////////////////////
            #  Bar sync for retrieve tensor memory ptr from shared memory
            # /////////////////////////////////////////////////////////////////////////////
            tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            # /////////////////////////////////////////////////////////////////////////////
            #  Retrieving tensor memory ptr and make accumulator tensor
            # /////////////////////////////////////////////////////////////////////////////
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_smem_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # if const_expr(self.debug_print):
            #     if (tidx == 0) and is_print_block:
            #         cute.printf("")
            #         cute.printf("[Epilog warp] tCtAcc_base: {}", tCtAcc_base)
            #         cute.printf("")

            # /////////////////////////////////////////////////////////////////////////////
            #  Partition for epilogue
            # /////////////////////////////////////////////////////////////////////////////
            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
                self.epilog_tmem_copy_and_partition(
                    epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
                )
            )

            tTR_rC = cute.make_fragment(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            tma_atom_c, bSG_sC, bSG_gC_partitioned = (
                self.epilog_gmem_copy_and_partition(
                    epi_tidx, tma_atom_c, tCgC, epi_tile, sC
                )
            )

            # if const_expr(self.debug_print):
            #     if (tidx == 0) and is_print_block:
            #         cute.printf("")
            #         cute.printf("[Epilog warp] tiled_copy_t2r: layout_src_tv: {} | layout_src_tv_tiled: {} | layout_dst_tv: {} | layout_dst_tv_tiled: {}", tiled_copy_t2r.layout_src_tv, tiled_copy_t2r.layout_src_tv_tiled, tiled_copy_t2r.layout_dst_tv, tiled_copy_t2r.layout_dst_tv_tiled)
            #         cute.printf("[Epilog warp] tTR_tAcc_base: {}", tTR_tAcc_base)
            #         cute.printf("[Epilog warp] tTR_rAcc: {}", tTR_rAcc)
            #         cute.printf("[Epilog warp] tiled_copy_r2s: layout_src_tv: {} | layout_src_tv_tiled: {} | layout_dst_tv: {} | layout_dst_tv_tiled: {}", tiled_copy_r2s.layout_src_tv, tiled_copy_r2s.layout_src_tv_tiled, tiled_copy_r2s.layout_dst_tv, tiled_copy_r2s.layout_dst_tv_tiled)
            #         cute.printf("[Epilog warp] tTR_rC: {}", tTR_rC)
            #         cute.printf("[Epilog warp] tRS_rC: {}", tRS_rC)
            #         cute.printf("[Epilog warp] tRS_sC: {}", tRS_sC)
            #         cute.printf("[Epilog warp] bSG_sC: {}", bSG_sC)
            #         cute.printf("[Epilog warp] bSG_gC_partitioned: {}", bSG_gC_partitioned)
            #         cute.printf("")

            # /////////////////////////////////////////////////////////////////////////////
            #  Persistent tile scheduling loop
            # /////////////////////////////////////////////////////////////////////////////
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            while work_tile.is_valid_tile:

                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // self.atom_thr_size,
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # /////////////////////////////////////////////////////////////////////////////
                #  Slice to per mma tile index
                # /////////////////////////////////////////////////////////////////////////////
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        *mma_tile_coord_mnl,
                    )
                ]

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_consumer_state.index)
                ]

                # /////////////////////////////////////////////////////////////////////////////
                #  Wait for accumulator buffer full
                # /////////////////////////////////////////////////////////////////////////////
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                # /////////////////////////////////////////////////////////////////////////////
                #  Store accumulator to global memory in subtiles
                # /////////////////////////////////////////////////////////////////////////////
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt

                # if const_expr(self.debug_print):
                #     is_first_work_tile = (cur_tile_coord[0] == 0) and (cur_tile_coord[1] == 0) and (cur_tile_coord[2] == 0)
                #     if (tidx == 0) and is_print_block and is_first_work_tile:
                #         cute.printf("")
                #         cute.printf("[Epilog warp] mma_tile_coord_mnl: ({}, {}, {})", mma_tile_coord_mnl[0], mma_tile_coord_mnl[1], mma_tile_coord_mnl[2])
                #         cute.printf("[Epilog warp] tTR_tAcc (post-group): {}", tTR_tAcc)
                #         cute.printf("[Epilog warp] subtile_cnt: {}", subtile_cnt)
                #         cute.printf("[Epilog warp] num_prev_subtiles: {}", num_prev_subtiles)
                #         cute.printf("[Epilog warp] bSG_gC (post-group): {}", bSG_gC)
                #         cute.printf("")

                for subtile_idx in cutlass.range(subtile_cnt):
                    # Load accumulator from tensor memory buffer to register
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    # Convert to C type
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                    tRS_rC.store(acc_vec)

                    # Store C to shared memory
                    c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, c_buffer)],
                    )
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    epilog_threads = 32 * len(self.epilog_warp_id)
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )

                    # TMA store C to global memory
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, c_buffer)],
                            bSG_gC[(None, subtile_idx)],
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )

                # /////////////////////////////////////////////////////////////////////////////
                #  Async arrive accumulator buffer empty
                # /////////////////////////////////////////////////////////////////////////////
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                # /////////////////////////////////////////////////////////////////////////////
                #  Advance to next tile
                # /////////////////////////////////////////////////////////////////////////////
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # /////////////////////////////////////////////////////////////////////////////
            #  Dealloc the tensor memory buffer
            # /////////////////////////////////////////////////////////////////////////////
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            epilog_threads = 32 * len(self.epilog_warp_id)
            cute.arch.barrier(
                barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads
            )
            if warp_idx == self.epilog_warp_id[0]:
                if use_2cta_instrs:
                    cute.arch.mbarrier_arrive(
                        tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1
                    )
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(
                    acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs
                )
            # /////////////////////////////////////////////////////////////////////////////
            #  Wait for C store complete
            # /////////////////////////////////////////////////////////////////////////////
            c_pipeline.producer_tail()

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_fragment(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rC: The partitioned tensor C (register source)
            - tRS_sC: The partitioned tensor C (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to:
        partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param atom: The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for none TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing (tma_atom_c, bSG_sC, bSG_gC) where:
            - tma_atom_c: The TMA copy atom
            - bSG_sC: The partitioned shared memory tensor C
            - bSG_gC: The partitioned global tensor C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )

        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        a_major_mode: tcgen05.OperandMajorMode,
        b_dtype: Type[cutlass.Numeric],
        b_major_mode: tcgen05.OperandMajorMode,
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
        debug_print: bool = False,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param a_major_mode: Major mode of operand A.
        :type a_major_mode: tcgen05.OperandMajorMode
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param b_major_mode: Major mode of operand B.
        :type b_major_mode: tcgen05.OperandMajorMode
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum of operand C.
        :type c_layout: utils.LayoutEnum
        :param sf_dtype: Data type of Scale factor.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Scale factor vector size.
        :type sf_vec_size: int
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # ACC stages
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        # Default C stages
        num_c_stage = 2

        # Calculate smem layout and size for one stage of A, B, SFA, SFB and C
        # a_smem_layout_stage_one:  S<3,4,3> o 0 o (MMA=(128,64),MMA_M=1,MMA_K=4,1):((256,1),0,64,0)
        # b_smem_layout_staged_one:  S<3,4,3> o 0 o (MMA=(64,64),MMA_M=1,MMA_K=4,1):((256,1),0,64,0)
        # sfa_smem_layout_staged_one:  (MMA=(((32,4),1),(16,4)),MMA_M=1,MMA_K=4,1):((((16,4),0),(0,1)),0,512,2048)
        # sfb_smem_layout_staged_one:  (MMA=(((32,4),1),(16,4)),MMA_M=1,MMA_K=4,1):((((16,4),0),(0,1)),0,512,2048)
        # c_smem_layout_staged_one:  S<2,4,3> o 0 o (epi_tileM=(8,16),epi_tileN=(32,1),(1,1)):((32,256),(1,0),(0,0))
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            num_stages=1,
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            num_stages=1,
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            num_stages=1,
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            num_stages=1,
        )

        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            epi_stage=1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        
        mbar_helpers_bytes = 1024
        
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B/SFA/SFB stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B/SFA/SFB stage
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B/SFA/SFB stages and reserved bytes
        # Add remaining unused smem to epilogue
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)

        if const_expr(debug_print):
            print()
            print("a_smem_layout_stage_one: ", a_smem_layout_stage_one)
            print("b_smem_layout_staged_one: ", b_smem_layout_staged_one)
            print("sfa_smem_layout_staged_one: ", sfa_smem_layout_staged_one)
            print("sfb_smem_layout_staged_one: ", sfb_smem_layout_staged_one)
            print("c_smem_layout_staged_one: ", c_smem_layout_staged_one)
            print(f"Bytes per A/B/SFA/SFB stage: {ab_bytes_per_stage=}")
            print(f"Bytes per C stage: {c_bytes_per_stage=}")
            print(f"Reserved bytes for mbar helpers: {mbar_helpers_bytes=}")
            print(
                f"Computed stages - AB stages: {num_ab_stage}, ACC stages: {num_acc_stage}, C stages: {num_c_stage}"
            )
            print()

        return num_acc_stage, num_ab_stage, num_c_stage

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
        debug_print: bool = False,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: Tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0)) # (CTA_tileM128, CTA_tileN128)
        gc = cute.zipped_divide(c, tiler=c_shape) # ((CTA_tileM128, CTA_tileN128), (RestM16, RestN32, RestL1))
        num_ctas_mnl = gc[(0, (None, None, None))].shape # (RestM16, RestN32, RestL1)
        cluster_shape_mnl = (*cluster_shape_mn, 1) # (2, 1, 1), then we have 8x32x1=256 clusters to activate

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape( # (CTA_M=2, CTA_N=1, num_persistent_clusters=74)
            params=tile_sched_params,
            # max_active_clusters = 74 = 148 // 2, 
            # since cluster_shape_mn is (2, 1), and we have 148 SMs in total
            max_active_clusters=max_active_clusters
        )
        
        if const_expr(debug_print):
            cute.printf("")
            cute.printf("zipped divided gc shape: {}", gc.shape)
            cute.printf("Number of CTAs in MNL: {}", num_ctas_mnl)
            cute.printf(
                "Computed tile scheduler parameters: "
                "problem_layout_ncluster_mnl: {}, "
                "problem_shape_ntile_mnl: {}",
                tile_sched_params.problem_layout_ncluster_mnl, # (CM=8, CN=32, CL=1):(1,8,256), col-major
                tile_sched_params.problem_shape_ntile_mnl, # (RestM16, RestN32, RestL1)
            )
            cute.printf("Computed grid shape for kernel launch: {}", grid)
            cute.printf("")

        return tile_sched_params, grid

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes and sf_vec_size are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes and sf_vec_size are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        # Check valid ab_dtype
        if ab_dtype not in {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        # Check valid sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # Check valid sf_dtype
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_dtype and sf_vec_size combinations
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            is_valid = False

        # Check valid c_dtype
        if c_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_layouts(
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the dtypes and sf_vec_size are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major dimension of the A tensor
        :type a_major: str
        :param b_major: The major dimension of the B tensor
        :type b_major: str
        :param c_major: The major dimension of the C tensor
        :type c_major: str

        :return: True if the layouts are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """
        Check if the mma tiler and cluster shape are valid

        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]

        :return: True if the mma tiler and cluster shape are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        # Skip invalid mma tile shape
        if not mma_tiler_mn[0] in [128, 256]:
            is_valid = False
        if not mma_tiler_mn[1] in [128, 256]:
            is_valid = False
        # Skip illegal cluster shape
        if cluster_shape_mn[0] % (2 if mma_tiler_mn[0] == 256 else 1) != 0:
            is_valid = False
        # Skip invalid cluster shape
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            # Special cluster shape check for scale factor multicasts.
            # Due to limited size of scale factors, we can't multicast among more than 4 CTAs.
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or not is_power_of_2(cluster_shape_mn[0])
            or not is_power_of_2(cluster_shape_mn[1])
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the gemm can be implemented

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor tensor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]
        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """
        can_implement = True
        # Skip unsupported types
        if not BlockScaledDenseGemmPersistentKernelSm100.is_valid_dtypes_and_scale_factor_vec_size(
            ab_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            can_implement = False
        # Skip unsupported layouts
        if not BlockScaledDenseGemmPersistentKernelSm100.is_valid_layouts(
            ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        # Skip invalid mma tile shape and cluster shape
        if not BlockScaledDenseGemmPersistentKernelSm100.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler_mn, cluster_shape_mn
        ):
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not BlockScaledDenseGemmPersistentKernelSm100.is_valid_tensor_alignment(
            m, n, k, l, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        return can_implement


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
    # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]


def run(
    mnkl: Tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    c_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float = 1e-01,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    **kwargs,
):
    """Execute a persistent batched dense blockscaled GEMM operation on Blackwell architecture with performance benchmarking.

    This function prepares input tensors, configures and launches the persistent GEMM kernel,
    optionally performs reference validation, and benchmarks the execution performance.

    :param mnkl: Problem size (M, N, K, L)
    :type mnkl: Tuple[int, int, int, int]
    :param ab_dtype: Data type for input tensors A and B
    :type ab_dtype: Type[cutlass.Numeric]
    :param sf_dtype: Data type for scale factor tensor
    :type sf_dtype: Type[cutlass.Numeric]
    :param sf_vec_size: Vector size for scale factor tensor
    :type sf_vec_size: int
    :param c_dtype: Data type for output tensor C
    :type c_dtype: Type[cutlass.Numeric]
    :param a_major/b_major/c_major: Memory layout of tensor A/B/C
    :type a_major/b_major/c_major: str
    :param mma_tiler_mn: MMA tiling size.
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster shape.
    :type cluster_shape_mn: Tuple[int, int]
    :param tolerance: Tolerance value for reference validation comparison, defaults to 1e-01
    :type tolerance: float, optional
    :param warmup_iterations: Number of warmup iterations before benchmarking, defaults to 0
    :type warmup_iterations: int, optional
    :param iterations: Number of benchmark iterations to run, defaults to 1
    :type iterations: int, optional
    :param skip_ref_check: Whether to skip reference result validation, defaults to False
    :type skip_ref_check: bool, optional
    :param use_cold_l2: Whether to use circular buffer strategy to ensure cold L2 cache, defaults to False
    :type use_cold_l2: bool, optional
    :raises RuntimeError: If CUDA GPU is not available
    :raises ValueError: If the configuration is invalid or unsupported by the kernel
    :return: Execution time of the GEMM kernel
    :rtype: float
    """
    print(f"Running Sm100 Persistent Dense BlockScaled GEMM test with:")
    print(f"mnkl: {mnkl}")
    print(f"AB dtype: {ab_dtype}, SF dtype: {sf_dtype}, SF Vec size: {sf_vec_size}")
    print(f"C dtype: {c_dtype}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {'True' if use_cold_l2 else 'False'}")

    # Unpack parameters
    m, n, k, l = mnkl

    # Skip unsupported testcase
    if not BlockScaledDenseGemmPersistentKernelSm100.can_implement(
        ab_dtype,
        sf_dtype,
        sf_vec_size,
        c_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        m,
        n,
        k,
        l,
        a_major,
        b_major,
        c_major,
    ):
        raise TypeError(
            f"Unsupported testcase {ab_dtype}, {sf_dtype}, {sf_vec_size}, {c_dtype},  {mma_tiler_mn}, {cluster_shape_mn}, {m}, {n}, {k}, {l}, {a_major}, {b_major}, {c_major}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    # Create tensor A/B/C
    a_ref = cutlass_torch.matrix(l, m, k, a_major == "m", cutlass.Float32)
    b_ref = cutlass_torch.matrix(l, n, k, b_major == "n", cutlass.Float32)
    c_ref = cutlass_torch.matrix(l, m, n, c_major == "m", cutlass.Float32)

    a_tensor, a_torch = cutlass_torch.cute_tensor_like(
        a_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, b_torch = cutlass_torch.cute_tensor_like(
        b_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    c_tensor, c_torch = cutlass_torch.cute_tensor_like(
        c_ref, c_dtype, is_dynamic_layout=True, assumed_align=16
    )
    
    divisibility = 2 if ab_dtype == cutlass.Float4E2M1FN else 1

    # Mark tensor to be byte aligned
    a_tensor.mark_compact_shape_dynamic(
        mode=1 if a_major == "k" else 0,
        stride_order=(2, 0, 1) if a_major == "k" else (2, 1, 0),
        divisibility=divisibility,
    )
    b_tensor.mark_compact_shape_dynamic(
        mode=1 if b_major == "k" else 0,
        stride_order=(2, 0, 1) if b_major == "k" else (2, 1, 0),
        divisibility=divisibility,
    )
    c_tensor.mark_compact_shape_dynamic(
        mode=1 if c_major == "n" else 0,
        stride_order=(2, 0, 1) if c_major == "n" else (2, 1, 0),
        divisibility=2 if c_dtype == cutlass.Float4E2M1FN else 1,
    )

    # Create scale factor tensor SFA/SFB
    def create_scale_factor_tensor(l, mn, k, sf_vec_size, dtype):
        def ceil_div(a, b):
            return (a + b - 1) // b

        sf_k = ceil_div(k, sf_vec_size)
        ref_shape = (l, mn, sf_k)

        atom_m = (32, 4)
        atom_k = 4
        mma_shape = (
            l,
            ceil_div(mn, atom_m[0] * atom_m[1]),
            ceil_div(sf_k, atom_k),
            atom_m[0],
            atom_m[1],
            atom_k,
        )

        ref_permute_order = (1, 2, 0)
        mma_permute_order = (3, 4, 1, 5, 2, 0)

        # Create f32 ref torch tensor (cpu)
        ref_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
            ref_shape,
            torch.float32,
            permute_order=ref_permute_order,
            init_type=cutlass_torch.TensorInitType.RANDOM,
            init_config=cutlass_torch.RandomInitConfig(
                min_val=1,
                max_val=3,
            ),
        )

        # Create f32 cute torch tensor (cpu)
        cute_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
            mma_shape,
            torch.float32,
            permute_order=mma_permute_order,
            init_type=cutlass_torch.TensorInitType.RANDOM,
            init_config=cutlass_torch.RandomInitConfig(
                min_val=0,
                max_val=1,
            ),
        )

        # convert ref f32 tensor to cute f32 tensor
        cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
            from_dlpack(ref_f32_torch_tensor_cpu),
            from_dlpack(cute_f32_torch_tensor_cpu),
        )
        cute_f32_torch_tensor = cute_f32_torch_tensor_cpu.cuda()

        # reshape makes memory contiguous
        ref_f32_torch_tensor_cpu = (
            ref_f32_torch_tensor_cpu.permute(2, 0, 1)
            .unsqueeze(-1)
            .expand(l, mn, sf_k, sf_vec_size)
            .reshape(l, mn, sf_k * sf_vec_size)
            .permute(*ref_permute_order)
        )
        # prune to mkl for reference check.
        ref_f32_torch_tensor_cpu = ref_f32_torch_tensor_cpu[:, :k, :]

        # Create dtype cute torch tensor (cpu)
        cute_tensor, cute_torch_tensor = cutlass_torch.cute_tensor_like(
            cute_f32_torch_tensor_cpu,
            dtype,
            is_dynamic_layout=True,
            assumed_align=16,
        )

        # Convert f32 cute tensor to dtype cute tensor
        cute_tensor = cutlass_torch.convert_cute_tensor(
            cute_f32_torch_tensor,
            cute_tensor,
            dtype,
            is_dynamic_layout=True,
        )
        return ref_f32_torch_tensor_cpu, cute_tensor, cute_torch_tensor

    sfa_ref, sfa_tensor, sfa_torch = create_scale_factor_tensor(
        l, m, k, sf_vec_size, sf_dtype
    )
    sfb_ref, sfb_tensor, sfb_torch = create_scale_factor_tensor(
        l, n, k, sf_vec_size, sf_dtype
    )

    # Configure gemm kernel
    gemm = BlockScaledDenseGemmPersistentKernelSm100(
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
        debug_print=DEBUG_MODE
    )

    # Compute max active clusters on current device
    hardware_info = cutlass.utils.HardwareInfo()
    max_sms = hardware_info.get_device_multiprocessor_count()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    print(f"Max active clusters: {max_active_clusters} for cluster shape {cluster_shape_mn} on device with {max_sms} SMs")

    # Initialize Stream
    current_stream = cutlass_torch.default_stream()

    # Compile gemm kernel
    compiled_gemm = cute.compile(
        gemm,
        a_tensor,
        b_tensor,
        sfa_tensor,
        sfb_tensor,
        c_tensor,
        max_active_clusters,
        current_stream,
    )

    # Compute reference result
    if not skip_ref_check:
        # Execute kernel once for reference checking
        compiled_gemm(
            a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor, current_stream
        )
        print("Verifying results...")
        res_a = torch.einsum("mkl,mkl->mkl", a_ref, sfa_ref)
        res_b = torch.einsum("nkl,nkl->nkl", b_ref, sfb_ref)
        ref = torch.einsum("mkl,nkl->mnl", res_a, res_b)

        # Convert c back to f32 for comparison.
        c_ref_device = c_ref.cuda()
        cute.testing.convert(
            c_tensor,
            from_dlpack(c_ref_device, assumed_align=16).mark_layout_dynamic(
                leading_dim=(1 if c_major == "n" else 0)
            ),
        )
        c_ref = c_ref_device.cpu()

        if c_dtype in (cutlass.Float32, cutlass.Float16, cutlass.BFloat16):
            torch.testing.assert_close(c_ref, ref, atol=tolerance, rtol=1e-02)
        elif c_dtype in (cutlass.Float8E5M2, cutlass.Float8E4M3FN):
            # Convert ref : f32 -> f8 -> f32
            ref_f8_ = torch.empty(*(l, m, n), dtype=torch.uint8, device="cuda").permute(
                1, 2, 0
            )
            ref_f8 = from_dlpack(ref_f8_, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            )
            ref_f8.element_type = c_dtype
            ref_device = ref.permute(2, 0, 1).contiguous().permute(1, 2, 0).cuda()
            ref_tensor = from_dlpack(ref_device, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            )
            cute.testing.convert(ref_tensor, ref_f8)
            cute.testing.convert(ref_f8, ref_tensor)
            ref = ref_device.cpu()
            torch.testing.assert_close(c_ref, ref, atol=tolerance, rtol=1e-02)
    def generate_tensors():
        a_tensor, _ = cutlass_torch.cute_tensor_like(
            a_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        b_tensor, _ = cutlass_torch.cute_tensor_like(
            b_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        c_tensor, _ = cutlass_torch.cute_tensor_like(
            c_ref, c_dtype, is_dynamic_layout=True, assumed_align=16
        )

        # Mark tensor to be byte aligned
        a_tensor.mark_compact_shape_dynamic(
            mode=1 if a_major == "k" else 0,
            stride_order=(2, 0, 1) if a_major == "k" else (2, 1, 0),
            divisibility=divisibility,
        )
        b_tensor.mark_compact_shape_dynamic(
            mode=1 if b_major == "k" else 0,
            stride_order=(2, 0, 1) if b_major == "k" else (2, 1, 0),
            divisibility=divisibility,
        )
        c_tensor.mark_compact_shape_dynamic(
            mode=1 if c_major == "n" else 0,
            stride_order=(2, 0, 1) if c_major == "n" else (2, 1, 0),
            divisibility=2 if c_dtype == cutlass.Float4E2M1FN else 1,
        )

        _, sfa_tensor, _ = create_scale_factor_tensor(l, m, k, sf_vec_size, sf_dtype)
        _, sfb_tensor, _ = create_scale_factor_tensor(l, n, k, sf_vec_size, sf_dtype)
        return cute.testing.JitArguments(
            a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor, current_stream
        )

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            a_torch.numel() * a_torch.element_size()
            + b_torch.numel() * b_torch.element_size()
            + sfa_torch.numel() * sfa_torch.element_size()
            + sfb_torch.numel() * sfb_torch.element_size()
            + c_torch.numel() * c_torch.element_size()
        )
        workspace_count = cute.testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = cute.testing.benchmark(
        compiled_gemm,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    # Profiling
    if PROFILE_MODE:
        import sys
        sys.path.insert(0, "..")
        from nvtx import switch_profile, add_nvtx_event

        flops = 2 * m * n * k * l * divisibility
        event_str = f"dense blockscaled gemm persistent ({mnkl=}, {flops=})"
        iters, start, end = 10, 6, 9
        for i in range(iters):
            switch_profile(
                iter_id=i,
                start=start,
                end=end,
            )

            with add_nvtx_event(event_str):
                compiled_gemm(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor, current_stream)

    return exec_time  # Return execution time in microseconds

if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    parser = argparse.ArgumentParser(
        description="Example of Sm100 Dense Persistent BlockScaled GEMM."
    )

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(512, 256, 256, 1),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--mma_tiler_mn",
        type=parse_comma_separated_ints,
        default=(128, 128),
        help="Mma tile shape (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        default=(1, 1),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.Float4E2M1FN)
    parser.add_argument("--sf_dtype", type=cutlass.dtype, default=cutlass.Float8E8M0FNU)
    parser.add_argument("--sf_vec_size", type=int, default=16)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )
    parser.add_argument(
        "--warmup_iterations", type=int, default=0, help="Warmup iterations"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run the kernel",
    )
    parser.add_argument(
        "--skip_ref_check", action="store_true", help="Skip reference checking"
    )
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    exec_time = run(
        args.mnkl,
        args.ab_dtype,
        args.sf_dtype,
        args.sf_vec_size,
        args.c_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.mma_tiler_mn,
        args.cluster_shape_mn,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
    )
    print(f"PASS with execution time: {exec_time} ms")
