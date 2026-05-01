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

import os
import argparse
from typing import Optional, Type, Tuple, Union
import cuda.bindings.driver as cuda

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack
from cutlass import const_expr

"""
A high-performance batched dense GEMM (C = A * B) example for the NVIDIA Blackwell SM100 architecture
using CUTE DSL.
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Blackwell's tcgen05.mma for matrix multiply-accumulate (MMA) operations (including 2cta mma instructions)
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Supports multi-stage pipeline to overlap computation and memory access

This GEMM works as follows:
1. Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
3. Load completed accumulator from tensor memory (TMEM) to registers (RMEM) using tcgen05.ld.
4. Type convert C matrix to output type.
5. Optionally store C matrix from registers (RMEM) to shared memory (SMEM) to global memory (GMEM) with TMA operations,
   or directly store C matrix from registers (RMEM) to global memory (GMEM) without TMA operations.
6. Optionally accept an elementwise lambda function epilogue_op to apply to the output tensor:
   e.g., relu can set epilogue_op = lambda x: cute.where(x > 0, x, cute.full_like(x, 0))

SM100 tcgen05.mma instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Write accumulator to TMEM
The accumulator in TMEM must then be loaded to registers before writing back to GMEM.

To run this example:

.. code-block:: bash

    python examples/blackwell/dense_gemm.py                                     \
      --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                  \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                             \
      --mnkl 8192,8192,8192,1                                                   \
      --use_tma_store --use_2cta_instrs

The above example command compute batched gemm with M=8192, N=8192, K=8192,
batch_count=1. The Blackwell tcgen05 MMA tile shape used 2 cta with 256x128
MMA tile and the cluster shape is (2,1). The input, mma accumulator and output
data type are set as fp16, fp32 and fp16, respectively.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/blackwell/dense_gemm.py                                \
      --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                 \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
      --mnkl 8192,8192,8192,1                                                  \
      --use_tma_store --use_2cta_instrs

Constraints:
* Supported input data types: fp16, bf16, tf32, int8, uint8, fp8 (e4m3fn, e5m2),
  see detailed valid dtype combinations in below DenseGemmKernelSm100 class documentation
* A/B tensor must have the same data type
* Mma tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
* Mma tiler N must be 8-256, step 8
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if use_2cta_instrs=True
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 4, 8, and 16 for TFloat32,
  Float16/BFloat16, and Int8/Uint8/Float8, respectively.
* OOB tiles are not allowed when TMA store is disabled
"""


DEBUG_MODE = int(os.environ.get("DEBUG_MODE", "0")) == 1


class DenseGemmKernelSm100:
    """
    This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Blackwell GPUs.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param use_2cta_instrs: Whether to use CTA group 2 for advanced thread cooperation
    :type use_2cta_instrs: bool
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tiler (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]
    :param use_tma_store: Whether to use Tensor Memory Access (TMA) for storing results
    :type use_tma_store: bool

    :note: In current version, A and B tensor must have the same data type
        - i.e., Float8E4M3FN for A and Float8E5M2 for B is not supported

    :note: Supported A/B data types:
        - TFloat32
        - Float16/BFloat16
        - Int8/Uint8
        - Float8E4M3FN/Float8E5M2

    :note: Supported accumulator data types:
        - Float32 (for all floating point A/B data types)
        - Float16 (only for fp16 and fp8 A/B data types)
        - Int32 (only for uint8/int8 A/B data types)

    :note: Supported C data types:
        - Float32 (for float32 and int32 accumulator data types)
        - Int32 (for float32 and int32 accumulator data types)
        - Float16/BFloat16 (for fp16 and fp8 accumulator data types)
        - Int8/Uint8 (for uint8/int8 accumulator data types)
        - Float8E4M3FN/Float8E5M2 (for float32 accumulator data types)

    :note: Constraints:
        - MMA tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
        - MMA tiler N must be 32-256, step 32
        - Cluster shape M must be multiple of 2 if use_2cta_instrs=True
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16

    Example:
        >>> gemm = DenseGemmKernelSm100(
        ...     acc_dtype=cutlass.Float32,
        ...     use_2cta_instrs=True,
        ...     mma_tiler_mn=(128, 128),
        ...     cluster_shape_mn=(2, 2)
        ... )
        >>> gemm(a_tensor, b_tensor, c_tensor, stream)
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_tma_store: bool,
        debug_print: bool = False,
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.
            - use_2cta_instrs: Boolean indicating if the tcgen05 MMA variant
              with cta_group=2 should be used.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        3. Output C tensor store mode:
            - use_tma_store: Boolean indicating whether to use Tensor Memory Access (TMA) for storing results.

        :param acc_dtype: Data type of the accumulator.
        :type acc_dtype: type[cutlass.Numeric]
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: Tuple[int, int]
        :param use_2cta_instrs: Boolean, True to use cta_group=2 MMA variant.
        :type use_2cta_instrs: bool
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        :param use_tma_store: Use Tensor Memory Access (TMA) or normal store for output C tensor.
        :type use_tma_store: bool
        """

        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler_mnk = (*mma_tiler_mn, 1) # K dimension is deferred in _setup_attributes
        self.use_tma_store = use_tma_store

        self.cta_group = ( # tcgen05.mma.cta_group::N, where N can be 1 or 2
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1 # TODO(REVIEW): what does occupancy mean, the block occupancy in one SM ?
        
        self.threads_per_cta = 128 # one warp group with 128 threads, to access 128 rows of tmem (one warp for 32 rows)
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100") # the same as sm90, 227KB
        
        # NOTE: TMA smem alignment due to swizzle
        # since the maximum swizzle pattern SW(B3, M4, S3) has a period of 2^(3+4+3) = 2^10 = 1024B
        # so we have to align to 1024B
        self.buffer_align_bytes = 1024
        
        self.debug_print = debug_print
        
        if const_expr(self.debug_print):
            print()
            print(f"Initialized DenseGemmKernelSm100 with acc_dtype={self.acc_dtype}, "
                  f"use_2cta_instrs={self.use_2cta_instrs}, "
                  f"mma_tiler_mn={mma_tiler_mn}, "
                  f"cluster_shape_mn={cluster_shape_mn}, "
                  f"cta_group={self.cta_group}, "
                  f"use_tma_store={use_tma_store}, "
                  f"occupancy={self.occupancy}, "
                  f"threads_per_cta={self.threads_per_cta}, "
                  f"buffer_align_bytes={self.buffer_align_bytes}, "
                  f"smem_capacity={self.smem_capacity} bytes"
            )
            print()

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        - Computing tensor memory allocation columns
        """
        # Configure tiled umma
        #
        # umma atom:
        #   Thr Layout VMNK: (CTA_GROUP=2,ATOM_M=1,ATOM_N=1,ATOM_K=1):(1,0,0,0)
        #   Shape MNK: (tileM=256, tileN=128, tiledK=16)
        #   TV Layout A: (CTA_GROUP=2, (CTA_M=128, CTA_K=16)):(128,(1,256))
        #   TV Layout B: (CTA_GROUP=2, (CTA_N=64, CTA_K=16)):(64,(1,128))
        #   TV Layout C: (CTA_GROUP=2,(CTA_M=128, CTA_N=128)):(128,(1,256))
        # NOTE:
        #   1. different from wgmma's thr layout (128,2,1,1), umma can use a CTA group (at most a CTA-pair for now)
        #       to finish a umma together, while each CTA only needs one thread to issue the umma instructions
        #       but one warp group to issue load/store, so the thr layout is just (2, (1,1,1))
        #   2. different from wgmma's mem role, where A can be in rmem/smem, B must in smem and C must in rmem,
        #       umma's  A can be in tmem/smem, B must in smem and C must in tmem
        #   3. umma's K dim is still 32B (16 for bf16/fp16), and N dim is still range(8, 256+8, 8),
        #       but umma's M dim raises up to either 64 or 128 from wgmma's only 64 for one CTA, and a CTA-pair can together handle M=128/256 resp.
        #       where one CTA handles a half-row-sliced C (M128, N128) with half-row-sliced A (M=128, K16),
        #       and half-col-sliced B (N64, K16), i.e. A is local in one CTA but B is shared in dist-smem across a CTA-pair
        #       so each CTA needs to access the whole B tile via dist-smem to finish its half-row-sliced C job
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype=self.a_dtype,
            a_leading_mode=self.a_major_mode,
            b_leading_mode=self.b_major_mode,
            acc_dtype=self.acc_dtype,
            cta_group=self.cta_group,
            mma_tiler_mn=self.mma_tiler_mnk[:2], # (tileM=256, tileN=128)
            a_source=tcgen05.OperandSource.SMEM, # SS
        )
        
        # Setup atom thread layout for tiled MMA (which equals to CTA layout by now)
        self.atom_thr_id = tiled_mma.thr_id
        self.atom_thr_shape = self.atom_thr_id.shape
        self.atom_thr_size = cute.size(self.atom_thr_shape)

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler_mnk = (
            self.mma_tiler_mnk[0],
            self.mma_tiler_mnk[1],
            mma_inst_shape_k * mma_inst_tile_k, # 16 x 4 = 64
        )
        
        # Compute CTA tile shape
        # (CTA_tileM=126, tileN=128, tileK=64)
        self.cta_tile_shape_mnk = ( # 2 CTAs slicing M dim per MMA tile
            self.mma_tiler_mnk[0] // cute.size(self.atom_thr_shape), # // 2
            self.mma_tiler_mnk[1], # N dim only sharded but not sliced by CTAs
            self.mma_tiler_mnk[2],
        )

        # Compute cluster layout
        # (CTA_GROUP=(2), CTA_M=1, CTA_N=1, CTA_K=1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (self.atom_thr_shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2]) # mcast along CTA_N dim
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1]) # mcast along CTA_M dim
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
        # (epiM=128:1, epiN=32:1)
        if const_expr(self.use_tma_store):
            self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
                cta_tile_shape=self.cta_tile_shape_mnk,
                use_2cta_instrs=self.use_2cta_instrs,
                layout_d=self.c_layout,
                elem_ty_d=self.c_dtype,
            )
        else:
            self.epi_tile = self.cta_tile_shape_mnk[:2]

        # Setup A/B/C stage count in shared memory
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.smem_capacity,
            self.occupancy,
            self.use_tma_store,
            self.debug_print,
        )

        # Compute A/B/C shared memory layout
        # sA: S<3,4,3> o 0 o (MMA=(128,16), MMA_M=1, MMA_K=4, STAGE=8):((64,1),0,16,8192)
        # sB: S<3,4,3> o 0 o (MMA=(64,16), MMA_N=1, MMA_K=4, STAGE=8):((64,1),0,16,4096)
        # sC: S<2,4,3> o 0 o (epi_M=(8,16), epi_N=(32,1), epi_stages=(1,2)):((32,256),(1,0),(0,4096))
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma=tiled_mma,
            mma_tiler_mnk=self.mma_tiler_mnk,
            a_dtype=self.a_dtype,
            num_stages=self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma=tiled_mma,
            mma_tiler_mnk=self.mma_tiler_mnk,
            b_dtype=self.b_dtype,
            num_stages=self.num_ab_stage,
        )
        self.c_smem_layout_staged = (
            sm100_utils.make_smem_layout_epi(
                epi_dtype=self.c_dtype,
                epi_layout=self.c_layout,
                epi_tile=self.epi_tile,
                epi_stage=self.num_c_stage,
            )
            if self.use_tma_store
            else None
        )

        # Compute the number of tensor memory allocation columns
        # MMA_C=(128, 128) with dtype fp32 => 128 rows x 128 cols in tmem
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler_mnk
        )
        
        self.tiled_mma = tiled_mma
        
        if const_expr(self.debug_print):
            print()
            print(f"MMA Tiler (M,N,K): {self.mma_tiler_mnk=}")
            print(f"CTA Tile Shape (M,N,K): {self.cta_tile_shape_mnk=}")
            print(f"Cluster Shape (M,N): {self.cluster_shape_mn=}")
            print(f"Cluster Layout: {self.cluster_layout_vmnk=}")
            print(f"Thread layout (CTA_GROUP, ATOM_M, ATOM_N, ATOM_K): {self.atom_thr_id=}")
            print(f"Number of multicast CTAs for A: {self.num_mcast_ctas_a=}")
            print(f"Number of multicast CTAs for B: {self.num_mcast_ctas_b=}")
            print(f"Epilogue Tile Shape (M,N): {self.epi_tile=}")
            print(f"Number of AB stages: {self.num_ab_stage=}")
            print(f"Number of accumulator stages: {self.num_acc_stage=}")
            print(f"Number of C stages: {self.num_c_stage=}")
            print(f"Number of TMEM alloc columns: {self.num_tmem_alloc_cols=}")
            print()
            
            print()
            print(f"A SMEM layout (a_smem_layout_staged) (MMA,MMA_M,MMA_K,STAGE): {self.a_smem_layout_staged}")
            print(f"B SMEM layout (b_smem_layout_staged) (MMA,MMA_N,MMA_K,STAGE): {self.b_smem_layout_staged}")
            print(f"C SMEM layout (c_smem_layout_staged) (MMA,MMA_M,MMA_N,STAGE): {self.c_smem_layout_staged}")
            print()
            
            print()
            print("self.tiled_mma: ", self.tiled_mma, f"\n\nshape_mnk: {self.tiled_mma.shape_mnk}", f"thr_id.shape: {self.atom_thr_shape}")
            print()
    
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes
        - Setup TMA load/store atoms and tensors
        - Compute grid size
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a: Input tensor A
        :type a: cute.Tensor
        :param b: Input tensor B
        :type b: cute.Tensor
        :param c: Output tensor C
        :type c: cute.Tensor
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        :raises AssertionError: If OOB (Out-Of-Bounds) tiles are present when TMA store is disabled.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        # Check if input data types are compatible with MMA instruction
        if const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        tiled_mma = self.tiled_mma

        # Setup TMA load for A
        # sA: S<3,4,3> o 0 o (MMA=(128,16),MMA_M=1,MMA_k=4,STAGE=8):((64,1),0,16,8192)
        # tma_atom_a: ThrID=(2:1), TV_src=(2,8192), TV_dst=(2,8192), where 8192 = 128 x 16 x 4, (2:1) for a CTA-pair
        # tma_tensor_a: (pM=2048,pK=1024,pL=1):(1@1,1@0,1@2)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A( # cp.async GMEM -> SMEM bulk tensor copy Operation
            cluster_shape_mnk=self.cluster_shape_mn, # (2, 1)
            atom_thr_id=self.atom_thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0)) # removing the pipeline stage dim
        a_smem_size = cute.cosize(self.a_smem_layout_staged.outer)
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            op=a_op,
            gmem_tensor=a,
            smem_layout=a_smem_layout,
            mma_tiler_mnk=self.mma_tiler_mnk,
            tiled_mma=tiled_mma,
            cluster_shape_vmnk=self.cluster_layout_vmnk.shape, # ((2),1,1,1)
            internal_type=( # if fp32 input, we just directly use tf32 in tma load, since it will be converted to fp32 in umma anyway
                cutlass.TFloat32 if a.element_type is cutlass.Float32 else None
            ),
        )

        # Setup TMA load for B
        # sB: S<3,4,3> o 0 o (MMA=(64,16),MMA_M=1,MMA_k=4,STAGE=8):((64,1),0,16,4096)
        # tma_atom_b: ThrID=(2:1), TV_src=(2,4096), TV_dst=(2,4096), where 4096 = 64 x 16 x 4, (2:1) for a CTA-pair
        # tma_tensor_b: (N=4096,K=1024,1):(1@1,1@0,1@2)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B( # cp.async GMEM -> SMEM bulk tensor copy Operation
            cluster_shape_mnk=self.cluster_shape_mn, 
            atom_thr_id=self.atom_thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0)) # removing the pipeline stage dim
        b_smem_size = cute.cosize(self.b_smem_layout_staged.outer)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            op=b_op,
            gmem_tensor=b,
            smem_layout=b_smem_layout,
            mma_tiler_mnk=self.mma_tiler_mnk,
            tiled_mma=tiled_mma,
            cluster_shape_vmnk=self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if b.element_type is cutlass.Float32 else None
            ),
        )

        # Setup store for C
        # sC: S<2,4,3> o 0 o (epi_M=(8,16),epi_N=(32,1),epi_STAGES=(1,2)):((32,256),(1,0),(0,4096))
        # tma_atom_c: ThrID=(1:0), TV_src=(1,4096), TV_dst=(1,4096), where 4096 = 8X16 x 32 x 2, (1:0) for a single CTA
        # tma_tensor_c: (2048,4096,1):(1@1,1@0,1@2)
        tma_atom_c, tma_tensor_c = None, None
        c_smem_size, c_cta_v_layout = 0, None
        if const_expr(self.use_tma_store):
            c_op = cpasync.CopyBulkTensorTileS2GOp() # cp.async SMEM -> GMEM bulk tensor copy Operation 
            c_cta_v_layout = cute.composition( # (128,32):(1@0,1@1), col-major
                cute.make_identity_layout(c.shape), self.epi_tile
            )
            epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0)) # removing the pipeline stage dim
            c_smem_size = cute.cosize(self.c_smem_layout_staged.outer)
            
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                op=c_op,
                gmem_tensor=c,
                smem_layout=epi_smem_layout,
                cta_tiler=c_cta_v_layout, # it's ok to just pass in `self.epi_tile`
            )

        # Compute grid size
        # NOTE: the number of TMA load bytes (tx_count for main pipeline) combines the sA and sB size one stage, 
        # and needs to times the CTA-pair number since we need all the sA, sB data loaded for both CTAs to start the umma 
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * self.atom_thr_size
        grid = self._compute_grid(c, self.cta_tile_shape_mnk, self.cluster_shape_mn)

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            # mainloop full/empty mbar array ptrs for each ab stage
            # to synchronize the producer (TMA-loading sA and sB) 
            # with the consumer (UMMA using sA and sB) for each ab stage in the mainloop
            ab_full_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            
            # tmem accumulation full mbar for each acc stage
            # to synchronize the producer (UMMA writing accumulators to tmem) 
            # with the consumer (tcgen05.ld loading accumulators from tmem to rmem) for each acc stage in the epilogue
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            
            # the mbar ptr to synchronize all threads in two CTAs before issuing tmem deallocation
            tmem_dealloc_mbar_ptr: cutlass.Int64
            
            # the smem buffer to hold the allocated tmem address
            tmem_holding_smem_buf: cutlass.Int32

            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, a_smem_size
                ],
                self.buffer_align_bytes,
            ]
            
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, b_smem_size
                ],
                self.buffer_align_bytes,
            ]
            
            # (EPI_M, EPI_N, EPI_STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, c_smem_size
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        
        if const_expr(self.debug_print):
            print()
            print(f"{self.a_dtype=}, {self.b_dtype=}, {self.c_dtype=}, {self.a_major_mode=}, {self.b_major_mode=}, {self.c_layout=}")
            print(f"{a_smem_size=}, {b_smem_size=}, {c_smem_size=}, {c_cta_v_layout=}")
            print(f"{a_copy_size=}, {b_copy_size=}, {self.num_tma_load_bytes=}")
            print()
            
            print()
            print("TMA A: a_op: ", a_op, "\ntma_atom_a: ", tma_atom_a)
            print()
            print("TMA B: b_op: ", b_op, "\ntma_atom_b: ", tma_atom_b)
            print()
            print("TMA C: c_op: ", c_op, "\ntma_atom_c: ", tma_atom_c)
            print()
            
            cute.printf("")
            cute.printf("tma_tensor_a: {}", tma_tensor_a)
            cute.printf("")
            cute.printf("tma_tensor_b: {}", tma_tensor_b)
            cute.printf("")
            cute.printf("tma_tensor_c: {}", tma_tensor_c)
            cute.printf("")
        
            cute.printf("grid: {}", grid)
        
        # Launch the kernel
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c if self.use_tma_store else c,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
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
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the batched GEMM computation.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = self.use_2cta_instrs
        is_thread0 = tidx == 127 and bidx == 15 and bidy == 31 and bidz == 0 # used only for debug print

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch Tma desc
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if const_expr(self.use_tma_store):
                # TODO(REVIEW): in Hopper, we do not use prefetch tma atom for C's S2G, why prefetch for Blackwell ?
                cpasync.prefetch_descriptor(tma_atom_c)

        # /////////////////////////////////////////////////////////////////////////////
        #  Setup cta/thread coordinates
        # /////////////////////////////////////////////////////////////////////////////
        
        # Coords inside cluster
        mma_tile_coord_v = bidx % self.atom_thr_size # CTA idx in the CTA-pair
        is_leader_cta = mma_tile_coord_v == 0 # CTA0 is the leader
        cta_rank_in_cluster = cute.arch.make_warp_uniform( # CTA idx in the cluster, which might be different from mma_tile_coord_v if cluster size > 2
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_mnk = cute.arch.block_in_cluster_idx() # CTA xyz coord in the cluster
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord( # CTA (pair_v, rest_pair_xyz) coord in the cluster
            cta_rank_in_cluster
        )
        
        # Coords outside cluster
        cta_coord = (bidx, bidy, bidz) # CTA idx in the grid
        mma_tile_coord_mnl = ( # CTA-pair idx in the grid, i.e. the (REST_M,REST_N,REST_L) idx
            cta_coord[0] // self.atom_thr_size,
            cta_coord[1],
            cta_coord[2],
        )
        
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("tidx: {}, warp_idx: {}, block_idx: ({}, {}, {})", tidx, warp_idx, bidx, bidy, bidz)
                cute.printf("mma_tile_coord_v: {}, is_leader_cta: {}, cta_rank_in_cluster: {}, ", mma_tile_coord_v, is_leader_cta, cta_rank_in_cluster)
                cute.printf("block_in_cluster_coord_mnk: {}", block_in_cluster_coord_mnk)
                cute.printf("block_in_cluster_coord_vmnk: {}", block_in_cluster_coord_vmnk)
                cute.printf("mma_tile_coord_mnl: {}", mma_tile_coord_mnl)
        
        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc and init: a+b full/empty, accumulator full, tensor memory dealloc barrier
        # /////////////////////////////////////////////////////////////////////////////
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Fetch smem data ptrs
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_smem_buf = storage.tmem_holding_smem_buf
        ab_full_empty_mbar_ptr = storage.ab_full_empty_mbar_ptr.data_ptr()
        acc_full_mbar_ptr = storage.acc_full_mbar_ptr.data_ptr()

        # /////////////////////////////////////////////////////////////////////////////
        #  Initialize mainloop ab_pipeline (barrier) and states
        # /////////////////////////////////////////////////////////////////////////////
        # NOTE: different from Hopper's warp-group level warp specialization pipeline,
        # Blackwell's pipeline only involves the first warp for each CTA (elected one lane to actually issue), where:
        #   1. warp0 for each CTA is the TMA producer, to load its A and (sharded) B from gmem to smem
        #   2. warp0 for each CTA is also the TMA consumer, to wait for the sA and sB to be ready
        #   3. only warp0 in the leader CTA (CTA0 in a CTA-pair) is responsible for issuing the UMMA
        # i.e., both producer and consumer warps are of size 1 and be the same warp itself
        num_tma_producer = 1
        ab_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, size=num_tma_producer
        )
        ab_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_ab_stage
        )
        
        num_tma_consumer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1 # mutlicast along M/N dim, counting itself twice, thus minus 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, size=num_tma_consumer
        )
        ab_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_ab_stage
        )
        
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=ab_full_empty_mbar_ptr,
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Initialize acc_pipeline (barrier) and states
        # /////////////////////////////////////////////////////////////////////////////
        # NOTE: this is the tmem -> rmem pipeline for the first step in the epilogue, where:
        #   1. warp0 for each CTA is the tmem producer (elected one lane), waiting for all the UMMA in the mainloop finished
        #       to arrive the acc full mbar with `tcgen05.commit.mbarrier::arrive::one`
        #   2. all threads (one warp group) in each CTA are the tmem consumers, waiting for the acc full mbar
        #       to issue `tcgen05.ld` to load the accumulators from tmem to rmem
        num_acc_producer = 1
        acc_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, size=num_acc_producer
        )
        acc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_acc_stage
        )
        
        num_acc_consumer = self.threads_per_cta
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, size=num_acc_consumer,
        )
        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_acc_stage
        )
        
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=acc_full_mbar_ptr,
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Tensor memory dealloc barrier init
        # /////////////////////////////////////////////////////////////////////////////
        if use_2cta_instrs:
            if warp_idx == 0: # tmem manager
                num_tmem_dealloc_threads = 32
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(
                        tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads
                    )
        cute.arch.mbarrier_init_fence() # fence.mbarrier_init: to ensure the mbarrier init is visible to all threads in the CTA

        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()
            
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                # FIXME: why print the stuff below causing hang ?
                # cute.printf("[ab_pipeline] num_stages: {}, producer_mask: {}, consumer_mask: {}, is_leader_cta: {}", ab_pipeline.num_stages, ab_pipeline.producer_mask, ab_pipeline.consumer_mask, ab_pipeline.is_leader_cta)
                # cute.printf("[acc_pipeline] num_stages: {}, producer_mask: {}, consumer_mask: {}", acc_pipeline.num_stages, acc_pipeline.producer_mask, acc_pipeline.consumer_mask)
            
        # /////////////////////////////////////////////////////////////////////////////
        #  Setup smem tensor A/B/C
        # /////////////////////////////////////////////////////////////////////////////
        
        # (MMA=(128,16), MMA_M=1, MMA_K=4, STAGE=8)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA=(64,16), MMA_N=1, MMA_K=4, STAGE=8)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        # (EPI_M=(8,16), EPI_N=(32,1), EPI_STAGE=2)
        sC = (
            storage.sC.get_tensor(
                c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
            )
            if self.use_tma_store
            else None
        )
        
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("sA: {}", sA)
                cute.printf("sB: {}", sB)
                cute.printf("sC: {}", sC)
                cute.printf("")

        # /////////////////////////////////////////////////////////////////////////////
        #  Compute multicast mask for A/B buffer full
        # /////////////////////////////////////////////////////////////////////////////
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            # NOTE: it calls `cute.make_layout_image_mask` internally
            a_full_mcast_mask = cpasync.create_tma_multicast_mask( # 0b(10), only for itself
                cta_layout_vmnk=cluster_layout_vmnk,
                cta_coord_vmnk=block_in_cluster_coord_vmnk, 
                mcast_mode=2 # multicast along N dim
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask( # 0b(10), only for itself 
                cta_layout_vmnk=cluster_layout_vmnk, 
                cta_coord_vmnk=block_in_cluster_coord_vmnk, 
                mcast_mode=1 # multicast along M dim
            )
            
            if const_expr(self.debug_print):
                if is_thread0:
                    cute.printf("")
                    cute.printf("a_full_mcast_mask: {}, b_full_mcast_mask: {}", a_full_mcast_mask, b_full_mcast_mask)
                    cute.printf("")

        # /////////////////////////////////////////////////////////////////////////////
        #  Local_tile partition global tensors
        # /////////////////////////////////////////////////////////////////////////////
        
        # NOTE: we don't manually compute tile coordinates for global local tiles below
        # since we carry the (REST_M, REST_N, REST_K) dims in the local tile iter the whole time
        # and at the right time, we will use `mma_tile_coord_mnl` to index the right local tile out
        # so they all start with (0, 0, 0) offset in the tiles for now
        
        # (tileM=256, tileK=64, restM=8, restK=16, restL=1)
        gA_mkl = cute.local_tile(
            input=mA_mkl,
            tiler=cute.slice_(self.mma_tiler_mnk, (None, 0, None)), # (M, K)
            coord=(None, None, None)
        )
        # (tileN=128, tileK=64, restN=32, restK=16, restL=1)
        gB_nkl = cute.local_tile(
            input=mB_nkl, 
            tiler=cute.slice_(self.mma_tiler_mnk, (0, None, None)), # (N, K)
            coord=(None, None, None)
        )
        # (tileM=256, tileN=128, restM=8, restN=32, restL=1)
        gC_mnl = cute.local_tile(
            input=mC_mnl,
            tiler=cute.slice_(self.mma_tiler_mnk, (None, None, 0)), # (M, N)
            coord=(None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3]) # restK dim for the iterations in the mainloop
        
        if const_expr(self.debug_print):
            if is_thread0:
                # FIXME: printing the tensors below causing incorrectness, need to investigate
                cute.printf("")
                cute.printf("k_tile_cnt: {}", k_tile_cnt)
                # cute.printf("mA_mkl:")
                # cute.print_tensor(mA_mkl)
                cute.printf("mA_mkl: {}", mA_mkl)
                # cute.printf("mB_nkl:")
                # cute.print_tensor(mB_nkl)
                cute.printf("mB_nkl: {}", mB_nkl)
                # cute.printf("mC_mnl:")
                # cute.print_tensor(mC_mnl)
                cute.printf("mC_mnl: {}", mC_mnl)
                
                cute.printf("")
                # cute.printf("gA_mkl:")
                # cute.print_tensor(gA_mkl)
                cute.printf("gA_mkl: {}", gA_mkl)
                # cute.printf("gB_nkl:")
                # cute.print_tensor(gB_nkl)
                cute.printf("gB_nkl: {}", gB_nkl)
                # cute.printf("gC_mnl:")
                # cute.print_tensor(gC_mnl)
                cute.printf("gC_mnl: {}", gC_mnl)
                cute.printf("")

        # /////////////////////////////////////////////////////////////////////////////
        #  Partition global tensor for TiledMMA_A/B/C
        # /////////////////////////////////////////////////////////////////////////////
        
        # NOTE: different from Hopper, which uses the consumer tidx to slice the tiled wgmma,
        # we use the CTA idx in the CTA-pair to slice tiled umma, since each CTA has only one warp for it
        # and each CTA is responsible for half-row-sliced A/C once, and half-column-sliced B twice (each shared sB distributed in each CTA)
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        
        # (MMA=(128,16), MMA_M=1, MMA_K=4, RestM=8, RestK=16, RestL=1)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA=(64,16), MMA_N=1, MMA_K=4, RestN=32, RestK=16, RestL=1)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA=(128,128), MMA_M=1, MMA_N=1, RestM=8, RestN=32, RestL=1)
        tCgC = thr_mma.partition_C(gC_mnl)
        
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("tCgA: {}", tCgA)
                cute.printf("tCgB: {}", tCgB)
                cute.printf("tCgC: {}", tCgC)
                cute.printf("")

        # /////////////////////////////////////////////////////////////////////////////
        #  Partition global/shared tensor for TMA load A/B
        # /////////////////////////////////////////////////////////////////////////////
        
        # TMA load A partition_S/D
        # sA_for_tma_partition: ((MMA, MMA_M, MMA_K)=((128,16),1,4), PIPE=8)
        # tCgA_for_tma_partition: ((MMA, MMA_M, MMA_K)=((128,16),1,4), RestM=8, RestK=16, RestL=1)
        sA_for_tma_partition = cute.group_modes(sA, 0, 3)
        tCgA_for_tma_partition = cute.group_modes(tCgA, 0, 3)
        
        a_cta_layout = cute.make_layout( # only need the 3rd N dim, (1):(0)
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        
        # tAsA: ((TMA_atom_v, rest_v)=(8192,1), PIPE=8)
        # tAgA: ((TMA_atom_v, rest_v)=((64,128),1), RestM=8, RestK=16, RestL=1)
        tAsA, tAgA = cpasync.tma_partition(
            atom=tma_atom_a,
            cta_coord=block_in_cluster_coord_vmnk[2], # along N
            cta_layout=a_cta_layout,
            smem_tensor=sA_for_tma_partition,
            gmem_tensor=tCgA_for_tma_partition,
        )
        
        # TMA load B partition_S/D
        # sB_for_tma_partition: ((MMA, MMA_N, MMA_K)=((64,16),1,4), PIPE=8)
        # tCgB_for_tma_partition: ((MMA, MMA_N, MMA_K)=((64,16),1,4), RestN=32, RestK=16, RestL=1)
        sB_for_tma_partition = cute.group_modes(sB, 0, 3)
        tCgB_for_tma_partition = cute.group_modes(tCgB, 0, 3)
        
        b_cta_layout = cute.make_layout( # only need the 2nd M dim: (1):(0)
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        
        # tBsB: ((TMA_atom_v, rest_v)=(4096,1), PIPE=8)
        # tBgB: ((TMA_atom_v, rest_v)=(((64,64),1), RestN=32, RestK=16, RestL=1)
        tBsB, tBgB = cpasync.tma_partition(
            atom=tma_atom_b,
            cta_coord=block_in_cluster_coord_vmnk[1], # along M
            cta_layout=b_cta_layout,
            smem_tensor=sB_for_tma_partition,
            gmem_tensor=tCgB_for_tma_partition,
        )

        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("a_cta_layout: {}", a_cta_layout)
                cute.printf("sA_for_tma_partition: {}", sA_for_tma_partition)
                cute.printf("tCgA_for_tma_partition: {}", tCgA_for_tma_partition)
                cute.printf("tAsA: {}", tAsA)
                cute.printf("tAgA: {}", tAgA)
                cute.printf("")
                
                cute.printf("")
                cute.printf("b_cta_layout: {}", b_cta_layout)
                cute.printf("sB_for_tma_partition: {}", sB_for_tma_partition)
                cute.printf("tCgB_for_tma_partition: {}", tCgB_for_tma_partition)
                cute.printf("tBsB: {}", tBsB)
                cute.printf("tBgB: {}", tBgB)
                cute.printf("")

        # /////////////////////////////////////////////////////////////////////////////
        #  Partition shared/tensor memory tensor for TiledMMA_A/B/C
        # /////////////////////////////////////////////////////////////////////////////
        
        # (MMA=1, MMA_M=1, MMA_K=4, STAGE=8):(0,0,2,1024)
        tCrA = thr_mma.make_fragment_A(sA)
        # (MMA=1, MMA_N=1, MMA_K=4, STAGE=8):(0,0,2,512)
        tCrB = thr_mma.make_fragment_B(sB)
        # NOTE: different from Hopper's private rmem layout per thread like tCrC: ((2,2,32),1,1):((1,2,4),0,0),
        # Blackwell's tCtAcc is shared across the CTA: (MMA=(128,128), MMA_M=1, MMA_N=1) : ((65536,1),0,0)
        acc_shape = thr_mma.partition_shape_C(self.mma_tiler_mnk[:2])
        # NOTE: this is only a fake fragment with layout, the actual tCtAcc needs to be in tmem,
        # but we haven't allocate and retrieve its addr yet
        tCtAcc_fake = thr_mma.make_fragment_C(acc_shape)
        
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("tCrA.layout: {}", tCrA.layout) # tCrA is not printable, so only print its layout
                cute.printf("tCrB.layout: {}", tCrB.layout) # tCrB is not printable, so only print its layout
                cute.printf("tCtAcc_fake.layout: {}", tCtAcc_fake.layout) # tCtAcc_fake is not printable, so only print its layout
                cute.printf("")

        # /////////////////////////////////////////////////////////////////////////////
        #  Cluster wait before tensor memory alloc
        # /////////////////////////////////////////////////////////////////////////////
        if cute.size(self.cluster_shape_mn) > 1:
            # wait for all CTAs in the cluster to arrive 
            # before warp0 of each CTA allocates the tmem
            cute.arch.cluster_wait()

        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc tensor memory buffer
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == 0: # tmem manager
            # Allocate a tmem buffer and store the tmem address in smem
            # using `tcgen05.alloc.cta_group::2 [smem_addr], cols`
            cute.arch.alloc_tmem(
                num_columns=self.num_tmem_alloc_cols, # 128 cols for 128 CTA_tileN
                smem_ptr_to_write_address=tmem_holding_smem_buf, 
                is_two_cta=use_2cta_instrs
            )
            
        # Bar.sync before each thread retrieves the stored tmem ptr from smem
        cute.arch.barrier()

        # /////////////////////////////////////////////////////////////////////////////
        #  Retrieving tensor memory ptr and make accumulator tensor
        # /////////////////////////////////////////////////////////////////////////////
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype, 
            alignment=16, 
            ptr_to_buffer_holding_addr=tmem_holding_smem_buf
        )
        # (MMA=(128, 128), MMA_M=1, MMA_N=1)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("tCtAcc: {}", tCtAcc)
                cute.printf("")

        # /////////////////////////////////////////////////////////////////////////////
        #  Partition for epilogue
        # /////////////////////////////////////////////////////////////////////////////
        
        # Make T2R tiled copy
        tiled_copy_t2r, tTR_tAcc, tTR_rAcc = self.epilog_tmem_copy_and_partition(
            tidx, 
            tAcc=tCtAcc, 
            tCgC=tCgC, 
            epi_tile=epi_tile, 
            use_2cta_instrs=use_2cta_instrs, 
            is_thread0=is_thread0
        )

        tTR_rC, tiled_copy_r2s, simt_atom = None, None, None
        tRS_rC, tRS_sC, bSG_sC, bSG_gC, tTR_gC = None, None, None, None, None
        if const_expr(self.use_tma_store):
            # Make R2S tiled copy
            # tTR_rC = cute.make_fragment(tTR_rAcc.shape, self.c_dtype) # deprecated API
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype) # new API, the bf16 version of tTR_rAcc
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition( 
                tiled_copy_t2r=tiled_copy_t2r, 
                tTR_rC=tTR_rC, 
                tidx=tidx, 
                sC=sC, 
                is_thread0=is_thread0
            )
            
            # Make S2G TMA tiled copy
            tma_atom_c, bSG_sC, bSG_gC = self.epilog_gmem_copy_and_partition(
                tidx=tidx, 
                atom=tma_atom_c, 
                tCgC=tCgC, 
                epi_tile=epi_tile, 
                sC=sC,
                is_thread0=is_thread0
            )
        else:
            # Make R2G tiled copy
            simt_atom, tTR_rC, tTR_gC = self.epilog_gmem_copy_and_partition(
                tidx=tidx, tiled_copy_t2r=tiled_copy_t2r, tCgC=tCgC, 
                epi_tile=epi_tile, sC=sC
            )

        # /////////////////////////////////////////////////////////////////////////////
        #  Slice to per mma tile index
        # /////////////////////////////////////////////////////////////////////////////
        
        # From: ((TMA_atom_v, rest_v)=((64,128),1), RestM=8, RestK=16, RestL=1)
        # To: ((TMA_atom_v, rest_v)=((64,128),1), RestK=16)
        tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])] # slice (RestM, RestL) idx
        # From: ((TMA_atom_v, rest_v)=(((64,64),1), RestN=32, RestK=16, RestL=1)
        # To: ((TMA_atom_v, rest_v)=(((64,64),1), RestK=16)
        tBgB = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])] # slice (RestN, RestL) idx
        if const_expr(self.use_tma_store):
            # From: ((ATOM_V, REST_V)=((32,128),1), EPI_M=1, EPI_N=4, RestM=8, RestN=32, RestL=1)
            # To: ((ATOM_V, REST_V)=((32,128),1), EPI_M=1, EPI_N=4)
            bSG_gC = bSG_gC[(None, None, None, *mma_tile_coord_mnl)] # slice (RestM, RestN, RestL) idx
        else:
            # From: (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
            # To: (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
            tTR_gC = tTR_gC[(None, None, None, None, None, *mma_tile_coord_mnl)] # slice (RestM, RestN, RestL) idx
        
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("tAgA_: {}", tAgA)
                cute.printf("tBgB_: {}", tBgB)
                if const_expr(self.use_tma_store):
                    cute.printf("bSG_gC_: {}", bSG_gC)
                else:
                    cute.printf("tTR_gC_: {}", tTR_gC)
                cute.printf("")

        # /////////////////////////////////////////////////////////////////////////////
        #  Mainloop: Pipelining TMA load A/B and UMMA
        # ///////////////////////////////////////////////////////////////////////////
        prefetch_k_tile_cnt = cutlass.min(self.num_ab_stage - 2, k_tile_cnt)

        if warp_idx == 0: # tma-load producer, as well as umma consumer/t2r producer if leader CTA
            # Peek for the first empty mbar to be arrived by the consumer w/o blocking
            peek_ab_empty_status = cutlass.Boolean(1)
            if ab_producer_state.count < k_tile_cnt:
                peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                    ab_producer_state
                )
            
            # /////////////////////////////////////////////////////////////////////////////
            #  Prefetch TMA load A/B
            # /////////////////////////////////////////////////////////////////////////////
            for prefetch_idx in cutlass.range(prefetch_k_tile_cnt, unroll=1):
                # Wait for current empty mbar to be arrived by the consumer
                ab_pipeline.producer_acquire(
                    ab_producer_state, 
                    try_acquire_token=peek_ab_empty_status
                )

                # TMA load A/B
                tma_full_mbar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
                cute.copy(
                    atom=tma_atom_a,
                    src=tAgA[(None, ab_producer_state.count)], # slice RestK idx
                    dst=tAsA[(None, ab_producer_state.index)], # slice ab stage idx
                    tma_bar_ptr=tma_full_mbar_ptr,
                    mcast_mask=a_full_mcast_mask,
                )
                cute.copy(
                    atom=tma_atom_b,
                    src=tBgB[(None, ab_producer_state.count)], # slice RestK idx
                    dst=tBsB[(None, ab_producer_state.index)], # slice ab stage idx
                    tma_bar_ptr=tma_full_mbar_ptr,
                    mcast_mask=b_full_mcast_mask,
                )

                # Peek for the next empty mbar to be arrived by the consumer w/o blocking
                ab_producer_state.advance()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )

            # Peek for the first full mbar to be arrived by the producer w/o blocking only by the leader CTA
            peek_ab_full_status = cutlass.Boolean(1)
            if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

            # /////////////////////////////////////////////////////////////////////////////
            # MMA mainloop
            # /////////////////////////////////////////////////////////////////////////////
            for k_tile in cutlass.range(k_tile_cnt):
                # /////////////////////////////////////////////////////////////////////////////
                # TMA Producer
                # /////////////////////////////////////////////////////////////////////////////
                
                # Wait for current empty mbar to be arrived by the consumer
                ab_pipeline.producer_acquire(
                    ab_producer_state, 
                    try_acquire_token=peek_ab_empty_status
                )

                if ab_producer_state.count < k_tile_cnt:
                    # TMA load A/B
                    tma_full_mbar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
                    cute.copy(
                        atom=tma_atom_a,
                        src=tAgA[(None, ab_producer_state.count)],
                        dst=tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=tma_full_mbar_ptr,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        atom=tma_atom_b,
                        src=tBgB[(None, ab_producer_state.count)],
                        dst=tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=tma_full_mbar_ptr,
                        mcast_mask=b_full_mcast_mask,
                    )
                    
                # /////////////////////////////////////////////////////////////////////////////
                # UMMA Consumer
                # /////////////////////////////////////////////////////////////////////////////
                if is_leader_cta:
                    # Wait for current full mbar to be arrived by the producer
                    ab_pipeline.consumer_wait(
                        ab_consumer_state,
                        try_wait_token=peek_ab_full_status
                    )

                    # Issuing UMMA for `num_kblocks` times looping over MMA_K dim
                    # tCtAcc += tCrA * tCrB
                    num_kblocks = cute.size(tCrA, mode=[2])
                    for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                        kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)
                        cute.gemm(
                            atom=tiled_mma,
                            d=tCtAcc,
                            a=tCrA[kblock_coord],
                            b=tCrB[kblock_coord],
                            c=tCtAcc,
                        )
                        
                        # Enable accumulate on tCtAcc after first kblock
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    # Arrive the empty mbar with `tcgen05.commit.mbarrier::arrive::one`
                    ab_pipeline.consumer_release(ab_consumer_state)

                # /////////////////////////////////////////////////////////////////////////////
                # TMA Producer
                # /////////////////////////////////////////////////////////////////////////////

                # Peek for the next empty mbar to be arrived by the consumer w/o blocking
                ab_producer_state.advance()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )

                # /////////////////////////////////////////////////////////////////////////////
                # UMMA Consumer
                # /////////////////////////////////////////////////////////////////////////////

                # Peek for the next full mbar to be arrived by the producer w/o blocking only by the leader CTA
                ab_consumer_state.advance()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt:
                    if is_leader_cta:
                        peek_ab_full_status = ab_pipeline.consumer_try_wait(
                            ab_consumer_state
                        )

            # Arrive the acc full mbar with `tcgen05.commit.mbarrier::arrive::one` 
            # by the leader CTA (UMMA consumer => T2R producer)
            if is_leader_cta:
                acc_pipeline.producer_commit(acc_producer_state)

        # /////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # /////////////////////////////////////////////////////////////////////////////

        # Release tmem allocation lock to allow different grid to allocate
        if warp_idx == 0: # tmem manager
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)

        # Wait for acc full buffer to be arrived by the t2r producer
        acc_pipeline.consumer_wait(acc_consumer_state)
        
        # Group the EPI_M and EPI_N modes together

        # From: (T2R=((T2R_COLS=32, T2R_ROWS=32),1), T2R_M=1, T2R_N=1, EPI_M=1, EPI_N=4)
        # to: (T2R=((T2R_COLS=32, T2R_ROWS=32),1), T2R_M=1, T2R_N=1, (EPI_M, EPI_N)=(1,4))
        tTR_tAcc = cute.group_modes(tTR_tAcc, begin=3, end=cute.rank(tTR_tAcc))
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3]) # EPI_M X EPI_N = 4
        if const_expr(self.use_tma_store):
            # From: ((ATOM_V, REST_V)=((32,128),1), EPI_M=1, EPI_N=4)
            # To: ((ATOM_V, REST_V)=((32,128),1), (EPI_M, EPI_N)=(1,4))
            bSG_gC = cute.group_modes(bSG_gC, begin=1, end=cute.rank(bSG_gC))
        else:
            # From: (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
            # To: (T2R, T2R_M, T2R_N, (EPI_M, EPI_N))
            tTR_gC = cute.group_modes(tTR_gC, begin=3, end=cute.rank(tTR_gC))
            
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("subtile_cnt: {}", subtile_cnt)
                cute.printf("tTR_tAcc_: {}", tTR_tAcc)
                if const_expr(self.use_tma_store):
                    cute.printf("_bSG_gC: {}", bSG_gC)
                else:
                    cute.printf("_tTR_gC: {}", tTR_gC)
                cute.printf("")

        # Make S2G TMA store pipeline
        c_pipeline = None
        if const_expr(self.use_tma_store):
            # Initialize tma store c_pipeline
            c_producer_group = pipeline.CooperativeGroup( # NOTE: this is only a dummy placeholder, no matter waht the arguments are
                pipeline.Agent.Thread, self.threads_per_cta
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

        # /////////////////////////////////////////////////////////////////////////////
        #  Store accumulator to global memory in subtiles
        # /////////////////////////////////////////////////////////////////////////////
        
        # TODO(REVIEW): why not range_constexpr here ?
        # for subtile_idx in cutlass.range_constexpr(subtile_cnt):
        for subtile_idx in range(subtile_cnt):
            # T2R copy to store accumulator from tmem to rmem
            cute.copy(
                atom=tiled_copy_t2r,
                src=tTR_tAcc[(None, None, None, subtile_idx)], # slice (EPI_M, EPI_N) idx
                dst=tTR_rAcc
            )

            if const_expr(self.use_tma_store):
                # Perform epilogue op on accumulator and convert to C type
                acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                tRS_rC.store(acc_vec)

                # R2S copy to store C from rmem to smem
                c_stage_idx = subtile_idx % self.num_c_stage
                cute.copy(
                    atom=tiled_copy_r2s,
                    src=tRS_rC,
                    dst=tRS_sC[(None, None, None, c_stage_idx)] # slice c stage idx
                )
                
                # Fence and barrier to make sure shared memory store is visible to TMA store
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                cute.arch.barrier()

                # S2G TMA store C from smem to gmem
                if warp_idx == 0: # issued only by warp0 (auto election inside)
                    cute.copy(
                        atom=tma_atom_c,
                        src=bSG_sC[(None, c_stage_idx)], # slice c stage idx
                        dst=bSG_gC[(None, subtile_idx)], # slice subtile idx
                    )
                    
                    # Fence and barrier to make sure TMA store is completed to recollect C buffer
                    c_pipeline.producer_commit() # `cp.async.bulk.commit_group`
                    c_pipeline.producer_acquire() # `cp.async.bulk.wait_group(num_stages-1)` 
                
                cute.arch.barrier() # wait warp0 before next iteration
            else:
                # Perform epilogue op on accumulator and convert to C type
                acc_vec = tTR_rAcc.load()
                acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                tTR_rC.store(acc_vec)

                # R2G copy to store C from rmem to gmem
                cute.copy(simt_atom, tTR_rC, tTR_gC[(None, None, None, subtile_idx)])

        # /////////////////////////////////////////////////////////////////////////////
        #  Dealloc the tensor memory buffer
        # /////////////////////////////////////////////////////////////////////////////
        
        cute.arch.barrier() # wait for all T2R->R2S->S2G copies to be issued before deallocating tmem
        if warp_idx == 0: # tmem manager
            if use_2cta_instrs:
                # Arrive at the peer CTA's dealloc mbar
                # by mapping the peer's mbar addr using `mapa.shared::cluster`
                cute.arch.mbarrier_arrive(
                    mbar_ptr=tmem_dealloc_mbar_ptr, 
                    peer_cta_rank_in_cluster=cta_rank_in_cluster ^ 1 # peer rank
                )
                # Wait for self CTA's dealloc mbar to be arrived by the peer CTA
                cute.arch.mbarrier_wait(mbar_ptr=tmem_dealloc_mbar_ptr, phase=0)
            
            # Deallocate the tmem buffer using `tcgen05.dealloc.cta_group::2.sync.aligned.b32`
            cute.arch.dealloc_tmem(
                tmem_ptr=tmem_ptr, 
                num_columns=self.num_tmem_alloc_cols, 
                is_two_cta=use_2cta_instrs
            )

        # /////////////////////////////////////////////////////////////////////////////
        #  Wait for C store complete
        # /////////////////////////////////////////////////////////////////////////////
        if const_expr(self.use_tma_store):
            c_pipeline.producer_tail() # `cp.async.bulk.wait_group(0)`

        # /////////////////////////////////////////////////////////////////////////////
        #  Wait for A/B buffer dangling empty mbar signals
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == 0:
            # Since we prefetch `prefetch_k_tile_cnt` times at first,
            # the producer state is prefetch_k_tile_cnt times in advance
            # and we need to reverse it to the actual state for next available smem buffer
            for i in range(prefetch_k_tile_cnt):
                ab_producer_state.reverse()
            
            # Call `producer_tail` to:
            #   1. first advance (num_stages-1) times to the last used smem buffer, 
            #       for which the consumer will arrive the corr. empty mbar at last
            #   2. then call `producer_acquire` to wait for the last empty mbar to be arrived by the consumer, 
            #       which ensures the producer won't early exit, causing the last empty mbar dangling
            ab_pipeline.producer_tail(ab_producer_state)

    @cute.jit
    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        tCgC: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
        is_thread0: cutlass.Boolean,
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
        # op: Ld32x32bOp(Repetition(32), Pack.NONE) => `tcgen05.ld.sync.aligned.32x32b.x32`, where:
        #   tcgen05.ld.sync.aligned.shape1.num{.pack}.b32    r, [taddr];
        #       .shape1 = { .16x64b, .16x128b, .16x256b, .32x32b } = num_rows x num_bits_per_thread
        #       .num    = { .x1, .x2, .x4, .x8, .x16, .x32, .x64, .x128 } = rep times along cols, i.e. num_cols
        #   so this copy atom means 32rows x 32cols of 32bits (4B, fp32) tmem, i.e. 1024 fp32 tC elems
        #   with properties:
        #       num_dp   = 32 (datapath / row / lane)
        #       num_bits = 32 (element bits)
        #       num_rep  = 32 (cols)
        #       pack     = Pack.NONE (no packing two 16-bit elements into one 32-bit)
        #
        # layout_src_tv: (32,1024):(0,1) => one warp handles 1024 fp32 elems (32rows x 32cols) in tmem, 
        #   all threads in the warp sharing the same base addr (so stride0=0)
        # layout_dst_tv: (32,32):(32,1) => each thread in one warp holds 32 fp32 elems in register, 
        #   contiguous in the thread dimension
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            cta_tile_shape=self.cta_tile_shape_mnk,
            layout_d=self.c_layout,
            elem_ty_d=self.c_dtype,
            elem_ty_acc=self.acc_dtype,
            epi_tile=epi_tile,
            use_2cta_instrs=use_2cta_instrs,
        )
        
        # Tile the tAcc of layout (MMA=(128, 128), MMA_M=1, MMA_N=1):((65536,1),0,0)
        # with epi_tile (epi_tileM=128, epi_tileN=32) into 
        # tAcc_epi: (epi_tileM=128, epi_tileN=32, epi_M=1, epi_N=4):(65536,1,0,32)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0)], # only take MMA dims
            epi_tile,
        )
        
        # Make t2r tiled copy
        # layout_src_tv_tiled: ((WARP, NUM_WARPS)=(32,4), (T2R_ROWS,T2R_COLS)=((32,32),1)):((0,1),((128,4),0))
        # layout_dst_tv_tiled: ((WARP, NUM_WARPS)=(32,4), T2R=(32,1)):((4,1),(128,0))
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            atom=copy_atom_t2r, 
            tmem_tensor=tAcc_epi[(None, None, 0, 0)] # only take epi tile dims (epi_tileM=128, epi_tileN=32)
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        
        # (T2R=((T2R_COLS=32, T2R_ROWS=32),1), T2R_M=1, T2R_N=1, EPI_M=1, EPI_N=4)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # from: (MMA=(128,128), MMA_M=1, MMA_N=1, RestM=8, RestN=32, RestL=1)
        # view to: (MMA=(128,128), RestM=8, RestN=32, RestL=1)
        gC_mnl_view = tCgC[((None, None), 0, 0, None, None, None)]
        
        # (EPI_TILE_M=128, EPI_TILE_N=32, EPI_M=1, EPI_N=4, RestM=8, RestN=32, RestL=1)
        gC_mnl_epi = cute.flat_divide(gC_mnl_view, epi_tile)
        
        # (T2R=(32,1), T2R_M=1, T2R_N=1, EPI_M=1, EPI_N=4, RestM=8, RestN=32, RestL=1)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        
        # (T2R=(32,1), T2R_M=1, T2R_N=1)
        # tTR_rAcc = cute.make_fragment( # deprecated API
        tTR_rAcc = cute.make_rmem_tensor( # new API
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, # fetch only (T2R, T2R_M, T2R_N) dims
            self.acc_dtype # fp32 rmem buffer for t2r dst
        )
        
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("copy_atom_t2r: layout_src_tv: {} | layout_dst_tv: {}", copy_atom_t2r.layout_src_tv, copy_atom_t2r.layout_dst_tv)
                cute.printf("tAcc_epi: {}", tAcc_epi)
                cute.printf("tiled_copy_t2r: layout_src_tv: {} | layout_src_tv_tiled: {} | layout_dst_tv: {} | layout_dst_tv_tiled: {}", tiled_copy_t2r.layout_src_tv, tiled_copy_t2r.layout_src_tv_tiled, tiled_copy_t2r.layout_dst_tv, tiled_copy_t2r.layout_dst_tv_tiled)
                cute.printf("tTR_tAcc: {}", tTR_tAcc)
                cute.printf("gC_mnl_view: {}", gC_mnl_view)
                cute.printf("gC_mnl_epi: {}", gC_mnl_epi)
                cute.printf("tTR_gC: {}", tTR_gC)
                cute.printf("tTR_rAcc: {}", tTR_rAcc)
                cute.printf("")
        
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    @cute.jit
    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
        is_thread0: cutlass.Boolean,
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

        :return: A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rC: The partitioned tensor C (register source)
            - tRS_sC: The partitioned tensor C (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        
        # Fetch some properties of the t2r tiled copy
        num_dp, num_bits, num_rep, pack = tcgen05.get_tmem_copy_properties(tiled_copy_t2r)
        
        # Make R2S copy atom from T2R tiled copy
        # since our T2R tiled copy is 32rows x 32cols, violating all `stmatrix` insts inside (m8n8xn, or m16n8xn)
        # it fall backs to `CopyUniversalOp`, where: layout_src_tv: (1,1):(0,0) | layout_dst_tv: (1,1):(0,0)
        # i.e. each thread needs to store 32 times for its 32 fp32 elements to smem
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            layout_d=self.c_layout, 
            elem_ty_d=self.c_dtype, 
            elem_ty_acc=self.acc_dtype,
            tiled_tmem_load=tiled_copy_t2r
        )
        
        # NOTE: different from Hopper's epilogue that making the R2Stiled copy using the `make_tiled_copy_S`
        # since the `register` is on the source side of the tiled mma as well as the CAtom tiled copy, 
        # here we use `make_tiled_copy_D` since the `register` is on the destination side of the T2R copy
        # but the underlying reason is the same: we want to use the same register TV layout
        # to directly copy the data without needing extra re-layout in registers
        # 
        # layout_src_tv_tiled: ((WARP,NUM_WARPS)=(32,4), VAL_PER_THREAD=(1,32)):((4,1),(0,128))
        # layout_dst_tv_tiled: ((WARP,NUM_WARPS)=(32,4),(1,32)):((4,1),(0,128))
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        
        # FROM: (epi_M=(8,16), epi_N=(32,1), epi_stages=(1,2))
        # TO: (R2S=(1,32), R2S_M=1, R2S_N=1, epi_stages=2)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        
        # FROM: (T2R=(32,1), T2R_M=1, T2R_N=1) 
        # TO: (R2S=(1,32), R2S_M=1, R2S_N=1)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        
        if const_expr(self.debug_print):
            if is_thread0:
                print(f"tiled_copy_t2r properties: num_dp: {num_dp} | num_bits: {num_bits} | num_rep: {num_rep} | pack: {pack}")
                cute.printf("")
                cute.printf("copy_atom_r2s: layout_src_tv: {} | layout_dst_tv: {}", copy_atom_r2s.layout_src_tv, copy_atom_r2s.layout_dst_tv)
                cute.printf("tiled_copy_r2s: layout_src_tv: {} | layout_src_tv_tiled: {} | layout_dst_tv: {} | layout_dst_tv_tiled: {}", tiled_copy_r2s.layout_src_tv, tiled_copy_r2s.layout_src_tv_tiled, tiled_copy_r2s.layout_dst_tv, tiled_copy_r2s.layout_dst_tv_tiled)
                cute.printf("tTR_rC: {}", tTR_rC)
                cute.printf("tRS_sC: {}", tRS_sC)
                cute.printf("tRS_rC: {}", tRS_rC)
                cute.printf("")
        
        return tiled_copy_r2s, tRS_rC, tRS_sC

    @cute.jit
    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        tCgC: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
        is_thread0: cutlass.Boolean,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to:
        - partition register array (source) and global memory (destination) for none TMA store version;
        - partition shared memory (source) and global memory (destination) for TMA store version.

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

        :return: A tuple containing either:
            - For TMA store: (tma_atom_c, bSG_sC, bSG_gC) where:
                - tma_atom_c: The TMA copy atom
                - bSG_sC: The partitioned shared memory tensor C
                - bSG_gC: The partitioned global tensor C
            - For non-TMA store: (simt_atom, tTR_rC, tTR_gC) where:
                - simt_atom: The SIMT copy atom
                - tTR_rC: The register tensor C
                - tTR_gC: The partitioned global tensor C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        
        # From: (MMA=(128,128), MMA_M=1, MMA_N=1, RestM=8, RestN=32, RestL=1)
        # to: (EPI_TILE_M=128, EPI_TILE_N=32, EPI_M=1, EPI_N=4, RestM=8, RestN=32, RestL=1)
        gC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None, None)],  # removing MMA_M and MMA_N dims
            epi_tile
        )
        
        if const_expr(self.use_tma_store):
            tma_atom_c = atom
            
            # sC_for_tma_partition: (epi_tile=((8,16),(32,1)), epi_stages=(1,2))
            # gC_for_tma_partition: (epi_tile=(128,32), EPI_M=1, EPI_N=4, RestM=8, RestN=32, RestL=1)
            sC_for_tma_partition = cute.group_modes(sC, 0, 2)
            gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
            
            # bSG_sC: ((ATOM_V, REST_V)=(4096,1), epi_stages=(1,2))
            # bSG_gC: ((ATOM_V, REST_V)=((32,128),1), EPI_M=1, EPI_N=4, RestM=8, RestN=32, RestL=1)
            bSG_sC, bSG_gC = cpasync.tma_partition(
                tma_atom_c,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=sC_for_tma_partition,
                gmem_tensor=gC_for_tma_partition,
            )
            
            if const_expr(self.debug_print):
                if is_thread0:
                    cute.printf("")
                    cute.printf("tma_atom_c: layout_src_tv: {} | layout_dst_tv: {}", tma_atom_c.layout_src_tv, tma_atom_c.layout_dst_tv)
                    cute.printf("gC_epi: {}", gC_epi)
                    cute.printf("sC_for_tma_partition: {}", sC_for_tma_partition)
                    cute.printf("gC_for_tma_partition: {}", gC_for_tma_partition)
                    cute.printf("bSG_sC: {}", bSG_sC)
                    cute.printf("bSG_gC: {}", bSG_gC)
                    cute.printf("")
            
            return tma_atom_c, bSG_sC, bSG_gC
        else:
            tiled_copy_t2r = atom
            
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_gC = thr_copy_t2r.partition_D(gC_epi)
            
            # (T2R, T2R_M, T2R_N)
            # tTR_rC = cute.make_fragment( # deprecated API
            tTR_rC = cute.make_rmem_tensor( # new API
                tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, 
                self.c_dtype
            )
            
            simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)
            
            return simt_atom, tTR_rC, tTR_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        smem_capacity: int,
        occupancy: int,
        use_tma_store: bool,
        debug_print: bool = False,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tile.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum of operand C in global memory.
        :type c_layout: utils.LayoutEnum
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int
        :param use_tma_store: Whether TMA store is enabled.
        :type use_tma_store: bool

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, epilogue stages)
        :rtype: tuple[int, int, int]
        """
        
        # Default one ACC stage to copy tmem to rmem, 
        # TODO(REVIEW): TMEM usually does not need double buffering, does it ?
        num_acc_stage = 1
        
        # Default C stages to copy C from smem to gmem
        num_c_stage = 2 if use_tma_store else 0

        # Calculate smem layout and size for one stage of A, B, and C with heuristics
        # sA: S<3,4,3> o 0 o (MMA=(128,16), MMA_M=1,MMA_K=4,1):((64,1),0,16,0)
        # sB: S<3,4,3> o 0 o (MMA=(64,16), MMA_M=1,MMA_K=4,1):((64,1),0,16,0)
        # sC: S<2,4,3> o 0 o (EPI_M=(8,16), EPI_N=(32,1),(1,1)):((32,256),(1,0),(0,0))
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma=tiled_mma,
            mma_tiler_mnk=mma_tiler_mnk,
            a_dtype=a_dtype,
            num_stages=1,
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma=tiled_mma,
            mma_tiler_mnk=mma_tiler_mnk,
            b_dtype=b_dtype,
            num_stages=1,
        )
        c_smem_layout_staged_one = (
            sm100_utils.make_smem_layout_epi(
                epi_dtype=c_dtype,
                epi_layout=c_layout,
                # selected by `sm100_utils.compute_epilogue_tile_shape` with heuristics
                epi_tile=epi_tile,
                epi_stage=1,
            )
            if use_tma_store
            else None
        )
        
        mbar_helpers_bytes = 1024
        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_stage_one
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        c_bytes_per_stage = (
            cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
            if use_tma_store
            else 0
        )
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B stage
        num_ab_stage = (
            smem_capacity - (occupancy + 1) * (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        if use_tma_store:
            num_c_stage += (
                smem_capacity
                - ab_bytes_per_stage * num_ab_stage
                - (occupancy + 1) * (mbar_helpers_bytes + c_bytes)
            ) // ((occupancy + 1) * c_bytes_per_stage)
        
        if const_expr(debug_print):
            print()
            print(
                f"Computed stages - ACC: {num_acc_stage=}, A/B: {num_ab_stage=}, C: {num_c_stage=}"
            )
            print(f"SMEM capacity: {smem_capacity=}, {occupancy=}")
            print(f"A/B bytes per stage: {ab_bytes_per_stage=}")
            print(f"Reserved bytes for mbar helpers: {(occupancy + 1) * mbar_helpers_bytes=}, {mbar_helpers_bytes=}")
            print(f"Reserved bytes for C stages: {(occupancy + 1) * c_bytes=}, {c_bytes_per_stage=}")
            print()
            
            print()
            print("a_smem_layout_stage_one: ", a_smem_layout_stage_one)
            print("b_smem_layout_staged_one: ", b_smem_layout_staged_one)
            print("c_smem_layout_staged_one: ", c_smem_layout_staged_one)
        
        return num_acc_stage, num_ab_stage, num_c_stage

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> Tuple[int, int, int]:
        """Compute grid shape for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]

        :return: Grid shape for kernel launch.
        :rtype: tuple[int, int, int]
        """

        cluster_shape_mnl = (*cluster_shape_mn, 1)

        grid = cute.round_up(
            (
                cute.ceil_div(c.layout.shape[0], cta_tile_shape_mnk[0]), # pM // CTA_tileM
                cute.ceil_div(c.layout.shape[1], cta_tile_shape_mnk[1]), # pN // CTA_tileN
                c.layout.shape[2], # pL
            ),
            cluster_shape_mnl, # (2, 1, 1)
        )

        return grid

    @staticmethod
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma, mma_tiler_mnk: Tuple[int, int, int]
    ) -> int:
        """
        Compute the number of tensor memory allocation columns.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler: The shape (M, N, K) of the MMA tile.
        :type mma_tiler: tuple[int, int, int]

        :return: The number of tensor memory allocation columns.
        :rtype: int
        """
        
        # tCtAcc_fake layout: (MMA_C=(128,128),1,1):((65536,1),0,0)
        # NOTE: since the 32-bit addr of tmem partitions into high 16-bit for rows and low 16-bit for columns, 
        # so each row has a stride of 64k = 2^16
        acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        # return sm100_utils.get_num_tmem_alloc_cols(tCtAcc_fake) # deprecated API
        
        # NOTE: tmem has 128 rows x 512 cols, 4B in each cell (128 x 512 x 4B = 256KB),
        # and each thread in a warp group will handle each row,
        # so we have to decide how many columns to allocate, in the range of [min_cols=32, max_cols=512] with power of 2,
        # and since tCtAcc_fake has a MMA_C layout of (128,128) with dtype fp32(4B), 
        # it needs 128 columns which satisfies the constraints
        return utils.get_num_tmem_alloc_cols(tCtAcc_fake) # new API

    @staticmethod
    def is_valid_dtypes(
        ab_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes are valid

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        if ab_dtype not in {
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.TFloat32,
            cutlass.Uint8,
            cutlass.Int8,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        if (
            acc_dtype not in {cutlass.Float32, cutlass.Float16, cutlass.Int32}
            or acc_dtype == cutlass.Float16
            and ab_dtype
            not in {cutlass.Float16, cutlass.Float8E4M3FN, cutlass.Float8E5M2}
            or acc_dtype == cutlass.Int32
            and ab_dtype not in {cutlass.Uint8, cutlass.Int8}
        ):
            is_valid = False
        if (
            acc_dtype == cutlass.Float32
            and c_dtype
            not in {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.BFloat16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
                cutlass.Int32,
                cutlass.Int8,
                cutlass.Uint8,
            }
            or acc_dtype == cutlass.Float16
            and c_dtype
            not in {
                cutlass.BFloat16,
                cutlass.Float16,
            }
            or acc_dtype == cutlass.Int32
            and c_dtype
            not in {
                cutlass.BFloat16,
                cutlass.Float16,
                cutlass.Float32,
                cutlass.Int32,
                cutlass.Int8,
                cutlass.Uint8,
            }
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """
        Check if the mma tiler and cluster shape are valid

        :param use_2cta_instrs: Whether to use 2 CTA groups
        :type use_2cta_instrs: bool
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]

        :return: True if the mma tiler and cluster shape are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        # Skip invalid mma tile shape
        if not (
            (not use_2cta_instrs and mma_tiler_mn[0] in [64, 128])
            or (use_2cta_instrs and mma_tiler_mn[0] in [128, 256])
        ):
            is_valid = False
        if mma_tiler_mn[1] not in range(32, 257, 32):
            is_valid = False
        # Skip illegal cluster shape
        if cluster_shape_mn[0] % (2 if use_2cta_instrs else 1) != 0:
            is_valid = False
        # Skip invalid cluster shape
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
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
    def is_valid_epilog_store_option(
        use_2cta_instrs: bool,
        use_tma_store: bool,
        m: int,
        n: int,
        mma_tiler_mn: Tuple[int, int],
    ) -> bool:
        """
        Check if the epilogue store option is valid

        :param use_2cta_instrs: Whether to use 2 CTA groups
        :type use_2cta_instrs: bool
        :param use_tma_store: Whether to use TMA store
        :type use_tma_store: bool
        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]

        :return: True if the epilogue store option is valid, False otherwise
        :rtype: bool
        """

        is_valid = True
        # None TMA store version does not have predication, can not support OOB tiles
        cta_tile_shape_mn = (
            mma_tiler_mn[0] // (2 if use_2cta_instrs else 1),
            mma_tiler_mn[1],
        )
        if not use_tma_store:
            if not (m % cta_tile_shape_mn[0] == 0 and n % cta_tile_shape_mn[1] == 0):
                is_valid = False
        return is_valid

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_tma_store: bool,
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
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param use_2cta_instrs: Whether to use 2 CTA groups
        :type use_2cta_instrs: bool
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]
        :param use_tma_store: Whether to use TMA store
        :type use_tma_store: bool
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
        if not DenseGemmKernelSm100.is_valid_dtypes(ab_dtype, acc_dtype, c_dtype):
            can_implement = False
        # Skip invalid mma tile shape and cluster shape
        if not DenseGemmKernelSm100.is_valid_mma_tiler_and_cluster_shape(
            use_2cta_instrs, mma_tiler_mn, cluster_shape_mn
        ):
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not DenseGemmKernelSm100.is_valid_tensor_alignment(
            m, n, k, l, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        # Skip invalid epilogue store option
        if not DenseGemmKernelSm100.is_valid_epilog_store_option(
            use_2cta_instrs, use_tma_store, m, n, mma_tiler_mn
        ):
            can_implement = False
        return can_implement


def run_dense_gemm(
    mnkl: Tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    use_2cta_instrs: bool,
    use_tma_store: bool,
    tolerance: float,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    measure_launch_overhead=False,
):
    """
    Prepare A/B/C tensors, launch GPU kernel, and reference checking.
    """
    print(f"Running B100 Dense GEMM test with:")
    print(f"mnkl: {mnkl}")
    print(f"AB dtype: {ab_dtype}, C dtype: {c_dtype}, Acc dtype: {acc_dtype}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"2CTA MMA instructions: {'True' if use_2cta_instrs else 'False'}")
    print(f"Use TMA Store: {'True' if use_tma_store else 'False'}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")

    # Unpack parameters
    m, n, k, l = mnkl

    # Skip unsupported testcase
    if not DenseGemmKernelSm100.can_implement(
        ab_dtype,
        acc_dtype,
        c_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        use_tma_store,
        m,
        n,
        k,
        l,
        a_major,
        b_major,
        c_major,
    ):
        raise TypeError(
            f"Unsupported testcase {ab_dtype}, {acc_dtype}, {c_dtype}, {use_2cta_instrs}, {mma_tiler_mn}, {cluster_shape_mn}, {use_tma_store}, {m}, {n}, {k}, {l}, {a_major}, {b_major}, {c_major}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    # Create and permute tensor A/B/C
    def create_and_permute_tensor(
        l, mode0, mode1, is_mode0_major, dtype, is_dynamic_layout=True
    ):
        # is_mode0_major: (l, mode1, mode0) -> (mode0, mode1, l)
        # else: (l, mode0, mode1) -> (mode0, mode1, l)
        shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
        permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
        is_unsigned = dtype in {cutlass.Uint8}
        # Temporarily use uint8 as torch does not support fp8 type
        torch_dtype = (
            cutlass_torch.dtype(dtype)
            if dtype not in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}
            else torch.uint8
        )

        # Create dtype torch tensor (cpu)
        torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
            shape,
            torch_dtype,
            permute_order=permute_order,
            init_type=cutlass_torch.TensorInitType.RANDOM,
            init_config=cutlass_torch.RandomInitConfig(
                min_val=0 if is_unsigned else -2, max_val=4 if is_unsigned else 2
            ),
        )
        # Create dtype torch tensor (gpu)
        torch_tensor = torch_tensor_cpu.cuda()

        # Create f32 torch tensor (cpu)
        f32_torch_tensor = torch_tensor_cpu.to(dtype=torch.float32)

        # Create dtype cute tensor (gpu)
        cute_tensor = from_dlpack(torch_tensor, assumed_align=16)
        cute_tensor.element_type = dtype
        if is_dynamic_layout:
            cute_tensor = cute_tensor.mark_layout_dynamic(
                leading_dim=(0 if is_mode0_major else 1)
            )
        cute_tensor = cutlass_torch.convert_cute_tensor(
            f32_torch_tensor,
            cute_tensor,
            dtype,
            is_dynamic_layout=is_dynamic_layout,
        )

        return f32_torch_tensor, cute_tensor, torch_tensor

    a_ref, a_tensor, a_torch = create_and_permute_tensor(
        l, m, k, a_major == "m", ab_dtype, is_dynamic_layout=True
    )
    b_ref, b_tensor, b_torch = create_and_permute_tensor(
        l, n, k, b_major == "n", ab_dtype, is_dynamic_layout=True
    )
    c_ref, c_tensor, c_torch = create_and_permute_tensor(
        l, m, n, c_major == "m", c_dtype, is_dynamic_layout=True
    )

    # Configure gemm kernel
    gemm = DenseGemmKernelSm100(
        acc_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        use_tma_store,
        debug_print=DEBUG_MODE
    )

    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    # Compile gemm kernel
    compiled_gemm = cute.compile(gemm, a_tensor, b_tensor, c_tensor, stream)

    # Launch GPU kernel
    
    # Warm up
    for i in range(warmup_iterations):
        compiled_gemm(a_tensor, b_tensor, c_tensor, stream)
    
    # Execution
    for i in range(iterations):
        compiled_gemm(a_tensor, b_tensor, c_tensor, stream)

    # Compute reference result
    if not skip_ref_check:
        if ab_dtype in {
            cutlass.Int8,
            cutlass.Uint8,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            ref = torch.einsum("mkl,nkl->mnl", a_ref.cpu(), b_ref.cpu())
        else:
            ref = (torch.einsum("mkl,nkl->mnl", a_ref, b_ref)).cpu()

        # Copy gpu result back
        gpu_c = c_torch.cpu()

        # Convert ref to c_type
        if c_dtype == cutlass.Float32:
            ref_c = ref
        elif c_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}:
            # m major: (l, n, m) -> (m, n, l)
            # n major: (l, m, n) -> (m, n, l)
            permute_order = (1, 2, 0) if c_major == "n" else (2, 1, 0)
            shape = (l, m, n) if c_major == "n" else (l, n, m)
            f8_torch_tensor = cutlass_torch.create_and_permute_torch_tensor(
                shape,
                torch.uint8,
                permute_order=permute_order,
                init_type=cutlass_torch.TensorInitType.SKIP,
            ).cuda()
            # Create dtype cute tensor (gpu)
            ref_c_tensor = from_dlpack(
                f8_torch_tensor, assumed_align=16
            ).mark_layout_dynamic(leading_dim=(1 if c_major == "n" else 0))
            ref_c_tensor.element_type = c_dtype
            ref_c_tensor = cutlass_torch.convert_cute_tensor(
                ref,
                ref_c_tensor,
                c_dtype,
                is_dynamic_layout=True,
            )

            ref_c = f8_torch_tensor.cpu()
        else:
            ref_c = ref.to(cutlass_torch.dtype(c_dtype))

        # Reference checking ref_c and gpu_c
        torch.testing.assert_close(
            gpu_c,
            ref_c,
            atol=tolerance,
            rtol=1e-05,
        )
        
    # Profiling
    profile_mode = os.environ.get("PROFILE_MODE", "0") == "1"
    if profile_mode:
        import sys
        sys.path.insert(0, "..")
        from nvtx import switch_profile, add_nvtx_event
        
        flops = 2 * m * n * k
        event_str = f"{mnkl=} ({flops=})"
        iters, start, end = 10, 6, 9
        for i in range(iters):
            switch_profile(
                iter_id=i,
                start=start,
                end=end,
            )
            
            with add_nvtx_event(event_str):
                compiled_gemm(a_tensor, b_tensor, c_tensor, stream)


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
            # or: return tuple([int(x.strip()) for x in s.split(",")])
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    parser = argparse.ArgumentParser(
        description="Example of MxNxKxL GEMM on Blackwell."
    )

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(256, 256, 512, 1),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--mma_tiler_mn",
        type=parse_comma_separated_ints,
        default=(128, 128),
        help="Mma tiler (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        default=(1, 1),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.TFloat32)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.Float32)
    parser.add_argument("--acc_dtype", type=cutlass.dtype, default=cutlass.Float32)
    parser.add_argument(
        "--use_2cta_instrs",
        action="store_true",
        help="Enable 2CTA MMA instructions feature",
    )
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument(
        "--use_tma_store", action="store_true", help="Use tma store or not"
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )
    parser.add_argument(
        "--warmup_iterations", type=int, default=0, help="Warmup iterations"
    )
    parser.add_argument("--iterations", type=int, default=1, help="Iterations")
    parser.add_argument(
        "--skip_ref_check", action="store_true", help="Skip reference checking"
    )

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    run_dense_gemm(
        args.mnkl,
        args.ab_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.mma_tiler_mn,
        args.cluster_shape_mn,
        args.use_2cta_instrs,
        args.use_tma_store,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
    )
    print("PASS")
