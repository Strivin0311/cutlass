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
using CUTE DSL with compiler generated software pipeline.
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

    python examples/blackwell/dense_gemm_software_pipeline.py                   \
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

    ncu python examples/blackwell/dense_gemm_software_pipeline.py              \
      --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                 \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
      --mnkl 8192,8192,8192,1                                                  \
      --use_tma_store --use_2cta_instrs

Constraints:
* Supported input data types: fp16, bf16, tf32, int8, uint8, fp8 (e4m3fn, e5m2),
  see detailed valid dtype combinations in below PipelinedDenseGemmKernelSm100 class documentation
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


class PipelinedDenseGemmKernelSm100:
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
        >>> gemm = PipelinedDenseGemmKernelSm100(
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
        # K dimension is deferred in _setup_attributes
        self.mma_tiler_mnk = (*mma_tiler_mn, 1) # (tileM256, tileN128)
        self.use_tma_store = use_tma_store

        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.buffer_align_bytes = 1024
        self.occupancy = 1 # we only want one CTA to reside on one SM
        self.threads_per_cta = 128 # one warp group
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100") # 227KB
        
        self.debug_print = debug_print
        
        if const_expr(self.debug_print):
            print()
            print(f"Initialized PipelinedDenseGemmKernelSm100 with configurations:")
            print(f"  Accumulator dtype: {self.acc_dtype=}")
            print(f"  Use 2CTA MMA instructions: {self.use_2cta_instrs=}")
            print(f"  MMA tiler (M, N): {self.mma_tiler_mnk[:2]=}")
            print(f"  Cluster shape (M, N): {self.cluster_shape_mn=}")
            print(f"  Use TMA store for output C: {self.use_tma_store=}")
            print(f"  CTA group for MMA: {self.cta_group=}")
            print(f"  Threads per CTA: {self.threads_per_cta=}")
            print(f"  Shared memory capacity (bytes): {self.smem_capacity=}")
            print(f"  Buffer alignment (bytes): {self.buffer_align_bytes=}")
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
        # Configure tiled mma
        # Thr Layout VMNK: (2,1,1,1):(1,0,0,0)
        # Shape MNK:       (256,128,16)
        # TV Layout A:     (2,(128,16)):(128,(1,256)) => sliced along M
        # TV Layout B:     (2,(64,16)):(64,(1,128)) => distributed along N
        # TV Layout C:     (2,(128,128)):(128,(1,256)) => sliced along M
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mnk[:2],
        )
        
        self.tiled_mma = tiled_mma
        self.atom_thr_id = tiled_mma.thr_id
        self.atom_thr_shape = self.atom_thr_id.shape
        self.atom_thr_size = cute.size(self.atom_thr_shape)

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler_mnk = ( # (tileM256, tileN128, tileK64)
            self.mma_tiler_mnk[0],
            self.mma_tiler_mnk[1],
            mma_inst_shape_k * mma_inst_tile_k, # 16 x 4 = 64
        )
        self.cta_tile_shape_mnk = ( # (CTA_tileM128, CTA_tileN128, CTA_tileK64)
            self.mma_tiler_mnk[0] // self.atom_thr_size, # 256 / 2 = 128
            self.mma_tiler_mnk[1],
            self.mma_tiler_mnk[2],
        )

        # Compute cluster layout
        #
        # ─────────────────────────────────────────────────────────────────────────────
        #  CuTe Layout Divide API Cheatsheet
        # ─────────────────────────────────────────────────────────────────────────────
        #
        #  All four variants divide a target layout/tensor by a tiler (tile shape).
        #  Given target shape (M=8, N=6) and tiler (4, 3):
        #    - inner tile  = (4, 3)   ← elements within one tile
        #    - outer tiles = (2, 2)   ← how many tiles fit (8/4, 6/3)
        #
        #  ┌─────────────────┬──────────────────────────────────┬────────────────────────────────────┐
        #  │  API            │  Result shape                    │  Semantics / typical use           │
        #  ├─────────────────┼──────────────────────────────────┼────────────────────────────────────┤
        #  │ logical_divide  │ ((4,2), (3,2))                   │ Each mode split into (tile, rest)  │
        #  │                 │                                  │ in-place; lowest-level algebra.    │
        #  ├─────────────────┼──────────────────────────────────┼────────────────────────────────────┤
        #  │ zipped_divide   │ ((4,3), (2,2))                   │ mode[0] = all intra-tile coords    │
        #  │                 │ [inner, outer]                   │ mode[1] = all inter-tile coords    │
        #  │                 │                                  │ → TV layout for partition_S/D      │
        #  ├─────────────────┼──────────────────────────────────┼────────────────────────────────────┤
        #  │ tiled_divide    │ ((2,2), (4,3))                   │ mode[0] = inter-tile (which tile)  │
        #  │                 │ [outer, inner]                   │ mode[1] = intra-tile (elem in tile)│
        #  │                 │                                  │ → cluster layout, local_tile coord │
        #  ├─────────────────┼──────────────────────────────────┼────────────────────────────────────┤
        #  │ flat_divide     │ (4, 3, 2, 2)                     │ intra-tile modes flattened first,  │
        #  │                 │ [t0,t1,..., r0,r1,...]           │ then inter-tile modes appended;    │
        #  │                 │                                  │ → epilog subtile split (tAcc_epi)  │
        #  └─────────────────┴──────────────────────────────────┴────────────────────────────────────┘
        #
        #  Example (flat_divide, used for epilog accumulator tiling):
        #    tAcc shape: (128, 128)  epi_tile: (128, 32)
        #    flat_divide → (128, 32, 1, 4)
        #                   ────────────  ────
        #                   intra-tile    inter: EPI_M=1, EPI_N=4
        #
        #  Example (tiled_divide, used here for cluster layout):
        #    cluster_shape_mn = (2, 1),  thr_id.shape = (2,)  [2-CTA pair]
        #    make_layout((2,1,1)) tiled_divide by (2,) →
        #      shape = ((1,1,1), (2,))   i.e. mode[0]=(V=1,M=1,N=1) mode[1]=(thr=2)
        #    → cluster_layout_vmnk.shape[1] = 1  (no B-multicast along M)
        #      cluster_layout_vmnk.shape[2] = 1  (no A-multicast along N)
        # ─────────────────────────────────────────────────────────────────────────────
        self.cluster_layout_vmnk = cute.tiled_divide( # (CTA_V(2), CTA_M1, CTA_N1, CTA_K1):((1),0,0,0)
            cute.make_layout((*self.cluster_shape_mn, 1)), # (2, 1, 1)
            (self.atom_thr_shape,), # (2,)
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2]) # along N dim
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1]) # along M dim
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
        if cutlass.const_expr(self.use_tma_store):
            self.epi_tile = sm100_utils.compute_epilogue_tile_shape( # (epi_tileM128:1, epi_tileN32:1)
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
            debug_print=self.debug_print,
        )

        # Compute A/B/C shared memory layout
        # sA: S<3,4,3> o 0 o (MMA=(128,16),MMA_M=1,MMA_K=4,MMA_STAGES=8):((64,1),0,16,8192)
        # sB: S<3,4,3> o 0 o (MMA=(64,16),MMA_N=1,MMA_K=4,MMA_STAGES=8):((64,1),0,16,4096)
        # sC: S<2,4,3> o 0 o (epi_tileM=(8,16), epi_tileN=(32,1), epi_stages=(1,2)):((32,256),(1,0),(0,4096))
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
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols( # 128 cols
            tiled_mma, self.mma_tiler_mnk
        )

        if const_expr(self.debug_print):
            print()
            print(f"Setup attributes dependent on GEMM inputs:")
            print(f"  MMA tiler (M, N, K): {self.mma_tiler_mnk=}")
            print(f"  CTA tile shape (M, N, K): {self.cta_tile_shape_mnk=}")
            print(f"  Cluster layout: {self.cluster_layout_vmnk=}")
            print(f"  Number of multicast CTAs for A: {self.num_mcast_ctas_a=}")
            print(f"  Number of multicast CTAs for B: {self.num_mcast_ctas_b=}")
            print(f"  Epilogue tile shape: {self.epi_tile=}")
            print(f"  Number of accumulator stages: {self.num_acc_stage=}")
            print(f"  Number of A/B stages: {self.num_ab_stage=}")
            print(f"  Number of C stages: {self.num_c_stage=}")
            print(f"  Number of tensor memory allocation columns: {self.num_tmem_alloc_cols=}")
            print()
            
            print()
            print(f"A SMEM layout (a_smem_layout_staged) (MMA,MMA_M,MMA_K,STAGE): {self.a_smem_layout_staged}")
            print(f"B SMEM layout (b_smem_layout_staged) (MMA,MMA_N,MMA_K,STAGE): {self.b_smem_layout_staged}")
            print(f"C SMEM layout (c_smem_layout_staged) (MMA,MMA_M,MMA_N,STAGE): {self.c_smem_layout_staged}")
            print()
            
            print()
            print("self.tiled_mma: ", tiled_mma, f"\n\nshape_mnk: {tiled_mma.shape_mnk}", f"thr_id.shape: {self.atom_thr_shape}")
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
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        tiled_mma = self.tiled_mma
        atom_thr_size = self.atom_thr_size

        # /////////////////////////////////////////////////////////////////////////////
        #  Setup TMA load for A
        # /////////////////////////////////////////////////////////////////////////////
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            cluster_shape_mnk=self.cluster_shape_mn, # (2, 1)
            atom_thr_id=self.atom_thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        a_smem_size = cute.cosize(self.a_smem_layout_staged.outer)
        
        # tma_atom_a: Src: (2,8192):(8192,1) | Dst: (2,8192):(8192,1), where CTA_tileM128 x tileK64 = 8192
        # tma_tensor_a: (pM=2048, pK=1024,1):(1@1,1@0,1@2)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            op=a_op,
            gmem_tensor=a,
            smem_layout=a_smem_layout,
            mma_tiler_mnk=self.mma_tiler_mnk,
            tiled_mma=tiled_mma,
            cluster_shape_vmnk=self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if a.element_type is cutlass.Float32 else None
            ),
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Setup TMA load for B
        # /////////////////////////////////////////////////////////////////////////////
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            cluster_shape_mnk=self.cluster_shape_mn, 
            atom_thr_id=self.atom_thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        b_smem_size = cute.cosize(self.b_smem_layout_staged.outer)
        
        # tma_atom_b: Src: (2,4096):(4096,1) | Dst: (2,4096):(4096,1), where CTA_tileN64 x tileK64 = 4096
        # tma_tensor_b: (pN=4096, pK=1024,1):(1@1,1@0,1@2)
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

        # /////////////////////////////////////////////////////////////////////////////
        #  Setup store for C
        # /////////////////////////////////////////////////////////////////////////////
        tma_atom_c, tma_tensor_c = None, None
        c_smem_size, c_cta_v_layout = 0, None
        if cutlass.const_expr(self.use_tma_store):
            c_op = cpasync.CopyBulkTensorTileS2GOp()
            c_cta_v_layout = cute.composition( # (128,32):(1@0,1@1), col-major
                cute.make_identity_layout(c.shape), self.epi_tile
            )
            
            epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
            c_smem_size = cute.cosize(self.c_smem_layout_staged.outer)
            
            # tma_atom_c: Src: (1,4096):(0,1) | Dst: (1,4096):(0,1), where epi_tileM128 x epi_tileN32 = 4096
            # tma_tensor_c: (pM2048, pN4096,1):(1@1,1@0,1@2)
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                op=c_op,
                gmem_tensor=c,
                smem_layout=epi_smem_layout,
                cta_tiler=c_cta_v_layout, # it's ok to just pass in `self.epi_tile`
            )

        # /////////////////////////////////////////////////////////////////////////////
        #  Compute grid size
        # /////////////////////////////////////////////////////////////////////////////
        
        # NOTE: the number of TMA load bytes (tx_count for main pipeline) combines the sA and sB size one stage, 
        # and needs to times the CTA-pair number since we need all the sA, sB data loaded for both CTAs to start the umma 
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size # tx_count
        
        grid = self._compute_grid(c, self.cta_tile_shape_mnk, self.cluster_shape_mn) # (pM/tileM=16,pN/tileN=32,pL=1)

        # /////////////////////////////////////////////////////////////////////////////
        #  Define shared storage for kernel
        # /////////////////////////////////////////////////////////////////////////////
        @cute.struct
        class SharedStorage:
            # mainloop full/empty mbar array ptrs for each ab stage
            ab_full_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            
            # tmem accumulation full mbar for each acc stage
            # NOTE: we don't need empty mbar since `self.num_acc_stage` is fixed to 1
            # and the epilogue is synchronized with the mainloop, so we only need one full signal
            # to notify the acc consumers to start T2R copy
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            
            # the mbar ptr to synchronize all threads in two CTAs before issuing tmem deallocation
            tmem_dealloc_mbar_ptr: cutlass.Int64
            
            # the smem buffer to hold the allocated tmem address
            tmem_holding_smem_buf: cutlass.Int32
            
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, a_smem_size,
                ],
                self.buffer_align_bytes,
            ]
            
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, b_smem_size,
                ],
                self.buffer_align_bytes,
            ]
            
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, c_smem_size,
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        
        if const_expr(self.debug_print):
            print()
            print(f"{self.a_dtype=}, {self.b_dtype=}, {self.c_dtype=}, {self.a_major_mode=}, {self.b_major_mode=}, {self.c_layout=}")
            print(f"{c_smem_size=}, {c_cta_v_layout=}")
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

        # /////////////////////////////////////////////////////////////////////////////
        #  Launch the kernel
        # /////////////////////////////////////////////////////////////////////////////            
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
        #  Prefetch tma descriptor
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(self.use_tma_store):
                cpasync.prefetch_descriptor(tma_atom_c)

        # /////////////////////////////////////////////////////////////////////////////
        #  Setup cta/thread coordinates
        # /////////////////////////////////////////////////////////////////////////////
        # Coords inside cluster
        mma_tile_coord_v = bidx % self.atom_thr_size # CTA idx in the CTA-pair
        is_leader_cta = mma_tile_coord_v == 0 c
        cta_rank_in_cluster = cute.arch.make_warp_uniform( # CTA idx in the cluster, which might be different from mma_tile_coord_v if cluster size > 2
            cute.arch.block_idx_in_cluster()
        )
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
        num_tma_producer = 1
        ab_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, size=num_tma_producer
        )
        num_tma_consumer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, size=num_tma_consumer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=ab_full_empty_mbar_ptr,
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        ab_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_ab_stage
        )
        ab_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_ab_stage
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Initialize acc_pipeline (barrier) and states
        # /////////////////////////////////////////////////////////////////////////////
        num_acc_producer = 1
        acc_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, size=num_acc_producer
        )
        num_acc_consumer = self.threads_per_cta
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, size=num_acc_consumer
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=acc_full_mbar_ptr,
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        acc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_acc_stage
        )
        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_acc_stage
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

        # /////////////////////////////////////////////////////////////////////////////
        #  Setup smem tensor A/B/C
        # /////////////////////////////////////////////////////////////////////////////
        
        # (MMA=(128,16), MMA_M=1, MMA_K=4, STAGE=8) => M-sliced within CTA-pair
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA=(64,16), MMA_N=1, MMA_K=4, STAGE=8) => N-shared within CTA-pair
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
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask( # 0b(10), only for itself
                cta_layout_vmnk=cluster_layout_vmnk, 
                cta_coord_vmnk=block_in_cluster_coord_vmnk,
                mcast_mode=2 # along N dim
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask( # 0b(10), only for itself
                cta_layout_vmnk=cluster_layout_vmnk, 
                cta_coord_vmnk=block_in_cluster_coord_vmnk, 
                mcast_mode=1 # along M dim
            )
            
            if const_expr(self.debug_print):
                if is_thread0:
                    cute.printf("")
                    cute.printf("a_full_mcast_mask: {}, b_full_mcast_mask: {}", a_full_mcast_mask, b_full_mcast_mask)
                    cute.printf("")

        # /////////////////////////////////////////////////////////////////////////////
        #  Local_tile partition global tensors
        # /////////////////////////////////////////////////////////////////////////////
        
        # (tileM=256, tileK=64, restM=8, restK=16, restL=1)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler_mnk, (None, 0, None)), (None, None, None)
        )
        # (tileN=128, tileK=64, restN=32, restK=16, restL=1)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler_mnk, (0, None, None)), (None, None, None)
        )
        # (tileM=256, tileN=128, restM=8, restN=32, restL=1)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler_mnk, (None, None, 0)), (None, None, None)
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
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v) # slice with CTA-pair idx
        
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
        a_cta_layout = cute.make_layout( # (1):(0)
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        
        # tAsA: ((TMA_atom_v, rest_v)=(8192,1), PIPE=8)
        # tAgA: ((TMA_atom_v, rest_v)=((64,128),1), RestM=8, RestK=16, RestL=1)
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
        
        # tBsB: ((TMA_atom_v, rest_v)=(4096,1), PIPE=8)
        # tBgB: ((TMA_atom_v, rest_v)=(((64,64),1), RestN=32, RestK=16, RestL=1)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("a_cta_layout: {}", a_cta_layout)
                cute.printf("tAsA: {}", tAsA)
                cute.printf("tAgA: {}", tAgA)
                cute.printf("")
                
                cute.printf("")
                cute.printf("b_cta_layout: {}", b_cta_layout)
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
        # (MMA=(128,128), MMA_M=1, MMA_N=1):((65536,1),0,0)
        acc_shape = thr_mma.partition_shape_C(self.mma_tiler_mnk[:2])
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

        # Bar sync for retrieve tensor memory ptr from shared memory
        cute.arch.barrier()

        # /////////////////////////////////////////////////////////////////////////////
        #  Retrieving tensor memory ptr and make accumulator tensor
        # /////////////////////////////////////////////////////////////////////////////
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_smem_buf
        )
        # (MMA=(128, 128), MMA_M=1, MMA_N=1):((65536,1),0,0)
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
        # tiled_copy_t2r: 
        #   layout_src_tv: (32,1024):(0,1) | layout_src_tv_tiled: ((32,4),((32,32),1)):((0,1),((128,4),0))
        #   layout_dst_tv: (32,32):(32,1) | layout_dst_tv_tiled: ((32,4),(32,1)):((4,1),(128,0))
        # tTR_tAcc: (((32,32),1),1,1,1,4):(((1,65536),0),0,0,0,32)
        # tTR_rAcc: ((32,1),1,1):((1,0),0,0)
        tiled_copy_t2r, tTR_tAcc, tTR_rAcc = self.epilog_tmem_copy_and_partition(
            tidx, 
            tAcc=tCtAcc, 
            tCgC=tCgC, 
            epi_tile=epi_tile,
            use_2cta_instrs=use_2cta_instrs,
        )

        tTR_rC, tiled_copy_r2s, simt_atom = None, None, None
        tRS_rC, tRS_sC, bSG_sC, bSG_gC, tTR_gC = None, None, None, None, None
        if cutlass.const_expr(self.use_tma_store):
            # Make R2S tiled copy
            # tiled_copy_r2s:
            #   layout_src_tv: (1,1):(0,0) | layout_src_tv_tiled: ((32,4),(1,32)):((4,1),(0,128))
            #   layout_dst_tv: (1,1):(0,0) | layout_dst_tv_tiled: ((32,4),(1,32)):((4,1),(0,128))
            # tTR_rC: ((32,1),1,1):((1,0),0,0)
            # tRS_rC: ((1,32),1,1):((0,1),0,0)
            # tRS_sC: (R2S=(1,32),1,1,epi_stages=(1,2)):((0,1),0,0,(0,4096))
            # tTR_rC = cute.make_fragment(tTR_rAcc.shape, self.c_dtype) # deprecated API
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype) # new API, the bf16 version of tTR_rAcc
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, tidx, sC
            )
            
            # Make S2G TMA tiled copy
            # bSG_sC: (TMA=(4096,1), epi_stages=(1,2)):((1,0),(0,4096))
            # bSG_gC: (TMA=((32,128),1),EPI_M=1, EPI_N=4, RestM=8, RestN=32, RestL=1):(((1@0,1@1),0),0,32@0,256@1,128@0,1@2)
            tma_atom_c, bSG_sC, bSG_gC = self.epilog_gmem_copy_and_partition(
                tidx, tma_atom_c, tCgC, epi_tile, sC
            )
        else:
            # Make R2G tiled copy
            simt_atom, tTR_rC, tTR_gC = self.epilog_gmem_copy_and_partition(
                tidx, tiled_copy_t2r, tCgC, epi_tile, sC
            )
            
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("tiled_copy_t2r: layout_src_tv: {} | layout_src_tv_tiled: {} | layout_dst_tv: {} | layout_dst_tv_tiled: {}", tiled_copy_t2r.layout_src_tv, tiled_copy_t2r.layout_src_tv_tiled, tiled_copy_t2r.layout_dst_tv, tiled_copy_t2r.layout_dst_tv_tiled)
                cute.printf("tTR_tAcc: {}", tTR_tAcc)
                cute.printf("tTR_rAcc: {}", tTR_rAcc)
                cute.printf("")
                
                cute.printf("")
                if cutlass.const_expr(self.use_tma_store):
                    cute.printf("tiled_copy_r2s: layout_src_tv: {} | layout_src_tv_tiled: {} | layout_dst_tv: {} | layout_dst_tv_tiled: {}", tiled_copy_r2s.layout_src_tv, tiled_copy_r2s.layout_src_tv_tiled, tiled_copy_r2s.layout_dst_tv, tiled_copy_r2s.layout_dst_tv_tiled)
                    cute.printf("tTR_rC: {}", tTR_rC)
                    cute.printf("tRS_rC: {}", tRS_rC)
                    cute.printf("tRS_sC: {}", tRS_sC)
                    cute.printf("bSG_sC: {}", bSG_sC)
                    cute.printf("bSG_gC: {}", bSG_gC)
                else:
                    cute.printf("simt_atom: {}", simt_atom)
                    cute.printf("tTR_rC: {}", tTR_rC)
                    cute.printf("tTR_gC: {}", tTR_gC)
                cute.printf("")

        # /////////////////////////////////////////////////////////////////////////////
        #  Slice to per mma tile index
        # /////////////////////////////////////////////////////////////////////////////
        
        # ((TMA_atom_v, rest_v)=((64,128),1), RestK=16)
        tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        # ((TMA_atom_v, rest_v)=(((64,64),1), RestK=16)
        tBgB = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
        if cutlass.const_expr(self.use_tma_store):
            # ((ATOM_V, REST_V)=((32,128),1), EPI_M=1, EPI_N=4)
            bSG_gC = bSG_gC[(None, None, None, *mma_tile_coord_mnl)]
        else:
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
            tTR_gC = tTR_gC[(None, None, None, None, None, *mma_tile_coord_mnl)]
            
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

        # ///////////////////////////////////////////////////////////////////////////////
        #  Mainloop: Pipelining TMA load A/B and UMMA
        # ///////////////////////////////////////////////////////////////////////////////
        prefetch_stages = self.num_ab_stage - 2
        prefetch_k_tile_cnt = cutlass.min(prefetch_stages, k_tile_cnt)
        
        if warp_idx == 0: # tma-load producer, as well as umma consumer/t2r producer if leader CTA
            # NOTE: we can just pass in `prefetch_stages` argument, 
            # to allow producer prefetching automatically, and no need to peek the states
            # which makes code much neater and less error-prone, as we don't need to manually maintain the prefetching logic and states
            # and what's better, it proves to a little bit better performance than the manual one, probably because of better scheduling flexibility with the automatic prefetching
            for k_tile in cutlass.range(k_tile_cnt, prefetch_stages=prefetch_stages):
                # /////////////////////////////////////////////////////////////////////////////
                # TMA Producer
                # /////////////////////////////////////////////////////////////////////////////
                
                # Wait for current empty mbar to be arrived by the consumer
                ab_pipeline.producer_acquire(ab_producer_state)

                #  TMA load A/B
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, ab_producer_state.count)],
                    tAsA[(None, ab_producer_state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    mcast_mask=a_full_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, ab_producer_state.count)],
                    tBsB[(None, ab_producer_state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    mcast_mask=b_full_mcast_mask,
                )
                
                # /////////////////////////////////////////////////////////////////////////////
                # UMMA Consumer
                # /////////////////////////////////////////////////////////////////////////////

                if is_leader_cta:
                    # Wait for current full mbar to be arrived by the producer
                    ab_pipeline.consumer_wait(ab_consumer_state)

                    # tCtAcc += tCrA * tCrB
                    num_kblocks = cute.size(tCrA, mode=[2])
                    for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                        kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[kblock_coord],
                            tCrB[kblock_coord],
                            tCtAcc,
                        )
                        
                        # Enable accumulate on tCtAcc after first kblock
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    # Arrive the empty mbar with `tcgen05.commit.mbarrier::arrive::one`
                    ab_pipeline.consumer_release(ab_consumer_state)

                ab_producer_state.advance()
                ab_consumer_state.advance()

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

        # tTR_tAcc: (T2R=((T2R_COLS=32, T2R_ROWS=32),1), T2R_M=1, T2R_N=1, (EPI_M, EPI_N)=(1,4))
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3]) # EPI_M x EPI_N = 4
        if cutlass.const_expr(self.use_tma_store):
            # bSG_gC: ((ATOM_V, REST_V)=((32,128),1), (EPI_M, EPI_N)=(1,4))
            bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
        else:
            # tTR_gC: (T2R, T2R_M, T2R_N, (EPI_M, EPI_N))
            tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))
            
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
        if cutlass.const_expr(self.use_tma_store):
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
        for subtile_idx in range(subtile_cnt):
            # T2R copy to store accumulator from tmem to rmem
            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

            if cutlass.const_expr(self.use_tma_store):
                # Perform epilogue op on accumulator and convert to C type
                acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                tRS_rC.store(acc_vec)

                # R2S copy to store C from rmem to smem
                c_stage_idx = subtile_idx % self.num_c_stage
                cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_stage_idx)])
                
                # Fence and barrier to make sure shared memory store is visible to TMA store
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                cute.arch.barrier()

                # S2G TMA store C from smem to gmem
                if warp_idx == 0:
                    cute.copy(
                        tma_atom_c,
                        bSG_sC[(None, c_stage_idx)],
                        bSG_gC[(None, subtile_idx)],
                    )
                    
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
                    tmem_dealloc_mbar_ptr, 
                    peer_cta_rank_in_cluster=cta_rank_in_cluster ^ 1 # peer rank
                )
                # Wait for self CTA's dealloc mbar to be arrived by the peer CTA
                cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
            
            # Deallocate the tmem buffer using `tcgen05.dealloc.cta_group::2.sync.aligned.b32`
            cute.arch.dealloc_tmem(
                tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs
            )

        # /////////////////////////////////////////////////////////////////////////////
        #  Wait for C store complete
        # /////////////////////////////////////////////////////////////////////////////
        if cutlass.const_expr(self.use_tma_store):
            c_pipeline.producer_tail() # `cp.async.bulk.wait_group(0)`

        # /////////////////////////////////////////////////////////////////////////////
        #  Wait A/B buffer empty
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == 0:
            # Reverse prefetch_k_tile_cnt times to next available buffer
            for i in range(prefetch_k_tile_cnt):
                ab_producer_state.reverse()
            
            ab_pipeline.producer_tail(ab_producer_state)

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        tCgC: cute.Tensor,
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
        
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0)],
            epi_tile,
        )
        
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_mnl_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None, None)], epi_tile
        )
        
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        
        # (T2R, T2R_M, T2R_N)
        # tTR_rAcc = cute.make_fragment( # deprecated API
        tTR_rAcc = cute.make_rmem_tensor(
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
        
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        
        if cutlass.const_expr(self.use_tma_store):
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
        else:
            tiled_copy_t2r = atom
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_gC = thr_copy_t2r.partition_D(gC_epi)
            # (T2R, T2R_M, T2R_N)
            tTR_rC = cute.make_fragment(
                tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.c_dtype
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
        # Default ACC stages
        num_acc_stage = 1
        # Default C stages
        num_c_stage = 2 if use_tma_store else 0

        # Calculate smem layout and size for one stage of A, B, and C
        # sA_stage_one: S<3,4,3> o 0 o (MMA=(128,16),MMA_M=1,MMA_K=4,MMA_STAGE=1):((64,1),0,16,0)
        # sB_stage_one: S<3,4,3> o 0 o (MMA=(64,16),MMA_N=1,MMA_K=4,MMA_STAGE=1):((64,1),0,16,0)
        # sC_stage_one: S<2,4,3> o 0 o (epi_tileM=(8,16), epi_tileN=(32,1), epi_stages=(1,1)):((32,256),(1,0),(0,0))
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
        c_smem_layout_staged_one = (
            sm100_utils.make_smem_layout_epi(
                c_dtype,
                c_layout,
                epi_tile,
                epi_stage=1,
            )
            if use_tma_store
            else None
        )
        
        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_stage_one
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        
        mbar_helpers_bytes = 1024
        
        c_bytes_per_stage = (
            cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
            if use_tma_store
            else 0
        )
        c_bytes = c_bytes_per_stage * num_c_stage
        
        # TODO(REVIEW): why use occupancy + 1 ?
        occ_factor = (occupancy + 1)

        # Calculate A/B stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B stage
        num_ab_stage = (
            smem_capacity # TODO(REVIEW): why not divide capacity with occupancy ?
            - occ_factor * (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        if use_tma_store:
            num_c_stage += (
                smem_capacity
                - ab_bytes_per_stage * num_ab_stage
                - occ_factor * (mbar_helpers_bytes + c_bytes)
            ) // (occ_factor * c_bytes_per_stage)
        
        if const_expr(debug_print):
            print()
            print("a_smem_layout_stage_one: ", a_smem_layout_stage_one)
            print("b_smem_layout_staged_one: ", b_smem_layout_staged_one)
            if use_tma_store:
                print("c_smem_layout_staged_one: ", c_smem_layout_staged_one)
            print(f"Bytes per A/B stage: {ab_bytes_per_stage=}")
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
                cute.ceil_div(c.layout.shape[0], cta_tile_shape_mnk[0]),
                cute.ceil_div(c.layout.shape[1], cta_tile_shape_mnk[1]),
                c.layout.shape[2],
            ),
            cluster_shape_mnl,
        )

        return grid

    @staticmethod
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma, mma_tiler: Tuple[int, int, int]
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
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        # num_tmem_cols = sm100_utils.get_num_tmem_alloc_cols(tCtAcc_fake) # deprecated API
        num_tmem_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        
        return num_tmem_cols

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
        if not PipelinedDenseGemmKernelSm100.is_valid_dtypes(ab_dtype, acc_dtype, c_dtype):
            can_implement = False
        # Skip invalid mma tile shape and cluster shape
        if not PipelinedDenseGemmKernelSm100.is_valid_mma_tiler_and_cluster_shape(
            use_2cta_instrs, mma_tiler_mn, cluster_shape_mn
        ):
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not PipelinedDenseGemmKernelSm100.is_valid_tensor_alignment(
            m, n, k, l, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        # Skip invalid epilogue store option
        if not PipelinedDenseGemmKernelSm100.is_valid_epilog_store_option(
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
):
    """
    Prepare A/B/C tensors, launch GPU kernel, and reference checking.
    """
    print(f"Running B100 software pipeline Dense GEMM test with:")
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
    if not PipelinedDenseGemmKernelSm100.can_implement(
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
    gemm = PipelinedDenseGemmKernelSm100(
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
