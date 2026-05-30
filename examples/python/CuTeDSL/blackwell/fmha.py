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
import enum
import math
import os
import time
from typing import Type, Tuple

import torch
import torch.nn.functional as F
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.cute.testing as testing
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Int64, Float32, Boolean

# ---------------------------------------------------------------------------
# Debug / profile mode switches
#   DEBUG_MODE=1   -> prints host-side compile info; device-side kernel prints
#                     from designated warp 0 / thread 0 in first tile only
#   PROFILE_MODE=1 -> wraps a dedicated benchmark loop with NVTX range markers
#                     so that nsys / ncu can isolate exactly the kernel launch
# ---------------------------------------------------------------------------
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"
PROFILE_MODE = os.environ.get("PROFILE_MODE", "0") == "1"

"""
A fused multi-head attention (FMHA) example for the NVIDIA Blackwell SM100 architecture using CUTE DSL

This example demonstrates an implementation of fused multi-head attention using a TMA + Blackwell SM100
TensorCore warp-specialized persistent kernel. The implementation integrates the Q*K^T matrix multiplication,
softmax normalization, and softmax(Q*K^T)*V into a single kernel, avoiding intermediate data movement between
global memory and shared memory, thus improving computational efficiency.

The kernel implements key optimizations including:
- Warp specialization for different computation phases (load, MMA, softmax, correction, epilogue)
- Pipeline stages between different warps for overlapping computation and memory access
- Support for different precision data types
- Optional causal masking for autoregressive models

To run this example:

.. code-block:: bash

    python examples/blackwell/fmha.py                                     \
      --qk_acc_dtype Float32 --pv_acc_dtype Float32                       \
      --mma_tiler_mn 128,128                                              \
      --q_shape 4,1024,8,64 --k_shape 4,1024,8,64                         \
      --is_persistent

The above example runs FMHA with batch size 4, sequence length 1024, 8 attention heads, and head
dimension 64. The Blackwell tcgen05 MMA tile shape is (128, 128), and the kernel uses fp16 for input/output
with fp32 for accumulation.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/blackwell/fmha.py                                 \
      --qk_acc_dtype Float32 --pv_acc_dtype Float32                       \
      --mma_tiler_mn 128,128                                              \
      --q_shape 4,1024,8,64 --k_shape 4,1024,8,64                         \
      --is_persistent --warmup_iterations 10                              \
      --iterations 10 --skip_ref_check

Constraints for this example:
* Supported head dimensions: 32, 64, and 128
* Number of heads in Q must be divisible by number of heads in K
* mma_tiler_mn must be 128,128
* Batch size must be the same for Q, K, and V tensors
* For causal masking, use --is_causal (note: specify without =True/False)
* For persistent scheduling, use --is_persistent (note: specify without =True/False)
"""

class FmhaStaticTileSchedulerParams:
    def __init__(
        self,
        is_persistent: bool,
        problem_shape_mbh: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        self.is_persistent = is_persistent
        self.problem_shape_mbh = problem_shape_mbh
        self._loc = loc
        self._ip = ip

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.is_persistent, self.problem_shape_mbh]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.is_persistent, self.problem_shape_mbh], self._values_pos
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return FmhaStaticTileSchedulerParams(*(tuple(obj_list)), loc=self._loc)


def create_fmha_static_tile_scheduler_params(
    is_persistent: bool,
    problem_shape_mbh: cute.Shape,
) -> FmhaStaticTileSchedulerParams:
    return FmhaStaticTileSchedulerParams(is_persistent, problem_shape_mbh)

class FmhaStaticTileScheduler:

    def __init__(
        self,
        params: FmhaStaticTileSchedulerParams,
        current_work_linear_idx: Int32,
        blk_coord: cute.Coord,
        grid_shape: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        self._params = params
        self._blk_coord = blk_coord
        self._grid_shape = grid_shape
        self._is_persistent = params.is_persistent
        self._current_work_linear_idx = current_work_linear_idx
        self._problem_shape_mbh = cute.make_layout(
            params.problem_shape_mbh, loc=loc, ip=ip
        )
        self._num_blocks = cute.size(self._problem_shape_mbh, loc=loc, ip=ip)
        self._is_first_block = True
        self.num_persistent_sm = cute.size(grid_shape, loc=loc, ip=ip)
        self._loc = loc
        self._ip = ip

    # called by host
    @staticmethod
    def get_grid_shape(
        params: FmhaStaticTileSchedulerParams,
        *,
        loc=None,
        ip=None,
    ) -> cute.Shape:
        if params.is_persistent:
            hardware_info = cutlass.utils.HardwareInfo()
            sm_count = hardware_info.get_device_multiprocessor_count()
            return (
                cutlass.min(
                    sm_count, cute.size(params.problem_shape_mbh, loc=loc, ip=ip)
                ),
                1,
                1,
            )
        else:
            return params.problem_shape_mbh

    @staticmethod
    def check_valid_work_for_seqlen_q(
        q_tiler: int,
        current_idx: Int32,
        seqlen_q: Int32,
    ) -> Boolean:
        return current_idx * q_tiler < seqlen_q

    def get_current_work(self, *, loc=None, ip=None) -> utils.WorkTileInfo:
        is_valid = (
            self._current_work_linear_idx < self._num_blocks # idx < m * b * h
            if self._is_persistent
            else self._is_first_block
        )

        blk_coord = (0, 0, 0)
        if self._is_persistent:
            # de-linearize the block idx to get the actual tile coordinate (midx, hidx, bidx)
            blk_coord = self._problem_shape_mbh.get_hier_coord(
                self._current_work_linear_idx, loc=loc, ip=ip
            )
        else:
            blk_coord = self._blk_coord

        # cur_tile_coord is (midx, 0, (hidx, bidx)), 0 for dummy nidx
        cur_tile_coord = (
            blk_coord[0],
            0,
            (blk_coord[1], blk_coord[2]),
        )

        return utils.WorkTileInfo(cur_tile_coord, is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, advance_count=1, loc=None, ip=None):
        if self._is_persistent:
            self._current_work_linear_idx += advance_count * self.num_persistent_sm
        self._is_first_block = False

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self._params)
        values.extend(cutlass.extract_mlir_values(self._current_work_linear_idx))
        values.extend(cutlass.extract_mlir_values(self._blk_coord))
        values.extend(cutlass.extract_mlir_values(self._grid_shape))
        return values

    def __new_from_mlir_values__(self, values):
        assert len(values) == 10
        new_params = cutlass.new_from_mlir_values(self._params, values[0:3])
        new_current_work_linear_idx = cutlass.new_from_mlir_values(
            self._current_work_linear_idx, [values[3]]
        )
        new_blk_coord = cutlass.new_from_mlir_values(self._blk_coord, values[4:7])
        new_grid_shape = cutlass.new_from_mlir_values(self._grid_shape, values[7:])
        return FmhaStaticTileScheduler(
            new_params, new_current_work_linear_idx, new_blk_coord, new_grid_shape
        )


def create_fmha_static_tile_scheduler(
    params: FmhaStaticTileSchedulerParams,
    blk_coord: cute.Coord,
    grid_shape: cute.Shape,
) -> FmhaStaticTileScheduler:
    return FmhaStaticTileScheduler(params, blk_coord[0], blk_coord, grid_shape)


class MaskType(enum.Enum):
    NO_MASK = enum.auto()
    RESIDUAL_MASK = enum.auto()
    CAUSAL_MASK = enum.auto()


class BlackwellFusedMultiHeadAttentionForward:
    def __init__(
        self,
        qk_acc_dtype: Type[cutlass.Numeric],
        pv_acc_dtype: Type[cutlass.Numeric],
        mma_tiler: Tuple[int, int, int],
        is_persistent: bool,
        mask_type: MaskType,
        debug_print: bool = False,
    ):
        """Initializes the configuration for a Blackwell Fused Multi-Head Attention (FMHA) kernel.

        This configuration includes several key aspects:

        1.  Data Type Settings:
            - qk_acc_dtype: Data type for Q*K^T matrix multiplication accumulator
            - pv_acc_dtype: Data type for P*V matrix multiplication accumulator

        2.  MMA Instruction Settings:
            - mma_tiler: The (M, N, K) shape of the MMA instruction unit
            - qk_mma_tiler: MMA shape for Q*K^T computation
            - pv_mma_tiler: MMA shape for P*V computation

        3.  Kernel Execution Mode:
            - is_persistent: Boolean indicating whether to use persistent kernel mode
            - mask_type: Specifies the type of mask to use (no mask, residual mask, or causal mask)

        :param qk_acc_dtype: Data type for Q*K^T matrix multiplication accumulator
        :type qk_acc_dtype: Type[cutlass.Numeric]
        :param pv_acc_dtype: Data type for P*V matrix multiplication accumulator
        :type pv_acc_dtype: Type[cutlass.Numeric]
        :param mma_tiler: The (M, N, K) shape of the MMA instruction
        :type mma_tiler: Tuple[int, int, int]
        :param is_persistent: Whether to use persistent kernel mode
        :type is_persistent: bool
        :param mask_type: Type of mask to use
        :type mask_type: MaskType
        """

        self.qk_acc_dtype = qk_acc_dtype
        self.pv_acc_dtype = pv_acc_dtype
        self.cta_tiler = (
            2 * mma_tiler[0],  # 2 Q tile per CTA (Q0, Q1)
            mma_tiler[1],
            mma_tiler[2],
        )
        self.qk_mma_tiler = mma_tiler # (tileQ128, tileK128, tileD128)
        self.pv_mma_tiler = ( # (tileP128, tileD128, tileV128)
            mma_tiler[0],
            mma_tiler[2],
            mma_tiler[1],
        )
        self.cluster_shape_mn = (1, 1)
        self.is_persistent = is_persistent
        self.mask_type = mask_type
        
        self.softmax0_warp_ids = (0, 1, 2, 3) # warp group 0
        self.softmax1_warp_ids = (4, 5, 6, 7) # warp group 1
        self.correction_warp_ids = (8, 9, 10, 11) # warp group 2
        
        # warp group 3
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.epilogue_warp_id = 14
        self.empty_warp_id = 15
        
        # all columns to be allocated in tmem
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.epilogue_warp_id,
                self.empty_warp_id,
            )
        )

        self.cta_sync_bar_id = 0
        self.tmem_alloc_sync_bar_id = 1

        # tmem column offsets
        self.tmem_s0_offset = 0
        self.tmem_s1_offset = 128
        self.tmem_o0_offset = 256
        self.tmem_o1_offset = 384
        self.tmem_p0_offset = 32
        self.tmem_p1_offset = 160

        # vec buffer for row_max & row_sum
        self.tmem_vec0_offset = 0
        self.tmem_vec1_offset = 128

        # Set specific register usage for each warp role
        self.num_regs_softmax = 192
        self.num_regs_correction = 96
        self.num_regs_other = 32
        self.num_regs_empty = 24

        self.buffer_align_bytes = 1024

        num_warps_per_warpgroup = 4
        self.softmax_warpgroup_count = (
            len((*self.softmax0_warp_ids, *self.softmax1_warp_ids))
            // num_warps_per_warpgroup
        )
        self.debug_print = debug_print
        
        if self.debug_print:
            print()
            print("Initialized BlackwellFusedMultiHeadAttentionForward with the following configuration:")
            print(f"  qk_acc_dtype: {self.qk_acc_dtype}")
            print(f"  pv_acc_dtype: {self.pv_acc_dtype}")
            print(f"  mma_tiler: {self.qk_mma_tiler} for Q*K^T, {self.pv_mma_tiler} for P*V")
            print(f"  mask_type: {self.mask_type}")
            print(f"  threads_per_cta: {self.threads_per_cta}")
            print(f"  softmax_warpgroup_count: {self.softmax_warpgroup_count}")
            print()
            

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        self.q_stage = 2
        self.kv_stage = 4 if self.q_dtype.width == 8 else 3
        self.acc_stage = 1
        self.softmax_corr_stage = 1
        self.mma_corr_stage = 2
        self.mma_softmax_stage = 1
        self.epi_stage = 2

    @cute.jit
    def __call__(
        self,
        q_iter: cute.Pointer,
        k_iter: cute.Pointer,
        v_iter: cute.Pointer,
        o_iter: cute.Pointer,
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32, Int32],
        cum_seqlen_q: cute.Tensor | None,
        cum_seqlen_k: cute.Tensor | None,
        scale_softmax_log2: Float32,
        scale_output: Float32,
        stream: cuda.CUstream,
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        This method prepares the input tensors for processing, validates their shapes and types,
        configures the computation parameters, and launches the CUDA kernel.

        The method handles:
        1. Tensor layout transformations for specific memory access patterns
        2. Validation of tensor shapes and data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch with appropriate parameters

        :param q_iter: The query tensor pointer
        :type q_iter: cute.Pointer
        :param k_iter: The key tensor pointer
        :type k_iter: cute.Pointer
        :param v_iter: The value tensor pointer
        :type v_iter: cute.Pointer
        :param o_iter: The output tensor pointer
        :type o_iter: cute.Pointer
        :param problem_size: The problem size with shape [b, s_q, s_k, h_q, h_k, d]. If cum_seqlen_q or cum_seqlen_k is not None, s_q and s_k are the max of the cumulative sequence length respectively.
        :type problem_size: Tuple[Int32, Int32, Int32, Int32, Int32, Int32]
        :param cum_seqlen_q: The cumulative sequence length tensor for query
        :type cum_seqlen_q: cute.Tensor | None
        :param cum_seqlen_k: The cumulative sequence length tensor for key
        :type cum_seqlen_k: cute.Tensor | None
        :param scale_softmax_log2: The log2 scale factor for softmax
        :type scale_softmax_log2: Float32
        :param scale_output: The scale factor for the output
        :type scale_output: Float32
        :param stream: The CUDA stream to execute the kernel on
        :type stream: cuda.CUstream
        :raises TypeError: If tensor data types don't match or aren't supported
        :raises RuntimeError: If tensor layouts aren't in supported formats
        """
        b, s_q, s_k, h_q, h_k, d = problem_size
        h_r = h_q // h_k
        qo_offset = 0 if cum_seqlen_q is None else -s_q * d * h_r * h_k
        kv_offset = 0 if cum_seqlen_k is None else -s_k * d * h_k
        b_qo = b if cum_seqlen_q is None else s_q * (1 + b)
        b_kv = b if cum_seqlen_k is None else s_k * (1 + b)
        stride_b_qo = h_r * h_k * s_q * d if cum_seqlen_q is None else d * h_r * h_k
        stride_b_kv = h_k * s_k * d if cum_seqlen_k is None else d * h_k

        # (s, d, ((h_r, h_k), b))
        q_layout = cute.make_layout(
            (s_q, d, ((h_r, h_k), b_qo)),
            stride=(d * h_r * h_k, 1, ((d, d * h_r), stride_b_qo)),
        )
        q = cute.make_tensor(q_iter + qo_offset, q_layout)
        # (s, d, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        k_layout = cute.make_layout(
            (s_k, d, ((h_r, h_k), b_kv)),
            stride=(d * h_k, 1, ((0, d), stride_b_kv)),
        )
        k = cute.make_tensor(k_iter + kv_offset, k_layout)
        # (d, s, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        # NOTE: we transpose V to Vt here to align O = PV = umma(P, Vt)
        v_layout = cute.make_layout(
            (d, s_k, ((h_r, h_k), b_kv)),
            stride=(1, d * h_k, ((0, d), stride_b_kv)),
        )
        v = cute.make_tensor(v_iter + kv_offset, v_layout)
        # (s, d, ((h_r, h_k), b))
        o_layout = cute.make_layout(
            (s_q, d, ((h_r, h_k), b_qo)),
            stride=(d * h_r * h_k, 1, ((d, d * h_r), stride_b_qo)),
        )
        o = cute.make_tensor(o_iter + qo_offset, o_layout)
        
        if cutlass.const_expr(self.debug_print):
            cute.printf("")
            cute.printf("Tensor configurations:")
            cute.printf("q_layout: {}", q_layout)
            cute.printf("k_layout: {}", k_layout)
            cute.printf("v_layout: {}", v_layout)
            cute.printf("o_layout: {}", o_layout)
            cute.printf("")
        
        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type

        self.tile_sched_params, grid = self._compute_grid(
            o_shape=cute.shape((s_q, d, ((h_r, h_k), b))),
            cta_tiler=self.cta_tiler, # (M128, K128)
            is_persistent=self.is_persistent,
        )

        self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.o_layout = utils.LayoutEnum.from_tensor(o)

        if cutlass.const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN): # NOTE: here is actually Vt
            raise RuntimeError("The layout of v is not supported")

        # check type consistency
        if cutlass.const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if cutlass.const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        
        self._setup_attributes()

        # Use 1 cta instead of 2
        cta_group = tcgen05.CtaGroup.ONE
        
        # NOTE: the intermediate tensor p is from tmem & k-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        
        # Thr Layout VMNK: (1,1,1,1):(0,0,0,0)
        # Permutation MNK: (_,_,_)
        # MMA Atom
        # ThrID:           1:0
        # Shape MNK:       (128,128,16)
        # TV Layout A:     (1,(128,16)):(128,(1,128))
        # TV Layout B:     (1,(128,16)):(128,(1,128))
        # TV Layout C:     (1,(128,128)):(128,(1,128))
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group=cta_group,
            mma_tiler_mn=self.qk_mma_tiler[:2],
            a_source=tcgen05.OperandSource.SMEM,
        )
        
        # Thr Layout VMNK: (1,1,1,1):(0,0,0,0)
        # Permutation MNK: (_,_,_)
        # MMA Atom
        # ThrID:           1:0
        # Shape MNK:       (128,128,16)
        # TV Layout A:     (1,(128,16)):(128,(1,128))
        # TV Layout B:     (1,(128,16)):(128,(1,128))
        # TV Layout C:     (1,(128,128)):(128,(1,128))
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group=cta_group,
            mma_tiler_mn=self.pv_mma_tiler[:2], # (M128, K128)
            a_source=p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1) # (1, 1, 1)
        self.cluster_layout_vmnk = cute.tiled_divide( # ((1),1,1,1):((0),0,0,0)
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        self.epi_tile = self.pv_mma_tiler[:2] # (M128, K128)

        # sQ: S<3,4,3> o 0 o (MMA_sA=(128,16), RestQ1, RestD=(4,2), PipeQ2):((64,1),0,(16,8192),16384)
        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )
        # sK: S<3,4,3> o 0 o (MMA_sB=(128,16), RestK1, RestD=(4,2), PipeKV3):((64,1),0,(16,8192),16384)
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.k_dtype,
            self.kv_stage,
        )
        # tP: S<3,4,3> o 0 o (MMA_tA=(128,16), RestP1, RestD=(4,2), PipeAcc1):((64,1),0,(16,8192),0)
        p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.q_dtype,
            self.acc_stage,
        )
        # sV: S<3,4,3> o 0 o (MMA_sB=((64,2),16), RestV1, RestD8, PipeKV3):(((1,8192),64),0,1024,16384)
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.v_dtype,
            self.kv_stage,
        )
        # sO: S<3,4,3> o 0 o (EPI_O=(8,16), EPI_D=(64,2), PipeEPI=(1,2)):((64,512),(1,8192),(0,16384))
        o_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.o_dtype,
            self.o_layout,
            self.epi_tile,
            self.epi_stage,
        )

        # TMA load op for QKV
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
        
        # TMA store op for O
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        # Make tma load atom/tensor for QKV
        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        # tma_atom_q: layout_src_tv=(1,8192):(0,1), layout_dst_tv=(1,8192):(0,1)
        # tma_tensor_q: (pQ2048, pD128, HB=((1,4),2)):(1@1,1@0,((1@2,1@3),1@4))
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            op=tma_load_op,
            gmem_tensor=q,
            smem_layout=q_smem_layout,
            mma_tiler_mnk=self.qk_mma_tiler,
            tiled_mma=qk_tiled_mma,
            cluster_shape_vmnk=self.cluster_layout_vmnk.shape,
        )
        
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        # tma_atom_k: layout_src_tv=(1,8192):(0,1), layout_dst_tv=(1,8192):(0,1)
        # tma_tensor_k: ((pK4096, pD128, HB=((1,4),2)):(1@1,1@0,((0,1@2),1@3))
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            op=tma_load_op,
            gmem_tensor=k,
            smem_layout=k_smem_layout,
            mma_tiler_mnk=self.qk_mma_tiler,
            tiled_mma=qk_tiled_mma,
            cluster_shape_vmnk=self.cluster_layout_vmnk.shape,
        )

        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        # tma_atom_v: layout_src_tv=(1,8192):(0,1), layout_dst_tv=(1,8192):(0,1)
        # tma_tensor_v: (pD128, pV4096, HB=((1,4),2)):(1@0,1@1,((0,1@2),1@3))
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            op=tma_load_op,
            gmem_tensor=v,
            smem_layout=v_smem_layout,
            mma_tiler_mnk=self.pv_mma_tiler,
            tiled_mma=pv_tiled_mma,
            cluster_shape_vmnk=self.cluster_layout_vmnk.shape,
        )

        o_cta_v_layout = cute.composition( # (128,128):(1@0,1@1)
            cute.make_identity_layout(o.shape), self.epi_tile # (M128, K128)
        )
        o_smem_layout = cute.select(o_smem_layout_staged, mode=[0, 1])
        # tma_atom_o: layout_src_tv=(1,8192):(0,1), layout_dst_tv=(1,8192):(0,1)
        # tma_tensor_o: (pO2048, pD128, HB=((1,4),2)):(1@1,1@0,((1@2,1@3),1@4))
        tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op=tma_store_op,
            gmem_tensor=o,
            smem_layout=o_smem_layout,
            cta_tiler=o_cta_v_layout,
        )

        q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        k_copy_size = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        self.tma_copy_q_bytes = q_copy_size
        self.tma_copy_kv_bytes = k_copy_size

        @cute.struct
        class SharedStorage:
            # Pipeline mbarriers
            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]
            load_kv_mbar_ptr: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mma_s0_mbar_ptr: cute.struct.MemRange[Int64, self.mma_softmax_stage * 2]
            mma_s1_mbar_ptr: cute.struct.MemRange[Int64, self.mma_softmax_stage * 2]
            s0_corr_mbar_ptr: cute.struct.MemRange[Int64, self.softmax_corr_stage * 2]
            s1_corr_mbar_ptr: cute.struct.MemRange[Int64, self.softmax_corr_stage * 2]
            s0_s1_sequence_mbar_ptr: cute.struct.MemRange[
                Int64, self.softmax_warpgroup_count
            ]
            corr_epi_mbar_ptr: cute.struct.MemRange[Int64, self.epi_stage * 2]
            mma_corr_mbar_ptr: cute.struct.MemRange[Int64, self.mma_corr_stage * 2]
            
            # Tmem dealloc mbarrier
            # the mbar ptr to synchronize all threads in the CTA before issuing tmem deallocation
            tmem_dealloc_mbar_ptr: Int64
            
            # Tmem holding buffer ptr
            # the smem buffer ptr to hold the allocated tmem address
            tmem_holding_smem_buf: Int32
            
            # Smem tensors Q/K/V/O
            # NOTE: V shares the same smem buf with K
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, cute.cosize(o_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[ # sK / sV
                cute.struct.MemRange[self.k_dtype, cute.cosize(k_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        if cutlass.const_expr(self.debug_print):
            print()
            print("[__call__] qk_tiled_mma: ", qk_tiled_mma)
            print("[__call__] pv_tiled_mma: ", pv_tiled_mma)
            print()
            print(f"[__call__] cluster_shape_mnk: {self.cluster_shape_mnk} | cluster_layout_vmnk: {self.cluster_layout_vmnk} | cta_group: {cta_group}")
            print(f"[__call__] q_major_mode: {self.q_major_mode}  k_major_mode: {self.k_major_mode}  v_major_mode: {self.v_major_mode}  o_layout: {self.o_layout}")
            print(f"[__call__] q_smem_layout_staged: {q_smem_layout_staged}")
            print(f"[__call__] k_smem_layout_staged: {k_smem_layout_staged}")
            print(f"[__call__] p_tmem_layout_staged: {p_tmem_layout_staged}")
            print(f"[__call__] v_smem_layout_staged: {v_smem_layout_staged}")
            print(f"[__call__] o_smem_layout_staged: {o_smem_layout_staged}")
            print(f"[__call__] o_cta_v_layout: {o_cta_v_layout} | epi_tile: {self.epi_tile}")
            print(f"[__call__] q_stage={self.q_stage}  kv_stage={self.kv_stage}  acc_stage={self.acc_stage}  mma_softmax_stage={self.mma_softmax_stage}")
            print(f"[__call__] softmax_corr_stage={self.softmax_corr_stage}  mma_corr_stage={self.mma_corr_stage}  epi_stage={self.epi_stage}")
            print()
            
            cute.printf("")
            cute.printf("[__call__] tma_atom_q: layout_src_tv={}, layout_dst_tv={}", tma_atom_q.layout_src_tv, tma_atom_q.layout_dst_tv)
            cute.printf("[__call__] tma_tensor_q.layout: {}", tma_tensor_q.layout)
            cute.printf("[__call__] tma_atom_k: layout_src_tv={}, layout_dst_tv={}", tma_atom_k.layout_src_tv, tma_atom_k.layout_dst_tv)
            cute.printf("[__call__] tma_tensor_k.layout: {}", tma_tensor_k.layout)
            cute.printf("[__call__] tma_atom_v: layout_src_tv={}, layout_dst_tv={}", tma_atom_v.layout_src_tv, tma_atom_v.layout_dst_tv)
            cute.printf("[__call__] tma_tensor_v.layout: {}", tma_tensor_v.layout)
            cute.printf("[__call__] tma_atom_o: layout_src_tv={}, layout_dst_tv={}", tma_atom_o.layout_src_tv, tma_atom_o.layout_dst_tv)
            cute.printf("[__call__] tma_tensor_o.layout: {}", tma_tensor_o.layout)
            cute.printf("")
            cute.printf("[__call__] tma_copy_q_bytes: {}", self.tma_copy_q_bytes)
            cute.printf("[__call__] tma_copy_kv_bytes: {}", self.tma_copy_kv_bytes)
            cute.printf("[__call__] grid: {}", grid) # (min(num_sm, pM//tileM * nh * batch), 1, 1)
            cute.printf("")

        # Launch the kernel synchronously
        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_o,
            tma_tensor_o,
            cum_seqlen_q,
            cum_seqlen_k,
            scale_softmax_log2,
            scale_output,
            q_smem_layout_staged,
            k_smem_layout_staged,
            p_tmem_layout_staged,
            v_smem_layout_staged,
            o_smem_layout_staged,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        mO_qdl: cute.Tensor,
        cum_seqlen_q: cute.Tensor | None,
        cum_seqlen_k: cute.Tensor | None,
        scale_softmax_log2: Float32,
        scale_output: Float32,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        p_tmem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: FmhaStaticTileSchedulerParams,
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.

        :param qk_tiled_mma: Tiled MMA for Q*K^T
        :type qk_tiled_mma: cute.TiledMma
        :param pv_tiled_mma: Tiled MMA for P*V
        :type pv_tiled_mma: cute.TiledMma
        :param tma_atom_q: TMA copy atom for query tensor
        :type tma_atom_q: cute.CopyAtom
        :param mQ_qdl: Partitioned query tensor
        :type mQ_qdl: cute.Tensor
        :param tma_atom_k: TMA copy atom for key tensor
        :type tma_atom_k: cute.CopyAtom
        :param mK_kdl: Partitioned key tensor
        :type mK_kdl: cute.Tensor
        :param tma_atom_v: TMA copy atom for value tensor
        :type tma_atom_v: cute.CopyAtom
        :param mV_dkl: Partitioned value tensor
        :type mV_dkl: cute.Tensor
        :param tma_atom_o: TMA copy atom for output tensor
        :type tma_atom_o: cute.CopyAtom
        :param mO_qdl: Partitioned output tensor
        :type mO_qdl: cute.Tensor
        :param scale_softmax_log2: The log2 scale factor for softmax
        :type scale_softmax_log2: Float32
        :param scale_output: The scale factor for the output
        :type scale_output: Float32
        :param q_smem_layout_staged: Shared memory layout for query tensor
        :type q_smem_layout_staged: cute.ComposedLayout
        :param k_smem_layout_staged: Shared memory layout for key tensor
        :type k_smem_layout_staged: cute.ComposedLayout
        :param p_tmem_layout_staged: Tensor memory layout for probability matrix
        :type p_tmem_layout_staged: cute.ComposedLayout
        :param v_smem_layout_staged: Shared memory layout for value tensor
        :type v_smem_layout_staged: cute.ComposedLayout
        :param o_smem_layout_staged: Shared memory layout for output tensor
        :type o_smem_layout_staged: cute.ComposedLayout
        :param tile_sched_params: Scheduling parameters for work distribution
        :type tile_sched_params: FmhaStaticTileSchedulerParams
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        
        # intra CTA coord
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        
        # inter CTA coord
        # dummy coord/layout since we do not use TMA multicast or 2-CTA cooperative
        # for this example
        cta_coord = 0
        cta_layout = cute.make_layout(1)

        # used only for debug print
        is_print_block = (bidx == 0) and (bidy == 0) and (bidz == 0) # first block
        is_print_thread = (tidx == 127) and is_print_block # the last thread in first warp

        if cutlass.const_expr(self.debug_print):
            if is_print_thread:
                cute.printf("")
                cute.printf("[kernel] warp_ids: load={} mma={} epi={} softmax0=0..3 softmax1=4..7 corr=8..11",
                            self.load_warp_id, self.mma_warp_id, self.epilogue_warp_id)
                cute.printf("[kernel] scale_softmax_log2 = {}  scale_output = {}",
                            scale_softmax_log2, scale_output)
                cute.printf("")

        # ///////////////////////////////////////////////////////////////////////////////
        # Prefetch tma desc
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)

        # ///////////////////////////////////////////////////////////////////////////////
        # Alloc smem storage and fetch data
        # ///////////////////////////////////////////////////////////////////////////////
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        load_q_mbar_ptr = storage.load_q_mbar_ptr.data_ptr()
        load_kv_mbar_ptr = storage.load_kv_mbar_ptr.data_ptr()
        mma_s0_mbar_ptr = storage.mma_s0_mbar_ptr.data_ptr()
        mma_s1_mbar_ptr = storage.mma_s1_mbar_ptr.data_ptr()
        s0_corr_mbar_ptr = storage.s0_corr_mbar_ptr.data_ptr()
        s1_corr_mbar_ptr = storage.s1_corr_mbar_ptr.data_ptr()
        s0_s1_sequence_mbar_ptr = storage.s0_s1_sequence_mbar_ptr.data_ptr()
        corr_epi_mbar_ptr = storage.corr_epi_mbar_ptr.data_ptr()
        mma_corr_mbar_ptr = storage.mma_corr_mbar_ptr.data_ptr()
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_smem_buf = storage.tmem_holding_smem_buf

        # ///////////////////////////////////////////////////////////////////////////////
        # Make pipelines
        # ///////////////////////////////////////////////////////////////////////////////
        #
        # Pipeline overview (7 pipelines coordinating 5 warp roles):
        #
        #   load_warp ──[load_q]──► mma_warp ──[mma_s0]──► softmax0_wg
        #   load_warp ──[load_kv]──► mma_warp ──[mma_s1]──► softmax1_wg
        #                            mma_warp ──[mma_corr]──► correction_wg
        #   softmax0_wg ──[s0_corr]──► correction_wg
        #   softmax1_wg ──[s1_corr]──► correction_wg
        #   softmax0_wg ──[s0_s1_sequence]──► softmax1_wg
        #   correction_wg ──[corr_epi]──► epilogue_warp
        #
        # Pipeline semantics (full/empty mbar meaning):
        #
        #   load_q  (TmaUmma):
        #     full  = "sQ[stage] written by TMA, mma_warp can issue UMMA reading sQ"
        #     empty = "mma_warp finished reading sQ[stage], load_warp can overwrite"
        #
        #   load_kv  (TmaUmma):
        #     full  = "sK[stage]/sV[stage] written by TMA, mma_warp can issue UMMA"
        #     empty = "mma_warp finished reading sK/sV[stage], load_warp can overwrite"
        #
        #   mma_s0/s1  (UmmaAsync):
        #     full  = "tmem S0/S1 ready (UMMA Q*K done), softmax0/1 can T2R load and process"
        #     empty = "softmax0/1 finished R2T store P0/P1, mma_warp can do P*V using tmem"
        #
        #   s0/s1_corr  (Async, multi-stage):
        #     full  = "vec0/vec1 in tmem written by softmax (old_max+new_max or row_sum+global_max),
        #              correction_wg can T2R load the vec and compute rescale factor"
        #     empty = "correction_wg finished reading vec, softmax can reuse this vec slot"
        #     NOTE:   each softmax_step produces 2 commits per KV-block:
        #               commit-1: old_max + new_max (before exp2, unblocks correction rescale)
        #               commit-2: row_sum + global_max (after all KV-blocks, unblocks epilog)
        #
        #   mma_corr  (UmmaAsync):
        #     full  = "tmem O0/O1 partial result written by UMMA (P*V done),
        #              correction_wg can T2T rescale tOtO in-place"
        #     empty = "correction_wg finished rescale, mma_warp can accumulate next P*V"
        #
        #   corr_epi  (Async):
        #     full  = "sO[stage] written by correction epilog, epilogue_warp can TMA store"
        #     empty = "epilogue_warp finished TMA store, correction can reuse sO slot"
        #
        #   s0_s1_sequence  (Async, 1-stage):
        #     full  = "softmax0 finished exp2 for current KV-block, softmax1 can start exp2"
        #     empty = "softmax1 finished exp2, softmax0 can proceed to next KV-block exp2"
        #     NOTE:   serializes exp2+R2T-store between softmax0 and softmax1 to avoid
        #             contention on tmem write port bandwidth
        #
        # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        # Data-flow timeline. Time flows left → right. Vertical alignment = same point in time. Sub-tasks of one step wrap downward under their column.
        # Notation: [op]=op  (p.↑)=commit(full)  (p.↓)=release(empty)  >>p=wait-full  <<p=wait-empty
        # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        #
        #               ◄───────────── PROLOGUE ──────────────────────────────────────────────────────►◄─────────── MAINLOOP (per KV-block i≥1) ──────────────────────────────────────────────────────────►◄──────────────── EPILOGUE ─────────────────────►
        #
        #  load_warp:   <<[TMA Q0→sQ]─(lq.↑Q0)  ──>  <<[TMA Q1→sQ]─(lq.↑Q1)                          <<[TMA Ki→sK]─(lkv.↑Ki)                                                                             (advance tile scheduler)
        #                             <<[TMA K0→sK]─(lkv.↑K0)       <<[TMA V0→sV]─(lkv.↑V0)                              <<[TMA Vi→sV]─(lkv.↑Vi)
        #
        #  mma_warp:    >>(lq.Q0,lkv.K0)              >>(lq.Q1)                   >>(lkv.V0,ms0.↓)     >>(lkv.Ki)          >>(lkv.V(i-1),mc.↓)         >>(lkv.Vi,mc.↓)                                    [Q0*Klast→S0]─(ms0.↑)
        #               [Q0*K0→S0]─(ms0.↑)─(lq.↓Q0)  [Q1*K0→S1]─(ms1.↑)         [P0*V0→O0]─(mc.↑O0)  [Q0*Ki→S0]─(ms0.↑) [P1*V(i-1)→O1]─(mc.↑O1)    [P0*Vi→O0]─(mc.↑O0)─(lkv.↓)                       [Q1*Klast→S1]─(ms1.↑)
        #                                              ─(lq.↓Q1,lkv.↓K0)          ─(lkv.↓V0)                               [Q1*Ki→S1]─(ms1.↑)                                                              s0_handle.commit / s1_handle.commit
        #
        #  softmax0:                                                                                     >>(ms0)             >>(s0s1.acq)                (hold sc0 slot, accumulate row_sum)                 [R2T vec0=(sum,gmax)]─(sc0.↑#2)
        #                                                                                                [T2R S0]─[row_max]  [exp2]─[→bf16]─(s0s1.↑)                                                        sc0.acq(empty step) ── ms0.↓last
        #                                                                                                [R2T vec0=(old,new)]─(sc0.↑#1)  [R2T P0]─(fence)─(ms0.↓)
        #
        #  softmax1:                                                                                     >>(ms1)             >>(s0s1, wait smx0 exp2)    (hold sc1 slot, accumulate row_sum)                 [R2T vec1=(sum,gmax)]─(sc1.↑#2)
        #                                                                                                [T2R S1]─[row_max]  [exp2]─[→bf16]─(s0s1.↓)
        #                                                                                                [R2T vec1=(old,new)]─(sc1.↑#1)  [R2T P1]─(fence)─(ms1.↓)
        #
        #  correction:  >>(sc0#1, skip)                                                                 >>(sc0#1)           >>(mc.O0)                    >>(sc1#1)           >>(mc.O1)                       >>(sc0#2) / >>(mc.O0)
        #               >>(sc1#1, hold vec1_handle)                                                     [T2R vec0]          [T2R O0, O0*=s0, R2T O0]     [T2R vec1]          [T2R O1, O1*=s1, R2T O1]        [T2R O0,O0*=(sout/sum0),R2T→sO0]─(ce.↑O0)
        #               >>(mc.O0, skip rescale)                                                         scale0=exp2(Δmax0)  ─(sc1_prev.↓, mc_O0.↓)       scale1=exp2(Δmax1)  ─(sc0_curr.↓, mc_O1.↓)         >>(sc1#2) / >>(mc.O1)
        #                                                                                                                                                                                                      [T2R O1,O1*=(sout/sum1),R2T→sO1]─(ce.↑O1)
        #
        #  epilogue:                                                                                                                                                                                           >>(ce.O0)─[TMA sO0→gmem]─(ce.↓)
        #                                                                                                                                                                                                      >>(ce.O1)─[TMA sO1→gmem]─(ce.↓)
        #
        # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

        # Load Q pipeline:
        #   producer: load warp loading Q from gmem to smem with tma
        #   consumer: mma warp loading Q from smem and do Q*K^T
        #   full  = sQ[stage] written by TMA
        #   empty = mma_warp finished reading sQ[stage]
        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=len([self.load_warp_id])),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=len([self.mma_warp_id])),
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=load_q_mbar_ptr,
        ).make_participants()
        
        # Load KV pipeline:
        #   producer: load warp loading K/V from gmem to smem with tma
        #   consumer: mma warp loading K/V from smem and do Q*K^T / P*V
        #   full  = sK[stage]/sV[stage] written by TMA
        #   empty = mma_warp finished reading sK/sV[stage]
        load_kv_producer, load_kv_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.kv_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=len([self.load_warp_id])),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=len([self.mma_warp_id])),
            tx_count=self.tma_copy_kv_bytes,
            barrier_storage=load_kv_mbar_ptr,
        ).make_participants()
        
        # MMA_S0/S1 = Q*K^T pipeline:
        #   producer: mma warp writing S0/S1 to tmem via UMMA Q*K^T
        #   consumer: softmax0/1 warpgroup T2R-loading S, doing softmax, R2T-storing P
        #   full  = tmem S0/S1 ready (UMMA done), softmax can T2R load
        #   empty = tmem P0/P1 ready (softmax R2T store done), mma can do P*V
        mma_s0_producer, mma_s0_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_softmax_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=len([self.mma_warp_id])),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=self.threads_per_warp * len(self.softmax0_warp_ids)),
            barrier_storage=mma_s0_mbar_ptr,
        ).make_participants()
        mma_s1_producer, mma_s1_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_softmax_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=len([self.mma_warp_id])),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=self.threads_per_warp * len(self.softmax1_warp_ids)),
            barrier_storage=mma_s1_mbar_ptr,
        ).make_participants()
        
        # Softmax-to-correction vec pipeline (s0_corr / s1_corr):
        #   producer: softmax0/1 warpgroup R2T-storing row-wise stats into tmem vec region
        #   consumer: correction warpgroup T2R-loading vec to compute rescale factor
        #   full  = tmem vec0/vec1 written by softmax (old_max+new_max, or row_sum+global_max)
        #   empty = correction finished reading vec, softmax can reuse this slot
        #   NOTE: each softmax_step commits twice per KV-block:
        #           commit-1 (before exp2): old_max + new_max  -> unblocks correction_rescale
        #           commit-2 (after all KV-blocks): row_sum + global_max -> unblocks epilog
        s0_corr_producer, s0_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.softmax_corr_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=self.threads_per_warp * len(self.softmax0_warp_ids)),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=self.threads_per_warp * len(self.correction_warp_ids)),
            barrier_storage=s0_corr_mbar_ptr,
        ).make_participants()
        s1_corr_producer, s1_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.softmax_corr_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=self.threads_per_warp * len(self.softmax1_warp_ids)),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=self.threads_per_warp * len(self.correction_warp_ids)),
            barrier_storage=s1_corr_mbar_ptr,
        ).make_participants()
        
        # Correction-to-epilogue pipeline:
        #   producer: correction warpgroup writing final O0/O1 (fp16/bf16) into smem sO
        #   consumer: epilogue warp TMA-storing sO to gmem
        #   full  = sO[0/1] ready in smem, epilogue can TMA store
        #   empty = epilogue finished TMA store, correction can reuse sO slot
        corr_epi_producer, corr_epi_consumer = pipeline.PipelineAsync.create(
            num_stages=self.epi_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=self.threads_per_warp * len(self.correction_warp_ids)),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=self.threads_per_warp * len([self.epilogue_warp_id])),
            barrier_storage=corr_epi_mbar_ptr,
        ).make_participants()
        
        # MMA-to-correction O pipeline:
        #   producer: mma warp writing partial O0/O1 to tmem via UMMA P*V
        #   consumer: correction warpgroup T2T-rescaling tOtO in-place
        #   full  = tmem O0/O1 partial result written by UMMA (P*V done), correction can rescale
        #   empty = correction finished rescale, mma can accumulate next P*V into tOtO
        mma_corr_producer, mma_corr_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_corr_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=len([self.mma_warp_id])),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=self.threads_per_warp * len(self.correction_warp_ids)),
            barrier_storage=mma_corr_mbar_ptr,
        ).make_participants()
        
        # Softmax sequence pipeline (s0_s1_sequence):
        #   producer: softmax0 warpgroup signals after finishing exp2+type-convert for one KV-block
        #   consumer: softmax1 warpgroup waits before starting exp2 for the same KV-block
        #   full  = softmax0 finished exp2, softmax1 can start exp2
        #   empty = softmax1 finished exp2+R2T-store, softmax0 can proceed to next block
        #   NOTE: serializes exp2+R2T-store between softmax0 and softmax1 to avoid
        #         contention on the tmem write port (both write to different P0/P1 regions
        #         but share the same physical tmem write bandwidth)
        s0_s1_sequence_producer, s0_s1_sequence_consumer = (
            pipeline.PipelineAsync.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=self.threads_per_warp * len(self.softmax0_warp_ids)),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=self.threads_per_warp * len(self.softmax1_warp_ids)),
                barrier_storage=s0_s1_sequence_mbar_ptr,
            ).make_participants()
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Make tile scheduler
        # ///////////////////////////////////////////////////////////////////////////////

        tile_sched = create_fmha_static_tile_scheduler(
            tile_sched_params, 
            blk_coord=cute.arch.block_idx(), 
            grid_shape=cute.arch.grid_dim()
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Tensor memory dealloc barrier init
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.empty_warp_id:
            cute.arch.mbarrier_init(
                tmem_dealloc_mbar_ptr,
                cnt=self.threads_per_warp
                * len(
                    ( # softmax warp group and correction warp group needs to arrive this mbar
                        *self.softmax0_warp_ids,
                        *self.softmax1_warp_ids,
                        *self.correction_warp_ids,
                    )
                ),
            )
        cute.arch.mbarrier_init_fence()

        # /////////////////////////////////////////////////////////////////////////////
        #  Generate smem tensor Q/K/V/O
        # /////////////////////////////////////////////////////////////////////////////
        
        # sQ: S<3,4,3> o 0 o (MMA_sA=(128,16), MMA_Q1, MMA_D=(4,2), PipeQ2):((64,1),0,(16,8192),16384)
        sQ = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        # sK: S<3,4,3> o 0 o (MMA_sB=(128,16), MMA_K1, MMA_D=(4,2), PipeKV3):((64,1),0,(16,8192),16384)
        sK = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )
        # sV: S<3,4,3> o 0 o (MMA_sB=((64,2),16), MMA_V1, MMA_D8, PipeKV3):(((1,8192),64),0,1024,16384)
        sV_ptr = cute.recast_ptr(sK.iterator, swizzle_=v_smem_layout_staged.inner) # shared with sK
        sV = cute.make_tensor(sV_ptr, v_smem_layout_staged.outer)
        # sO: S<3,4,3> o 0 o (EPI_O=(8,16), EPI_D=(64,2), PipeEPI=(1,2)):((64,512),(1,8192),(0,16384))
        sO = storage.sO.get_tensor(
            o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner
        )
        
        # /////////////////////////////////////////////////////////////////////////////
        #  Make tiled mma fragment
        # /////////////////////////////////////////////////////////////////////////////
        
        cta_idx = 0 # default 1 cta
        
        # tSrQ: (MMA_sA1, MMA_Q=1, MMA_D=(4,2), PipeQ2):(0,0,(2,1024),2048)
        # tSrK: (MMA_sB1, MMA_K=1, MMA_D=(4,2), PipeKV3):(0,0,(2,1024),2048)
        # tStS/tStS0/tStS1: (MMA_TMEM_C=(Row128,Col128),1,1):((65536,1),0,0)
        qk_thr_mma = qk_tiled_mma.get_slice(cta_idx)
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        qk_acc_shape = qk_thr_mma.partition_shape_C(
            (self.qk_mma_tiler[0], self.qk_mma_tiler[1]) # (tileM, tileN)
        )
        tStS: cute.Tensor = qk_thr_mma.make_fragment_C(qk_acc_shape)
        
        # tOrV: (MMA_sB1, MMA_V1, MMA_D8, PipeKV3):(0,0,128,2048)
        # tOtO/tOtO0/tOtO1: (MMA_TMEM_C=(Row128,Col128),1,1):((65536,1),0,0)
        pv_thr_mma = pv_tiled_mma.get_slice(cta_idx)
        tOrV = pv_thr_mma.make_fragment_B(sV)
        pv_acc_shape = pv_thr_mma.partition_shape_C(
            (self.pv_mma_tiler[0], self.pv_mma_tiler[1])
        )
        tOtO = pv_thr_mma.make_fragment_C(pv_acc_shape)

        # Shard tmem for double buffer of {S0, S1} and {O0, O1}
        tStS0 = cute.make_tensor(tStS.iterator + self.tmem_s0_offset, tStS.layout) # 0~128
        tStS1 = cute.make_tensor(tStS.iterator + self.tmem_s1_offset, tStS.layout) # 128~256
        tOtO0 = cute.make_tensor(tOtO.iterator + self.tmem_o0_offset, tOtO.layout) # 256~384
        tOtO1 = cute.make_tensor(tOtO.iterator + self.tmem_o1_offset, tOtO.layout) # 384~512

        # Reuse the same tmem buffer of tS for tP
        # 
        # NOTE: since P is bf16 while S is fp32, thus tP only takes half 64cols out of 128cols of S
        # so below, we add an offset to use [32, 32+64) cols for P0 in S0, and [32+128, 32+128+64) cols for P1 in S1
        # 
        # and note that the `tOrP` is viewing the fp32 buffer of tS as bf16, 
        # so the ptr offset needs a `elems_per_acc_dtype` scaling factor
        # 
        # tP: (MMA_tA=(128,16), MMA_P1, MMA_D=(4,2), PipeAcc1):((64,1),0,(16,8192),0)
        # tOrP/tOrP0/tOrP1: (MMA_tA=(128,16), MMA_P1, MMA_D=(4,2)):((65536,1),0,(16,64))
        elems_per_acc_dtype = self.qk_acc_dtype.width // self.q_dtype.width
        tP = cute.make_tensor(tStS.iterator, p_tmem_layout_staged.outer)
        tOrP = pv_thr_mma.make_fragment_A(tP)[None, None, None, 0]
        tOrP0 = cute.make_tensor(
            tOrP.iterator
            + elems_per_acc_dtype * self.tmem_p0_offset, # 2 * 32 = 64
            tOrP.layout,
        )
        tOrP1 = cute.make_tensor(
            tOrP.iterator
            + elems_per_acc_dtype * self.tmem_p1_offset, # 2 * 160 = 320
            tOrP.layout,
        )

        if cutlass.const_expr(self.debug_print):
            if is_print_thread:
                cute.printf("")
                cute.printf("[kernel] sQ: {}", sQ)
                cute.printf("[kernel] sK: {}", sK)
                cute.printf("[kernel] sV: {}", sV)
                cute.printf("[kernel] sO: {}", sO)
                cute.printf("")
                cute.printf("[kernel] tSrQ.layout: {}", tSrQ.layout)
                cute.printf("[kernel] tSrK.layout: {}", tSrK.layout)
                cute.printf("[kernel] tOrV.layout: {}", tOrV.layout)
                cute.printf("")
                cute.printf("[kernel] tStS.layout (QK acc, tmem): {}", tStS.layout)
                cute.printf("[kernel] tOtO.layout (PV acc, tmem): {}", tOtO.layout)
                cute.printf("")
                cute.printf("[kernel] tP.layout: {}", tP.layout)
                cute.printf("[kernel] tOrP.layout: {}", tOrP.layout)
                cute.printf("[kernel] tOrP0.layout (P0 tmem frag): {}", tOrP0.layout)
                cute.printf("[kernel] tOrP1.layout (P1 tmem frag): {}", tOrP1.layout)
                cute.printf("")

        # /////////////////////////////////////////////////////////////////////////////
        #  Wait before tensor memory alloc
        # /////////////////////////////////////////////////////////////////////////////
        cute.arch.barrier( # equals to `__syncthreads()` here
            barrier_id=self.cta_sync_bar_id,
            number_of_threads=self.threads_per_cta,
        )
        
        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY Warp
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.empty_warp_id:
            # cute.arch.warpgroup_reg_dealloc(self.num_regs_empty) # deprecated
            cute.arch.setmaxregister_decrease(self.num_regs_empty) # 24

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD Warp
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            # cute.arch.warpgroup_reg_dealloc(self.num_regs_other) # deprecated
            cute.arch.setmaxregister_decrease(self.num_regs_other) # 32

            # /////////////////////////////////////////////////////////////////////////////
            #  Persistent tile scheduling loop
            # /////////////////////////////////////////////////////////////////////////////
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx # (midx, 0, (hidx, bidx))
                batch_coord = curr_block_coord[2][1]
                continue_cond = False # to simulate `continue` in python loop
                cuseqlen_q = Int32(0)
                seqlen_q = mQ_qdl.shape[0]
                
                if cutlass.const_expr(cum_seqlen_q is not None): # varlen case
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = (
                        not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q( # tileM * midx < seqlen_q
                            q_tiler=self.cta_tiler[0], # tileM
                            current_idx=curr_block_coord[0], # midx
                            seqlen_q=seqlen_q,
                        )
                    )
                
                if not continue_cond:
                    mQ_qdl_ = mQ_qdl
                    mK_kdl_ = mK_kdl
                    mV_dkl_ = mV_dkl
                    seqlen_k = mK_kdl.shape[0]
                    curr_block_coord_q = curr_block_coord
                    curr_block_coord_kv = curr_block_coord

                    # Offset mQ/mK/mV for varlen case
                    if cutlass.const_expr(cum_seqlen_q is not None):
                        logical_offset_mQ = (
                            mQ_qdl.shape[0] - seqlen_q, # offset in Q dimension
                            0, # no offset in D dimension
                            (0, cuseqlen_q + seqlen_q), # offset in batch dimension
                        )
                        mQ_qdl_ = cute.domain_offset(coord=logical_offset_mQ, tensor=mQ_qdl)
                        curr_block_coord_q = (
                            curr_block_coord[0], # midx
                            curr_block_coord[1], # nidx = 0
                            (curr_block_coord[2][0], Int32(0)), # (hidx, 0)
                        )

                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                        logical_offset_mK = (
                            mK_kdl.shape[0] - seqlen_k, # offset in K dimension
                            0, # no offset in D dimension
                            (0, cuseqlen_k + seqlen_k), # offset in batch dimension
                        )
                        logical_offset_mV = (
                            0,  # no offset in D dimension
                            mK_kdl.shape[0] - seqlen_k, # offset in K dimension
                            (0, cuseqlen_k + seqlen_k), # offset in batch dimension
                        )
                        mK_kdl_ = cute.domain_offset(logical_offset_mK, mK_kdl)
                        mV_dkl_ = cute.domain_offset(logical_offset_mV, mV_dkl)
                        curr_block_coord_kv = (
                            curr_block_coord[0], # midx
                            curr_block_coord[1], # nidx = 0
                            (curr_block_coord[2][0], Int32(0)), # (hidx, 0)
                        )

                    # ///////////////////////////////////////////////////////////////////////////////
                    #  TMA partition Q/K/V
                    # ///////////////////////////////////////////////////////////////////////////////
                    
                    # mQ_qdl_: (Q2048, D128, HB=((1,4),2)):(1@1,1@0,((1@2,1@3),1@4))
                    # gQ_qdl: (tileQ128, tileD128, restQ16, restD1, HB=((1,4),2)):(1@1,1@0,128@1,128@0,((1@2,1@3),1@4))
                    # tSgQ_qdl: (MMA_sA=(128,16), MMA_Q1, MMA_D8, restQ16, restD1, HB=((1,4),2)):((1@1,1@0),0,16@0,128@1,128@0,((1@2,1@3),1@4))
                    # tQgQ_qdl: (TMA_atom=(TMA_atomV=(64,128), TMA_restV=2), restQ16, restD1, HB=((1,4),2)):(((1@0,1@1),64@0),128@1,128@0,((1@2,1@3),1@4))
                    # tQgQ: (TMA_atom=(TMA_atomV=(64,128), TMA_restV=2), restQ16):(((1@0,1@1),64@0),128@1)
                    # tQsQ: (TMA_atom=(TMA_atomV=8192, TMA_restV=2), PipeQ2):((1,8192),16384)
                    gQ_qdl = cute.local_tile(
                        mQ_qdl_,
                        tiler=cute.select(self.qk_mma_tiler, mode=[0, 2]), # (tileQ, tileD)
                        coord=(None, None, None)
                    )
                    tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
                    tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_q,
                        cta_coord=cta_coord,
                        cta_layout=cta_layout,
                        smem_tensor=cute.group_modes(sQ, 0, 3), # group MMA dims
                        gmem_tensor=cute.group_modes(tSgQ_qdl, 0, 3), # group MMA dims
                    )
                    # since tileD == D, we slice out the dummy restD dim, and the current HB batch as well
                    tQgQ = tQgQ_qdl[None, None, 0, curr_block_coord_q[2]]

                    # mK_kdl_: (K4096, D128, HB=((1,4),2)):(1@1,1@0,((0,1@2),1@3))
                    # gK_kdl: (tileK128, tileD128, restK32, restD1, HB=((1,4),2)):(1@1,1@0,128@1,128@0,((0,1@2),1@3))
                    # tSgK_kdl: (MMA_sB=(128,16), MMA_K1, MMA_D8, restK32, restD1, HB=((1,4),2)):((1@1,1@0),0,16@0,128@1,128@0,((0,1@2),1@3))
                    # tKgK_kdl: (TMA_atom=(TMA_atomV=(64,128), TMA_restV=2), restK32, restD1, HB=((1,4),2)):(((1@0,1@1),64@0),128@1,128@0,((0,1@2),1@3))
                    # tKgK: (TMA_atom=(TMA_atomV=(64,128), TMA_restV=2), restK32):(((1@0,1@1),64@0),128@1)
                    # tKsK: (TMA_atom=(TMA_atomV=8192, TMA_restV=2), PipeKV3):((1,8192),16384)
                    gK_kdl = cute.local_tile(
                        mK_kdl_, 
                        tiler=cute.select(self.qk_mma_tiler, mode=[1, 2]), # (tileK, tileD)
                        coord=(None, None, None),
                    )
                    tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                    tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_k,
                        cta_coord=cta_coord,
                        cta_layout=cta_layout,
                        smem_tensor=cute.group_modes(sK, 0, 3), # group MMA dims
                        gmem_tensor=cute.group_modes(tSgK_kdl, 0, 3), # group MMA dims
                    )
                    # since tileD == D, we slice out the dummy restD dim, and the current HB batch as well
                    tKgK = tKgK_kdl[None, None, 0, curr_block_coord_kv[2]]

                    # mV_dkl_: (D128, K4096, HB=((1,4),2)):(1@0,1@1,((0,1@2),1@3))
                    # gV_dkl: (tileD128, tileK128, restD1, restK32, HB=((1,4),2)):(1@0,1@1,128@0,128@1,((0,1@2),1@3))
                    # tSgV_dkl: (MMA_sB=(128,16), MMA_D1, MMA_K8, restD1, restK32, HB=((1,4),2)):((1@0,1@1),0,16@1,128@0,128@1,((0,1@2),1@3))
                    # tVgV_dkl: (TMA_atom=(TMA_atomV=(64,128), TMA_restV=2), restD1, restK32, HB=((1,4),2)):(((1@0,1@1),64@0),128@0,128@1,((0,1@2),1@3))
                    # tVgV: (TMA_atom=(TMA_atomV=(64,128), TMA_restV=2), restK32):(((1@0,1@1),64@0),128@1)
                    # tVsV: (TMA_atom=(TMA_atomV=8192, TMA_restV=2), PipeKV3):((1,8192),16384)
                    gV_dkl = cute.local_tile(
                        mV_dkl_,
                        tiler=cute.select(self.pv_mma_tiler, mode=[1, 2]), # (tileD, tileV)
                        coord=(None, None, None),
                    )
                    tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
                    tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_v,
                        cta_coord=cta_coord,
                        cta_layout=cta_layout,
                        smem_tensor=cute.group_modes(sV, 0, 3), # group MMA dims
                        gmem_tensor=cute.group_modes(tSgV_dkl, 0, 3), # group MMA dims
                    )
                    tVgV = tVgV_dkl[None, 0, None, curr_block_coord_kv[2]]
                    
                    if cutlass.const_expr(self.debug_print):
                        is_first_work_tile = (curr_block_coord[0] == 0) and (curr_block_coord[1] == 0) and (curr_block_coord[2] == (0,0))
                        if (tidx == 32 * self.load_warp_id) and is_print_block and is_first_work_tile:
                            cute.printf("")
                            cute.printf("[kernel] After TMA partition, before TMA copy")
                            cute.printf("[kernel] curr_block_coord_q: {}", curr_block_coord_q)
                            cute.printf("[kernel] curr_block_coord_kv: {}", curr_block_coord_kv)
                            cute.printf("")
                            cute.printf("[kernel] mQ_qdl_.layout: {}", mQ_qdl_.layout)
                            cute.printf("[kernel] mK_kdl_.layout: {}", mK_kdl_.layout)
                            cute.printf("[kernel] mV_dkl_.layout: {}", mV_dkl_.layout)
                            cute.printf("")
                            cute.printf("[kernel] gQ_qdl.layout: {}", gQ_qdl.layout)
                            cute.printf("[kernel] gK_kdl.layout: {}", gK_kdl.layout)
                            cute.printf("[kernel] gV_dkl.layout: {}", gV_dkl.layout)
                            cute.printf("")
                            cute.printf("[kernel] tSgQ_qdl.layout: {}", tSgQ_qdl.layout)
                            cute.printf("[kernel] tSgK_kdl.layout: {}", tSgK_kdl.layout)
                            cute.printf("[kernel] tSgV_dkl.layout: {}", tSgV_dkl.layout)
                            cute.printf("")
                            cute.printf("[kernel] tQsQ layout: {}", tQsQ.layout)
                            cute.printf("[kernel] tKsK layout: {}", tKsK.layout)
                            cute.printf("[kernel] tVsV layout: {}", tVsV.layout)
                            cute.printf("")
                            cute.printf("[kernel] tQgQ_qdl layout: {}", tQgQ_qdl.layout)
                            cute.printf("[kernel] tKgK_kdl layout: {}", tKgK_kdl.layout)
                            cute.printf("[kernel] tVgV_dkl layout: {}", tVgV_dkl.layout)
                            cute.printf("")
                            cute.printf("[kernel] tQgQ layout: {}", tQgQ.layout)
                            cute.printf("[kernel] tKgK layout: {}", tKgK.layout)
                            cute.printf("[kernel] tVgV layout: {}", tVgV.layout)
                            cute.printf("")
                    
                    # ///////////////////////////////////////////////////////////////////////////////
                    #  Prologue: TMA copy Q0/Q1/K0/V0
                    # ///////////////////////////////////////////////////////////////////////////////
                    
                    # Q0
                    q0_handle = load_q_producer.acquire_and_advance() # NOTE: the returned handler stores the state before advancing
                    q0_midx = 2 * curr_block_coord_q[0] # 2 * midx
                    cute.copy(
                        tma_atom_q,
                        tQgQ[None, q0_midx],
                        tQsQ[None, q0_handle.index],
                        tma_bar_ptr=q0_handle.barrier,
                    )
                    # K0
                    k_handle = load_kv_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_k,
                        tKgK[None, 0],
                        tKsK[None, k_handle.index],
                        tma_bar_ptr=k_handle.barrier,
                    )
                    # Q1
                    q1_handle = load_q_producer.acquire_and_advance()
                    q1_midx = q0_midx + 1
                    cute.copy(
                        tma_atom_q,
                        tQgQ[None, q1_midx],
                        tQsQ[None, q1_handle.index],
                        tma_bar_ptr=q1_handle.barrier,
                    )
                    # V0
                    v_handle = load_kv_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_v,
                        tVgV[None, 0],
                        tVsV[None, v_handle.index],
                        tma_bar_ptr=v_handle.barrier,
                    )
                    
                    # ///////////////////////////////////////////////////////////////////////////////
                    #  Mainloop: TMA copy Ki/Vi
                    # ///////////////////////////////////////////////////////////////////////////////

                    # Loop over remaining K/V tiles to load
                    seqlen_kv_loop_steps = (
                        self.get_trip_count(curr_block_coord, self.cta_tiler, seqlen_k)
                    )
                    for kv_nidx in cutlass.range(1, seqlen_kv_loop_steps, unroll=1):
                        # Ki
                        k_handle = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_k,
                            tKgK[None, kv_nidx],
                            tKsK[None, k_handle.index],
                            tma_bar_ptr=k_handle.barrier,
                        )
                        # Vi
                        v_handle = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_v,
                            tVgV[None, kv_nidx],
                            tVsV[None, v_handle.index],
                            tma_bar_ptr=v_handle.barrier,
                        )

                # Advance to next Q tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA Warp
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            # cute.arch.warpgroup_reg_dealloc(self.num_regs_other) # deprecated
            cute.arch.setmaxregister_decrease(self.num_regs_other) # 32

            # Alloc tmem buffer
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            cute.arch.alloc_tmem(
                num_columns=tmem_alloc_cols,
                smem_ptr_to_write_address=tmem_holding_smem_buf
            )
            cute.arch.barrier(
                barrier_id=self.tmem_alloc_sync_bar_id,
                number_of_threads=self.threads_per_warp, # only mma warp need to access the tmem
            )

            # /////////////////////////////////////////////////////////////////////////////
            #  Persistent tile scheduling loop
            # /////////////////////////////////////////////////////////////////////////////
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                batch_coord = curr_block_coord[2][1]
                continue_cond = False
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = (
                        not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                            q_tiler=self.cta_tiler[0],
                            curr_block_coord=curr_block_coord[0],
                            seqlen_q=seqlen_q,
                        )
                    )

                if not continue_cond:
                    seqlen_k = mK_kdl.shape[0]
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                        
                    # ///////////////////////////////////////////////////////////////////////////////
                    #  Prologue: GEMM Q0K0, GEMM Q1K0, GEMM P0V0
                    # ///////////////////////////////////////////////////////////////////////////////

                    # --- GEMM_Q0K0 (Q0 * K0 -> S0) ---
                    # 1. wait for Q0 to be full
                    q0_handle = load_q_consumer.wait_and_advance()
                    tSrQ0 = tSrQ[None, None, None, q0_handle.index]
                    # 2. wait for K0 to be full
                    k_handle = load_kv_consumer.wait_and_advance()
                    tSrK0 = tSrK[None, None, None, k_handle.index]
                    # 3. acquire S0 to be empty
                    s0_handle = mma_s0_producer.acquire_and_advance()
                    # 4. gemm over MMA_D dim
                    num_kphases = cute.size(tSrQ0, mode=[2]) # MMA_D=(4,2) -> 8 phases
                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                        kphase_coord_0 = (None, None, kphase_idx)
                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0) # only the first kphase doesn't need to accumulate
                        cute.gemm( # Issuing UMMA
                            atom=qk_tiled_mma,
                            d=tStS0,
                            a=tSrQ0[kphase_coord_0],
                            b=tSrK0[kphase_coord_0],
                            c=tStS0,
                        )
                    # 5. commit S0 to be full
                    s0_handle.commit()
                    
                    # NOTE: K0 will be used in the GEMM_Q1K0 below, 
                    # so we need to keep it until then before release

                    # --- GEMM_Q1K0 (Q1 * K0 -> S1) ---
                    # 1. wait for Q1 to be full
                    # NOTE: K0 is ready in GEMM_Q0K0, so no need to wait
                    q1_handle = load_q_consumer.wait_and_advance()
                    tSrQ1 = tSrQ[None, None, None, q1_handle.index]
                    # 2. acquire S1 to be empty
                    s1_handle = mma_s1_producer.acquire_and_advance()
                    # 3. gemm over MMA_D dim
                    num_kphases = cute.size(tSrQ1, mode=[2])
                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                        kphase_coord_1 = (None, None, kphase_idx)
                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0) # only the first kphase doesn't need to accumulate
                        cute.gemm( # Issuing UMMA
                            qk_tiled_mma,
                            tStS1,
                            tSrQ1[kphase_coord_1],
                            tSrK0[kphase_coord_1],
                            tStS1,
                        )
                    # 4. commit S1 to be full
                    # Arrive the full mbar with `tcgen05.commit.mbarrier::arrive::one`
                    # the same across all producer commit in the whole example
                    s1_handle.commit()
                    # 5. release K0 to be empty
                    # Arrive the empty mbar with `tcgen05.commit.mbarrier::arrive::one`
                    # the same across all consumer release in the whole example
                    k_handle.release()

                    # NOTE: Q0 & Q1 are still needed in the whole seqlen_kv loop
                    # so we need to release them after the whole seqlen_kv loop done

                    # --- GEMM_P0V0 (P0 * V0 -> O0_partial) ---
                    # NOTE: O0 needs to be accumulated in the seqlen_kv loop
                    # 1. wait for V0 to be full
                    v_handle = load_kv_consumer.wait_and_advance()
                    tOrVi = tOrV[None, None, None, v_handle.index]
                    # 2. acquire corrected O0_partial to be empty
                    # NOTE: acquire corr first to take it out of the critical path since softmax takes longer
                    o0_handle = mma_corr_producer.acquire_and_advance()
                    # 3. acquire S0 to be empty => P0 to be full
                    # NOTE: 
                    #   1. this acquire returns the ownership of all of S0 to the mma warp
                    #       including the P0 part (inplaced in S0)
                    #   2. actually, the MMA warp is the producer of `S=QK => p = softmax(S)` pipeline
                    #       while the consumer of `P = softmax(S) => O_partial = PV` pipeline
                    #       so this second `acquire S0 to be empty` is been reusing to say `wait P0 to be full`
                    #       avoiding to add extra pipeline for `softmax => GEMM_PV`
                    s0_handle = mma_s0_producer.acquire_and_advance()
                    # 4. gemm over MMA_D dim
                    num_kphases = cute.size(tOrP0, mode=[2])
                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                        kphase_coord_2 = (None, None, kphase_idx)
                        pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                        cute.gemm(
                            pv_tiled_mma,
                            tOtO0,
                            tOrP0[kphase_coord_2],
                            tOrVi[kphase_coord_2],
                            tOtO0,
                        )
                    # 5. commit accumulated O0_partial to be full
                    o0_handle.commit()
                    
                    # NOTE: V0 will be used in the GEMM_P1V(i-1) in the first iter of the mainloop below, 
                    # so we don't release it here

                    # ///////////////////////////////////////////////////////////////////////////////
                    #  Mainloop: GEMM Q0Ki, GEMM P1V(i-1), GEMM Q1Ki, GEMM P0Vi
                    # ///////////////////////////////////////////////////////////////////////////////
                    seqlen_kv_loop_steps = (
                        self.get_trip_count(curr_block_coord, self.cta_tiler, seqlen_k)
                    )
                    # NOTE: since O0,O1 need to be accumulated across the mainloop, we need to set a global flag
                    pv_whether_acc = False
                    for i in cutlass.range(1, seqlen_kv_loop_steps, unroll=1):
                        # --- GEMM_Q0Ki (Q0 * Ki -> S0) ---
                        # 1. wait for Ki to be full
                        k_handle = load_kv_consumer.wait_and_advance()
                        tSrKi = tSrK[None, None, None, k_handle.index]
                        # 2. gemm over MMA_D dim
                        inner_num_kphases = cute.size(tSrQ0, mode=[2])
                        for kphase_idx in cutlass.range(inner_num_kphases, unroll_full=True):
                            kphase_coord_3 = (None, None, kphase_idx)
                            # NOTE: since P0 is shared with S0, and consumed in the GEMM_P0V(i-1) in previous iter, 
                            # we need to override it for S0 in the first phase
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                            cute.gemm(
                                qk_tiled_mma,
                                tStS0,
                                tSrQ0[kphase_coord_3],
                                tSrKi[kphase_coord_3],
                                tStS0,
                            )
                        # 3. commit S0 to be full => release P0 to be empty
                        s0_handle.commit()
                        
                        # NOTE: Ki will be used in the GEMM_Q1Ki below, so we don't release it here

                        # --- GEMM_P1V(i-1) (P1 * V(i-1) -> O1_partial) ---
                        # NOTE: V(i-1) is ready in GEMM_P0V(i-1) in the previous iter
                        # 1. acquire corrected O1_partial to be empty
                        o1_handle = mma_corr_producer.acquire_and_advance()
                        # 2. acquire S1 to be empty => wait P1 to be full
                        s1_handle = mma_s1_producer.acquire_and_advance()
                        # 3. gemm over MMA_D dim
                        inner_num_kphases = cute.size(tOrP0, mode=[2])
                        for kphase_idx in cutlass.range(inner_num_kphases, unroll_full=True):
                            kphase_coord_4 = (None, None, kphase_idx)
                            pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                            cute.gemm( 
                                pv_tiled_mma,
                                tOtO1,
                                tOrP1[kphase_coord_4],
                                tOrVi[kphase_coord_4],
                                tOtO1,
                            )
                            pv_whether_acc = True
                        # 4. commit accumulated O1_partial to be full
                        o1_handle.commit()
                        # 5. release V(i-1) to be empty
                        v_handle.release()

                        # --- GEMM_Q1Ki (Q1 * Ki -> S1) ---
                        # NOTE: Q1 is ready in GEMM_Q1K0; Ki is ready in GEMM_Q0Ki
                        # 1. gemm over MMA_D dim
                        inner_num_kphases = cute.size(tSrQ1, mode=[2])
                        for kphase_idx in cutlass.range(inner_num_kphases, unroll_full=True):
                            kphase_coord_5 = (None, None, kphase_idx)
                            # NOTE: since P1 is shared with S1, and consumed in the GEMM_P1V(i-1) above, 
                            # we need to override it for S1 in the first phase
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                            cute.gemm(
                                qk_tiled_mma,
                                tStS1,
                                tSrQ1[kphase_coord_5],
                                tSrKi[kphase_coord_5],
                                tStS1,
                            )
                        # 2. commit S1 to be full => release P1 to be empty
                        s1_handle.commit()
                        # 3. release Ki to be empty
                        k_handle.release()

                        # --- GEMM_P0Vi (P0 * Vi -> O0_partial) ---
                        # 1. wait for Vi to be full
                        v_handle = load_kv_consumer.wait_and_advance()
                        tOrVi = tOrV[None, None, None, v_handle.index]
                        # 2. acquire corrected O0_partial to be empty
                        o0_handle = mma_corr_producer.acquire_and_advance()
                        # 3. acquire S0 to be empty => wait P0 to be full
                        s0_handle = mma_s0_producer.acquire_and_advance()
                        # 4. gemm over MMA_D dim
                        inner_num_kphases = cute.size(tOrP0, mode=[2])
                        for kphase_idx in cutlass.range(inner_num_kphases, unroll_full=True):
                            kphase_coord_6 = (None, None, kphase_idx)
                            pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(
                                pv_tiled_mma,
                                tOtO0,
                                tOrP0[kphase_coord_6],
                                tOrVi[kphase_coord_6],
                                tOtO0,
                            )
                        # 5. commit accumulated O0_partial to be full
                        o0_handle.commit()
                        
                        # NOTE: Vi will be used in the GEMM_P1V(i-1) in the next iter, so we don't release it here 

                    # release Q0 & Q1 to be empty for next Q tile
                    q0_handle.release()
                    q1_handle.release()
                    
                    # ///////////////////////////////////////////////////////////////////////////////
                    #  Epilogue: GEMM P1V(i_end)
                    # ///////////////////////////////////////////////////////////////////////////////

                    # --- GEMM_P1V(i_end) (P1 * V(i_end) -> O1_partial) ---
                    # 1. acquire corrected O1_partial to be empty
                    o1_handle = mma_corr_producer.acquire_and_advance()
                    # 2. acquire S1 to be empty => wait P1 to be full
                    s1_handle = mma_s1_producer.acquire_and_advance()
                    # 3. gemm over MMA_D dim
                    num_kphases = cute.size(tOrP1, mode=[2])
                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                        kphase_coord_7 = (None, None, kphase_idx)
                        pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        cute.gemm(
                            pv_tiled_mma,
                            tOtO1,
                            tOrP1[kphase_coord_7],
                            tOrVi[kphase_coord_7],
                            tOtO1,
                        )
                    # 4. commit accumulated O1_partial to be full
                    o1_handle.commit()
                    # 5. release V(i_end) to be empty
                    v_handle.release()

                    # Commit S0 and S1 to be full => release P0/P1 to be empty
                    # NOTE: this commit is counter-intuitive but necessary,
                    # since the MMA warp is the consumer of the `softmax => GEMM_PV` pipeline,
                    # so to begin with the next tile, the softmax producer will acquire for P0/P1 to be empty, 
                    #   reusing the signal of `consumer wait S0/S1 to be full`,
                    # and it will hang there if we don't release P0/P1 to be empty here,
                    #   reusing the signal of `producer commit S0/S1 to be full`
                    s0_handle.commit()
                    s1_handle.commit()

                # Advance to next Q tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Dealloc tmem buffer
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
            cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0) # wait for softmax/correction warp group to finish using tmem
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                Float32,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_smem_buf,
            )
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue Warp
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.epilogue_warp_id:
            # cute.arch.warpgroup_reg_dealloc(self.num_regs_other) # deprecated
            cute.arch.setmaxregister_decrease(self.num_regs_other) # 32

            # /////////////////////////////////////////////////////////////////////////////
            #  Persistent tile scheduling loop
            # /////////////////////////////////////////////////////////////////////////////
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                batch_coord = curr_block_coord[2][1]
                continue_cond = False
                cuseqlen_q = Int32(0)
                seqlen_q = mQ_qdl.shape[0]

                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = (
                        not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                            q_tiler=self.cta_tiler[0],
                            curr_block_coord=curr_block_coord[0],
                            seqlen_q=seqlen_q,
                        )
                    )
                
                if not continue_cond:
                    curr_block_coord_o = curr_block_coord
                    mO_qdl_ = mO_qdl
                    if cutlass.const_expr(cum_seqlen_q is not None):
                        logical_offset_mO = (
                            mO_qdl_.shape[0] - seqlen_q, # qidx
                            0, # kidx
                            (0, cuseqlen_q + seqlen_q), # batch idx
                        )
                        mO_qdl_ = cute.domain_offset(logical_offset_mO, mO_qdl_)
                        curr_block_coord_o = (
                            curr_block_coord[0],
                            curr_block_coord[1],
                            (curr_block_coord[2][0], 0),
                        )

                    o0_coord = 2 * curr_block_coord_o[0]
                    o1_coord = o0_coord + 1
                    gO_qdl = cute.flat_divide(
                        mO_qdl_, cute.select(self.pv_mma_tiler, mode=[0, 1])
                    )
                    gO = gO_qdl[None, None, None, 0, curr_block_coord_o[2]]
                    tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_o,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sO, 0, 2),
                        cute.group_modes(gO, 0, 2),
                    )

                    # O0 O1 using the same pipeline
                    # wait from corr, issue tma store on smem
                    
                    # O0
                    # 1. wait for O0 final
                    o0_handle = corr_epi_consumer.wait_and_advance()
                    # 2. copy O0 to gmem (S2G)
                    cute.copy(tma_atom_o, tOsO[None, 0], tOgO[None, o0_coord])
                    cute.arch.cp_async_bulk_commit_group()
                    # O1
                    # 1. wait for O1 final
                    o1_handle = corr_epi_consumer.wait_and_advance()
                    # 2. copy O1 to gmem (S2G)
                    cute.copy(tma_atom_o, tOsO[None, 1], tOgO[None, o1_coord])
                    cute.arch.cp_async_bulk_commit_group()

                    # Ensure O0 buffer is ready to be released
                    cute.arch.cp_async_bulk_wait_group(1, read=True)
                    o0_handle.release()
                    # Ensure O1 buffer is ready to be released
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    o1_handle.release()

                # Advance to next Q tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax WarpGroup 0
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.softmax1_warp_ids[0]:
            # increase register for softmax after decreasing
            # cute.arch.warpgroup_reg_alloc(self.num_regs_softmax) # deprecated
            cute.arch.setmaxregister_increase(self.num_regs_softmax) # 192

            self.softmax(
                stage=0, # S0
                seqlen_k=mK_kdl.shape[0],
                cum_seqlen_q=cum_seqlen_q,
                cum_seqlen_k=cum_seqlen_k,
                scale_softmax_log2=scale_softmax_log2,
                qk_thr_mma=qk_thr_mma,
                tStS=tStS,
                tStSi=tStS0,
                mma_si_consumer=mma_s0_consumer,
                si_corr_producer=s0_corr_producer,
                s0_s1_sequence_consumer=s0_s1_sequence_consumer,
                s0_s1_sequence_producer=s0_s1_sequence_producer,
                tile_sched=tile_sched,
                is_print_block=is_print_block,
            )
            
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr) # arrive the tmem dealloc mbar when finishing using tmem for softmax

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax WarpGroup 1
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx < self.correction_warp_ids[0]
            and warp_idx >= self.softmax1_warp_ids[0]
        ):
            # increase register for softmax after decreasing
            # cute.arch.warpgroup_reg_alloc(self.num_regs_softmax) # deprecated
            cute.arch.setmaxregister_increase(self.num_regs_softmax) # 192

            self.softmax(
                stage=1, # S1
                seqlen_k=mK_kdl.shape[0],
                cum_seqlen_q=cum_seqlen_q,
                cum_seqlen_k=cum_seqlen_k,
                scale_softmax_log2=scale_softmax_log2,
                qk_thr_mma=qk_thr_mma,
                tStS=tStS,
                tStSi=tStS1,
                mma_si_consumer=mma_s1_consumer,
                si_corr_producer=s1_corr_producer,
                s0_s1_sequence_consumer=s0_s1_sequence_consumer,
                s0_s1_sequence_producer=s0_s1_sequence_producer,
                tile_sched=tile_sched,
                is_print_block=is_print_block,
            )
            
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr) # arrive the tmem dealloc mbar when finishing using tmem for softmax

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction WarpGroup
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            # cute.arch.warpgroup_reg_dealloc(self.num_regs_correction) # deprecated
            cute.arch.setmaxregister_decrease(self.num_regs_correction) # 96

            # cS: (tileQ128, tileK128):(1@0,1@1)
            # tScS: (MMA_TMEM_C=(128,128), restQ1, restK1):((1@0,1@1),0,0)
            # vec_layout: (128,2):(1,128) => fp32 row_max + fp32 row_sum takes 2 tmem cols
            cS = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
            tScS = qk_thr_mma.partition_C(cS)
            vec_layout = cute.make_layout((128, 2))
            
            # NOTE: we use the first two cols out of tStS{i} for its vec (row_max + row_sum)
            # tStS: (MMA_TMEM_C=(128,128), MMA_Q1, MMA_K1):((65536,1),0,0)
            # tStS_vec_layout: (128,2):(65536,1)
            tStS_vec_layout = cute.composition(tStS.layout, vec_layout)
            tStS_vec0 = cute.make_tensor(
                tStS.iterator + self.tmem_vec0_offset, # 0
                tStS_vec_layout
            )
            tStS_vec1 = cute.make_tensor(
                tStS.iterator + self.tmem_vec1_offset, # 128
                tStS_vec_layout
            )

            # tScS_vec_layout: (128,2):(1@0,1@1)
            tScS_vec_layout = cute.composition(tScS.layout, vec_layout)
            tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)

            # T2R Atom with `tcgen05.ld.sync.aligned.32x32b.x2`
            # layout_src_tv: (32,64):(0,1) => TMEM_ROW32 x TMEM_COL2 = 64 fp32 in tmem for one warp
            # layout_dst_tv: (32,2):(2,1) => 64 fp32 / 32 threads = 2 fp32 per thread in rmem
            tmem_load_v_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(2)),
                self.qk_acc_dtype,
            )
            
            # layout_src_tv_tiled: ((32,4), TMEM_LD_ATOM_SRC=((AtomCol2, AtomRow32),1)):((0,1),((128,4),0)) => 4 x (32x2) = 256 fp32 in tmem for a warp group
            # layout_dst_tv_tiled: ((32,4), TMEM_LD_ATOM_DST=((AtomCol2, AtomRow1),1)):((4,1),(128,0)) => still 2 fp32 per thread in rmem, but tiled for a warp group
            tiled_tmem_load_vec = tcgen05.make_tmem_copy(tmem_load_v_atom, tStS_vec0)
            
            # tTMEM_LOAD_VECtS0: (TMEM_LD_ATOM_SRC=((AtomCol2, AtomRow32),1),1,1):(((1,65536),0),0,0)
            # tTMEM_LOAD_VECtS1: (TMEM_LD_ATOM_SRC=((AtomCol2, AtomRow32),1),1,1):(((1,65536),0),0,0)
            # tTMEM_LOAD_VECcS: (TMEM_LD_ATOM_DST=((AtomCol2, AtomRow1),1),1,1):((1@1,0),0,0)
            thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids)) # tidx within the correction warp group
            thr_tmem_load_vec = tiled_tmem_load_vec.get_slice(thread_idx)
            tTMEM_LOAD_VECtS0 = thr_tmem_load_vec.partition_S(tStS_vec0)
            tTMEM_LOAD_VECtS1 = thr_tmem_load_vec.partition_S(tStS_vec1)
            tTMEM_LOAD_VECcS = thr_tmem_load_vec.partition_D(tScS_vec)
            
            if cutlass.const_expr(self.debug_print):
                if (thread_idx == 0) and is_print_block:
                    cute.printf("")
                    cute.printf("[kernel] Entering correction loop, print tile info and tensor layouts for the first tile")
                    cute.printf("[kernel] cS.layout: {}", cS.layout)
                    cute.printf("[kernel] tScS.layout: {}", tScS.layout)
                    cute.printf("[kernel] tStS.layout: {}", tStS.layout)
                    cute.printf("[kernel] vec_layout: {}", vec_layout)
                    cute.printf("[kernel] tStS_vec_layout: {}", tStS_vec_layout)
                    cute.printf("[kernel] tScS_vec_layout: {}", tScS_vec_layout)
                    cute.printf("")
                    cute.printf("[kernel] tmem_load_v_atom: layout_src_tv: {}, layout_dst_tv: {}", tmem_load_v_atom.layout_src_tv, tmem_load_v_atom.layout_dst_tv)
                    cute.printf("[kernel] tiled_tmem_load_vec: layout_src_tv_tiled: {}, layout_dst_tv_tiled: {}", tiled_tmem_load_vec.layout_src_tv_tiled, tiled_tmem_load_vec.layout_dst_tv_tiled)
                    cute.printf("[kernel] tTMEM_LOAD_VECtS0.layout: {}", tTMEM_LOAD_VECtS0.layout)
                    cute.printf("[kernel] tTMEM_LOAD_VECtS1.layout: {}", tTMEM_LOAD_VECtS1.layout)
                    cute.printf("[kernel] tTMEM_LOAD_VECcS.layout: {}", tTMEM_LOAD_VECcS.layout)
                    cute.printf("")

            # /////////////////////////////////////////////////////////////////////////////
            #  Persistent tile scheduling loop
            # /////////////////////////////////////////////////////////////////////////////
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                batch_coord = curr_block_coord[2][1]
                seqlen_k = mK_kdl.shape[0]
                continue_cond = False

                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = (
                        not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                            q_tiler=self.cta_tiler[0],
                            curr_block_coord=curr_block_coord[0],
                            seqlen_q=seqlen_q,
                        )
                    )

                if not continue_cond:
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                    
                    # Ignore first signal from softmax as no correction is required
                    vec0_handle = s0_corr_consumer.wait_and_advance()
                    vec0_handle.release()
                    vec1_handle = s1_corr_consumer.wait_and_advance()
                    
                    # NOTE: the first vec1 signal is ignored in the first iter below
                    
                    seqlen_kv_loop_steps = (
                        self.get_trip_count(curr_block_coord, self.cta_tiler, seqlen_k)
                    )
                    for i in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                        # Wait for vec0 (old_row_max, new_row_max) to be full by softmax warp group 0
                        vec0_handle = s0_corr_consumer.wait_and_advance()
                        
                        # T2R copy vec0 from tmem to rmem
                        # tTMEM_LOAD_VECrS = cute.make_fragment( # deprecated
                        tTMEM_LOAD_VECrS = cute.make_rmem_tensor(
                            tTMEM_LOAD_VECcS.shape, self.qk_acc_dtype
                        )
                        cute.copy(
                            tiled_tmem_load_vec, 
                            tTMEM_LOAD_VECtS0,
                            tTMEM_LOAD_VECrS
                        )
                        
                        # Compute rescale factor = exp2((new_row_max - old_row_max) * scale_softmax_log2)
                        scale_ = scale_softmax_log2 * (
                            tTMEM_LOAD_VECrS[0] - tTMEM_LOAD_VECrS[1]
                        )
                        scale = cute.math.exp2(scale_, fastmath=True)
                        
                        # Wait for O0 to be full by MMA warp
                        o0_handle = mma_corr_consumer.wait_and_advance()
                        
                        # Rescale the O0 partial result in-place in tmem
                        self.correction_rescale(pv_thr_mma, tOtO0, scale)
                        
                        # Release vec1
                        vec1_handle.release()
                        
                        # Release O0 after rescaling to let MMA warp to start new O_partial=PV
                        # NOTE: we need to use `tcgen05.wait::st` to ensure the store of rescaled tO is finished
                        # before we notify MMA warp
                        cute.arch.fence_view_async_tmem_op(kind="store")
                        o0_handle.release()

                        # Wait for vec1 (old_row_max, new_row_max) to be full by softmax warp group 1
                        vec1_handle = s1_corr_consumer.wait_and_advance()
                        
                        # T2R copy vec1 from tmem to rmem
                        cute.copy(
                            tiled_tmem_load_vec, 
                            tTMEM_LOAD_VECtS1,
                            tTMEM_LOAD_VECrS
                        )
                        
                        # Compute rescale factor = exp2((new_row_max - old_row_max) * scale_softmax_log2)
                        scale_ = scale_softmax_log2 * (
                            tTMEM_LOAD_VECrS[0] - tTMEM_LOAD_VECrS[1]
                        )
                        scale = cute.math.exp2(scale_, fastmath=True)
                        
                        # Wait for O1 to be full by MMA warp
                        o1_handle = mma_corr_consumer.wait_and_advance()
                        
                        # Rescale the O1 partial result in-place in tmem
                        self.correction_rescale(pv_thr_mma, tOtO1, scale)
                        
                        # Release vec0
                        vec0_handle.release()
                        
                        # Release O1 after rescaling to let MMA warp to start new O_partial=PV
                        cute.arch.fence_view_async_tmem_op(kind="store")
                        o1_handle.release()

                    # Release vec1
                    vec1_handle.release()

                    # Wait for final vec0 (global_row_sum, global_row_max)
                    # to be full by softmax warp group 0
                    vec0_handle = s0_corr_consumer.wait_and_advance()
                    
                    # T2R copy final vec0 from tmem to rmem
                    # tTMEM_LOAD_VECrS = cute.make_fragment( # deprecated
                    tTMEM_LOAD_VECrS = cute.make_rmem_tensor(
                        tTMEM_LOAD_VECcS.shape, self.qk_acc_dtype
                    )
                    cute.copy(tiled_tmem_load_vec, tTMEM_LOAD_VECtS0, tTMEM_LOAD_VECrS)
                    
                    # Release final vec0 after we ensure the tmem is finished and ready for next iter ???
                    cute.arch.fence_view_async_tmem_op(kind="load")
                    vec0_handle.release()
                    
                    # Wait for final tO0 to be full by MMA warp
                    o0_handle = mma_corr_consumer.wait_and_advance()
                    
                    # Acquire sO0 to be empty
                    o0_final_handle = corr_epi_producer.acquire_and_advance()
                    
                    # Scale final tO0 by T2R copying to rO0
                    # and then R2S copy to sO0
                    self.correction_epilog(
                        pv_thr_mma,
                        tOtO0,
                        scale=scale_output / tTMEM_LOAD_VECrS[0], # scale with inv_row_sum
                        sO=sO[None, None, 0],
                    )
                    
                    # Release final tO0 after writing to sO0
                    o0_handle.release()
                    
                    # Commit sO0 to be full
                    # to let epilogue warp to copy sO0 to gmem
                    o0_final_handle.commit()

                    # Wait for final vec1 (global_row_sum, global_row_max) 
                    # to be full by softmax warp group 1
                    vec1_handle = s1_corr_consumer.wait_and_advance()
                    
                    # T2R copy final vec1 from tmem to rmem
                    cute.copy(tiled_tmem_load_vec, tTMEM_LOAD_VECtS1, tTMEM_LOAD_VECrS)
                    
                    # Release final vec1 after we ensure the tmem is finished and ready for next iter ???
                    cute.arch.fence_view_async_tmem_op(kind="load")
                    vec1_handle.release()
                    
                    # Wait for final tO1 to be full by MMA warp
                    o1_handle = mma_corr_consumer.wait_and_advance()
                    
                    # Acquire sO1 to be empty
                    o1_final_handle = corr_epi_producer.acquire_and_advance()
                    
                    # Scale final tO1 by T2R copying to rO1
                    # and then R2S copy to sO1
                    self.correction_epilog(
                        pv_thr_mma,
                        tOtO1,
                        scale_output / tTMEM_LOAD_VECrS[0],
                        sO[None, None, 1],
                    )
                    
                    # Release final tO1 after writing to sO1
                    o1_handle.release()
                    
                    # Commit sO1 to be full
                    # to let epilogue warp to copy sO1 to gmem
                    o1_final_handle.commit()
                
                # Advance to next Q tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr) # arrive the tmem dealloc mbar when finishing using tmem for correction

    @cute.jit
    def softmax_step(
        self,
        stage: int,
        need_apply_mask: bool,
        iter_args: tuple,
        value_args: tuple,
        pipeline_args: tuple,
        atom_args: tuple,
        tensor_args: tuple,
        is_print_thread: bool = False,
    ) -> Tuple[
        Float32,
        Float32,
        pipeline.PipelineProducer.ImmutableResourceHandle,
        pipeline.PipelineConsumer,
        pipeline.PipelineProducer,
        pipeline.PipelineConsumer,
        pipeline.PipelineProducer,
    ]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages

        :param stage: Processing stage (0 for first half, 1 for second half)
        :type stage: int
        :param need_apply_mask: Whether to apply attention masking
        :type need_apply_mask: bool
        :param iter_args: Tuple containing the counting tensor, row_max, row_sum, and vector buffer's handle for current iteration
        :type iter_args: tuple
        :param value_args: Tuple containing seqlen_k and scale_softmax_log2
        :type value_args: tuple
        :param pipeline_args: Tuple containing pipeline related arguments for MMA, correction, and sequence synchronization
        :type pipeline_args: tuple
        :param atom_args: Tuple containing mma & copy atoms
        :type atom_args: tuple
        :param tensor_args: Tuple containing softmax related tensors
        :type tensor_args: tuple
        :return: Updated state values (row_max, row_sum, and pipeline related arguments)
        :rtype: tuple
        """
        cS, row_max, row_sum, vec_i_handle = iter_args
        seqlen_k, scale_softmax_log2 = value_args
        (
            mma_si_consumer,
            si_corr_producer,
            s0_s1_sequence_consumer,
            s0_s1_sequence_producer,
        ) = pipeline_args
        (
            qk_thr_mma,
            tiled_tmem_load,
            tiled_tmem_store,
            tiled_tmem_store_vec,
            thr_tmem_load,
            thr_tmem_store,
            thr_tmem_store_vec,
        ) = atom_args
        (
            tTMEM_LOADtS,
            tTMEM_STORE_VECtS,
            tTMEM_STOREtS_x4,
        ) = tensor_args
        
        # /////////////////////////////////////////////////////////////////////////////
        #  Make tensor for tS, tP, vec
        # /////////////////////////////////////////////////////////////////////////////

        # vec_layout: (128,2):(1,128)  
        #   — 128 rows × 2 cols: col0=old_row_max, col1=new_row_max
        vec_layout = cute.make_layout((128, 2))
        tilePlikeFP32 = self.qk_mma_tiler[1] // Float32.width * self.o_dtype.width # tileK128 // 2 = 64
        P_fp32_layout = cute.make_layout((128, tilePlikeFP32))
        
        # tScS: (MMA_TMEM_C=(128,128), MMA_Q1, MMA_K1):((1@0,1@1),0,0)
        tScS = qk_thr_mma.partition_C(cS)
        # tScS_vec_layout: (128,2):(1@0,1@1)  — coord layout for the 2-col vec region of S
        tScS_vec_layout = cute.composition(tScS.layout, vec_layout)
        # tScS_vec: (128,2):(1@0,1@1)  — coord tensor for vec region (col0=old_max, col1=new_max)
        tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)

        # tScS_P_layout: (128,64):(1@0,1@1) 
        #   — coord layout for P in fp32 units (64 fp32 cols = 128 bf16 cols)
        tScS_P_layout = cute.composition(tScS.layout, P_fp32_layout)
        # tScS_P: (128,64):(1@0,1@1)  
        #   — coord tensor for the P region in S tmem, viewed as fp32
        tScS_P = cute.make_tensor(tScS.iterator, tScS_P_layout)
        
        # /////////////////////////////////////////////////////////////////////////////
        #  T2R load tS to rS
        # /////////////////////////////////////////////////////////////////////////////
        
        # Wait for Si to be full by MMA warp
        si_handle = mma_si_consumer.wait_and_advance()
        
        # tTMEM_LOADtS: (TMEM_LD_ATOM_SRC=((AtomCol32, AtomRow32),1), restCol4, restRow1, restL1):(((1,65536),0),32,0,0)
        # tTMEM_LOADcS: (TMEM_LD_ATOM_DST=(32,1), restCol4, restRow1, restL1):((1@1,0),32@1,0,0)
        # tTMEM_LOADrS: (TMEM_LD_ATOM_DST=(32,1), restCol4, restRow1, restL1):((1,0),32,0,0)
        tTMEM_LOADcS = thr_tmem_load.partition_D(tScS)
        tTMEM_LOADrS = cute.make_rmem_tensor(
            tTMEM_LOADcS.shape, self.qk_acc_dtype
        )
        
        # T2R copy tS to rS with `tcgen05.ld.sync.aligned.32x32b.x32`
        cute.copy(tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADrS)
        
        # Apply softmax mask
        if need_apply_mask:
            self.apply_mask(tTMEM_LOADrS, tTMEM_LOADcS, seqlen_k)
            
        # /////////////////////////////////////////////////////////////////////////////
        #  Reduce rS to new_row_max and R2T store vec (old_row_max, new_row_max)
        # /////////////////////////////////////////////////////////////////////////////
            
        # tTMEM_STORE_VECcS: ((2,1),1,1):((1@1,0),0,0)
        tTMEM_STORE_VECcS = thr_tmem_store_vec.partition_S(tScS_vec)

        # max-reduce rS to get new_row_max
        old_row_max = row_max
        row_max = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, row_max, 0)
        
        # safe handle special case when row_max is -inf
        row_max_safe = row_max
        if row_max == -cutlass.Float32.inf:
            # to resolve `(-inf) - (-inf)` to `(-inf) - 0` 
            # to get `(-inf)` as the result instead of NaN
            row_max_safe = 0.0
        
        # R2S copy (old_row_max, row_max_safe) vec 
        # from rS to tS with `tcgen05.st.aligned.32x32b.x2`
        # tTMEM_STORE_VECrS: ((2,1),1,1):((1,0),0,0)
        # tTMEM_STORE_VECrS = cute.make_fragment( # deprecated
        tTMEM_STORE_VECrS = cute.make_rmem_tensor(
            tTMEM_STORE_VECcS.shape, self.qk_acc_dtype
        )
        tTMEM_STORE_VECrS[0] = old_row_max
        tTMEM_STORE_VECrS[1] = row_max_safe
        cute.copy(tiled_tmem_store_vec, tTMEM_STORE_VECrS, tTMEM_STORE_VECtS)
        
        # Commit row_max to be full for correction WG
        # TODO(REVIEW): we have to use `tcgen05.wait::st` 
        # to ensure the R2T store is finished and tmem is ready, 
        # before we notify the correction WG
        cute.arch.fence_view_async_tmem_op(kind="store")
        vec_i_handle.commit()
        
        # /////////////////////////////////////////////////////////////////////////////
        #  Apply softmax on rS to rP && R2T store rP to tP
        # /////////////////////////////////////////////////////////////////////////////
        
        # tTMEM_STOREcS: ((32,1), restRow1, restCol2):((1@1,0),0,32@1)
        # tTMEM_STORErS_x4: ((32,1),1, restCol2):((1,0),0,32) rmem in fp32
        #   which serves as the fp32 view of `tTMEM_STORErS_x4_e` to issue R2T copy as the src
        # tTMEM_STORErS_x4_e: ((32,1), restCol4,1,1):((1,0),32,0,0) rmem in bf16
        #   which serves as the bf16 downcasted rmem buffer for rP
        tTMEM_STOREcS = thr_tmem_store.partition_S(tScS_P)
        # tTMEM_STORErS_x4 = cute.make_fragment(tTMEM_STOREcS.shape, self.qk_acc_dtype) # deprecated
        tTMEM_STORErS_x4 = cute.make_rmem_tensor(
            tTMEM_STOREcS.shape, self.qk_acc_dtype
        )
        tTMEM_STORErS_x4_e = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErS_x4.iterator, dtype=self.q_dtype),
            tTMEM_LOADrS.layout,
        )

        scale = scale_softmax_log2
        minus_row_max_scale = (0.0 - row_max_safe) * scale

        # Sequence barrier wait
        if cutlass.const_expr(stage == 0): # s0
            # Acquire s1 to be finished
            sequence_producer_handle = s0_s1_sequence_producer.acquire_and_advance()
        else: # s1
            # Wait s0 to be finished
            sequence_consumer_handle = s0_s1_sequence_consumer.wait_and_advance()
        
        frg_cnt = 4
        frg_tile = cute.size(tTMEM_LOADrS) // frg_cnt  # 128 fp32 / 4 = 32 fp32 per fragment
        # tTMEM_LOADrS_frg: (frg_tile=32, frg_cnt=4):(1,32) in fp32 
        #   — fp32 S scores divided into 4 fragments for pipelined exp2
        tTMEM_LOADrS_frg = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(frg_tile))
        # tTMEM_STORErS_x4_e_frg: (frg_tile=32, frg_cnt=4):(1,32) in bf16
        #   — bf16 downcasted P, 4 fragments matching above
        tTMEM_STORErS_x4_e_frg = cute.logical_divide(
            tTMEM_STORErS_x4_e, cute.make_layout(frg_tile)
        )
        
        # Apply unnormalized stable softmax: exp(x * scale - row_max * scale)
        for j in range(frg_cnt): # 4 fragments
            for k in range(0, cute.size(tTMEM_LOADrS_frg, mode=[0]), 2): # for each 2-fp32 elem in one 32 fragment
                # f = x * scale + (-row_max * scale) = fma(x, scale, -row_max * scale)
                tTMEM_LOADrS_frg[k, j], tTMEM_LOADrS_frg[k + 1, j] = (
                    cute.arch.fma_packed_f32x2( # `fma.packed.f32x2`
                        (tTMEM_LOADrS_frg[k, j], tTMEM_LOADrS_frg[k + 1, j]),
                        (scale, scale),
                        (minus_row_max_scale, minus_row_max_scale),
                    )
                )
                
                # exp2(f)
                tTMEM_LOADrS_frg[k, j] = cute.math.exp2(
                    tTMEM_LOADrS_frg[k, j], fastmath=True
                )
                tTMEM_LOADrS_frg[k + 1, j] = cute.math.exp2(
                    tTMEM_LOADrS_frg[k + 1, j], fastmath=True
                )
            
            # Downcast to bf16 and store rP
            s_vec = tTMEM_LOADrS_frg[None, j].load()
            tTMEM_STORErS_x4_e_frg[None, j].store(s_vec.to(self.q_dtype))
        
        # Sequence barrier arrive
        if cutlass.const_expr(stage == 0): # s0
            # Commit s0 to be finished
            sequence_producer_handle.commit()
        else: # s1
            # Release s1 to be finished
            sequence_consumer_handle.release()
        
        # R2T copy fp32-view of bf16 rP to tP with `tcgen05.st.aligned.32x32b.x32`
        cute.copy(tiled_tmem_store, tTMEM_STORErS_x4, tTMEM_STOREtS_x4)
        
        # Release Si to be empty => commit Pi to be full for MMA warp
        # NOTE: we have to use `tcgen05.wait::st` to ensure the R2T store is finished and tmem is ready, 
        # before we notify the MMA warp
        cute.arch.fence_view_async_tmem_op(kind="store")
        si_handle.release()
        
        # /////////////////////////////////////////////////////////////////////////////
        #  Update row_sum
        # /////////////////////////////////////////////////////////////////////////////

        # Acquire old_row_sum to be consumed by correction WG and allowed to update
        vec_i_handle = si_corr_producer.acquire_and_advance()
        
        # Rescale old_row_sum with the factor = exp(old_row_max * scale - new_row_max * scale)
        acc_scale_ = scale * (old_row_max - row_max_safe)
        # NOTE: we need to scale to 1/2 since the `local_row_sum_0` below 
        # splits the `old_row_sum` to a packed tuple
        acc_scale = cute.math.exp2(acc_scale_, fastmath=True) * 0.5
        row_sum *= acc_scale
        
        # Prepare reduction unrolled row-sum rmem buffer
        reduction_unroll = 4
        frg_tile = cute.size(tTMEM_LOADrS) // reduction_unroll # 128 fp32 / 4 = 32 fp32 per fragment for reduction
        local_row_sum_0 = (row_sum, row_sum)
        local_row_sum_1 = (0.0, 0.0)
        local_row_sum_2 = (0.0, 0.0)
        local_row_sum_3 = (0.0, 0.0)

        # Reduce row_sum for the row this register holds
        # tTMEM_LOADrS_frg: (32,4):(1,32) => 128 fp32 rS to reduce
        tTMEM_LOADrS_frg = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(frg_tile))

        # Reduce within fragment
        for j in cutlass.range_constexpr(0, cute.size(tTMEM_LOADrS_frg, mode=[0]), 2): # for each 2-fp32 elem in one 32 fragment
            # unroll 0
            local_row_sum_0 = cute.arch.add_packed_f32x2(
                local_row_sum_0, (tTMEM_LOADrS_frg[j, 0], tTMEM_LOADrS_frg[j + 1, 0])
            )
            # unroll 1
            local_row_sum_1 = cute.arch.add_packed_f32x2(
                local_row_sum_1, (tTMEM_LOADrS_frg[j, 1], tTMEM_LOADrS_frg[j + 1, 1])
            )
            # unroll 2
            local_row_sum_2 = cute.arch.add_packed_f32x2(
                local_row_sum_2, (tTMEM_LOADrS_frg[j, 2], tTMEM_LOADrS_frg[j + 1, 2])
            )
            # unroll 3
            local_row_sum_3 = cute.arch.add_packed_f32x2(
                local_row_sum_3, (tTMEM_LOADrS_frg[j, 3], tTMEM_LOADrS_frg[j + 1, 3])
            )

        # Reduce across fragments
        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_1)
        local_row_sum_2 = cute.arch.add_packed_f32x2(local_row_sum_2, local_row_sum_3)
        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_2)
        row_sum = local_row_sum_0[0] + local_row_sum_0[1]
        
        if cutlass.const_expr(self.debug_print and stage == 0):
            if is_print_thread:
                cute.printf("")
                cute.printf("[softmax_step0] ---- Coord / layout tensors ----")
                cute.printf("[softmax_step0] vec_layout: {}", vec_layout)
                cute.printf("[softmax_step0] tilePlikeFP32: {}", tilePlikeFP32)
                cute.printf("[softmax_step0] tScS.layout: {}", tScS.layout)
                cute.printf("[softmax_step0] tScS_vec_layout: {}", tScS_vec_layout)
                cute.printf("[softmax_step0] tScS_vec.layout: {}", tScS_vec.layout)
                cute.printf("[softmax_step0] tScS_P_layout: {}", tScS_P_layout)
                cute.printf("[softmax_step0] tScS_P.layout: {}", tScS_P.layout)
                cute.printf("")
                cute.printf("[softmax_step0] ---- T2R load S ----")
                cute.printf("[softmax_step0] tiled_tmem_load: layout_src_tv_tiled: {}, layout_dst_tv_tiled: {}", tiled_tmem_load.layout_src_tv_tiled, tiled_tmem_load.layout_dst_tv_tiled)
                cute.printf("[softmax_step0] tTMEM_LOADtS.layout: {}", tTMEM_LOADtS.layout)
                cute.printf("[softmax_step0] tTMEM_LOADcS.layout: {}", tTMEM_LOADcS.layout)
                cute.printf("[softmax_step0] tTMEM_LOADrS.layout: {}", tTMEM_LOADrS.layout)
                cute.printf("")
                cute.printf("[softmax_step0] ---- R2T store vec ----")
                cute.printf("[softmax_step0] tiled_tmem_store_vec: layout_src_tv_tiled: {}, layout_dst_tv_tiled: {}", tiled_tmem_store_vec.layout_src_tv_tiled, tiled_tmem_store_vec.layout_dst_tv_tiled)
                cute.printf("[softmax_step0] tTMEM_STORE_VECtS.layout: {}", tTMEM_STORE_VECtS.layout)
                cute.printf("[softmax_step0] tTMEM_STORE_VECcS.layout: {}", tTMEM_STORE_VECcS.layout)
                cute.printf("[softmax_step0] tTMEM_STORE_VECrS.layout: {}", tTMEM_STORE_VECrS.layout)
                cute.printf("")
                cute.printf("[softmax_step0] ---- R2T store P ----")
                cute.printf("[softmax_step0] tiled_tmem_store: layout_src_tv_tiled: {}, layout_dst_tv_tiled: {}", tiled_tmem_store.layout_src_tv_tiled, tiled_tmem_store.layout_dst_tv_tiled)
                cute.printf("[softmax_step0] tTMEM_STOREtS_x4.layout: {}", tTMEM_STOREtS_x4.layout)
                cute.printf("[softmax_step0] tTMEM_STOREcS.layout: {}", tTMEM_STOREcS.layout)
                cute.printf("[softmax_step0] tTMEM_STORErS_x4.layout: {}", tTMEM_STORErS_x4.layout)
                cute.printf("[softmax_step0] tTMEM_STORErS_x4_e.layout: {}", tTMEM_STORErS_x4_e.layout)
                cute.printf("")
                cute.printf("[softmax_step0] ---- Fragments (frg_cnt=4) ----")
                cute.printf("[softmax_step0] frg_tile: {}", frg_tile)
                cute.printf("[softmax_step0] tTMEM_LOADrS_frg.layout: {}", tTMEM_LOADrS_frg.layout)
                cute.printf("[softmax_step0] tTMEM_STORErS_x4_e_frg.layout: {}", tTMEM_STORErS_x4_e_frg.layout)
                cute.printf("")

        return (
            row_max,
            row_sum,
            vec_i_handle,
            mma_si_consumer,
            si_corr_producer,
            s0_s1_sequence_consumer,
            s0_s1_sequence_producer,
        )

    # For both softmax0 and softmax1 warp group
    @cute.jit
    def softmax(
        self,
        stage: int,
        seqlen_k: Int32,
        cum_seqlen_q: cute.Tensor | None,
        cum_seqlen_k: cute.Tensor | None,
        scale_softmax_log2: Float32,
        qk_thr_mma: cute.core.ThrMma,
        tStS: cute.Tensor,
        tStSi: cute.Tensor,
        mma_si_consumer: pipeline.PipelineConsumer,
        si_corr_producer: pipeline.PipelineProducer,
        s0_s1_sequence_consumer: pipeline.PipelineConsumer,
        s0_s1_sequence_producer: pipeline.PipelineProducer,
        tile_sched: FmhaStaticTileScheduler,
        is_print_block: bool = False,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.

        :param stage: Processing stage (0 for first half, 1 for second half of attention matrix)
        :type stage: int
        :param scale_softmax_log2: Log2 scale factor for softmax operation
        :type scale_softmax_log2: Float32
        :param qk_thr_mma: Thread MMA operation for QK matrix multiplication
        :type qk_thr_mma: cute.core.ThrMma
        :param tStS: Shared tensor for softmax input/output
        :type tStS: cute.Tensor
        :param tStSi: Input tensor containing attention scores
        :type tStSi: cute.Tensor
        :param mma_si_pipeline: Pipeline for synchronizing with MMA operations
        :type mma_si_pipeline: pipeline.PipelineAsync
        :param si_corr_pipeline: Pipeline for synchronizing with correction operations
        :type si_corr_pipeline: pipeline.PipelineAsync
        :param s0_s1_sequence_pipeline: Pipeline for synchronizing between stage 0 and 1
        :type s0_s1_sequence_pipeline: pipeline.PipelineAsync
        :param tile_sched_params: Parameters for tile scheduling
        :type tile_sched_params: FmhaStaticTileSchedulerParams
        :param is_print_block: Flag to enable printing of block information
        :type is_print_block: bool
        """
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % ( # tidx within this softmax warp group
            self.threads_per_warp
            * (
                len(self.softmax0_warp_ids)
                if stage == 0
                else len(self.softmax1_warp_ids)
            )
        )
        
        # /////////////////////////////////////////////////////////////////////////////
        #  Make tmem (coord) tensor of S/P
        # /////////////////////////////////////////////////////////////////////////////

        # cS_base: (tileQ128, tileK128):(1@0,1@1)
        # vec_layout: (128,2):(1,128)
        # P_fp32_layout: (128,64):(1,128)
        tilePlikeFP32 = self.qk_mma_tiler[1] // 32 * self.o_dtype.width # tileK128 // 32 * 16 = 64
        vec_layout = cute.make_layout((128, 2))
        P_fp32_layout = cute.make_layout((128, tilePlikeFP32))
        cS_base = cute.make_identity_tensor(
            (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
        )
        
        # tStS: (MMA_TMEM_C=(128,128), MMA_Q1, MMA_K1):((65536,1),0,0)
        # tScS: (MMA_TMEM_C=(ROW128, COL128), MMA_Q1, MMA_K1):((1@0,1@1),0,0)
        tScS = qk_thr_mma.partition_C(cS_base)
        # tStS_vec_layout: (Row128, Col2):(65536,1)
        tStS_vec_layout = cute.composition(tStS.layout, vec_layout)
        tmem_vec_offset = self.tmem_vec0_offset if stage == 0 else self.tmem_vec1_offset # {0: 0, 1: 128}
        # tStS_vec: (Row128, Col2):(65536,1)
        tStS_vec = cute.make_tensor(tStS.iterator + tmem_vec_offset, tStS_vec_layout)
        # tScS_vec_layout: (Row128, Col2):(1@0,1@1)
        tScS_vec_layout = cute.composition(tScS.layout, vec_layout)
        # tScS_vec: (Row128, Col2):(1@0,1@1)
        tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)
        # tStS_P_layout: (Row128, Col64):(65536,1)
        tStS_P_layout = cute.composition(tStS.layout, P_fp32_layout)
        tmem_p_offset = self.tmem_p0_offset if stage == 0 else self.tmem_p1_offset # {0: 32, 1: 160}
        # tStS_P: (Row128, Col64):(65536,1)
        tStS_P = cute.make_tensor(tStS.iterator + tmem_p_offset, tStS_P_layout)
        
        # /////////////////////////////////////////////////////////////////////////////
        #  Make tmem load tiled copy for T2R copy tS -> rS
        # /////////////////////////////////////////////////////////////////////////////
        
        # tmem_load_atom with `tcgen05.ld.sync.aligned.32x32b.x32`
        # layout_src_tv: (32,1024):(0,1) => Row32 x Col32 x 4B = 1024 fp32 elems per warp in tmem
        # layout_dst_tv: (32,32):(32,1) => 1024 / 32 = 32 fp32 elems per thread in rmem
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.qk_acc_dtype,
        )
        
        # tiled_tmem_load (T2R):
        # layout_src_tv_tiled: ((32,4), TMEM_LD_ATOM_SRC=(1024,1)):((0,1),(4,0)) => 4 x (32x32) = 4096 fp32 elems in tmem for a warp group
        # layout_dst_tv_tiled: ((32,4), TMEM_LD_ATOM_DST=(32,1)):((128,1),(4,0)) => still 32 fp32 elems per thread in rmem, but tiled for a warp group
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStSi)
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        
        # tStSi: (MMA_TMEM_C=(128,128), MMA_Q1, MMA_K1):((65536,1),0,0)
        # tTMEM_LOADtS: (TMEM_LD_ATOM_SRC=((AtomCol32, AtomRow32),1), restCol4, restRow1,1):(((1,65536),0),32,0,0)
        tTMEM_LOADtS = thr_tmem_load.partition_S(tStSi)
        
        # /////////////////////////////////////////////////////////////////////////////
        #  Make tmem store tiled copy for R2T copy (old_row_max, new_row_max) vec
        # /////////////////////////////////////////////////////////////////////////////
        
        # tmem_store_vec_atom with `tcgen05.st.sync.aligned.32x32b.x2`
        # layout_src_tv: (32,2):(2,1) => 64 / 32 = 2 fp32 elems per thread in rmem
        # layout_dst_tv: (32,64):(0,1) => 32 x 2 x 4B = 64 fp32 elems per warp in tmem
        tmem_store_vec_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(2)),
            self.qk_acc_dtype,
        )
        
        # tiled_tmem_store_vec (R2T):
        # layout_src_tv_tiled: ((32,4), TMEM_ST_ATOM_SRC=(AtomCol2, AtomRow1)):((4,1),(128,0)) => still 2 fp32 elems per thread in rmem, but tiled for a warp group
        # layout_dst_tv_tiled: ((32,4), TMEM_ST_ATOM_DST=((AtomCol2, AtomRow32),1)):((0,1),((128,4),0)) => 4 x (32x2) = 256 fp32 elems in tmem for a warp group
        tiled_tmem_store_vec = tcgen05.make_tmem_copy(tmem_store_vec_atom, tStS_vec)
        thr_tmem_store_vec = tiled_tmem_store_vec.get_slice(thread_idx)
        
        # tTMEM_STORE_VECcS: (TMEM_ST_ATOM_SRC=(AtomCol2, AtomRow1), RestCol1, RestRow1):((1@1,0),0,0)
        tTMEM_STORE_VECcS = thr_tmem_store_vec.partition_S(tScS_vec)
        # tTMEM_STORE_VECtS: (TMEM_ST_ATOM_DST=((AtomCol2, AtomRow32),1), RestCol1, RestRow1):(((1,65536),0),0,0)
        tTMEM_STORE_VECtS = thr_tmem_store_vec.partition_D(tStS_vec)
        
        # /////////////////////////////////////////////////////////////////////////////
        #  Make tmem store tiled copy for R2T copy rP -> tP
        # /////////////////////////////////////////////////////////////////////////////
        
        # tmem_store_atom with `tcgen05.st.sync.aligned.32x32b.x32`
        # layout_src_tv: (32,32):(32,1) => 1024 / 32 = 32 fp32 elems per thread in rmem
        # layout_dst_tv: (32,1024):(0,1) => Row32 x Col32 x 4B = 1024 fp32 elems per warp in tmem
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),
            self.qk_acc_dtype,
        )
        
        # tiled_tmem_store for R2T copy rP -> tP:
        # layout_src_tv_tiled: ((32,4),(32,1)):((4,1),(128,0)) => still 32 fp32 elems per thread in rmem, but tiled for a warp group
        # layout_dst_tv_tiled: ((32,4),((32,32),1)):((0,1),((128,4),0)) => 4 x (32x32) = 4096 fp32 elems in tmem for a warp group
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStS_P)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)
        # tTMEM_STOREtS_x4: (TMEM_ST_ATOM_DST=((AtomCol32, AtomRow32),1), restRow1, restCol2):(((1,65536),0),0,32)
        tTMEM_STOREtS_x4 = thr_tmem_store.partition_D(tStS_P)
        
        if cutlass.const_expr(self.debug_print and stage == 0):
            if (thread_idx == 0) and is_print_block:
                cute.printf("")
                cute.printf("[softmax0] Entering softmax loop, print tile info and tensor layouts")
                cute.printf("[softmax0] tStS.layout: {}", tStS.layout)
                cute.printf("[softmax0] tilePlikeFP32: {}", tilePlikeFP32)
                cute.printf("[softmax0] cS_base.layout: {}", cS_base.layout)
                cute.printf("[softmax0] vec_layout: {}", vec_layout)
                cute.printf("[softmax0] P_fp32_layout: {}", P_fp32_layout)
                cute.printf("")
                cute.printf("[softmax0] tScS.layout: {}", tScS.layout)
                cute.printf("[softmax0] tStS_vec_layout: {}", tStS_vec_layout)
                cute.printf("[softmax0] tStS_vec.layout: {}", tStS_vec.layout)
                cute.printf("[softmax0] tScS_vec_layout: {}", tScS_vec_layout)
                cute.printf("[softmax0] tScS_vec.layout: {}", tScS_vec.layout)
                cute.printf("[softmax0] tStS_P_layout: {}", tStS_P_layout)
                cute.printf("[softmax0] tStS_P.layout: {}", tStS_P.layout)
                cute.printf("")
                cute.printf("[softmax0] tStSi.layout: {}", tStSi.layout)
                cute.printf("[softmax0] tmem_load_atom: layout_src_tv: {}, layout_dst_tv: {}", tmem_load_atom.layout_src_tv, tmem_load_atom.layout_dst_tv)
                cute.printf("[softmax0] tiled_tmem_load: layout_src_tv_tiled: {}, layout_dst_tv_tiled: {}", tiled_tmem_load.layout_src_tv_tiled, tiled_tmem_load.layout_dst_tv_tiled)
                cute.printf("[softmax0] tTMEM_LOADtS.layout: {}", tTMEM_LOADtS.layout)
                cute.printf("")
                cute.printf("[softmax0] tmem_store_vec_atom: layout_src_tv: {}, layout_dst_tv: {}", tmem_store_vec_atom.layout_src_tv, tmem_store_vec_atom.layout_dst_tv)
                cute.printf("[softmax0] tiled_tmem_store_vec: layout_src_tv_tiled: {}, layout_dst_tv_tiled: {}", tiled_tmem_store_vec.layout_src_tv_tiled, tiled_tmem_store_vec.layout_dst_tv_tiled)
                cute.printf("[softmax0] tTMEM_STORE_VECtS.layout: {}", tTMEM_STORE_VECtS.layout)
                cute.printf("[softmax0] tTMEM_STORE_VECcS.layout: {}", tTMEM_STORE_VECcS.layout)
                cute.printf("")
                cute.printf("[softmax0] tmem_store_atom: layout_src_tv: {}, layout_dst_tv: {}", tmem_store_atom.layout_src_tv, tmem_store_atom.layout_dst_tv)
                cute.printf("[softmax0] tiled_tmem_store: layout_src_tv_tiled: {}, layout_dst_tv_tiled: {}", tiled_tmem_store.layout_src_tv_tiled, tiled_tmem_store.layout_dst_tv_tiled)
                cute.printf("[softmax0] tTMEM_STOREtS_x4.layout: {}", tTMEM_STOREtS_x4.layout)
                cute.printf("")

        # /////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduling loop
        # /////////////////////////////////////////////////////////////////////////////
        work_tile = tile_sched.initial_work_tile_info()
        while work_tile.is_valid_tile:
            curr_block_coord = work_tile.tile_idx
            batch_coord = curr_block_coord[2][1]
            seqlen_k_ = seqlen_k
            continue_cond = False

            if cutlass.const_expr(cum_seqlen_q is not None):
                cuseqlen_q = cum_seqlen_q[batch_coord]
                seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                continue_cond = (
                    not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        q_tiler=self.cta_tiler[0],
                        current_idx=curr_block_coord[0],
                        seqlen_q=seqlen_q,
                    )
                )

            if not continue_cond:
                if cutlass.const_expr(cum_seqlen_k is not None):
                    cuseqlen_k = cum_seqlen_k[batch_coord]
                    seqlen_k_ = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                
                # Init row_max, row_sum for this Q tile
                # which will be iterately updated across each K/V tile
                row_max = -Float32.inf
                row_sum = 0.0
                
                # Prepare args
                value_args = (seqlen_k_, scale_softmax_log2)
                atom_args = (
                    qk_thr_mma,
                    tiled_tmem_load,
                    tiled_tmem_store,
                    tiled_tmem_store_vec,
                    thr_tmem_load,
                    thr_tmem_store,
                    thr_tmem_store_vec,
                )
                tensor_args = (
                    tTMEM_LOADtS,
                    tTMEM_STORE_VECtS,
                    tTMEM_STOREtS_x4,
                )

                logical_offset = (
                    curr_block_coord[0] * self.cta_tiler[0] # qidx
                    + stage * self.qk_mma_tiler[0], # q0 / q1
                    0, # kidx
                )
                cS = cute.domain_offset(logical_offset, cS_base)
                
                # Acquire final vec in last iter to be consumed by correction WG
                vec_i_handle = si_corr_producer.acquire_and_advance()
                
                # Unmasked softmax iterations
                unmask_count = self.get_unmasked_trip_count(
                    curr_block_coord,
                    self.cta_tiler,
                    seqlen_k_,
                )
                for i in cutlass.range(0, unmask_count, 1, unroll=1):
                    cS_iter = cute.domain_offset((0, i * self.qk_mma_tiler[1]), cS)
                    iter_args = (cS_iter, row_max, row_sum, vec_i_handle)
                    pipeline_args = (
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    )
                    (
                        row_max,
                        row_sum,
                        vec_i_handle,
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    ) = self.softmax_step(
                        stage,
                        False, # need_apply_mask = False for unmasked iterations
                        iter_args,
                        value_args,
                        pipeline_args,
                        atom_args,
                        tensor_args,
                        is_print_thread=thread_idx == 0 \
                            and i == 0 \
                            and is_print_block \
                            and (curr_block_coord[0] == 0) and (curr_block_coord[1] == 0) and (curr_block_coord[2] == (0,0)),
                    )
                
                # Masked softmax iterations
                mask_count = self.get_masked_trip_count(
                    curr_block_coord,
                    self.cta_tiler,
                    seqlen_k_,
                )
                for i in cutlass.range(
                    unmask_count, unmask_count + mask_count, 1, unroll=1
                ):
                    cS_iter = cute.domain_offset((0, i * self.qk_mma_tiler[1]), cS)
                    iter_args = (cS_iter, row_max, row_sum, vec_i_handle)
                    pipeline_args = (
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    )
                    
                    (
                        row_max,
                        row_sum,
                        vec_i_handle,
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    ) = self.softmax_step(
                        stage,
                        True, # need_apply_mask = True for masked iterations
                        iter_args,
                        value_args,
                        pipeline_args,
                        atom_args,
                        tensor_args,
                        is_print_thread=thread_idx == 0 \
                            and i == 0 \
                            and is_print_block \
                            and (curr_block_coord[0] == 0) and (curr_block_coord[1] == 0) and (curr_block_coord[2] == (0,0)),
                    )
                
                # Wait for MMA final commit ???
                si_handle = mma_si_consumer.wait_and_advance()
                
                # Final store (final_row_sum, final_row_max) vec to tmem for correction WG
                # tTMEM_STORE_VECrS = cute.make_fragment( # deprecated
                tTMEM_STORE_VECrS = cute.make_rmem_tensor(
                    tTMEM_STORE_VECcS.shape, self.qk_acc_dtype
                )
                tTMEM_STORE_VECrS[0] = row_sum
                tTMEM_STORE_VECrS[1] = row_max
                
                # R2T copy with `tcgen05.st.aligned.32x32b.x2` from rmem to tmem
                cute.copy(tiled_tmem_store_vec, tTMEM_STORE_VECrS, tTMEM_STORE_VECtS)
                
                # Final commit vec to be full for correction WG
                # NOTE: we have to use `tcgen05.wait::st` to ensure the R2T store is finished and tmem is ready,
                # before we notify the correction WG
                cute.arch.fence_view_async_tmem_op(kind="store")
                vec_i_handle.commit()
                
                # Wait for correction WG to finish consuming the final vec ???
                si_corr_producer.acquire()
                
                # Release Si to be empty ???
                si_handle.release()

            # Advance to next Q tile
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        scale: Float32,
    ):
        """Rescale intermediate attention results based on softmax normalization factor.

        This method performs a crucial correction step in the attention computation pipeline.
        When processing attention in blocks, the softmax normalization factors may change
        as new blocks are processed. This method rescales previously computed partial
        output values to account for updated normalization factors.

        The implementation uses efficient tensor memory operations to:
        1. Load existing partial attention output from tensor memory
        2. Apply the scaling factor to all elements
        3. Store the rescaled results back to tensor memory

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.core.ThrMma
        :param tOtO: Tensor representing partial attention output to be rescaled
        :type tOtO: cute.Tensor
        :param scale: Scaling factor to apply to the partial results
        :type scale: Float32
        """
        pv_tiled_mma_shape = (
            self.pv_mma_tiler[0],
            self.pv_mma_tiler[1],
        )
        cO = cute.make_identity_tensor(pv_tiled_mma_shape)
        tOcO = thr_mma.partition_C(cO)

        corr_tile_size = 16  # tuneable parameter
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )

        tOtO_i_layout = cute.composition(
            tOtO.layout, cute.make_layout((128, corr_tile_size))
        )
        tOcO_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((128, corr_tile_size))
        )

        tOtO_i = cute.make_tensor(tOtO.iterator, tOtO_i_layout)
        tOcO_i = cute.make_tensor(tOcO.iterator, tOcO_i_layout)

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i)
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i)
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids))
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i)

        tTMEM_STOREtO = thr_tmem_store.partition_D(tOtO_i)

        # tTMrO = cute.make_fragment( # deprecated
        tTMrO = cute.make_rmem_tensor(
            (tTMEM_LOADcO.shape, 128 // corr_tile_size), self.pv_acc_dtype
        )
        for i in range(self.cta_tiler[2] // corr_tile_size):
            tTMrO_i_ = tTMrO[None, i]
            tTMrO_i_layout = cute.composition(
                tTMrO_i_.layout, cute.make_layout(tTMrO.shape[0])
            )
            tTMrO_i = cute.make_tensor(tTMrO_i_.iterator, tTMrO_i_layout)
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )
            tTMEM_STOREtO_i = cute.make_tensor(
                tTMEM_STOREtO.iterator + i * corr_tile_size, tTMEM_STOREtO.layout
            )

            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO_i)
            for j in range(0, cute.size(tTMrO_i), 2):
                tTMrO_i[j], tTMrO_i[j + 1] = cute.arch.mul_packed_f32x2(
                    (tTMrO_i[j], tTMrO_i[j + 1]),
                    (scale, scale),
                )
            cute.copy(tiled_tmem_store, tTMrO_i, tTMEM_STOREtO_i)

    @cute.jit
    def correction_epilog(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        scale: Float32,
        sO: cute.Tensor,
    ):
        """Apply final scaling and transformation to attention output before writing to global memory.

        This correction_epilog function handles the final processing step for attention output values.
        It applies a scaling factor to the accumulated attention results and prepares the
        data for efficient transfer back to global memory.

        The method performs:
        1. Loading of accumulated attention results from tensor memory
        2. Application of the final output scaling factor
        3. Type conversion if necessary (typically from higher precision accumulator to output precision)
        4. Reorganization of data for optimal memory access patterns
        5. Preparation for efficient TMA store operations

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.core.ThrMma
        :param tOtO: Tensor containing accumulated attention output
        :type tOtO: cute.Tensor
        :param scale: Final scaling factor to apply to the output
        :type scale: Float32
        :param sO: Shared memory tensor for the final output
        :type sO: cute.Tensor
        """

        pv_tiled_mma_shape = (
            self.pv_mma_tiler[0],
            self.pv_mma_tiler[1],
        )
        cO = cute.make_identity_tensor(pv_tiled_mma_shape)

        corr_tile_size = 32 * 8 // self.o_dtype.width
        tOsO = thr_mma.partition_C(sO)
        tOcO = thr_mma.partition_C(cO)

        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((128, corr_tile_size)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((128, corr_tile_size)))
        tOsO_i = cute.logical_divide(tOsO, cute.make_layout((128, corr_tile_size)))
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids))

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils.get_tmem_load_op(
            self.pv_mma_tiler,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=False,
        )

        tiled_tmem_load = tcgen05.make_tmem_copy(
            tmem_copy_atom, tOtO_i[(None, None), 0]
        )
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        
        smem_copy_atom = sm100_utils.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tTMEM_LOADsO = thr_tmem_load.partition_D(tOsO_i[(None, None), None])
        tTMEM_LOADoO = thr_tmem_load.partition_D(tOcO_i[(None, None), None])

        for i in range(self.cta_tiler[2] // corr_tile_size):
            # T2R copy O from tmem to rmem in fp32
            tTMEM_LOADtO_i = tTMEM_LOADtO[None, 0, 0, i]
            tTMEM_LOADsO_i = tTMEM_LOADsO[None, 0, 0, i]
            # tTMrO = cute.make_fragment( # deprecated
            tTMrO = cute.make_rmem_tensor(
                tTMEM_LOADoO[None, 0, 0, i].shape, self.pv_acc_dtype
            )
            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO)
            
            # Rescale the output with the inv_row_sum scale factor in fp32
            for j in range(0, cute.size(tTMrO), 2):
                tTMrO[j], tTMrO[j + 1] = cute.arch.mul_packed_f32x2(
                    (tTMrO[j], tTMrO[j + 1]),
                    (scale, scale),
                )
                
            # Convert the output to the desired output type (bf16)
            # tSMrO = cute.make_fragment(tTMrO.shape, self.o_dtype) # deprecated
            tSMrO = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
            o_vec = tTMrO.load()
            tSMrO.store(o_vec.to(self.o_dtype))
            
            # R2S copy the rescaled output from rmem to smem
            cute.copy(tiled_smem_store, tSMrO, tTMEM_LOADsO_i)

        # fence view async shared to let the R2S copy visible to epilogue's TMA S2G store
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

    def get_trip_count(
        self,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_k: Int32,
    ) -> Int32:
        result = 0
        if (
            self.mask_type == MaskType.NO_MASK
            or self.mask_type == MaskType.RESIDUAL_MASK
        ):
            result = cute.ceil_div(seqlen_k, tile_shape[1])
        elif self.mask_type == MaskType.CAUSAL_MASK:
            max_blocks_k = cute.ceil_div(seqlen_k, tile_shape[1])
            max_blocks_q = cute.ceil_div(
                (blk_coord[0] + 1) * tile_shape[0], tile_shape[1]
            )
            result = cutlass.min(max_blocks_k, max_blocks_q)
        return result

    @cute.jit
    def get_masked_trip_count(
        self,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_k: Int32,
    ) -> Int32:
        result = 0
        if self.mask_type == MaskType.NO_MASK:
            result = 0
        elif self.mask_type == MaskType.RESIDUAL_MASK:
            if seqlen_k % tile_shape[1] != 0:
                result = 1
            else:
                result = 0
        elif self.mask_type == MaskType.CAUSAL_MASK:
            trip_count = self.get_trip_count(blk_coord, tile_shape, seqlen_k)
            result = cutlass.min(
                trip_count,
                cute.ceil_div(tile_shape[0], tile_shape[1]),
            )
        return result

    @cute.jit
    def get_unmasked_trip_count(
        self,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_k: Int32,
    ) -> Int32:
        result = 0
        if self.mask_type == MaskType.NO_MASK:
            result = self.get_trip_count(blk_coord, tile_shape, seqlen_k)
        elif self.mask_type == MaskType.RESIDUAL_MASK:
            if seqlen_k % tile_shape[1] != 0:
                result = self.get_trip_count(blk_coord, tile_shape, seqlen_k) - 1
            else:
                result = self.get_trip_count(blk_coord, tile_shape, seqlen_k)
        elif self.mask_type == MaskType.CAUSAL_MASK:
            result = self.get_trip_count(
                blk_coord, tile_shape, seqlen_k
            ) - self.get_masked_trip_count(blk_coord, tile_shape, seqlen_k)
        return result

    @cute.jit
    def apply_mask(
        self,
        acc_qk: cute.Tensor,
        index_qk: cute.Tensor,
        seqlen_k: Int32,
    ):
        if self.mask_type == MaskType.RESIDUAL_MASK:
            for i in range(cute.size(acc_qk)):
                pos = index_qk[i]
                if pos[1] >= seqlen_k:
                    acc_qk[i] = -Float32.inf
        elif self.mask_type == MaskType.CAUSAL_MASK:
            for i in range(cute.size(acc_qk)):
                pos = index_qk[i]
                if pos[0] < pos[1] or pos[1] >= seqlen_k:
                    acc_qk[i] = -Float32.inf

    @staticmethod
    def _compute_grid(
        o_shape: cute.Shape,
        cta_tiler: Tuple[int, int, int],
        is_persistent: bool,
    ) -> Tuple[FmhaStaticTileSchedulerParams, Tuple[int, int, int]]:
        tile_sched_params = create_fmha_static_tile_scheduler_params(
            is_persistent,
            problem_shape_mbh=( # actually is (m, h, b)
                cute.ceil_div(cute.size(o_shape[0]), cta_tiler[0]), # pM2048 // tileM128 = 16
                cute.size(o_shape[2][0]), # h4
                cute.size(o_shape[2][1]), # b2
            ),
        )
        grid = FmhaStaticTileScheduler.get_grid_shape(tile_sched_params)
        return tile_sched_params, grid


def run(
    q_shape: Tuple[int, int, int, int] | Tuple[int, Tuple[int, ...], int, int],
    k_shape: Tuple[int, int, int, int] | Tuple[int, Tuple[int, ...], int, int],
    in_dtype: Type[cutlass.Numeric],
    out_dtype: Type[cutlass.Numeric],
    qk_acc_dtype: Type[cutlass.Numeric],
    pv_acc_dtype: Type[cutlass.Numeric],
    mma_tiler_mn: Tuple[int, int],
    is_persistent: bool,
    is_causal: bool,
    scale_q: float,
    scale_k: float,
    scale_v: float,
    inv_scale_o: float,
    scale_softmax: float,
    tolerance: float,
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
    use_cold_l2: bool = False,
    **kwargs,
):
    """Execute Fused Multi-Head Attention (FMHA) on Blackwell architecture and validate results.

    This function creates random input tensors for query, key, and value, then performs the
    complete FMHA computation pipeline. It supports configurable data types, tiling parameters,
    and various attention masking options. Results can be validated against a PyTorch reference
    implementation or run multiple times for performance measurement.

    The implementation leverages specialized tensor memory operations and efficient math
    operations optimized for Blackwell architecture, including pipelined computation stages
    for maximum throughput.

    :param q_shape: Query tensor shape (B, S_q, H, D) where B=batch size, S_q=query sequence length,
                    H=number of heads, D=head dimension.
                    If S_q is a tuple, it is the variable sequence length.
    :type q_shape: Tuple[int, int, int, int] | Tuple[int, Tuple[int, ...], int, int]
    :param k_shape: Key tensor shape (B, S_k, H_k, D) where B=batch size, S_k=key sequence length,
                    H_k=number of key heads (H must be divisible by H_k), D=head dimension.
                    If S_k is a tuple, it is the variable sequence length.
    :type k_shape: Tuple[int, int, int, int] | Tuple[int, Tuple[int, ...], int, int]
    :param in_dtype: Input data type for query, key and value tensors
    :type in_dtype: Type[cutlass.Numeric]
    :param out_dtype: Output data type for attention output
    :type out_dtype: Type[cutlass.Numeric]
    :param qk_acc_dtype: Accumulator data type for query-key matrix multiplication
    :type qk_acc_dtype: Type[cutlass.Numeric]
    :param pv_acc_dtype: Accumulator data type for probability-value matrix multiplication
    :type pv_acc_dtype: Type[cutlass.Numeric]
    :param mma_tiler_mn: Matrix multiply accumulate tile shape (M, N)
    :type mma_tiler_mn: Tuple[int, int]
    :param is_persistent: Whether to use persistent kernel optimization
    :type is_persistent: bool
    :param is_causal: Whether to apply causal masking
    :type is_causal: bool
    :param scale_q: Scaling factor for query tensor
    :type scale_q: float
    :param scale_k: Scaling factor for key tensor
    :type scale_k: float
    :param scale_v: Scaling factor for value tensor
    :type scale_v: float
    :param inv_scale_o: Inverse scaling factor for output tensor
    :type inv_scale_o: float
    :param scale_softmax: Attention score scaling factor (defaults to 1/sqrt(D) if set to 0)
    :type scale_softmax: float
    :param tolerance: Maximum acceptable error for validation
    :type tolerance: float
    :param warmup_iterations: Number of warmup iterations
    :type warmup_iterations: int
    :param iterations: Number of iterations to run for performance testing
    :type iterations: int
    :param skip_ref_check: Skip validation against reference implementation
    :type skip_ref_check: bool
    :param use_cold_l2: Whether to use circular buffer strategy to ensure cold L2 cache
    :type use_cold_l2: bool

    :raises ValueError: If input shapes are incompatible or head dimension is unsupported
    :raises RuntimeError: If GPU is unavailable for computation
    :return: Execution time of the FMHA kernel in microseconds
    :rtype: float
    """

    if DEBUG_MODE:
        print(f"Running Blackwell SM100 FMHA test with:")
        print(f"  q_shape: {q_shape}")
        print(f"  k_shape: {k_shape}")
        print(f"  in_dtype: {in_dtype}")
        print(f"  out_dtype: {out_dtype}")
        print(f"  qk_acc_dtype: {qk_acc_dtype}")
        print(f"  pv_acc_dtype: {pv_acc_dtype}")
        print(f"  mma_tiler_mn: {mma_tiler_mn}")
        print(f"  is_persistent: {is_persistent}")
        print(f"  is_causal: {is_causal}")
        print(f"  scale_q: {scale_q}")
        print(f"  scale_k: {scale_k}")
        print(f"  scale_v: {scale_v}")
        print(f"  inv_scale_o: {inv_scale_o}")
        print(f"  scale_softmax: {scale_softmax}")
        print(f"  tolerance: {tolerance}")
        print(f"  warmup_iterations: {warmup_iterations}")
        print(f"  iterations: {iterations}")
        print(f"  skip_ref_check: {skip_ref_check}")
        print(f"  use_cold_l2: {use_cold_l2}")

    # Unpack parameters
    b, s_q, h_q, d = q_shape
    b_, s_k, h_k, d_ = k_shape

    if b != b_:
        raise ValueError("q & k must have the same batch size")

    if d != d_:
        raise ValueError("q & k must have the same head dimension")

    if d not in {32, 64, 128}:
        raise ValueError("head dimension must be 32, 64, or 128")

    if h_q % h_k != 0:
        raise ValueError("h_q must be divisible by h_k")

    if isinstance(s_q, tuple) and len(s_q) != b:
        raise ValueError("variable_seqlen s_q must have the length of batch size")
    if isinstance(s_k, tuple) and len(s_k) != b:
        raise ValueError("variable_seqlen s_k must have the length of batch size")

    if in_dtype not in {cutlass.Float8E4M3FN, cutlass.Float16}:
        raise ValueError("in_dtype must be Float8E4M3FN or Float16")

    if out_dtype not in {cutlass.Float8E4M3FN, cutlass.Float16}:
        raise ValueError("out_dtype must be Float8E4M3FN or Float16")

    if qk_acc_dtype not in {Float32}:
        raise ValueError("qk_acc_dtype must be Float32")

    if pv_acc_dtype not in {Float32}:
        raise ValueError("pv_acc_dtype must be Float32")

    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    h_r = h_q // h_k

    # Prepare pytorch tensors: Q, K, V (random from 0 to 2) and O (all zero)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    def create_cumulative_sequence_lengths(s):
        s_cumsum = [0]
        for i in range(len(s)):
            s_cumsum.append(s_cumsum[-1] + s[i])

        s_cumsum_cute_tensor, s_cumsum_torch_tensor = cutlass_torch.cute_tensor_like(
            torch.tensor(s_cumsum, dtype=torch.int32),
            Int32,
            is_dynamic_layout=True,
            assumed_align=16,
        )

        return s_cumsum_cute_tensor, s_cumsum_torch_tensor

    cum_seqlen_q, cum_seqlen_q_torch = (
        create_cumulative_sequence_lengths(s_q)
        if isinstance(s_q, tuple)
        else (None, None)
    )
    cum_seqlen_k, cum_seqlen_k_torch = (
        create_cumulative_sequence_lengths(s_k)
        if isinstance(s_k, tuple)
        else (None, None)
    )

    def create_and_pad_tensor(
        shape, padding, dtype, s_cumsum=None, is_dynamic_layout=True
    ):
        # (b, s, h, d)
        shape_ = tuple(map(lambda x, y: x + y, shape, padding))
        if s_cumsum is not None:
            if shape_[0] != 1 or padding[0] != 0:
                raise ValueError("Invalid tensor creation for variable sequence length")
            # (s_total + padding, h, d)
            shape_ = shape_[1:]
            padding = padding[1:]

        # Create f32 torch tensor (cpu)
        f32_torch_tensor_full = cutlass_torch.create_and_permute_torch_tensor(
            shape_,
            torch.float32,
            permute_order=None,
            init_type=cutlass.torch.TensorInitType.RANDOM,
            init_config=cutlass.torch.RandomInitConfig(
                min_val=-2 if dtype.is_float or dtype.signed else 0, max_val=2
            ),
        )
        # Create dtype cute & torch tensor (gpu)
        _, torch_tensor_full = cutlass_torch.cute_tensor_like(
            f32_torch_tensor_full,
            dtype,
            is_dynamic_layout,
            assumed_align=16,
        )

        # Offset the tensor
        slices = tuple(slice(s, e) for s, e in zip(padding, shape_))
        torch_tensor = torch_tensor_full[slices].detach()
        f32_torch_tensor = f32_torch_tensor_full[slices].detach()
        torch_tensor._keep_alive = torch_tensor_full
        f32_torch_tensor._keep_alive = f32_torch_tensor_full

        # Create dtype cute tensor with offset (gpu)
        cute_tensor = from_dlpack(torch_tensor, assumed_align=16)
        cute_tensor.element_type = dtype

        # From ragged to jagged
        if s_cumsum is not None:
            torch_tensor = torch.nested.nested_tensor_from_jagged(
                values=torch_tensor, offsets=s_cumsum
            )
            f32_torch_tensor = torch.nested.nested_tensor_from_jagged(
                values=f32_torch_tensor, offsets=s_cumsum.cpu()
            )

        return (
            f32_torch_tensor,
            cute_tensor,
            torch_tensor,
        )

    qo_shape = (b, s_q, h_r * h_k, d)
    kv_shape = (b, s_k, h_k, d)
    qo_padding = (0, 0, 0, 0, 0)
    kv_padding = (0, 0, 0, 0, 0)

    if isinstance(s_q, tuple):
        qo_shape = (1, sum(s_q), h_r * h_k, d)
        qo_padding = (0, max(s_q), 0, 0, 0)

    if isinstance(s_k, tuple):
        kv_shape = (1, sum(s_k), h_k, d)
        kv_padding = (0, max(s_k), 0, 0, 0)

    q_ref, q_tensor, q_torch = create_and_pad_tensor(
        qo_shape,
        qo_padding,
        in_dtype,
        s_cumsum=cum_seqlen_q_torch,
        is_dynamic_layout=True,
    )
    k_ref, k_tensor, k_torch = create_and_pad_tensor(
        kv_shape,
        kv_padding,
        in_dtype,
        s_cumsum=cum_seqlen_k_torch,
        is_dynamic_layout=True,
    )
    v_ref, v_tensor, v_torch = create_and_pad_tensor(
        kv_shape,
        kv_padding,
        in_dtype,
        s_cumsum=cum_seqlen_k_torch,
        is_dynamic_layout=True,
    )
    _, o_tensor, o_torch = create_and_pad_tensor(
        qo_shape,
        qo_padding,
        out_dtype,
        s_cumsum=cum_seqlen_q_torch,
        is_dynamic_layout=True,
    )

    mma_tiler = (*mma_tiler_mn, d)

    mask_type = MaskType.NO_MASK
    if is_causal:
        mask_type = MaskType.CAUSAL_MASK
    else:
        if isinstance(s_k, tuple):
            for i in range(len(s_k)):
                if s_k[i] % mma_tiler_mn[1] != 0:
                    mask_type = MaskType.RESIDUAL_MASK
        else:
            if s_k % mma_tiler_mn[1] != 0:
                mask_type = MaskType.RESIDUAL_MASK

    fmha = BlackwellFusedMultiHeadAttentionForward(
        qk_acc_dtype,
        pv_acc_dtype,
        mma_tiler,
        is_persistent,
        mask_type,
        debug_print=DEBUG_MODE,
    )

    # Initialize Stream
    current_stream = cutlass_torch.default_stream()

    if scale_softmax == 0.0:  # default to 1/sqrt(d)
        scale_softmax = 1.0 / math.sqrt(d)
    log2_e = math.log2(
        math.exp(1.0)
    )  # gpu uses exp2 for perf concerns, we need an extra factor 'log2_e' here

    scale_softmax = scale_q * scale_k * scale_softmax
    scale_softmax_log2 = scale_softmax * log2_e
    scale_output = scale_v * inv_scale_o

    problem_size = (
        b,
        max(s_q) if isinstance(s_q, tuple) else s_q,
        max(s_k) if isinstance(s_k, tuple) else s_k,
        h_q,
        h_k,
        d,
    )

    if DEBUG_MODE:
        print("Compiling kernel with cute.compile ...")
    start_time = time.time()
    # compile fmha kernel
    compiled_fmha = cute.compile(
        fmha,
        q_tensor.iterator,
        k_tensor.iterator,
        v_tensor.iterator,
        o_tensor.iterator,
        problem_size,
        cum_seqlen_q,
        cum_seqlen_k,
        scale_softmax_log2,
        scale_output,
        current_stream,
    )
    compilation_time = time.time() - start_time
    if DEBUG_MODE:
        print(f"Compilation time: {compilation_time:.4f} seconds")

    def run_torch_fmha(q, k, v, scale_softmax=1.0, scale_output=1.0, is_causal=False):
        h_q = q.shape[2]
        h_k = k.shape[2]

        # as we initialize q, k, v with shape (b, s, h, d) and SDPA of torch needs them to be (b, h, s, d)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # For the situation that torch has not supported, we need to handle it manually
        situation1 = (h_q != h_k or is_causal) and (q.is_nested or k.is_nested)
        situation2 = (q.is_nested and not k.is_nested) or (
            not q.is_nested and k.is_nested
        )
        if situation1 or situation2:
            # Once torch supports the situation, we can remove this fallback
            batch_size = q.size(0)
            ref_list = []
            for batch_idx in range(batch_size):
                q_i = q[batch_idx]
                k_i = k[batch_idx]
                v_i = v[batch_idx]

                ref_i = F.scaled_dot_product_attention(
                    q_i,
                    k_i,
                    v_i,
                    attn_mask=None,
                    dropout_p=0.0,
                    scale=scale_softmax,
                    is_causal=is_causal,
                    enable_gqa=(h_q != h_k),
                )
                ref_i = ref_i.transpose(0, 1) * scale_output
                ref_list.append(ref_i)
            if q.is_nested:
                ref = torch.nested.nested_tensor(ref_list, layout=torch.jagged)
            else:
                ref = torch.stack(ref_list)
        else:
            ref = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                scale=scale_softmax,
                is_causal=is_causal,
                enable_gqa=(h_q != h_k),
            )
            ref = ref.transpose(1, 2) * scale_output
        return ref

    if not skip_ref_check:
        # Execute kernel once for reference checking
        if DEBUG_MODE:
            print("Executing FMHA kernel for reference check ...")
        compiled_fmha(
            q_tensor.iterator,
            k_tensor.iterator,
            v_tensor.iterator,
            o_tensor.iterator,
            problem_size,
            cum_seqlen_q,
            cum_seqlen_k,
            scale_softmax_log2,
            scale_output,
            current_stream,
        )
        if DEBUG_MODE:
            print("Verifying results...")
        o_ref = run_torch_fmha(
            q_ref, k_ref, v_ref, scale_softmax, scale_output, is_causal
        )

        if o_ref.is_nested:
            o_ref = o_ref.values()

        if o_torch.is_nested:
            o_torch = o_torch.values()

        # convert o back to f32 for comparison
        o_fp32, o_fp32_torch = cutlass_torch.cute_tensor_like(
            torch.empty(*o_torch.shape, dtype=torch.float32),
            Float32,
            is_dynamic_layout=True,
            assumed_align=16,
        )
        cute.testing.convert(o_tensor, o_fp32)
        o_result = o_fp32_torch.cpu()

        if out_dtype.is_float and out_dtype.width <= 8:
            ref_narrow_precision, _ = cutlass_torch.cute_tensor_like(
                torch.empty(*o_ref.shape, dtype=torch.uint8),
                out_dtype,
                is_dynamic_layout=True,
                assumed_align=16,
            )

            ref_o_f32, ref_o_f32_torch = cutlass_torch.cute_tensor_like(
                o_ref,
                cutlass.Float32,
                is_dynamic_layout=True,
                assumed_align=16,
            )

            # convert ref : f32 -> fp4/fp8 -> f32
            cute.testing.convert(ref_o_f32, ref_narrow_precision)
            cute.testing.convert(ref_narrow_precision, ref_o_f32)

            o_ref = ref_o_f32_torch.cpu()

            # override tolerance
            tolerance = 0.13

        # Assert close results
        torch.testing.assert_close(o_result, o_ref, atol=tolerance, rtol=1e-05)
        print("Results verified successfully!")

    def generate_tensors():
        _, q_tensor_workspace, _ = create_and_pad_tensor(
            qo_shape,
            qo_padding,
            in_dtype,
            s_cumsum=cum_seqlen_q_torch,
            is_dynamic_layout=True,
        )
        _, k_tensor_workspace, _ = create_and_pad_tensor(
            kv_shape,
            kv_padding,
            in_dtype,
            s_cumsum=cum_seqlen_k_torch,
            is_dynamic_layout=True,
        )
        _, v_tensor_workspace, _ = create_and_pad_tensor(
            kv_shape,
            kv_padding,
            in_dtype,
            s_cumsum=cum_seqlen_k_torch,
            is_dynamic_layout=True,
        )
        _, o_tensor_workspace, _ = create_and_pad_tensor(
            qo_shape,
            qo_padding,
            out_dtype,
            s_cumsum=cum_seqlen_q_torch,
            is_dynamic_layout=True,
        )
        return testing.JitArguments(
            q_tensor_workspace.iterator,
            k_tensor_workspace.iterator,
            v_tensor_workspace.iterator,
            o_tensor_workspace.iterator,
            problem_size,
            cum_seqlen_q,
            cum_seqlen_k,
            scale_softmax_log2,
            scale_output,
            current_stream,
        )

    workspace_count = 1
    if use_cold_l2:
        q_torch_effective = q_torch.values() if q_torch.is_nested else q_torch
        k_torch_effective = k_torch.values() if k_torch.is_nested else k_torch
        v_torch_effective = v_torch.values() if v_torch.is_nested else v_torch
        o_torch_effective = o_torch.values() if o_torch.is_nested else o_torch
        one_workspace_bytes = (
            q_torch_effective.numel() * q_torch_effective.element_size()
            + k_torch_effective.numel() * k_torch_effective.element_size()
            + v_torch_effective.numel() * v_torch_effective.element_size()
            + o_torch_effective.numel() * o_torch_effective.element_size()
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = testing.benchmark(
        compiled_fmha,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    if PROFILE_MODE:
        import sys
        sys.path.insert(0, "..")
        from nvtx import switch_profile, add_nvtx_event

        b, sq, hq, d = q_shape
        _, sk, _, _ = k_shape
        flops = b * sq * sk * hq * d * 4  # 4 for 2 matmul
        flops = flops // 2 if is_causal else flops  # causal will have half flops on average
        
        event_str = (
            f"fmha_fwd ({b=},{sq=},{sk=},{hq=},{d=},{is_causal=},{flops=})"
        )
        iters, start, end = 10, 6, 9
        for i in range(iters):
            switch_profile(iter_id=i, start=start, end=end)
            with add_nvtx_event(event_str):
                compiled_fmha(
                    q_tensor.iterator,
                    k_tensor.iterator,
                    v_tensor.iterator,
                    o_tensor.iterator,
                    problem_size,
                    cum_seqlen_q,
                    cum_seqlen_k,
                    scale_softmax_log2,
                    scale_output,
                    current_stream,
                )

    return exec_time  # Return execution time in microseconds


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str):
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    def parse_nested_comma_separated_ints(s: str):
        try:
            s = s.strip()
            if "(" not in s:
                return tuple(int(x.strip()) for x in s.split(","))

            start = s.find("(")
            end = s.find(")")
            if start == -1 or end == -1:
                raise ValueError("Mismatched parentheses")

            before = s[:start].strip().rstrip(",")
            middle = s[start + 1 : end].strip()
            after = s[end + 1 :].strip().lstrip(",")

            result = []
            if before:
                result.extend(int(x.strip()) for x in before.split(","))

            if middle:
                nested_tuple = tuple(int(x.strip()) for x in middle.split(","))
                result.append(nested_tuple)

            if after:
                result.extend(int(x.strip()) for x in after.split(","))

            return tuple(result)

        except ValueError as e:
            if str(e) == "Mismatched parentheses":
                raise argparse.ArgumentTypeError("Mismatched parentheses in input")
            else:
                raise argparse.ArgumentTypeError(
                    "Invalid format. Expected comma-separated integers with optional parentheses for nested tuple."
                )

    parser = argparse.ArgumentParser(description="Example of FMHA on Blackwell.")

    parser.add_argument(
        "--in_dtype",
        type=cutlass.dtype,
        default=cutlass.Float16,
        help="Input data type",
    )

    parser.add_argument(
        "--out_dtype",
        type=cutlass.dtype,
        default=cutlass.Float16,
        help="Output data type",
    )

    parser.add_argument(
        "--qk_acc_dtype",
        type=cutlass.dtype,
        default=Float32,
        help="QK accumulator data type",
    )

    parser.add_argument(
        "--pv_acc_dtype",
        type=cutlass.dtype,
        default=Float32,
        help="PV accumulator data type",
    )

    parser.add_argument(
        "--mma_tiler_mn",
        type=parse_comma_separated_ints,
        default=(128, 128),
        help="MMA tile shape (M, N)",
    )

    parser.add_argument(
        "--is_persistent",
        action="store_true",
        help="Is persistent",
    )

    parser.add_argument(
        "--is_causal",
        action="store_true",
        help="Whether to use casual mask",
    )

    parser.add_argument(
        "--q_shape",
        type=parse_nested_comma_separated_ints,
        default=(1, 256, 8, 128),
        help="Shape of Q (B, S_q, H, D)",
    )

    parser.add_argument(
        "--k_shape",
        type=parse_nested_comma_separated_ints,
        default=(1, 256, 8, 128),
        help="Shape of K (B, S_k, H_k, D)",
    )

    parser.add_argument(
        "--scale_q",
        type=float,
        default=1.0,
        help="Scaling factors to dequantize Q",
    )

    parser.add_argument(
        "--scale_k",
        type=float,
        default=1.0,
        help="Scaling factors to dequantize K",
    )

    parser.add_argument(
        "--scale_v",
        type=float,
        default=1.0,
        help="Scaling factors to dequantize V",
    )

    parser.add_argument(
        "--inv_scale_o",
        type=float,
        default=1.0,
        help="Scaling factor to quantize O",
    )

    parser.add_argument(
        "--scale_softmax",
        type=float,
        default=0.0,
        help="Scaling factor to scale S (i.e. Q*K); if zero, defaults to 1/sqrt(D)",
    )

    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )

    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=0,
        help="Number of iterations for warmup",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations after warmup",
    )

    parser.add_argument(
        "--skip_ref_check",
        action="store_true",
        help="Skip reference check",
    )

    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )

    args = parser.parse_args()

    if len(args.q_shape) != 4:
        parser.error("--q_shape must contain exactly 4 values")

    if len(args.k_shape) != 4:
        parser.error("--k_shape must contain exactly 4 values")

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    run(
        args.q_shape,
        args.k_shape,
        args.in_dtype,
        args.out_dtype,
        args.qk_acc_dtype,
        args.pv_acc_dtype,
        args.mma_tiler_mn,
        args.is_persistent,
        args.is_causal,
        args.scale_q,
        args.scale_k,
        args.scale_v,
        args.inv_scale_o,
        args.scale_softmax,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
    )

    print("PASS")
