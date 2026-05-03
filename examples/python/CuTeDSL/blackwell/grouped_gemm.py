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
import functools
import os
from typing import List, Type, Union
from inspect import isclass

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.torch as cutlass_torch
from cutlass import const_expr

"""
A grouped GEMM example for the NVIDIA Blackwell SM100 architecture using CUTE DSL

This example demonstrates an implementation of grouped GEMM using a TMA plus Blackwell SM100 TensorCore
warp-specialized persistent kernel.
The grouped GEMM workload computes a batch of GEMM operations with distinct problem sizes. Pointers to matrices
in global memory are passed to the kernel in an array (also held in global memory). Similarly, problem shapes and
strides are also stored in arrays in GMEM.

This differs from "Batched Array" GEMM since the size of each GEMM problem in the grouped GEMM concept may be distinct.

To run this example:

.. code-block:: bash

    python examples/blackwell/grouped_gemm.py                                                 \
      --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                                \
      --mma_tiler_mn 128,64 --cluster_shape_mn 1,1                                            \
      --problem_sizes_mnkl "(8192,1280,32,1),(16,384,1536,1),(640,1280,16,1),(640,160,16,1)"  \
      --num_groups 4  --tensormap_update_mode SMEM

The above example command makes 4 groups of different m, n, k sizes. The Blackwell tcgen05 MMA tile shape
is specified as (128, 64) and the cluster shape is (1,1). The input, mma accumulator and output data type
are set as fp16, fp32 and fp16, respectively.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/blackwell/grouped_gemm.py                                             \
      --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                                \
      --mma_tiler_mn 128,64 --cluster_shape_mn 1,1                                            \
      --problem_sizes_mnkl "(8192,1280,32,1),(16,384,1536,1),(640,1280,16,1),(640,160,16,1)"  \
      --num_groups 4  --tensormap_update_mode SMEM                                            \
      --warmup_iterations 1 --iterations 10 --skip_ref_check

There are some constrains for this example. Besides the constrains from the Blackwell dense GEMM persistent example,
there are also the following constrains:
* Only fp16 and bf16 data types are supported as inputs.
* Output data types could be fp16, bf16 or fp32.
* The contiguous dimension of each tensor must be at least 16 bytes aligned.
* The l mode(aka, batch size) for each group must be 1.
* The majorness for A, B and C must be the same across all groups.

--------------------------------------------------------------------------------
TensorMap (TMA Descriptor) Lifecycle per Warp — GMEM Update Mode
--------------------------------------------------------------------------------
A TMA descriptor (tensormap, 128B) encodes base_ptr + shape + stride for one
tensor.  In grouped GEMM, each group has different tensors, so the descriptor
must be updated at runtime when the group changes.
In GMEM mode the descriptor lives in GMEM and is updated in-place.
In SMEM mode the update is staged through an SMEM buffer to hide GMEM latency.

[Epilog warps (warp 0-3)] — manage tensormap C
    # --- kernel startup ---
    init_tensormap_from_atom(tma_atom_c, tensormap_c_ptr)  # copy initial descriptor to GMEM
    fence_tensormap_initialization()                        # fence.acq_rel.cta: make GMEM write visible

    # --- persistent tile loop ---
    while tile:
        if group_changed:
            update_tensormap(real_c, tma_atom_c,            # wait in-flight TMA, then overwrite GMEM descriptor
                             tensormap_c_ptr, ...)
            fence_tensormap_update(tensormap_c_ptr)         # fence_tma_desc_acquire: ensure TMA HW sees new desc
        cute.copy(tma_atom_c, ...,
                  tma_desc_ptr=tensormap_c_ptr)             # TMA store C using updated descriptor

[TMA warp (warp 5)] — manage tensormap A and B
    # --- kernel startup ---
    init_tensormap_from_atom(tma_atom_a, tensormap_a_ptr)  # copy initial descriptor to GMEM
    init_tensormap_from_atom(tma_atom_b, tensormap_b_ptr)  # copy initial descriptor to GMEM

    # --- persistent tile loop ---
    while tile:
        if group_changed:
            if first_group:
                fence_tensormap_initialization()            # ensure init writes are visible before first update
            update_tensormap(real_a, real_b, tma_atom_a,   # wait in-flight TMA, then overwrite GMEM descriptors
                             tma_atom_b, ...)
            fence_tensormap_update(tensormap_a_ptr)         # fence_tma_desc_acquire for A
            fence_tensormap_update(tensormap_b_ptr)         # fence_tma_desc_acquire for B
        cute.copy(tma_atom_a, ...,
                  tma_desc_ptr=tensormap_a_ptr)             # TMA load A using updated descriptor
        cute.copy(tma_atom_b, ...,
                  tma_desc_ptr=tensormap_b_ptr)             # TMA load B using updated descriptor

--------------------------------------------------------------------------------
TensorMap (TMA Descriptor) Lifecycle per Warp — SMEM Update Mode
--------------------------------------------------------------------------------
In SMEM mode, descriptor updates are first written to an SMEM staging buffer,
then flushed back to GMEM via cp.fence.proxy.tensormap (release).
A/B descriptor initialization is delegated to the MMA warp so that the TMA
warp can overlap waiting on ab_empty barriers with the init work.
The TMA warp waits on a named barrier (tensormap_ab_init_bar) before proceeding
to the first update.

[Epilog warps (warp 0-3)] — manage tensormap C  (same as GMEM mode for C)
    # --- kernel startup ---
    init_tensormap_from_atom(tma_atom_c, tensormap_c_smem_ptr) # copy initial descriptor to SMEM
    fence_tensormap_initialization()                            # noop in SMEM mode (SMEM visibility via barrier)

    # --- persistent tile loop ---
    while tile:
        if group_changed:
            update_tensormap(real_c, tma_atom_c,               # write new desc to SMEM, wait in-flight TMA,
                             tensormap_c_ptr,                   # then cp.fence.proxy.tensormap release → GMEM
                             tensormap_c_smem_ptr, ...)
            fence_tensormap_update(tensormap_c_ptr)             # fence_tma_desc_acquire: ensure TMA HW sees new desc
        cute.copy(tma_atom_c, ...,
                  tma_desc_ptr=tensormap_c_ptr)                 # TMA store C using updated GMEM descriptor

[MMA warp (warp 4)] — init tensormap A and B on behalf of TMA warp
    # --- kernel startup (SMEM mode only, delegate_tensormap_ab_init=True) ---
    init_tensormap_from_atom(tma_atom_a, tensormap_a_smem_ptr) # copy initial descriptor to SMEM
    init_tensormap_from_atom(tma_atom_b, tensormap_b_smem_ptr) # copy initial descriptor to SMEM
    barrier.arrive(tensormap_ab_init_bar)                       # signal TMA warp: init complete

[TMA warp (warp 5)] — manage tensormap A and B
    # --- kernel startup ---
    barrier.wait(tensormap_ab_init_bar)                         # wait for MMA warp to finish init in SMEM

    # --- persistent tile loop ---
    while tile:
        if group_changed:
            if first_group:
                fence_tensormap_initialization()                # noop in SMEM mode
            update_tensormap(real_a, real_b, tma_atom_a,       # write new desc to SMEM, wait in-flight TMA,
                             tma_atom_b, tensormap_a_ptr,       # then cp.fence.proxy.tensormap release → GMEM
                             tensormap_b_ptr,
                             tensormap_a_smem_ptr,
                             tensormap_b_smem_ptr, ...)
            fence_tensormap_update(tensormap_a_ptr)             # fence_tma_desc_acquire for A
            fence_tensormap_update(tensormap_b_ptr)             # fence_tma_desc_acquire for B
        cute.copy(tma_atom_a, ...,
                  tma_desc_ptr=tensormap_a_ptr)                 # TMA load A using updated GMEM descriptor
        cute.copy(tma_atom_b, ...,
                  tma_desc_ptr=tensormap_b_ptr)                 # TMA load B using updated GMEM descriptor

--------------------------------------------------------------------------------
Why SMEM Mode Hides Latency: delegate_tensormap_ab_init Timeline
--------------------------------------------------------------------------------
The key difference is WHERE the A/B init work is placed relative to the TMA
warp's tile-scheduling work, and whether init writes go to SMEM or GMEM.

GMEM mode  (delegate=False, init written to GMEM by TMA warp itself):

  TMA warp:  [init_A→GMEM][init_B→GMEM] | [delinearize_z][fence_init][update_AB][fence_upd][TMA load]...
  MMA warp:  [idle (nothing to do yet) ] | [tmem_ptr_sync][mma work]...
              ^^^^^^^^^^^^^^^^^^^^^^^^
              TMA warp is blocked here writing 2x128B to slow GMEM before it can
              even start scheduling work.  The init cost is fully exposed.

SMEM mode  (delegate=True, init written to SMEM by MMA warp):

  TMA warp:  [delinearize_z ...]  (barrier_wait — usually no stall) [fence_init(noop)][update_AB][fence_upd][TMA load]...
  MMA warp:  [init_A→SMEM][init_B→SMEM][barrier_arrive] | [tmem_ptr_sync][mma work]...
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              MMA warp does the init concurrently while TMA warp runs
              delinearize_z (linear scan over group list = ALU work).
              By the time TMA warp reaches the barrier, init is already done.

Two compounding benefits:
  1. Overlap:  init latency is hidden behind TMA warp's scheduler ALU work.
  2. Faster write:  SMEM write (128B) is much faster than GMEM write (128B),
     so even if TMA warp does stall at the barrier, the wait is shorter.

--------------------------------------------------------------------------------
Why GMEM Mode Does NOT Delegate init to the MMA Warp
--------------------------------------------------------------------------------
One might ask: why not also delegate A/B init to the MMA warp in GMEM mode,
to get the same overlap benefit?

If we did delegate in GMEM mode:

  TMA warp:  [delinearize_z...] → [barrier_wait] ← still blocked on slow GMEM write
  MMA warp:  [init_A→GMEM (slow)] [init_B→GMEM (slow)] [barrier_arrive] | tmem work...

The GMEM write latency is unchanged regardless of which warp issues it.
Delegating only moves the work to a different warp — it does NOT create any
new overlap because:

  - The TMA warp's delinearize_z ALU time ≈ or < GMEM write latency,
    so TMA warp would still stall at the barrier waiting for the MMA warp
    to finish writing GMEM.
  - An extra named barrier synchronization point is introduced, adding overhead
    with no latency benefit.

The SMEM delegate trick works only because SMEM writes are so fast (a few
cycles for 128B) that they complete well within the time TMA warp spends on
its ALU-bound scheduler work — creating genuine overlap.
In GMEM mode, the slow GMEM write destroys this overlap opportunity entirely.
"""


DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"
PROFILE_MODE = os.environ.get("PROFILE_MODE", "0") == "1"


class GroupedGemmPersistentKernelSm100:
    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        tensormap_update_mode: utils.TensorMapUpdateMode = utils.TensorMapUpdateMode.SMEM,
        debug_print: bool = False,
    ):
        """Initializes the configuration for a Blackwell grouped GEMM kernel.

        Besides configurations for dense persistent GEMM, there is an extra config specific to grouped GEMM:

        Tensormap Update Mode:
        - tensormap_update_mode: Specifies whether the tensormap is
            updated in global memory(GMEM) or shared memory(SMEM).
           The 2 modes are functionally equivalent and the difference are:
            - We buffer 3 tensormaps in SMEM for A, B, and C tensors (each TMA descriptor takes 128B) when TMA updates performed on SMEM.
            - Performance varies between modes depending on problem size; optimal choice differs across workloads.

        :param acc_dtype: Data type of the accumulator.
        :type acc_dtype: type[cutlass.Numeric]
        :param use_2cta_instrs: Boolean, True to use cta_group=2 MMA variant.
        :type use_2cta_instrs: bool
        :param mma_tiler_mn: tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: tuple[int, int]
        :param cluster_shape_mn: tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: tuple[int, int]
        :param tensormap_update_mode: Mode for updating the tensormap (GMEM or SMEM), defaults to SMEM.
        :type tensormap_update_mode: utils.TensorMapUpdateMode, optional
        """
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn # (CM2, CN1)
        # K dimension is deferred in _setup_attributes
        self.mma_tiler_mnk = (*mma_tiler_mn, 1) # (tileM256, tileN128, tileK1)
        self.cta_group = (
            tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.tensormap_update_mode = tensormap_update_mode
        # Delegate tensormap ab initialization to MMA warp 
        # when SMEM mode is used for better latency hiding
        self.delegate_tensormap_ab_init = (
            tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
        )

        # TODO(REVIEW): why no support to multicast ?
        self.num_mcast_ctas_a = 1
        self.num_mcast_ctas_b = 1
        self.is_a_mcast = False
        self.is_b_mcast = False

        self.occupancy = 1 # we only want one CTA to reside on one SM
        
        self.buffer_align_bytes = 1024
        
        # Set specialized warp ids
        self.epilog_warp_id = (0, 1, 2, 3) # the first warp group forms the epilogue consumer warps
        self.mma_warp_id = 4 # a single warp for umma consumer / acc producer
        self.tma_warp_id = 5 # a single warp for tma producer
        
        self.epilogue_threads = 32 * len(self.epilog_warp_id)
        self.tmem_ptr_read_threads = 32 + self.epilogue_threads  # all threads in mma warp and epilogue warps can read tmem ptr from shared memory
        self.threads_per_cta = 32 + self.tmem_ptr_read_threads
        self.tensormap_init_threads = 32 + 32 # tma producer warp + umma consumer warp for tensormap initialization when delegated to mma warp
        
        # Set barrier id for cta sync, epilog sync, tmem ptr sync and tensormap update sync
        self.cta_sync_bar_id = 0
        self.epilog_sync_bar_id = 1
        self.tmem_ptr_sync_bar_id = 2
        # Barrier ID used by MMA/TMA warps to signal A/B tensormap initialization completion
        self.tensormap_ab_init_bar_id = 4
        
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

        self.debug_print = debug_print

        if const_expr(self.debug_print):
            print()
            print(f"Initialized GroupedGemmPersistentKernelSm100 with configurations:")
            print(f"  acc_dtype: {self.acc_dtype=}")
            print(f"  use_2cta_instrs: {self.use_2cta_instrs=}")
            print(f"  mma_tiler: {self.mma_tiler_mnk=}")
            print(f"  cluster_shape_mn: {self.cluster_shape_mn=}")
            print(f"  CTA group for MMA: {self.cta_group=}")
            print(f"  tensormap_update_mode: {self.tensormap_update_mode=}")
            print(f"  delegate_tensormap_ab_init: {self.delegate_tensormap_ab_init=}")
            print(f"  epilogue_threads: {self.epilogue_threads=}")
            print(f"  tmem_ptr_read_threads: {self.tmem_ptr_read_threads=}")
            print(f"  threads_per_cta: {self.threads_per_cta=}")
            print(f"  warp ids: {self.epilog_warp_id=}, {self.mma_warp_id=}, {self.tma_warp_id=}")
            print(f"  barrier ids: {self.cta_sync_bar_id=}, {self.epilog_sync_bar_id=}, {self.tmem_ptr_sync_bar_id=}, {self.tensormap_ab_init_bar_id=}")
            print(f"  smem_capacity: {self.smem_capacity=} bytes")
            print(f"  Occupancy: {self.occupancy=}")
            print(f"  Buffer alignment: {self.buffer_align_bytes=} bytes")
            print(f"  Bytes per tensormap: {self.bytes_per_tensormap=}")
            print(f"  Reserved smem for mbar: {self.reserved_smem_bytes} bytes")
            print(f"  Reserved smem for tensormap management: {self.tensor_memory_management_bytes} bytes")
            print()

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        Most of the implementation follows standard dense GEMM patterns,
        with the key difference being additional consideration for SMEM
        buffer needed for tensormap updates.
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
        self.atom_thr_id = tiled_mma.thr_id # 1:0
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
            self.mma_tiler_mnk[0] // self.atom_thr_size,
            self.mma_tiler_mnk[1],
            self.mma_tiler_mnk[2],
        )
        self.cluster_tile_shape_mnk = tuple( # (CGA_tileM256, CGA_tileN128, CGA_tileK64)
            x * y for x, y in zip(self.cta_tile_shape_mnk, (*self.cluster_shape_mn, 1))
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide( # ((2),1,1,1):((1),0,0,0)
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (self.atom_thr_shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2]) # 1, along N dim
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1]) # 1, along M dim
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
        self.epi_tile = utils.compute_epilogue_tile_shape( # (epi_tileM128:1, epi_tileN32:1)
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_epi_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.smem_capacity,
            self.occupancy,
            debug_print=self.debug_print,
        )

        # Compute A/B/C shared memory layout
        # sA: S<3,4,3> o 0 o (MMA=(128,16),MMA_M=1,MMA_K=4,MMA_STAGES=8):((64,1),0,16,8192)
        # sB: S<3,4,3> o 0 o (MMA=(64,16),MMA_N=1,MMA_K=4,MMA_STAGES=8):((64,1),0,16,4096)
        # sC: S<2,4,3> o 0 o (epi_tileM=(8,16), epi_tileN=(32,1), epi_stages=(1,4)):((32,256),(1,0),(0,4096))
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
        self.epi_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_epi_stage,
        )

        tensor_smem_bytes = self._get_tensor_smem_bytes(
            self.a_smem_layout_staged,
            self.a_dtype,
            self.b_smem_layout_staged,
            self.b_dtype,
            self.epi_smem_layout_staged,
            self.c_dtype,
        )
        mbar_smem_bytes = self._get_mbar_smem_bytes(
            num_acc_stage=self.num_acc_stage,
            num_ab_stage=self.num_ab_stage,
            num_epi_stage=self.num_epi_stage,
        )
        tensormap_smem_bytes = self._get_tensormap_smem_bytes(
            self.tensormap_update_mode
        )
        if (
            mbar_smem_bytes
            + tensormap_smem_bytes
            + GroupedGemmPersistentKernelSm100.tensor_memory_management_bytes
            > self.reserved_smem_bytes
        ):
            raise ValueError(
                f"smem consumption for mbar and tensormap {mbar_smem_bytes + tensormap_smem_bytes} exceeds the "
                f"reserved smem bytes {self.reserved_smem_bytes}"
            )

        # Compute the number of tensor memory allocation columns
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols( # tileN128 x num_acc_stages2 = 256 cols
            tiled_mma, self.mma_tiler_mnk, self.num_acc_stage
        )

        if const_expr(self.debug_print):
            print()
            print(f"Setup attributes dependent on GEMM inputs:")
            print(f"  MMA tiler (M, N, K): {self.mma_tiler_mnk=}")
            print(f"  CTA tile shape (M, N, K): {self.cta_tile_shape_mnk=}")
            print(f"  Cluster tile shape (M, N, K): {self.cluster_tile_shape_mnk=}")
            print(f"  Cluster layout: {self.cluster_layout_vmnk=}")
            print(f"  Number of multicast CTAs for A: {self.num_mcast_ctas_a=}")
            print(f"  Number of multicast CTAs for B: {self.num_mcast_ctas_b=}")
            print(f"  Epilogue tile shape: {self.epi_tile=}")
            print(f"  Number of accumulator stages: {self.num_acc_stage=}")
            print(f"  Number of A/B stages: {self.num_ab_stage=}")
            print(f"  Number of epi stages: {self.num_epi_stage=}")
            print(f"  Number of tmem alloc cols: {self.num_tmem_alloc_cols=}")
            print(f"  Tensor memory smem bytes: {tensor_smem_bytes=}")
            print(f"  Mbar smem bytes: {mbar_smem_bytes=}")
            print(f"  Tensormap smem bytes: {tensormap_smem_bytes=}")
            print()

            print()
            print(f"A SMEM layout (a_smem_layout_staged) (MMA,MMA_M,MMA_K,STAGE): {self.a_smem_layout_staged}")
            print(f"B SMEM layout (b_smem_layout_staged) (MMA,MMA_N,MMA_K,STAGE): {self.b_smem_layout_staged}")
            print(f"Epi SMEM layout (epi_smem_layout_staged) (EPI_M,EPI_N,STAGE): {self.epi_smem_layout_staged}")
            print()

            print()
            print("tiled_mma: ", tiled_mma, f"\n\nshape_mnk: {tiled_mma.shape_mnk}", f"thr_id.shape: {self.atom_thr_shape}")
            print()

    @cute.jit
    def __call__(
        self,
        initial_a: cute.Tensor,
        initial_b: cute.Tensor,
        initial_c: cute.Tensor,
        group_count: cutlass.Constexpr[int],
        problem_shape_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        total_num_clusters: cutlass.Constexpr[int],
        tensormap_cute_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        For grouped GEMM, tensor shapes, tensor strides, and tensor address are all provided
        by different tensors in global memory. The "initial" tensors only carry data type and
        majorness information.

        :param initial_a: Initial tensor A, used for data type and majorness information.
        :type initial_a: cute.Tensor
        :param initial_b: Initial tensor B, used for data type and majorness information.
        :type initial_b: cute.Tensor
        :param initial_c: Initial tensor C, used for data type and majorness information.
        :type initial_c: cute.Tensor
        :param group_count: The number of GEMM groups.
        :type group_count: cutlass.Constexpr[int]
        :param problem_shape_mnkl: Tensor containing the (M, N, K, L) shape for each group.
        :type problem_shape_mnkl: cute.Tensor
        :param strides_abc: Tensor containing the strides for A, B, and C for each group.
        :type strides_abc: cute.Tensor
        :param tensor_address_abc: Tensor containing the base addresses for A, B, and C for each group.
        :type tensor_address_abc: cute.Tensor
        :param total_num_clusters: Total number of clusters needed for all groups.
        :type total_num_clusters: cutlass.Constexpr[int]
        :param tensormap_cute_tensor: Tensor for storing tensormaps.
        :type tensormap_cute_tensor: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr[int]
        :param stream: CUDA stream for asynchronous execution.
        :type stream: cuda.CUstream
        :raises TypeError: If A and B data types do not match.
        """
        self.a_dtype = initial_a.element_type
        self.b_dtype = initial_b.element_type
        self.c_dtype = initial_c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(initial_a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(initial_b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(initial_c)
        if const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        tiled_mma = self.tiled_mma
        atom_thr_size = self.atom_thr_size

        # /////////////////////////////////////////////////////////////////////////////
        #  Setup TMA load for A
        # /////////////////////////////////////////////////////////////////////////////
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, self.atom_thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        
        # tma_atom_a: Src: (2,8192):(8192,1) | Dst: (2,8192):(8192,1), where CTA_tileM128 x tileK64 = 8192
        # tma_tensor_a: (pM_min=128,pK=1024,1):(1@1,1@0,1@2)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            initial_a,
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
        
        # tma_atom_b: Src: (2,4096):(4096,1) | Dst: (2,4096):(4096,1), where CTA_tileN64 x tileK64 = 4096
        # tma_tensor_b: (pN=4096, pK=1024,1):(1@1,1@0,1@2)
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            initial_b,
            b_smem_layout,
            self.mma_tiler_mnk,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # /////////////////////////////////////////////////////////////////////////////
        #  Setup TMA store for C
        # /////////////////////////////////////////////////////////////////////////////
        tma_atom_c = None
        tma_tensor_c = None
        c_cta_v_layout = cute.composition(
            cute.make_identity_layout(initial_c.shape), self.epi_tile
        )
        epi_smem_layout = cute.slice_(self.epi_smem_layout_staged, (None, None, 0))
        
        # tma_atom_c: Src: (1,4096):(0,1) | Dst: (1,4096):(0,1), where epi_tileM128 x epi_tileN32 = 4096
        # tma_tensor_c: (pM_min=128, pN4096,1):(1@1,1@0,1@2)
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            initial_c,
            epi_smem_layout,
            c_cta_v_layout,
        )

        self.tile_sched_params, grid = self._compute_grid( # (CGA_M2, 1, num_persist_clusters=74)
            total_num_clusters, self.cluster_shape_mn, max_active_clusters
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Define shared storage for kernel
        # /////////////////////////////////////////////////////////////////////////////
        self.size_tensormap_in_i64 = (
            0
            if self.tensormap_update_mode == utils.TensorMapUpdateMode.GMEM
            else self.num_tensormaps
            * self.bytes_per_tensormap
            // 8
        )

        @cute.struct
        class SharedStorage:
            # the smem buffer to hold tensormap when tensormap update performed on SMEM
            tensormap_buffer: cute.struct.MemRange[
                cutlass.Int64, self.size_tensormap_in_i64
            ]
            
            # mainloop full/empty mbar array ptrs for each ab stage
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            
            # tmem accumulation full/empty mbar for each acc stage
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            
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
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.epi_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            

        self.shared_storage = SharedStorage

        if const_expr(self.debug_print):
            print()
            print(f"{self.a_dtype=}, {self.b_dtype=}, {self.c_dtype=}, {self.a_major_mode=}, {self.b_major_mode=}, {self.c_layout=}")
            print(f"{a_copy_size=}, {b_copy_size=}, {self.num_tma_load_bytes=}")
            print(f"{self.tile_sched_params=}")
            print(f"{self.size_tensormap_in_i64=}")
            print()

            print()
            print("TMA A: a_op: ", a_op, "\ntma_atom_a: ", tma_atom_a)
            print()
            print("TMA B: b_op: ", b_op, "\ntma_atom_b: ", tma_atom_b)
            print()
            print("TMA C: tma_atom_c: ", tma_atom_c)
            print()

            cute.printf("")
            cute.printf("tma_tensor_a: {}", tma_tensor_a)
            cute.printf("")
            cute.printf("tma_tensor_b: {}", tma_tensor_b)
            cute.printf("")
            cute.printf("tma_tensor_c: {}", tma_tensor_c)
            cute.printf("")
            cute.printf("total_num_clusters: {}", total_num_clusters)
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
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            group_count,
            problem_shape_mnkl,
            strides_abc,
            tensor_address_abc,
            tensormap_cute_tensor,
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
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        group_count: cutlass.Constexpr[int],
        problem_sizes_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        ptrs_abc: cute.Tensor,
        tensormaps: cute.Tensor,
    ):
        """
        GPU device kernel performing the grouped GEMM computation.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bid = cute.arch.block_idx()
        grid_dim = cute.arch.grid_dim()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = self.use_2cta_instrs
        
        # used only for debug print
        is_print_block = (bid[0] == 0) and (bid[1] == 0) and (bid[2] == 0) # pick a leader CTA
        is_print_thread = (tidx == 127) and is_print_block

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch tma descriptor
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        # /////////////////////////////////////////////////////////////////////////////
        #  Setup cta/thread coordinates
        # /////////////////////////////////////////////////////////////////////////////
        # Coord inside cluster
        mma_tile_coord_v = bid[0] % self.atom_thr_size # CTA idx in the CTA-pair
        is_leader_cta = mma_tile_coord_v == 0 # leader CTA in the CTA-pair
        cta_rank_in_cluster = cute.arch.make_warp_uniform( # CTA idx in the cluster, which might be different from mma_tile_coord_v if cluster size > 2
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord( # CTA (pair_v, rest_pair_xyz) coord in the cluster
            cta_rank_in_cluster
        )

        if const_expr(self.debug_print):
            if is_print_thread:
                cute.printf("")
                cute.printf("tidx: {}, warp_idx: {}, block_idx: ({}, {}, {})", tidx, warp_idx, bid[0], bid[1], bid[2])
                cute.printf("mma_tile_coord_v: {}, is_leader_cta: {}, cta_rank_in_cluster: {}", mma_tile_coord_v, is_leader_cta, cta_rank_in_cluster)

        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc and init: tensormap buffer, a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        # /////////////////////////////////////////////////////////////////////////////
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Fetch smem data ptrs
        tensormap_a_smem_ptr = None
        tensormap_b_smem_ptr = None
        tensormap_c_smem_ptr = None
        if const_expr(
            self.tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
        ):
            tensormap_smem_ptr = storage.tensormap_buffer.data_ptr()
            tensormap_a_smem_ptr = tensormap_smem_ptr
            tensormap_b_smem_ptr = (
                tensormap_a_smem_ptr + self.bytes_per_tensormap // 8
            )
            tensormap_c_smem_ptr = (
                tensormap_b_smem_ptr + self.bytes_per_tensormap // 8
            )
        ab_full_mbar_ptr = storage.ab_full_mbar_ptr.data_ptr()
        ab_empty_mbar_ptr = storage.ab_empty_mbar_ptr.data_ptr()
        acc_full_mbar_ptr = storage.acc_full_mbar_ptr.data_ptr()
        acc_empty_mbar_ptr = storage.acc_empty_mbar_ptr.data_ptr()
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_smem_buf = storage.tmem_holding_smem_buf

        # /////////////////////////////////////////////////////////////////////////////
        #  Initialize pipeline mbarriers manually
        # /////////////////////////////////////////////////////////////////////////////

        # Init barrier for loading A, B with TMA
        if warp_idx == self.epilog_warp_id[0]:
            for k_stage in range(self.num_ab_stage):
                num_tma_producer = 1
                num_tma_consumer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
                with cute.arch.elect_one():
                    # NOTE: only TMA producer of the leader CTA will arrive the ab full mbar,
                    # since the TMA's tx count of both CTAs will be automatically routed to the mbar of the leader CTA's umma warp by the hardware
                    cute.arch.mbarrier_init(ab_full_mbar_ptr + k_stage, num_tma_producer)
                    cute.arch.mbarrier_init(
                        ab_empty_mbar_ptr + k_stage, num_tma_consumer
                    )
        
        # Accumulator barrier init
        if warp_idx == self.mma_warp_id:
            for acc_stage in range(self.num_acc_stage):
                num_acc_producer = 1
                num_acc_consumer = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
                with cute.arch.elect_one():
                    # NOTE: only the umma consumer/acc producer of the leader CTA will arrive the acc full mbar,
                    # which uses the `tcgen05.commit`'s mcast_mask mechanism to multicast to the epilogue warps in both CTAs
                    # And accordingly, each CTA's epilogue warps will arrive the acc empty mbar of the leader CTA's,
                    # i.e. the epilogue warps in non-leader CTA have to mapa the acc empty mbar and arrive it remotely
                    cute.arch.mbarrier_init(acc_full_mbar_ptr + acc_stage, num_acc_producer)
                    cute.arch.mbarrier_init(
                        acc_empty_mbar_ptr + acc_stage, num_acc_consumer
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
        # (EPI_M=(8,16), EPI_N=(32,1), EPI_STAGE=4)
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        if const_expr(self.debug_print):
            if is_print_thread:
                cute.printf("")
                cute.printf("sA layout: {}", sA.layout)
                cute.printf("sB layout: {}", sB.layout)
                cute.printf("sC layout: {}", sC.layout)

        # /////////////////////////////////////////////////////////////////////////////
        #  Compute multicast mask for A/B buffer full and empty
        # /////////////////////////////////////////////////////////////////////////////
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        ab_empty_mcast_mask = None
        if const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask( # 0b(01), only for self
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2 # along N dim
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask( # 0b(01), only for self
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1 # along M dim
            )
            
            ab_empty_mcast_mask = a_full_mcast_mask | b_full_mcast_mask # 0b(01) for self, updated below
        
        acc_full_mcast_mask = None
        if const_expr(use_2cta_instrs):
            # When umma is done, we need to arrive both the self and peer acc full mbars
            acc_full_mcast_mask = cute.make_layout_image_mask( # 0b(11), for both self and peer
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mode=0 # along V (CTA pair) dim
            )
            block_in_cluster_coord_vmnk_peer = (
                block_in_cluster_coord_vmnk[0] ^ 1, # use peer CTA rank
                *block_in_cluster_coord_vmnk[1:],
            )
            a_full_mcast_mask_peer = cpasync.create_tma_multicast_mask( # 0b(10), only for peer
                cluster_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=2
            )
            b_full_mcast_mask_peer = cpasync.create_tma_multicast_mask( # 0b(10), only for peer
                cluster_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=1
            )
            
            # When umma is done, we need to arrive both the self and peer ab empty mbars
            ab_empty_mcast_mask = ( # 0b(11), for both self and peer
                a_full_mcast_mask_peer
                | b_full_mcast_mask_peer
                | cutlass.Int16(
                    0 if ab_empty_mcast_mask is None else ab_empty_mcast_mask
                )
            )

        if const_expr(self.debug_print):
            if is_print_thread:
                cute.printf("")
                cute.printf("a_full_mcast_mask: {}, b_full_mcast_mask: {}, ab_empty_mcast_mask: {}, acc_full_mcast_mask: {}",
                    a_full_mcast_mask, b_full_mcast_mask, ab_empty_mcast_mask, acc_full_mcast_mask)
                cute.printf("block_in_cluster_coord_vmnk: {}, block_in_cluster_coord_vmnk_peer: {}",
                    block_in_cluster_coord_vmnk, block_in_cluster_coord_vmnk_peer if const_expr(use_2cta_instrs) else "N/A"
                )
                cute.printf("acc_full_mcast_mask: {}", acc_full_mcast_mask if const_expr(use_2cta_instrs) else "N/A")
                cute.printf("a_full_mcast_mask_peer: {}, b_full_mcast_mask_peer: {}",
                    a_full_mcast_mask_peer if const_expr(use_2cta_instrs) else "N/A",
                    b_full_mcast_mask_peer if const_expr(use_2cta_instrs) else "N/A"
                )

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

        if const_expr(self.debug_print):
            if is_print_thread:
                cute.printf("")
                cute.printf("gA_mkl layout: {}", gA_mkl.layout)
                cute.printf("gB_nkl layout: {}", gB_nkl.layout)
                cute.printf("gC_mnl layout: {}", gC_mnl.layout)

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
            if is_print_thread:
                cute.printf("")
                cute.printf("tCgA layout: {}", tCgA.layout)
                cute.printf("tCgB layout: {}", tCgB.layout)
                cute.printf("tCgC layout: {}", tCgC.layout)

        # /////////////////////////////////////////////////////////////////////////////
        #  Partition global/shared tensor for load A, B with TMA
        # /////////////////////////////////////////////////////////////////////////////
        
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
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
        b_cta_layout = cute.make_layout(
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
            if is_print_thread:
                cute.printf("")
                cute.printf("a_cta_layout: {}, b_cta_layout: {}", a_cta_layout, b_cta_layout)
                cute.printf("tAsA layout: {}", tAsA.layout)
                cute.printf("tAgA layout: {}", tAgA.layout)
                cute.printf("tBsB layout: {}", tBsB.layout)
                cute.printf("tBgB layout: {}", tBgB.layout)

        # /////////////////////////////////////////////////////////////////////////////
        #  Partition shared/tensor memory tensor for TiledMMA_A/B/C
        # /////////////////////////////////////////////////////////////////////////////
        # (MMA=1, MMA_M=1, MMA_K=4, STAGE=8):(0,0,2,1024)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA=1, MMA_N=1, MMA_K=4, STAGE=8):(0,0,2,512)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA=(128,128), MMA_M=1, MMA_N=1, ACC_STAGE=2):((65536,1),0,0,128)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler_mnk[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        if const_expr(self.debug_print):
            if is_print_thread:
                cute.printf("")
                cute.printf("tCrA layout: {}", tCrA.layout)
                cute.printf("tCrB layout: {}", tCrB.layout)
                cute.printf("tCtAcc_fake layout: {}", tCtAcc_fake.layout)

        # /////////////////////////////////////////////////////////////////////////////
        #  Create static persistent tile scheduler
        # /////////////////////////////////////////////////////////////////////////////
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, bid, grid_dim
        )
        
        # grouped gemm tile scheduler helper will compute the group index for the tile we're working on
        group_gemm_ts_helper = utils.GroupedGemmTileSchedulerHelper(
        # group_gemm_ts_helper = utils.StaticPersistentGroupTileScheduler(
            group_count,
            tile_sched_params,
            self.cluster_tile_shape_mnk,
            utils.create_initial_search_state(),
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
        #  Get tensormap buffer address
        # /////////////////////////////////////////////////////////////////////////////
        
        # tensormaps: (num_sms=148, num_tensormaps=3, num_int64_per_tma_desc=16)
        
        tensormap_workspace_idx = ( # flatten block idx to index the SM
            bid[2] * grid_dim[1] * grid_dim[0] + bid[1] * grid_dim[0] + bid[0]
        )

        tensormap_manager = utils.TensorMapManager(
            self.tensormap_update_mode, 
            self.bytes_per_tensormap
        )
        
        tensormap_a_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 0, None)].iterator
        )
        tensormap_b_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 1, None)].iterator
        )
        tensormap_c_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 2, None)].iterator
        )
        
        # Setup tensormap initialization pointer based on the mode
        # GMEM mode: directly write to the gmem tensormap buffer
        # SMEM mode: write to the smem buffer first and then copy to gmem
        if const_expr(
            self.tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
        ):
            tensormap_a_init_ptr = tensormap_a_smem_ptr
            tensormap_b_init_ptr = tensormap_b_smem_ptr
            tensormap_c_init_ptr = tensormap_c_smem_ptr
        else:
            tensormap_a_init_ptr = tensormap_a_ptr
            tensormap_b_init_ptr = tensormap_b_ptr
            tensormap_c_init_ptr = tensormap_c_ptr

        # /////////////////////////////////////////////////////////////////////////////
        #  Specialized TMA load warp
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.tma_warp_id:
            # Initialize tensormaps for A, B if not delegated to the mma warp
            # which will call `copy_tensormap` to copy all the TMA desc information to the `dst_ptr`
            # including the (1) base address, shape and stride, (2) dtype, (3) swizzle, (4) LBO SBO, etc,
            # in which (1) will be updated whenever a new group first enters the mainloop by `update_tensormap`
            if const_expr(self.delegate_tensormap_ab_init == False):
                tensormap_manager.init_tensormap_from_atom(
                    copy_atom=tma_atom_a, 
                    dst_ptr=tensormap_a_init_ptr, 
                    warp_id=self.tma_warp_id
                )
                tensormap_manager.init_tensormap_from_atom(
                    copy_atom=tma_atom_b, 
                    dst_ptr=tensormap_b_init_ptr, 
                    warp_id=self.tma_warp_id
                )
            
            # /////////////////////////////////////////////////////////////////////////////
            #  Persistent tile scheduling loop (TMA warp)
            # /////////////////////////////////////////////////////////////////////////////
            tensormap_init_done = cutlass.Boolean(False)
            total_k_tile_cnt = cutlass.Int32(0) # tile count we have searched
            last_group_idx = cutlass.Int32(-1) # group index of last tile
            
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx # block coord
                grouped_gemm_cta_tile_info = group_gemm_ts_helper.delinearize_z(
                    cur_tile_coord,
                    problem_sizes_mnkl,
                )
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                is_group_changed = cur_group_idx != last_group_idx
                
                # Skip tensormap update if we're working on the same group
                if is_group_changed:
                    # Construct gmem tensor A/B based on real address, shape and stride information
                    real_tensor_a = self.make_tensor_for_tensormap_update(
                        cur_group_idx,
                        self.a_dtype,
                        problem_shape_mnk=(
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        strides_abc=strides_abc,
                        tensor_address_abc=ptrs_abc,
                        tensor_index=0,  # 0 for tensor A
                    )
                    real_tensor_b = self.make_tensor_for_tensormap_update(
                        cur_group_idx,
                        self.b_dtype,
                        problem_shape_mnk=(
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        strides_abc=strides_abc,
                        tensor_address_abc=ptrs_abc,
                        tensor_index=1,  # 1 for tensor B
                    )
                    
                    # Wait tensormap initialization complete before update
                    if tensormap_init_done == False:
                        if const_expr(self.delegate_tensormap_ab_init):
                            # Wait for the mma warp to finish tensormap initialization
                            cute.arch.barrier(
                                barrier_id=self.tensormap_ab_init_bar_id,
                                number_of_threads=self.tensormap_init_threads,
                            )
                        
                        # If in GMEM tensormap update mode, the TMA desc is directly written to the gmem buffer,
                        # so we need to call `fence.acq_rel.cta` (i.e. `__threadfence_block()`) 
                        # to ensure visibility of tensormap initialization
                        # to all the threads in the CTA before we update the tensormap
                        # noop for SMEM tensormap update mode
                        tensormap_manager.fence_tensormap_initialization()
                        tensormap_init_done = True

                    # Update tensormap for the current group by the tma warp, varying on the mode:
                    #   GMEM mode:
                    #       step1. wait until all in-flight TMA finished by `cp_async_bulk_commit_group` and `cp_async_bulk_wait_group(0, read=True)`
                    #       step2. update tma desc directly in gmem, including the base addrs, shapes and strides
                    #       step3. call `fence.proxy.tensormap::generic.release.gpu` 
                    #               to ensure tensormap write is visible with release order
                    #
                    #   SMEM mode:
                    #       step1. update tma desc in smem w/o waiting in-flight TMA, including the base addrs, shapes and strides
                    #       step2. wait for all in-flight TMA to finish, the same as step1 in GMEM mode
                    #       step3. call `tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned`
                    #               to copy updated tma desc from smem to gmem and ensure tensormap write is visible with release order
                    # aligning with the read fence in `fence_tensormap_update` with acquire order below
                    tensormap_manager.update_tensormap(
                        tensor_gmem=(real_tensor_a, real_tensor_b),
                        tma_copy_atom=(tma_atom_a, tma_atom_b),
                        tensormap_gmem_ptr=(tensormap_a_ptr, tensormap_b_ptr),
                        warp_id=self.tma_warp_id,
                        tensormap_smem_ptr=(tensormap_a_smem_ptr, tensormap_b_smem_ptr),
                    )

                mma_tile_coord_mnl = ( # cluster coord
                    grouped_gemm_cta_tile_info.cta_tile_idx_m
                    // self.atom_thr_size,
                    grouped_gemm_cta_tile_info.cta_tile_idx_n,
                    0,
                )

                # Slice to per mma tile index
                # ((atom_v, rest_v)=((64,128),1), RestK16):(((1@0,1@1),0),64@0)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2]) # slice RestM8 and RestL1 idx
                ]
                # ((atom_v, rest_v)=(64,64),1), RestK16):(((1@0,1@1),0),64@0)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2]) # slice RestN32 and RestL1 idx
                ]

                if const_expr(self.debug_print):
                    is_first_work_tile = (cur_tile_coord[0] == 0) and (cur_tile_coord[1] == 0) and (cur_tile_coord[2] == 0)
                    if (tidx == 32 * self.tma_warp_id) and is_print_block and is_first_work_tile:
                        cute.printf("")
                        cute.printf("[TMA warp] mma_tile_coord_mnl: ({}, {}, {}), cur_group_idx: {}, cur_k_tile_cnt: {}",
                            mma_tile_coord_mnl[0], mma_tile_coord_mnl[1], mma_tile_coord_mnl[2], cur_group_idx, cur_k_tile_cnt)
                        cute.printf("[TMA warp] tAgA_slice layout: {}", tAgA_slice.layout)
                        cute.printf("[TMA warp] tBgB_slice layout: {}", tBgB_slice.layout)

                num_prev_k_blk = total_k_tile_cnt
                total_k_tile_cnt += cur_k_tile_cnt

                # Init the first stage idx and the mbar phase
                tma_wr_k_tile = cutlass.Int32(0)
                smem_wr_stage_idx = (num_prev_k_blk + tma_wr_k_tile) % self.num_ab_stage
                tma_wr_ab_empty_phase = (
                    # NOTE:
                    #   1. each new round across a whole smem stages, we need to toggle the phase
                    #       since the mbar's parity will automatically toggle when one mbar.wait is done
                    #   2. since the mbar's parity is initialized to 0, thus the producer needs to initialize its phase to 1 (i.e. ^1)
                    #       to make the first wait directly succeed  (mbar.wait when parity == phase until the expected count is arrived)
                    (num_prev_k_blk + tma_wr_k_tile) // self.num_ab_stage % 2 ^ 1
                )
                
                # Peek for the first ab empty mbar to be arrived by the consumer w/o blocking
                peek_ab_empty_status = cute.arch.mbarrier_conditional_try_wait(
                    tma_wr_k_tile < cur_k_tile_cnt,
                    ab_empty_mbar_ptr + smem_wr_stage_idx,
                    tma_wr_ab_empty_phase,
                )
                
                # Ensure the update to tensormap has completed before using it
                if is_group_changed:
                    # Call `fence.proxy.tensormap::generic.acquire.gpu` 
                    # to ensure tensormap read is visible with acquire order before we use it to load TMA
                    # aligning with the write fence in `update_tensormap` with release order above
                    tensormap_manager.fence_tensormap_update(tensormap_a_ptr)
                    tensormap_manager.fence_tensormap_update(tensormap_b_ptr)
                
                # /////////////////////////////////////////////////////////////////////////////
                #  Tma load loop
                # /////////////////////////////////////////////////////////////////////////////
                for k_tile in cutlass.range(cur_k_tile_cnt, unroll=1):
                    tma_wr_k_tile_next = tma_wr_k_tile + 1
                    smem_wr_next_stage_idx = (num_prev_k_blk + tma_wr_k_tile_next) % self.num_ab_stage
                    tma_wr_ab_empty_phase_next = (
                        tma_wr_ab_empty_phase ^ 1 if smem_wr_next_stage_idx == 0 
                        else tma_wr_ab_empty_phase
                    )

                    smem_full_mbar_ptr = ab_full_mbar_ptr + smem_wr_stage_idx

                    # Wait for current ab empty mbar to be arrived by the consumer
                    if peek_ab_empty_status == 0: # token == 0, peek failed
                        cute.arch.mbarrier_wait(
                            ab_empty_mbar_ptr + smem_wr_stage_idx, 
                            tma_wr_ab_empty_phase
                        )

                    # Arrive ab full mbar and expect full transaction bytes 
                    # NOTE: it is only arrived by the leader CTA
                    # since only the leader CTA is the umma consumer who waits for the ab full mbar
                    if is_leader_cta:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                smem_full_mbar_ptr, self.num_tma_load_bytes
                            )

                    # TMA load A/B
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, tma_wr_k_tile)],
                        tAsA[(None, smem_wr_stage_idx)],
                        tma_bar_ptr=smem_full_mbar_ptr,
                        mcast_mask=a_full_mcast_mask,
                        # NOTE: we need to manually specify the tma desc ptr by fetching from the tensor map
                        tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                            tensormap_a_ptr,
                            cute.AddressSpace.generic,
                        ),
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, tma_wr_k_tile)],
                        tBsB[(None, smem_wr_stage_idx)],
                        tma_bar_ptr=smem_full_mbar_ptr,
                        mcast_mask=b_full_mcast_mask,
                        tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                            tensormap_b_ptr,
                            cute.AddressSpace.generic,
                        ),
                    )

                    # Peek for the first ab empty mbar to be arrived by the consumer w/o blocking
                    peek_ab_empty_status = cute.arch.mbarrier_conditional_try_wait(
                        tma_wr_k_tile_next < cur_k_tile_cnt,
                        ab_empty_mbar_ptr + smem_wr_next_stage_idx,
                        tma_wr_ab_empty_phase_next,
                    )

                    tma_wr_k_tile = tma_wr_k_tile_next
                    smem_wr_stage_idx = smem_wr_next_stage_idx
                    tma_wr_ab_empty_phase = tma_wr_ab_empty_phase_next

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx

            # Wait for the last ab empty mbar to avoid dangling signals
            cute.arch.mbarrier_wait(
                (ab_empty_mbar_ptr + ((total_k_tile_cnt - 1) % self.num_ab_stage)),
                phase=(((total_k_tile_cnt - 1) // self.num_ab_stage) % 2),
            )

        # /////////////////////////////////////////////////////////////////////////////
        #  Specialized MMA warp
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            # Initialize tensormap A, B for TMA warp (one-time thing when kernel launches)
            if const_expr(self.delegate_tensormap_ab_init):
                # Call the `copy_tensormap` to copy the `tma_atom` (with the initialized TMA desc inside)
                # to the tensormap buffer pointed by `dst_ptr`, by the warp with idx `warp_id` (elected one lane and sync the warp after)
                tensormap_manager.init_tensormap_from_atom(
                    copy_atom=tma_atom_a, 
                    dst_ptr=tensormap_a_init_ptr,
                    warp_id=self.mma_warp_id
                )
                tensormap_manager.init_tensormap_from_atom(
                    copy_atom=tma_atom_b,
                    dst_ptr=tensormap_b_init_ptr,
                    warp_id=self.mma_warp_id
                )
                
                # Signal tensormap initialization has finished to the tma warp
                cute.arch.barrier(
                    barrier_id=self.tensormap_ab_init_bar_id,
                    number_of_threads=self.tensormap_init_threads
                )
            
            # Bar sync for retrieve tmem ptr from shared mem
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=self.tmem_ptr_read_threads,
            )

            # Retrieving tensor memory ptr and make accumulator tensor
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_smem_buf,
            )
            
            # (MMA=(128,128), MMA_M=1, MMA_N=1, ACC_STAGE=2):((65536,1),0,0,128)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            
            if const_expr(self.debug_print):
                if (tidx == 32 * self.mma_warp_id) and is_print_block:
                    cute.printf("")
                    cute.printf("[MMA warp] tCtAcc_base: {}", tCtAcc_base)
                    cute.printf("")

            # /////////////////////////////////////////////////////////////////////////////
            #  Persistent tile scheduling loop (MMA warp)
            # /////////////////////////////////////////////////////////////////////////////
            total_k_tile_cnt = cutlass.Int32(0) # tile count we have searched
            
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx # block coord
                # MMA warp is only interested in number of tiles along K dimension
                (
                    cur_k_tile_cnt,
                    cur_group_idx,
                ) = group_gemm_ts_helper.search_cluster_tile_count_k(
                    cur_tile_coord,
                    problem_sizes_mnkl,
                )
                
                # Set tensor memory buffer for current tile
                # (MMA=(128,128), MMA_M=1, MMA_N=1):((65536,1),0,0)
                acc_buf_stage_idx = tile_sched.num_tiles_executed % self.num_acc_stage
                tCtAcc = tCtAcc_base[(None, None, None, acc_buf_stage_idx)]
                
                if const_expr(self.debug_print):
                    is_first_work_tile = (cur_tile_coord[0] == 0) and (cur_tile_coord[1] == 0) and (cur_tile_coord[2] == 0)
                    if (tidx == 32 * self.mma_warp_id) and is_print_block and is_first_work_tile:
                        cute.printf("")
                        cute.printf("[MMA warp] tCtAcc: {}", tCtAcc)
                        cute.printf("")

                num_prev_k_blk = total_k_tile_cnt
                total_k_tile_cnt += cur_k_tile_cnt

                # Init the first stage idx and the mbar phase
                mma_rd_k_tile = cutlass.Int32(0)
                smem_rd_stage_idx = (num_prev_k_blk + mma_rd_k_tile) % self.num_ab_stage
                mma_rd_ab_full_phase = (
                    # NOTE: different from the producer, the consumer does not need to toggle the phase for the first wait
                    (num_prev_k_blk + mma_rd_k_tile) // self.num_ab_stage % 2
                )
                
                # Peek for the first ab full mbar to be arrived by the producer w/o blocking only by the leader CTA
                need_check_rd_buffer_full = (
                    mma_rd_k_tile < cur_k_tile_cnt and is_leader_cta
                )
                peek_ab_full_status = cute.arch.mbarrier_conditional_try_wait(
                    need_check_rd_buffer_full,
                    ab_full_mbar_ptr + smem_rd_stage_idx,
                    mma_rd_ab_full_phase,
                )

                # Wait for the first acc empty mbar to be arrived by the epilogue consumer only by the leader CTA
                if is_leader_cta:
                    acc_empty_phase = (
                        # NOTE: as the acc producer, we need to toggle the phase for the first wait to directly pass
                        tile_sched.num_tiles_executed // self.num_acc_stage % 2 ^ 1
                    )
                    cute.arch.mbarrier_wait(
                        acc_empty_mbar_ptr + acc_buf_stage_idx, acc_empty_phase
                    )

                # Reset the ACCUMULATE field for each tile
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                # /////////////////////////////////////////////////////////////////////////////
                #  Mma mainloop
                # /////////////////////////////////////////////////////////////////////////////
                for k_tile in range(cur_k_tile_cnt):
                    mma_rd_k_tile_next = cutlass.Int32(k_tile + 1)
                    smem_rd_next_stage_idx = (num_prev_k_blk + mma_rd_k_tile_next) % self.num_ab_stage
                    mma_rd_ab_full_phase_next = (
                        mma_rd_ab_full_phase ^ 1
                        if smem_rd_next_stage_idx == 0
                        else mma_rd_ab_full_phase
                    )
                    
                    # Wait for current ab full mbar to be arrived by the tma producer from the leader CTA
                    # by the leader CTA itself
                    if is_leader_cta:
                        if peek_ab_full_status == 0:
                            cute.arch.mbarrier_wait(
                                ab_full_mbar_ptr + smem_rd_stage_idx, mma_rd_ab_full_phase
                            )

                        # tCtAcc += tCrA * tCrB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (None, None, kblock_idx, smem_rd_stage_idx)

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )
                            
                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Arrive the ab empty mbar with `tcgen05.commit.mbarrier::arrive::one`
                        # only by the leader CTA
                        with cute.arch.elect_one():
                            tcgen05.commit(
                                ab_empty_mbar_ptr + smem_rd_stage_idx,
                                ab_empty_mcast_mask, # mcast to self and peer's ab empty mbar when using 2-CTA
                                self.cta_group,
                            )

                    # Peek for the next ab full mbar to be arrived by the tma producer w/o blocking only by the leader CTA
                    need_check_rd_buffer_full = (
                        mma_rd_k_tile_next < cur_k_tile_cnt and is_leader_cta
                    )

                    peek_ab_full_status = cute.arch.mbarrier_conditional_try_wait(
                        need_check_rd_buffer_full,
                        ab_full_mbar_ptr + smem_rd_next_stage_idx,
                        mma_rd_ab_full_phase_next,
                    )

                    mma_rd_k_tile = mma_rd_k_tile_next
                    smem_rd_stage_idx = smem_rd_next_stage_idx
                    mma_rd_ab_full_phase = mma_rd_ab_full_phase_next

                # Arrive the acc full mbar with `tcgen05.commit.mbarrier::arrive::one`
                # only by the leader CTA (UMMA consumer => T2R producer)
                if is_leader_cta:
                    with cute.arch.elect_one():
                        tcgen05.commit(
                            acc_full_mbar_ptr + acc_buf_stage_idx,
                            mask=acc_full_mcast_mask, # mcast to self and peer's acc full mbar when using 2-CTA
                            cta_group=self.cta_group,
                        )

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Wait for the last acc empty mbar to avoid dangling signals only by the leader CTA
            # NOTE: the inner logic selects the leader CTA automatically
            if is_leader_cta:
                cute.arch.mbarrier_wait(
                    (acc_empty_mbar_ptr + ((total_k_tile_cnt - 1) % self.num_ab_stage)),
                    phase=(((total_k_tile_cnt - 1) // self.num_ab_stage) % 2),
                )

        # /////////////////////////////////////////////////////////////////////////////
        #  Specialized epilogue warps
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.mma_warp_id:
            # Initialize tensormap for C by the first epilogue warp (one-time thing when kernel launches)
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_c,
                tensormap_c_init_ptr,
                self.epilog_warp_id[0],
            )
            
            # Alloc tensor memory buffer
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols,
                    smem_ptr_to_write_address=tmem_holding_smem_buf,
                    is_two_cta=use_2cta_instrs,
                )

            # Bar sync for retrieve tensor memory ptr from shared memory
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=self.tmem_ptr_read_threads,
            )

            # Retrieve tensor memory ptr and make accumulator tensor
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_smem_buf,
            )
            
            # (MMA=(128,128), MMA_M=1, MMA_N=1, ACC_STAGE=2):((65536,1),0,0,128)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            
            if const_expr(self.debug_print):
                if (tidx == 0) and is_print_block:
                    cute.printf("")
                    cute.printf("[Epilog warp] tCtAcc_base: {}", tCtAcc_base)
                    cute.printf("")

            # /////////////////////////////////////////////////////////////////////////////
            #  Partition for epilogue
            # /////////////////////////////////////////////////////////////////////////////
            epi_tidx = tidx
            
            # tiled_copy_t2r: 
            #   layout_src_tv: (32,1024):(0,1) | layout_src_tv_tiled: ((32,4),((32,32),1)):((0,1),((128,4),0))
            #   layout_dst_tv: (32,32):(32,1) | layout_dst_tv_tiled: ((32,4),(32,1)):((4,1),(128,0))
            # tTR_tAcc_base: (T2R=((T2R_COLS=32,T2R_ROWS=32),1), T2R_M=1,T2R_N=1, EPI_M=1,EPI_N=4, EPI_STAGES=2):(((1,65536),0),0,0,0,32,128)
            # tTR_rAcc: ((32,1),1,1):((1,0),0,0)
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
            )
            
            # Make R2S tiled copy
            # tiled_copy_r2s:
            #   layout_src_tv: (1,1):(0,0) | layout_src_tv_tiled: ((32,4),(1,32)):((4,1),(0,128))
            #   layout_dst_tv: (1,1):(0,0) | layout_dst_tv_tiled: ((32,4),(1,32)):((4,1),(0,128))
            # tTR_rC: ((32,1),1,1):((1,0),0,0) | tRS_rC: ((1,32),1,1):((0,1),0,0)
            # tRS_sC: (R2S=(1,32), 1,1, epi_stages=(1,4)):((0,1),0,0,(0,4096))
            # tTR_rC = cute.make_fragment(tTR_rAcc.shape, self.c_dtype) # deprecated API
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype) # new API, the bf16 version of tTR_rAcc
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            
            # Make S2G TMA tiled copy
            # bSG_sC: (TMA=(4096,1), epi_stages=(1,4)):((1,0),(0,4096))
            # bSG_gC_partitioned: (TMA=((32,128),1),EPI_M=1,EPI_N=4,RestM=8,RestN=32,RestL=1):(((1@0,1@1),0),0,32@0,256@1,128@0,1@2)
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(tma_atom_c, tCgC, epi_tile, sC)

            if const_expr(self.debug_print):
                if (tidx == 0) and is_print_block:
                    cute.printf("")
                    cute.printf("[Epilog warp] tiled_copy_t2r: layout_src_tv: {} | layout_src_tv_tiled: {} | layout_dst_tv: {} | layout_dst_tv_tiled: {}", tiled_copy_t2r.layout_src_tv, tiled_copy_t2r.layout_src_tv_tiled, tiled_copy_t2r.layout_dst_tv, tiled_copy_t2r.layout_dst_tv_tiled)
                    cute.printf("[Epilog warp] tTR_tAcc_base: {}", tTR_tAcc_base)
                    cute.printf("[Epilog warp] tTR_rAcc: {}", tTR_rAcc)
                    cute.printf("")
                    cute.printf("[Epilog warp] tiled_copy_r2s: layout_src_tv: {} | layout_src_tv_tiled: {} | layout_dst_tv: {} | layout_dst_tv_tiled: {}", tiled_copy_r2s.layout_src_tv, tiled_copy_r2s.layout_src_tv_tiled, tiled_copy_r2s.layout_dst_tv, tiled_copy_r2s.layout_dst_tv_tiled)
                    cute.printf("[Epilog warp] tTR_rC: {}", tTR_rC)
                    cute.printf("[Epilog warp] tRS_rC: {}", tRS_rC)
                    cute.printf("[Epilog warp] tRS_sC: {}", tRS_sC)
                    cute.printf("[Epilog warp] bSG_sC: {}", bSG_sC)
                    cute.printf("[Epilog warp] bSG_gC_partitioned: {}", bSG_gC_partitioned)
                    cute.printf("")

            # /////////////////////////////////////////////////////////////////////////////
            #  Persistent tile scheduling loop (epilog warp)
            # /////////////////////////////////////////////////////////////////////////////
            # Wait tensormap initialization complete before update
            # NOTE: if in GMEM tensormap update mode, the TMA desc is directly written to the gmem buffer and we need to 
            # call `fence.acq_rel.cta` (i.e. `__threadfence_block()`) to ensure visibility of tensormap initialization
            # but if in SMEM tensormap update mode, it's a noop
            tensormap_manager.fence_tensormap_initialization()
            
            total_k_tile_cnt = cutlass.Int32(0) # tile count we have searched
            last_group_idx = cutlass.Int32(-1) # group index of last tile
            
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx # block coord
                grouped_gemm_cta_tile_info = group_gemm_ts_helper.delinearize_z(
                    cur_tile_coord,
                    problem_sizes_mnkl,
                )
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                
                is_group_changed = cur_group_idx != last_group_idx
                if is_group_changed:
                    # Construct gmem tensor C based on real address, shape and stride information
                    real_tensor_c = self.make_tensor_for_tensormap_update(
                        cur_group_idx,
                        self.c_dtype,
                        problem_shape_mnk=(
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        strides_abc=strides_abc,
                        tensor_address_abc=ptrs_abc,
                        tensor_index=2,  # 2 for tensor C
                    )
                    
                    # Update tensormap for the current group
                    # by the first epilogue warp since only the first warp will issue the TMA store
                    # NOTE: it will use either `fence.proxy.tensormap::generic.release.gpu` 
                    # or `tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned`
                    # to ensure tensormap write is visible with release order
                    # aligning with the read fence in `fence_tensormap_update` with acquire order below
                    tensormap_manager.update_tensormap(
                        tensor_gmem=((real_tensor_c),),
                        tma_copy_atom=((tma_atom_c),),
                        tensormap_gmem_ptr=((tensormap_c_ptr),),
                        warp_id=self.epilog_warp_id[0],
                        tensormap_smem_ptr=(tensormap_c_smem_ptr,),
                    )

                mma_tile_coord_mnl = ( # cluster coord
                    grouped_gemm_cta_tile_info.cta_tile_idx_m
                    // self.atom_thr_size,
                    grouped_gemm_cta_tile_info.cta_tile_idx_n,
                    0,
                )
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                total_k_tile_cnt += cur_k_tile_cnt

                # /////////////////////////////////////////////////////////////////////////////
                #  Slice to per mma tile index
                # /////////////////////////////////////////////////////////////////////////////
                
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                # (TMA=((32,128),1),EPI_M=1, EPI_N=4)
                bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]

                # Set tensor memory buffer for current tile
                # (T2R=((T2R_COLS=32,T2R_ROWS=32),1), T2R_M=1,T2R_N=1, EPI_M=1,EPI_N=4)
                acc_buf_stage_idx = tile_sched.num_tiles_executed % self.num_acc_stage
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_buf_stage_idx)]

                # Wait for the first acc full mbar to be arrived by the epilogue producer only by the leader CTA
                acc_full_phase = (
                    # NOTE: as the acc consumer, we do not need to toggle the phase for the first wait
                    tile_sched.num_tiles_executed // self.num_acc_stage % 2
                )
                cute.arch.mbarrier_wait(acc_full_mbar_ptr + acc_buf_stage_idx, acc_full_phase)

                # Group the EPI_M and EPI_N modes together
                # tTR_tAcc: (T2R=((T2R_COLS=32, T2R_ROWS=32),1), T2R_M=1,T2R_N=1, (EPI_M, EPI_N)=(1,4))
                # bSG_gC: ((ATOM_V, REST_V)=((32,128),1), (EPI_M, EPI_N)=(1,4))
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                
                # Ensure the update to tensormap has completed before using it
                # by the first epilogue warp since only the first warp will issue the TMA store
                if is_group_changed:
                    if warp_idx == self.epilog_warp_id[0]:
                        # Call `fence.proxy.tensormap::generic.acquire.gpu` 
                        # to ensure tensormap read is visible with acquire order before we use it to load TMA
                        # aligning with the write fence in `update_tensormap` with release order above
                        tensormap_manager.fence_tensormap_update(tensormap_c_ptr)
                
                # /////////////////////////////////////////////////////////////////////////////
                #  Store accumulator to global memory in subtiles
                # /////////////////////////////////////////////////////////////////////////////
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3]) # EPI_M x EPI_N = 4
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt

                if const_expr(self.debug_print):
                    is_first_work_tile = (cur_tile_coord[0] == 0) and (cur_tile_coord[1] == 0) and (cur_tile_coord[2] == 0)
                    if (tidx == 0) and is_print_block and is_first_work_tile:
                        cute.printf("")
                        cute.printf("[Epilog warp] mma_tile_coord_mnl: ({}, {}, {})", mma_tile_coord_mnl[0], mma_tile_coord_mnl[1], mma_tile_coord_mnl[2])
                        cute.printf("[Epilog warp] tTR_tAcc (post-group): {}", tTR_tAcc)
                        cute.printf("[Epilog warp] subtile_cnt: {}", subtile_cnt)
                        cute.printf("[Epilog warp] num_prev_subtiles: {}", num_prev_subtiles)
                        cute.printf("[Epilog warp] bSG_gC (post-group): {}", bSG_gC)
                        cute.printf("")

                for subtile_idx in range(subtile_cnt):
                    # T2R copy to store accumulator from tmem to rmem
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    # Perform epilogue op on accumulator and convert to C type
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    tRS_rC.store(acc_vec.to(self.c_dtype))
                    
                    # R2S copy to store C from rmem to smem
                    epi_stage_idx = (num_prev_subtiles + subtile_idx) % self.num_epi_stage
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, epi_stage_idx)],
                    )
                    
                    # Fence and barrier all the epilogue threads
                        # to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=self.epilogue_threads,
                    )
                    
                    # S2G TMA store C from smem to gmem
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, epi_stage_idx)],
                            bSG_gC[(None, subtile_idx)],
                            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                                tensormap_c_ptr,
                                cute.AddressSpace.generic,
                            ),
                        )
                        
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(
                            # NOTE: with `read=True`, it only waits for the source smem buffer completely read
                            # then it can be reused, no need to wait for the gmem store to be completed
                            self.num_epi_stage - 1, read=True
                        )
                    
                    cute.arch.barrier( # wait warp0 before next iteration
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=self.epilogue_threads,
                    )
                
                # Arrive the acc empty mbar at the leader CTA
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(
                        acc_empty_mbar_ptr + acc_buf_stage_idx,
                        peer_cta_rank_in_cluster=( # find the leader CTA rank
                            cta_rank_in_cluster // 2 * 2 if use_2cta_instrs else None
                        )
                    )

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx

            # /////////////////////////////////////////////////////////////////////////////
            #  Dealloc the tensor memory buffer
            # /////////////////////////////////////////////////////////////////////////////
            if warp_idx == self.epilog_warp_id[0]:
                # Relinquishes the right to allocate TMEM 
                # so that other CTAs potentially in a different grid can allocate.
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            
            cute.arch.barrier( # wait for all the epilogue threads to finish before tmem deallocation
                barrier_id=self.epilog_sync_bar_id,
                number_of_threads=self.epilogue_threads,
            )
            
            if warp_idx == self.epilog_warp_id[0]:
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

            # Wait for all TMA store to finish (at least the smem read)
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def make_tensor_for_tensormap_update(
        self,
        group_idx: cutlass.Int32,
        dtype: Type[cutlass.Numeric],
        problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        tensor_index: int,
    ):
        """Extract stride and tensor address for a given group and construct a global tensor.

        This function is used within the kernel to dynamically create a CUTE tensor
        representing A, B, or C for the current group being processed, using the
        group-specific address, shape, and stride information.

        :param group_idx: The index of the current group within the grouped GEMM.
        :type group_idx: cutlass.Int32
        :param dtype: The data type of the tensor elements (e.g., cutlass.Float16).
        :type dtype: Type[cutlass.Numeric]
        :param problem_shape_mnk: The (M, N, K) problem shape for the current group.
        :type problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]
        :param strides_abc: Tensor containing strides for A, B, C for all groups. Layout: (group_count, 3, 2).
        :type strides_abc: cute.Tensor
        :param tensor_address_abc: Tensor containing global memory addresses for A, B, C for all groups. Layout: (group_count, 3).
        :type tensor_address_abc: cute.Tensor
        :param tensor_index: Specifies which tensor to create: 0 for A, 1 for B, 2 for C.
        :type tensor_index: int
        :return: A CUTE tensor representing the requested global memory tensor (A, B, or C) for the specified group.
        :rtype: cute.Tensor
        :raises TypeError: If the provided dtype is not a subclass of cutlass.Numeric.
        """
        ptr_i64 = tensor_address_abc[(group_idx, tensor_index)]
        if const_expr(
            not isclass(dtype) or not issubclass(dtype, cutlass.Numeric)
        ):
            raise TypeError(
                f"dtype must be a type of cutlass.Numeric, got {type(dtype)}"
            )
        tensor_gmem_ptr = cute.make_ptr(
            dtype, ptr_i64, cute.AddressSpace.gmem, assumed_align=16
        )

        strides_tensor_gmem = strides_abc[(group_idx, tensor_index, None)]
        strides_tensor_reg = cute.make_fragment(
            cute.make_layout(2),
            strides_abc.element_type,
        )
        cute.autovec_copy(strides_tensor_gmem, strides_tensor_reg)
        stride_mn = strides_tensor_reg[0]
        stride_k = strides_tensor_reg[1]
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)

        if const_expr(tensor_index == 0):  # tensor A
            m = problem_shape_mnk[0]
            k = problem_shape_mnk[2]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((m, k, c1), stride=(stride_mn, stride_k, c0)),
            )
        elif const_expr(tensor_index == 1):  # tensor B
            n = problem_shape_mnk[1]
            k = problem_shape_mnk[2]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((n, k, c1), stride=(stride_mn, stride_k, c0)),
            )
        else:  # tensor C
            m = problem_shape_mnk[0]
            n = problem_shape_mnk[1]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((m, n, c1), stride=(stride_mn, stride_k, c0)),
            )

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
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
        # Make tiledCopy for tensor memory load(t2r)
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
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
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
        tma_atom_c: cute.CopyAtom,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to partition
        shared memory (source) and global memory (destination) for TMA store version.

        :param tma_atom_c: The TMA copy atom configured for storing tensor C.
        :type tma_atom_c: cute.CopyAtom
        :param gC_mnl: The global memory tensor C.
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler defining the granularity of the operation.
        :type epi_tile: cute.Tile
        :param sC: The shared memory epilogue buffer tensor.
        :type sC: cute.Tensor
        :return: A tuple containing:
                 - tma_atom_c: The input TMA copy atom (passed through).
                 - bSG_sC: The source shared memory tensor partitioned for the TMA operation.
                 - tCgC: The destination global memory tensor partitioned for the TMA operation.
        :rtype: tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
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
        mma_tiler_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        smem_capacity: int,
        occupancy: int,
        debug_print: bool = False,
    ) -> tuple[int, int, int]:
        """Computes the number of stages for accumulator, A/B operands, and epilogue based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
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

        :return: A tuple containing the computed number of stages for:
                 (accumulator stages, A/B operand stages, epilogue stages)
        :rtype: tuple[int, int, int]
        """
        # Default accumulator and epilogue stages
        num_acc_stage = 2
        num_epi_stage = 2

        # Calculate smem layout and size for one stage of A, B, and Epilogue
        # sA_stage_one: S<3,4,3> o 0 o (MMA=(128,16),MMA_M=1,MMA_K=4,MMA_STAGE=1):((64,1),0,16,0)
        # sB_stage_one: S<3,4,3> o 0 o (MMA=(64,16),MMA_N=1,MMA_K=4,MMA_STAGE=1):((64,1),0,16,0)
        # sC_stage_one: S<2,4,3> o 0 o (epi_tileM=(8,16),epi_tileN=(32,1),epi_stages=(1,1)):((32,256),(1,0),(0,0))
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # stage=1
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # stage=1
        )
        epi_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,  # stage=1
        )
        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_stage_one
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)

        epi_bytes_per_stage = cute.size_in_bytes(c_dtype, epi_smem_layout_staged_one)
        epi_bytes = epi_bytes_per_stage * num_epi_stage

        # Calculate A/B stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial epilogue bytes
        # Divide remaining by bytes needed per A/B stage
        num_ab_stage = (
            smem_capacity // occupancy
            - GroupedGemmPersistentKernelSm100.reserved_smem_bytes
            - epi_bytes
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        remaining_smem = (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (GroupedGemmPersistentKernelSm100.reserved_smem_bytes + epi_bytes)
        )
        num_epi_stage += remaining_smem // (occupancy * epi_bytes_per_stage)

        if const_expr(debug_print):
            print()
            print("a_smem_layout_stage_one: ", a_smem_layout_stage_one)
            print("b_smem_layout_staged_one: ", b_smem_layout_staged_one)
            print("epi_smem_layout_staged_one: ", epi_smem_layout_staged_one)
            print(f"Bytes per A/B stage: {ab_bytes_per_stage=}")
            print(f"Bytes per epi stage: {epi_bytes_per_stage=}")
            print(
                f"Computed stages - AB stages: {num_ab_stage}, ACC stages: {num_acc_stage}, Epi stages: {num_epi_stage}"
            )
            print()

        return num_acc_stage, num_ab_stage, num_epi_stage

    @staticmethod
    def _compute_grid(
        total_num_clusters: int,
        cluster_shape_mn: tuple[int, int],
        max_active_clusters: cutlass.Constexpr[int],
    ) -> tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]:
        """Compute tile scheduler parameters and grid shape for grouped GEMM operations.

        :param total_num_clusters: Total number of clusters to process across all groups.
        :type total_num_clusters: int
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr[int]

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: tuple[utils.PersistentTileSchedulerParams, tuple[int, ...]]
        """
        # Create problem shape with M, N dimensions from cluster shape
        # and L dimension representing the total number of clusters.
        problem_shape_ntile_mnl = (
            cluster_shape_mn[0],
            cluster_shape_mn[1],
            cutlass.Int32(total_num_clusters),
        )

        tile_sched_params = utils.PersistentTileSchedulerParams(
            problem_shape_ntile_mnl, (*cluster_shape_mn, 1)
        )

        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def _get_mbar_smem_bytes(**kwargs_stages: int) -> int:
        """Calculate shared memory consumption for memory barriers based on provided stages.

        Each stage requires 2 barriers, and each barrier consumes 8 bytes of shared memory.
        The total consumption is the sum across all provided stages. This function calculates the total
        shared memory needed for these barriers.

        :param kwargs_stages: Variable keyword arguments where each key is a stage name
                              (e.g., num_acc_stage, num_ab_stage) and each value is the
                              number of stages of that type.
        :type kwargs_stages: int
        :return: Total shared memory bytes required for all memory barriers.
        :rtype: int
        """
        num_barriers_per_stage = 2 # full/empty mbar pair
        num_bytes_per_barrier = 8 # int64, 8B
        mbar_smem_consumption = sum(
            [
                num_barriers_per_stage * num_bytes_per_barrier * stage
                for stage in kwargs_stages.values()
            ]
        )
        return mbar_smem_consumption

    @staticmethod
    def _get_tensormap_smem_bytes(
        tensormap_update_mode: utils.TensorMapUpdateMode,
    ) -> int:
        """Get the SMEM consumption for the tensormap buffer based on the update mode.

        :param tensormap_update_mode: Specifies whether tensormaps are updated in GMEM or SMEM.
        :type tensormap_update_mode: utils.TensorMapUpdateMode
        :return: The shared memory bytes required for the tensormap buffer. Returns 0 if mode is GMEM.
        :rtype: int
        :raises ValueError: If an invalid tensormap update mode is provided.
        """
        if tensormap_update_mode == utils.TensorMapUpdateMode.GMEM:
            return 0
        elif tensormap_update_mode == utils.TensorMapUpdateMode.SMEM:
            return (
                GroupedGemmPersistentKernelSm100.bytes_per_tensormap * GroupedGemmPersistentKernelSm100.num_tensormaps
            )
        else:
            raise ValueError(f"Invalid tensormap update mode: {tensormap_update_mode}")

    @staticmethod
    def _get_tensor_smem_bytes(
        a_smem_layout_staged: cute.Layout,
        a_dtype: Type[cutlass.Numeric],
        b_smem_layout_staged: cute.Layout,
        b_dtype: Type[cutlass.Numeric],
        epi_smem_layout_staged: cute.Layout,
        c_dtype: Type[cutlass.Numeric],
    ) -> int:
        """Compute the total SMEM consumption for tensor A, B and C."""
        ab_bytes = cute.size_in_bytes(
            a_dtype, a_smem_layout_staged
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged)

        epi_bytes = cute.size_in_bytes(c_dtype, epi_smem_layout_staged)
        return ab_bytes + epi_bytes

    @staticmethod
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler: tuple[int, int, int],
        num_acc_stage: int,
    ) -> int:
        """
        Compute the number of tensor memory allocation columns.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler: The shape (M, N, K) of the MMA tile.
        :type mma_tiler: tuple[int, int, int]
        :param acc_stage: The stage of the accumulator tensor.
        :type acc_stage: int

        :return: The number of tensor memory allocation columns.
        :rtype: int
        """
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)

        return num_tmem_alloc_cols

    # Size of smem we reserved for mbarrier, tensor memory management and tensormap update
    reserved_smem_bytes = 1024
    bytes_per_tensormap = 128 # 128 bytes for TMA desc
    num_tensormaps = 3 # A/B/C
    # size of smem used for tensor memory management
    tensor_memory_management_bytes = 12


# Create tensor and return the pointer, tensor, and stride
def create_tensor_and_stride(
    l: int,
    mode0: int,
    mode1: int,
    is_mode0_major: bool,
    dtype: type[cutlass.Numeric],
    is_dynamic_layout: bool = True,
    torch_tensor_cpu: torch.Tensor = None,
) -> tuple[int, torch.Tensor, cute.Tensor, torch.Tensor, tuple[int, int]]:
    """Create a GPU tensor from scratch or based on an existing CPU tensor.

    :param torch_tensor_cpu: Optional existing CPU tensor to reuse. If None, creates a new one.
    :type torch_tensor_cpu: torch.Tensor, optional
    """
    if torch_tensor_cpu is None:
        # Create new CPU tensor
        torch_tensor_cpu = cutlass_torch.matrix(l, mode0, mode1, is_mode0_major, dtype)

    # Create GPU tensor from CPU tensor (new or existing)
    cute_tensor, torch_tensor = cutlass_torch.cute_tensor_like(
        torch_tensor_cpu, dtype, is_dynamic_layout, assumed_align=16
    )
    return (
        torch_tensor.data_ptr(),
        torch_tensor,
        cute_tensor,
        torch_tensor_cpu,
        torch_tensor.stride()[:-1],
    )


def create_tensors_for_all_groups(
    problem_sizes_mnkl: List[tuple[int, int, int, int]],
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    torch_fp32_tensors_abc: List[List[torch.Tensor]] = None,
) -> tuple[
    List[List[int]],
    List[List[torch.Tensor]],
    List[tuple],
    List[List[tuple]],
    List[List[torch.Tensor]],
]:
    if torch_fp32_tensors_abc is not None and len(torch_fp32_tensors_abc) != len(
        problem_sizes_mnkl
    ):
        raise ValueError("torch_fp32_tensors_abc must have one entry per group")

    # Initialize lists to store tensors for all groups
    new_torch_fp32_tensors_abc = (
        [] if torch_fp32_tensors_abc is None else torch_fp32_tensors_abc
    )
    torch_tensors_abc = []
    cute_tensors_abc = []
    strides_abc = []
    ptrs_abc = []

    # Iterate through all groups and create tensors for each group
    for group_idx, (m, n, k, l) in enumerate(problem_sizes_mnkl):
        # Get existing CPU tensors if available, otherwise None
        existing_cpu_a = (
            torch_fp32_tensors_abc[group_idx][0] if torch_fp32_tensors_abc else None
        )
        existing_cpu_b = (
            torch_fp32_tensors_abc[group_idx][1] if torch_fp32_tensors_abc else None
        )
        existing_cpu_c = (
            torch_fp32_tensors_abc[group_idx][2] if torch_fp32_tensors_abc else None
        )

        # Create tensors (reusing CPU tensors if provided)
        (
            ptr_a,
            torch_tensor_a,
            cute_tensor_a,
            tensor_fp32_a,
            stride_mk_a,
        ) = create_tensor_and_stride(
            l, m, k, a_major == "m", ab_dtype, torch_tensor_cpu=existing_cpu_a
        )
        (
            ptr_b,
            torch_tensor_b,
            cute_tensor_b,
            tensor_fp32_b,
            stride_nk_b,
        ) = create_tensor_and_stride(
            l, n, k, b_major == "n", ab_dtype, torch_tensor_cpu=existing_cpu_b
        )
        (
            ptr_c,
            torch_tensor_c,
            cute_tensor_c,
            tensor_fp32_c,
            stride_mn_c,
        ) = create_tensor_and_stride(
            l, m, n, c_major == "m", c_dtype, torch_tensor_cpu=existing_cpu_c
        )

        # Only append to new_torch_fp32_tensors_abc if we created new CPU tensors
        if torch_fp32_tensors_abc is None:
            new_torch_fp32_tensors_abc.append(
                [tensor_fp32_a, tensor_fp32_b, tensor_fp32_c]
            )

        ptrs_abc.append([ptr_a, ptr_b, ptr_c])
        torch_tensors_abc.append([torch_tensor_a, torch_tensor_b, torch_tensor_c])
        strides_abc.append([stride_mk_a, stride_nk_b, stride_mn_c])
        cute_tensors_abc.append(
            (
                cute_tensor_a,
                cute_tensor_b,
                cute_tensor_c,
            )
        )

    return (
        ptrs_abc,
        torch_tensors_abc,
        cute_tensors_abc,
        strides_abc,
        new_torch_fp32_tensors_abc,
    )


def run(
    num_groups: int,
    problem_sizes_mnkl: tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    use_2cta_instrs: bool,
    tensormap_update_mode: utils.TensorMapUpdateMode,
    tolerance: float,
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
    use_cold_l2: bool = False,
    debug_print: bool = False,
    **kwargs,
):
    """Run grouped GEMM example with specified configurations.

    :param use_cold_l2: Whether to use circular buffer strategy to ensure cold L2 cache, defaults to False
    :type use_cold_l2: bool, optional
    :return: Execution time of the GEMM kernel in microseconds
    :rtype: float
    """
    print(f"Running Blackwell Grouped GEMM test with:")
    print(f"{num_groups} groups")
    for i, (m, n, k, l) in enumerate(problem_sizes_mnkl):
        print(f"Group {i}: {m}x{n}x{k}x{l}")
    print(f"AB dtype: {ab_dtype}, C dtype: {c_dtype}, Acc dtype: {acc_dtype}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"2CTA MMA instructions: {'True' if use_2cta_instrs else 'False'}")
    print(f"Tensor map update mode: {tensormap_update_mode}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {'True' if use_cold_l2 else 'False'}")

    # Skip unsupported types
    if ab_dtype not in {
        cutlass.Float16,
        cutlass.BFloat16,
    }:
        raise ValueError(f"Skip unsupported ab_dtype {ab_dtype}")
    if c_dtype not in {cutlass.Float16, cutlass.BFloat16, cutlass.Float32}:
        raise ValueError(f"Skip unsupported c_dtype {c_dtype}")
    # Skip unsupported acc dtype
    if acc_dtype not in {cutlass.Float32, cutlass.Float16}:
        raise ValueError(f"Skip unsupported acc_dtype {acc_dtype}")
    # Skip invalid ab_dtype and acc_dtype combination
    if ab_dtype == cutlass.BFloat16 and acc_dtype == cutlass.Float16:
        raise ValueError("Skip invalid ab_dtype and acc_dtype combination")
    # Skip invalid mma tile shape
    if not (
        (not use_2cta_instrs and mma_tiler_mn[0] in [64, 128])
        or (use_2cta_instrs and mma_tiler_mn[0] in [128, 256])
    ):
        raise ValueError(f"Skip invalid mma tiler M {mma_tiler_mn[0]}")
    if mma_tiler_mn[1] not in range(32, 257, 32):
        raise ValueError(f"Skip invalid mma tiler N {mma_tiler_mn[1]}")
    # Skip illegal cluster shape
    if cluster_shape_mn[0] % (2 if use_2cta_instrs else 1) != 0:
        raise ValueError(
            f"cluster_shape_m need align with use_2cta_instrs config {cluster_shape_mn}"
        )
    # Skip invalid cluster shape
    is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
    if (
        cluster_shape_mn[0] * cluster_shape_mn[1] > 16
        or cluster_shape_mn[0] <= 0
        or cluster_shape_mn[1] <= 0
        or not is_power_of_2(cluster_shape_mn[0])
        or not is_power_of_2(cluster_shape_mn[1])
    ):
        raise ValueError(f"Skip invalid cluster shape {cluster_shape_mn}")

    # Skip illegal problem shape for load/store alignment
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
        raise ValueError("Skip invalid problem alignment")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    # Create tensors for all groups using the new function
    (
        ptrs_abc,
        torch_tensors_abc,
        cute_tensors_abc,
        strides_abc,
        torch_fp32_tensors_abc,
    ) = create_tensors_for_all_groups(
        problem_sizes_mnkl,
        ab_dtype,
        c_dtype,
        a_major,
        b_major,
        c_major,
    )

    # Choose A, B, C with the smallest size to create initial tensormaps
    key_size_a = lambda item: item[1][0] * item[1][2]
    key_size_b = lambda item: item[1][1] * item[1][2]
    key_size_c = lambda item: item[1][0] * item[1][1]
    # Find the indices of the groups with the smallest tensor sizes
    min_a_idx, _ = min(enumerate(problem_sizes_mnkl), key=key_size_a)
    min_b_idx, _ = min(enumerate(problem_sizes_mnkl), key=key_size_b)
    min_c_idx, _ = min(enumerate(problem_sizes_mnkl), key=key_size_c)
    initial_cute_tensors_abc = [
        cute_tensors_abc[min_a_idx][0],  # A with smallest (m, k)
        cute_tensors_abc[min_b_idx][1],  # B with smallest (n, k)
        cute_tensors_abc[min_c_idx][2],  # C with smallest (m, n)
    ]

    hardware_info = utils.HardwareInfo()
    max_sms = hardware_info.get_device_multiprocessor_count()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    print(f"Max active clusters: {max_active_clusters} for cluster shape {cluster_shape_mn} on device with {max_sms} SMs")

    # Prepare tensormap buffer for each SM
    num_tensormap_buffers = max_sms
    tensormap_shape = (
        num_tensormap_buffers, # one tensormap buffer per SM
        GroupedGemmPersistentKernelSm100.num_tensormaps, # 3 tensormaps for A/B/C
        GroupedGemmPersistentKernelSm100.bytes_per_tensormap // 8, # split a TMA desc into int64 
    )
    tensor_of_tensormap, tensor_of_tensormap_torch = cutlass_torch.cute_tensor_like(
        torch.empty(tensormap_shape, dtype=torch.int64),
        cutlass.Int64, # use int64 to store tensormap
        is_dynamic_layout=False,
    )

    grouped_gemm = GroupedGemmPersistentKernelSm100(
        acc_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        tensormap_update_mode,
        debug_print=debug_print,
    )

    # layout (num_groups, 4):(4, 1)
    (
        tensor_of_dim_size_mnkl,
        tensor_of_dim_size_mnkl_torch,
    ) = cutlass_torch.cute_tensor_like(
        torch.tensor(problem_sizes_mnkl, dtype=torch.int32),
        cutlass.Int32,
        is_dynamic_layout=False,
        assumed_align=16,
    )
    # layout (num_groups, 3, 2):(6, 2, 1)
    tensor_of_strides_abc, tensor_of_strides_abc_torch = cutlass_torch.cute_tensor_like(
        torch.tensor(strides_abc, dtype=torch.int32),
        cutlass.Int32,
        is_dynamic_layout=False,
        assumed_align=16,
    )

    # layout (num_groups,3):(3, 1)
    tensor_of_ptrs_abc, tensor_of_ptrs_abc_torch = cutlass_torch.cute_tensor_like(
        torch.tensor(ptrs_abc, dtype=torch.int64),
        cutlass.Int64,
        is_dynamic_layout=False,
        assumed_align=16,
    )

    # Compute total number of cluster tiles we need to compute for given grouped GEMM problem
    def compute_total_num_clusters(
        problem_sizes_mnkl: List[tuple[int, int, int, int]],
        cluster_tile_shape_mn: tuple[int, int],
    ) -> int:
        total_num_clusters = 0
        for m, n, _, _ in problem_sizes_mnkl:
            num_clusters_mn = tuple(
                (x + y - 1) // y for x, y in zip((m, n), cluster_tile_shape_mn)
            )
            total_num_clusters += functools.reduce(lambda x, y: x * y, num_clusters_mn)
        return total_num_clusters

    # Compute cluster tile shape
    def compute_cluster_tile_shape(
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        use_2cta_instrs: bool,
    ) -> tuple[int, int]:
        cta_tile_shape_mn = list(mma_tiler_mn)
        if use_2cta_instrs:
            cta_tile_shape_mn[0] = cta_tile_shape_mn[0] // 2
        return tuple(x * y for x, y in zip(cta_tile_shape_mn, cluster_shape_mn))

    cluster_tile_shape_mn = compute_cluster_tile_shape(
        mma_tiler_mn, cluster_shape_mn, use_2cta_instrs
    )
    total_num_clusters = compute_total_num_clusters(
        problem_sizes_mnkl, cluster_tile_shape_mn
    )

    # Initialize Stream
    current_stream = cutlass_torch.default_stream()

    # Compile grouped GEMM kernel
    compiled_grouped_gemm = cute.compile(
        grouped_gemm,
        initial_cute_tensors_abc[0],
        initial_cute_tensors_abc[1],
        initial_cute_tensors_abc[2],
        num_groups,
        tensor_of_dim_size_mnkl,
        tensor_of_strides_abc,
        tensor_of_ptrs_abc,
        total_num_clusters,
        tensor_of_tensormap,
        max_active_clusters,
        current_stream,
    )

    if not skip_ref_check:
        compiled_grouped_gemm(
            initial_cute_tensors_abc[0],
            initial_cute_tensors_abc[1],
            initial_cute_tensors_abc[2],
            tensor_of_dim_size_mnkl,
            tensor_of_strides_abc,
            tensor_of_ptrs_abc,
            tensor_of_tensormap,
            current_stream,
        )

        # Compute reference result
        for i, (a, b, c) in enumerate(torch_tensors_abc):
            ref = torch.einsum(
                "mkl,nkl->mnl",
                a.cpu().to(dtype=torch.float32),
                b.cpu().to(dtype=torch.float32),
            )
            print(f"checking group {i}")
            torch.testing.assert_close(
                c.cpu(),
                ref.to(cutlass_torch.dtype(c_dtype)),
                atol=tolerance,
                rtol=1e-05,
            )

    def generate_tensors():
        # Reuse existing CPU tensors and create new GPU tensors from them
        (
            ptrs_abc_workspace,
            torch_tensors_abc_workspace,
            cute_tensors_abc_workspace,
            strides_abc_workspace,
            _,
        ) = create_tensors_for_all_groups(
            problem_sizes_mnkl,
            ab_dtype,
            c_dtype,
            a_major,
            b_major,
            c_major,
            torch_fp32_tensors_abc,
        )

        initial_cute_tensors_abc_workspace = [
            cute_tensors_abc_workspace[min_a_idx][0],  # A with smallest (m, k)
            cute_tensors_abc_workspace[min_b_idx][1],  # B with smallest (n, k)
            cute_tensors_abc_workspace[min_c_idx][2],  # C with smallest (m, n)
        ]

        # Create new tensors for this workspace
        tensor_of_strides_abc_workspace, _ = cutlass_torch.cute_tensor_like(
            torch.tensor(strides_abc_workspace, dtype=torch.int32),
            cutlass.Int32,
            is_dynamic_layout=False,
            assumed_align=16,
        )

        tensor_of_ptrs_abc_workspace, _ = cutlass_torch.cute_tensor_like(
            torch.tensor(ptrs_abc_workspace, dtype=torch.int64),
            cutlass.Int64,
            is_dynamic_layout=False,
            assumed_align=16,
        )

        tensormap_workspace, _ = cutlass_torch.cute_tensor_like(
            torch.empty(tensormap_shape, dtype=torch.int64),
            cutlass.Int64,
            is_dynamic_layout=False,
        )

        return testing.JitArguments(
            initial_cute_tensors_abc_workspace[0],
            initial_cute_tensors_abc_workspace[1],
            initial_cute_tensors_abc_workspace[2],
            tensor_of_dim_size_mnkl,
            tensor_of_strides_abc_workspace,
            tensor_of_ptrs_abc_workspace,
            tensormap_workspace,
            current_stream,
        )

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            sum(
                [
                    sum(
                        [
                            torch_tensor.numel() * torch_tensor.element_size()
                            for torch_tensor in group_tensors
                        ]
                    )
                    for group_tensors in torch_tensors_abc
                ]
            )
            +
            # Add size of strides tensor
            tensor_of_strides_abc_torch.numel()
            * tensor_of_strides_abc_torch.element_size()
            +
            # Add size of ptrs tensor
            tensor_of_ptrs_abc_torch.numel() * tensor_of_ptrs_abc_torch.element_size()
            +
            # Add size of tensormap tensor
            tensor_of_tensormap_torch.numel() * tensor_of_tensormap_torch.element_size()
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = testing.benchmark(
        compiled_grouped_gemm,
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
        flops = 0
        for m, n, k, l in problem_sizes_mnkl:
            flops += 2 * m * n * k * l
        event_str = f"grouped gemm ({num_groups=}, {flops=})"
        iters, start, end = 10, 6, 9
        args_gen = generate_tensors()
        for i in range(iters):
            switch_profile(iter_id=i, start=start, end=end)
            with add_nvtx_event(event_str):
                compiled_grouped_gemm(
                    args_gen.args[0],
                    args_gen.args[1],
                    args_gen.args[2],
                    args_gen.args[3],
                    args_gen.args[4],
                    args_gen.args[5],
                    args_gen.args[6],
                    current_stream,
                )

    return exec_time  # Return execution time in microseconds


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    def parse_comma_separated_tuples(s: str) -> List[tuple[int, ...]]:
        if s.strip().startswith("("):
            # Split on ),( to separate tuples
            tuples = s.strip("()").split("),(")
            result = []
            tuple_len = None

            for t in tuples:
                # Parse individual tuple
                nums = [int(x.strip()) for x in t.split(",")]

                # Validate tuple length consistency
                if tuple_len is None:
                    tuple_len = len(nums)
                elif len(nums) != tuple_len:
                    raise argparse.ArgumentTypeError(
                        "All tuples must have the same length"
                    )

                result.append(tuple(nums))
            return result

        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers or list of tuples"
        )

    parser = argparse.ArgumentParser(
        description="Example of Grouped GEMM on Blackwell."
    )
    parser.add_argument(
        "--num_groups",
        type=int,
        default=2,
        help="Number of groups",
    )
    parser.add_argument(
        "--problem_sizes_mnkl",
        type=parse_comma_separated_tuples,
        default=((128, 128, 128, 1), (128, 128, 128, 1)),
        help="a tuple of problem sizes for each group (comma-separated tuples)",
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
    parser.add_argument(
        "--tensormap_update_mode",
        type=str,
        default="SMEM",
        help="Tensor map update mode",
    )
    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.Float16)
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

    if (
        len(args.problem_sizes_mnkl) != 0
        and len(args.problem_sizes_mnkl) != args.num_groups
    ):
        parser.error("--problem_sizes_mnkl must contain exactly num_groups tuples")

    # l mode must be 1 for all groups
    for _, _, _, l in args.problem_sizes_mnkl:
        if l != 1:
            parser.error("l must be 1 for all groups")

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    if args.tensormap_update_mode not in ["GMEM", "SMEM"]:
        parser.error("--tensormap_update_mode must be GMEM or SMEM")

    if args.tensormap_update_mode == "GMEM":
        tensormap_update_mode = utils.TensorMapUpdateMode.GMEM
    else:
        tensormap_update_mode = utils.TensorMapUpdateMode.SMEM

    torch.manual_seed(2025)

    exec_time = run(
        args.num_groups,
        args.problem_sizes_mnkl,
        args.ab_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.mma_tiler_mn,
        args.cluster_shape_mn,
        args.use_2cta_instrs,
        tensormap_update_mode,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
        debug_print=DEBUG_MODE,
    )
    print(f"PASS with execution time: {exec_time} ms")
