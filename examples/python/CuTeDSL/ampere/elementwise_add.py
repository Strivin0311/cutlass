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
import time
from typing import Type

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from cutlass import const_expr

DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"
PROFILE_MODE = os.environ.get("PROFILE_MODE", "0") == "1"

"""
An Elementwise Addition Example using CuTe DSL.

This example kernel copies data from global memory to register memory (rmem), performs the elementwise
addition operation, and stores the result back to global memory.

Primary goals of this example are to demonstrate how basic global memory copies can be expressed in
CuTe DSL and illustrate canonical partitioning patterns in CuTe. It also implements canonical
predication for tensors whose shape is not multiple of tile size to guard OOB reads.

Thread-value (or TV) layouts are central to canonical partitioning patterns in CuTe. They provide a
mapping from thread and a thread's value to the set of coordinates within a tile that we have sliced
out from a data tensor.

The input tensors are row-major layout, that leading dimension is the right most dimension. In order
to efficiently copy data from global memory, we must map threads contiguously on row dimension.

Thread ID mapping to 2D coordinates with layout `(4,32):(32,1)`:

    +----+----+----+----+-----+----+
    |    | 0  | 1  | 2  | ... | 31 |
    +----+----+----+----+-----+----+
    | 0  | T0 | T1 | T2 | ... | T31|
    +----+----+----+----+-----+----+
    | 1  |T32 |T33 |T34 | ... |T63 |
    +----+----+----+----+-----+----+
    | 2  |T64 |T65 |T66 | ... |T95 |
    +----+----+----+----+-----+----+
    | 3  |T96 |T97 |T98 | ... |T127|
    +----+----+----+----+-----+----+

As Ampere GPU supports a maximum of 128bit per load/store instruction and each element is 32bit, we
can load 4 elements per instruction. Having additional contiguous values allows for vectorization
across threads (coalesced accesses) and is required for saturating the memory bandwidth.

We use `(4,4):(4,1)` as the val layout in this example. Notice that the major mode is the same as
the major mode of the input tensor - without which vectorization would not be possible.

If you already know the TV layout you want to use for your tiled copy, CuTe DSL provides utility
`cute.make_layout_tv` to build the tiled copy type around it and the atom of your choice.

.. code-block:: python

    thr_layout = cute.make_layout((4, 32), stride=(32, 1))
    val_layout = cute.make_layout((4, 4), stride=(4, 1))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    # Tile input tensor to thread blocks: ((TileM,TileN),(RestM,RestN))
    gA = cute.zipped_divide(mA, tiler_mn)

Then we can build tiled copy for input and output tensors with `cute.make_tiled_copy_tv` utility, which
infers the tiler and tv layout for the tiled copy automatically, where `tiler` is the tile size per thread
block and `tv_layout` is the TV layout which maps thread index and inter-thread index of data array per
thread to logical coordinates of elements in input and output tensors.

.. code-block:: python

    blkA = gA[((None, None), bidx)]  # (TileM,TileN)

    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    tiled_copy_A = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)

    # get slice of tiled_copy_A for current thread
    thr_copy_A = tiled_copy_A.get_slice(tidx)

    # partition per thread block tensor as source of tiled copy
    thrA = thr_copy_A.partition_S(blkA)

    # allocate fragment for gmem->rmem
    frgA = cute.make_fragment_like(thrA)

    # copy data from global memory to register memory
    cute.copy(copy_atom_load, thrA, frgA)


To run this example:

.. code-block:: bash

    python examples/ampere/elementwise_add.py --M 3 --N 12
    python examples/ampere/elementwise_add.py --M 1024 --N 512
    python examples/ampere/elementwise_add.py --M 1024 --N 1024 --benchmark --warmup_iterations 2 --iterations 1000

To collect performance with NCU profiler:

.. code-block:: bash

    # Don't iterate too many times when profiling with ncu
    ncu python examples/ampere/elementwise_add.py --M 2048 --N 2048 --benchmark --iterations 10 --skip_ref_check
"""


@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    cC: cute.Tensor,  # coordinate tensor
    shape: cute.Shape,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # slice for CTAs
    # logical id -> address
    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]  # (TileM,TileN)
    blkB = gB[blk_coord]  # (TileM,TileN)
    blkC = gC[blk_coord]  # (TileM,TileN)
    blkCrd = cC[blk_coord]  # (TileM, TileN)

    if const_expr(DEBUG_MODE):
        if tidx == 0 and bidx == 0:
            cute.printf("grid_dim = {}", cute.arch.grid_dim())
            cute.printf("block_dim = {}", cute.arch.block_dim())
            cute.printf("shape = {}", shape)
            cute.printf("[DSL INFO] Sliced Tensors per thread block (tid=0, bid=0):\n")
            cute.printf("blkA:")
            cute.print_tensor(blkA)
            cute.printf("blkB:")
            cute.print_tensor(blkB)
            cute.printf("blkC:")
            cute.print_tensor(blkC)
            cute.printf("blkCrd:")
            cute.print_tensor(blkCrd)

    # Declare the atoms which will be used later for memory copy
    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), copy_internal_type=gA.element_type)
    copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), copy_internal_type=gC.element_type)

    # Make tiled copy
    tiled_copy_A = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_B = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_C = cute.make_tiled_copy_tv(copy_atom_store, thr_layout, val_layout)
    
    if const_expr(DEBUG_MODE):
        if tidx == 0 and bidx == 0:
            print("\ncopy_atom_load: ", copy_atom_load)
            print("\ncopy_atom_store: ", copy_atom_store)
            print("\ntiled_copy_A: ", tiled_copy_A)
            print("\ntiled_copy_B: ", tiled_copy_B)
            print("\ntiled_copy_C: ", tiled_copy_C)
            
    # Slice to thread view of tiled copy
    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    thr_copy_C = tiled_copy_C.get_slice(tidx)

    # Partition the thread view of block A/B/C
    thrA = thr_copy_A.partition_S(blkA)
    thrB = thr_copy_B.partition_S(blkB)
    thrC = thr_copy_C.partition_S(blkC)
    thrCrd = thr_copy_C.partition_S(blkCrd)

    # Allocate register fragments to copy from gmem to rmem
    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)
    
    # Make predicate with coord tensor to guard OOB access
    frgPred = cute.make_fragment_like(
        thrCrd,
        dtype=cutlass.Boolean
    )
    for i in range(cute.size(frgPred)):
        # NOTE: thrCrd[i] is the i-th coord that this thread is responsible for, 
        # and shape is the overall shape of the input/output tensor. 
        # We want to guard against OOB access by checking if thrCrd[i] < shape in any dimension. 
        # cute.elem_less does elementwise comparison and returns a boolean tensor, 
        # and since we are comparing coordinate with shape, the result will be False if any dimension exceeds the boundary, 
        # which is what we want for predication.
        val = cute.elem_less(thrCrd[i], shape)
        frgPred[i] = val

    if const_expr(DEBUG_MODE):
        if tidx == 0 and bidx == 0:
            cute.printf("")
            cute.printf("[DSL INFO] Sliced Tensors per thread (tid=0, bid=0):\n")
            cute.printf("thrA:")
            cute.print_tensor(thrA)
            cute.printf("thrB:")
            cute.print_tensor(thrB)
            cute.printf("thrC:")
            cute.print_tensor(thrC)
            cute.printf("thrCrd:")
            cute.print_tensor(thrCrd)
            
            cute.printf("")
            cute.printf("frgPred:")
            cute.print_tensor(frgPred)
            

    ##########################################################
    # Move data to reg address space
    ##########################################################

    cute.copy(copy_atom_load, src=thrA, dst=frgA, pred=frgPred)
    cute.copy(copy_atom_load, src=thrB, dst=frgB, pred=frgPred)

    if const_expr(DEBUG_MODE):
        if tidx == 0 and bidx == 0:
            cute.printf("")
            cute.printf("frgA:")
            cute.print_tensor(frgA)
            cute.printf("frgB:")
            cute.print_tensor(frgB)

    # Load data before use. The compiler will optimize the copy and load
    # operations to convert some memory ld/st into register uses.
    result = frgA.load() + frgB.load()

    # Save the results back to registers. Here we reuse b's registers.
    frgC.store(result)

    # Copy the results back to c
    cute.copy(copy_atom_store, src=frgC, dst=thrC, pred=frgPred)


@cute.jit
def elementwise_add(mA, mB, mC, copy_bits: cutlass.Constexpr = 128):
    dtype = mA.element_type
    vector_size = copy_bits // dtype.width

    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0)) # (4,32):(32,1), row-major
    val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0)) # (4,4):(4,1), row-major, every `vector_size` elems form a vectorized access unit
    # repeat each val layout for each thread,
    # so the tiler_mn layout is (TileM, TileN) = (ThrM x ValM, ThrN x ValN) = (4x4, 32x4) = (16,128)
    # and the tv_layout is packed (thr_layout, val_layout): ((32,4),(4,4)):((64,4),(16,1))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    if const_expr(DEBUG_MODE):
        cute.printf("")
        cute.printf("[DSL INFO] thr_layout: {}", thr_layout)
        cute.printf("[DSL INFO] val_layout: {}", val_layout)
        cute.printf("[DSL INFO]   tiler_mn = {}", tiler_mn)
        cute.printf("[DSL INFO]   tv_layout = {}", tv_layout)
        
        cute.printf("")
        cute.printf("[DSL INFO] Input Tensors:")
        cute.printf("[DSL INFO]   mA.layout={}", mA.layout)
        cute.printf("[DSL INFO]   mB.layout={}", mB.layout)
    
    # =========================================================================
    # Blocked vs. Raked thread-value layout — full tile walkthrough
    #
    # Parameters for this example:
    #   thr_layout = (4,5):(5,1)  →  20 threads in a 4-row × 5-col grid
    #                                 T(i,j) has flat thread id = 5*i + j
    #                                 i.e. T00=id0, T01=id1, T02=id2, T03=id3, T04=id4,
    #                                      T10=id5, T11=id6, ..., T14=id9,
    #                                      T20=id10, ..., T24=id14,
    #                                      T30=id15, ..., T34=id19
    #   val_layout = (2,3):(3,1)  →  each thread owns 2 val-rows × 3 val-cols = 6 values
    #                                 V(p,q) has flat val id = 3*p + q
    #                                 i.e. p=0: V0(q=0), V1(q=1), V2(q=2)   ← val-row 0
    #                                      p=1: V3(q=0), V4(q=1), V5(q=2)   ← val-row 1
    #   tile size = (ThrM×ValM, ThrN×ValN) = (4×2, 5×3) = (8 rows, 15 cols)
    #
    # Cell notation: "Tij,Vk" means tile element is owned by thread T(row=i,col=j)
    #                and is the k-th value in that thread's private fragment.
    #
    # =========================================================================
    # BLOCKED layout  (intuitive but poor memory access pattern)
    # =========================================================================
    #
    #   Assignment:  (m, n)  →  thread T(m//ValM, n//ValN) = T(m//2, n//3)
    #                            val_id = 3*(m%2) + (n%3)
    #
    #   Each thread T(i,j) owns a contiguous ValM×ValN = 2×3 rectangle:
    #     m ∈ [2i, 2i+2),   n ∈ [3j, 3j+3)
    #
    #          n=  0      1      2   |  3      4      5   |  6      7      8   |  9     10     11   | 12     13     14
    #               ←── j=0 ────────→   ←── j=1 ────────→   ←── j=2 ────────→   ←── j=3 ────────→   ←── j=4 ────────→
    #   m=0 i=0: T00,V0 T00,V1 T00,V2 | T01,V0 T01,V1 T01,V2 | T02,V0 T02,V1 T02,V2 | T03,V0 T03,V1 T03,V2 | T04,V0 T04,V1 T04,V2
    #   m=1 i=0: T00,V3 T00,V4 T00,V5 | T01,V3 T01,V4 T01,V5 | T02,V3 T02,V4 T02,V5 | T03,V3 T03,V4 T03,V5 | T04,V3 T04,V4 T04,V5
    #            ─────────────────────────────────────────────── i=0, p=0 and p=1 ──────────────────────────────────────────────────
    #   m=2 i=1: T10,V0 T10,V1 T10,V2 | T11,V0 T11,V1 T11,V2 | T12,V0 T12,V1 T12,V2 | T13,V0 T13,V1 T13,V2 | T14,V0 T14,V1 T14,V2
    #   m=3 i=1: T10,V3 T10,V4 T10,V5 | T11,V3 T11,V4 T11,V5 | T12,V3 T12,V4 T12,V5 | T13,V3 T13,V4 T13,V5 | T14,V3 T14,V4 T14,V5
    #   m=4 i=2: T20,V0 T20,V1 T20,V2 | T21,V0 T21,V1 T21,V2 | T22,V0 T22,V1 T22,V2 | T23,V0 T23,V1 T23,V2 | T24,V0 T24,V1 T24,V2
    #   m=5 i=2: T20,V3 T20,V4 T20,V5 | T21,V3 T21,V4 T21,V5 | T22,V3 T22,V4 T22,V5 | T23,V3 T23,V4 T23,V5 | T24,V3 T24,V4 T24,V5
    #   m=6 i=3: T30,V0 T30,V1 T30,V2 | T31,V0 T31,V1 T31,V2 | T32,V0 T32,V1 T32,V2 | T33,V0 T33,V1 T33,V2 | T34,V0 T34,V1 T34,V2
    #   m=7 i=3: T30,V3 T30,V4 T30,V5 | T31,V3 T31,V4 T31,V5 | T32,V3 T32,V4 T32,V5 | T33,V3 T33,V4 T33,V5 | T34,V3 T34,V4 T34,V5
    #
    #   T00's 6 elements:  (m,n) = (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)  — contiguous 2×3 block
    #
    #   Coalescing analysis — row m=0, thread-row i=0 (T00..T04) all load V0:
    #     T00 → n= 0,  T01 → n= 3,  T02 → n= 6,  T03 → n= 9,  T04 → n=12
    #     Address stride between consecutive threads = ValN = 3  ← NOT coalesced
    #     (each thread's V0 is 3 elements apart; a warp needs multiple cache lines)
    #
    # =========================================================================
    # RAKED layout  (what make_layout_tv produces — coalescing-friendly)
    # =========================================================================
    #
    #   Assignment:  (m, n)  →  thread T(m%ThrM, n%ThrN) = T(m%4, n%5)
    #                            val coords: p = m//ThrM = m//4,  q = n//ThrN = n//5
    #                            val_id = 3*p + q = 3*(m//4) + (n//5)
    #
    #   Each thread T(i,j) owns scattered positions:
    #     m = ThrM*p + i = 4p+i,   n = ThrN*q + j = 5q+j,   p ∈ [0,2),  q ∈ [0,3)
    #
    #          n=  0      1      2      3      4   |  5      6      7      8      9   | 10     11     12     13     14
    #               ←───────── q=0 ──────────────→   ←───────── q=1 ──────────────→   ←───────── q=2 ──────────────→
    #   m=0 i=0,p=0: T00,V0 T01,V0 T02,V0 T03,V0 T04,V0 | T00,V1 T01,V1 T02,V1 T03,V1 T04,V1 | T00,V2 T01,V2 T02,V2 T03,V2 T04,V2
    #   m=1 i=1,p=0: T10,V0 T11,V0 T12,V0 T13,V0 T14,V0 | T10,V1 T11,V1 T12,V1 T13,V1 T14,V1 | T10,V2 T11,V2 T12,V2 T13,V2 T14,V2
    #   m=2 i=2,p=0: T20,V0 T21,V0 T22,V0 T23,V0 T24,V0 | T20,V1 T21,V1 T22,V1 T23,V1 T24,V1 | T20,V2 T21,V2 T22,V2 T23,V2 T24,V2
    #   m=3 i=3,p=0: T30,V0 T31,V0 T32,V0 T33,V0 T34,V0 | T30,V1 T31,V1 T32,V1 T33,V1 T34,V1 | T30,V2 T31,V2 T32,V2 T33,V2 T34,V2
    #                ──────────────────────────────────────── p=0, val-row 0 (V0,V1,V2) ─────────────────────────────────────────────
    #   m=4 i=0,p=1: T00,V3 T01,V3 T02,V3 T03,V3 T04,V3 | T00,V4 T01,V4 T02,V4 T03,V4 T04,V4 | T00,V5 T01,V5 T02,V5 T03,V5 T04,V5
    #   m=5 i=1,p=1: T10,V3 T11,V3 T12,V3 T13,V3 T14,V3 | T10,V4 T11,V4 T12,V4 T13,V4 T14,V4 | T10,V5 T11,V5 T12,V5 T13,V5 T14,V5
    #   m=6 i=2,p=1: T20,V3 T21,V3 T22,V3 T23,V3 T24,V3 | T20,V4 T21,V4 T22,V4 T23,V4 T24,V4 | T20,V5 T21,V5 T22,V5 T23,V5 T24,V5
    #   m=7 i=3,p=1: T30,V3 T31,V3 T32,V3 T33,V3 T34,V3 | T30,V4 T31,V4 T32,V4 T33,V4 T34,V4 | T30,V5 T31,V5 T32,V5 T33,V5 T34,V5
    #                ──────────────────────────────────────── p=1, val-row 1 (V3,V4,V5) ─────────────────────────────────────────────
    #
    #   T00's 6 elements:  (m,n) = (0,0),(0,5),(0,10),(4,0),(4,5),(4,10)  — scattered by (ThrM, ThrN) strides
    #
    #   Coalescing analysis — row m=0, thread-row i=0 (T00..T04) all load V0 (q=0):
    #     T00 → n= 0,  T01 → n= 1,  T02 → n= 2,  T03 → n= 3,  T04 → n= 4
    #     Address stride between consecutive threads = 1  ← perfectly coalesced!
    #     (5 consecutive addresses → served in a single cache-line transaction)
    #
    #   Same property holds for V1 (q=1): threads access n = 5,6,7,8,9   (stride=1)
    #                           for V2 (q=2): threads access n = 10,11,12,13,14 (stride=1)
    #                           for V3..V5 (p=1, m=4): same pattern repeats
    #
    # =========================================================================
    # Summary: why raked layout is the right choice
    # =========================================================================
    #
    #   In raked layout, for any fixed val-index k = 3*p + q, ALL threads
    #   in a thread-row (same i, varying j=0..ThrN-1) access tile columns
    #     n = ThrN*q + j  for j = 0,1,...,ThrN-1
    #   which is exactly a contiguous range [ThrN*q, ThrN*q + ThrN). Stride = 1.
    #
    #   In blocked layout, for val-index k=0 (p=0,q=0), the same threads access
    #     n = ValN*j + 0  for j = 0,1,...,ThrN-1  →  n = 0, 3, 6, 9, 12. Stride = ValN.
    #
    #   The "cost" of raked: each individual thread's values are scattered with
    #   stride ThrN in n (not contiguous). But GPU threads always execute in
    #   lock-step within a warp and issue loads simultaneously — so what matters
    #   is cross-thread address locality at each step, which raked maximizes.
    # =========================================================================
    thr_layout_ = cute.make_layout((4, 5), stride=(5, 1))
    val_layout_ = cute.make_layout((2, 3), stride=(3, 1))
    tiler_mn_, layout_tv_ = cute.make_layout_tv(thr_layout_, val_layout_)
    if const_expr(DEBUG_MODE):
        cute.printf("")
        cute.printf("[DSL INFO] Example of another tiling configuration for the same input tensors")
        cute.printf("[DSL INFO] thr_layout_: {}", thr_layout_)
        cute.printf("[DSL INFO] val_layout_: {}", val_layout_)
        cute.printf("[DSL INFO]   tiler_mn_ = {}", tiler_mn_)
        cute.printf("[DSL INFO]   layout_tv_ = {}", layout_tv_)

    gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM,TileN),(RestM,RestN))
    gB = cute.zipped_divide(mB, tiler_mn)  # ((TileM,TileN),(RestM,RestN))
    gC = cute.zipped_divide(mC, tiler_mn)  # ((TileM,TileN),(RestM,RestN))
    if const_expr(DEBUG_MODE):
        cute.printf("")
        cute.printf("[DSL INFO]   gA.layout={}", gA.layout)
        cute.printf("[DSL INFO]   gB.layout={}", gB.layout)
        cute.printf("[DSL INFO]   gC.layout={}", gC.layout)

    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)
    if const_expr(DEBUG_MODE):
        cute.printf("")
        cute.printf("[DSL INFO]   coord tensor")
        cute.print_tensor(cC)

    elementwise_add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1], # RestM x RestN
        block=[cute.size(tv_layout, mode=[0]), 1, 1], # ThrM x ThrN
    )


def run_elementwise_add(
    M,
    N,
    dtype: Type[cutlass.Numeric],
    is_a_dynamic_layout=False,
    is_b_dynamic_layout=False,
    is_result_dynamic_layout=False,
    skip_ref_check=False,
    benchmark=True,
    warmup_iterations=2,
    iterations=200,
):
    if const_expr(DEBUG_MODE):
        print(f"\nRunning Elementwise Add test with:")
        print(f"Tensor dimensions: [{M}, {N}]")
        print(f"Input and Output Data type: {dtype}")

    torch_dtype = cutlass_torch.dtype(dtype)
    if dtype.is_integer:
        a = torch.randint(0, 10, (M, N), device=torch.device("cuda"), dtype=torch_dtype)
        b = torch.randint(0, 10, (M, N), device=torch.device("cuda"), dtype=torch_dtype)
    else:
        a = torch.randn(M, N, device=torch.device("cuda"), dtype=torch_dtype)
        b = torch.randn(M, N, device=torch.device("cuda"), dtype=torch_dtype)

    c = torch.zeros_like(a)

    if const_expr(DEBUG_MODE):
        print(f"Input tensor shapes:")
        print(f"a: {a.shape}, dtype: {a.dtype}")
        print(f"b: {b.shape}, dtype: {b.dtype}")
        print(f"c: {c.shape}, dtype: {c.dtype}\n")

    if not is_a_dynamic_layout:
        a_tensor = from_dlpack(a).mark_layout_dynamic()
    else:
        a_tensor = a

    if not is_b_dynamic_layout:
        b_tensor = from_dlpack(b).mark_layout_dynamic()
    else:
        b_tensor = b

    if not is_result_dynamic_layout:
        c_tensor = from_dlpack(c).mark_layout_dynamic()
    else:
        c_tensor = c

    if const_expr(DEBUG_MODE):
        print("Compiling kernel with cute.compile ...")
    start_time = time.time()
    compiled_func = cute.compile(elementwise_add, a_tensor, b_tensor, c_tensor)
    compilation_time = time.time() - start_time
    if const_expr(DEBUG_MODE):
        print(f"Compilation time: {compilation_time:.4f} seconds")

    if const_expr(DEBUG_MODE):
        print("Executing vector add kernel...")

    # Get current CUstream from torch
    current_stream = cutlass_torch.current_stream()

    if not skip_ref_check:
        compiled_func(a_tensor, b_tensor, c_tensor)
        if const_expr(DEBUG_MODE):
            print("Verifying results...")
        torch.testing.assert_close(a + b, c)
        if const_expr(DEBUG_MODE):
            print("Results verified successfully!")

    if not benchmark:
        return

    def generate_tensors():
        if dtype.is_integer:
            a = torch.randint(
                0, 10, (M, N), device=torch.device("cuda"), dtype=torch_dtype
            )
            b = torch.randint(
                0, 10, (M, N), device=torch.device("cuda"), dtype=torch_dtype
            )
        else:
            a = torch.randn(M, N, device=torch.device("cuda"), dtype=torch_dtype)
            b = torch.randn(M, N, device=torch.device("cuda"), dtype=torch_dtype)

        c = torch.zeros_like(a)

        if not is_a_dynamic_layout:
            a_tensor = from_dlpack(a).mark_layout_dynamic()
        else:
            a_tensor = a

        if not is_b_dynamic_layout:
            b_tensor = from_dlpack(b).mark_layout_dynamic()
        else:
            b_tensor = b

        if not is_result_dynamic_layout:
            c_tensor = from_dlpack(c).mark_layout_dynamic()
        else:
            c_tensor = c

        return testing.JitArguments(a_tensor, b_tensor, c_tensor)

    avg_time_us = testing.benchmark(
        compiled_func,
        workspace_generator=generate_tensors,
        workspace_count=10,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    # Print execution results
    print(f"Kernel execution time: {avg_time_us / 1e3:.4f} ms")
    print(
        f"Achieved memory throughput: {(3 * a.numel() * dtype.width // 8) / (avg_time_us / 1e6) / 1e9:.2f} GB/s"
    )
    if const_expr(DEBUG_MODE):
        print(f"First few elements of result: \n{c[:3, :3]}")

    # Profiling
    if PROFILE_MODE:
        import sys
        sys.path.insert(0, "..")
        from nvtx import switch_profile, add_nvtx_event

        bytes_moved = 3 * M * N * (dtype.width // 8)
        event_str = f"elementwise_add (M={M}, N={N}, bytes_moved={bytes_moved})"
        iters, start, end = 10, 6, 9
        for i in range(iters):
            switch_profile(
                iter_id=i,
                start=start,
                end=end,
            )
            with add_nvtx_event(event_str):
                compiled_func(a_tensor, b_tensor, c_tensor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="example of elementwise add to demonstrate the numpy/pytorch as input for kernels"
    )
    parser.add_argument("--M", default=1024, type=int)
    parser.add_argument("--N", default=1024, type=int)
    parser.add_argument("--warmup_iterations", default=2, type=int)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--benchmark", action="store_true")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(f"Ampere GPU is required to run this example!")

    run_elementwise_add(
        args.M,
        args.N,
        dtype=cutlass.Float32,
        is_a_dynamic_layout=True,
        is_b_dynamic_layout=True,
        is_result_dynamic_layout=True,
        skip_ref_check=args.skip_ref_check,
        benchmark=args.benchmark,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
    )
    print("\nPASS")
