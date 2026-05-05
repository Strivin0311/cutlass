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
import operator
import os
import time
from typing import Type, List

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
An Elementwise Apply Example using CuTe DSL.

This example kernel demonstrates the meta-programming capability of the CuTe DSL by allowing
customization of elementwise operations through lambda functions. The kernel copies data from
global memory to register memory (rmem), applies a user-defined operation to the elements,
and stores the result back to global memory.

Primary goals of this example:
1. Demonstrate meta-programming capability by passing lambda functions to customize elementwise operations
2. Show how to apply different operations (add, multiply, etc.) using the same kernel structure
3. Illustrate how to parameterize CUDA kernels with operation types at compile time

To run this example:

.. code-block:: bash

    # Run with addition operation
    python examples/ampere/elementwise_apply.py --M 1024 --N 512 --op add

    # Run with multiplication operation
    python examples/ampere/elementwise_apply.py --M 1024 --N 512 --op mul

    # Run with subtraction operation
    python examples/ampere/elementwise_apply.py --M 1024 --N 512 --op sub

    # Benchmark performance
    python examples/ampere/elementwise_apply.py --M 2048 --N 2048 --op add --benchmark --warmup_iterations 2 --iterations 10

The example demonstrates how to express complex CUDA kernels with customizable operations
while maintaining high performance through efficient memory access patterns.
"""


@cute.kernel
def elementwise_apply_kernel(
    op: cutlass.Constexpr,
    inputs: List[cute.Tensor],
    gC: cute.Tensor,
    cC: cute.Tensor,  # coordinate tensor
    shape: cute.Shape,
    tv_layout: cute.Layout,  # (tid, vid) -> logic coord
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    is_print_thread = (tidx == 0) and (bidx == 0)

    # --- Slice for CTAs ---
    cta_coord = ((None, None), bidx)

    # NOTE: Leverage the meta-programming capability of the DSL to slice the tensors for each input
    # All for loops below on input tensors would be fully unrolled automatically at compile time
    #
    # ctaInputs[0]:
    # tensor(raw_ptr(0x000072986ea00000: f32, gmem, align<4>) o (16,128):(1024,1), data=
    #        [[ 0.000000,  1.000000,  2.000000, ...,  125.000000,  126.000000,  127.000000, ],
    #         [ 1024.000000,  1025.000000,  1026.000000, ...,  1149.000000,  1150.000000,  1151.000000, ],
    #         [ 2048.000000,  2049.000000,  2050.000000, ...,  2173.000000,  2174.000000,  2175.000000, ],
    #         ...
    #         [ 13312.000000,  13313.000000,  13314.000000, ...,  13437.000000,  13438.000000,  13439.000000, ],
    #         [ 14336.000000,  14337.000000,  14338.000000, ...,  14461.000000,  14462.000000,  14463.000000, ],
    #         [ 15360.000000,  15361.000000,  15362.000000, ...,  15485.000000,  15486.000000,  15487.000000, ]])
    ctaInputs = [t[cta_coord] for t in inputs]  # (TileM, TileN)
    
    # slice the tile for this CTA
    ctaC = gC[cta_coord]  # (TileM, TileN)
    
    if const_expr(DEBUG_MODE):
        if is_print_thread:
            cute.printf("")
            cute.printf("ctaInputs[0]:")
            cute.print_tensor(ctaInputs[0])
            cute.printf("")
            cute.printf("ctaInputs[0][(15,0)]:")
            cute.printf(ctaInputs[0][(15,0)])
            cute.printf("")
            cute.printf("ctaInputs[0][15]:")
            cute.printf(ctaInputs[0][15])
    
    # ctaCrd:
    # tensor((0,0) o (16,128):(1@0,1@1), data=
    #        [[ (0,0),  (0,1),  (0,2), ...,  (0,125),  (0,126),  (0,127), ],
    #         [ (1,0),  (1,1),  (1,2), ...,  (1,125),  (1,126),  (1,127), ],
    #         [ (2,0),  (2,1),  (2,2), ...,  (2,125),  (2,126),  (2,127), ],
    #         ...
    #         [ (13,0),  (13,1),  (13,2), ...,  (13,125),  (13,126),  (13,127), ],
    #         [ (14,0),  (14,1),  (14,2), ...,  (14,125),  (14,126),  (14,127), ],
    #         [ (15,0),  (15,1),  (15,2), ...,  (15,125),  (15,126),  (15,127), ]])
    ctaCrd = cC[cta_coord]  # (TileM, TileN)

    if const_expr(DEBUG_MODE):
        if is_print_thread:
            cute.printf("grid_dim = {}", cute.arch.grid_dim())
            cute.printf("block_dim = {}", cute.arch.block_dim())
            cute.printf("shape = {}", shape)
            cute.printf("tidx = {}, bidx = {}", tidx, bidx)
            
            cute.printf("")
            cute.printf("ctaCrd:")
            cute.print_tensor(ctaCrd)

    # --- Compose with CTA TV layout ---
    # tv_layout: ((32,4),(4,4)):((64,4),(16,1))
    # (tid, vid) -> address
    #
    # `cute.composition(A, B)` is pure function composition: R[x] = A[B[x]].
    # Given:
    #   A = ctaA,    layout (16, 128):(1024, 1)   — shape (M=16, N=128), row-major
    #   B = tv_layout = ((32,4),(4,4)):((64,4),(16,1))
    #
    # B maps each (tid, vid) index to a flat offset inside A's domain [0, 16*128).
    # Because CuTe always flattens col-major (mode-0 fastest), the flat offset f
    # maps back to A's 2-D coordinate as:
    #   (M, N) = (f % 16,  f // 16)
    # and A's address stride is then:  addr = M * 1024 + N * 1
    #
    # To find R's stride for each mode of B, take one step in that mode
    # (i.e. advance f by stride_B), convert to (M, N), and apply A's strides:
    #
    #   mode t_inner (size 32, stride_B = 64):
    #     (M, N) = (64 % 16, 64 // 16) = (0, 4)
    #     addr   = 0 * 1024 + 4 * 1   = 4      → R stride for t_inner = 4
    #
    #   mode t_outer (size  4, stride_B =  4):
    #     (M, N) = ( 4 % 16,  4 // 16) = (4, 0)
    #     addr   = 4 * 1024 + 0 * 1   = 4096   → R stride for t_outer = 4096
    #
    #   mode v0      (size  4, stride_B = 16):
    #     (M, N) = (16 % 16, 16 // 16) = (0, 1)
    #     addr   = 0 * 1024 + 1 * 1   = 1      → R stride for v0      = 1
    #
    #   mode v1      (size  4, stride_B =  1):
    #     (M, N) = ( 1 % 16,  1 // 16) = (1, 0)
    #     addr   = 1 * 1024 + 0 * 1   = 1024   → R stride for v1      = 1024
    #
    # Result: R = tidfrgInputs[0], layout ((32,4),(4,4)):((4,4096),(1,1024))
    # Interpretation:
    #   - t_inner (warp-lane, stride 4)   → steps along N by 4 elements → coalesced
    #   - t_outer (warp-group, stride 4096) → steps along M by one row
    #   - v0      (val slow,  stride 1)   → steps along N by 1 → vectorisable
    #   - v1      (val fast,  stride 1024)  → steps along M by one row
    # The rightmost val mode (v0, stride=1 in address) is exactly the N-contiguous
    # direction, so cute.copy can issue a 128-bit vectorised load over v0.

    
    # tidfrgInputs[0]:
    # tensor(raw_ptr(0x000072986ea00000: f32, gmem, align<4>) o ((32,4),(4,4)):((4,4096),(1,1024)), data=
    #        [[ 0.000000,  1.000000,  2.000000, ...,  3073.000000,  3074.000000,  3075.000000, ],
    #         [ 4.000000,  5.000000,  6.000000, ...,  3077.000000,  3078.000000,  3079.000000, ],
    #         [ 8.000000,  9.000000,  10.000000, ...,  3081.000000,  3082.000000,  3083.000000, ],
    #         ...
    #         [ 12404.000000,  12405.000000,  12406.000000, ...,  15477.000000,  15478.000000,  15479.000000, ],
    #         [ 12408.000000,  12409.000000,  12410.000000, ...,  15481.000000,  15482.000000,  15483.000000, ],
    #         [ 12412.000000,  12413.000000,  12414.000000, ...,  15485.000000,  15486.000000,  15487.000000, ]])
    tidfrgInputs = [cute.composition(t, tv_layout) for t in ctaInputs]
    tidfrgC = cute.composition(ctaC, tv_layout)

    # --- coord-layout composition: same mechanics, coordinate strides instead of address strides ---
    # A = ctaCrd,   layout (16, 128):(1@0, 1@1)   — identity coordinate tensor;
    # B = tv_layout = ((32,4),(4,4)):((64,4),(16,1))
    #
    # A's "stride" lives in coordinate space:
    #   1@0  means Δcoord = (+1,  0)   (mode-0 increments)
    #   1@1  means Δcoord = ( 0, +1)   (mode-1 increments)
    # So if the base coord is (b0, b1), then A[(m,n)] gets (b0 + m*1, b1 + n*1) = (b0 + m, b1 + n)
    # and if the A's coord stride turns to (1@1, 1@0), then A[(m,n)] gets (b0 + n*1, b1 + m*1) = (b0 + n, b1 + m)
    # generally, if the A's coord stride is (a@0, b@1), then A[(m,n)] gets (b0 + m*a, b1 + n*b),
    # and if the A's coord stride turns to (a@1, b@0), then A[(m,n)] gets (b0 + n*a, b1 + m*b)
    #
    # For R = A ∘ B: advance f by stride_B, col-major decode (M, N) = (f%16, f//16),
    # then the R coord-stride = M * 1@0  +  N * 1@1:
    #
    #   mode t_inner (size 32, stride_B = 64):
    #     (M, N) = (64 % 16, 64 // 16) = (0, 4)
    #     Δcoord = 0*1@0 + 4*1@1        → R coord-stride = 4@1
    #
    #   mode t_outer (size  4, stride_B =  4):
    #     (M, N) = ( 4 % 16,  4 // 16) = (4, 0)
    #     Δcoord = 4*1@0 + 0*1@1        → R coord-stride = 4@0
    #
    #   mode v0      (size  4, stride_B = 16):
    #     (M, N) = (16 % 16, 16 // 16) = (0, 1)
    #     Δcoord = 0*1@0 + 1*1@1        → R coord-stride = 1@1
    #
    #   mode v1      (size  4, stride_B =  1):
    #     (M, N) = ( 1 % 16,  1 // 16) = (1, 0)
    #     Δcoord = 1*1@0 + 0*1@1        → R coord-stride = 1@0
    #
    # Result: R = tidfrgCrd, layout ((32,4),(4,4)):((4@1,4@0),(1@1,1@0))
    # Reading off the val modes v0,v1: each thread owns a (4M × 4N) block where
    #   v0 (stride 1@1) steps in dim-1 (N) by 1  →  4 consecutive N-elements per row
    #   v1 (stride 1@0) steps in dim-0 (M) by 1  →  4 consecutive M-rows
    # Consistent with the address layout result: v0 is the N-contiguous vectorisable axis.

    # tidfrgCrd:
    # tensor((0,0) o ((32,4),(4,4)):((4@1,4@0),(1@1,1@0)), data=
    #        [[ (0,0),  (0,1),  (0,2), ...,  (3,1),  (3,2),  (3,3), ],
    #         [ (0,4),  (0,5),  (0,6), ...,  (3,5),  (3,6),  (3,7), ],
    #         [ (0,8),  (0,9),  (0,10), ...,  (3,9),  (3,10),  (3,11), ],
    #         ...
    #         [ (12,116),  (12,117),  (12,118), ...,  (15,117),  (15,118),  (15,119), ],
    #         [ (12,120),  (12,121),  (12,122), ...,  (15,121),  (15,122),  (15,123), ],
    #         [ (12,124),  (12,125),  (12,126), ...,  (15,125),  (15,126),  (15,127), ]])
    tidfrgCrd = cute.composition(ctaCrd, tv_layout)
    
    if const_expr(DEBUG_MODE):
        if is_print_thread:
            cute.printf("")
            cute.printf("tidfrgCrd:")
            cute.print_tensor(tidfrgCrd)
            cute.printf("")
            cute.printf("tidfrgInputs[0]:")
            cute.print_tensor(tidfrgInputs[0])
            cute.printf("tidfrgInputs[0][((31,3), (0,0))]:")
            cute.printf(tidfrgInputs[0][((31,3), (0,0))])
            cute.printf("tidfrgInputs[0][127]:") # 127 = 31 + 3 x 32
            cute.printf(tidfrgInputs[0][127])

    # --- Slice for threads ---
    # vid -> address
    thr_coord = (tidx, (None, None))
    
    # thrInputs[0]:
    # tensor(raw_ptr(0x000072986ea00000: f32, gmem, align<4>) o (4,4):(1,1024), data=
    #        [[ 0.000000,  1024.000000,  2048.000000,  3072.000000, ],
    #         [ 1.000000,  1025.000000,  2049.000000,  3073.000000, ],
    #         [ 2.000000,  1026.000000,  2050.000000,  3074.000000, ],
    #         [ 3.000000,  1027.000000,  2051.000000,  3075.000000, ]])
    thrInputs = [t[thr_coord] for t in tidfrgInputs]  # (V)
    thrC = tidfrgC[thr_coord]  # (V)
    
    # thrCrd:
    # tensor((0,0) o (4,4):(1@1,1@0), data=
    #        [[ (0,0),  (1,0),  (2,0),  (3,0), ],
    #         [ (0,1),  (1,1),  (2,1),  (3,1), ],
    #         [ (0,2),  (1,2),  (2,2),  (3,2), ],
    #         [ (0,3),  (1,3),  (2,3),  (3,3), ]])
    thrCrd = tidfrgCrd[thr_coord]

    # --- Allocate fragments for gmem->rmem ---
    frgInputs = [cute.make_fragment_like(t, t.element_type) for t in thrInputs]
    frgC = cute.make_fragment_like(thrC, gC.element_type)
    frgPred = cute.make_fragment_like(thrCrd, cutlass.Boolean)

    for i in cutlass.range(cute.size(frgPred), unroll=1):
        frgPred[i] = cute.elem_less(thrCrd[i], shape)

    if const_expr(DEBUG_MODE):
        if is_print_thread:
            cute.printf("")
            cute.printf("thrCrd:")
            cute.print_tensor(thrCrd)
            cute.printf("")
            cute.printf("thrInputs[0]:")
            cute.print_tensor(thrInputs[0])

    ##########################################################
    # Move data to reg address space
    ##########################################################

    # Declare the atoms which will be used later for memory copy
    # Compile time validation: expect same element type for all input tensors so as to reuse the copy atom for load
    assert all(t.element_type == inputs[0].element_type for t in inputs)

    copy_atom_load = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        copy_internal_type=inputs[0].element_type,
        num_bits_per_copy=inputs[0].element_type.width,
    )
    copy_atom_store = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        copy_internal_type=gC.element_type,
        num_bits_per_copy=gC.element_type.width,
    )

    for thrInput, frgInput in zip(thrInputs, frgInputs):
        cute.copy(copy_atom_load, thrInput, frgInput, pred=frgPred)

    # Load data before use. The compiler will optimize the copy and load
    # operations to convert some memory ld/st into register uses.
    result = op(*[frgInput.load() for frgInput in frgInputs])

    # Save the results back to registers. Here we reuse b's registers.
    frgC.store(result)

    # Copy the results back to c
    cute.copy(copy_atom_store, frgC, thrC, pred=frgPred)


@cute.jit
def elementwise_apply(
    op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    result: cute.Tensor,
    stream: cuda.CUstream,
):
    """CUDA kernel applying binary operator on each element of two n-D input tensors in
    CuTe Python and store to result tensor.

    :param op: Binary operator or lambda function to apply element-wise
    :type op: cutlass.Constexpr
    :param a: First input tensor
    :type a: cute.Tensor
    :param b: Second input tensor
    :type b: cute.Tensor
    :param result: Output tensor to store the results of op(a, b)
    :type result: cute.Tensor
    :return: None
    :rtype: None

    .. code-block:: python

        # Example 1: Adding two tensors
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device="cuda")
        y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32, device="cuda")
        result = torch.empty_like(x)
        elementwise_apply(operator.add, from_dlpack(x), from_dlpack(y), from_dlpack(result))
        # result:
        # tensor([[6.0, 8.0],
        #         [10.0, 12.0]], device='cuda:0')

        # Example 2: Using a lambda function
        elementwise_apply(lambda a, b: a * a + b * b, from_dlpack(x), from_dlpack(y), from_dlpack(result))
        # result:
        # tensor([[  2.,   8.],
        #         [ 54., 512.]], device='cuda:0')
    """

    # Baseline: naive TV layout
    #   * mA layout: (4096, 4096):(4096, 1)
    #   * TV layout map to (512, 4) tile
    #   * tidx maps to mode-0 but input layout is contiguous on mode-1, performance will be bad
    # tv_layout = cute.make_layout((128, (4, 4)), stride=(4, (512, 1)))
    # cta_tiler = (512, 4)

    # Opt-1: better TV layout with better 1D thread layout (SOL with 1D thread layout)
    #   * mA layout: (4096, 4096):(4096, 1)
    #   * TV layout map to (4, 512) tile
    #   * tidx maps to mode-1 which is leading mode of input tensor for coalesced load
    # tv_layout = cute.make_layout((128, (4, 4)), stride=(16, (4, 1)))
    # cta_tiler = (4, 512)

    # Opt-2: 2D tile but worse
    #   * mA layout: (4096, 4096):(4096, 1)
    #   * TV layout map to (128, 16) logical tile
    #   * V layout is bad as contiguous mode is not on right-most
    #     * `cute.copy` only supports vectorize when stride-1 of v-layout on right-most )
    # tv_layout = cute.make_layout(((32, 4), (4, 4)), stride=((4, 512), (1, 128)))
    # cta_tiler = (128, 16)

    # Opt-3: SOL with 2D thread tile
    #   * mA layout: (4096, 4096):(4096, 1)
    #   * TV layout map to (16, 128) logical tile
    #   * tidx maps to mode-1 and input layout is contiguous on mode-1 for coalesced load-store
    # tv_layout = ((32,4),(4,4)):((64,4),(16,1))
    thr_layout = cute.make_layout((4, 32), stride=(32, 1))
    val_layout = cute.make_layout((4, 4), stride=(4, 1))

    # --- Part 1: How to define thr_layout and val_layout ---
    #
    # Given:
    #   input layout  (M, N):(strideM, 1)  — N is the memory-contiguous direction
    #   block size    T threads  (here 128)
    #   val count     V elements per thread  (here 16)
    #   total tile    T * V = 2048 elements = TileM * TileN
    #
    # Choose tile shape: make TileN large to exploit N-contiguity.  Here TileM=16, TileN=128.
    #
    # thr_layout = (t_out, t_in) describes how threads tile the (TileM, TileN) space:
    #   - t_in  is the fast tid axis (warp lane, tid % 32 = 32 threads).
    #     It must walk along N so adjacent lanes access adjacent N elements → coalesced.
    #   - t_out is the slow tid axis (warp index, tid // 32 = 4 warps).
    #     It walks along M, stepping over the M-block each warp covers.
    #   Convention: write (t_out, t_in) so that mode-1 (t_in) is fastest.
    #     thr_layout = (4, 32):(32, 1)   ← shape only; make_layout_tv fills the strides
    #
    # val_layout = (v_out, v_in) describes the per-thread (valM × valN) sub-block:
    #   - v_in  is the fast vid axis (4 elements along N).
    #     It must walk along N so cute.copy can issue a 128-bit vectorised load.
    #   - v_out is the slow vid axis (4 elements along M).
    #     It walks along M, stepping one row at a time.
    #   Convention: write (v_out, v_in) so that mode-1 (v_in) is fastest.
    #     val_layout = (4, 4):(4, 1)   ← v_in is stride-1 in val space → fastest flat offset

    # --- Part 2: How make_layout_tv derives tv_layout from thr_layout and val_layout ---
    #
    # Step 1 — write down the logical (M, N) coordinate each (t, v) index lands on:
    #   t_out controls which M-block the warp covers
    #   t_in  controls which N-block the lane covers
    #   v_out steps along M with block size valM=4
    #   v_in  steps along N with block size valN=4
    #
    #   M_coord = t_out * size(v_out) + v_out = t_out * valM + v_out   ∈ [0, 16)
    #   N_coord = t_in  * valN + v_in  = t_in  * valN + v_in    ∈ [0, 128)
    #
    # Step 2 — convert to a col-major flat offset (TileM=16 is the fast dimension):
    #
    #   flat_coord  = M_coord + N_coord * TileM
    #               = (t_out*valM + v_out) + (t_in*valN + v_in) * TileM
    #               = (t_out*4 + v_out) + (t_in*4 + v_in) * 16
    #               = t_in*64 + t_out*4 + v_in*16 + v_out*1
    #
    # Step 3 — read off the flat stride for each mode directly from the expanded formula:
    #
    #   mode    size   flat stride
    #   t_in     32       64
    #   t_out     4        4
    #   v_in      4       16
    #   v_out     4        1
    #
    # Step 4 — assemble tv_layout:
    #   Convention: tid modes in mode-0, vid modes in mode-1;
    #   within each group, inner (fast) mode listed first, outer (slow) mode second.
    #
    #   tid part: (t_in=32, t_out=4) : (64, 4)
    #   vid part: (v_in=4,  v_out=4) : (16,  1)
    #
    # Result: tv_layout = ((32,4),(4,4)):((64,4),(16,1))
    #                       ^tid^   ^vid^   ^tid^  ^vid^

    # --- Part 3: Given a tv_layout, how to verify it is correct ---
    #
    # For each mode, decode its flat stride back to a (M_step, N_step) tile coordinate
    # and then to an address step:
    #   (M_step, N_step) = (flat_stride % TileM,  flat_stride // TileM)
    #   addr_step        =  M_step * strideM      + N_step * 1
    #
    # Applied to tv_layout = ((32,4),(4,4)):((64,4),(16,1)), TileM=16, strideM=1024:
    #
    #   mode    size  flat_stride  (M_step, N_step)  addr_step  verdict
    #   t_in     32      64       (64%16, 64//16)=(0,4)    4    32 lanes × 4-N gap → fully coalesced ✓
    #   t_out     4       4       ( 4%16,  4//16)=(4,0)  4096   4 warps × 1 M-row gap ✓
    #   v_in      4      16       (16%16, 16//16)=(0,1)     1   4 consecutive N → 128-bit load ✓
    #   v_out     4       1       ( 1%16,  1//16)=(1,0)  1024   4 consecutive M-rows ✓
    #
    # Two key criteria for a good TV layout:
    #   1. t_in (fastest tid mode) must decode to the N direction (M_step=0, addr_step small)
    #      → ensures the whole warp accesses contiguous memory → coalesced.
    #   2. v_in (fastest vid mode, listed first inside the vid group) must decode to addr_step=1
    #      → ensures cute.copy can issue a 128-bit vectorised load over v_in.
    #
    # Counter-example (Opt-2): tv_layout = ((32,4),(4,4)):((4,512),(1,128)), TileM=128
    #   v_in flat_stride=1 → (1%128, 1//128) = (1, 0) → addr_step = 1*strideM = 4096 ≠ 1 → no vectorisation ✗
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    if const_expr(DEBUG_MODE):
        cute.printf("")
        cute.printf("[DSL INFO] thr_layout: {}", thr_layout)
        cute.printf("[DSL INFO] val_layout: {}", val_layout)
        cute.printf("[DSL INFO]   tiler_mn = {}", tiler_mn)
        cute.printf("[DSL INFO]   tv_layout = {}", tv_layout)

        cute.printf("")
        cute.printf("[DSL INFO] Input Tensors:")
        cute.printf("[DSL INFO]   a = {}", a.layout)
        cute.printf("[DSL INFO]   b = {}", b.layout)
        cute.printf("[DSL INFO]   result = {}", result.layout)

    gA = cute.zipped_divide(a, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    gB = cute.zipped_divide(b, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    gC = cute.zipped_divide(result, tiler_mn)  # ((TileM, TileN), (RestM, RestN))

    if const_expr(DEBUG_MODE):
        cute.printf("")
        cute.printf("[DSL INFO] Tiled Tensors:")
        cute.printf("[DSL INFO]   gA.layout={}", gA.layout)
        cute.printf("[DSL INFO]   gB.layout={}", gB.layout)
        cute.printf("[DSL INFO]   gC.layout={}", gC.layout)

    # idC:
    # tensor((0,0) o (2048,1024):(1@0,1@1), data=
    #        [[ (0,0),  (0,1),  (0,2), ...,  (0,1021),  (0,1022),  (0,1023), ],
    #         [ (1,0),  (1,1),  (1,2), ...,  (1,1021),  (1,1022),  (1,1023), ],
    #         [ (2,0),  (2,1),  (2,2), ...,  (2,1021),  (2,1022),  (2,1023), ],
    #         ...
    #         [ (2045,0),  (2045,1),  (2045,2), ...,  (2045,1021),  (2045,1022),  (2045,1023), ],
    #         [ (2046,0),  (2046,1),  (2046,2), ...,  (2046,1021),  (2046,1022),  (2046,1023), ],
    #         [ (2047,0),  (2047,1),  (2047,2), ...,  (2047,1021),  (2047,1022),  (2047,1023), ]])
    idC = cute.make_identity_tensor(result.shape) # (M, N)
    
    # cC:
    # tensor((0,0) o ((16,128),(128,8)):((1@0,1@1),(16@0,128@1)), data=
    #        [[ (0,0),  (16,0),  (32,0), ...,  (2000,896),  (2016,896),  (2032,896), ],
    #         [ (1,0),  (17,0),  (33,0), ...,  (2001,896),  (2017,896),  (2033,896), ],
    #         [ (2,0),  (18,0),  (34,0), ...,  (2002,896),  (2018,896),  (2034,896), ],
    #         ...
    #         [ (13,127),  (29,127),  (45,127), ...,  (2013,1023),  (2029,1023),  (2045,1023), ],
    #         [ (14,127),  (30,127),  (46,127), ...,  (2014,1023),  (2030,1023),  (2046,1023), ],
    #         [ (15,127),  (31,127),  (47,127), ...,  (2015,1023),  (2031,1023),  (2047,1023), ]])
    cC = cute.zipped_divide(idC, tiler=tiler_mn) # ((TileM, TileN), (RestM, RestN))

    if const_expr(DEBUG_MODE):
        cute.printf("")
        cute.printf("idC:")
        cute.print_tensor(idC)
        cute.printf("idC[2049]: {}", idC[2049]) # (1,1)
        cute.printf("")
        cute.printf("cC:")
        cute.print_tensor(cC)

    # Launch the kernel asynchronously
    # Async token(s) can also be specified as dependencies
    elementwise_apply_kernel(
        op,
        [gA, gB],  # Group input tensors into a list as a single argument
        gC,
        cC,
        result.shape,
        tv_layout,
    ).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
        stream=stream,
    )


def run_elementwise_apply_and_verify(
    op,
    M,
    N,
    dtype: Type[cutlass.Numeric],
    skip_ref_check=False,
    benchmark=True,
    warmup_iterations=2,
    iterations=100,
):
    if not torch.cuda.is_available():
        raise RuntimeError(f"Ampere GPU is required to run this example!")

    # Create non default CUDA stream from PyTorch
    torch_stream = torch.cuda.Stream()
    # Get the raw stream pointer as a CUstream
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    if DEBUG_MODE:
        print(f"\nRunning Elementwise Apply test with:")
        print(f"Tensor dimensions: [{M}, {N}]")
        print(f"Input and Output Data type: {dtype}")

    torch_dtype = cutlass_torch.dtype(dtype)

    # Allocate tensors with random values.
    # a = torch.randn(M, N, device=torch.device("cuda"), dtype=torch_dtype)
    a = torch.arange(M*N, device=torch.device("cuda"), dtype=torch_dtype).reshape(M, N)
    b = torch.randn(M, N, device=torch.device("cuda"), dtype=torch_dtype)
    c = torch.zeros_like(a)

    if DEBUG_MODE:
        print(f"Input tensor shapes:")
        print(f"a: {a.shape}, dtype: {a.dtype}")
        print(f"b: {b.shape}, dtype: {b.dtype}")
        print(f"c: {c.shape}, dtype: {c.dtype}\n")

    epsilon = 1.2
    if op in (operator.truediv, operator.floordiv):
        b = torch.where(b == 0, torch.tensor(epsilon), b)

    if DEBUG_MODE:
        print("Compiling kernel with cute.compile ...")
    start_time = time.time()
    compiled_func = cute.compile(
        elementwise_apply,
        op,
        from_dlpack(a),
        from_dlpack(b),
        from_dlpack(c).mark_layout_dynamic(),
        current_stream,
    )
    compilation_time = time.time() - start_time
    if DEBUG_MODE:
        print(f"Compilation time: {compilation_time:.4f} seconds")

    if DEBUG_MODE:
        print("Executing elementwise apply kernel...")

    if not skip_ref_check:
        compiled_func(
            from_dlpack(a),
            from_dlpack(b),
            from_dlpack(c).mark_layout_dynamic(),
            current_stream,
        )
        if DEBUG_MODE:
            print("Verifying results...")
        torch.testing.assert_close(op(a, b), c)
        if DEBUG_MODE:
            print("Results verified successfully!")

    if not benchmark:
        return

    avg_time_us = testing.benchmark(
        compiled_func,
        kernel_arguments=testing.JitArguments(
            from_dlpack(a),
            from_dlpack(b),
            from_dlpack(c).mark_layout_dynamic(),
            current_stream,
        ),
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        use_cuda_graphs=True,
        stream=current_stream,
    )

    # Print execution results
    print(f"Kernel execution time: {avg_time_us / 1e3:.4f} ms")
    print(
        f"Achieved memory throughput: {(3 * a.numel() * dtype.width // 8) / (avg_time_us / 1e6) / 1e9:.2f} GB/s"
    )
    if DEBUG_MODE:
        print(f"First few elements of result: \n{c[:3, :3]}")

    # Profiling
    if PROFILE_MODE:
        import sys
        sys.path.insert(0, "..")
        from nvtx import switch_profile, add_nvtx_event

        bytes_moved = 3 * M * N * (dtype.width // 8)
        event_str = f"elementwise_apply (M={M}, N={N}, bytes_moved={bytes_moved})"
        iters, start, end = 10, 6, 9
        for i in range(iters):
            switch_profile(
                iter_id=i,
                start=start,
                end=end,
            )
            with add_nvtx_event(event_str):
                compiled_func(
                    from_dlpack(a),
                    from_dlpack(b),
                    from_dlpack(c).mark_layout_dynamic(),
                    current_stream,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="example of elementwise apply to demonstrate building elementwise kernels"
    )
    parser.add_argument("--M", default=1024, type=int)
    parser.add_argument("--N", default=1024, type=int)
    parser.add_argument("--op", default="add", type=str)
    parser.add_argument("--warmup_iterations", default=0, type=int)
    parser.add_argument("--iterations", default=1, type=int)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    run_elementwise_apply_and_verify(
        getattr(operator, args.op),
        args.M,
        args.N,
        dtype=cutlass.Float32,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        skip_ref_check=args.skip_ref_check,
        benchmark=args.benchmark,
    )
    print("\nPASS")
