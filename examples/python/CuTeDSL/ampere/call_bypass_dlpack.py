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

import sys
import os
from typing import Tuple
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr


"""
An Example demonstrating how to call off-the-shelf kernel by-passing dlpack protocol

MOTIVATION
----------
The standard way to bring a PyTorch tensor into a CuTe DSL JIT function is through the
DLPack protocol (from_dlpack). However, DLPack has a known limitation: when a tensor has
a shape-1 dimension, DLPack forces that dimension's stride to 1, silently discarding any
alignment guarantee the user specified via assumed_align. This corrupts downstream
vectorization decisions inside the kernel.

Concrete failure case:

.. code-block:: python

    @cute.kernel
    def fails_kernel(gX: cute.Tensor):
        bidx, _, _ = cute.arch.block_idx()
        mX = gX[None, bidx, None]  # We wish to retain alignment
        # assert mX.iterator.alignment == 16  # <-- This will be WRONG!

    @cute.jit
    def fails(gX_: cute.Tensor):
        gX = gX_
        fails_kernel(gX).launch(grid=(1, 1, 1), block=(128, 1, 1))

    gX_torch = torch.rand((128, 1, 128), device="cuda", dtype=torch.bfloat16)
    fails(from_dlpack(gX_torch, assumed_align=16))
    # DLPack converts the shape-1 mode with stride=1, propagating alignment incorrectly.

KEY TECHNIQUE
-------------
Bypass DLPack entirely by using raw pointers:
  1. Call tensor.data_ptr() to get the raw CUDA device pointer.
  2. Wrap it with make_ptr(dtype, raw_ptr, gmem, assumed_align=N) to create a
     cute.Pointer with an exact, user-controlled alignment guarantee.
  3. Pass the flat pointer + separate shape scalars (as cutlass.Int32) to a thin
     @cute.jit wrapper function.
  4. Inside the JIT function, reconstruct a full CuTe tensor with
     cute.make_ordered_layout + cute.make_tensor — the layout is built from scratch,
     under our complete control, with no DLPack involvement.

The thin JIT wrapper is compiled fully inlined, introducing zero runtime overhead.

To run this example:

.. code-block:: bash

    python examples/ampere/call_bypass_dlpack.py
"""

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tensorop_gemm import TensorOpGemm


@cute.jit
def tensor_op_gemm_wrapper(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    m: cutlass.Int32,
    n: cutlass.Int32,
    k: cutlass.Int32,
    l: cutlass.Int32,
):
    print(f"\n[DSL INFO] Input Parameters:")
    print(f"[DSL INFO]   mnkl: {(m, n, k, l)}")

    # [KEY STEP 1] Provide alignment hints for dynamic shape values.
    # cute.assume(x, divby=N) asserts that `x` is always divisible by N at runtime.
    # The compiler uses this to emit wider (more efficient) memory instructions.
    # Without the hint, the compiler must assume worst-case (non-aligned) shapes,
    # which prevents full vectorization of load/store operations.
    m = cute.assume(m, divby=8)
    n = cute.assume(n, divby=8)

    # [KEY STEP 2] Reconstruct CuTe layouts from raw shape scalars.
    # make_ordered_layout(shape, order): builds a compact strided layout where
    #   order[i] = j means dimension i has the j-th smallest stride (0 = innermost).
    #
    # These layouts must match the physical memory of the PyTorch tensors created
    # in run_tensor_op_gemm_wrapper (see comments there for derivation):
    #   A (M, K, L): order=(0, 1, 2) → M innermost (stride=1), K middle, L outermost
    #   B (N, K, L): order=(0, 1, 2) → N innermost (stride=1), K middle, L outermost
    #   C (M, N, L): order=(1, 0, 2) → N innermost (stride=1), M middle, L outermost
    #                (C is N-major in the MN plane — standard GEMM output convention)
    a_layout = cute.make_ordered_layout((m, k, l), order=(0, 1, 2))
    b_layout = cute.make_ordered_layout((n, k, l), order=(0, 1, 2))
    c_layout = cute.make_ordered_layout((m, n, l), order=(1, 0, 2))

    # [KEY STEP 3] Construct CuTe tensors from raw pointers and explicit layouts.
    # This is the payoff of the bypass-dlpack pattern: make_tensor(ptr, layout)
    # creates a fully-specified CuTe tensor without going through DLPack at all.
    # The pointer carries the exact alignment guarantee set by make_ptr() in the caller.
    mA = cute.make_tensor(a_ptr, layout=a_layout)
    mB = cute.make_tensor(b_ptr, layout=b_layout)
    mC = cute.make_tensor(c_ptr, layout=c_layout)

    print(f"[DSL INFO]   mA: {mA}")
    print(f"[DSL INFO]   mB: {mB}")
    print(f"[DSL INFO]   mC: {mC}")

    # [KEY STEP 4] Instantiate and call the off-the-shelf kernel from inside a JIT function.
    # When a @cute.jit function calls another JIT callable (TensorOpGemm here),
    # the compiler automatically inlines the callee — no cute.compile() call is needed.
    # This is zero-overhead composition: the wrapper JIT and the inner JIT fuse into
    # a single compiled CUDA kernel.
    tensor_op_gemm = TensorOpGemm(
        a_ptr.value_type, c_ptr.value_type, cutlass.Float32, (2, 2, 1)
    )
    print(f"\n[DSL INFO] Created TensorOpGemm instance")
    print(f"[DSL INFO]   Input dtype: {a_ptr.value_type}")
    print(f"[DSL INFO]   Output dtype: {c_ptr.value_type}")
    print(f"[DSL INFO]   Accumulation dtype: {cutlass.Float32}")
    print(f"[DSL INFO]   Atom layout: {(2, 2, 1)}")

    # No need to compile inside jit function
    tensor_op_gemm(mA, mB, mC)
    print(f"\n[DSL INFO] Executed TensorOpGemm")


def run_tensor_op_gemm_wrapper(mnkl: Tuple[int, int, int, int]):
    print(f"\nRunning TensorOpGemm test with:")
    print(f"Tensor dimensions: {mnkl}")

    # Allocate PyTorch tensors with deliberately chosen physical layouts.
    # The goal: each tensor's innermost dimension (stride=1) matches the dimension
    # that the JIT wrapper reconstructs as innermost via make_ordered_layout.
    #
    # Strategy: allocate in an order where the desired innermost dimension is physically
    # last (PyTorch row-major = last allocated dim is stride-1), then use .permute()
    # to relabel axes without copying. permute() changes shape/stride metadata only.
    #
    # A: logical shape (M, K, L), desired physical strides (1, M, M*K) → M innermost
    #   torch.randn(L, K, M) → shape (L,K,M), strides (K*M, M, 1) [row-major, M stride=1]
    #   .permute(2, 1, 0)    → shape (M, K, L), strides (1, M, K*M)  ✓
    # (M,K,L)
    a = torch.randn(
        mnkl[3], mnkl[2], mnkl[0], dtype=torch.float16, device="cuda"
    ).permute(2, 1, 0)
    # B: logical shape (N, K, L), desired physical strides (1, N, N*K) → N innermost
    #   torch.randn(L, K, N).permute(2, 1, 0) → shape (N, K, L), strides (1, N, K*N)  ✓
    # (N,K,L)
    b = torch.randn(
        mnkl[3], mnkl[2], mnkl[1], dtype=torch.float16, device="cuda"
    ).permute(2, 1, 0)
    # C: logical shape (M, N, L), desired physical strides (N, 1, M*N) → N innermost
    #   torch.randn(L, M, N) → shape (L,M,N), strides (M*N, N, 1)
    #   .permute(1, 2, 0)    → shape (M, N, L), strides (N, 1, M*N)  ✓
    # (M,N,L)
    c = torch.randn(
        mnkl[3], mnkl[0], mnkl[1], dtype=torch.float16, device="cuda"
    ).permute(1, 2, 0)

    print(f"Input tensor shapes:")
    print(f"a: {a.shape}, dtype: {a.dtype}")
    print(f"b: {b.shape}, dtype: {b.dtype}")
    print(f"c: {c.shape}, dtype: {c.dtype}\n")

    # [KEY STEP: Bypass DLPack]
    # Instead of from_dlpack(tensor), we extract the raw GPU pointer via data_ptr()
    # and wrap it with make_ptr() to attach explicit type and alignment metadata.
    #
    # make_ptr(dtype, raw_ptr, address_space, assumed_align=N):
    #   - dtype:          element type used by the compiler for type checking
    #   - raw_ptr:        the raw CUDA device pointer (from tensor.data_ptr())
    #   - address_space:  gmem = global GPU memory
    #   - assumed_align:  alignment guarantee in bytes (32 = 256-bit aligned,
    #                     enabling float16 x 16 vectorized loads)
    # This bypasses DLPack entirely, so no shape-1 alignment corruption can occur.
    a_ptr = make_ptr(
        cutlass.Float16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    b_ptr = make_ptr(
        cutlass.Float16, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    c_ptr = make_ptr(
        cutlass.Float16, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )

    # Pass raw pointers + separate shape scalars to the JIT wrapper.
    # Shapes are passed as individual values (cutlass.Int32 inside the JIT function);
    # the JIT function reconstructs the full CuTe tensor layouts internally.
    tensor_op_gemm_wrapper(a_ptr, b_ptr, c_ptr, *mnkl)
    torch.cuda.synchronize()

    ref = torch.einsum("mkl,nkl->mnl", a, b)
    torch.testing.assert_close(c, ref, atol=1e-05, rtol=1e-05)
    print(f"\n[DSL INFO] Results verified successfully!")
    print(f"First few elements of result: \n{c[:3, :3, :3]}")


if __name__ == "__main__":
    run_tensor_op_gemm_wrapper((512, 256, 128, 16))
