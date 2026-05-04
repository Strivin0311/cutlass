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

"""
Demonstrating How to Define a Custom Type That Can Cross JIT Boundaries (JIT-to-JIT Calling)

MOTIVATION
----------
When a @cute.jit function (JIT-A) calls another @cute.jit function (JIT-B), only values
representable as MLIR SSA values (e.g., raw pointers, integer scalars) can cross the call
boundary at runtime. A custom Python object that bundles both a dynamic field (pointer)
and a static field (layout convention) cannot be passed by default.

This example shows how to make such a custom type JIT-compatible by implementing two protocols:

  1. JitArgument Protocol — enables Python → compiled JIT function calls:
       __c_pointers__()              returns raw C pointers for the CUDA runtime call
       __get_mlir_types__()          returns MLIR type signatures for compiler code generation
       __new_from_mlir_values__(v)   reconstructs the Python object from MLIR function arguments

  2. DynamicExpression Protocol — additionally enables JIT-A → JIT-B inlined calls:
       __extract_mlir_values__()     serializes the object into MLIR SSA values at the call site
       __new_from_mlir_values__(v)   deserializes back from MLIR arguments inside JIT-B body

KEY DESIGN INSIGHT
------------------
BufferWithLayout splits its state into two tiers:
  - DYNAMIC: ptr (cute.Pointer) — the actual GPU address, variable at runtime.
    This flows through MLIR SSA and is the only value crossing the JIT boundary.
  - STATIC:  stride_order (tuple) — the layout convention, fixed at compile time.
    This NEVER enters MLIR. In __new_from_mlir_values__, it is copied from the
    "old" Python object's attribute rather than being reconstructed from MLIR.

Because all protocol methods delegate entirely to self.ptr (one pointer = one slot),
the required consistency constraint is automatically satisfied:
    len(__c_pointers__()) == len(__get_mlir_types__()) == len(__extract_mlir_values__())

The example uses GEMM as a concrete workload, but the technique is general: any custom
type with mixed static/dynamic state can be made JIT-passable with this pattern.

To run this example:

.. code-block:: bash

    python examples/ampere/call_from_jit.py

Default configuration:
- Batch dimension (L): 16
- Matrix dimensions: M=512, N=256, K=128
- Precision: Float16 inputs with Float32 accumulation
"""

import os
import sys
from typing import Type, Tuple

import torch

import cutlass
import cutlass.cute as cute
from cutlass.torch import dtype as torch_dtype
from cutlass.cute.runtime import make_ptr


# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tensorop_gemm import TensorOpGemm


# BufferWithLayout is a custom type designed to cross JIT call boundaries.
# It demonstrates the essential design pattern for JIT-compatible custom types:
#   - Bundle a DYNAMIC field (ptr) that must travel through MLIR SSA.
#   - Bundle a STATIC field (stride_order) that is baked in at compile time and
#     directly copied in __new_from_mlir_values__ without entering MLIR at all.
class BufferWithLayout:
    def __init__(self, ptr: cute.Pointer, stride_order: tuple[int, int, int]):
        # DYNAMIC FIELD: the raw GPU memory pointer.
        # This is the only part that becomes an MLIR SSA value when crossing JIT
        # boundaries. Its concrete address is unknown at compile time.
        self.ptr = ptr

        # STATIC FIELD: describes which dimension of the tensor is innermost (stride=1).
        # stride_order[i] = j means dimension i has the j-th smallest stride (0 = innermost).
        # This is a compile-time constant — it does NOT flow through MLIR; instead it is
        # baked into generated code by being directly copied in __new_from_mlir_values__.
        self.stride_order = stride_order

    def to_tensor(
        self, shape: tuple[int, int, int], *, loc=None, ip=None
    ) -> cute.Tensor:
        # Called INSIDE a JIT function to materialize a CuTe tensor from:
        #   - self.ptr:          the dynamic GPU pointer (an MLIR SSA value here)
        #   - self.stride_order: the static layout convention (compile-time constant)
        #   - shape:             provided by the caller (Constexpr tuple or dynamic values)
        assert len(shape) == len(self.stride_order), (
            f"Shape {shape} and stride_order {self.stride_order} must have the "
            "same rank."
        )
        # Build a CuTe layout matching the physical memory of the tensor.
        # stride_order[i]=j: dimension i has the j-th smallest stride.
        # For stride_order=(2,1,0) and shape=(l, m_or_n, k):
        #   dim0(l)     gets rank 2 (outermost),
        #   dim1(m/n)   gets rank 1 (middle),
        #   dim2(k)     gets rank 0 (innermost, stride=1)
        # This matches PyTorch row-major tensors allocated as (L, M, K): strides (M*K, K, 1).
        layout = cute.make_ordered_layout(shape, self.stride_order)
        # The input shape is ordered as (l, mn, k) — batch first, matching PyTorch allocation.
        # TensorOpGemm expects (mn, k, l) — spatial dims first, batch last.
        # cute.select(layout, mode=[1, 2, 0]) permutes the modes (no data copy).
        res = cute.make_tensor(self.ptr, cute.select(layout, mode=[1, 2, 0]))
        return res

    # =========================================================================
    # JitArgument Protocol + DynamicExpression Protocol implementation
    #
    # These four methods make BufferWithLayout passable across JIT call boundaries.
    # Together they define how the compiler serializes/deserializes this object
    # when it needs to cross from Python into compiled code, or from JIT-A into JIT-B.
    #
    # Required consistency constraint (MUST always hold):
    #   len(__c_pointers__()) == len(__get_mlir_types__()) == len(__extract_mlir_values__())
    #
    # For BufferWithLayout, all methods delegate entirely to self.ptr (one pointer = one
    # slot), so the constraint is trivially satisfied. The static field `stride_order`
    # is handled out-of-band: it is directly copied in __new_from_mlir_values__ and
    # never appears in any of the three length-constrained lists.
    # =========================================================================

    def __c_pointers__(self):
        """[JitArgument] Called at runtime when Python invokes a compiled JIT function.

        The JIT executor calls this to obtain the raw C pointers that are forwarded
        to the underlying compiled CUDA function. Each returned pointer corresponds to
        one MLIR function parameter (must align one-for-one with __get_mlir_types__).

        For BufferWithLayout: only the dynamic ptr field is exposed here.
        stride_order is NOT exposed — it is a compile-time constant, not a runtime value.
        """
        return self.ptr.__c_pointers__()

    def __get_mlir_types__(self):
        """[JitArgument] Called at compile time to determine the MLIR function signature.

        Returns the MLIR types for each dynamic value this object contributes to the
        compiled function's parameter list. Must be length-consistent with
        __c_pointers__() and __extract_mlir_values__().

        For BufferWithLayout: one entry (the pointer MLIR type), delegated to self.ptr.
        """
        return self.ptr.__get_mlir_types__()

    def __extract_mlir_values__(self):
        """[DynamicExpression] Called inside JIT-A when it issues a call to JIT-B.

        At the MLIR call site inside JIT-A's generated code, this method serializes
        the object into a list of MLIR SSA values that become the actual arguments
        passed to JIT-B. Must be length-and-type consistent with __get_mlir_types__().

        For BufferWithLayout: extracts the single pointer SSA value from self.ptr.
        """
        return self.ptr.__extract_mlir_values__()

    def __new_from_mlir_values__(self, values):
        """[JitArgument + DynamicExpression] Reconstructs this Python object from MLIR values.

        This method is invoked in two scenarios:
          1. Building the function BODY of a compiled JIT function (JitArgument):
             the compiler passes the formal MLIR parameters here to construct a usable
             Python object for use inside the function body.
          2. Building the function BODY of JIT-B called from JIT-A (DynamicExpression):
             same reconstruction, triggered by the inlined inner-call mechanism.

        KEY POINT: `values` carries ONLY the dynamic field (ptr). The static field
        (stride_order) is NOT in `values` — it is copied from `self` (the prototype
        object used at compile time). This is the mechanism by which the static/dynamic
        split works: the compiler always has access to the "old" object's Python attributes.

        :param values: MLIR SSA values for the dynamic fields (just the pointer here)
        :return: A new BufferWithLayout with ptr rebuilt from `values` and
                 stride_order copied from the current object's attribute.
        """
        return BufferWithLayout(
            self.ptr.__new_from_mlir_values__(values), self.stride_order
        )


@cute.jit
def tensor_op_gemm_wrapper(
    buffer_a: BufferWithLayout,
    buffer_b: BufferWithLayout,
    buffer_c: BufferWithLayout,
    mnkl: cutlass.Constexpr[tuple[int, int, int, int]],
    acc_dtype: Type[cutlass.Numeric],
    atom_layout_mnk: cutlass.Constexpr[tuple[int, int, int]],
):
    print(f"\n[DSL INFO] Input Parameters:")
    print(f"[DSL INFO]   mnkl: {mnkl}")
    print(f"[DSL INFO]   buffer_a: {buffer_a}")
    print(f"[DSL INFO]   buffer_b: {buffer_b}")
    print(f"[DSL INFO]   buffer_c: {buffer_c}")
    print(f"[DSL INFO]   acc_dtype: {acc_dtype}")
    print(f"[DSL INFO]   atom_layout_mnk: {atom_layout_mnk}")

    # [KEY STEP 1] Reconstruct CuTe tensors from the BufferWithLayout objects.
    # At this point inside the JIT function, buffer_a/b/c are "rehydrated" Python objects:
    # their ptr field is an MLIR SSA value (rebuilt via __new_from_mlir_values__), and
    # their stride_order is the compile-time constant copied from the original object.
    #
    # cute.select(mnkl, mode=[3, 0, 2]) picks elements [mnkl[3], mnkl[0], mnkl[2]]
    # = (L, M, K), providing the (l, mn, k) shape that to_tensor() expects (batch-first).
    # to_tensor() then applies stride_order to produce the final (M, K, L) CuTe tensor
    # with the correct physical layout.
    mA = buffer_a.to_tensor(cute.select(mnkl, mode=[3, 0, 2]))  # (l,m,k) -> tensor (m,k,l)
    mB = buffer_b.to_tensor(cute.select(mnkl, mode=[3, 1, 2]))  # (l,n,k) -> tensor (n,k,l)
    mC = buffer_c.to_tensor(cute.select(mnkl, mode=[3, 0, 1]))  # (l,m,n) -> tensor (m,n,l)

    print(f"\n[DSL INFO] Created Tensors:")
    print(f"[DSL INFO]   mA = {mA}")
    print(f"[DSL INFO]   mB = {mB}")
    print(f"[DSL INFO]   mC = {mC}")

    # [KEY STEP 2] Instantiate the JIT kernel callable inside a JIT function.
    # TensorOpGemm is a callable object wrapping a @cute.jit GEMM implementation.
    # It can be constructed here because all its constructor arguments (dtypes, atom
    # layout) are compile-time constants (Constexpr), so no runtime state is needed.
    tensor_op_gemm = TensorOpGemm(
        buffer_a.ptr.value_type,
        buffer_c.ptr.value_type,
        acc_dtype,
        atom_layout_mnk,
    )
    print(f"\n[DSL INFO] Created TensorOpGemm instance")
    print(f"[DSL INFO]   Input dtype: {buffer_a.ptr.value_type}")
    print(f"[DSL INFO]   Output dtype: {buffer_c.ptr.value_type}")
    print(f"[DSL INFO]   Accumulation dtype: {acc_dtype}")
    print(f"[DSL INFO]   Atom layout: {atom_layout_mnk}")

    # [KEY STEP 3] Call a JIT kernel from inside another JIT function (JIT-to-JIT).
    # The CuTe DSL compiler automatically inlines tensor_op_gemm into this wrapper —
    # no cute.compile() call is needed inside a JIT function. The outer and inner JIT
    # functions are fused into a single compiled CUDA kernel with zero call overhead.
    # No need to compile inside jit function
    tensor_op_gemm(mA, mB, mC)
    print(f"\n[DSL INFO] Executed TensorOpGemm")


def run_tensor_op_gemm_wrapper(mnkl: Tuple[int, int, int, int]):
    print(f"\nRunning TensorOpGemm test with:")
    print(f"Tensor dimensions: {mnkl}")

    ab_dtype = cutlass.Float16
    c_dtype = cutlass.Float16

    # Allocate standard contiguous PyTorch tensors in (L, M/N, K) order (batch-first,
    # row-major). Physical strides for shape (L, M, K) are (M*K, K, 1) — K is innermost.
    # This matches stride_order=(2, 1, 0): dim2(K) gets rank 0 (innermost), which
    # is exactly what to_tensor() will reconstruct inside the JIT function.
    a = torch.randn(
        mnkl[3], mnkl[0], mnkl[2], dtype=torch_dtype(ab_dtype), device="cuda"
    )
    b = torch.randn(
        mnkl[3], mnkl[1], mnkl[2], dtype=torch_dtype(ab_dtype), device="cuda"
    )
    c = torch.randn(
        mnkl[3], mnkl[0], mnkl[1], dtype=torch_dtype(c_dtype), device="cuda"
    )

    print(f"Input tensor shapes:")
    print(f"a: {a.shape}, dtype: {a.dtype}")
    print(f"b: {b.shape}, dtype: {b.dtype}")
    print(f"c: {c.shape}, dtype: {c.dtype}\n")

    # [KEY STEP: Build BufferWithLayout objects]
    # Each BufferWithLayout bundles two pieces of information:
    #
    #   1. make_ptr(dtype, data_ptr(), gmem, assumed_align=32):
    #      Bypasses DLPack — extracts the raw GPU pointer directly from the PyTorch
    #      tensor with an explicit 32-byte alignment guarantee.
    #      This becomes the DYNAMIC field (ptr) that flows through MLIR SSA.
    #
    #   2. stride_order=(2, 1, 0):
    #      The layout convention — dim2 (K) is innermost (rank 0, stride=1),
    #      dim1 (M or N) is middle (rank 1), dim0 (L) is outermost (rank 2).
    #      This matches the physical strides (M*K, K, 1) of our row-major allocations.
    #      This becomes the STATIC field: baked into code via __new_from_mlir_values__,
    #      never passed through MLIR SSA.
    buffer_a = BufferWithLayout(
        make_ptr(ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
        (2, 1, 0),
    )
    buffer_b = BufferWithLayout(
        make_ptr(ab_dtype, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
        (2, 1, 0),
    )
    buffer_c = BufferWithLayout(
        make_ptr(c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
        (2, 1, 0),
    )

    # Pass BufferWithLayout objects to the JIT wrapper.
    # mnkl is passed as a plain Python tuple — it becomes cutlass.Constexpr inside the
    # JIT function, meaning it is a compile-time constant baked into the generated code.
    # No stride arguments are needed: the layout convention is encoded in stride_order.
    tensor_op_gemm_wrapper(
        buffer_a,
        buffer_b,
        buffer_c,
        mnkl,  # pass shape as static value (Constexpr)
        # no stride passing — stride convention is encoded in stride_order
        cutlass.Float32,
        (2, 2, 1),
    )
    torch.cuda.synchronize()

    ref = torch.einsum("lmk,lnk->lmn", a, b)
    torch.testing.assert_close(c, ref, atol=1e-05, rtol=1e-05)
    print(f"\n[DSL INFO] Results verified successfully!")
    print(f"First few elements of result: \n{c[:3, :3, :3]}")


if __name__ == "__main__":
    run_tensor_op_gemm_wrapper((512, 256, 128, 16))
