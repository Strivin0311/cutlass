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
from typing import Tuple, Type
import math
import cuda.bindings.driver as cuda

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
from cutlass import const_expr
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils

"""
A high-performance batched dense GEMM (C = A * B) example for the NVIDIA Hopper architecture
using CuTe DSL.
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Hopper's WGMMA for matrix multiply-accumulate (MMA) operations
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Supports multi-stage pipeline to overlap computation and memory access

This GEMM works as follows:
1. Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. Perform matrix multiply-accumulate (MMA) operations using WGMMA instruction.
3. Store results from registers (RMEM) to shared memory (SMEM), then to global memory (GMEM) with TMA operations.

Hopper WGMMA instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Perform MMA operation and store the result in Accumulator(register)

To run this example:

.. code-block:: bash

    python examples/hopper/dense_gemm.py                                   \
      --mnkl 8192,8192,8192,1 --tile_shape_mn 128,256                      \
      --cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16           \
      --c_dtype Float16 --acc_dtype Float32                                \
      --a_major k --b_major k --c_major n

The above example command compute batched gemm with M=8192, N=8192, K=8192,
batch_count=1. The Hopper WGMMA tile shape is 128x256x64 and the cluster shape
is (1,1). The input, mma accumulator and output data type are set as fp16, fp32
and fp16, respectively.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/hopper/dense_gemm.py                               \
      --mnkl 8192,8192,8192,1 --tile_shape_mn 128,256                      \
      --cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16           \
      --c_dtype Float16 --acc_dtype Float32                                \
      --a_major k --b_major k --c_major n

Constraints:
* Supported input data types: fp16, fp8 (e4m3fn, e5m2)
* For fp16 types, A and B must have the same data type
* For fp8 types, A and B can have different types (e4m3fn or e5m2) but both must be 8-bit
* Fp8 types only support k-major layout
* CTA tile shape M must be 64/128
* CTA tile shape N must be 64/128/256
* Cluster shape M/N must be positive and power of 2, total cluster size <= 4
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 8, 16 for Float16, and Float8, respectively.
"""


DEBUG_MODE = int(os.environ.get("DEBUG_MODE", "0")) == 1


# /////////////////////////////////////////////////////////////////////////////
#  Helpers to parse args
# /////////////////////////////////////////////////////////////////////////////
def parse_comma_separated_ints(s: str):
    try:
        return tuple([int(x.strip()) for x in s.split(",")])
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers."
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example of MxNxKxL GEMM on Hopper.")

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(4096, 4096, 4096, 1),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--tile_shape_mn",
        type=parse_comma_separated_ints,
        choices=[(128, 128), (128, 256), (128, 64), (64, 64)],
        default=(128, 128),
        help="Cta tile shape (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        choices=[(1, 1), (2, 1), (1, 2), (2, 2)],
        default=(1, 1),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument(
        "--a_dtype",
        type=cutlass.dtype,
        default=cutlass.Float16,
    )
    parser.add_argument(
        "--b_dtype",
        type=cutlass.dtype,
        default=cutlass.Float16,
    )
    parser.add_argument(
        "--c_dtype",
        type=cutlass.dtype,
        default=cutlass.Float16,
    )
    parser.add_argument(
        "--acc_dtype",
        type=cutlass.dtype,
        default=cutlass.Float32,
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

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")
    if len(args.tile_shape_mn) != 2:
        parser.error("--tile_shape_mn must contain exactly 2 values")
    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    return args


# /////////////////////////////////////////////////////////////////////////////
#  Host setup and device kernel launch
# /////////////////////////////////////////////////////////////////////////////


class HopperWgmmaGemmKernel:
    """
    This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Hopper GPUs.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param tile_shape_mn: Shape of the CTA tile (M,N)
    :type tile_shape_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: Data type requirements:
        - For 16-bit types: A and B must have the same data type
        - For 8-bit types: A and B can have different types (Float8E4M3FN/Float8E5M2) as long as both are 8-bit
        - Float8 types only support k-major layout

    :note: Supported data types:
        - Float16
        - Float8E4M3FN/Float8E5M2

    :note: Supported accumulation types:
        - Float32 (for all floating point inputs)

    :note: Constraints:
        - CTA tile M must be 64/128
        - CTA tile N must be 64/128/256
        - CTA tile K must be 64
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 4

    Example:
        >>> gemm = HopperWgmmaGemmKernel(
        ...     acc_dtype=cutlass.Float32,
        ...     tile_shape_mn=(128, 256),
        ...     cluster_shape_mn=(1, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, c_tensor, stream)
    """

    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        tile_shape_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        debug_print: bool = False,
    ):
        """
        Initializes the configuration for a Hopper dense GEMM kernel.

        This configuration includes data types for operands, tile shape, cluster configuration,
        and thread layout.

        :param acc_dtype: Data type for accumulation during computation
        :type acc_dtype: type[cutlass.Numeric]
        :param tile_shape_mn: Shape of the CTA tile (M,N)
        :type tile_shape_mn: Tuple[int, int]
        :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
        :type cluster_shape_mn: Tuple[int, int]
        """

        self.acc_dtype = acc_dtype

        self.cluster_shape_mn = cluster_shape_mn
        self.mma_inst_shape_mn = None
        # K dimension is deferred in _setup_attributes
        self.tile_shape_mnk = (*tile_shape_mn, 1) # determine tileK automatically later
        # For large tile size, using two warp groups is preferred because using only one warp
        # group may result in register spill
        self.atom_layout_mnk = (
            (2, 1, 1)
            if self.tile_shape_mnk[0] > 64 and self.tile_shape_mnk[1] > 128
            else (1, 1, 1)
        )
        self.num_mcast_ctas_a = None
        self.num_mcast_ctas_b = None
        self.is_a_mcast = False
        self.is_b_mcast = False
        self.tiled_mma = None

        self.occupancy = 1
        self.mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = self.mma_warp_groups * self.num_threads_per_warp_group
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90") # get the dynamic shared memory capacity for sm_90, which is (228-1) KB

        self.debug_print = debug_print

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024 # TMA smem alignment

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
        """

        # check the cta tile shape
        if self.tile_shape_mnk[0] not in [64, 128]:
            raise ValueError("CTA tile shape M must be 64/128")
        if self.tile_shape_mnk[1] not in [64, 128, 256]:
            raise ValueError("CTA tile shape N must be 64/128/256")

        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            a_dtype=self.a_dtype,
            b_dtype=self.b_dtype,
            a_leading_mode=self.a_layout.sm90_mma_major_mode(), # K if row-major, MN if col-major
            b_leading_mode=self.b_layout.sm90_mma_major_mode(), # K if row-major, MN if col-major
            acc_dtype=self.acc_dtype,
            atom_layout_mnk=self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
            a_source=cute.nvgpu.warpgroup.OperandSource.SMEM,
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.tile_shape_mnk = (
            self.tile_shape_mnk[0],
            self.tile_shape_mnk[1],
            mma_inst_shape_k * mma_inst_tile_k, # 16 x 4 = 64
        )

        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        self.num_mcast_ctas_a = self.cluster_shape_mn[1]
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        is_cooperative = self.atom_layout_mnk == (2, 1, 1)
        self.epi_tile = self._sm90_compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative
        )

        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
            debug_print=self.debug_print,
        )
        
        self.epi_tiled_copy_r2s = self._make_epi_tiled_copy(
            tiled_mma=self.tiled_mma,
            c_layout=self.c_layout,
            c_dtype=self.c_dtype,
            acc_dtype=self.acc_dtype,
            debug_print=self.debug_print,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stream: cuda.CUstream,
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
        """

        # setup static attributes before smem/grid/tma computation
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if const_expr(
            self.a_dtype.width == 16 and self.a_dtype != self.b_dtype
        ):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if const_expr(self.a_dtype.width != self.b_dtype.width):
            raise TypeError(
                f"Type width mismatch: {self.a_dtype.width} != {self.b_dtype.width}"
            )
        if const_expr(self.a_dtype.width != 16 and self.a_dtype.width != 8):
            raise TypeError(f"a_dtype should be float16 or float8")

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]), # (tileM, tileK)
            self.cluster_shape_mn[1],
            debug_print=self.debug_print,
            title="Make TMA atom/tensor for A",
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]), # (tileN, tileK)
            self.cluster_shape_mn[0],
            debug_print=self.debug_print,
            title="Make TMA atom/tensor for B",
        )

        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile, # (epiM, epiN)
            debug_print=self.debug_print,
            title="Make TMA atom/tensor for C",
        )

        grid = self._compute_grid(c, self.tile_shape_mnk, self.cluster_shape_mn)

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2 # x2 since each stage will have a pair of empty/full mbars
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        
        if const_expr(self.debug_print):
            print()
            print(f"{self.tile_shape_mnk=}, {self.atom_layout_mnk=}, {self.cluster_shape_mn=}, {self.cta_layout_mnk=}")
            print(f"{self.num_mcast_ctas_a=}, {self.num_mcast_ctas_b=}, {self.is_a_mcast=}, {self.is_b_mcast=}")
            print(f"{self.mma_warp_groups=}, {self.num_threads_per_warp_group=}, {self.threads_per_cta=}")
            print(f"{self.ab_stage=}, {self.epi_stage=}, {self.epi_tile=}, {self.smem_capacity=} ({self.smem_capacity / 1024:.1f} KB), {self.occupancy=}, {self.buffer_align_bytes=}")
            print()
        
            print()
            print(f"{self.mma_warp_groups=}, {self.num_threads_per_warp_group=}, {self.threads_per_cta=}")
            print(f"{self.a_layout=}, {self.a_layout.sm90_mma_major_mode()=} | {self.b_layout=}, {self.b_layout.sm90_mma_major_mode()=} | {self.c_layout=}")
            print(f"{self.a_dtype=}, {self.b_dtype=}, {self.c_dtype=}, {self.acc_dtype=}")
            print()
        
            print()
            print("self.tiled_mma: ", self.tiled_mma, f"shape_mnk: {self.tiled_mma.shape_mnk}")
            print()
            
            cute.printf("tma_tensor_a: {}", tma_tensor_a)
            cute.printf("tma_tensor_b: {}", tma_tensor_b)
            cute.printf("tma_tensor_c: {}", tma_tensor_c)
        
            cute.printf("grid: {}", grid)

        # Launch the kernel synchronously
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            self.tiled_mma,
            self.epi_tiled_copy_r2s,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )
        return

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        epi_tiled_copy_r2s: cute.TiledCopy,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
    ):
        """
        GPU device kernel performing the batched GEMM computation.

        :param tma_atom_a: TMA copy atom for A tensor
        :type tma_atom_a: cute.CopyAtom
        :param mA_mkl: Input tensor A
        :type mA_mkl: cute.Tensor
        :param tma_atom_b: TMA copy atom for B tensor
        :type tma_atom_b: cute.CopyAtom
        :param mB_nkl: Input tensor B
        :type mB_nkl: cute.Tensor
        :param tma_atom_c: TMA copy atom for C tensor
        :type tma_atom_c: cute.CopyAtom
        :param mC_mnl: Output tensor C
        :type mC_mnl: cute.Tensor
        :param tiled_mma: Tiled MMA object
        :type tiled_mma: cute.TiledMma
        :param cta_layout_mnk: CTA layout
        :type cta_layout_mnk: cute.Layout
        :param a_smem_layout_staged: Shared memory layout for A
        :type a_smem_layout_staged: cute.ComposedLayout
        :param b_smem_layout_staged: Shared memory layout for B
        :type b_smem_layout_staged: cute.ComposedLayout
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        """
        
        # ///////////////////////////////////////////////////////////////////////////////
        #  Get cta/warp/thread idx
        # ///////////////////////////////////////////////////////////////////////////////
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        is_thread0 = tidx == 129 and bidx == 15 and bidy == 15

        cidx, cidy, cidxz = cute.arch.cluster_idx()
        cdimx, cdimy, cdimz = cute.arch.cluster_dim()
        cluster_id = cidx + cdimx * cidy

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.mma_warp_groups, stride=self.num_threads_per_warp_group
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch Tma desc
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            
        # /////////////////////////////////////////////////////////////////////////////
        #  Make CTA layout
        # /////////////////////////////////////////////////////////////////////////////

        # CTA Swizzle to promote L2 data reuse
        group_size_m = 8
        s_shape = (
            (group_size_m, cdimx // group_size_m),
            cdimy,
        )
        s_stride = ((1, cdimy * group_size_m), group_size_m)
        s_layout = cute.make_layout(s_shape, stride=s_stride)
        num_reg_cids = cute.size(s_shape)
        cid_m, cid_n = s_layout.get_flat_coord(cluster_id % num_reg_cids)
        
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("s_layout: {}, num_reg_cids: {}", s_layout, num_reg_cids)
                cute.printf("cid_m: {}, cid_n: {}, cluster_id: {}, bdim: ({}, {}, {}), cdim: ({}, {}, {}), bidx: ({}, {}, {}), cidx: ({}, {}, {})", cid_m, cid_n, cluster_id, *cute.arch.block_dim(), cdimx, cdimy, cdimz, bidx, bidy, bidz, cidx, cidy, cidxz)

        # Deal with the tail part
        if cluster_id >= num_reg_cids:
            tail_size_m = cdimx % group_size_m
            tail_layout = cute.make_layout(
                (tail_size_m, cdimy), stride=(1, tail_size_m)
            )
            tail_cid = cluster_id - num_reg_cids
            tail_cid_m, tail_cid_n = tail_layout.get_flat_coord(tail_cid)
            cid_m = cute.size(s_shape, mode=[0]) + tail_cid_m
            cid_n = tail_cid_n

        # Get the pid from cluster id
        bidx_in_cluster = cute.arch.block_in_cluster_idx()
        pid_m = cid_m * self.cluster_shape_mn[0] + bidx_in_cluster[0]
        pid_n = cid_n * self.cluster_shape_mn[1] + bidx_in_cluster[1]

        tile_coord_mnkl = (pid_m, pid_n, None, bidz)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)
        
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("block_in_cluster_idx: {}, block_idx_in_cluster: {}, ", bidx_in_cluster, cute.arch.block_idx_in_cluster())
                cute.printf("pid_m: {}, pid_n: {}, tile_coord_mnkl: {}, cluster_coord_mnk: {}", pid_m, pid_n, tile_coord_mnkl, cluster_coord_mnk)
                cute.printf("cta_layout_mnk: {}, cta_rank_in_cluster: {}, cluster_coord_mnk: {}", cta_layout_mnk, cta_rank_in_cluster, cluster_coord_mnk)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get mcast mask
        # ///////////////////////////////////////////////////////////////////////////////
        a_mcast_mask = cute.make_layout_image_mask(
            # multicast the same A along the given mode1 (i.e. N dimension) of the cluster
            # e.g. if cluster_shape_mn is (2,1), then two CTAs in the same cluster will have different A slices
            # so the mcast mask is 01 or 10 in binary (but the better way is to use 00 with CopyBulkTensorTileG2SOp, i.e. no multicast at all)
            # while if cluster_shape_mn is (1,2), then two CTAs in the same cluster will have the same A slice,
            # so the mcast mask is 11 in binary with CopyBulkTensorTileG2SMulticastOp
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            # multicast the same B along the given mode0 (i.e. M dimension) of the cluster
            # e.g. if cluster_shape_mn is (2,1), then two CTAs in the same cluster will have the same B slice,
            # so the mcast mask is 11 in binary with CopyBulkTensorTileG2SMulticastOp,
            # while if cluster_shape_mn is (1,2), then two CTAs in the same cluster will have different B slices,
            # so the mcast mask is 01 or 10 in binary (but the better way is to use 00 with CopyBulkTensorTileG2SOp, i.e. no multicast at all)
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )

        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
        
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("a_mcast_mask: {}, b_mcast_mask: {}", a_mcast_mask, b_mcast_mask)

        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc and init AB full/empty + ACC full mbar (pipeline)
        # /////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        
        # Get the tma copy bytes for each stage,
        # note that we need to wait both A/B and ready, thus the tma copy bytes is the sum of A and B copy size, 
        # which is determined by the smem layout per stage and data type
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        # Get mbar arrays ptr
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        # Define producer group for mainloop pipeline
        # only warp0 will be the producer to load the data, 
        # thus size=1 (the lane0 in warp0 will participate the pipeline and accquire/issue the tma load)
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, size=1
        )
        
        # Define consumer group for mainloop pipeline
        # all warps in the CTA will consume the data produced by the producer group, 
        # thus size=num_warps (the lane0 in each warp will participate the pipeline and wait/release the tma data)
        # note that when cluster feature is enabled, we need to `x mcast_size` of multicast CTAs,
        # which equals to the sum of number of multicast CTAs for A(row) and B(col), minus 1, 
        # since the current CTA itself counts twice in A(row) and B(col)
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        num_warps = self.threads_per_cta // 32
        consumer_arrive_cnt = mcast_size * num_warps
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, size=consumer_arrive_cnt
        )

        # since cta_layout_mnk = (2,1,1), thus cta_layout_vmnk is (1,2,1,1)
        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        
        # Make the mainloop pipeline for producer and consumer synchronization of TMA smem buffer using mbarriers in the `create` function
        #   1. it will creat a pair of mbarriers for empty and full state for each stage,
        #       thus the total number of mbarriers needed is `ab_stage * 2`, and then call `mbarrier_init_fence`
        #   2. and during each stage, producer will wait the empty mbar, arrive the full mbar and expect `tma_copy_bytes` of `tx_count`, 
        #       and then issue the TMA load to update the `tx_count` on full mbar;
        #   3. while the consumer will wait the full mbar, i.e. the TMA load completion, then consume the data,
        #       and arrive the empty mbar and expect 0 `tx_count`, just to notify the producer that the stage is consumed and empty again for the next iteration
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
        )
        
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("a_smem_layout: {}, b_smem_layout: {}, tma_copy_bytes: {}", a_smem_layout, b_smem_layout, tma_copy_bytes)
                cute.printf("mcast_size: {}, consumer_arrive_cnt: {}, cta_layout_vmnk: {}", mcast_size, consumer_arrive_cnt, cta_layout_vmnk)

        #  Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Generate smem tensor A/B/C
        # ///////////////////////////////////////////////////////////////////////////////
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        
        # NOTE: smem of C reuses the one of A, since during the epilogue, A/B smem is no longer needed, 
        # thus we can reuse A's smem for C to save smem usage
        sC_ptr = cute.recast_ptr(
            sA.iterator, swizzle_=epi_smem_layout_staged.inner, dtype=self.c_dtype
        )
        sC = cute.make_tensor(sC_ptr, epi_smem_layout_staged.outer)
        
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("sA:")
                cute.print_tensor(sA)
                cute.printf("sB:")
                cute.print_tensor(sB)
                cute.printf("sC:")
                cute.print_tensor(sC)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Local_tile partition global tensors
        # ///////////////////////////////////////////////////////////////////////////////
        
        # (bM, bK, RestK)
        gA_mkl = cute.local_tile(
            mA_mkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(1, None, 1)
        )
        # (bN, bK, RestK)
        gB_nkl = cute.local_tile(
            mB_nkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(None, 1, 1)
        )
        # (bM, bN)
        gC_mnl = cute.local_tile(
            mC_mnl, self.tile_shape_mnk, tile_coord_mnkl, proj=(1, 1, None)
        )
        
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("mA_mkl:")
                cute.print_tensor(mA_mkl)
                cute.printf("mB_nkl:")
                cute.print_tensor(mB_nkl)
                cute.printf("mC_mnl:")
                cute.print_tensor(mC_mnl)
                cute.printf("")
                cute.printf("gA_mkl:")
                cute.print_tensor(gA_mkl)
                cute.printf("gB_nkl:")
                cute.print_tensor(gB_nkl)
                cute.printf("gC_mnl:")
                cute.print_tensor(gC_mnl)

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition shared tensor for TMA load A/B
        # //////////////////////////////////////////////////////////////////////////////
        
        #  TMA load A partition_S/D
        
        # The cta layout is the tma multicast layout
        # since cluster_shape_mn is (2,1,1), then two CTAs in the same cluster will have different A slices w/o multicast,
        # so the cta layout for A is dummy `(1):(0)` and cta coordinate is dummy `0`
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        
        # Since sA layout: ( (8,16), (64,1), stages=(1,4) ) while tma_atom_a is 8192
        # so we need to group the first two modes of sA together (8x16x64=8192) for one tma atom,
        # so sA_for_tma_partition layout: ( ((8,16),(64,1)), stages=(1,4) )
        sA_for_tma_partition = cute.group_modes(sA, 0, 2)
        # gA_mkl layout: ( 128, 64, rest_k=16 ), and we need to group the first two modes together (128x64=8192) for one tma atom,
        # so gA_for_tma_partition layout: ( (128,64), rest_k=16 )
        gA_for_tma_partition = cute.group_modes(gA_mkl, 0, 2)
        
        # Then, tma partition will further partition the given shared/global tensors into tiles 
        # according to the given tma atom, the cta layout with cta coordinate
        # and return the corr. tiled shared/global tensors,
        # i.e. tAsA with layout: ( (8192,1), stages=(1,4) ) 
        # and tAgA_mkl with layout: ( ((64,128),1), rest_k=16 )
        tAsA, tAgA_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            sA_for_tma_partition,
            gA_for_tma_partition,
        )

        # TMA load B partition_S/D
        
        # While the cta layout for B is `(2):(0)` since two CTAs in the same cluster will have the same B slice with multicast,
        # and the cta coordinate is either `0` or `1` indicating the multicast idx of the current CTA,
        # which equals to the cluster coordinate along the multicast mode
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        
        # Similar to A, since tma_atom_b is 16384, we need to group the first two modes of sB together (8x32x64=16384) for one tma atom,
        # so 
        #   sB layout: ( (8,32), (64,1), stages=(1,4) ) while 
        #   sB_for_tma_partition layout: ( ((8,32),(64,1)), stages=(1,4) )
        #   gB_nkl layout: ( 256, 64, rest_k=16 ), and we need to group the first two modes together (256x64=16384) for one tma atom,
        #   gB_for_tma_partition layout: ( (256,64), rest_k=16 )
        sB_for_tma_partition = cute.group_modes(sB, 0, 2)
        gB_for_tma_partition = cute.group_modes(gB_nkl, 0, 2)
        
        # And tma partition will return 
        # tBsB with layout: ( (16384,1), stages=(1,4) ) 
        # and tBgB_nkl with layout: ( ((64,256),1), rest_k=16 )
        tBsB, tBgB_nkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            sB_for_tma_partition,
            gB_for_tma_partition,
        )
        
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("a_cta_layout: {}, a_cta_crd: {}", a_cta_layout, a_cta_crd)
                cute.printf("b_cta_layout: {}, b_cta_crd: {}", b_cta_layout, b_cta_crd)
                cute.printf("sA_for_tma_partition:")
                cute.print_tensor(sA_for_tma_partition)
                cute.printf("gA_for_tma_partition:")
                cute.print_tensor(gA_for_tma_partition)
                cute.printf("sB_for_tma_partition:")
                cute.print_tensor(sB_for_tma_partition)
                cute.printf("gB_for_tma_partition:")
                cute.print_tensor(gB_for_tma_partition)
                cute.printf("tAsA:")
                cute.print_tensor(tAsA)
                cute.printf("tAgA_mkl:")
                cute.print_tensor(tAgA_mkl)
                
                # cute.printf("tBsB:")
                # cute.print_tensor(tBsB) # FIXME: due to multicast, print_tensor will incur with illegal memory access
                cute.printf("tBsB: {}", tBsB)
                
                cute.printf("tBgB_nkl:")
                cute.print_tensor(tBgB_nkl)
            
        # //////////////////////////////////////////////////////////////////////////////
        #  Partition shared/global tensor for TiledMMA A/B/C
        # //////////////////////////////////////////////////////////////////////////////
        
        # since each thread in the same warp group shares the same thread slice of tiled_mma, 
        # we can directly get the thread slice with either actual tidx or warp_group_idx
        # thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))
        thr_mma = tiled_mma.get_slice(tidx)
        
        # since sA layout: ( (8,16), (64,1), stages=(1,4) ) and wgmma has limitations on m-dim(=64) and k-dim(=16)
        # so we need to partition sA into tCsA with layout: ( mk=(64,16), 1, kloop=4, stages=(1,4) ) 
        # note that m=64 but M=8x16=128, so the M-dim will be parallelized along 2 WGs (atom_layout_mnk=(2,1,1))
        tCsA = thr_mma.partition_A(sA)
        
        # since sB layout: ( (8,32), (64,1), stages=(1,4) ) and wgmma has limitations on n-dim(=N) and k-dim(=16)
        # so we need to partition sB into tCsB with layout: ( nk=(256,16), 1, kloop=4, stages=(1,4) ) 
        # note that n=N=256, so it only needs to be handled by current WG (atom_layout_mnk=(2,1,1))
        tCsB = thr_mma.partition_B(sB)

        # gC_mnl with layout: (128,256) is the tile of C on global memory this block will compute and store,
        # and since we have 2 WGs with 256 threads, then each thread needs to hold 128 elements of C,
        # so tCgC with layout: ((2,2,32),1,1) is the sub-tile of gC_mnl that this thread will hold,
        tCgC = thr_mma.partition_C(gC_mnl)
        
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("warp_group_idx: {}, warp_group_thread_layout: {}, slice: {}", warp_group_idx, warp_group_thread_layout, warp_group_thread_layout(warp_group_idx))
                # thr_mma is not printable, but we want to print its attributes
                cute.printf("thr_mma.tv_layout_A: {}, thr_mma.tv_layout_A_tiled: {}", thr_mma.tv_layout_A, thr_mma.tv_layout_A_tiled)
                cute.printf("thr_mma.tv_layout_B: {}, thr_mma.tv_layout_B_tiled: {}", thr_mma.tv_layout_B, thr_mma.tv_layout_B_tiled)
                cute.printf("thr_mma.tv_layout_C: {}, thr_mma.tv_layout_C_tiled: {}", thr_mma.tv_layout_C, thr_mma.tv_layout_C_tiled)
                cute.printf("tCsA:")
                cute.print_tensor(tCsA)
                cute.printf("tCsB:")
                cute.print_tensor(tCsB)
                cute.printf("tCgC:")
                cute.print_tensor(tCgC)

        # //////////////////////////////////////////////////////////////////////////////
        #  Make fragments for TiledMMA A/B/C
        # //////////////////////////////////////////////////////////////////////////////
        
        # since tCsA layout is ( mk=(64,16), 1, kloop=4, stages=(1,4) )
        # so tCrA layout will be (1, 1, kloop=4, stages=(1,4) ),
        # wile the first two modes are dummy with stride=0 because tCsA in smem will be handled by current WG as a whole
        tCrA = tiled_mma.make_fragment_A(tCsA)
        
        # since tCsB layout is ( nk=(256,16), 1, kloop=4, stages=(1,4) )
        # so tCrB layout will be (1, 1, kloop=4, stages=(1,4) ),
        # while the first two modes are dummy with stride=0 because tCsB in smem will be handled by current WG as a whole
        tCrB = tiled_mma.make_fragment_B(tCsB)
        
        # since tCgC layout is ((2,2,32),1,1), then tCrC (accumulators) layout will also be ( (2,2,32), 1, 1 )
        tCrC = cute.make_fragment(tCgC.shape, self.acc_dtype)
        
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                # cute.printf("tCrA:")
                # cute.print_tensor(tCrA)
                cute.printf("tCrA.layout: {}", tCrA.layout) # tCrA is not printable, but we want to print its layout
                
                # cute.printf("tCrB:")
                # cute.print_tensor(tCrB)
                cute.printf("tCrB.layout: {}", tCrB.layout) # tCrB is not printable, but we want to print its layout
                
                cute.printf("tCrC:")
                cute.print_tensor(tCrC)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Cluster wait
        # ///////////////////////////////////////////////////////////////////////////////
        
        # TODO(REVIEW): do we need this cluster wait, since `pipeline.PipelineTmaAsync.create` 
        # has already issued `cluster_arrive_relaxed` and `cluster_wait` inside ?
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()
        
        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch
        # /////////////////////////////////////////////////////////////////////////////
        prologue_mmas = 1
        k_tile_cnt = cute.size(gA_mkl, mode=[2]) # rest_k dim as the k_tile cnt (16)
        num_k_blocks = cute.size(tCrA, mode=[2]) # kloop dim as the k_block cnt (4)
        prefetch_k_tile_cnt = cutlass.max(cutlass.min(self.ab_stage, k_tile_cnt), 0) # min(rest_k, stages) = stages = 4
        
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("prologue_mmas: {}, prefetch_k_tile_cnt: {}, num_k_blocks: {}", prologue_mmas, prefetch_k_tile_cnt, num_k_blocks)

        # Init producer state for mainloop pipeline
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, stages=self.ab_stage
        )
        
        # Prefetch a full stage of A,B using TMA
        if warp_idx == 0: # producer
            # /////////////////////////////////////////////////////////////////////////////
            # Prefetch TMA load
            # /////////////////////////////////////////////////////////////////////////////
            for prefetch_idx in cutlass.range(prefetch_k_tile_cnt, unroll=1):
                # /////////////////////////////////////////////////////////////////////////////
                #  Wait for A/B buffers to be empty before loading into them
                #  Also sets the transaction barrier for the A/B buffers
                # /////////////////////////////////////////////////////////////////////////////
                
                # producer acquire will wait the empty mbar using `mbarrier.try_wait.parity.shared`
                # and arrive the full mbar using `mbarrier_arrive_and_expect_tx`, where the tx_count = tma_copy_bytes
                mainloop_pipeline.producer_acquire(mainloop_producer_state)
                
                # /////////////////////////////////////////////////////////////////////////////
                #  Slice to global/shared memref to current k_tile
                # /////////////////////////////////////////////////////////////////////////////
                
                # tAgA_mkl layout: ( ((64,128),1), rest_k=16 ) 
                # and we need to slice the rest_k dim according to the current k_tile indicated by mainloop_producer_state.count
                # resulting in tAgA_k with layout: ( ((64,128),1), 1 )
                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                
                # tAsA layout: ( (8192,1), stages=(1,4) ) 
                # and we need to slice the stage dim according to the current pipeline stage indicated by mainloop_producer_state.index
                # resulting in tAsA_pipe with layout: ( (8192,1), 1 )
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                # tBgB_nkl layout: ( ((64,256),1), rest_k=16 )
                # and we need to slice the rest_k dim according to the current k_tile indicated by mainloop_producer_state.count
                # resulting in tBgB_k with layout: ( ((64,256),1), 1 )
                tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                
                # tBsB layout: ( (16384,1), stages=(1,4) )
                # and we need to slice the stage dim according to the current pipeline stage indicated by mainloop_producer_state.index
                # resulting in tBsB_pipe with layout: ( (16384,1), 1 )
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                # /////////////////////////////////////////////////////////////////////////////
                #  TMA load A/B
                # /////////////////////////////////////////////////////////////////////////////
                cute.copy(
                    atom=tma_atom_a,
                    src=tAgA_k,
                    dst=tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=a_mcast_mask, # 0(0b00), since A won't be multicast for cluster shape (2,1,1)
                )
                cute.copy(
                    atom=tma_atom_b,
                    src=tBgB_k,
                    dst=tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=b_mcast_mask, # 3(0b11), since B will be multicast along m mode for cluster shape (2,1,1)
                )
                
                # NOTE: Mainloop pipeline's producer commit is a NOP
                # since TMA instruction itself updates the transaction count in full mbar
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

        # /////////////////////////////////////////////////////////////////////////////
        #  Prologue MMAs
        # /////////////////////////////////////////////////////////////////////////////

        # NOTE: we need to use two separate pipeline states:
        #   1. consumer_read_state to issue consumer_wait 
        #       to wait the current stage of full mbar before consuming the current data
        #       as the tail ptr of the pipeline queue
        #   2. consumer_release_state to issue consumer_release 
        #       to arrive the empty mbar after consuming the previous data
        #       as the head ptr of the pipeline queue
        mainloop_consumer_read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        mainloop_consumer_release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        peek_ab_full_status = cutlass.Boolean(1)
        if mainloop_consumer_read_state.count < k_tile_cnt:
            # consumer will peek if the full mbar is arrived without blocking, using `mbarrier_try_wait`
            # and return the token in peek_ab_full_status, where:
            #   1. token == 1, then arrived and no need to wait
            #   2. token == 0, then not arrived and need to wait
            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                mainloop_consumer_read_state
            )

        # NOTE: we only allocate tCrC for the accumulator of WGMMA, but we don't zeros it,
        # so the first WGMMA needs to set ACCUMULATE to False to avoid using the uninitialized value in tCrC, 
        # and then set it back to True for the rest WGMMA in the mainloop
        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        for k_tile in cutlass.range_constexpr(prologue_mmas):
            # Wait for the current stage of full mbar, i.e. A/B smem buffer to be ready,  using `mbarrier_wait`
            # since we try wait earlier, so if peek_ab_full_status is true (token == 1),
            # then consumer won't need to check the full mbar here and can directly go to consume the data;
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_read_state, peek_ab_full_status
            )

            # NOTE: case1: Before the first WGMMA, we need call `wgmma.fence.sync.aligned` to 
            #   1. ensure the memory operations by generic proxy to registers of the accumulator (like zeros), and the operand A if RS, are visible to wgmma proxy.
            #   2. hand over the ownership of the accumulator registers and operand A if RS from generic proxy to wgmma proxy (so even no zeros, we still need ti call this fence).
            # 
            # case2: The same as to when we modify the registers between two WGMMA calls in the mainloop, 
            # we also need to call `wgmma.fence.sync.aligned` to ensure the visibility of the modified registers to wgmma proxy for the next WGMMA call.
            # 
            # case3: Otherwise, we don't have to call fence between each iteration of WGMMA in the mainloop, since no memory operations by generic proxy in between.
            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (
                    None,
                    None,
                    k_block_idx,
                    mainloop_consumer_read_state.index,
                )
                tCrA_1phase = tCrA[k_block_coord]
                tCrB_1phase = tCrB[k_block_coord]

                # D = A*B + C, where A and B are from smem and C is from register, 
                # and the result is stored back to register D (which can be alias to C)
                cute.gemm(
                    atom=tiled_mma,
                    d=tCrC,
                    a=tCrA_1phase,
                    b=tCrB_1phase,
                    c=tCrC,
                )
                
                # After the first WGMMA, we set ACCUMULATE to True 
                # for the rest WGMMA in the mainloop to accumulate on the result in tCrC
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            # Commit the num_k_blocks WGMMA calls in the current k_tile
            # to let wgmma engine runs them as a batch
            cute.nvgpu.warpgroup.commit_group()
            
            # Advance to next block of global A/B and next stage of smem A/B buffer
            mainloop_consumer_read_state.advance()
            
            # Peek next stage of A/B full mbar
            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state
                )

        # /////////////////////////////////////////////////////////////////////////////
        #  MAINLOOP
        # /////////////////////////////////////////////////////////////////////////////
        for k_tile in cutlass.range(prologue_mmas, k_tile_cnt, 1, unroll=1):
            # /////////////////////////////////////////////////////////////////////////////
            #  Wait for TMA copies to complete
            # /////////////////////////////////////////////////////////////////////////////
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_read_state, peek_ab_full_status
            )
            
            # /////////////////////////////////////////////////////////////////////////////
            #  WGMMA
            # /////////////////////////////////////////////////////////////////////////////
            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (
                    None,
                    None,
                    k_block_idx,
                    mainloop_consumer_read_state.index,
                )
                tCrA_1phase = tCrA[k_block_coord]
                tCrB_1phase = tCrB[k_block_coord]

                cute.gemm(
                    atom=tiled_mma,
                    d=tCrC,
                    a=tCrA_1phase,
                    b=tCrB_1phase,
                    c=tCrC,
                )

            cute.nvgpu.warpgroup.commit_group()
            
            # Wait on the wgmma barrier 
            # to allow at most `prologue_mmas` wgmmas unfinished
            # i.e. the queue window size is `prologue_mmas` so the previous wgmmas out of the window must complete
            # and we already issued `prologue_mmas + 1` wgmmas (including current one) at this point,
            # so this wait equals to waiting for the very earliest issued wgmma (i.e. the current k_tile - prologue_mmas) to complete 
            cute.nvgpu.warpgroup.wait_group(prologue_mmas)

            # When the current k_tile's WGMMA is done, then we can release the current stage of A/B buffer 
            # by arriving the empty mbar for the current stage, using `mbarrier_arrive_and_expect_tx` where tx_count = 0
            # NOTE: this is a huge difference with the `mbarrier_arrive_and_expect_tx` used in producer's `acquire`, since:
            #   1. producer's `acquire` will arrive the full mbar with tx_count = tma_copy_bytes after issuing the TMA load, 
            #       to not only notify the consumer that "I am arrived", but also let TMA engine update the mbar's tx count later,
            #       so that consumer's `wait` will actually wait for the TMA load to complete by checking the tx count in the mbar;
            #   2. while consumer's `release` here will arrive the empty mbar with tx_count = 0 after consuming the data,
            #       just to notify the producer that "I am arrived and have consumed the data", 
            #       so that producer's `acquire` only needs to wait for the consumer to arrive before issuing the next TMA load
            mainloop_pipeline.consumer_release(mainloop_consumer_release_state)

            # Advance read and release states to next stage 
            # (i.e. moving both the head/tail ptrs of the pipeline queue)
            mainloop_consumer_read_state.advance()
            mainloop_consumer_release_state.advance()

            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state
                )
                
            # /////////////////////////////////////////////////////////////////////////////
            #  TMA load
            # /////////////////////////////////////////////////////////////////////////////
            if warp_idx == 0 and mainloop_producer_state.count < k_tile_cnt: # producer
                # /////////////////////////////////////////////////////////////////////////////
                #  Wait for A/B buffers to be empty before loading into them
                #  Also sets the transaction barrier for the A/B buffers
                # /////////////////////////////////////////////////////////////////////////////
                mainloop_pipeline.producer_acquire(mainloop_producer_state)

                # /////////////////////////////////////////////////////////////////////////////
                #  Slice to global/shared memref to current k_tile
                # /////////////////////////////////////////////////////////////////////////////
                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                # /////////////////////////////////////////////////////////////////////////////
                #  TMA load A/B
                # /////////////////////////////////////////////////////////////////////////////
                cute.copy(
                    tma_atom_a,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_k,
                    tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=b_mcast_mask,
                )
                
                # Mainloop pipeline's producer commit is a NOP
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

        # /////////////////////////////////////////////////////////////////////////////
        #  EPILOG
        # /////////////////////////////////////////////////////////////////////////////
        
        # Wait all issued wgmmas to complete, to ensure the accumulated results in tCrC are ready to be stored
        cute.nvgpu.warpgroup.wait_group(0)

        if cute.size(self.cluster_shape_mn) > 1:
            # Wait for all threads in the cluster to finish, avoid early release of smem
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        else:
            # For cluster that has a single thread block, it might have more than one warp groups.
            # Wait for all warp groups in the thread block to finish, because smem for tensor A in
            # the mainloop is reused in the epilogue.
            cute.arch.sync_threads()

        # Get the thread slice of the epilogue R2S tiled copy 
        thr_copy_r2s = epi_tiled_copy_r2s.get_slice(tidx)
        
        # Partition the dst shared memory of sC with layout (epi_tile=((8,16), 32), epi_stage=(1,4))
        # with this tiled copy with tiler-mn (m128, n16) and copy atom TV layout of (32, (2,4)),
        # to get the tiled dst shared memory tRS_sD for this thread
        # with layout: (R2S=(2,4), R2S_M=1, R2S_N=2, PIPE_D=(1,4))
        # i.e. this thread copies (2,4) elems in one tiled copy atom, and will repeat 2 times along N 
        # (since each tiled copy a (m128, n16) tiler in C, so repeating 2 times along N will cover the (m128, n32) epi_tile of C),
        # and 4 times along the pipeline stage to finish copying a subtile (m128, n128) of C
        # then repeating the whole thing for the next subtile (m128, n128) of C to cover the whole tile (m128, n256) of C
        tRS_sD = thr_copy_r2s.partition_D(sC)
        
        # Since tCrC has a mma layout of ( (2,2,32), 1, 1 ) to hold 128 fp32 elems of C in this thread
        # while the tiled copy atom (stmatrix inst) needs 8=(2,4) contiguous fp32 elems in one go,
        # so we need to retile tCrC to have a layout of (8,16):(1,8) to match the tiled copy atom layout,
        # where 8 elems in one tiled copy, and repeat 16 = R2S_N=2 x PIPE_D=4 x LOOP=(256/128)=2 times to cover the whole tile (m128, n256) of C
        tRS_rAcc = epi_tiled_copy_r2s.retile(tCrC)
        
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("thr_copy_r2s.layout_src_tv: {}", thr_copy_r2s.layout_src_tv)
                cute.printf("thr_copy_r2s.layout_src_tv_tiled: {}", thr_copy_r2s.layout_src_tv_tiled)
                cute.printf("thr_copy_r2s.layout_dst_tv: {}", thr_copy_r2s.layout_dst_tv)
                cute.printf("thr_copy_r2s.layout_dst_tv_tiled: {}", thr_copy_r2s.layout_dst_tv_tiled)
                cute.printf("tRS_sD:")
                cute.print_tensor(tRS_sD)
                cute.printf("tRS_rAcc:")
                cute.print_tensor(tRS_rAcc)

        # Allocate D registers as a register buffer for a single tiled copy in each stage
        # tRS_rD_layout is (R2S=((2,2,2),1), R2S_M=1, R2S_N=2)
        # which means in each stage, this thread will copy 8=(2,2,2) fp32 elems from src registers and repeat 2 times along N
        # NOTE: the input tensors of partition_D / partition_S are both sC, 
        # instead of more natually passing tCrC for partition_S and passing sC for partition_D,
        # because the tiled_copy_r2s determines the mapping based on the smem layout,
        # so we only need the dst sC to figure out how the dst tRS_sD, as well as src tRS_rD look like
        rD_staged_shape = cute.shape(thr_copy_r2s.partition_S(sC))
        tRS_rD_layout = cute.make_layout(rD_staged_shape[:3]) # removing pipeline stage mode
        tRS_rD = cute.make_fragment_like(tRS_rD_layout, self.acc_dtype) # fp32 buffer to slice from tCrC
        tRS_rD_out = cute.make_fragment_like(tRS_rD_layout, self.c_dtype) # c_dtype buffer for type conversion output before copying to shared memory
        size_tRS_rD = cute.size(tRS_rD) # 16 fp32 elems in one tiled copy of tRS_rD
        
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("rD_staged_shape: {}, tRS_rD_layout: {}, size_tRS_rD: {}", rD_staged_shape, tRS_rD_layout, size_tRS_rD)
                cute.printf("tRS_rD:")
                cute.print_tensor(tRS_rD)

        # Make tma copy partitions for the epilogue store of C from shared memory to global memory
        # where:
        #   1. sepi_for_tma_partition is just group the non-pipeline modes together with layout: (epi_tile=((8,16),(32,1)), epi_stages=(1,4))
        #   2. tCgC_for_tma_partition is just divide the global memory tensor gC_mnl by the epi_tile with layout (m128, n32),
        #       resulting in (epi_tile=(128,32), rest_n=(1,8)) tiled layout, which indicates how many epi_tiles (rest_n) we need 
        #       to finish copying the whole gC_mnl
        sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
        tCgC_for_tma_partition = cute.zipped_divide(gC_mnl, self.epi_tile)

        # And tma partition will return 
        #  1. bSG_sD with layout: (TMA_TILE=(4096,1), epi_stages=(1,4)) as the src shared memory
        #  2. bSG_gD with layout: (TMA_TILE=((32,128),1), rest_n=(1,8)) as the dst global memory
        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
            atom=tma_atom_c,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            smem_tensor=sepi_for_tma_partition,
            gmem_tensor=tCgC_for_tma_partition,
        )

        epi_tile_num = cute.size(tCgC_for_tma_partition, mode=[1]) # rest_n=8, number of epi_tiles
        epi_tile_shape = tCgC_for_tma_partition.shape[1] # rest_n_shape=(1,8)
        epi_tile_layout = cute.make_layout( # rest_n_layout=(1,8):(8,1)
            epi_tile_shape, stride=(epi_tile_shape[1], 1)
        )
        
        if const_expr(self.debug_print):
            if is_thread0:
                cute.printf("")
                cute.printf("epi_tile_num: {}, epi_tile_shape: {}, epi_tile_layout: {}", epi_tile_num, epi_tile_shape, epi_tile_layout)
                cute.printf("sepi_for_tma_partition:")
                cute.print_tensor(sepi_for_tma_partition)
                cute.printf("tCgC_for_tma_partition:")
                cute.print_tensor(tCgC_for_tma_partition)
                cute.printf("bSG_sD:")
                cute.print_tensor(bSG_sD)
                cute.printf("bSG_gD:")
                cute.print_tensor(bSG_gD)

        # Initialize tma store c_pipeline using `TmaStoreFence` instead of `MbarrierArray`
        # which servers as an alias API for `cp.async.bulk.commit_group` and `cp.async.bulk.wait_group`
        # since we don't need to know if the 
        c_producer_group = pipeline.CooperativeGroup( # NOTE: this is only a dummy placeholder, no matter waht the arguments are
            pipeline.Agent.Thread, self.threads_per_cta
        )
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=c_producer_group,
        )

        for epi_idx in cutlass.range_constexpr(epi_tile_num): # 8
            # Copy from accumulators to D registers (fp32 -> fp32)
            for epi_v in cutlass.range_constexpr(size_tRS_rD): # 16 = (2,2,2) x 2
                # NOTE: it's ok to directly use tCrC for the src tensor, 
                # since it is element-wise copied to tRS_rD without any reduction, 
                # so no need to worry about the layout difference between tCrC and tRS_rAcc
                # but it's still a good habit to use the retiled tiled_copy layout tRS_rAcc, instead of original tiled_mma layout tCrC
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v] 

            # Type conversion (fp32 -> c_dtype) in registers
            tRS_rD_out.store(tRS_rD.load().to(self.c_dtype))

            # R2S-Copy from D registers to shared memory
            epi_stage_idx = epi_idx % self.epi_stage
            cute.copy(
                atom=epi_tiled_copy_r2s,
                src=tRS_rD_out,
                dst=tRS_sD[(None, None, None, epi_stage_idx)] # slice the stage mode with epi_buffer index
            )

            # Fence-proxy between the R2S copy using generic proxy above
            # with the S2G copy using TMA proxy below, to ensure the visibility of the copied data in shared memory 
            # to the TMA proxy for the following S2G copy
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            
            # And we also need to sync (bar.sync) all the threads in the CTA
            # to ensure all threads have finished the R2S copy 
            # and the data in shared memory is all ready to allow the warp0 below to issue the S2G copy using TMA
            cute.arch.barrier()

            # S2G-Copy from shared memory to global memory using TMA
            if warp_idx == 0:
                gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                cute.copy(
                    atom=tma_atom_c,
                    src=bSG_sD[(None, epi_stage_idx)],
                    dst=bSG_gD[(None, gmem_coord)],
                )
                
                # `producer_commit` for TMAStoreFence will issue `cp.async.bulk.commit_group` 
                # to commit the current group of TMA store insts
                c_pipeline.producer_commit()
                
                # `producer_acquire` for TMAStoreFence will issue `cp.async.bulk.wait_group(num_stages-1)` 
                # to wait for the earliest committed TMA store group to complete, 
                # to allow only `num_stages-1` committed but unfinished TMA store groups in the pipeline,
                # to avoid the next stage buffer's data being overwritten before it is consumed by the TMA store in the next iteration
                c_pipeline.producer_acquire()

            # Wait for warp0 to issue the S2G copy before entering the next epi_tile iteration
            cute.arch.barrier()

        if warp_idx == 0:
            # `producer_tail` for TMAStoreFence will issue `cp.async.bulk.wait_group(0)`
            # to wait for all committed TMA store groups to complete before the producer (warp0) can exit
            c_pipeline.producer_tail()

        return

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/B operand stages, epilogue stages)
        :rtype: tuple[int, int]
        """

        epi_stage = 4
        # epi_smem will reuse smem ab.
        epi_bytes = 0

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None)) # (tileM, tileK)
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None)) # (tileN, tileK)
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        ab_stage = (
            smem_capacity // occupancy - mbar_helpers_bytes - epi_bytes
        ) // ab_bytes_per_stage
        
        return ab_stage, epi_stage

    @staticmethod
    def _sm90_compute_tile_shape_or_override(
        tile_shape_mnk: tuple[int, int, int],
        element_type: type[cutlass.Numeric],
        is_cooperative: bool = False,
        epi_tile_override: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        """Compute the epilogue tile shape or use override if provided.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param element_type: Data type of elements
        :type element_type: type[cutlass.Numeric]
        :param is_cooperative: Whether to use cooperative approach
        :type is_cooperative: bool
        :param epi_tile_override: Optional override for epilogue tile shape
        :type epi_tile_override: Tuple[int, int] or None

        :return: Computed epilogue tile shape
        :rtype: Tuple[int, int]
        """
        if epi_tile_override is not None:
            return epi_tile_override
        if is_cooperative:
            tile_m = min(128, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(32, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)
        else:
            n_perf = 64 if element_type.width == 8 else 32
            tile_m = min(64, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(n_perf, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        epi_tile: tuple[int, int],
        a_dtype: type[cutlass.Numeric],
        a_layout: utils.LayoutEnum,
        b_dtype: type[cutlass.Numeric],
        b_layout: utils.LayoutEnum,
        ab_stage: int,
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        epi_stage: int,
        debug_print: bool = False,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
        """Create shared memory layouts for A, B, and C tensors.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param a_dtype: Data type for matrix A
        :type a_dtype: type[cutlass.Numeric]
        :param a_layout: Layout enum for matrix A
        :type a_layout: utils.LayoutEnum
        :param b_dtype: Data type for matrix B
        :type b_dtype: type[cutlass.Numeric]
        :param b_layout: Layout enum for matrix B
        :type b_layout: utils.LayoutEnum
        :param ab_stage: Number of stages for A/B tensors
        :type ab_stage: int
        :param c_dtype: Data type for output matrix C
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum for the output matrix C
        :type c_layout: utils.LayoutEnum
        :param epi_stage: Number of epilogue stages
        :type epi_stage: int

        :return: Tuple of shared memory layouts for A, B, and C
        :rtype: Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]
        """
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None)) # (tileM, tileK)

        a_is_k_major = (
            a_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
        )
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]
        a_smem_layout_atom_kind = sm90_utils.get_smem_layout_atom( # K_SW128 if k-major
            layout=a_layout,
            element_type=a_dtype,
            # how many elems in one row, if it's 64 with 2B per elem, 
            # then one row has 128B and can use SW128, i.e. SW(B3, M4, S3),
            # where 2^(B3 + M4) = 2^7 = 128B
            major_mode_size=a_major_mode_size,
        )
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom( # (8,64) if k-major
            kind=a_smem_layout_atom_kind,
            element_type=a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            atom=a_smem_layout_atom,
            trg_shape=cute.append(a_smem_shape, ab_stage),
            # if in k-major, then the first extending order is m mode (along rows)
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        b_is_k_major = (
            b_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
        )
        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom_kind = sm90_utils.get_smem_layout_atom( # K_SW128 if k-major
            layout=b_layout,
            element_type=b_dtype,
            major_mode_size=b_major_mode_size,
        )
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom( # (8,64) if k-major
            kind=b_smem_layout_atom_kind,
            element_type=b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            atom=b_smem_layout_atom,
            trg_shape=cute.append(b_smem_shape, ab_stage),
            # if in k-major, then the first extending order is n mode (along rows)
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        c_smem_shape = epi_tile
        c_major_mode_size = epi_tile[1 if c_layout.is_n_major_c() else 0]
        c_smem_layout_atom_kind = sm90_utils.get_smem_layout_atom( # K_SW64 if n-major
            layout=c_layout,
            element_type=c_dtype,
            # Since c_major_size is 32, then one row has 64B if c_dtype is 2B-wide, 
            # so can use SW64, i.e. SW(B2, M4, S3), where 2^(B2 + M4) = 2^6 = 64B
            major_mode_size=c_major_mode_size,
        )
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom( # (8,32) if n-major
            kind=c_smem_layout_atom_kind,
            element_type=c_dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            atom=c_smem_layout_atom,
            trg_shape=cute.append(c_smem_shape, epi_stage),
            # if in n-major, then the first extending order is m mode (along rows)
            order=(0, 1, 2) if c_layout.is_n_major_c() else (1, 0, 2),
        )

        if const_expr(debug_print):
            print()
            print(f"{a_is_k_major=}, {a_major_mode_size=}, {a_smem_layout_atom_kind=}")
            print(f"{b_is_k_major=}, {b_major_mode_size=}, {b_smem_layout_atom_kind=}")
            print(f"{c_layout.is_n_major_c()=}, {c_major_mode_size=}, {c_smem_layout_atom_kind=}")
            print("a_smem_layout_atom: ", a_smem_layout_atom)
            print("b_smem_layout_atom: ", b_smem_layout_atom)
            print("c_smem_layout_atom: ", c_smem_layout_atom)
            print("a_smem_layout_staged: ", a_smem_layout_staged)
            print("b_smem_layout_staged: ", b_smem_layout_staged)
            print("epi_smem_layout_staged: ", epi_smem_layout_staged)

        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    @staticmethod
    def _make_epi_tiled_copy(
        tiled_mma: cute.TiledMma,
        c_layout: utils.LayoutEnum,
        c_dtype: type[cutlass.Numeric],
        acc_dtype: type[cutlass.Numeric],
        debug_print: bool = False,
    ) -> cute.TiledCopy:
        # Select the appropriate smem store operation based on C's layout and data type
        # and when the c_dtype (smem store dtype) is 2B-wide (like bf16/fp16), 
        # it will select `stmatrix.m8n8.x4.b16` intruction 
        # to warp-level vectorized store a (m=8x4, n=8) 2B smem tile in one go within a warp, where:
        #   1. one `m8n8` tile will be copied by 32 threads in this warp and 1 thread will copy 2 contiguous elems
        #       so this 8x8 matrix will result in (8,(4,2)) thread layout for 32 threads in this warp
        #       where one group of 4 contiguous threads will together copy 8 elements in one row ( inner layout: (4,2) )
        #       and 8 groups forming a warp will together copy all of the 8 rows ( outer layout: 8 )
        #   2. and since it has `x4`, so this copy atom will repeat 4 times, thus finall resulting in
        #       `(32,(2,4)):(2,(1,64))` src TV layout, where each thread in 32 will copy 2 contiguous elements
        #       and it will repeat 4 times in last mode with stride of 32x2=64 elems
        #       so each thread will copy 8 elements in total, so the dst TV layout is simply `(32,8):(8,1)`
        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            layout_d=c_layout, # if c_layout.is_m_major_c, then stmatrix will transpose
            elem_ty_d=c_dtype, # if bf16, use stmatrix.b16
            elem_ty_acc=acc_dtype, # no use
        )

        # Since the `copy_atom_r2s` is only the smallest copy atom to copy 32x8 elems in one warp, 
        # and using the info from tiled_mma, we also need to know how to use all the 4x2=8 warps in the 2 mma warp groups
        # to copy a larger tile of C, as a larger tiled copy atom, to copy (32x8) x (4x2) = m128 x n16 elems in one go:
        #   1. the Tiler-M layout of m128 is (8,8,2):(1,16,8), while the Tiler-N layout of n16 is (4,2,2):(2,1,8)
        #   2. and the TV layout for all 8 mma warps is: (4,8,8),(2,2,2), where each thread in 256 will copy 8 elems
        tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(atom=copy_atom_r2s, mma=tiled_mma)

        # Using the two copy atoms above to make the final R2S tiled copy for the accumulator C of the tiled mma
        # sharing the same Tiler-MN as `tiled_copy_C_Atom` but different TV layout of ((4,64),((2,2,2),1))
        # which is more suitable for R2S copy with a pipeling-stage mode placeholder 1, instead of (4,8,8),(2,2,2) for tiled_mma
        # NOTE: the src (R) and dst (S) memory are all assumed in the same dtype with 2B, 
        # so the type conversion (fp32->c_dtype) is done in the src registers by the user
        tiled_copy_r2s = cute.make_tiled_copy_S(
            atom=copy_atom_r2s,
            tiled_copy=tiled_copy_C_Atom,
        )
        
        if const_expr(debug_print):
            print()
            print("c_layout.is_m_major_c(): ", c_layout.is_m_major_c())
            print("copy_atom_r2s: ", copy_atom_r2s)
            print("tiled_copy_C_Atom: ", tiled_copy_C_Atom)
            print("tiled_copy_r2s: ", tiled_copy_r2s)
            
        
        return tiled_copy_r2s

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
    ) -> tuple[int, int, int]:
        """Compute grid shape for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]

        :return: Grid shape for kernel launch.
        :rtype: tuple[int, int, int]
        """

        c_shape = (tile_shape_mnk[0], tile_shape_mnk[1])
        gc = cute.zipped_divide(c, tiler=c_shape)
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        clusters = cute.ceil_div(cute.get(gc.layout, mode=[1]).shape, cluster_shape_mnl)
        grid = tuple(x * y for x, y in zip(clusters, cluster_shape_mnl))
        return grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int],
        debug_print: bool = False,
        title: str = "",
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for C tensor storage.

        :param tensor_c: Output tensor C
        :type tensor_c: cute.Tensor
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]

        :return: TMA atom and tensor for C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        
        # NOTE: this step is also the first step inside of `make_tiled_tma_atom`
        # and it can be executed in advance here since ID∘X = X
        # so we can directly pass in epi_tile as the cta tiler just like tma atom of A/B
        # 
        # and the reason why cute needs to use this ID composition, is to make the cta tiler a TMA layout with the basis stride:
        # e.g. epi_tiler=(128,32) => c_cta_v_layout: (128,32):(1@0,1@1)
        # it is different from the layout with normal stride, mapping a coord (m,n) to a flatten offset integer
        # which will map a coord to the multi-dim TMA coord tuple (d0, d1) to TMA
        # since TMA will store the strides of each dim itself and figure its own offset
        # 
        # e.g. for smem tma tensor mA_mkl: (m=2048,n=1024,l=1):(1@1,1@0,1@2), where `1@k` means the k-th basis vector,
        # so the n is the innermost dim0, m is second dim1, and l is the outermost dim2, 
        # indicating it is a row-major layout of (m,k) and the l is the "batch" dim
        # 
        # and for gmem tma tensor gA_mkl: (128,64,16):(1@1,1@0,64@0), where `N@k` means the vector 
        # with the same direction as the k-th basis vector but with length/mode of N, 
        # so the k is the innermost dim0 with step size 1, m is the second dim1 with step size 1, 
        # and l is also the dim0, but with step size 64 == k,
        # indication it is a row-major layout of (m,k), and l is actually the "rest_k" dim
        c_cta_v_layout = cute.composition(
            cute.make_identity_layout(tensor_c.shape), epi_tile
        )
        
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op=cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            gmem_tensor=tensor_c,
            smem_layout=epi_smem_layout,
            cta_tiler=c_cta_v_layout, # or directly pass in `epi_tile`
            num_multicast=1,
        )
        
        if const_expr(debug_print):
            print("")
            print(f"{title}: epi_smem_layout: {epi_smem_layout}, c_cta_v_layout: {c_cta_v_layout}")
            print()
            print(f"tma_atom_c: {tma_atom_c}")
            print()
            

        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
        debug_print: bool = False,
        title: str = "",
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

        :param tensor: Input tensor (A or B)
        :type tensor: cute.Tensor
        :param smem_layout_staged: Shared memory layout for the tensor
        :type smem_layout_staged: cute.ComposedLayout
        :param smem_tile: Shared memory tile shape
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: Multicast dimension
        :type mcast_dim: int

        :return: TMA atom and tensor
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        op = (
            # cp.async GMEM -> SMEM bulk tensor copy Operation
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            # cp.async GMEM -> SMEM bulk tensor multicast copy Operation
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0)) # removing the pipeline stage mode
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op=op,
            gmem_tensor=tensor,
            smem_layout=smem_layout,
            cta_tiler=smem_tile,
            num_multicast=mcast_dim,
        )
        
        if const_expr(debug_print):
            print(f"{title}: op: {op}, smem_layout: {smem_layout}, cta_tiler: {smem_tile}, mcast_dim: {mcast_dim}")
            print()
            print(f"tma_atom: {tma_atom}")
        
        return tma_atom, tma_tensor

    @staticmethod
    def is_valid_dtypes(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
    ) -> bool:
        """
        Check if the dtypes are valid

        :param a_dtype: The data type of tensor A
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of tensor B
        :type b_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: major mode of tensor A
        :type a_major: str
        :param b_major: major mode of tensor B
        :type b_major: str

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        # tested a_dtype
        if a_dtype not in {
            cutlass.Float16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        # tested b_dtype
        if b_dtype not in {
            cutlass.Float16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        # tested acc_dtype
        if acc_dtype not in {cutlass.Float32, cutlass.Float16}:
            is_valid = False
        # tested c_dtype
        if c_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        # make sure a_dtype == b_dtype for Float16
        if a_dtype.width == 16 and a_dtype != b_dtype:
            is_valid = False
        # make sure a_dtype.width == b_dtype.width (i.e, Float8E4M3FN or Float8E5M2)
        if a_dtype.width != b_dtype.width:
            is_valid = False

        # for Float8 types, this implementation only supports k-major layout
        if (a_dtype.width == 8 and a_major != "k") or (
            b_dtype.width == 8 and b_major != "k"
        ):
            is_valid = False

        return is_valid


def run(
    mnkl: Tuple[int, int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    tile_shape_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float,
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
    use_cold_l2: bool = False,
    **kwargs,
):
    """
    Prepare A/B/C tensors, launch GPU kernel, and reference checking.

    :param mnkl: Problem size (M, N, K, L)
    :type mnkl: Tuple[int, int, int, int]
    :param a_dtype: Data type for input tensor A
    :type a_dtype: Type[cutlass.Numeric]
    :param b_dtype: Data type for input tensor B
    :type b_dtype: Type[cutlass.Numeric]
    :param c_dtype: Data type for output tensor C
    :type c_dtype: Type[cutlass.Numeric]
    :param acc_dtype: Data type for accumulation during matrix multiplication
    :type acc_dtype: Type[cutlass.Numeric]
    :param a_major/b_major/c_major: Memory layout of tensor A/B/C
    :type a_major/b_major/c_major: str
    :param tile_shape_mn: CTA tile shape (M, N)
    :type tile_shape_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster shape (M, N)
    :type cluster_shape_mn: Tuple[int, int]
    :param tolerance: Tolerance value for reference validation comparison
    :type tolerance: float
    :param warmup_iterations: Number of warmup iterations before benchmarking, defaults to 0
    :type warmup_iterations: int, optional
    :param iterations: Number of benchmark iterations to run, defaults to 1
    :type iterations: int, optional
    :param skip_ref_check: Whether to skip reference result validation, defaults to False
    :type skip_ref_check: bool, optional
    :param use_cold_l2: Whether to use circular buffer strategy to ensure cold L2 cache, defaults to False
    :type use_cold_l2: bool, optional
    :return: Execution time of the GEMM kernel in microseconds
    :rtype: float
    """

    print(f"Running Hopper Dense GEMM with:")
    print(f"mnkl: {mnkl}")
    print(
        f"A dtype: {a_dtype}, B dtype: {b_dtype}, C dtype: {c_dtype}, Acc dtype: {acc_dtype}"
    )
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Tile Shape: {tile_shape_mn}, Cluster Shape: {cluster_shape_mn}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {use_cold_l2}")

    # Unpack parameters
    m, n, k, l = mnkl

    # Skip unsupported types
    if not HopperWgmmaGemmKernel.is_valid_dtypes(
        a_dtype, b_dtype, acc_dtype, c_dtype, a_major, b_major
    ):
        raise TypeError(
            f"Skipping due to unsupported combination of types and majors: {a_dtype}, {b_dtype}, {acc_dtype}, {c_dtype}, {a_major=}, {b_major=}"
        )

    # Prepare pytorch tensors: A, B (random from 0 to 2) and C (all zero)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    # Create and permute tensor A/B/C
    def create_and_permute_tensor(
        l, mode0, mode1, is_mode0_major, dtype, is_dynamic_layout=True
    ):
        # is_mode0_major: (l, mode1, mode0) -> (mode0, mode1, l)
        # else : (l, mode0, mode1) -> (mode0, mode1, l)
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
        torch_tensor_cpu = cutlass.torch.create_and_permute_torch_tensor(
            shape,
            torch_dtype,
            permute_order=permute_order,
            init_type=cutlass.torch.TensorInitType.RANDOM,
            init_config=cutlass.torch.RandomInitConfig(
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
        cute_tensor = cutlass.torch.convert_cute_tensor(
            f32_torch_tensor,
            cute_tensor,
            dtype,
            is_dynamic_layout=is_dynamic_layout,
        )

        return f32_torch_tensor, cute_tensor, torch_tensor

    a, mA, a_torch = create_and_permute_tensor(l, m, k, a_major == "m", a_dtype)
    b, mB, b_torch = create_and_permute_tensor(l, n, k, b_major == "n", b_dtype)
    c, mC, c_torch = create_and_permute_tensor(l, m, n, c_major == "m", c_dtype)

    gemm = HopperWgmmaGemmKernel(acc_dtype, tile_shape_mn, cluster_shape_mn, debug_print=DEBUG_MODE)

    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    # compile gemm kernel
    compiled_gemm = cute.compile(gemm, mA, mB, mC, stream)

    if not skip_ref_check:
        # execution
        compiled_gemm(mA, mB, mC, stream)

        torch.cuda.synchronize()

        # Ref check
        ref = (torch.einsum("mkl,nkl->mnl", a, b)).cpu()

        if c_dtype in (cutlass.Float8E4M3FN, cutlass.Float8E5M2):
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

        torch.testing.assert_close(c_torch.cpu(), ref_c, atol=tolerance, rtol=1e-03)

    def generate_tensors():
        _, mA_workspace, _ = create_and_permute_tensor(l, m, k, a_major == "m", a_dtype)
        _, mB_workspace, _ = create_and_permute_tensor(l, n, k, b_major == "n", b_dtype)
        _, mC_workspace, _ = create_and_permute_tensor(l, m, n, c_major == "m", c_dtype)
        return testing.JitArguments(mA_workspace, mB_workspace, mC_workspace, stream)

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            a_torch.numel() * a_torch.element_size()
            + b_torch.numel() * b_torch.element_size()
            + c_torch.numel() * c_torch.element_size()
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = testing.benchmark(
        compiled_gemm,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    return exec_time  # Return execution time in microseconds


if __name__ == "__main__":
    args = parse_arguments()
    run(
        args.mnkl,
        args.a_dtype,
        args.b_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.tile_shape_mn,
        (2, 1) if DEBUG_MODE else args.cluster_shape_mn,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        True if DEBUG_MODE else args.skip_ref_check,
        args.use_cold_l2,
    )
    print("PASS")
