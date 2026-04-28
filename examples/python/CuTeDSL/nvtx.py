# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import wraps
from typing import Any, Callable, Literal, Optional, TypeAlias, TypeVar, cast, Sequence

import torch
import torch.distributed as dist

# Global flag to enable/disable the profiler
_PROFILER_ENABLED = False
_EMIT_NVTX_CTX: None | torch.autograd.profiler.emit_nvtx = None


# fixed the mypy type check missing bug
# when a func is wrapped
# issue: https://stackoverflow.com/questions/65621789/mypy-untyped-decorator-makes-function-my-method-untyped
F = TypeVar("F", bound=Callable[..., Any])
ProfileType: TypeAlias = Literal["memory", "nsys"]


def wrap_to_list(x: Any | Sequence[Any], broadcast_to_length: int = 1) -> list[Any]:
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
        assert broadcast_to_length == 1
        return list(x)
    else:
        return [x] * broadcast_to_length


def _forward_pre_hook(layer, inputs):
    """
    Pre-hook for the forward pass of a layer. If profiling is enabled,
    pushes a range onto the NVTX stack.

    Args:
    - layer: The layer for which this hook is called.
    - inputs: The inputs to the layer.
    """
    if _PROFILER_ENABLED:
        torch.cuda.nvtx.range_push(layer.__class__.__name__ + "_fwd")


def _forward_post_hook(module, inputs, outputs):
    """
    Post-hook for the forward pass of a module. If profiling is enabled,
    pops a range from the NVTX stack.

    Args:
    - module: The module for which this hook is called.
    - inputs: The inputs to the module.
    - outputs: The outputs from the module.
    """
    if _PROFILER_ENABLED:
        torch.cuda.nvtx.range_pop()


def _register_hook_recursively(module, pre_hook, post_hook):
    """
    Recursively registers pre and post hooks to all submodules of the given module.

    Args:
    - module: The root module to register hooks for.
    - pre_hook: The pre-hook function to be registered.
    - post_hook: The post-hook function to be registered.
    """
    if not isinstance(module, torch.nn.Module):
        return

    # Recursively apply hooks to all submodules
    for submodule in module.children():
        _register_hook_recursively(submodule, pre_hook, post_hook)

    # Register hooks
    if pre_hook is not None:
        module.register_forward_pre_hook(hook=pre_hook)
    if post_hook is not None:
        module.register_forward_hook(hook=post_hook)


def register_profile_hook(model):
    """
    Registers profiling hooks for a PyTorch model or list of models.

    Args:
    - model: A PyTorch model or a list of PyTorch models.
    """
    if isinstance(model, torch.nn.Module):
        _register_hook_recursively(model, _forward_pre_hook, _forward_post_hook)
    elif isinstance(model, list):
        raise RuntimeError("register_profile_hook given a list of models")


# NOTE: since normally "switch_profile" is used in the training loop instead of inside the model,
# we don't have to make it compatible with torch.compile
def switch_profile(
    iter_id: int,
    start: int,
    end: int,
    profile_ranks: list[int] = [0],
    profile_type: ProfileType | list[ProfileType] = "nsys",
    event_name: Optional[str] = None,
    mem_snapshot_name: Optional[str] = None,
    record_shape: bool = True,
    enable: bool = True,
):
    """
    Controls the profiler state based on the iteration number. Turns on profiling
    at the start iteration and turns it off at the end iteration.

    Args:
        iter_id (int): The current iteration number.
        start (int): The iteration number to start profiling.
        end (int): The iteration number to end profiling.
        profile_ranks (list[int]): List of ranks to be profiled.
            Defaults to [0] to profile only rank0.
        profile_type (ProfileType | list[ProfileType], optional):
            The profiler type or list of profiler types to be used.
            Supports "nsys" or "memory".
        event_name (str, optional): Custom name for the profiling event.
            If None, defaults to 'iter{iter_id}'.
        mem_snapshot_name (str, optional): Name of the memory snapshot file.
            If None, defaults to 'memory_snapshot_iter({start}-{end})_r{rank}'.
        record_shape (bool, optional): Whether to record the operand shape of each operation
            with `torch.autograd.profiler.emit_nvtx`,
            NOTE: this might increase the CPU overhead for extra recording,
            as well as much more recompilation when using torch.compile.
        enable (bool): Whether to enable profiling. Useful to unify the code with a flag to control.
    """
    if not enable:
        return

    if not dist.is_initialized():
        assert profile_ranks == [0], (
            "profile_ranks can only contain rank0 "
            "if torch.distributed is not initialized"
        )
        rank = 0
    else:
        rank = dist.get_rank()
        if rank not in profile_ranks:
            return

    global _PROFILER_ENABLED
    global _EMIT_NVTX_CTX

    event_name = f"iter_{iter_id}" if event_name is None else event_name
    mem_snapshot_name = (
        f"memory_snapshot_iter({start}-{end})_r{rank}"
        if mem_snapshot_name is None or mem_snapshot_name == ""
        else mem_snapshot_name
    )
    profile_type = wrap_to_list(profile_type)  # type: ignore[assignment]

    # Start profiling
    if iter_id == start:
        if "nsys" in profile_type:
            if record_shape:
                emit_nvtx_ctx = torch.autograd.profiler.emit_nvtx(record_shapes=True)
                _EMIT_NVTX_CTX = emit_nvtx_ctx.__enter__()
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push(event_name)

        if "memory" in profile_type:
            torch.cuda.memory._record_memory_history()

        _PROFILER_ENABLED = True

    # Stop profiling
    elif iter_id == end:
        if "nsys" in profile_type:
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
            if record_shape:
                _EMIT_NVTX_CTX.__exit__(None, None, None)  # type: ignore[union-attr]
                _EMIT_NVTX_CTX = None

        if "memory" in profile_type:
            torch.cuda.memory._dump_snapshot(f"{mem_snapshot_name}.pickle")

        _PROFILER_ENABLED = False

    # Continue profiling
    elif iter_id > start and iter_id < end:
        if "nsys" in profile_type:
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push(event_name)


@torch.library.custom_op("odin::nvtx_range_push", mutates_args=())
def nvtx_range_push(event_name: str) -> None:
    """torch.ops.odin.nvtx_range_push"""
    torch.cuda.nvtx.range_push(event_name)


@nvtx_range_push.register_fake
def _(event_name: str) -> None:
    pass


@torch.library.custom_op("odin::nvtx_range_pop", mutates_args=())
def nvtx_range_pop() -> None:
    """torch.ops.odin.nvtx_range_pop"""
    torch.cuda.nvtx.range_pop()


@nvtx_range_pop.register_fake
def _() -> None:
    pass


# NOTE: since torch.compile does not support @contextlib.contextmanager,
# we use the class-based context manager
class add_nvtx_event:
    """
    Context manager to add an NVTX event around a code block.

    Args:
        event_name (str): The name of the event to be recorded.
    """

    def __init__(self, event_name: str):
        self.enter_name = event_name

    def __enter__(self):
        if torch.compiler.is_compiling():
            # NOTE: torch.compile supports neither retrieving the attributes from "self"
            # nor modifying a variable not in the current scope
            # so we have no choice but assign a constant event name when compiling
            nvtx_range_push("torch compile region")
        else:
            torch.cuda.nvtx.range_push(self.enter_name)
        return self

    def __exit__(self, *excinfo):
        if torch.compiler.is_compiling():
            nvtx_range_pop()
        else:
            torch.cuda.nvtx.range_pop()


def instrument_nvtx(func: F) -> F:
    """
    Decorator that records an NVTX range for the duration of the function call.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The wrapped function that is now being profiled.
    """

    @wraps(func)
    def wrapped_fn(*args, **kwargs):
        if torch.compiler.is_compiling():
            # NOTE: we can not access func.__qualname__ when compiling
            # thus use func.__name__ instead
            func_name = func.__name__
        else:
            func_name = func.__qualname__

        with add_nvtx_event(func_name):
            ret_val = func(*args, **kwargs)
        return ret_val

    return cast(F, wrapped_fn)
