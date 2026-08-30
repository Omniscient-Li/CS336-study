from __future__ import annotations

import importlib.util
import pathlib

# tests/ 的上一级目录就是 chapter2/hw1，作业文件都在这
_HW1_DIR = pathlib.Path(__file__).resolve().parent.parent


def get_flashattention_autograd_function_pytorch() -> type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).
    """
    # 按文件路径加载用户的实现（和 chapter1 tests/adapters.py 同一套路）
    spec = importlib.util.spec_from_file_location(
        "flashattention_autograd_function_pytorch",
        str(_HW1_DIR / "flashattention_autograd_function_pytorch.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Autograd_function_pytorch


def get_flashattention_autograd_function_triton() -> type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.
    """
    # Triton 版还没实现，跑测试时用 pytest -k pytorch 过滤掉 triton 的用例
    raise NotImplementedError
