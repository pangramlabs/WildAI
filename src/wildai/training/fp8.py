# Adapted from nanochat (https://github.com/karpathy/nanochat), Copyright (c) 2025 Andrej Karpathy, MIT License; see NOTICE.
"""FP8 matmuls with tensorwise dynamic scaling, as used to train every WildAI model.

Each of a Linear layer's three GEMMs (forward, grad-input, grad-weight) quantizes its operands with one scale per
tensor and calls `torch._scaled_mm`: inputs and weights in e4m3, gradients in e5m2. Master weights stay in full
precision. Requires a GPU with FP8 tensor cores (compute capability 8.9 or newer, e.g. H100).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

EPS = 1e-12
MIN_FEATURES = 128  # smaller layers gain nothing from FP8


def supports_fp8(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) >= (8, 9)


@torch.no_grad()
def _to_fp8(x: torch.Tensor, fp8_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize with one scale for the whole tensor; returns the FP8 data and the inverse scale `_scaled_mm` expects."""
    fp8_max = torch.finfo(fp8_dtype).max
    amax = x.float().abs().max()
    # float64 division keeps compiled and eager numerics identical
    scale = (fp8_max / amax.double().clamp(min=EPS)).float()
    x_fp8 = (x.float() * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return x_fp8, scale.reciprocal()


def _to_col_major(x: torch.Tensor) -> torch.Tensor:
    return x.t().contiguous().t()


@torch._dynamo.allow_in_graph
class _Float8Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, input_2d: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        input_fp8, input_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)
        ctx.save_for_backward(input_fp8, input_inv, weight_fp8, weight_inv)
        return torch._scaled_mm(input_fp8, weight_fp8.t(), scale_a=input_inv, scale_b=weight_inv, out_dtype=input_2d.dtype, use_fast_accum=True)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        in_fp8, in_inv, w_fp8, w_inv = ctx.saved_tensors
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)
        grad_input = torch._scaled_mm(go_fp8, _to_col_major(w_fp8), scale_a=go_inv, scale_b=w_inv, out_dtype=grad_output.dtype, use_fast_accum=False)
        grad_weight = torch._scaled_mm(
            go_fp8.t().contiguous(),
            _to_col_major(in_fp8),
            scale_a=go_inv,
            scale_b=in_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )
        return grad_input, grad_weight


class Float8Linear(nn.Linear):
    """`nn.Linear` whose matmul runs in FP8; shares the original layer's parameters."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(torch.bfloat16)
        shape = input.shape
        output = _Float8Matmul.apply(input.reshape(-1, shape[-1]), self.weight)
        output = output.reshape(*shape[:-1], output.shape[-1])
        return output if self.bias is None else output + self.bias.to(output.dtype)

    @classmethod
    def from_linear(cls, module: nn.Linear) -> Float8Linear:
        with torch.device("meta"):
            new = cls(module.in_features, module.out_features, bias=False)
        new.weight = module.weight
        new.bias = module.bias
        return new


def fp8_eligible(module: nn.Module) -> bool:
    return (
        isinstance(module, nn.Linear)
        and not isinstance(module, Float8Linear)
        and module.in_features % 16 == 0
        and module.out_features % 16 == 0
        and min(module.in_features, module.out_features) >= MIN_FEATURES
    )


def convert_to_fp8(module: nn.Module, eligible: Callable[[nn.Module], bool] = fp8_eligible) -> int:
    """Swap eligible Linear layers for `Float8Linear` in place; returns how many were converted."""
    converted = 0
    for name, child in module.named_children():
        converted += convert_to_fp8(child, eligible)
        if eligible(child):
            setattr(module, name, Float8Linear.from_linear(child))
            converted += 1
    return converted
