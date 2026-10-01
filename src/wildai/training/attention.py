# Adapted from nanochat (https://github.com/karpathy/nanochat), Copyright (c) 2025 Andrej Karpathy, MIT License; see NOTICE.
"""Causal attention with Flash Attention 3 on Hopper GPUs and a PyTorch SDPA fallback everywhere else.

FA3 is fetched from the Hugging Face kernel hub (`varunneal/flash-attention-3`) the first time a model is placed on a
Hopper GPU (`prepare(device)`); importing this module has no side effects. Tensors use FA3's `(B, T, H, D)` layout.
`window_size = (left, 0)` limits attention to the `left` previous tokens; `left >= T` means full causal attention.
"""

from __future__ import annotations

import os
from types import ModuleType

import torch
import torch.nn.functional as F

_FA3: ModuleType | None = None


def prepare(device: torch.device) -> bool:
    """Load FA3 if `device` is a Hopper GPU. Returns whether FA3 is active."""
    global _FA3
    if _FA3 is None and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] == 9:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        try:
            from kernels import get_kernel

            _FA3 = get_kernel("varunneal/flash-attention-3").flash_attn_interface
        except Exception:
            _FA3 = None
    return _FA3 is not None


def _use_fa3(q: torch.Tensor) -> bool:
    return _FA3 is not None and q.is_cuda and q.dtype == torch.bfloat16


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int) -> torch.Tensor:
    """SDPA over `(B, H, T, D)` tensors; queries are the last `Tq` positions of the `Tk` keys."""
    tq, tk = q.size(2), k.size(2)
    gqa = q.size(1) != k.size(1)
    if (window < 0 or window >= tq) and tq == tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)
    if tq == 1:
        if 0 <= window < tk:
            start = max(0, tk - (window + 1))
            k, v = k[:, :, start:], v[:, :, start:]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=gqa)
    rows = (tk - tq) + torch.arange(tq, device=q.device).unsqueeze(1)
    cols = torch.arange(tk, device=q.device).unsqueeze(0)
    mask = cols <= rows
    if 0 <= window < tk:
        mask = mask & ((rows - cols) <= window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=gqa)


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: tuple[int, int]) -> torch.Tensor:
    """Training/scoring attention over a full sequence."""
    if _use_fa3(q):
        return _FA3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
    y = _sdpa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), window_size[0])
    return y.transpose(1, 2)


def cached_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache_seqlens: torch.Tensor,
    window_size: tuple[int, int],
) -> torch.Tensor:
    """Generation attention: writes `k, v` into the caches at `cache_seqlens` (in place) and attends over the prefix."""
    if _use_fa3(q):
        return _FA3.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens, causal=True, window_size=window_size)
    t_new = q.size(1)
    pos = int(cache_seqlens[0].item())  # all rows of a generation batch share one position
    k_cache[:, pos : pos + t_new] = k
    v_cache[:, pos : pos + t_new] = v
    end = pos + t_new
    y = _sdpa(q.transpose(1, 2), k_cache[:, :end].transpose(1, 2), v_cache[:, :end].transpose(1, 2), window_size[0])
    return y.transpose(1, 2)
