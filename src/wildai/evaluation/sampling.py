"""Batched ancestral sampling with a KV cache, with temperature, top-p and top-k truncation.

All prompts in a batch have the same length, so the batch is prefilled at once. A row stops when it samples a stop
token (`<|bos|>`, the document boundary, for base models); the stop token is not part of the continuation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from wildai.training.model import GPT, KVCache


@dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None


def sample_next(logits: torch.Tensor, sampling: Sampling, rng: torch.Generator) -> torch.Tensor:
    """One token per row of `(B, V)` logits. Top-p and top-k compose: a token must survive both; the most likely
    token always survives."""
    logits = logits.float()
    if sampling.temperature == 0.0:
        return torch.argmax(logits, dim=-1)
    logits = logits / sampling.temperature
    if sampling.top_p is None and sampling.top_k is None:
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1, generator=rng)[:, 0]
    sorted_logits, sorted_ids = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    keep = torch.ones_like(probs, dtype=torch.bool)
    if sampling.top_p is not None:
        keep &= torch.cumsum(probs, dim=-1) - probs < sampling.top_p
    if sampling.top_k is not None and sampling.top_k < keep.size(1):
        keep[:, sampling.top_k :] = False
    keep[:, 0] = True
    probs = probs * keep
    probs = probs / probs.sum(dim=-1, keepdim=True)
    choice = torch.multinomial(probs, num_samples=1, generator=rng)
    return sorted_ids.gather(1, choice)[:, 0]


@torch.inference_mode()
def generate_batch(model: GPT, prompts: list[list[int]], sampling: Sampling, max_new_tokens: int, stop_ids: set[int], rng: torch.Generator) -> list[list[int]]:
    """Continuations (token ids, without the stop token) of equal-length prompts."""
    assert len({len(p) for p in prompts}) == 1, "prompts in a batch must have the same length"
    device = model.get_device()
    cache = KVCache(model.config, len(prompts), len(prompts[0]) + max_new_tokens, device, model.compute_dtype)
    logits = model(torch.tensor(prompts, dtype=torch.long, device=device), kv_cache=cache)[:, -1, :]
    continuations: list[list[int]] = [[] for _ in prompts]
    active = torch.ones(len(prompts), dtype=torch.bool, device=device)
    filler = min(stop_ids)
    for _ in range(max_new_tokens):
        rows = active.nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            break
        sampled = sample_next(logits[rows], sampling, rng)
        next_ids = torch.full((len(prompts),), filler, dtype=torch.long, device=device)
        next_ids[rows] = sampled
        for row, token in zip(rows.tolist(), sampled.tolist()):
            if token in stop_ids:
                active[row] = False
            else:
                continuations[row].append(token)
        if not bool(active.any()):
            break
        logits = model(next_ids.unsqueeze(1), kv_cache=cache)[:, -1, :]
    return continuations
