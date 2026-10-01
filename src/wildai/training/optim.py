# Adapted from nanochat (https://github.com/karpathy/nanochat), Copyright (c) 2025 Andrej Karpathy, MIT License; see NOTICE.
# The Muon step is adapted from modded-nanogpt (https://github.com/KellerJordan/modded-nanogpt).
"""Combined Muon (transformer matrices) + AdamW (embeddings, output head, scalars) optimizer.

Muon: Nesterov momentum, Polar-Express orthogonalization (https://arxiv.org/abs/2505.16932), NorMuon per-neuron
variance normalization (https://arxiv.org/abs/2510.05491) and cautious weight decay, fused into one compiled kernel
per stack of same-shape matrices. `DistMuonAdamW` is the multi-GPU version: it reduce-scatters gradients (so no DDP
wrapper is needed), shards optimizer state ZeRO-2 style and all-gathers updated parameters.

Every param group is a dict with `kind` ('adamw' or 'muon') and its hyperparameters: AdamW groups carry `lr`,
`betas`, `eps`, `weight_decay`; Muon groups carry `lr`, `momentum`, `ns_steps`, `beta2`, `weight_decay`. Muon
groups must hold parameters of a single shape.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

ParamGroup = dict[str, Any]


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor, grad: Tensor, exp_avg: Tensor, exp_avg_sq: Tensor,
    step_t: Tensor, lr_t: Tensor, beta1_t: Tensor, beta2_t: Tensor, eps_t: Tensor, wd_t: Tensor,
) -> None:  # fmt: skip
    """Decoupled weight decay, moment updates, bias correction and the parameter update in one graph.

    Hyperparameters arrive as 0-D CPU tensors so changing their values never triggers recompilation.
    """
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t**step_t
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    p.add_(exp_avg / denom, alpha=-(lr_t / bias1))


# Polar Express coefficients for 5 iterations (safety factor 2e-2, cushion 2).
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor, stacked_params: Tensor, momentum_buffer: Tensor, second_momentum_buffer: Tensor,
    momentum_t: Tensor, lr_t: Tensor, wd_t: Tensor, beta2_t: Tensor,
    ns_steps: int, red_dim: int, bf16: bool,
) -> None:  # fmt: skip
    """Nesterov momentum -> Polar Express -> NorMuon variance reduction -> cautious update."""
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    X = g.bfloat16() if bf16 else g
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X.mT @ X
            X = a * X + X @ (b * A + c * (A @ A))
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X @ X.mT
            X = a * X + (b * A + c * (A @ A)) @ X
    g = X

    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    v_norm_new = ((v_mean * red_dim_size) * step_size.float().square()).sum(dim=(-2, -1), keepdim=True).sqrt()
    g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)

    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class _HyperTensors:
    """0-D CPU tensors holding the current hyperparameters (avoids recompiling the fused kernels)."""

    def __init__(self) -> None:
        def zero() -> Tensor:
            return torch.tensor(0.0, dtype=torch.float32, device="cpu")

        self.adamw_step, self.adamw_lr, self.adamw_beta1, self.adamw_beta2 = zero(), zero(), zero(), zero()
        self.adamw_eps, self.adamw_wd = zero(), zero()
        self.muon_momentum, self.muon_lr, self.muon_wd, self.muon_beta2 = zero(), zero(), zero(), zero()

    def adamw(self, group: ParamGroup, step: int) -> tuple[Tensor, ...]:
        self.adamw_step.fill_(step)
        self.adamw_lr.fill_(group["lr"])
        self.adamw_beta1.fill_(group["betas"][0])
        self.adamw_beta2.fill_(group["betas"][1])
        self.adamw_eps.fill_(group["eps"])
        self.adamw_wd.fill_(group["weight_decay"])
        return self.adamw_step, self.adamw_lr, self.adamw_beta1, self.adamw_beta2, self.adamw_eps, self.adamw_wd

    def muon(self, group: ParamGroup, shape: torch.Size) -> tuple[Tensor, ...]:
        self.muon_momentum.fill_(group["momentum"])
        self.muon_beta2.fill_(group["beta2"])
        # Tall matrices get a larger step, as in modded-nanogpt.
        self.muon_lr.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self.muon_wd.fill_(group["weight_decay"])
        return self.muon_momentum, self.muon_lr, self.muon_wd, self.muon_beta2


def _muon_state(state: dict[str, Tensor], count: int, shape: torch.Size, like: Tensor) -> tuple[Tensor, Tensor, int]:
    if "momentum_buffer" not in state:
        state["momentum_buffer"] = torch.zeros(count, *shape, dtype=like.dtype, device=like.device)
    if "second_momentum_buffer" not in state:
        factored = (count, shape[-2], 1) if shape[-2] >= shape[-1] else (count, 1, shape[-1])
        state["second_momentum_buffer"] = torch.zeros(factored, dtype=like.dtype, device=like.device)
    red_dim = -1 if shape[-2] >= shape[-1] else -2
    return state["momentum_buffer"], state["second_momentum_buffer"], red_dim


class MuonAdamW(torch.optim.Optimizer):
    """Single-device version."""

    def __init__(self, param_groups: list[ParamGroup], bf16_orthogonalization: bool) -> None:
        super().__init__(param_groups, defaults={})
        self.bf16 = bf16_orthogonalization
        self._hyper = _HyperTensors()

    def _step_adamw(self, group: ParamGroup) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            adamw_step_fused(p, p.grad, state["exp_avg"], state["exp_avg_sq"], *self._hyper.adamw(group, state["step"]))

    def _step_muon(self, group: ParamGroup) -> None:
        params: list[Tensor] = group["params"]
        p = params[0]
        momentum_buffer, second_buffer, red_dim = _muon_state(self.state[p], len(params), p.shape, p)
        stacked_grads = torch.stack([q.grad for q in params])
        stacked_params = torch.stack(params)
        muon_step_fused(
            stacked_grads, stacked_params, momentum_buffer, second_buffer,
            *self._hyper.muon(group, p.shape), group["ns_steps"], red_dim, self.bf16,
        )  # fmt: skip
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self) -> None:  # type: ignore[override]
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            elif group["kind"] == "muon":
                self._step_muon(group)
            else:
                raise ValueError(f"unknown optimizer kind: {group['kind']}")


class DistMuonAdamW(torch.optim.Optimizer):
    """Multi-GPU version: averages gradients across ranks itself, so the model is not wrapped in DDP.

    AdamW parameters with at least 1024 elements are reduce-scattered along dim 0 (which must divide by the world
    size), updated shard-wise and all-gathered; smaller ones are all-reduced and updated everywhere. Each Muon group's
    stack of K matrices is split into ceil(K / world) chunks, one per rank. Communication is launched asynchronously in
    three phases (reduce, compute + gather, finish) so it overlaps with the updates.
    """

    def __init__(self, param_groups: list[ParamGroup], bf16_orthogonalization: bool) -> None:
        super().__init__(param_groups, defaults={})
        self.bf16 = bf16_orthogonalization
        self._hyper = _HyperTensors()

    def _reduce_adamw(self, group: ParamGroup, world_size: int) -> dict[Tensor, dict[str, Any]]:
        infos: dict[Tensor, dict[str, Any]] = {}
        for p in group["params"]:
            grad = p.grad
            if p.numel() < 1024:
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                infos[p] = {"future": future, "grad_slice": grad, "is_small": True}
            else:
                assert grad.shape[0] % world_size == 0, f"dim 0 ({grad.shape[0]}) must divide by world size ({world_size})"
                grad_slice = torch.empty_like(grad[: grad.shape[0] // world_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                infos[p] = {"future": future, "grad_slice": grad_slice, "is_small": False}
        return infos

    def _reduce_muon(self, group: ParamGroup, world_size: int) -> dict[str, Any]:
        params = group["params"]
        chunk_size = (len(params) + world_size - 1) // world_size
        p = params[0]
        stacked_grads = torch.zeros(chunk_size * world_size, *p.shape, dtype=p.dtype, device=p.device)
        stacked_grads[: len(params)].copy_(torch.stack([q.grad for q in params]))
        grad_chunk = torch.empty(chunk_size, *p.shape, dtype=p.dtype, device=p.device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()
        return {"future": future, "grad_chunk": grad_chunk, "stacked_grads": stacked_grads, "chunk_size": chunk_size}

    def _compute_adamw(self, group: ParamGroup, infos: dict[Tensor, dict[str, Any]], gathers: list[dict[str, Any]], rank: int, world_size: int) -> None:
        for p in group["params"]:
            info = infos[p]
            info["future"].wait()
            rows = p.shape[0] // world_size
            p_slice = p if info["is_small"] else p[rank * rows : (rank + 1) * rows]
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p_slice)
                state["exp_avg_sq"] = torch.zeros_like(p_slice)
            state["step"] += 1
            adamw_step_fused(p_slice, info["grad_slice"], state["exp_avg"], state["exp_avg_sq"], *self._hyper.adamw(group, state["step"]))
            if not info["is_small"]:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gathers.append({"future": future, "params": None})

    def _compute_muon(self, group: ParamGroup, info: dict[str, Any], gathers: list[dict[str, Any]], rank: int) -> None:
        info["future"].wait()
        params = group["params"]
        chunk_size = info["chunk_size"]
        p = params[0]
        start = rank * chunk_size
        owned = min(chunk_size, max(0, len(params) - start))
        momentum_buffer, second_buffer, red_dim = _muon_state(self.state[p], chunk_size, p.shape, p)
        updated = torch.zeros(chunk_size, *p.shape, dtype=p.dtype, device=p.device)
        if owned > 0:
            stacked = torch.stack(params[start : start + owned])
            muon_step_fused(
                info["grad_chunk"][:owned], stacked, momentum_buffer[:owned], second_buffer[:owned],
                *self._hyper.muon(group, p.shape), group["ns_steps"], red_dim, self.bf16,
            )  # fmt: skip
            updated[:owned].copy_(stacked)
        stacked_params = info["stacked_grads"]  # reuse the reduce buffer as the gather output
        future = dist.all_gather_into_tensor(stacked_params, updated, async_op=True).get_future()
        gathers.append({"future": future, "stacked_params": stacked_params, "params": params})

    @torch.no_grad()
    def step(self) -> None:  # type: ignore[override]
        rank, world_size = dist.get_rank(), dist.get_world_size()
        reduces = [self._reduce_adamw(g, world_size) if g["kind"] == "adamw" else self._reduce_muon(g, world_size) for g in self.param_groups]
        gathers: list[dict[str, Any]] = []
        for group, info in zip(self.param_groups, reduces):
            if group["kind"] == "adamw":
                self._compute_adamw(group, info, gathers, rank, world_size)
            else:
                self._compute_muon(group, info, gathers, rank)
        for gather in gathers:
            gather["future"].wait()
            if gather["params"] is not None:
                params = gather["params"]
                torch._foreach_copy_(params, list(gather["stacked_params"][: len(params)].unbind(0)))
