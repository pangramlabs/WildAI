# Adapted from nanochat (https://github.com/karpathy/nanochat), Copyright (c) 2025 Andrej Karpathy, MIT License; see NOTICE.
"""The training loop: one run from its `RunSpec`, on one GPU or many (launched with torchrun)."""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from wildai.training.checkpoint import save_checkpoint
from wildai.training.data import TrainBatches
from wildai.training.fp8 import convert_to_fp8, supports_fp8
from wildai.training.mixture import Pools, RunSpec, plan_run
from wildai.training.model import GPT
from wildai.training.recipe import Recipe
from wildai.training.schedule import Schedule


@dataclass(frozen=True)
class DistEnv:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def log(self, message: str) -> None:
        if self.is_main:
            print(message, flush=True)


def init_distributed() -> DistEnv:
    """torchrun sets RANK/LOCAL_RANK/WORLD_SIZE; without them this is a single-process run."""
    if "RANK" in os.environ:
        rank, local_rank, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        return DistEnv(rank, local_rank, world, device)
    return DistEnv(0, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def build_model(recipe: Recipe, depth: int, device: torch.device) -> GPT:
    with torch.device("meta"):
        model = GPT(recipe.gpt_config(depth))
    model.to_empty(device=device)
    model.init_weights()
    return model


def train(spec: RunSpec, recipe: Recipe, pools: Pools, out_dir: Path, log_every: int = 10) -> Path:
    """Train the run and save its final checkpoint; returns the checkpoint directory `out_dir / spec.name`."""
    env = init_distributed()
    torch.manual_seed(spec.seed)  # the weight initialisation is the first use of the RNG on every rank
    torch.set_float32_matmul_precision("high")
    plan = plan_run(spec, pools, recipe)
    summary = plan.summary(recipe.sequence_len)
    env.log(f"run {spec.name}: {summary.model_dump_json()}")

    model = build_model(recipe, spec.depth, env.device)
    if recipe.fp8 and supports_fp8(env.device):
        env.log(f"fp8 matmuls in {convert_to_fp8(model)} linear layers")
    elif recipe.fp8:
        env.log("this GPU has no FP8 support; training in bf16")
    forward = torch.compile(model, dynamic=False) if recipe.compile else model
    optimizer = recipe.optimizer(model, spec.depth, distributed=env.world_size > 1)
    schedule = Schedule.from_recipe(recipe, plan.steps, recipe.muon_weight_decay(spec.depth))
    rows_per_step = recipe.rows_per_step(spec.depth)
    batches = TrainBatches(plan, pools, recipe.sequence_len, rows_per_step, recipe.device_batch_size, env.rank, env.world_size, env.device)
    tokens_per_step = rows_per_step * recipe.sequence_len
    env.log(f"{plan.steps} steps x {tokens_per_step:,} tokens on {env.world_size} GPU(s), {batches.grad_accum_steps} micro-batches each; params {model.param_counts()}")

    stream = iter(batches)
    micro_batches = next(stream)
    smooth_loss = 0.0
    for step in range(plan.steps):
        t0 = time.time()
        for micro in micro_batches:
            loss = forward(micro.inputs, micro.targets)
            train_loss = loss.detach()
            (loss / len(micro_batches)).backward()
        lr_multiplier = schedule.lr_multiplier(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lr_multiplier
            if group["kind"] == "muon":
                group["momentum"] = schedule.muon_momentum(step)
                group["weight_decay"] = schedule.muon_weight_decay(step)
        optimizer.step()
        model.zero_grad(set_to_none=True)
        if step + 1 < plan.steps:
            micro_batches = next(stream)  # packs the next step while the GPU works on this one
        loss_value = train_loss.item()
        dt = time.time() - t0
        smooth_loss = 0.9 * smooth_loss + 0.1 * loss_value
        if step % log_every == 0 or step == plan.steps - 1:
            debiased = smooth_loss / (1 - 0.9 ** (step + 1))
            env.log(f"step {step:06d}/{plan.steps} | loss {debiased:.4f} | lrm {lr_multiplier:.3f} | dt {dt * 1000:.0f}ms | tok/s {tokens_per_step / dt:,.0f}")
        if step == 0:  # nanochat's GC tuning: freeze setup objects, then collect manually
            gc.collect()
            gc.freeze()
            gc.disable()
        elif step % 5000 == 0:
            gc.collect()
    next(stream, None)  # finishes the stream, which checks that every planned token was used
    gc.unfreeze()
    gc.enable()

    directory = out_dir / spec.name
    if env.is_main:
        meta = {
            "run": spec.model_dump(mode="json"),
            "recipe": recipe.model_dump(mode="json"),
            "data": summary.model_dump(mode="json"),
            "world_size": env.world_size,
        }
        env.log(f"saved {save_checkpoint(directory, plan.steps, model, meta)}")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return directory
