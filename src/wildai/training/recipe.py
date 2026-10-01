# Adapted from nanochat (https://github.com/karpathy/nanochat), Copyright (c) 2025 Andrej Karpathy, MIT License; see NOTICE.
"""The training recipe shared by every WildAI model, as typed defaults.

Only width and depth vary between model sizes: `n_embd = 64 * depth` rounded up to a multiple of the 128-wide heads.
Batch size (tokens per optimizer step) grows with depth, and the learning rates and Muon weight decay are rescaled
from a depth-12, 2**19-token reference exactly as nanochat's `base_train` does:

- every learning rate except the smear gate's is multiplied by sqrt(B / 2**19);
- AdamW rates are further multiplied by (n_embd / 768) ** -0.5;
- Muon weight decay is `weight_decay * sqrt(B / 2**19) * P_12 / P_depth`, where P counts transformer matrices plus
  the output head.
"""

from __future__ import annotations

import math

import torch
from pydantic import BaseModel, ConfigDict

from wildai.training.model import GPT, GPTConfig, ParamCounts
from wildai.training.optim import DistMuonAdamW, MuonAdamW, ParamGroup


class Recipe(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # architecture
    sequence_len: int = 2048
    vocab_size: int = 32768
    aspect_ratio: int = 64
    head_dim: int = 128
    window_pattern: str = "SSSL"
    # batch: tokens per optimizer step; None uses `default_tokens_per_step(depth)`
    tokens_per_step: int | None = None
    device_batch_size: int = 8  # sequences per GPU per micro-step; only affects memory, not the maths
    # optimizer
    embedding_lr: float = 0.3
    unembedding_lr: float = 0.008
    matrix_lr: float = 0.02
    scalar_lr: float = 0.5
    smear_lr: float = 0.2
    weight_decay: float = 0.28
    reference_depth: int = 12
    reference_tokens_per_step: int = 2**19
    # schedule
    warmup_steps: int = 40
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05
    muon_momentum_warmup_steps: int = 400
    # numerics
    fp8: bool = True
    compile: bool = True

    def gpt_config(self, depth: int) -> GPTConfig:
        n_embd = math.ceil(depth * self.aspect_ratio / self.head_dim) * self.head_dim
        n_head = n_embd // self.head_dim
        return GPTConfig(
            sequence_len=self.sequence_len,
            vocab_size=self.vocab_size,
            n_layer=depth,
            n_head=n_head,
            n_kv_head=n_head,
            n_embd=n_embd,
            window_pattern=self.window_pattern,
        )

    def batch_tokens(self, depth: int) -> int:
        tokens = self.tokens_per_step or default_tokens_per_step(depth)
        assert tokens % self.sequence_len == 0, "tokens per step must be a whole number of sequences"
        return tokens

    def rows_per_step(self, depth: int) -> int:
        return self.batch_tokens(depth) // self.sequence_len

    def param_counts(self, depth: int) -> ParamCounts:
        with torch.device("meta"):
            return GPT(self.gpt_config(depth)).param_counts()

    def lr_scale(self, depth: int) -> float:
        return (self.batch_tokens(depth) / self.reference_tokens_per_step) ** 0.5

    def muon_weight_decay(self, depth: int) -> float:
        reference = self.param_counts(self.reference_depth).scaling_params
        return self.weight_decay * self.lr_scale(depth) * reference / self.param_counts(depth).scaling_params

    def param_groups(self, model: GPT, depth: int) -> list[ParamGroup]:
        """Muon for the transformer matrices (grouped by shape), AdamW for everything else."""
        scale = self.lr_scale(depth)
        adamw_scale = scale * (model.config.n_embd / 768) ** -0.5
        matrices = list(model.transformer.h.parameters())
        smear = [model.smear_gate.weight, model.smear_lambda, model.backout_lambda]
        groups = [
            _adamw([model.lm_head.weight], self.unembedding_lr * adamw_scale, (0.8, 0.96), 0.01),
            _adamw([model.transformer.wte.weight], self.embedding_lr * adamw_scale, (0.8, 0.995), 0.001),
            _adamw(list(model.value_embeds.parameters()), self.embedding_lr * adamw_scale * 0.5, (0.8, 0.995), 0.01),
            _adamw([model.resid_lambdas], self.scalar_lr * scale * 0.01, (0.8, 0.95), 0.05),
            _adamw([model.x0_lambdas], self.scalar_lr * scale, (0.96, 0.95), 0.0),
            _adamw(smear, self.smear_lr, (0.8, 0.95), 0.0),
        ]
        weight_decay = self.muon_weight_decay(depth)
        for shape in sorted({p.shape for p in matrices}):
            params = [p for p in matrices if p.shape == shape]
            groups.append(
                {
                    "kind": "muon",
                    "params": params,
                    "lr": self.matrix_lr * scale,
                    "momentum": 0.95,
                    "ns_steps": 5,
                    "beta2": 0.9,
                    "weight_decay": weight_decay,
                }
            )
        assert sum(len(g["params"]) for g in groups) == len(list(model.parameters())), "every parameter needs a group"
        for group in groups:
            group["initial_lr"] = group["lr"]
        return groups

    def optimizer(self, model: GPT, depth: int, distributed: bool) -> MuonAdamW | DistMuonAdamW:
        bf16 = model.compute_dtype == torch.bfloat16
        groups = self.param_groups(model, depth)
        return DistMuonAdamW(groups, bf16) if distributed else MuonAdamW(groups, bf16)


def _adamw(params: list[torch.nn.Parameter], lr: float, betas: tuple[float, float], weight_decay: float) -> ParamGroup:
    return {"kind": "adamw", "params": params, "lr": lr, "betas": betas, "eps": 1e-10, "weight_decay": weight_decay}


def default_tokens_per_step(depth: int) -> int:
    """2**18 tokens up to depth 6 (19.9M, 35.8M), 2**19 up to depth 16 (86.2M-268M), 2**20 beyond (477M, 973M)."""
    if depth <= 6:
        return 2**18
    if depth <= 16:
        return 2**19
    return 2**20
