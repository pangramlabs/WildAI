# Adapted from nanochat (https://github.com/karpathy/nanochat), Copyright (c) 2025 Andrej Karpathy, MIT License; see NOTICE.
"""Per-step learning-rate, Muon-momentum and Muon weight-decay schedules (nanochat's `base_train`)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from wildai.training.recipe import Recipe


@dataclass(frozen=True)
class Schedule:
    num_steps: int
    warmup_steps: int
    warmdown_ratio: float
    final_lr_frac: float
    momentum_warmup_steps: int
    weight_decay: float

    @classmethod
    def from_recipe(cls, recipe: Recipe, num_steps: int, weight_decay: float) -> Schedule:
        return cls(num_steps, recipe.warmup_steps, recipe.warmdown_ratio, recipe.final_lr_frac, recipe.muon_momentum_warmup_steps, weight_decay)

    @property
    def warmdown_steps(self) -> int:
        return round(self.warmdown_ratio * self.num_steps)

    def lr_multiplier(self, step: int) -> float:
        """Linear warmup, constant, then linear decay to `final_lr_frac` over the last `warmdown_ratio` of steps."""
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        if step <= self.num_steps - self.warmdown_steps:
            return 1.0
        progress = (self.num_steps - step) / self.warmdown_steps
        return progress + (1 - progress) * self.final_lr_frac

    def muon_momentum(self, step: int) -> float:
        """0.85 -> 0.97 over the first 400 steps, then 0.97 -> 0.90 during the warmdown (warmup takes precedence)."""
        warmdown_start = self.num_steps - self.warmdown_steps
        if step < self.momentum_warmup_steps:
            frac = step / self.momentum_warmup_steps
            return (1 - frac) * 0.85 + frac * 0.97
        if step >= warmdown_start:
            progress = (step - warmdown_start) / self.warmdown_steps
            return 0.97 * (1 - progress) + 0.90 * progress
        return 0.97

    def muon_weight_decay(self, step: int) -> float:
        """Cosine decay from the scaled weight decay to zero."""
        return self.weight_decay * 0.5 * (1 + math.cos(math.pi * step / self.num_steps))
