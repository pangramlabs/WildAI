"""Quantities derived from a fitted law: the cost of not filtering, the value of an AI token, the loss-minimising AI
share, and the compute-equivalent gain (CEG) of a mix against a reference mix.

At a fixed model size compute is proportional to training tokens (C ~ 6 N D), so a ratio of token counts that reach the
same loss is a compute ratio (Davidson et al.'s compute-equivalent gain).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from wildai.laws.forms import FittedLaw

# AI share of the tokens of our 2026 web crawl (22.3 %), the mix the filtering runs train on.
WEB_2026_AI_SHARE = 0.22315880974511895


def cost_of_not_filtering(model: FittedLaw, n_params: float, human_tpp: float, share: float = WEB_2026_AI_SHARE) -> float:
    """Tokens a mix with AI share ``share`` needs to reach the loss of training on human_tpp * N human tokens alone,
    divided by those human tokens: the compute-equivalent gain of filtering the mix (inf if the mix never gets there)."""
    total = human_tpp * n_params
    target = model.loss1(n_params, total, 0.0)

    def excess(log_tokens: float) -> float:
        tokens = math.exp(log_tokens)
        return model.loss1(n_params, (1 - share) * tokens, share * tokens) - target

    grid = np.geomspace(0.05 * total, 200 * total, 400)
    losses = model.loss(n_params, (1 - share) * grid, share * grid)
    reached = np.flatnonzero(losses <= target)
    if not len(reached):
        return math.inf
    k = int(reached[0])
    if k == 0:
        return float(grid[0] / total)
    return math.exp(brentq(excess, math.log(grid[k - 1]), math.log(grid[k]), xtol=1e-13)) / total


def ai_token_value(model: FittedLaw, n_params: float, human_tokens: float, ratio: np.ndarray | float) -> np.ndarray:
    """Average value of the AI tokens added at ratio r, in human tokens: the loss change they cause divided by the loss
    change the same number of fresh human tokens would cause, [L(H, 0) - L(H, rH)] / [L(H, 0) - L((1 + r) H, 0)]."""
    ratio = np.atleast_1d(np.asarray(ratio, float))
    base = model.loss1(n_params, human_tokens, 0.0)
    with_ai = model.loss(n_params, human_tokens, ratio * human_tokens)
    with_human = model.loss(n_params, (1 + ratio) * human_tokens, 0.0)
    return (base - with_ai) / (base - with_human)


def optimal_ratio(model: FittedLaw, n_params: float, human_tokens: float, r_max: float = 64.0) -> float:
    """The AI-to-human ratio in [0, r_max] that minimises the predicted loss with the human corpus fixed.

    A 700-point grid (0 and a log grid from 1e-3) finds the global minimum (the loss can have two local minima), then a
    bounded search between the neighbouring grid points refines it, in log r away from zero.
    """
    grid = np.concatenate([[0.0], np.geomspace(1e-3, r_max, 700)])
    losses = model.loss(n_params, human_tokens, grid * human_tokens)
    k = int(np.argmin(losses))
    hi = grid[min(k + 1, len(grid) - 1)]

    def loss_at(r: float) -> float:
        return model.loss1(n_params, human_tokens, r * human_tokens)

    if k <= 1:
        result = minimize_scalar(loss_at, bounds=(0.0, hi), method="bounded", options={"xatol": 1e-12})
        r = float(result.x)
    else:
        result = minimize_scalar(lambda z: loss_at(math.exp(z)), bounds=(math.log(grid[k - 1]), math.log(hi)), method="bounded", options={"xatol": 1e-10})
        r = float(math.exp(result.x))
    return r if result.fun <= losses[k] else float(grid[k])


def optimal_share(model: FittedLaw, n_params: float, human_tokens: float, r_max: float = 64.0) -> float:
    """The loss-minimising AI share of all training tokens, r_opt / (1 + r_opt)."""
    r = optimal_ratio(model, n_params, human_tokens, r_max)
    return r / (1 + r)


def largest_loss_reduction(model: FittedLaw, n_params: float, human_tokens: float, r_max: float = 64.0) -> float:
    """The largest relative reduction in loss that adding AI text to a fixed human corpus achieves (0 if none)."""
    base = model.loss1(n_params, human_tokens, 0.0)
    best = model.loss1(n_params, human_tokens, optimal_ratio(model, n_params, human_tokens, r_max) * human_tokens)
    return max(0.0, 1.0 - best / base)


InversionStatus = Literal["ok", "multiple_crossings", "below_range", "above_range"]


@dataclass(frozen=True)
class ReferenceMix:
    """Loss of a reference mix (fixed AI share) at a fixed size, as a function of its training tokens.

    Tokens are searched only over the human budgets the law was fitted on, so no conversion extrapolates the reference.
    """

    model: FittedLaw
    n_params: float
    human_tpp_range: tuple[float, float]
    share: float = WEB_2026_AI_SHARE
    points: int = 1025

    def loss(self, log_tokens: float) -> float:
        tokens = math.exp(log_tokens)
        return self.model.loss1(self.n_params, (1 - self.share) * tokens, self.share * tokens)

    @cached_property
    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Log tokens and the reference loss on the search grid."""
        lo, hi = self.human_tpp_range
        log_tokens = np.linspace(math.log(self.n_params * lo / (1 - self.share)), math.log(self.n_params * hi / (1 - self.share)), self.points)
        tokens = np.exp(log_tokens)
        return log_tokens, self.model.loss(self.n_params, (1 - self.share) * tokens, self.share * tokens)

    def tokens_for(self, loss: float) -> tuple[float | None, InversionStatus]:
        """Training tokens of the reference mix that reach ``loss``, with the reason when there is no single answer."""
        log_tokens, losses = self.grid
        residual = losses - loss
        roots = [float(z) for z, y in zip(log_tokens, residual) if abs(y) < 1e-13]
        for i in np.flatnonzero(residual[:-1] * residual[1:] < 0):
            roots.append(brentq(lambda z: self.loss(z) - loss, log_tokens[i], log_tokens[i + 1], xtol=1e-12))
        unique: list[float] = []
        for z in sorted(roots):
            if not unique or abs(z - unique[-1]) > 1e-8:
                unique.append(z)
        if len(unique) > 1:
            return None, "multiple_crossings"
        if not unique:
            return None, "below_range" if loss < losses.min() else "above_range"
        return math.exp(unique[0]), "ok"

    def ceg(self, loss: float, tokens: float) -> tuple[float | None, InversionStatus]:
        """Compute-equivalent gain over the reference mix of a run that reached ``loss`` with ``tokens`` training tokens."""
        equivalent, status = self.tokens_for(loss)
        return (equivalent / tokens if equivalent is not None else None), status
