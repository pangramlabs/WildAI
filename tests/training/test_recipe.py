"""The recipe's derived quantities and schedules."""

from __future__ import annotations

import math

import pytest

from wildai.training.recipe import Recipe
from wildai.training.registry import load_models
from wildai.training.schedule import Schedule

RECIPE = Recipe()


def test_param_counts_match_the_papers_n() -> None:
    n_by_depth = {row.depth: row.n_params for row in load_models().values()}
    assert all(RECIPE.param_counts(depth).paper_n == n for depth, n in n_by_depth.items())
    assert RECIPE.gpt_config(26).n_embd == 1664 and RECIPE.gpt_config(26).n_head == 13


@pytest.mark.parametrize(("depth", "tokens"), [(4, 2**18), (6, 2**18), (9, 2**19), (16, 2**19), (20, 2**20), (26, 2**20)])
def test_tokens_per_step(depth: int, tokens: int) -> None:
    assert RECIPE.batch_tokens(depth) == tokens


def test_rescaling_from_the_depth_12_reference() -> None:
    assert RECIPE.muon_weight_decay(12) == pytest.approx(0.28)
    assert RECIPE.lr_scale(4) == pytest.approx(math.sqrt(0.5))
    ratio = RECIPE.param_counts(12).scaling_params / RECIPE.param_counts(20).scaling_params
    assert RECIPE.muon_weight_decay(20) == pytest.approx(0.28 * math.sqrt(2) * ratio)


def test_schedule() -> None:
    schedule = Schedule.from_recipe(RECIPE, num_steps=1000, weight_decay=0.2)
    assert schedule.lr_multiplier(0) == pytest.approx(1 / 40)
    assert schedule.lr_multiplier(39) == 1.0
    assert schedule.lr_multiplier(350) == 1.0  # warmdown covers the last 650 steps
    assert schedule.lr_multiplier(1000) == pytest.approx(0.05)
    assert schedule.lr_multiplier(675) == pytest.approx(0.5 + 0.5 * 0.05)
    assert schedule.muon_momentum(0) == pytest.approx(0.85)
    assert schedule.muon_momentum(400) == pytest.approx(0.97 * (1 - 50 / 650) + 0.90 * 50 / 650)
    assert schedule.muon_weight_decay(0) == pytest.approx(0.2)
    assert schedule.muon_weight_decay(500) == pytest.approx(0.1)
