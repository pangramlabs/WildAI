"""Training batches do not depend on the number of GPUs."""

from __future__ import annotations

import torch

from tests.training.helpers import SMALL_RECIPE
from wildai.training.data import TrainBatches
from wildai.training.mixture import Arm, Pools, RunSpec, plan_run


def step_rows(pools: Pools, world_size: int, device_batch_size: int) -> list[torch.Tensor]:
    plan = plan_run(RunSpec(name="x", depth=4, arm=Arm.AI, human_steps=20, steps=30), pools, SMALL_RECIPE)
    rows_per_step = SMALL_RECIPE.rows_per_step(4)
    per_rank = [list(TrainBatches(plan, pools, SMALL_RECIPE.sequence_len, rows_per_step, device_batch_size, rank, world_size, torch.device("cpu"))) for rank in range(world_size)]
    steps = []
    for step in range(plan.steps):
        micro = [m for rank in range(world_size) for m in per_rank[rank][step]]
        steps.append(torch.cat([torch.cat([m.inputs, m.targets[:, -1:]], 1) for m in micro]))
    return steps


def test_data_order_is_independent_of_world_size(pools: Pools) -> None:
    single = step_rows(pools, world_size=1, device_batch_size=8)
    for world_size, batch in [(2, 2), (4, 1), (8, 1)]:
        assert all(torch.equal(a, b) for a, b in zip(single, step_rows(pools, world_size, batch)))


def test_targets_are_inputs_shifted(pools: Pools) -> None:
    plan = plan_run(RunSpec(name="x", depth=4, arm=Arm.CONTROL, human_steps=5, steps=5), pools, SMALL_RECIPE)
    batches = TrainBatches(plan, pools, SMALL_RECIPE.sequence_len, 8, 4, 0, 1, torch.device("cpu"))
    for micro_batches in batches:
        for micro in micro_batches:
            assert micro.inputs.shape == (4, 64)
            assert torch.equal(micro.inputs[:, 1:], micro.targets[:, :-1])
            assert (micro.inputs[:, 0] == 256).all()  # every row starts with <|bos|>
