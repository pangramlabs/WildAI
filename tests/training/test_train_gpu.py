"""End-to-end training on one GPU (compiled, FP8 where supported): the loss on the training data falls."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.training.helpers import SMALL_RECIPE
from wildai.training.checkpoint import load_model
from wildai.training.data import TrainBatches
from wildai.training.mixture import Arm, Pools, RunSpec, plan_run
from wildai.training.trainer import build_model, train

pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")]


def test_training_lowers_the_loss(pools: Pools, tmp_path: Path) -> None:
    recipe = SMALL_RECIPE.model_copy(update={"fp8": True, "compile": True, "device_batch_size": 4})
    spec = RunSpec(name="gpu-test", depth=4, arm=Arm.AI, human_steps=80, steps=120, seed=5)
    path = train(spec, recipe, pools, tmp_path, log_every=40)
    device = torch.device("cuda")
    trained, meta = load_model(path, device)
    assert meta["data"]["ai_tokens"] * 2 == meta["data"]["human_tokens"]
    torch.manual_seed(spec.seed)
    initial = build_model(recipe, spec.depth, device)
    micro = next(iter(TrainBatches(plan_run(spec, pools, recipe), pools, recipe.sequence_len, recipe.rows_per_step(4), 8, 0, 1, device)))[0]
    with torch.no_grad():
        before = initial(micro.inputs, micro.targets).item()
        after = trained(micro.inputs, micro.targets).item()
    assert after < before - 1.0
