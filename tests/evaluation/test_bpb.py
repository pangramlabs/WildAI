"""Packed bits-per-byte scoring."""

from __future__ import annotations

import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from wildai.evaluation.bpb import BpbSuite, score_target
from wildai.evaluation.runtime import CONFIG_DIR, LoadedModel


def write_set(directory: Path, texts: list[str]) -> Path:
    directory.mkdir(parents=True)
    pq.write_table(pa.table({"text": texts}), directory / "shard_00000.parquet", row_group_size=50)
    return directory


def test_paper_budgets() -> None:
    suite = BpbSuite.load(CONFIG_DIR / "bpb_targets.json")
    assert suite.batch_size == 16
    assert suite.targets == {"c4": 480, "fw22": 480, "fw26": 369, "fw26_human": 293, "fw26_ai": 66, "cosmopedia": 564}


def test_score_equals_a_direct_computation(loaded: LoadedModel, tmp_path: Path) -> None:
    # documents of exactly one row (32 letters + BOS) pack one per row, in order
    texts = [chr(97 + i % 26) * 32 for i in range(40)]
    score = score_target(loaded, write_set(tmp_path / "set", texts), steps=5, batch_size=4)
    ids = torch.tensor([[256, *text.encode()] for text in texts[:20]])
    with torch.no_grad():
        nats = loaded.model(ids[:, :-1], ids[:, 1:], loss_reduction="none").sum().item()
    assert score.bytes == 20 * 32 and score.tokens == 20 * 32
    assert score.bpb == pytest.approx(nats / (math.log(2) * 20 * 32), rel=1e-6)


def test_each_document_is_scored_at_most_once(loaded: LoadedModel, tmp_path: Path) -> None:
    directory = write_set(tmp_path / "set", [chr(97 + i % 26) * (5 + i % 40) for i in range(60)])
    with pytest.raises(ValueError, match="fills only"):
        score_target(loaded, directory, steps=10, batch_size=4)
