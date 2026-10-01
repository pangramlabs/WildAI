"""Document-bounded Paloma scoring."""

from __future__ import annotations

import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from wildai.evaluation.paloma import score_source, score_texts, windows
from wildai.evaluation.runtime import LoadedModel


def test_windows_predict_every_token_once() -> None:
    tokens = list(range(100))
    spans = list(windows(tokens, 32))
    targets = [t for w in spans for t in w[1:]]
    assert targets == tokens[1:]
    assert all(len(w) <= 33 for w in spans)
    assert list(windows([7], 32)) == []


def test_documents_are_scored_separately(loaded: LoadedModel) -> None:
    texts = ["hello world", "a" * 70]
    score = score_texts(loaded, texts, batch_size=3)
    nats = 0.0
    with torch.no_grad():
        for text in texts:
            for w in windows([256, *text.encode()], 32):
                ids = torch.tensor([w])
                nats += loaded.model(ids[:, :-1], ids[:, 1:], loss_reduction="none").sum().item()
    assert score.bytes == sum(len(t) for t in texts)
    assert score.bpb == pytest.approx(nats / (math.log(2) * score.bytes), rel=1e-5)


def test_source_is_the_mean_over_domains(loaded: LoadedModel, tmp_path: Path) -> None:
    directory = tmp_path / "source"
    directory.mkdir()
    pq.write_table(pa.table({"text": ["abc" * 10, "xyz" * 30, "hello"], "domain": ["b", "a", "a"]}), directory / "part.parquet")
    score = score_source(loaded, directory, batch_size=8)
    assert list(score.domains) == ["a", "b"]
    assert score.macro_bpb == pytest.approx((score.domains["a"].bpb + score.domains["b"].bpb) / 2)
