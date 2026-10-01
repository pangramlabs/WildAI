"""Shared fixtures for the hub tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import tiktoken

from wildai.hub.card import CardSettings
from wildai.hub.catalog import Catalog
from wildai.hub.export import Exporter
from wildai.hub.tokenizer import to_hf_tokenizer
from wildai.laws.data import Run, Study

from .helpers import CORPUS, EDGE_CASES, TinyCheckpoint, train_encoding, write_checkpoint


@pytest.fixture(scope="session")
def encoding() -> tiktoken.Encoding:
    return train_encoding()


@pytest.fixture(scope="session")
def sample_texts() -> list[str]:
    """Corpus documents plus edge cases: whitespace runs, digits, non-Latin scripts, emoji, code, special-token strings."""
    return [*dict.fromkeys(CORPUS), *EDGE_CASES]


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory, encoding: tiktoken.Encoding) -> TinyCheckpoint:
    """Depth 4, width 64, two 32-dim heads; context 512 gives windows of 128, 128, 128, 512 tokens."""
    return write_checkpoint(tmp_path_factory.mktemp("ckpt"), depth=4, width=64, n_head=2, vocab_size=encoding.n_vocab)


@dataclass(frozen=True)
class StagedRelease:
    root: Path  # staging root: <root>/<repo name>/<model name>/
    repo_dir: Path
    names: tuple[str, ...]
    repo_id: str


@pytest.fixture(scope="session")
def staged_release(tmp_path_factory: pytest.TempPathFactory, tiny_checkpoint: TinyCheckpoint, encoding: tiktoken.Encoding) -> StagedRelease:
    """A two-model repository (a control and an AI run of one group) exported from the tiny checkpoint."""
    common = {"group": "g900", "size": "19.9m", "depth": tiny_checkpoint.depth, "n_params": 1_000_000, "seed": 7, "split": "fit"}
    control = Run(
        name="19.9m-g900-control", arm="control", added_ratio=0.0, human_tokens=4e7, ai_tokens=0.0, total_tokens=4e7, steps=100, **common
    )
    ai = Run(name="19.9m-g900-ai-r0.5", arm="ai", added_ratio=0.5, human_tokens=4e7, ai_tokens=2e7, total_tokens=6e7, steps=150, **common)
    losses = {(ai.name, "c4"): 1.2345, (ai.name, "fw26"): 1.1, (ai.name, "paloma"): 1.3, (ai.name, "paloma_ptb"): 1.5}
    losses |= {(control.name, "c4"): 1.25}
    settings = CardSettings(dataset="pangram/wildai-data")
    exporter = Exporter(Catalog(Study([control, ai], losses)), to_hf_tokenizer(encoding), settings)
    root = tmp_path_factory.mktemp("staging")
    for run in (control, ai):
        exporter.export(run.name, tiny_checkpoint.model_path, root)
    exporter.write_repo_root(root)
    return StagedRelease(root=root, repo_dir=root / "WildAI-models", names=(control.name, ai.name), repo_id="pangram/WildAI-models")
