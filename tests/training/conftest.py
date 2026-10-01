"""Shared fixtures: tokenized synthetic human and AI pools."""

from __future__ import annotations

import pytest

from tests.training.helpers import byte_tokenizer, write_raw_pool
from wildai.training.mixture import Pools
from wildai.training.pools import build_store


@pytest.fixture(scope="session")
def pools(tmp_path_factory: pytest.TempPathFactory) -> Pools:
    root = tmp_path_factory.mktemp("pools")
    tokenizer = byte_tokenizer()
    write_raw_pool(root / "raw/human", "h", 3000, seed=1)
    write_raw_pool(root / "raw/ai", "a", 2000, seed=2)
    build_store([root / "raw/human"], root / "data/human", tokenizer, num_threads=1)
    build_store([root / "raw/ai"], root / "data/ai", tokenizer, num_threads=1)
    return Pools.from_directory(root / "data")
