"""Pool stores: key order, exact token spans, and partial stores as prefixes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tests.training.helpers import byte_tokenizer, write_raw_pool
from wildai.data.hashing import keyed_hash
from wildai.training.pools import SELECTION_KEY, PoolStore, build_store


def test_store_holds_every_document_in_key_order(tmp_path: Path) -> None:
    write_raw_pool(tmp_path / "raw", "d", 500, seed=4)
    build_store([tmp_path / "raw"], tmp_path / "store", byte_tokenizer(), num_threads=1)
    store = PoolStore(tmp_path / "store")
    raw = pq.read_table(tmp_path / "raw").to_pylist()
    texts = {row["id"]: row["text"] for row in raw}
    ids = pq.read_table(tmp_path / "store/index.parquet", columns=["id"])["id"].to_pylist()
    assert sorted(ids) == sorted(texts) and store.info.documents == 500
    assert [keyed_hash(i, SELECTION_KEY) for i in ids] == store.keys.tolist() == sorted(store.keys.tolist())
    for row in (0, 17, 499):
        assert store.document(row).tolist() == [256, *texts[ids[row]].encode()]
    assert store.info.tokens == int(store.lengths.sum()) == len(store.tokens)


def test_partial_store_is_a_prefix_of_the_full_store(tmp_path: Path) -> None:
    write_raw_pool(tmp_path / "raw", "d", 800, seed=5)
    build_store([tmp_path / "raw"], tmp_path / "full", byte_tokenizer(), num_threads=1)
    build_store([tmp_path / "raw"], tmp_path / "part", byte_tokenizer(), fraction=0.3, num_threads=1)
    full, part = PoolStore(tmp_path / "full"), PoolStore(tmp_path / "part")
    count = len(part.keys)
    assert 150 < count < 330
    assert np.array_equal(part.keys, full.keys[:count]) and np.array_equal(part.lengths, full.lengths[:count])
