"""EditLens preselection, natural draw and pool building on tiny fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.data.builders import candidates, pangram_sidecar, web_documents, write
from wildai.data.arrow_schema import arrow_schema
from wildai.data.hashing import keyed_hash
from wildai.data.parquet_io import read_dataset
from wildai.data.pools import build_pools, pool_name
from wildai.data.pools_config import EditLensPreselection, NaturalDraw, PoolSettings
from wildai.data.preselect import hash_threshold, human_quotas, preselect_editlens, preselect_natural
from wildai.data.schema import CandidateDocument, PoolDocument


def editlens_sidecar(ids: list[str], buckets: list[int]) -> pa.Table:
    return pa.table({"id": ids, "editlens_bucket": pa.array(buckets, pa.int64()), "editlens_score": [b / 3 for b in buckets],
                     "editlens_probs": [[0.25] * 4] * len(ids), "editlens_windows": [1] * len(ids)})


def test_human_quotas_follow_ai_tokens() -> None:
    quotas = human_quotas({"a": 300, "b": 100}, {"a": 10_000, "b": 20}, budget=400)
    assert quotas == {"a": 300, "b": 20}  # b is capped by what it holds
    with pytest.raises(ValueError):
        human_quotas({"a": 0}, {"a": 1}, 1)


def test_hash_threshold_reaches_the_quota() -> None:
    hashes = np.array([5, 1, 3, 9], dtype=np.uint64)
    tokens = np.array([10, 10, 10, 10])
    assert hash_threshold(hashes, tokens, 15) == 3 and hash_threshold(hashes, tokens, 100) == 9
    assert hash_threshold(hashes, tokens, 0) is None


def test_preselect_editlens(tmp_path: Path) -> None:
    for dump, n in (("CC-MAIN-2025-30", 40), ("CC-MAIN-2026-25", 40)):
        ids = [f"{dump}-{i}" for i in range(n)]
        buckets = [i % 4 for i in range(n)]  # ten of each bucket
        write(web_documents(ids, dump), tmp_path / "docs" / dump / "part-00000.parquet")
        write(editlens_sidecar(ids, buckets), tmp_path / "editlens" / dump / "part-00000.parquet")
    settings = EditLensPreselection(human_token_budget=10 * 121, key="k")  # about five human documents per crawl
    summary = preselect_editlens(tmp_path / "docs", tmp_path / "editlens", tmp_path / "out", settings)
    for dump in ("CC-MAIN-2025-30", "CC-MAIN-2026-25"):
        table = read_dataset(tmp_path / "out" / dump)
        assert table.schema == arrow_schema(CandidateDocument)
        buckets = table["editlens_bucket"].to_pylist()
        assert 1 not in buckets and buckets.count(2) == buckets.count(3) == 10  # every AI candidate, no bucket 1
        assert summary[dump]["ai"] == 20 and 4 <= summary[dump]["human"] <= 6
        assert set(table.filter(pa.compute.equal(table["editlens_bucket"], 0))["selection"].to_pylist()) == {"editlens_human"}


def test_preselect_natural_is_label_blind(tmp_path: Path) -> None:
    ids = [f"d{i}" for i in range(50)]
    write(web_documents(ids, "CC-MAIN-2026-25"), tmp_path / "docs" / "CC-MAIN-2026-25" / "part-00000.parquet")
    write(web_documents(["x"], "CC-MAIN-2025-30"), tmp_path / "docs" / "CC-MAIN-2025-30" / "part-00000.parquet")
    written = preselect_natural(tmp_path / "docs", tmp_path / "out", NaturalDraw(dumps=["CC-MAIN-2026-25"], documents=7, key="n"))
    table = read_dataset(tmp_path / "out")
    assert written == 7 and set(table["selection"].to_pylist()) == {"natural"}
    assert sorted(table["id"].to_pylist()) == sorted(sorted(ids, key=lambda i: keyed_hash(i, "n"))[:7])
    assert table["editlens_bucket"].null_count == 7


@pytest.fixture
def labeled(tmp_path: Path) -> tuple[Path, Path]:
    """Two candidate shards; ``dup`` appears in both (first as AI) and ``bad`` could not be labeled."""

    first = ["a1", "a2", "h1", "m1", "dup"]
    second = ["h2", "dup", "bad", "a3"]
    write(candidates(first), tmp_path / "cand" / "X" / "part-00000.parquet")
    write(candidates(second), tmp_path / "cand" / "X" / "part-00001.parquet")
    write(pangram_sidecar(first, ["AI", "AI", "Human", "Mixed", "AI"]), tmp_path / "pangram" / "X" / "part-00000.parquet")
    write(pangram_sidecar(second, ["Human", "Human", None, "AI"]), tmp_path / "pangram" / "X" / "part-00001.parquet")
    return tmp_path / "cand", tmp_path / "pangram"


def test_build_pools_splits_dedups_and_sorts(labeled: tuple[Path, Path], tmp_path: Path) -> None:
    settings = PoolSettings(key="pool", bins=4, rows_per_shard=2)
    summary = build_pools(*labeled, tmp_path / "pools", settings, natural=False)
    assert summary["duplicate_ids_dropped"] == 1
    pools = {name: read_dataset(tmp_path / "pools" / name) for name in ("human", "ai", "mixed", "_unlabeled")}
    assert sorted(pools["ai"]["id"].to_pylist()) == ["a1", "a2", "a3", "dup"]  # the first occurrence (AI) wins
    assert sorted(pools["human"]["id"].to_pylist()) == ["h1", "h2"]
    assert pools["mixed"]["id"].to_pylist() == ["m1"] and pools["_unlabeled"]["id"].to_pylist() == ["bad"]
    for name in ("human", "ai", "mixed"):
        table = pools[name]
        assert table.schema == arrow_schema(PoolDocument)
        assert set(table["label"].to_pylist()) == {name}
        hashes = table["sampling_hash"].to_pylist()
        assert hashes == sorted(hashes) and hashes[0] == keyed_hash(table["id"][0].as_py(), "pool")
    assert (tmp_path / "pools" / "ai" / "_pool.json").exists() and summary["pools"]["ai"]["documents"] == 4


def test_build_pools_natural_leaves_out_mixed(labeled: tuple[Path, Path], tmp_path: Path) -> None:
    summary = build_pools(*labeled, tmp_path / "pools", PoolSettings(key="pool", bins=2), natural=True)
    natural = read_dataset(tmp_path / "pools" / "natural")
    assert sorted(natural["id"].to_pylist()) == ["a1", "a2", "a3", "dup", "h1", "h2"]
    assert summary["left_out"] == 1 and not (tmp_path / "pools" / "mixed").exists()
    assert pool_name(None, natural=True) == "_unlabeled_natural" and pool_name("Mixed", natural=False) == "mixed"


def test_pool_rebuild_replaces_old_shards(labeled: tuple[Path, Path], tmp_path: Path) -> None:
    build_pools(*labeled, tmp_path / "pools", PoolSettings(key="pool", bins=2, rows_per_shard=1), natural=False)
    build_pools(*labeled, tmp_path / "pools", PoolSettings(key="pool", bins=2, rows_per_shard=10), natural=False)
    assert len(list((tmp_path / "pools" / "ai").glob("*.parquet"))) == 1
    assert pq.read_table(tmp_path / "pools" / "ai" / "part-00000.parquet").num_rows == 4
