"""Schemas, hashing, Parquet I/O, PII anonymization, WARC frames and configs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import pyarrow as pa
import pytest
from pydantic import BaseModel

from wildai.data.arrow_schema import ArrowType, arrow_schema, conform, table_from_models
from wildai.data.commoncrawl import select_frame
from wildai.data.config import CONFIG_DIR, load_config
from wildai.data.hashing import bottom_k, keyed_hash, keyed_hashes
from wildai.data.parquet_io import ShardedWriter, dataset_dirs, list_shards, read_dataset
from wildai.data.pii import anonymize
from wildai.data.schema import PoolDocument


class Inner(BaseModel):
    score: Annotated[float, ArrowType(pa.float32())]


class Example(BaseModel):
    name: str
    kind: Literal["a", "b"]
    count: int | None
    tags: list[str]
    inner: list[Inner]
    hashed: Annotated[int, ArrowType(pa.uint64())]


def test_arrow_schema_maps_every_supported_annotation() -> None:
    schema = arrow_schema(Example)
    assert schema.field("name").type == pa.string() and not schema.field("name").nullable
    assert schema.field("kind").type == pa.string()
    assert schema.field("count").type == pa.int64() and schema.field("count").nullable
    assert schema.field("tags").type == pa.list_(pa.string())
    assert schema.field("inner").type.value_type == pa.struct([pa.field("score", pa.float32(), nullable=False)])
    assert schema.field("hashed").type == pa.uint64()


def test_table_from_models_and_conform_round_trip() -> None:
    rows = [Example(name="x", kind="a", count=None, tags=["t"], inner=[Inner(score=0.5)], hashed=2**63 + 1)]
    table = table_from_models(rows, Example)
    assert table.to_pylist()[0]["hashed"] == 2**63 + 1
    widened = table.append_column("extra", pa.array([1]))
    assert conform(widened, Example).schema == arrow_schema(Example)
    with pytest.raises(ValueError, match="missing"):
        conform(table.drop_columns(["tags"]), Example)


def test_pool_schema_has_the_training_interface() -> None:
    schema = arrow_schema(PoolDocument)
    for name in ("id", "text", "label", "token_count", "sampling_hash"):
        assert name in schema.names
    assert schema.field("sampling_hash").type == pa.uint64()


def test_keyed_hash_is_stable_and_key_dependent() -> None:
    assert keyed_hash("doc", "k") == keyed_hash("doc", "k")
    assert keyed_hash("doc", "k") != keyed_hash("doc", "other")
    assert keyed_hash("doc", "k") == 0xa2344cb14b4ffd8d  # BLAKE2b-128 keyed by "k", first 8 bytes big-endian
    assert list(keyed_hashes(["a", "b"], "k")) == [keyed_hash("a", "k"), keyed_hash("b", "k")]


def test_bottom_k_ignores_input_order() -> None:
    items = [f"id{i}" for i in range(100)]
    forward = bottom_k(items, 10, "key", id_of=str)
    backward = bottom_k(list(reversed(items)), 10, "key", id_of=str)
    assert forward == backward
    assert [keyed_hash(i, "key") for i in forward] == sorted(keyed_hash(i, "key") for i in items)[:10]


def test_sharded_writer_cuts_shards(tmp_path: Path) -> None:
    schema = pa.schema([("x", pa.int64())])
    with ShardedWriter(tmp_path / "out", schema, rows_per_shard=3) as writer:
        writer.write(pa.table({"x": list(range(7))}))
    assert [p.name for p in list_shards(tmp_path / "out")] == ["part-00000.parquet", "part-00001.parquet", "part-00002.parquet"]
    assert read_dataset(tmp_path / "out")["x"].to_pylist() == list(range(7))


def test_dataset_dirs_skips_hidden_directories(tmp_path: Path) -> None:
    table = pa.table({"x": [1]})
    for sub in ("a", "b/c", ".cache/d"):
        with ShardedWriter(tmp_path / sub, table.schema) as writer:
            writer.write(table)
    assert [p.relative_to(tmp_path).as_posix() for p in dataset_dirs(tmp_path)] == ["a", "b/c"]


def test_anonymize_follows_fineweb_rules() -> None:
    text = "Mail jane.doe@example.net or bob@test.org, server 8.8.8.8, lan 192.168.1.1, Upper@Example.COM"
    result = anonymize(text)
    assert "jane.doe@example.net" not in result.text and "bob@test.org" not in result.text
    assert "email@example.com" in result.text and "firstname.lastname@example.org" in result.text  # rotation
    assert "8.8.8.8" not in result.text and "192.168.1.1" in result.text  # only public addresses
    assert "Upper@Example.COM" in result.text  # FineWeb's pattern is lower-case only
    assert (result.emails, result.ips) == (2, 1) and result.changed
    assert not anonymize("nothing here").changed


def test_anonymize_matches_datatrove_per_document() -> None:
    formatters = pytest.importorskip("datatrove.pipeline.formatters")
    texts = ["a@b.co and c@d.io and e@f.org", "ips 1.1.1.1 10.0.0.1 9.9.9.9 8.8.4.4", "none"]
    for text in texts:
        assert anonymize(text).text == formatters.PIIFormatter().format(text)


def test_select_frame_is_seeded_sorted_and_whole_crawl_by_default() -> None:
    paths = [f"crawl-data/X/segments/{i}/warc/f{i}.warc.gz" for i in range(50)]
    frame = select_frame("X", paths, 5, seed=1)
    assert frame == select_frame("X", paths, 5, seed=1) and frame != select_frame("X", paths, 5, seed=2)
    assert frame.paths == sorted(frame.paths, key=paths.index) and len(set(frame.paths)) == 5
    assert frame.inclusion_probability == 0.1
    assert select_frame("X", paths, None, seed=1).paths == paths
    with pytest.raises(ValueError):
        select_frame("X", paths, 51, seed=1)


def test_shipped_configs_validate() -> None:
    from wildai.data.collect.config import CollectionConfig
    from wildai.data.evalsets.common import EvalSetsConfig
    from wildai.data.measure.filter_audit import FilterAuditConfig
    from wildai.data.monthly_sample import MonthlySampleConfig
    from wildai.data.pools_config import PoolsConfig
    from wildai.data.release.config import ReleaseConfig
    from wildai.labeling.config import LabelingConfig

    models = {"collection.yaml": CollectionConfig, "evalsets.yaml": EvalSetsConfig, "filter_audit.yaml": FilterAuditConfig,
              "monthly_sample.yaml": MonthlySampleConfig, "pools.yaml": PoolsConfig, "release.yaml": ReleaseConfig,
              "labeling.yaml": LabelingConfig}
    assert {p.name for p in CONFIG_DIR.glob("*.yaml")} == set(models)
    loaded = {name: load_config(CONFIG_DIR / name, model) for name, model in models.items()}
    assert len(loaded["collection.yaml"].fineweb.dumps) == 16 and len(loaded["collection.yaml"].common_crawl.dumps) == 12
    assert len(loaded["monthly_sample.yaml"].all_months()) == 62
    assert len(loaded["evalsets.yaml"].paloma.sources) == 16
    release = loaded["release.yaml"]
    assert release.hub.repo_id == "pangram/WildAI" and release.hub.private and not release.include_pangram_windows


def test_configs_reject_unknown_keys(tmp_path: Path) -> None:
    from wildai.data.measure.filter_audit import FilterAuditConfig

    path = tmp_path / "c.yaml"
    path.write_text("dump: X\nwarc_files: 1\ndocuments: 1\nseed: 1\ntypo: 3\n")
    with pytest.raises(ValueError, match="typo"):
        load_config(path, FilterAuditConfig)
