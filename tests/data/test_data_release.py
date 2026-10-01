"""Release export, dataset card and upload on tiny fixtures (no network)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from tests.data.builders import WordCounter, candidates, pangram_sidecar, web_documents, write
from wildai.citation import ARXIV_ID
from wildai.data.parquet_io import read_dataset
from wildai.data.pools import build_pools
from wildai.data.pools_config import PoolSettings
from wildai.data.release.card import render
from wildai.data.release.config import HubSettings, ReleaseConfig
from wildai.data.release.export import Exporter, anonymize_rows
from wildai.data.release.push import push
from wildai.data.release.schema import FilterAuditRecord, MonthlyRecord, PoolRecord, release_schema


def weborganizer_sidecar(ids: list[str]) -> pa.Table:
    return pa.table({"id": ids, "topic": ["Health"] * len(ids), "topic_score": pa.array([0.9] * len(ids), pa.float32()),
                     "format": ["Tutorial"] * len(ids), "format_score": pa.array([0.8] * len(ids), pa.float32())})


@pytest.fixture
def pools_dir(tmp_path: Path) -> Path:
    ids = ["a1", "h1", "m1", "a2"]
    texts = ["write to jane@mail.com now " * 20, "human text " * 30, "mixed text " * 30, "ai text " * 30]
    write(candidates(ids, texts=texts, pii_anonymized=False), tmp_path / "cand" / "X" / "part-00000.parquet")
    write(pangram_sidecar(ids, ["AI", "Human", "Mixed", "AI"]), tmp_path / "pangram" / "X" / "part-00000.parquet")
    build_pools(tmp_path / "cand", tmp_path / "pangram", tmp_path / "pools", PoolSettings(key="k", bins=2), natural=False)
    for pool in ("human", "ai", "mixed"):
        for shard in (tmp_path / "pools" / pool).glob("*.parquet"):
            write(weborganizer_sidecar(pq.read_table(shard, columns=["id"])["id"].to_pylist()), tmp_path / "wo" / pool / shard.name)
    return tmp_path / "pools"


def test_anonymize_rows_recounts_changed_rows() -> None:
    table = pa.table({"text": ["mail a@b.com x", "clean"], "token_count": [99, 99]})
    out = anonymize_rows(table, [True, True], WordCounter())
    assert out["text"][0].as_py() == "mail email@example.com x" and out["token_count"].to_pylist() == [3, 99]
    assert out["pii_anonymized_after_labeling"].to_pylist() == [True, False]


def test_release_schema_options() -> None:
    assert "text" not in release_schema(PoolRecord, with_text=False).names
    assert "pangram_windows" not in release_schema(PoolRecord).names
    assert "pangram_windows" in release_schema(MonthlyRecord, with_windows=True).names
    assert {"english", "fineweb_quality", "removed_at"} <= set(release_schema(FilterAuditRecord).names)


def test_export_pools_and_card(pools_dir: Path, tmp_path: Path) -> None:
    release = tmp_path / "release"
    Exporter(release, ReleaseConfig(), WordCounter()).pools(pools_dir, tmp_path / "wo")
    ai = read_dataset(release / "ai")
    assert ai.schema == release_schema(PoolRecord) and set(ai["pangram_label"].to_pylist()) == {"AI"}
    flagged = dict(zip(ai["id"].to_pylist(), ai["pii_anonymized_after_labeling"].to_pylist()))
    assert flagged == {"a1": True, "a2": False}
    assert "jane@mail.com" not in "".join(ai["text"].to_pylist())
    labels = read_dataset(release / "labels")
    assert "text" not in labels.column_names and len(labels) == 4

    card = render(release, ReleaseConfig())
    front = yaml.safe_load(card.split("---")[1])
    assert [c["config_name"] for c in front["configs"]] == ["human", "ai", "mixed", "labels"]
    assert front["license"] == "cc-by-nc-sa-4.0" and "extra_gated_fields" not in front
    assert "CC BY-NC-SA 4.0" in card and "| `ai` | common_crawl | 2 |" in card
    gated = yaml.safe_load(render(release, ReleaseConfig(hub=HubSettings(gated=True))).split("---")[1])
    assert gated["license"] == "cc-by-nc-sa-4.0" and "extra_gated_fields" in gated and "extra_gated_prompt" in gated
    assert "monthly_sample" not in card and "filter_audit" not in card  # only the configs that were exported


def test_parallel_pool_export_writes_the_same_tables(pools_dir: Path, tmp_path: Path) -> None:
    Exporter(tmp_path / "one", ReleaseConfig(), WordCounter()).pools(pools_dir, tmp_path / "wo")
    Exporter(tmp_path / "two", ReleaseConfig(), WordCounter(), workers=2).pools(pools_dir, tmp_path / "wo")
    for config in ("human", "ai", "mixed", "labels"):
        assert read_dataset(tmp_path / "one" / config).equals(read_dataset(tmp_path / "two" / config))


def test_export_monthly_sample_keeps_file_order(tmp_path: Path) -> None:
    ids = ["m3", "m1", "m2"]
    sample = web_documents(ids, "CC-MAIN-2026-30").append_column("month", pa.array(["2026-07"] * 3))
    write(sample, tmp_path / "sample" / "2026-07.parquet")
    write(pangram_sidecar(ids, ["AI", "Human", "Mixed"]), tmp_path / "pangram" / "2026-07.parquet")
    write(weborganizer_sidecar(ids), tmp_path / "wo" / "2026-07.parquet")
    config = ReleaseConfig(include_pangram_windows=True)
    Exporter(tmp_path / "release", config, WordCounter()).monthly_sample(tmp_path / "sample", tmp_path / "pangram", tmp_path / "wo")
    table = read_dataset(tmp_path / "release" / "monthly_sample")
    assert table["id"].to_pylist() == ids and table.schema == release_schema(MonthlyRecord, with_windows=True)


def test_export_filter_audit(tmp_path: Path) -> None:
    ids = ["x1", "x2"]
    sample = pa.table({"id": ids, "text": ["mail z@y.org", "plain"], "url": ["u1", "u2"], "date": ["d", "d"],
                       "dump": ["CC-MAIN-2026-30"] * 2, "warc_path": ["w", "w"]})
    write(sample, tmp_path / "audit" / "sample" / "part-00000.parquet")
    stages = pa.table({"id": ids, "language": ["en", "de"], "language_score": [0.9, 0.2], "english": [True, False],
                       "gopher_repetition": [True, False], "gopher_quality": [True, False], "c4": [True, False],
                       "fineweb_quality": [False, False], "removed_at": ["fineweb_quality", "language"],
                       "removal_reason": ["line_punct_ratio", None]})
    write(stages, tmp_path / "audit" / "fineweb_stages.parquet")
    write(pangram_sidecar(ids, ["AI", None]), tmp_path / "pangram" / "sample" / "part-00000.parquet")
    write(weborganizer_sidecar(ids), tmp_path / "wo" / "sample" / "part-00000.parquet")
    Exporter(tmp_path / "release", ReleaseConfig(), WordCounter()).filter_audit(tmp_path / "audit", tmp_path / "pangram", tmp_path / "wo")
    fineweb = read_dataset(tmp_path / "release" / "filter_audit" / "fineweb")
    assert fineweb["pangram_label"].to_pylist() == ["AI", None] and fineweb["pii_anonymized_after_labeling"].to_pylist() == [True, False]
    assert not (tmp_path / "release" / "filter_audit" / "dclm").exists()  # DCLM's audit text is not released


class FakeHub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def create_repo(self, repo_id: str, **kwargs: object) -> None:
        self.calls.append(("create_repo", {"repo_id": repo_id, **kwargs}))

    def update_repo_settings(self, repo_id: str, **kwargs: object) -> None:
        self.calls.append(("update_repo_settings", {"repo_id": repo_id, **kwargs}))

    def upload_large_folder(self, repo_id: str, folder_path: str, **kwargs: object) -> None:
        self.calls.append(("upload_large_folder", {"repo_id": repo_id, "folder_path": folder_path, **kwargs}))

    def create_collection(self, title: str, **kwargs: object) -> SimpleNamespace:
        self.calls.append(("create_collection", {"title": title, **kwargs}))
        return SimpleNamespace(slug=f"{kwargs['namespace']}/{title.lower()}-0123")

    def add_collection_item(self, collection_slug: str, item_id: str, item_type: str, **kwargs: object) -> None:
        self.calls.append(("add_collection_item", {"slug": collection_slug, "item_id": item_id, "item_type": item_type, **kwargs}))


def test_push_creates_a_private_repo_under_pangram(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("card")
    hub = FakeHub()
    url = push(tmp_path, HubSettings(), hub)
    assert url == "https://huggingface.co/datasets/pangram/WildAI"
    assert hub.calls[0] == ("create_repo", {"repo_id": "pangram/WildAI", "repo_type": "dataset", "private": True, "exist_ok": True})
    assert [c[0] for c in hub.calls] == ["create_repo", "upload_large_folder", "create_collection", "add_collection_item", "add_collection_item"]
    assert hub.calls[2][1] | {"description": ""} == {"title": "WildAI", "namespace": "pangram", "description": "", "private": True, "exists_ok": True}
    assert [(c[1]["item_id"], c[1]["item_type"]) for c in hub.calls[3:]] == [(ARXIV_ID, "paper"), ("pangram/WildAI", "dataset")]
    gated = FakeHub()
    push(tmp_path, HubSettings(gated=True, private=False), gated)
    assert gated.calls[1] == ("update_repo_settings", {"repo_id": "pangram/WildAI", "gated": "manual", "repo_type": "dataset"})
    with pytest.raises(FileNotFoundError):
        push(tmp_path / "missing", HubSettings(), FakeHub())
