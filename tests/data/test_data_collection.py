"""FineWeb row-group reading, the FineWeb collector and the monthly sample, on local Parquet fixtures."""

from __future__ import annotations

from pathlib import Path

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wildai.data.collect.config import FineWebCollection
from wildai.data.collect.fineweb import collect_dump, take_tokens, to_documents
from wildai.data.fineweb_hf import FINEWEB_COLUMNS, ParquetDump, iter_row_groups, shuffled_row_groups, warc_path
from wildai.data.monthly_sample import MonthlyFrame, MonthlySampleConfig, month_seed, select_month
from wildai.data.parquet_io import read_dataset


def fineweb_file(path: Path, dump: str, start: int, n: int, months: list[str], row_group_size: int = 5) -> None:
    rows = [{"text": f"text of document {start + i} " + "word " * (20 + i % 7), "id": f"<urn:uuid:{dump}-{start + i}>",
             "dump": dump, "url": f"https://site/{start + i}", "date": f"{months[i % len(months)]}-15T00:00:00Z",
             "file_path": f"s3://commoncrawl/crawl-data/{dump}/segments/7/warc/f{i % 3}.warc.gz",
             "language": "en", "language_score": 0.95, "token_count": 30 + i % 7} for i in range(n)]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=row_group_size)


@pytest.fixture
def dump(tmp_path: Path) -> ParquetDump:
    for k in range(3):
        fineweb_file(tmp_path / "CC-MAIN-2021-10" / f"000_0000{k}.parquet", "CC-MAIN-2021-10", 100 * k, 20, ["2021-02", "2021-03"])
    return ParquetDump(fsspec.filesystem("file"), str(tmp_path / "CC-MAIN-2021-10"))


def test_row_groups_are_listed_and_shuffled_reproducibly(dump: ParquetDump) -> None:
    groups = dump.row_groups()
    assert len(groups) == 12 and {g.rows for g in groups} == {5}
    pairs = [("CC-MAIN-2021-10", g) for g in groups]
    order = shuffled_row_groups(pairs, seed=7)
    assert order == shuffled_row_groups(list(reversed(pairs)), seed=7) != shuffled_row_groups(pairs, seed=8)
    tables = list(iter_row_groups({"CC-MAIN-2021-10": dump}, seed=7))
    assert sum(len(t) for t in tables) == 60 and tables[0].column_names == list(FINEWEB_COLUMNS)


def test_to_documents_and_token_target(dump: ParquetDump) -> None:
    table = dump.read(dump.row_groups()[0])
    documents = to_documents(table, min_chars=0)
    assert documents["source"].to_pylist() == ["fineweb"] * 5 and all(documents["pii_anonymized"].to_pylist())
    assert documents["warc_path"][0].as_py().startswith("crawl-data/CC-MAIN-2021-10/")
    assert len(to_documents(table, min_chars=10_000)) == 0
    assert warc_path("s3://commoncrawl/crawl-data/X/a.warc.gz") == "crawl-data/X/a.warc.gz"
    parts = list(take_tokens([documents, documents], target_tokens=int(sum(documents["token_count"].to_pylist())) + 1))
    assert len(parts) == 2 and len(parts[1]) == 1


def test_collect_dump_stops_at_the_token_target(dump: ParquetDump, tmp_path: Path) -> None:
    settings = FineWebCollection(dumps=["CC-MAIN-2021-10"], seed=3, target_tokens_per_dump=500, min_chars=10)
    counts = collect_dump(dump, "CC-MAIN-2021-10", settings, tmp_path / "out")
    collected = read_dataset(tmp_path / "out")
    assert counts["documents"] == len(collected) and counts["tokens"] >= 500
    assert counts["tokens"] - collected["token_count"][-1].as_py() < 500  # the last document crossed the target


def test_select_month_keeps_only_the_month_in_hash_order(dump: ParquetDump) -> None:
    tables = [to_documents(t, 0) for t in iter_row_groups({"CC-MAIN-2021-10": dump}, seed=1)]
    chosen = select_month(iter(tables), "2021-03", month_seed(20260917, "2021-03"), n=10)
    assert len(chosen) == 10 and set(chosen["month"].to_pylist()) == {"2021-03"}
    assert all(d.startswith("2021-03") for d in chosen["date"].to_pylist())
    again = select_month(iter(list(reversed(tables))), "2021-03", month_seed(20260917, "2021-03"), n=10)
    assert chosen["id"].to_pylist() == again["id"].to_pylist()
    with pytest.raises(RuntimeError, match="only"):
        select_month(iter(tables), "2021-03", 1, n=1000)


def test_monthly_config_lookup() -> None:
    config = MonthlySampleConfig(frames=[MonthlyFrame(source="fineweb", seed=10, months={"2021-01": ["CC-MAIN-2021-04"]})])
    frame, dumps = config.month("2021-01")
    assert frame.source == "fineweb" and dumps == ["CC-MAIN-2021-04"] and month_seed(10, "2021-01") == 202111
    with pytest.raises(KeyError):
        config.month("1999-01")
