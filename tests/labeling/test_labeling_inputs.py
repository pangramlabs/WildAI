"""EditLens text preparation and the sidecar runner."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wildai.data.parquet_io import read_joined
from wildai.labeling.editlens_text import clean_text, select_windows
from wildai.labeling.sidecar import label_directory, my_shards


def test_clean_text_matches_editlens_preprocessing() -> None:
    assert clean_text("Hello   World\n\nNew  line") == "hello world new line"
    assert clean_text("Sure! Here is your essay:\nThe essay body.") == "the essay body."
    assert clean_text("Sure") == "sure"  # a single line is never dropped
    assert clean_text("<think>plan</think> Answer text") == "answer text"
    assert clean_text("I love it 👍") == "i love it :thumbs_up:"
    assert clean_text("## Title: A\nBody") == "body"  # leading symbols do not hide a preamble


def test_select_windows() -> None:
    ids = list(range(10))
    assert select_windows(ids, window=20, k=3) == [ids]
    assert select_windows(ids, window=4, k=3) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    long = list(range(50))  # ten windows of five: first, middle and last
    assert select_windows(long, window=5, k=3) == [long[0:5], long[20:25], long[45:50]]
    assert select_windows(long, window=5, k=1) == [long[0:5]]


def test_label_directory_writes_aligned_sidecars_and_resumes(tmp_path: Path) -> None:
    (tmp_path / "in").mkdir()
    for k in range(3):
        pq.write_table(pa.table({"id": [f"{k}-{i}" for i in range(4)], "text": ["x"] * 4}), tmp_path / "in" / f"part-0000{k}.parquet")
    calls: list[str] = []

    def label(table: pa.Table, shard: Path) -> pa.Table:
        calls.append(shard.name)
        return pa.table({"id": table["id"], "score": [1.0] * len(table)})

    assert [p.name for p in my_shards(tmp_path / "in", 1, 2)] == ["part-00001.parquet"]
    written = label_directory(tmp_path / "in", tmp_path / "out", ["id", "text"], label, shard_index=0, num_shards=2)
    assert [p.name for p in written] == ["part-00000.parquet", "part-00002.parquet"]
    label_directory(tmp_path / "in", tmp_path / "out", ["id", "text"], label)  # only the missing shard is labeled
    assert calls == ["part-00000.parquet", "part-00002.parquet", "part-00001.parquet"]
    joined = read_joined(tmp_path / "in" / "part-00001.parquet", {"s": tmp_path / "out"})
    assert joined.column_names == ["id", "text", "score"]

    def misaligned(table: pa.Table, shard: Path) -> pa.Table:
        return pa.table({"id": list(reversed(table["id"].to_pylist()))})

    with pytest.raises(RuntimeError, match="aligned"):
        label_directory(tmp_path / "in", tmp_path / "other", ["id"], misaligned)
