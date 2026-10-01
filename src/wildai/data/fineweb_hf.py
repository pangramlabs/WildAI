"""Read Hugging Face FineWeb dumps at a pinned revision, in a seeded random order of Parquet row groups.

FineWeb stores each crawl as ~300 Parquet files of ~1M documents in row groups of 1,000. Each file holds the output of a
subset of the crawl's WARC files, so reading row groups in a keyed-hash order is a cluster sample spread over the whole
crawl. Only the row groups actually read are downloaded (``HfFileSystem`` range requests).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

from wildai.data.hashing import keyed_hash, seeded_key

FINEWEB_REPO = "HuggingFaceFW/fineweb"
FINEWEB_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"  # the v1.4.0 branch
FINEWEB_COLUMNS = ("text", "id", "dump", "url", "date", "file_path", "language_score", "token_count")


@dataclass(frozen=True)
class RowGroup:
    file: str
    index: int
    rows: int

    @property
    def name(self) -> str:
        return f"{self.file}#{self.index}"


class ParquetDump:
    """The Parquet files of one crawl under a directory of any fsspec filesystem."""

    def __init__(self, fs: fsspec.AbstractFileSystem, directory: str) -> None:
        self.fs = fs
        self.directory = directory.rstrip("/")

    @classmethod
    def fineweb(cls, dump: str, revision: str = FINEWEB_REVISION) -> ParquetDump:
        from huggingface_hub import HfFileSystem

        return cls(HfFileSystem(), f"datasets/{FINEWEB_REPO}@{revision}/data/{dump}")

    def files(self) -> list[str]:
        return sorted(self.fs.glob(f"{self.directory}/*.parquet"))

    def row_groups(self) -> list[RowGroup]:
        groups = []
        for path in self.files():
            with self.fs.open(path, "rb") as handle:
                metadata = pq.ParquetFile(handle).metadata
            groups.extend(RowGroup(path, i, metadata.row_group(i).num_rows) for i in range(metadata.num_row_groups))
        return groups

    def read(self, group: RowGroup, columns: Sequence[str] = FINEWEB_COLUMNS) -> pa.Table:
        with self.fs.open(group.file, "rb") as handle:
            return pq.ParquetFile(handle).read_row_group(group.index, columns=list(columns))


def shuffled_row_groups(groups: Sequence[tuple[str, RowGroup]], seed: int) -> list[tuple[str, RowGroup]]:
    """(dump, row group) pairs in a seeded keyed-hash order; the same seed gives the same order on any machine.

    Groups are keyed by dump, file name and index, not by the repository path, so the order does not depend on the
    revision string or the filesystem the files are read from. Row groups of several dumps interleave at random.
    """

    key = seeded_key("fineweb-row-groups", seed)
    return sorted(groups, key=lambda item: keyed_hash(f"{item[0]}/{item[1].file.rsplit('/', 1)[-1]}#{item[1].index}", key))


def iter_row_groups(dumps: dict[str, ParquetDump], seed: int, columns: Sequence[str] = FINEWEB_COLUMNS) -> Iterator[pa.Table]:
    """Tables of successive row groups of one or more dumps (name -> dump) in seeded random order."""

    groups = [(name, group) for name, dump in dumps.items() for group in dump.row_groups()]
    for name, group in shuffled_row_groups(groups, seed):
        yield dumps[name].read(group, columns)


def warc_path(file_path: str) -> str:
    """FineWeb's ``file_path`` (``s3://commoncrawl/crawl-data/...``) as a crawl-relative ``crawl-data/...`` path."""

    marker = "crawl-data/"
    return file_path[file_path.index(marker):] if marker in file_path else file_path
