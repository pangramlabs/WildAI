"""Evaluation-set settings, manifests and writing."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TypeVar

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from wildai.data.config import StrictModel
from wildai.data.parquet_io import write_table_atomic
from wildai.data.tokens import Gpt2TokenCounter, TokenCounter

T = TypeVar("T")
SHARD = "shard_00000.parquet"
MANIFEST = "manifest.json"


class HubTextSet(StrictModel):
    """The first ``target_chars`` characters (documents of at least ``min_chars``) of a pinned dataset's stream."""

    repo: str
    revision: str
    config: str
    split: str
    target_chars: int = Field(gt=0)
    min_chars: int = 200


class FineWebYearSet(StrictModel):
    """An equal character budget from each listed FineWeb dump, read in seeded row-group order, then shuffled."""

    revision: str
    dumps: list[str]
    target_chars: int = Field(gt=0)
    min_chars: int = 200
    seed: int


class NaturalTailSet(StrictModel):
    """The end of the natural pool's hash order, disjoint from every training pool."""

    target_tokens: int = Field(gt=0)
    """GPT-2 tokens of the natural set; its human and AI partitions split exactly these documents."""


class PalomaSet(StrictModel):
    repo: str
    revision: str
    split: str
    sources: dict[str, str]
    """Our name -> Paloma subset name (16 sources)."""


class EvalSetsConfig(StrictModel):
    c4: HubTextSet
    cosmopedia: HubTextSet
    fw22: FineWebYearSet
    fw26: NaturalTailSet
    paloma: PalomaSet


class EvalSetManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    source: str
    revision: str | None
    selection: str
    documents: int
    characters: int
    gpt2_tokens: int
    domains: dict[str, int] | None = None


def take_chars(items: Iterable[T], target_chars: int, min_chars: int, text_of: Callable[[T], str] = str) -> Iterator[T]:
    """Items whose text has at least ``min_chars`` characters, until their total reaches ``target_chars``."""

    total = 0
    for item in items:
        length = len(text_of(item))
        if length < min_chars:
            continue
        yield item
        total += length
        if total >= target_chars:
            return


def write_eval_set(output: Path, name: str, texts: list[str], *, source: str, revision: str | None, selection: str,
                   domains: list[str] | None = None, counter: TokenCounter | None = None) -> EvalSetManifest:
    """Write one evaluation set and its manifest; refuses to overwrite an existing set."""

    directory = output / name
    if (directory / MANIFEST).exists():
        raise FileExistsError(f"{directory} already holds an evaluation set")
    columns: dict[str, pa.Array] = {"text": pa.array(texts, pa.string())}
    if domains is not None:
        columns["domain"] = pa.array(domains, pa.string())
    write_table_atomic(pa.table(columns), directory / SHARD)
    counter = counter or Gpt2TokenCounter()
    domain_counts = dict(sorted(Counter(domains).items())) if domains is not None else None
    manifest = EvalSetManifest(name=name, source=source, revision=revision, selection=selection, documents=len(texts),
                               characters=sum(map(len, texts)), gpt2_tokens=sum(counter.count_batch(texts)),
                               domains=domain_counts)
    (directory / MANIFEST).write_text(json.dumps(manifest.model_dump(), indent=1))
    return manifest
