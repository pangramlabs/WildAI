"""Columns of the released tables. ``id`` is always the Common Crawl WARC-Record-ID of the page."""

from __future__ import annotations

from typing import Annotated

import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from wildai.data.arrow_schema import ArrowType, arrow_schema
from wildai.data.schema import Selection, Source, UInt64
from wildai.labeling.pangram.models import PangramLabel, WindowScore

Float32 = Annotated[float, ArrowType(pa.float32())]


class ReleaseDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    url: str
    date: str
    """Capture time (WARC-Date)."""
    dump: str
    """Common Crawl crawl, e.g. ``CC-MAIN-2026-25``."""
    source: Source
    """``fineweb``: Hugging Face FineWeb v1.4.0; ``common_crawl``: our FineWeb-recipe run on Common Crawl WARC files."""
    warc_path: str | None
    """WARC file of the capture inside the crawl; ``None`` where it was not kept (the pool's FineWeb documents)."""
    token_count: int
    """GPT-2 tokens of the released text."""
    truncated: bool
    """The released text is shorter than the page's extracted text."""
    pii_anonymized_after_labeling: bool
    """E-mail or IP addresses were anonymized at release, after the labels were computed on the original text."""
    pangram_label: PangramLabel | None
    pangram_fraction_ai: float | None
    pangram_fraction_ai_assisted: float | None
    pangram_fraction_human: float | None
    pangram_version: str | None
    topic: str | None
    """WebOrganizer topic."""
    topic_score: Float32 | None
    format: str | None
    """WebOrganizer format."""
    format_score: Float32 | None


class PoolRecord(ReleaseDocument):
    """A document of the WildAI pool (configs ``human``, ``ai``, ``mixed``; ``labels`` without ``text``)."""

    language_score: float
    selection: Selection
    """Why Pangram labeled it: EditLens AI buckets, EditLens human bucket, or a label-blind draw."""
    sampling_hash: UInt64
    """The pool order: taking documents in increasing hash gives a uniform sample."""


class MonthlyRecord(ReleaseDocument):
    """A document of the monthly measurement sample."""

    month: str
    language_score: float


class FilterAuditRecord(ReleaseDocument):
    """A document of the filter audit's FineWeb sample, with whether it is alive after each FineWeb stage."""

    language: str
    language_score: float
    english: bool
    gopher_repetition: bool
    gopher_quality: bool
    c4: bool
    fineweb_quality: bool
    removed_at: str | None
    removal_reason: str | None


WINDOWS_FIELD = pa.field("pangram_windows", pa.list_(pa.struct(list(arrow_schema(WindowScore)))))


def release_schema(model: type[ReleaseDocument], *, with_text: bool = True, with_windows: bool = False) -> pa.Schema:
    """The Arrow schema of a released table; ``labels`` drops the text, windows are opt-in."""

    schema = arrow_schema(model)
    if not with_text:
        schema = schema.remove(schema.get_field_index("text"))
    return schema.append(WINDOWS_FIELD) if with_windows else schema
