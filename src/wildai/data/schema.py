"""The table layouts every data stage reads and writes, declared once as Pydantic models.

Documents move through the pipeline as sharded Parquet whose columns are exactly these models' fields
(:func:`wildai.data.arrow_schema.arrow_schema` turns a model into its Arrow schema):

* :class:`WebDocument` -- a collected web document (Hugging Face FineWeb or our FineWeb-recipe rerun on Common Crawl).
* :class:`MonthlyDocument` -- a web document drawn for the monthly AI-share measurement.
* :class:`CandidateDocument` -- a web document chosen for Pangram labeling (by EditLens, or at random).
* :class:`PoolDocument` -- one row of a training pool (``pools/{human,ai,mixed,natural}``): a candidate with its Pangram
  label. **This is the interface the training mixture builder reads**: ``id``, ``text``, ``label``, ``token_count``
  (and ``sampling_hash``, the pool order); the other columns are provenance for analysis and the dataset release.

Labelers write sidecars keyed by ``id`` (see :mod:`wildai.labeling`), aligned row-for-row with the document shard they label.
"""

from __future__ import annotations

from typing import Annotated, Literal

import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from wildai.data.arrow_schema import ArrowType
from wildai.labeling.pangram.models import WindowScore

Source = Literal["fineweb", "common_crawl"]
"""Where a document's text came from: Hugging Face FineWeb, or our FineWeb-recipe rerun on Common Crawl WARC files."""

PoolLabel = Literal["human", "ai", "mixed"]
"""A pool's label: the Pangram document label, lower-cased."""

UInt64 = Annotated[int, ArrowType(pa.uint64())]


class WebDocument(BaseModel):
    """A web document as collected, before any labeling."""

    model_config = ConfigDict(frozen=True)

    id: str
    """Common Crawl WARC-Record-ID, e.g. ``<urn:uuid:...>``; FineWeb keeps it as its own ``id``."""
    text: str
    url: str
    date: str
    """WARC-Date of the capture (ISO 8601)."""
    dump: str
    """Common Crawl crawl, e.g. ``CC-MAIN-2026-25``."""
    source: Source
    warc_path: str | None
    """Path of the WARC file inside the crawl (``crawl-data/...warc.gz``); ``None`` for imported rows that did not keep it."""
    language_score: float
    """fastText English score from FineWeb's language filter."""
    token_count: int
    """GPT-2 tokens of ``text`` (no special tokens), see :mod:`wildai.data.tokens`."""
    pii_anonymized: bool
    """Whether e-mail and public IP addresses were anonymized (FineWeb's PII step)."""
    truncated: bool = False
    """Whether ``text`` is shorter than the extracted page. The pipeline never truncates; imported rows may be."""


class MonthlyDocument(WebDocument):
    """A document of the monthly measurement sample; ``month`` is the capture month the document was drawn for."""

    month: str
    """``YYYY-MM``."""


POOL_LABELS: tuple[PoolLabel, ...] = ("human", "ai", "mixed")


Selection = Literal["editlens_ai", "editlens_human", "natural"]
"""Why a document became a Pangram candidate: EditLens AI buckets, EditLens human bucket, or a label-blind random draw."""


class CandidateDocument(WebDocument):
    """A document selected for Pangram labeling (:mod:`wildai.data.preselect`)."""

    selection: Selection
    editlens_bucket: int | None
    """EditLens bucket (0 human ... 3 AI); ``None`` for label-blind draws, which are not EditLens-labeled."""
    editlens_score: float | None


class PoolDocument(CandidateDocument):
    """One document of a training pool; each pool is sorted by ``sampling_hash``, so any prefix is a uniform sample.

    Training needs ``id``, ``text``, ``label`` and ``token_count`` (GPT-2 tokens, the paper's unit for pool sizes;
    training re-tokenizes with its own tokenizer). The Pangram fields come from :mod:`wildai.labeling.pangram`.
    """

    label: PoolLabel
    """The Pangram document label, lower-cased."""
    sampling_hash: UInt64
    """Keyed BLAKE2b hash of ``id`` (:func:`wildai.data.hashing.keyed_hash`); the pool's order."""
    pangram_fraction_ai: float
    pangram_fraction_ai_assisted: float
    pangram_fraction_human: float
    pangram_version: str
    pangram_windows: list[WindowScore]
