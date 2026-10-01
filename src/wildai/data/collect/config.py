"""Collection settings (``configs/data/collection.yaml``)."""

from __future__ import annotations

from pydantic import Field

from wildai.data.config import StrictModel
from wildai.data.fineweb_hf import FINEWEB_REVISION


class FineWebCollection(StrictModel):
    """Documents sampled from Hugging Face FineWeb."""

    revision: str = FINEWEB_REVISION
    dumps: list[str]
    seed: int
    target_tokens_per_dump: int = Field(gt=0)
    """Stop reading a dump once this many GPT-2 tokens are collected."""
    min_chars: int = 200


class RecipeSettings(StrictModel):
    """How the FineWeb recipe runs on a set of WARC files (see :mod:`wildai.data.collect.recipe`)."""

    batch_files: int = Field(16, gt=0)
    """WARC files downloaded and filtered together; raw WARCs are deleted once their batch is filtered."""
    extraction_timeout: float = 0.1
    """Trafilatura's per-document time limit in seconds (FineWeb's value); extraction is load-sensitive at this limit."""
    minhash: bool = True
    """Per-crawl MinHash near-duplicate removal (FineWeb's 5-gram, 14 buckets x 8 hashes, 64-bit)."""
    pii: bool = True
    """Anonymize e-mail and public IP addresses (FineWeb's PIIFormatter)."""
    min_chars: int = 200
    """Documents shorter than this are dropped when materialized, as for the FineWeb sample."""


class CommonCrawlCollection(StrictModel):
    """The FineWeb recipe rerun on Common Crawl WARC files of crawls FineWeb has not released."""

    dumps: list[str]
    seed: int
    warc_files_per_crawl: int | None = None
    """WARC files drawn uniformly from each crawl; ``None`` processes the whole crawl, as FineWeb does."""
    recipe: RecipeSettings = RecipeSettings()


class CollectionConfig(StrictModel):
    fineweb: FineWebCollection
    common_crawl: CommonCrawlCollection
