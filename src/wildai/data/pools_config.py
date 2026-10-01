"""Candidate selection and pool settings (``configs/data/pools.yaml``)."""

from __future__ import annotations

from pydantic import Field

from wildai.data.config import StrictModel


class EditLensPreselection(StrictModel):
    human_bucket: int = 0
    ai_min_bucket: int = 2
    human_token_budget: int = Field(gt=0)
    """GPT-2 tokens of EditLens-human candidates sent to Pangram, over all crawls."""
    key: str


class NaturalDraw(StrictModel):
    dumps: list[str]
    """Crawls to draw from (document directories are named by crawl)."""
    documents: int = Field(gt=0)
    key: str


class PoolSettings(StrictModel):
    key: str
    """Hash key of the pools' order."""
    bins: int = Field(1024, gt=0)
    """Hash ranges sorted one at a time (memory holds one range)."""
    rows_per_shard: int = Field(100_000, gt=0)


class PoolsConfig(StrictModel):
    editlens: EditLensPreselection
    natural: NaturalDraw
    pools: PoolSettings
