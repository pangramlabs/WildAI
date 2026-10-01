"""Typed readers for the result tables in `results/` that the paper's figures and tables are drawn from.

Every table is produced by the package (`wildai.*`) or shipped with the release; results/README.md documents the schemas.
"""

from __future__ import annotations

import csv
import json
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

RESULTS = Path(__file__).resolve().parents[1] / "results"
LABELS = ("Human", "Mixed", "AI")


class Row(BaseModel):
    model_config = ConfigDict(frozen=True)


def _read(path: Path, row: type[Row]) -> list:
    with path.open(encoding="utf-8") as handle:
        return [row.model_validate(r) for r in csv.DictReader(handle)]


class Model(Row):
    name: str
    group: str  # g001..g099: a human-only control and the runs built on its human documents; f01..f19: filtering pairs
    arm: Literal["control", "ai", "human", "repeat", "natural", "filtered"]
    depth: int
    n_params: int
    seed: int
    added_ratio: float | None  # design r of an addition; the matched AI ratio of a repetition model; empty for filtering
    human_tokens: int
    ai_tokens: int
    total_tokens: int
    steps: int
    split: Literal["fit", "held_out", "filtering", "repetition"]

    @field_validator("added_ratio", mode="before")
    @classmethod
    def _empty_is_none(cls, value: str) -> str | None:
        return value or None

    @property
    def ratio(self) -> float:
        """Realized AI tokens per human token."""
        return self.ai_tokens / self.human_tokens

    @property
    def human_tpp(self) -> float:
        return self.human_tokens / self.n_params


class Loss(Row):
    name: str
    target: str
    bpb: float


@cache
def models() -> dict[str, Model]:
    return {m.name: m for m in _read(RESULTS / "models.csv", Model)}


@cache
def losses() -> dict[tuple[str, str], float]:
    """(model name, target) -> bits per byte."""
    return {(r.name, r.target): r.bpb for r in _read(RESULTS / "losses.csv", Loss)}


@cache
def controls() -> dict[str, Model]:
    """Group -> its human-only control."""
    return {m.group: m for m in models().values() if m.arm == "control"}


class MonthlyShare(Row):
    month: str  # YYYY-MM
    dumps: str
    source: Literal["fineweb", "common_crawl_random_warc"]
    documents: int
    human_documents: int
    mixed_documents: int
    ai_documents: int
    tokens: int
    human_tokens: int
    mixed_tokens: int
    ai_tokens: int
    ai_share: float  # AI-labeled tokens / all tokens
    ai_or_mixed_share: float
    ai_share_lo95: float
    ai_share_hi95: float


def monthly_ai_share() -> list[MonthlyShare]:
    return _read(RESULTS / "web/monthly_ai_share.csv", MonthlyShare)


class Stage(BaseModel):
    stage: str
    kept: int  # documents of every label still in the pipeline after this stage
    Human: float  # % of each label's documents still in
    Mixed: float
    AI: float


class Pipeline(BaseModel):
    extraction: str
    labels: dict[str, int]  # documents per label in the sample
    stages: list[Stage]


def filter_audit() -> dict[str, Pipeline]:
    audit = json.loads((RESULTS / "web/filter_audit.json").read_text(encoding="utf-8"))
    return {name: Pipeline.model_validate(p) for name, p in audit["pipelines"].items()}


class TopicFormatCount(Row):
    period: str  # YYYY-MM for the monthly sample, YYYY for the training pool
    topic: str
    format: str
    label: str  # Human, Mixed or AI
    documents: int
    tokens: int


def topic_format_counts(which: Literal["monthly_sample", "pool"]) -> list[TopicFormatCount]:
    name = {"monthly_sample": "monthly_topic_format.csv", "pool": "pool_topic_format.csv"}[which]
    return _read(RESULTS / "web" / name, TopicFormatCount)


class Downstream(Row):
    name: str
    core: float  # CORE score (DCLM's 22 tasks, centered accuracy)


def downstream() -> dict[str, float]:
    """Model name -> CORE."""
    return {r.name: r.core for r in _read(RESULTS / "downstream.csv", Downstream)}


class PhraseRate(Row):
    """AI-typical phrases per 1,000 words of generated text, with a 95 % bootstrap interval over prompts."""

    name: str
    prompts: Literal["writingprompts", "webtext"]
    rate: float
    low: float
    high: float


def ai_phrase_rates() -> list[PhraseRate]:
    return _read(RESULTS / "generation/ai_phrase_rates.csv", PhraseRate)


class GenerationLabels(Row):
    """Percent of generations Pangram labels AI, with a 95 % bootstrap interval over prompts."""

    name: str
    prompts: Literal["writingprompts", "webtext"]
    scored: int
    ai_pct: float
    ai_pct_low: float
    ai_pct_high: float
    mixed_pct: float


def generation_labels() -> list[GenerationLabels]:
    return _read(RESULTS / "generation/pangram_labels.csv", GenerationLabels)
