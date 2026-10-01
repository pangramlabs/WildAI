"""Release settings (``configs/data/release.yaml``)."""

from __future__ import annotations

from pydantic import Field

from wildai.citation import BIBTEX
from wildai.data.config import StrictModel


class HubSettings(StrictModel):
    org: str = "pangram"
    name: str = "WildAI"
    private: bool = True
    """Create the dataset repository as private."""
    gated: bool = False
    collection: str = "WildAI"
    """Title of the organization collection the dataset joins, beside the models and the paper."""
    """Require access requests (adds ``extra_gated_*`` fields to the card and turns on manual approval)."""

    @property
    def repo_id(self) -> str:
        return f"{self.org}/{self.name}"


class CardText(StrictModel):
    """Card text that describes the released data rather than the pipeline; the authors set it for each release."""

    extension_note: str = ""
    """Anything the ``common_crawl`` rows did differently from the recipe (e.g. which WARC files, dedup, PII)."""
    pangram_note: str = "Labels come from the public Pangram API; `pangram_version` records the detector version."
    contact: str = "the dataset maintainers through the Hugging Face discussion tab"
    license_text: str = "The text comes from FineWeb (ODC-By 1.0) and Common Crawl, and remains subject to Common Crawl's terms of use."
    """Notes on the license beyond the release license itself (stated at the top of the card)."""
    citation: str = BIBTEX
    gated_prompt: str = "Please describe how you will use WildAI. Access is granted for research use."


class ReleaseConfig(StrictModel):
    hub: HubSettings = HubSettings()
    card: CardText = CardText()
    include_pangram_windows: bool = False
    """Export per-window Pangram scores (undecided; off by default)."""
    rows_per_shard: int = Field(100_000, gt=0)
