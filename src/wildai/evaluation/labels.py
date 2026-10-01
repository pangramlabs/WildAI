"""Share of generations a detector labels AI, with a bootstrap over prompts.

The paper labels the first continuation (`sample_index` 0) of each WritingPrompts prompt with Pangram 3.3.2 through
the public Pangram API. Any labeler that maps texts to AI / not-AI plugs in through `AILabeler`, e.g. a thin adapter
over `wildai.labeling.pangram` that returns `label == "AI"` for each text.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict

from wildai.evaluation.generate import Generation

BOOTSTRAP_DRAWS = 2000


class AILabeler(Protocol):
    def is_ai(self, ids: Sequence[str], texts: Sequence[str]) -> list[bool]:
        """Whether each text is labeled AI-generated (ids identify texts, e.g. for caching)."""
        ...


class LabelShare(BaseModel):
    model_config = ConfigDict(frozen=True)

    ai_percent: float
    low: float
    high: float
    labeled: int


def ai_label_share(generations: Sequence[Generation], labeler: AILabeler, rng: np.random.Generator, sample_index: int = 0, draws: int = BOOTSTRAP_DRAWS) -> LabelShare:
    """Percent of prompts whose `sample_index`-th continuation is labeled AI, with a 95 % prompt-bootstrap interval."""
    chosen = [g for g in generations if g.sample_index == sample_index]
    labels = np.array(labeler.is_ai([f"{g.prompt_id}/{g.sample_index}" for g in chosen], [g.text for g in chosen]), dtype=float)
    boot = 100 * labels[rng.integers(0, len(labels), (draws, len(labels)))].mean(1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return LabelShare(ai_percent=100 * float(labels.mean()), low=float(low), high=float(high), labeled=len(labels))
