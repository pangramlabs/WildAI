"""Rate of AI-typical phrases in generated text, per 1,000 words, with a bootstrap over prompts.

The 26 phrases (such as "delve", "tapestry" and "it is important to note") are matched case-insensitively as
substrings, non-overlapping, with one alternation in list order. Words are runs of letters, optionally joined by one
apostrophe or hyphen. A model's rate pools every continuation: 1,000 x total phrase hits / total words. The 95 %
interval resamples prompts with replacement (all continuations of a prompt together), 2,000 draws.

Several generation files are scored with one random generator seeded 0, in the order given (the paper's tables and
figures pass each figure's models in sorted order, so their intervals depend on that order):

    python -m wildai.evaluation.phrases gens_a.jsonl gens_b.jsonl --output rates.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from wildai.evaluation.generate import Generation, read_generations

AI_TYPICAL_PHRASES = (
    "delve",
    "tapestry",
    "testament to",
    "it's important to note",
    "it is important to note",
    "in conclusion",
    "moreover",
    "furthermore",
    "vibrant",
    "crucial role",
    "landscape of",
    "realm of",
    "underscores",
    "showcasing",
    "boasts",
    "in today's",
    "navigate the",
    "unleash",
    "elevate",
    "foster",
    "seamless",
    "comprehensive guide",
    "game-changer",
    "ever-evolving",
    "a beacon of",
    "not just",
)
PHRASE_RE = re.compile("|".join(re.escape(phrase) for phrase in AI_TYPICAL_PHRASES))
WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?")
BOOTSTRAP_DRAWS = 2000


class PhraseRate(BaseModel):
    model_config = ConfigDict(frozen=True)

    rate: float  # AI-typical phrases per 1,000 words
    low: float
    high: float
    prompts: int
    words: int
    hits: int


def count_words(text: str) -> int:
    return sum(1 for _ in WORD_RE.finditer(text))


def count_phrases(text: str) -> int:
    return len(PHRASE_RE.findall(text.lower()))


def phrase_rate(generations: Iterable[Generation], rng: np.random.Generator, draws: int = BOOTSTRAP_DRAWS) -> PhraseRate:
    hits: dict[str, int] = {}
    words: dict[str, int] = {}
    for generation in generations:
        hits[generation.prompt_id] = hits.get(generation.prompt_id, 0) + count_phrases(generation.text)
        words[generation.prompt_id] = words.get(generation.prompt_id, 0) + count_words(generation.text)
    h, w = np.array(list(hits.values())), np.array(list(words.values()))
    resamples = rng.integers(0, len(h), (draws, len(h)))
    boot = 1000 * h[resamples].sum(1) / w[resamples].sum(1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return PhraseRate(rate=1000 * h.sum() / w.sum(), low=float(low), high=float(high), prompts=len(h), words=int(w.sum()), hits=int(h.sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("generations", type=Path, nargs="+", help="JSON-lines files from wildai.evaluation.generate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--output", type=Path, help="CSV with one row per file")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for path in args.generations:
        rate = phrase_rate(read_generations(path), rng, args.draws)
        rows.append({"generations": str(path), **rate.model_dump()})
        print(f"{path}: {rate.rate:.3f} per 1,000 words [{rate.low:.3f}, {rate.high:.3f}]")
    if args.output:
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
