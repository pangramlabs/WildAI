"""Sampling, the generation driver, AI-typical phrase rates and the labeler hook."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from wildai.evaluation.generate import Generation, GenerationTask, Prompt, generate, load_task
from wildai.evaluation.labels import ai_label_share
from wildai.evaluation.phrases import count_phrases, count_words, phrase_rate
from wildai.evaluation.runtime import LoadedModel
from wildai.evaluation.sampling import Sampling, generate_batch, sample_next


def test_top_k_one_is_greedy() -> None:
    logits = torch.randn(5, 50)
    rng = torch.Generator().manual_seed(0)
    assert torch.equal(sample_next(logits, Sampling(1.0, None, 1), rng), logits.argmax(-1))
    assert torch.equal(sample_next(logits, Sampling(0.0), rng), logits.argmax(-1))


def test_top_p_keeps_only_the_nucleus() -> None:
    logits = torch.tensor([[10.0, 9.0, -10.0, -10.0]])
    rng = torch.Generator().manual_seed(0)
    draws = {int(sample_next(logits, Sampling(1.0, 0.5, None), rng)) for _ in range(50)}
    assert draws == {0}


def test_generation_is_reproducible(loaded: LoadedModel) -> None:
    prompts = [[256, 104, 105], [256, 106, 107]]

    def run() -> list[list[int]]:
        return generate_batch(loaded.model, prompts, Sampling(1.0, 0.95, 20), 12, {256}, torch.Generator().manual_seed(3))

    first = run()
    assert first == run() and all(len(c) <= 12 for c in first)


def test_generate_driver_and_task_files(loaded: LoadedModel) -> None:
    task, prompts = load_task("writingprompts")
    assert len(prompts) == 500 and task.samples_per_prompt == 8 and task.top_k == 20 and task.seed == 8201
    webtext, webtext_prompts = load_task("webtext")
    assert len(webtext_prompts) == 5000 and webtext.max_new_tokens == 1024 and webtext.top_k is None
    small = GenerationTask(**{**task.model_dump(), "samples_per_prompt": 2, "max_new_tokens": 5})
    records = generate(loaded, small, [Prompt(id="p1", prompt="hi"), Prompt(id="p2", prompt="hello")], batch_size=3)
    assert [(r.prompt_id, r.sample_index) for r in records] == [("p1", 0), ("p1", 1), ("p2", 0), ("p2", 1)]


def test_phrase_counting() -> None:
    text = "Moreover, it's important to note this vibrant tapestry. Not just a game-changer; in today's world."
    assert count_phrases(text) == 7
    assert count_words("It's a well-known fact, isn't it?") == 6


def test_phrase_rate_pools_prompts() -> None:
    generations = [
        Generation(prompt_id="a", sample_index=0, text="moreover " + "word " * 99),
        Generation(prompt_id="a", sample_index=1, text="word " * 100),
        Generation(prompt_id="b", sample_index=0, text="delve delve " + "word " * 98),
    ]
    rate = phrase_rate(generations, np.random.default_rng(0))
    assert rate.rate == 1000 * 3 / 300 and rate.prompts == 2
    assert rate.low <= rate.rate <= rate.high


class EveryOtherLabeler:
    def is_ai(self, ids: Sequence[str], texts: Sequence[str]) -> list[bool]:
        return [i % 2 == 0 for i in range(len(texts))]


def test_label_share_uses_one_continuation_per_prompt() -> None:
    generations = [Generation(prompt_id=f"p{i}", sample_index=s, text="x") for i in range(10) for s in range(2)]
    share = ai_label_share(generations, EveryOtherLabeler(), np.random.default_rng(0))
    assert share.labeled == 10 and share.ai_percent == 50.0
