"""5-shot MMLU accuracy by answer-letter likelihood (`cais/mmlu`, all subjects, test split).

Each question is preceded by five dev-set examples of its subject (shuffled once per question by a generator seeded
42 and consumed in test order), rendered as `Question: ...\nA. ...\nAnswer: X\n\n`. The prediction is the letter among
" A", " B", " C", " D" with the lowest mean loss.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from datasets import load_dataset
from pydantic import BaseModel, ConfigDict

from wildai.evaluation.runtime import LoadedModel

LETTERS = ("A", "B", "C", "D")
CHOICES = tuple(f" {letter}" for letter in LETTERS)


class MmluScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    accuracy: float
    correct: int
    total: int


@dataclass(frozen=True)
class Question:
    prompt: str
    answer: int


def render(question: str, choices: list[str], answer: int | None = None) -> str:
    text = f"Question: {question}\n" + "".join(f"{letter}. {choice}\n" for letter, choice in zip(LETTERS, choices)) + "Answer:"
    return text if answer is None else text + f" {LETTERS[answer]}\n\n"


def build_questions(num_fewshot: int = 5, seed: int = 42) -> list[Question]:
    test = load_dataset("cais/mmlu", "all", split="test")
    dev = load_dataset("cais/mmlu", "all", split="dev")
    shots_by_subject: dict[str, list[dict[str, object]]] = {}
    for row in dev:
        shots_by_subject.setdefault(row["subject"], []).append(row)
    rng = random.Random(seed)
    questions = []
    for row in test:
        shots = list(shots_by_subject.get(row["subject"], []))
        rng.shuffle(shots)
        prefix = "".join(render(s["question"], s["choices"], int(s["answer"])) for s in shots[:num_fewshot])
        questions.append(Question(prefix + render(row["question"], row["choices"]), int(row["answer"])))
    return questions


@torch.no_grad()
def continuation_losses(loaded: LoadedModel, context: str, choices: tuple[str, ...]) -> list[float]:
    """Mean loss of each continuation after the shared context (sequences cropped from the left to the context size)."""
    full = [loaded.tokenizer.encode_documents([context + choice])[0] for choice in choices]
    shared = next((i for i in range(min(map(len, full))) if any(t[i] != full[0][i] for t in full)), min(map(len, full)))
    rows, starts = [], []
    for tokens in full:
        crop = max(0, len(tokens) - loaded.sequence_len)
        tokens = tokens[crop:]
        rows.append(tokens)
        starts.append(min(max(shared - crop, 1), len(tokens) - 1))
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), loaded.tokenizer.bos_id, dtype=torch.long)
    for i, tokens in enumerate(rows):
        ids[i, : len(tokens)] = torch.tensor(tokens)
    ids = ids.to(loaded.device)
    logits = loaded.model(ids)
    losses = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), torch.roll(ids, -1, dims=1).view(-1), reduction="none").view(ids.shape)
    return [losses[i, start - 1 : len(tokens) - 1].mean().item() for i, (tokens, start) in enumerate(zip(rows, starts))]


def evaluate_mmlu(loaded: LoadedModel, num_fewshot: int = 5, limit: int | None = None) -> MmluScore:
    questions = build_questions(num_fewshot)[:limit]
    correct = 0
    for question in questions:
        losses = continuation_losses(loaded, question.prompt, CHOICES)
        correct += int(losses.index(min(losses)) == question.answer)
    return MmluScore(accuracy=correct / len(questions), correct=correct, total=len(questions))
