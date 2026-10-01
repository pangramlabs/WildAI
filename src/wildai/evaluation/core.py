# Adapted from nanochat (https://github.com/karpathy/nanochat), Copyright (c) 2025 Andrej Karpathy, MIT License; see wildai/training/NOTICE.
"""The CORE metric (DCLM, https://arxiv.org/abs/2406.11794) on nanochat's public eval bundle.

Each task is scored by in-context likelihood: multiple-choice and schema tasks pick the option with the lowest mean
loss over its continuation, language-modelling tasks require the greedy continuation to match exactly. A task's
accuracy is centered against its random baseline, `(acc - baseline) / (1 - baseline)`, and CORE is the mean over
tasks. The bundle (`core.yaml`, `eval_meta_data.csv`, `eval_data/`) is downloaded on first use.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from jinja2 import Template
from pydantic import BaseModel, ConfigDict

from wildai.evaluation.runtime import LoadedModel

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

MC_TEMPLATE = Template(
    """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
)
SCHEMA_TEMPLATE = Template(
    """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
)
LM_TEMPLATE = Template(
    """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
)


class TaskScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    accuracy: float
    centered: float
    correct: int
    total: int


class CoreScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    core: float
    tasks: dict[str, TaskScore]


@dataclass(frozen=True)
class Task:
    label: str
    task_type: str
    num_fewshot: int
    continuation_delimiter: str
    data: list[dict[str, Any]]
    random_baseline: float  # percent


def ensure_bundle(bundle_dir: Path) -> Path:
    """Download and unpack the eval bundle into `bundle_dir` if it is not there yet."""
    if (bundle_dir / "core.yaml").exists():
        return bundle_dir
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "eval_bundle.zip"
        urllib.request.urlretrieve(EVAL_BUNDLE_URL, archive)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(tmp)
        bundle_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(Path(tmp) / "eval_bundle"), bundle_dir)
    return bundle_dir


def load_tasks(bundle_dir: Path, max_per_task: int = -1) -> list[Task]:
    config = yaml.safe_load((bundle_dir / "core.yaml").read_text(encoding="utf-8"))
    with (bundle_dir / "eval_meta_data.csv").open(encoding="utf-8") as handle:
        baselines = {row["Eval Task"]: float(row["Random baseline"]) for row in csv.DictReader(handle)}
    tasks = []
    for entry in config["icl_tasks"]:
        lines = (bundle_dir / "eval_data" / entry["dataset_uri"]).read_text(encoding="utf-8").splitlines()
        data = [json.loads(line) for line in lines if line.strip()]
        random.Random(1337).shuffle(data)  # fixed order so --max-per-task subsamples consistently
        if max_per_task > 0:
            data = data[:max_per_task]
        tasks.append(
            Task(
                entry["label"],
                entry["icl_task_type"],
                entry["num_fewshot"][0],
                entry.get("continuation_delimiter", " "),
                data,
                baselines[entry["label"]],
            )
        )
    return tasks


def _common_length(sequences: list[list[int]], from_end: bool) -> int:
    shortest = min(len(s) for s in sequences)
    for i in range(shortest):
        index = -1 - i if from_end else i
        if any(s[index] != sequences[0][index] for s in sequences):
            return i
    return shortest


def _prompts(task: Task, item: dict[str, Any], fewshot: list[dict[str, Any]]) -> list[str]:
    context = {"fewshot_examples": fewshot, "continuation_delimiter": task.continuation_delimiter, "item": item}
    if task.task_type == "multiple_choice":
        return [MC_TEMPLATE.render(choice=choice, **context) for choice in item["choices"]]
    if task.task_type == "schema":
        return [SCHEMA_TEMPLATE.render(context=option, **context) for option in item["context_options"]]
    if task.task_type == "language_modeling":
        return [LM_TEMPLATE.render(include_continuation=False, **context).strip(), LM_TEMPLATE.render(include_continuation=True, **context)]
    raise ValueError(f"unsupported task type {task.task_type}")


def _spans(task: Task, tokens: list[list[int]]) -> tuple[list[list[int]], list[int], list[int]]:
    """Token sequences to score and the [start, end) of the scored continuation in each."""
    if task.task_type == "multiple_choice":  # shared context, different continuations
        start = _common_length(tokens, from_end=False)
        return tokens, [start] * len(tokens), [len(t) for t in tokens]
    if task.task_type == "schema":  # different contexts, shared continuation
        suffix = _common_length(tokens, from_end=True)
        return tokens, [len(t) - suffix for t in tokens], [len(t) for t in tokens]
    without, with_continuation = tokens
    assert with_continuation[: len(without)] == without, "the prompt without the continuation must be a prefix"
    return [with_continuation], [len(without)], [len(with_continuation)]


@torch.no_grad()
def evaluate_example(loaded: LoadedModel, task: Task, index: int) -> bool:
    item = task.data[index]
    fewshot: list[dict[str, Any]] = []
    if task.num_fewshot > 0:
        rng = random.Random(1234 + index)
        fewshot = [task.data[i] for i in rng.sample([i for i in range(len(task.data)) if i != index], task.num_fewshot)]
    tokens = loaded.tokenizer.encode_documents(_prompts(task, item, fewshot))
    sequences, starts, ends = _spans(task, tokens)
    # Like nanochat, prompts are not cropped to the training context; the rotary tables cover 10x that length.
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), loaded.tokenizer.bos_id, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        ids[row, : len(sequence)] = torch.tensor(sequence)
    ids = ids.to(loaded.device)
    logits = loaded.model(ids)
    losses = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), torch.roll(ids, -1, dims=1).view(-1), reduction="none").view(ids.shape)
    if task.task_type == "language_modeling":
        start, end = starts[0], ends[0]
        return bool(torch.all(logits.argmax(dim=-1)[0, start - 1 : end - 1] == ids[0, start:end]).item())
    mean_losses = [losses[row, start - 1 : end - 1].mean().item() for row, (start, end) in enumerate(zip(starts, ends))]
    return mean_losses.index(min(mean_losses)) == item["gold"]


def evaluate_core(loaded: LoadedModel, bundle_dir: Path, max_per_task: int = -1) -> CoreScore:
    scores: dict[str, TaskScore] = {}
    for task in load_tasks(ensure_bundle(bundle_dir), max_per_task):
        correct = sum(evaluate_example(loaded, task, i) for i in range(len(task.data)))
        accuracy = correct / max(1, len(task.data))
        baseline = 0.01 * task.random_baseline
        scores[task.label] = TaskScore(accuracy=accuracy, centered=(accuracy - baseline) / (1 - baseline), correct=correct, total=len(task.data))
        print(f"CORE {task.label}: accuracy {accuracy:.4f}, centered {scores[task.label].centered:.4f}", flush=True)
    return CoreScore(core=sum(s.centered for s in scores.values()) / len(scores), tasks=scores)
