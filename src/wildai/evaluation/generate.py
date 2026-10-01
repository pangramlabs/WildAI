"""Sample continuations of a prompt set from a checkpoint (the paper's generation appendix).

Two prompt sets are configured in `configs/evaluation/generation/`:
- `writingprompts`: 500 WritingPrompts test prompts, 8 samples each, temperature 1.0, top-p 0.95, top-k 20, up to 256
  new tokens, seed 8201;
- `webtext`: 5,000 WebText test articles cut to their first 35 GPT-2 tokens, 1 sample each, temperature 1.0,
  top-p 0.95, up to 1,024 new tokens, seed 21021.

Each prompt is used verbatim as `[<|bos|>, *tokens]`. Requests (prompt x sample, in file order) are grouped by prompt
length, shortest first, into batches, and one random generator seeded with the task seed is used throughout, so the
output depends on the batch size. Generation stops at `<|bos|>` or the token limit. Output: one JSON line per
continuation with `prompt_id`, `sample_index`, `text`, `token_ids` and `finish_reason`.

    python -m wildai.evaluation.generate --model 268m-g072-ai-r0.5 --task writingprompts --output gens.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict

from wildai.evaluation.runtime import CONFIG_DIR, LoadedModel, add_model_arguments, load_from_arguments
from wildai.evaluation.sampling import Sampling, generate_batch

TASKS_DIR = CONFIG_DIR / "generation"


class GenerationTask(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    prompts: str  # relative to the task file
    prompts_sha256: str
    samples_per_prompt: int
    temperature: float
    top_p: float | None
    top_k: int | None
    max_new_tokens: int
    seed: int

    @property
    def sampling(self) -> Sampling:
        return Sampling(self.temperature, self.top_p, self.top_k)


class Prompt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    prompt: str


class Generation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    prompt_id: str
    sample_index: int
    text: str
    token_ids: list[int] = []
    finish_reason: str = "length"


def load_task(name_or_path: str) -> tuple[GenerationTask, list[Prompt]]:
    path = Path(name_or_path) if name_or_path.endswith(".json") else TASKS_DIR / f"{name_or_path}.json"
    task = GenerationTask.model_validate_json(path.read_text(encoding="utf-8"))
    raw = (path.parent / task.prompts).read_bytes()
    if hashlib.sha256(raw).hexdigest() != task.prompts_sha256:
        raise ValueError(f"{task.prompts} does not match its recorded sha256")
    prompts = [Prompt.model_validate_json(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return task, prompts


def generate(loaded: LoadedModel, task: GenerationTask, prompts: list[Prompt], batch_size: int) -> list[Generation]:
    bos = loaded.tokenizer.bos_id
    requests = [(prompt, sample) for prompt in prompts for sample in range(task.samples_per_prompt)]
    tokens = {prompt.id: [bos, *loaded.tokenizer.encode(prompt.prompt)] for prompt in prompts}
    by_length: dict[int, list[tuple[Prompt, int]]] = defaultdict(list)
    for prompt, sample in requests:
        by_length[len(tokens[prompt.id])].append((prompt, sample))
    rng = torch.Generator(device=loaded.device)
    rng.manual_seed(task.seed)
    outputs: dict[tuple[str, int], list[int]] = {}
    for length in sorted(by_length):
        group = by_length[length]
        for start in range(0, len(group), batch_size):
            batch = group[start : start + batch_size]
            continuations = generate_batch(loaded.model, [tokens[p.id] for p, _ in batch], task.sampling, task.max_new_tokens, {bos}, rng)
            outputs.update({(p.id, sample): ids for (p, sample), ids in zip(batch, continuations)})
    return [
        Generation(
            prompt_id=prompt.id,
            sample_index=sample,
            text=loaded.tokenizer.decode(outputs[prompt.id, sample]),
            token_ids=outputs[prompt.id, sample],
            finish_reason="stop" if len(outputs[prompt.id, sample]) < task.max_new_tokens else "length",
        )
        for prompt, sample in requests
    ]


def read_generations(path: Path) -> list[Generation]:
    return [Generation.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--task", default="writingprompts", help="writingprompts, webtext, or a task JSON file")
    parser.add_argument("--batch-size", type=int, default=32, help="requests per batch (the paper used 32 at 973M and 128 at 268M)")
    parser.add_argument("--limit-prompts", type=int, help="only the first N prompts (for quick checks)")
    parser.add_argument("--output", type=Path, required=True, help="JSON-lines file of generations")
    args = parser.parse_args()

    task, prompts = load_task(args.task)
    loaded = load_from_arguments(args)
    generations = generate(loaded, task, prompts[: args.limit_prompts], args.batch_size)
    with args.output.open("w", encoding="utf-8") as handle:
        for generation in generations:
            handle.write(generation.model_dump_json() + "\n")
    print(f"wrote {len(generations)} generations to {args.output}")


if __name__ == "__main__":
    main()
