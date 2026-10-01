# Adapted from nanochat (https://github.com/karpathy/nanochat), Copyright (c) 2025 Andrej Karpathy, MIT License; see wildai/training/NOTICE.
"""Bits per byte of a checkpoint on the paper's evaluation sets (C4, FW22, FW26, FW26-H, FW26-AI, Cosmopedia).

Each evaluation set is a directory of parquet files with a `text` column, at `<eval-dir>/<target>/` (built by
`wildai.data.evalsets`). Its documents
are read in file order, prefixed with `<|bos|>` and packed into rows of 2,049 tokens with nanochat's BOS best-fit
packer (`wildai.training.packing`, cropped remainders discarded); a step scores 16 rows (16 x 2,048 target tokens).
The score is total nats over total UTF-8 bytes of the scored target tokens (special tokens have 0 bytes), over a fixed
number of steps per target (`configs/evaluation/bpb_targets.json`):

    c4, fw22: 480 steps (15.7M tokens); fw26: 369 (12.1M); fw26_human: 293 (9.6M); fw26_ai: 66 (2.2M); cosmopedia: 564 (18.5M)

Every document is scored at most once: each budget is the number of full steps the set fills, and asking for more fails.

    python -m wildai.evaluation.bpb --model 268m-g072-ai-r0.5 --eval-dir data/eval --targets c4 fw26_ai   # released model
    python -m wildai.evaluation.bpb --checkpoint checkpoints/my-run --eval-dir data/eval                   # local checkpoint
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from pydantic import BaseModel, ConfigDict

from wildai.evaluation.runtime import CONFIG_DIR, LoadedModel, add_model_arguments, load_from_arguments
from wildai.training.packing import BestFitPacker, Piece, row_tokens
from wildai.training.tokenizer import Tokenizer

TOKENIZE_BATCH = 128  # documents per packer refill, as in nanochat's loader


class BpbSuite(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_size: int
    targets: dict[str, int]  # target name -> steps

    @classmethod
    def load(cls, path: Path) -> BpbSuite:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class BpbScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    bpb: float
    steps: int
    tokens: int
    bytes: int
    nats: float


def tokenized_batches(directory: Path, tokenizer: Tokenizer) -> list[list[np.ndarray]]:
    """The set's documents as `[BOS, *tokens]`, grouped 128 at a time within each parquet row group."""
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {directory}")
    batches: list[list[np.ndarray]] = []
    for path in files:
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            texts = parquet.read_row_group(group, columns=["text"]).column("text").to_pylist()
            for start in range(0, len(texts), TOKENIZE_BATCH):
                encoded = tokenizer.encode_documents(texts[start : start + TOKENIZE_BATCH])
                batches.append([np.asarray(ids, dtype=np.int32) for ids in encoded])
    return batches


def documents(batches: list[list[np.ndarray]]) -> Iterator[list[Piece]]:
    """The set's documents, once, in packer refill batches."""
    for batch in batches:
        yield [Piece(tokens, 0) for tokens in batch]


@torch.no_grad()
def score_rows(loaded: LoadedModel, rows: Iterator[list[Piece]], steps: int, batch_size: int) -> BpbScore:
    nats, total_bytes, tokens = 0.0, 0, 0
    for step in range(steps):
        batch = [row for _, row in zip(range(batch_size), rows)]
        if len(batch) < batch_size:
            raise ValueError(f"the evaluation set fills only {step} steps of {batch_size} rows; {steps} requested")
        ids = torch.from_numpy(np.stack([row_tokens(row) for row in batch]).astype(np.int64)).to(loaded.device)
        inputs, targets = ids[:, :-1].contiguous(), ids[:, 1:].contiguous()
        loss = loaded.model(inputs, targets, loss_reduction="none").view(-1)
        target_bytes = loaded.token_bytes[targets.view(-1)]
        nats += float((loss * (target_bytes > 0)).sum().item())
        total_bytes += int(target_bytes.sum().item())
        tokens += targets.numel()
    return BpbScore(
        bpb=nats / (math.log(2) * total_bytes),
        steps=steps,
        tokens=tokens,
        bytes=total_bytes,
        nats=nats,
    )


def score_target(loaded: LoadedModel, directory: Path, steps: int, batch_size: int) -> BpbScore:
    batches = tokenized_batches(directory, loaded.tokenizer)
    packer = BestFitPacker(documents(batches), loaded.sequence_len + 1)
    return score_rows(loaded, iter(packer), steps, batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--eval-dir", type=Path, required=True, help="directory with one subdirectory per target")
    parser.add_argument("--suite", type=Path, default=CONFIG_DIR / "bpb_targets.json", help="targets and their step budgets")
    parser.add_argument("--targets", nargs="+", help="subset of the suite's targets (default: all)")
    parser.add_argument("--output", type=Path, help="write the scores as JSON")
    args = parser.parse_args()

    suite = BpbSuite.load(args.suite)
    loaded = load_from_arguments(args)
    scores: dict[str, BpbScore] = {}
    for target in args.targets or list(suite.targets):
        scores[target] = score_target(loaded, args.eval_dir / target, suite.targets[target], suite.batch_size)
        print(f"{target}: {scores[target].bpb:.6f} bpb ({scores[target].tokens:,} tokens)", flush=True)
    if args.output:
        payload = {"model": args.model or str(args.checkpoint), "step": loaded.step, "scores": {k: v.model_dump() for k, v in scores.items()}}
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
