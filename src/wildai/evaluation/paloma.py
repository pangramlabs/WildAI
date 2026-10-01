"""Paloma: document-bounded bits per byte on the 16 sources of Paloma's validation split, macro-averaged.

Each source is a directory `<eval-dir>/paloma/<source>/` of parquet files with `text` and `domain` columns (the
official validation split, domain labels kept). Every document is scored on its own, as `[<|bos|>, *tokens]` split
into windows of `sequence_len + 1` tokens that share one boundary token, so every token after the first is predicted
exactly once and attention never crosses documents. A domain's BPB is its total nats over its total bytes; a source's
score is the unweighted mean over its domains; the paper's Paloma number is the unweighted mean over the 16 sources.

    python -m wildai.evaluation.paloma --model 268m-g072-ai-r0.5 --eval-dir evalsets/ --output paloma.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
import torch
from pydantic import BaseModel, ConfigDict

from wildai.evaluation.runtime import LoadedModel, add_model_arguments, load_from_arguments

PALOMA_SOURCES = (
    "c4",
    "mc4_en",
    "wikitext_103",
    "ptb",
    "redpajama",
    "falcon_refinedweb",
    "dolma",
    "m2d2_s2orc",
    "m2d2_wikipedia",
    "c4_100_domains",
    "reddit_100",
    "programming",
    "twitteraae",
    "manosphere",
    "gab",
    "4chan",
)


class DomainScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    bpb: float
    docs: int
    windows: int
    bytes: int
    nats: float


class SourceScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    macro_bpb: float
    domains: dict[str, DomainScore]


def windows(tokens: list[int], sequence_len: int) -> Iterator[list[int]]:
    """Windows of up to `sequence_len + 1` tokens whose prediction targets do not overlap."""
    for start in range(0, max(len(tokens) - 1, 0), sequence_len):
        window = tokens[start : start + sequence_len + 1]
        if len(window) >= 2:
            yield window


@torch.no_grad()
def score_texts(loaded: LoadedModel, texts: list[str], batch_size: int) -> DomainScore:
    bos = loaded.tokenizer.bos_id
    spans = [w for doc in loaded.tokenizer.encode_documents(texts) for w in windows(doc, loaded.sequence_len)]
    nats, total_bytes = 0.0, 0
    for start in range(0, len(spans), batch_size):
        batch = spans[start : start + batch_size]
        width = max(len(w) - 1 for w in batch)
        inputs = torch.full((len(batch), width), bos, dtype=torch.long)
        targets = torch.full((len(batch), width), -1, dtype=torch.long)
        for row, window in enumerate(batch):
            inputs[row, : len(window) - 1] = torch.tensor(window[:-1])
            targets[row, : len(window) - 1] = torch.tensor(window[1:])
        inputs, targets = inputs.to(loaded.device), targets.to(loaded.device)
        loss = loaded.model(inputs, targets, loss_reduction="none").view_as(targets)
        valid = targets >= 0
        target_bytes = torch.where(valid, loaded.token_bytes[targets.clamp_min(0)], torch.zeros_like(targets))
        nats += float((loss * (valid & (target_bytes > 0))).sum().item())
        total_bytes += int(target_bytes.sum().item())
    return DomainScore(bpb=nats / (math.log(2) * total_bytes), docs=len(texts), windows=len(spans), bytes=total_bytes, nats=nats)


def score_source(loaded: LoadedModel, directory: Path, batch_size: int) -> SourceScore:
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in sorted(directory.glob("*.parquet")):
        table = pq.read_table(path, columns=["text", "domain"])
        for text, domain in zip(table["text"].to_pylist(), table["domain"].to_pylist()):
            grouped[str(domain)].append(text)
    if not grouped:
        raise FileNotFoundError(f"no parquet files in {directory}")
    domains = {domain: score_texts(loaded, grouped[domain], batch_size) for domain in sorted(grouped)}
    return SourceScore(macro_bpb=sum(d.bpb for d in domains.values()) / len(domains), domains=domains)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--eval-dir", type=Path, required=True, help="directory containing paloma/<source>/")
    parser.add_argument("--sources", nargs="+", default=list(PALOMA_SOURCES), choices=PALOMA_SOURCES)
    parser.add_argument("--batch-size", type=int, default=8, help="windows per forward pass")
    parser.add_argument("--output", type=Path, help="write the scores as JSON")
    args = parser.parse_args()

    loaded = load_from_arguments(args)
    sources: dict[str, SourceScore] = {}
    for source in args.sources:
        sources[source] = score_source(loaded, args.eval_dir / "paloma" / source, args.batch_size)
        print(f"paloma_{source}: {sources[source].macro_bpb:.6f} bpb ({len(sources[source].domains)} domains)", flush=True)
    macro = sum(s.macro_bpb for s in sources.values()) / len(sources)
    print(f"paloma (mean of {len(sources)} sources): {macro:.6f} bpb")
    if args.output:
        payload = {
            "model": args.model or str(args.checkpoint),
            "step": loaded.step,
            "paloma": macro,
            "sources": {k: v.model_dump() for k, v in sources.items()},
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
