"""Choose which collected documents Pangram labels.

``editlens`` -- EditLens preselection for the training pools. Every document in an AI bucket (2 or 3 of 0-3) is a
candidate; bucket 1 (lightly AI-edited, ambiguous) is dropped; bucket-0 (human) documents are drawn per crawl, up to
``human_token_budget`` GPT-2 tokens in all, split across crawls in proportion to each crawl's AI-candidate tokens, so the
human and AI candidates cover the same crawl dates. Within a crawl, human candidates are the documents with the smallest
keyed hash of their id.

``natural`` -- a label-blind uniform draw of ``documents`` documents from the given crawls, from which the FW26 evaluation
set is built and the web's AI share is measured; no EditLens involved. (The filtering experiments train on the human and
AI pools mixed at that share, not on this draw.)

    python -m wildai.data.preselect editlens --documents-root data/documents --editlens-root data/labels/editlens \
        --output-dir data/candidates/editlens
    python -m wildai.data.preselect natural --documents-root data/documents/common_crawl --output-dir data/candidates/natural

Output: :class:`wildai.data.schema.CandidateDocument` Parquet, one directory per input directory (``natural``: one).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from wildai.data.arrow_schema import arrow_schema
from wildai.data.config import default_config, load_config
from wildai.data.hashing import keyed_hashes
from wildai.data.parquet_io import ShardedWriter, dataset_dirs, list_shards, mirror, read_joined
from wildai.data.pools_config import EditLensPreselection, NaturalDraw, PoolsConfig
from wildai.data.schema import CandidateDocument, Selection

CANDIDATE_SCHEMA = arrow_schema(CandidateDocument)


def _with_editlens(directory: Path, root: Path, editlens_root: Path) -> Iterator[pa.Table]:
    sidecars = {"editlens": mirror(directory, root, editlens_root)}
    for shard in list_shards(directory):
        yield read_joined(shard, sidecars)


def _candidates(table: pa.Table, selection: Selection) -> pa.Table:
    n = len(table)
    table = table.append_column("selection", pa.array([selection] * n, pa.string()))
    if "editlens_bucket" not in table.column_names:
        table = table.append_column("editlens_bucket", pa.nulls(n, pa.int64()))
        table = table.append_column("editlens_score", pa.nulls(n, pa.float64()))
    return table.select(CANDIDATE_SCHEMA.names).cast(CANDIDATE_SCHEMA)


def human_quotas(ai_tokens: dict[str, int], human_available: dict[str, int], budget: int) -> dict[str, int]:
    """Human-candidate tokens per directory, proportional to its AI-candidate tokens and capped by what it holds."""

    total = sum(ai_tokens.values())
    if total == 0:
        raise ValueError("no AI candidates to match")
    return {d: min(human_available.get(d, 0), round(budget * t / total)) for d, t in ai_tokens.items()}


def hash_threshold(hashes: np.ndarray, tokens: np.ndarray, quota: int) -> int | None:
    """The largest hash to keep so the smallest-hash documents sum to at least ``quota`` tokens (all if they fall short);
    ``None`` keeps nothing."""

    if quota <= 0 or not len(hashes):
        return None
    order = np.argsort(hashes, kind="stable")
    reached = int(np.searchsorted(np.cumsum(tokens[order]), quota))
    return int(hashes[order[min(reached, len(order) - 1)]])


def below(hashes: np.ndarray, threshold: int | None) -> pa.Array:
    if threshold is None:
        return pa.array(np.zeros(len(hashes), dtype=bool))
    return pa.array(hashes <= np.uint64(threshold))


def preselect_editlens(root: Path, editlens_root: Path, output: Path, settings: EditLensPreselection) -> dict[str, dict]:
    directories = dataset_dirs(root)
    ai_tokens: dict[str, int] = defaultdict(int)
    human: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for directory in directories:  # pass 1: AI-candidate tokens and the human candidates' hashes
        name = str(directory.relative_to(root))
        key = f"{settings.key}/{name}"
        hashes, tokens = [], []
        for table in _with_editlens(directory, root, editlens_root):
            bucket = table["editlens_bucket"]
            ai_tokens[name] += int(pc.sum(table.filter(pc.greater_equal(bucket, settings.ai_min_bucket))["token_count"]).as_py() or 0)
            humans = table.filter(pc.equal(bucket, settings.human_bucket))
            hashes.append(keyed_hashes(humans["id"].to_pylist(), key))
            tokens.append(humans["token_count"].to_numpy())
        human[name] = (np.concatenate(hashes), np.concatenate(tokens))
    quotas = human_quotas(ai_tokens, {d: int(t.sum()) for d, (_h, t) in human.items()}, settings.human_token_budget)
    summary = {}
    for directory in directories:  # pass 2: write AI candidates and the human draw
        name = str(directory.relative_to(root))
        key = f"{settings.key}/{name}"
        threshold = hash_threshold(*human[name], quotas[name])
        counts = {"ai": 0, "human": 0}
        with ShardedWriter(output / name, CANDIDATE_SCHEMA) as writer:
            for table in _with_editlens(directory, root, editlens_root):
                bucket = table["editlens_bucket"]
                ai = table.filter(pc.greater_equal(bucket, settings.ai_min_bucket))
                humans = table.filter(pc.equal(bucket, settings.human_bucket))
                humans = humans.filter(below(keyed_hashes(humans["id"].to_pylist(), key), threshold))
                writer.write(_candidates(ai, "editlens_ai"))
                writer.write(_candidates(humans, "editlens_human"))
                counts["ai"] += len(ai)
                counts["human"] += len(humans)
        summary[name] = {**counts, "ai_tokens": ai_tokens[name], "human_token_quota": quotas[name]}
    return summary


def preselect_natural(root: Path, output: Path, settings: NaturalDraw) -> int:
    directories = [d for d in dataset_dirs(root) if d.name in settings.dumps]
    if {d.name for d in directories} != set(settings.dumps):
        raise FileNotFoundError(f"missing crawls under {root}: {sorted(set(settings.dumps) - {d.name for d in directories})}")
    shards = [shard for directory in directories for shard in list_shards(directory)]
    hashes = np.concatenate([keyed_hashes(pq.read_table(s, columns=["id"])["id"].to_pylist(), settings.key) for s in shards])
    ones = np.ones(len(hashes), dtype=np.int64)  # a quota of documents is a quota of unit weights
    threshold = hash_threshold(hashes, ones, settings.documents)
    written = 0
    with ShardedWriter(output, CANDIDATE_SCHEMA) as writer:
        for shard in shards:
            table = pq.read_table(shard)
            chosen = table.filter(below(keyed_hashes(table["id"].to_pylist(), settings.key), threshold))
            writer.write(_candidates(chosen, "natural"))
            written += len(chosen)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["editlens", "natural"])
    parser.add_argument("--config", type=Path, default=default_config("pools.yaml"))
    parser.add_argument("--documents-root", type=Path, required=True)
    parser.add_argument("--editlens-root", type=Path, help="EditLens sidecars, mirroring --documents-root (editlens mode)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config, PoolsConfig)
    if args.mode == "editlens":
        if args.editlens_root is None:
            parser.error("editlens mode needs --editlens-root")
        for name, counts in preselect_editlens(args.documents_root, args.editlens_root, args.output_dir, config.editlens).items():
            print(f"{name}: {counts['ai']:,} AI and {counts['human']:,} human candidates")
    else:
        print(f"natural: {preselect_natural(args.documents_root, args.output_dir, config.natural):,} documents")


if __name__ == "__main__":
    main()
