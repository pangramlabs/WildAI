"""FW26: current-web text from the natural 2026 pool, with its Pangram-labeled human and AI partitions.

FW26 takes documents from the end of the natural pool's hash order, skipping any document that is also in a training
pool, until ``target_tokens`` GPT-2 tokens are collected. FW26-H and FW26-AI are the
human- and AI-labeled documents of exactly that set. The chosen ids are written to ``_fw26_ids.parquet`` next to the
sets, so training can confirm it never reads them.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from wildai.data.evalsets.common import EvalSetManifest, NaturalTailSet, write_eval_set
from wildai.data.parquet_io import iter_batches, read_dataset, write_table_atomic

IDS_FILE = "_fw26_ids.parquet"
PARTITIONS = {"fw26_human": "human", "fw26_ai": "ai"}


def ids_in_pools(candidates: pa.Array, pools: list[Path]) -> set[str]:
    """The candidate ids that occur in any of the pools (streamed, so the pools never sit in memory)."""

    found: set[str] = set()
    for pool in pools:
        for batch in iter_batches(pool, columns=["id"]):
            found.update(batch.column(0).filter(pc.is_in(batch.column(0), value_set=candidates)).to_pylist())
    return found


def select_tail(natural: pa.Table, target_tokens: int, pools: list[Path]) -> pa.Table:
    """The shortest run from the end of the hash order holding ``target_tokens`` tokens of documents in no pool."""

    order = pc.sort_indices(natural, [("sampling_hash", "descending")])
    tail = natural.take(order)
    excluded: set[str] = set()
    while True:
        eligible = tail.filter(pc.invert(pc.is_in(tail["id"], value_set=pa.array(sorted(excluded), pa.string()))))
        cumulative = pc.cumulative_sum(eligible["token_count"]).to_numpy()
        if not len(cumulative) or cumulative[-1] < target_tokens:
            raise ValueError(f"the natural pool holds fewer than {target_tokens:,} tokens outside the training pools")
        chosen = eligible.slice(0, int((cumulative < target_tokens).sum()) + 1)
        overlap = ids_in_pools(chosen["id"], pools)
        if not overlap:
            return chosen.take(pc.sort_indices(chosen, [("sampling_hash", "ascending")]))
        excluded |= overlap


def build_fw26(spec: NaturalTailSet, natural_pool: Path, training_pools: list[Path], output: Path) -> list[EvalSetManifest]:
    natural = read_dataset(natural_pool, columns=["id", "text", "label", "token_count", "sampling_hash"])
    chosen = select_tail(natural, spec.target_tokens, training_pools)
    write_table_atomic(chosen.select(["id", "label", "token_count", "sampling_hash"]), output / IDS_FILE)
    selection = (f"end of the natural pool's hash order, {spec.target_tokens:,} GPT-2 tokens of documents in none of "
                 f"{len(training_pools)} training pools")
    manifests = [write_eval_set(output, "fw26", chosen["text"].to_pylist(), source="natural pool", revision=None,
                                selection=selection)]
    for name, label in PARTITIONS.items():
        part = chosen.filter(pc.equal(chosen["label"], label))
        manifests.append(write_eval_set(output, name, part["text"].to_pylist(), source="natural pool", revision=None,
                                        selection=f"the {label}-labeled documents of fw26"))
    return manifests
