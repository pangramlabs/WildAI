"""FW22: plain FineWeb text from the 2021 crawls, before ChatGPT and years before the training crawls.

Each 2021 dump contributes an equal character budget, read in seeded random row-group order (documents of at least
``min_chars`` characters); the selected documents are then ordered by a keyed hash of their id. No detector or quality
classifier is involved.
"""

from __future__ import annotations

from pathlib import Path

from wildai.data.evalsets.common import EvalSetManifest, FineWebYearSet, take_chars, write_eval_set
from wildai.data.fineweb_hf import FINEWEB_REPO, ParquetDump, iter_row_groups
from wildai.data.hashing import keyed_hash, seeded_key


def dump_budgets(dumps: list[str], target_chars: int) -> dict[str, int]:
    base, extra = divmod(target_chars, len(dumps))
    return {dump: base + (i < extra) for i, dump in enumerate(sorted(dumps))}


def build_fw22(spec: FineWebYearSet, output: Path, name: str = "fw22") -> EvalSetManifest:
    documents: list[dict[str, str]] = []
    for dump, budget in dump_budgets(spec.dumps, spec.target_chars).items():
        source = {dump: ParquetDump.fineweb(dump, spec.revision)}
        rows = (row for table in iter_row_groups(source, spec.seed, ("id", "text")) for row in table.to_pylist())
        documents.extend(take_chars(rows, budget, spec.min_chars, text_of=lambda r: r["text"]))
    key = seeded_key("fw22-order", spec.seed)
    documents.sort(key=lambda d: (keyed_hash(d["id"], key), d["id"]))
    selection = (f"{spec.target_chars:,} characters split equally over {len(spec.dumps)} dumps, seeded row-group order "
                 f"(seed {spec.seed}), then ordered by keyed hash")
    return write_eval_set(output, name, [d["text"] for d in documents], source=FINEWEB_REPO, revision=spec.revision,
                          selection=selection)
