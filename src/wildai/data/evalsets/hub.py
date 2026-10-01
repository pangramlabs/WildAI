"""C4 and Cosmopedia: the first ``target_chars`` characters of a pinned Hugging Face dataset, in its streaming order.

Neither set can overlap the training pools: C4 is built from a 2019 crawl and Cosmopedia is synthetic.
"""

from __future__ import annotations

from pathlib import Path

from wildai.data.evalsets.common import EvalSetManifest, HubTextSet, take_chars, write_eval_set


def build_hub_set(name: str, spec: HubTextSet, output: Path) -> EvalSetManifest:
    from datasets import load_dataset

    stream = load_dataset(spec.repo, spec.config, split=spec.split, streaming=True, revision=spec.revision)
    texts = list(take_chars((row["text"] for row in stream), spec.target_chars, spec.min_chars))
    selection = (f"first {spec.target_chars:,} characters of documents with at least {spec.min_chars} characters, "
                 f"{spec.config}/{spec.split} in streaming order")
    return write_eval_set(output, name, texts, source=spec.repo, revision=spec.revision, selection=selection)
