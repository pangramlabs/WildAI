"""Paloma: the validation split of its 16 sources, one evaluation set per source with each document's domain.

Paloma is gated on Hugging Face (accept the AI2 ImpACT license at https://huggingface.co/datasets/allenai/paloma first).
A document's domain is its ``subdomain`` field, else its ``source``; losses are macro-averaged over domains, then sources.
"""

from __future__ import annotations

from pathlib import Path

from wildai.data.evalsets.common import EvalSetManifest, PalomaSet, write_eval_set


def build_paloma(spec: PalomaSet, output: Path) -> list[EvalSetManifest]:
    from datasets import load_dataset

    manifests = []
    for name, subset in spec.sources.items():
        rows = load_dataset(spec.repo, subset, split=spec.split, streaming=True, revision=spec.revision)
        texts, domains = [], []
        for row in rows:
            if row.get("text"):
                texts.append(row["text"])
                domains.append(str(row.get("subdomain") or row.get("source") or subset))
        manifests.append(write_eval_set(output, f"paloma_{name}", texts, source=f"{spec.repo}/{subset}",
                                        revision=spec.revision, selection=f"the whole {spec.split} split", domains=domains))
    return manifests
