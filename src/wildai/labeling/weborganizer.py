"""WebOrganizer topic and format labels (24 topics, 24 formats; https://huggingface.co/WebOrganizer).

Each classifier reads ``url + "\\n\\n" + text`` (truncated to 8,192 tokens) and we keep the most likely class and its
probability. The classifiers run the ``gte-base-en-v1.5`` architecture through ``trust_remote_code`` at pinned revisions.

    CUDA_VISIBLE_DEVICES=0 python -m wildai.labeling.weborganizer --input-dir data/monthly_sample \
        --output-dir data/labels/weborganizer/monthly_sample

Sidecar columns (:class:`WebOrganizerLabels`): ``topic``, ``topic_score``, ``format``, ``format_score``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from wildai.data.arrow_schema import ArrowType, table_from_models
from wildai.data.config import default_config, load_config
from wildai.labeling.config import LabelingConfig, ModelPin, WebOrganizerSettings
from wildai.labeling.sidecar import label_directory

Float32 = Annotated[float, ArrowType(pa.float32())]


class WebOrganizerLabels(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    topic: str
    topic_score: Float32
    format: str
    format_score: Float32


def classifier_input(url: str, text: str) -> str:
    return f"{url}\n\n{text}"


class Classifier:
    """One WebOrganizer classifier at a pinned revision."""

    def __init__(self, pin: ModelPin, settings: WebOrganizerSettings, device: str) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.settings = settings
        self.tokenizer = AutoTokenizer.from_pretrained(pin.repo, revision=pin.revision, trust_remote_code=True)
        config = AutoConfig.from_pretrained(pin.repo, revision=pin.revision, trust_remote_code=True)
        config.use_memory_efficient_attention = settings.memory_efficient_attention
        config.unpad_inputs = settings.memory_efficient_attention
        dtype = torch.bfloat16 if settings.dtype == "bfloat16" else torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(pin.repo, revision=pin.revision, config=config,
                                                                   trust_remote_code=True, dtype=dtype)
        self.model = model.to(device).eval()
        self._rebuild_position_buffers()
        self.labels = {int(k): v for k, v in self.model.config.id2label.items()}

    def _rebuild_position_buffers(self) -> None:
        """Recompute the position ids and rotary caches, which the checkpoint does not store.

        They are non-persistent buffers; Transformers 5 allocates them without running the constructor code that fills
        them, which gives random (float32) or NaN (bfloat16) predictions unless they are rebuilt on the model's device.
        """

        embeddings = self.model.new.embeddings
        parameter = next(self.model.parameters())
        if hasattr(embeddings, "position_ids"):
            positions = int(self.model.config.max_position_embeddings)
            embeddings.register_buffer("position_ids", self.torch.arange(positions, device=parameter.device),
                                       persistent=False)
        rotary = getattr(embeddings, "rotary_emb", None)
        if rotary is not None:
            length = round(float(rotary.max_position_embeddings) * float(getattr(rotary, "scaling_factor", 1.0)))
            rotary._set_cos_sin_cache(length, device=parameter.device, dtype=parameter.dtype)

    def classify(self, inputs: Sequence[str]) -> list[tuple[str, float]]:
        order = sorted(range(len(inputs)), key=lambda i: len(inputs[i]))  # similar lengths share a padded batch
        out: list[tuple[str, float]] = [("", 0.0)] * len(inputs)
        for start in range(0, len(order), self.settings.batch_size):
            chunk = order[start:start + self.settings.batch_size]
            batch = self.tokenizer([inputs[i] for i in chunk], padding=True, truncation=True,
                                   max_length=self.settings.max_length, return_tensors="pt").to(self.model.device)
            with self.torch.inference_mode():
                probs = self.torch.softmax(self.model(**batch).logits.float(), dim=-1)
            if not self.torch.isfinite(probs).all():
                raise RuntimeError("non-finite WebOrganizer probabilities")
            values, indices = probs.max(dim=-1)
            for i, index, value in zip(chunk, indices.tolist(), values.tolist()):
                out[i] = (self.labels[index], float(value))
        return out


class WebOrganizer:
    def __init__(self, settings: WebOrganizerSettings, device: str = "cuda:0") -> None:
        self.topic = Classifier(settings.topic, settings, device)
        self.format = Classifier(settings.format, settings, device)

    def label(self, ids: Sequence[str], urls: Sequence[str], texts: Sequence[str]) -> list[WebOrganizerLabels]:
        inputs = [classifier_input(u or "", t or "") for u, t in zip(urls, texts, strict=True)]
        topics, formats = self.topic.classify(inputs), self.format.classify(inputs)
        return [WebOrganizerLabels(id=i, topic=t, topic_score=ts, format=f, format_score=fs)
                for i, (t, ts), (f, fs) in zip(ids, topics, formats)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=default_config("labeling.yaml"))
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args(argv)
    organizer = WebOrganizer(load_config(args.config, LabelingConfig).weborganizer, args.device)

    def label(table: pa.Table, _shard: Path) -> pa.Table:
        rows = organizer.label(table["id"].to_pylist(), table["url"].to_pylist(), table["text"].to_pylist())
        return table_from_models(rows, WebOrganizerLabels)

    label_directory(args.input_dir, args.output_dir, ["id", "url", "text"], label,
                    shard_index=args.shard_index, num_shards=args.num_shards)


if __name__ == "__main__":
    main()
