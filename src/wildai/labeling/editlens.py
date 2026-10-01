"""EditLens labels: how much AI editing a document shows, in four buckets from human-written (0) to AI-generated (3).

The public model ``pangram/editlens_Llama-3.2-3B`` is a LoRA adapter plus a LayerNorm+Linear bucket head on
``meta-llama/Llama-3.2-3B`` (both gated on Hugging Face; the adapter is CC BY-NC-SA 4.0, non-commercial use only, and the
base model is under the Llama 3.2 Community License). We load the base in 4-bit NF4 as the adapter was trained, score up
to three evenly spaced 512-token windows per document and average the bucket probabilities (see
:mod:`wildai.labeling.editlens_text`).

    CUDA_VISIBLE_DEVICES=0 python -m wildai.labeling.editlens --input-dir data/documents/fineweb/CC-MAIN-2025-26 \
        --output-dir data/labels/editlens/CC-MAIN-2025-26 --shard-index 0 --num-shards 8

Sidecar columns (:class:`EditLensLabels`): ``editlens_bucket`` (argmax of the averaged probabilities),
``editlens_score`` (expected bucket / (buckets - 1), in [0, 1]), ``editlens_probs`` and ``editlens_windows``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import numpy as np
import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from wildai.data.arrow_schema import ArrowType, table_from_models
from wildai.data.config import default_config, load_config
from wildai.labeling.config import EditLensSettings, LabelingConfig
from wildai.labeling.editlens_text import clean_text, select_windows
from wildai.labeling.sidecar import label_directory

Float32 = Annotated[float, ArrowType(pa.float32())]


class EditLensLabels(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    editlens_bucket: int
    editlens_score: Float32
    editlens_probs: list[Float32]
    editlens_windows: int


def adapter_buckets(settings: EditLensSettings) -> int:
    """Number of buckets, read from the shape of the adapter's saved head."""

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(settings.adapter.repo, "adapter_model.safetensors", revision=settings.adapter.revision)
    with safe_open(path, framework="pt") as weights:
        for name in weights.keys():
            if "score" in name and "linear.weight" in name:
                return int(weights.get_slice(name).get_shape()[0])
    raise ValueError(f"{settings.adapter.repo} has no bucket head")


class EditLensScorer:
    def __init__(self, settings: EditLensSettings, device: str = "cuda:0") -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

        class NormedLinear(torch.nn.Module):
            """EditLens's bucket head: LayerNorm followed by a bias-free linear layer."""

            def __init__(self, hidden: int, buckets: int) -> None:
                super().__init__()
                self.norm = torch.nn.LayerNorm(hidden, device=device, dtype=torch.bfloat16)
                self.linear = torch.nn.Linear(hidden, buckets, bias=False, device=device, dtype=torch.bfloat16)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(self.norm(x))

        self.torch = torch
        self.settings = settings
        self.buckets = adapter_buckets(settings)
        base = settings.base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base.repo, revision=base.revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        kwargs: dict[str, object] = {"num_labels": self.buckets, "dtype": torch.bfloat16}
        if settings.quantization == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                               bnb_4bit_use_double_quant=True,
                                                               bnb_4bit_compute_dtype=torch.bfloat16)
            kwargs["device_map"] = {"": device}
        model = AutoModelForSequenceClassification.from_pretrained(base.repo, revision=base.revision, **kwargs)
        model.config.pad_token_id = self.tokenizer.pad_token_id
        model.score = NormedLinear(model.config.hidden_size, self.buckets)
        model = PeftModel.from_pretrained(model, settings.adapter.repo, revision=settings.adapter.revision)
        self.model = (model if settings.quantization == "nf4" else model.to(device)).eval()

    def _probabilities(self, windows: list[list[int]]) -> np.ndarray:
        batch = self.tokenizer.pad({"input_ids": windows}, return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.inference_mode():
            logits = self.model(**batch).logits.float()
        return self.torch.softmax(logits, dim=-1).cpu().numpy()

    def score(self, ids: Sequence[str], texts: Sequence[str]) -> list[EditLensLabels]:
        encoded = self.tokenizer([clean_text(t or "") for t in texts], add_special_tokens=True)["input_ids"]
        windows, owner = [], []
        for doc, token_ids in enumerate(encoded):
            for window in select_windows(token_ids, self.settings.max_length, self.settings.windows):
                windows.append(window)
                owner.append(doc)
        order = sorted(range(len(windows)), key=lambda i: len(windows[i]))  # similar lengths share a padded batch
        totals = np.zeros((len(texts), self.buckets))
        counts = np.zeros(len(texts), dtype=int)
        for start in range(0, len(order), self.settings.batch_size):
            chunk = order[start:start + self.settings.batch_size]
            probs = self._probabilities([windows[i] for i in chunk])
            np.add.at(totals, [owner[i] for i in chunk], probs)
            np.add.at(counts, [owner[i] for i in chunk], 1)
        mean = totals / np.maximum(counts, 1)[:, None]
        expected = mean @ np.arange(self.buckets) / (self.buckets - 1)
        return [EditLensLabels(id=i, editlens_bucket=int(m.argmax()), editlens_score=float(e), editlens_probs=m.tolist(),
                               editlens_windows=int(c)) for i, m, e, c in zip(ids, mean, expected, counts)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=default_config("labeling.yaml"))
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args(argv)
    settings = load_config(args.config, LabelingConfig).editlens
    scorer = EditLensScorer(settings, args.device)

    def label(table: pa.Table, _shard: Path) -> pa.Table:
        ids, texts = table["id"].to_pylist(), table["text"].to_pylist()
        rows = [row for start in range(0, len(ids), settings.read_rows)
                for row in scorer.score(ids[start:start + settings.read_rows], texts[start:start + settings.read_rows])]
        return table_from_models(rows, EditLensLabels)

    label_directory(args.input_dir, args.output_dir, ["id", "text"], label,
                    shard_index=args.shard_index, num_shards=args.num_shards)


if __name__ == "__main__":
    main()
