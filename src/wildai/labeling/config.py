"""Labeler settings (``configs/data/labeling.yaml``); models are pinned to exact Hugging Face revisions."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from wildai.data.config import StrictModel

PANGRAM_PUBLIC_API = "https://text.external-api.pangram.com"


class PangramSettings(StrictModel):
    base_url: str = PANGRAM_PUBLIC_API
    model: str = "default"
    """Model selector; ``GET /models`` lists the selectors a key may use."""
    expected_version: str | None = None
    """If set, every result must report this model version (e.g. ``3.3.2``)."""
    max_chars: int = Field(200_000, gt=0)
    words_per_unit: int = Field(100, gt=0)
    max_units_per_job: int = Field(1000, gt=0)
    max_items_per_job: int = Field(1000, gt=0)
    max_outstanding_jobs: int = Field(4, gt=0)
    poll_interval: float = Field(15.0, gt=0)
    keep_windows: bool = True
    """Store per-window scores in the sidecar (the release exports them only if asked to)."""


class ModelPin(StrictModel):
    repo: str
    revision: str


class EditLensSettings(StrictModel):
    base_model: ModelPin = ModelPin(repo="meta-llama/Llama-3.2-3B", revision="13afe5124825b4f3751f836b40dafda64c1ed062")
    adapter: ModelPin = ModelPin(repo="pangram/editlens_Llama-3.2-3B", revision="b5f8044f631f5b455eafbcb569dcf175f2b0726d")
    quantization: Literal["nf4", "bf16"] = "nf4"
    """``nf4`` loads the base model in 4-bit NF4, the setting the adapter was trained with (QLoRA)."""
    max_length: int = Field(512, gt=0)
    """Tokens per window."""
    windows: int = Field(3, gt=0)
    """Evenly spaced windows scored per document (first, middle and last for 3); probabilities are averaged."""
    batch_size: int = Field(128, gt=0)
    read_rows: int = Field(4096, gt=0)
    """Documents tokenized together, so windows of similar length can share a padded batch."""


class WebOrganizerSettings(StrictModel):
    topic: ModelPin = ModelPin(repo="WebOrganizer/TopicClassifier", revision="8d158c9d514cdc21a7c8e9bd94e5dc483d49e024")
    format: ModelPin = ModelPin(repo="WebOrganizer/FormatClassifier", revision="67e7104a18a512a9ade9e03dca0cbf36eb977f3d")
    max_length: int = Field(8192, gt=0)
    batch_size: int = Field(32, gt=0)
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    memory_efficient_attention: bool = True
    """xformers attention on unpadded inputs (needs ``xformers`` and a GPU). The checkpoints' remote code builds its
    padded-attention mask with a helper Transformers 5.10+ no longer provides, so ``False`` needs an older Transformers."""


class LabelingConfig(StrictModel):
    editlens: EditLensSettings = EditLensSettings()
    pangram: PangramSettings = PangramSettings()
    weborganizer: WebOrganizerSettings = WebOrganizerSettings()
