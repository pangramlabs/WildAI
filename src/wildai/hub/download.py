"""The released models, tokenizer and dataset on the Hugging Face Hub: the default source wherever a tool needs one.

Tools use a local path when one is given and otherwise fetch what they need from these repositories, into the Hugging
Face cache (``HF_HOME``), so each file is downloaded once.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

MODELS_REPO = "pangram/WildAI-models"
DATASET_REPO = "pangram/WildAI"


def model_dir(name: str) -> Path:
    """The folder of one released model (``<name>/`` in the model repository)."""
    folder = Path(snapshot_download(MODELS_REPO, allow_patterns=[f"{name}/*"])) / name
    if not (folder / "model.safetensors").exists():
        raise FileNotFoundError(f"{MODELS_REPO} has no model named {name!r} (names are listed in results/models.csv)")
    return folder


def tokenizer_file() -> Path:
    """The released tokenizer, ``tokenizer.json`` at the root of the model repository."""
    return Path(hf_hub_download(MODELS_REPO, "tokenizer.json"))


def dataset_dir(config: str) -> Path:
    """The parquet files of one config of the released dataset (``<config>/`` in the dataset repository)."""
    folder = Path(snapshot_download(DATASET_REPO, repo_type="dataset", allow_patterns=[f"{config}/**"])) / config
    if not any(folder.rglob("*.parquet")):
        raise FileNotFoundError(f"{DATASET_REPO} has no config {config!r}")
    return folder
