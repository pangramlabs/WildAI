"""Shared setup for the evaluation tools: a model with its tokenizer, on a device.

A model is either a released model, by name (downloaded from ``pangram/WildAI-models``) or as a local copy of its folder,
or a training checkpoint written by ``wildai.training.train``. The tokenizer defaults to the model's own (released
models carry ``tokenizer.json``) or else the released one.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from wildai.training.checkpoint import load_model
from wildai.training.model import GPT
from wildai.training.released import load_released, released_config
from wildai.training.tokenizer import JSON_FILE, Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs" / "evaluation"


@dataclass(frozen=True)
class LoadedModel:
    model: GPT
    tokenizer: Tokenizer
    token_bytes: torch.Tensor
    device: torch.device
    step: int

    @property
    def sequence_len(self) -> int:
        return self.model.config.sequence_len


def released_folder(model: str) -> Path:
    """A local released-model folder, or the named model downloaded from the Hub."""
    if Path(model).is_dir():
        return Path(model)
    from wildai.hub.download import model_dir

    return model_dir(model)


def load(model: str | None = None, checkpoint: Path | None = None, tokenizer: Path | None = None, device: str | None = None) -> LoadedModel:
    """Load a released ``model`` (name or folder) or a training ``checkpoint``; exactly one must be given."""
    if (model is None) == (checkpoint is None):
        raise ValueError("give exactly one of model and checkpoint")
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_float32_matmul_precision("high")
    if checkpoint is not None:
        gpt, meta = load_model(checkpoint, target)
        step = int(meta["step"])
    else:
        folder = released_folder(model)
        gpt, step = load_released(folder, target), int(released_config(folder)["release"]["steps"])
        tokenizer = tokenizer or folder / JSON_FILE
    loaded = Tokenizer.load(tokenizer)
    if loaded.vocab_size != gpt.config.vocab_size:
        raise ValueError(f"tokenizer has {loaded.vocab_size} tokens, the model {gpt.config.vocab_size}")
    return LoadedModel(gpt, loaded, loaded.token_bytes(target), target, step)


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--model", help="a released model: its name (results/models.csv), downloaded from the Hub, or a local copy of its folder")
    which.add_argument("--checkpoint", type=Path, help="a training checkpoint: its directory (latest step) or a model_<step>.pt file")
    parser.add_argument("--tokenizer", type=Path, help="tokenizer.json or tokenizer.pkl, or a directory with one (default: the model's own, else the released one)")
    parser.add_argument("--device", default=None, help="torch device (default: cuda if available)")


def load_from_arguments(args: argparse.Namespace) -> LoadedModel:
    return load(args.model, args.checkpoint, args.tokenizer, args.device)
