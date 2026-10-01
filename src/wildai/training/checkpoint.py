# Adapted from nanochat (https://github.com/karpathy/nanochat), Copyright (c) 2025 Andrej Karpathy, MIT License; see NOTICE.
"""Save and load model checkpoints.

A checkpoint directory holds `model_<step:06d>.pt` (the model's `state_dict`: fp32 matrices and scalars, embedding
tables in the compute dtype) and `meta_<step:06d>.json`, whose `model_config` field describes the architecture.
Released models are Hugging Face folders instead; `wildai.training.released` loads those.

    model, meta = load_model(Path("checkpoints/19.9m-g001-control"), torch.device("cuda"))
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import torch

from wildai.training.model import GPT, GPTConfig, compute_dtype_for

_STEP = re.compile(r"model_(\d+)\.pt$")


def checkpoint_paths(directory: Path, step: int) -> tuple[Path, Path]:
    return directory / f"model_{step:06d}.pt", directory / f"meta_{step:06d}.json"


def latest_step(directory: Path) -> int:
    steps = [int(m.group(1)) for p in directory.glob("model_*.pt") if (m := _STEP.search(p.name))]
    if not steps:
        raise FileNotFoundError(f"no model_<step>.pt in {directory}")
    return max(steps)


def resolve(path: Path, step: int | None = None) -> tuple[Path, Path]:
    """Model and meta paths for a checkpoint directory (latest step unless given) or a `model_<step>.pt` file."""
    path = Path(path)
    if path.is_file():
        match = _STEP.search(path.name)
        if not match:
            raise ValueError(f"expected a model_<step>.pt file, got {path}")
        return checkpoint_paths(path.parent, int(match.group(1)))
    return checkpoint_paths(path, latest_step(path) if step is None else step)


def read_meta(path: Path, step: int | None = None) -> dict[str, Any]:
    return json.loads(resolve(path, step)[1].read_text(encoding="utf-8"))


def load_model(path: Path, device: torch.device, step: int | None = None, compute_dtype: torch.dtype | None = None) -> tuple[GPT, dict[str, Any]]:
    """Load a checkpoint for evaluation; returns the model (in eval mode) and its metadata."""
    model_path, meta_path = resolve(path, step)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    config = GPTConfig.from_dict(meta["model_config"])
    dtype = compute_dtype or compute_dtype_for(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    if dtype == torch.float32:
        state = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in state.items()}
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights(dtype)  # builds the rotary tables; every parameter is then replaced by the checkpoint's
    model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    return model, meta


def _atomic_write(path: Path, write: Any) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_checkpoint(directory: Path, step: int, model: GPT, meta: dict[str, Any]) -> Path:
    """Write the model weights and metadata (with `step` and `model_config` filled in); returns the model path."""
    directory.mkdir(parents=True, exist_ok=True)
    model_path, meta_path = checkpoint_paths(directory, step)
    meta = {**meta, "step": step, "model_config": model.config.to_dict()}
    _atomic_write(model_path, lambda tmp: torch.save(model.state_dict(), tmp))
    _atomic_write(meta_path, lambda tmp: Path(tmp).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8"))
    return model_path
