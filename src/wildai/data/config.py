"""Load the YAML configs in ``configs/data/`` into the Pydantic models their stages define."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs" / "data"


class StrictModel(BaseModel):
    """Base for config models: immutable, and unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)


M = TypeVar("M", bound=BaseModel)


def load_config(path: Path, model: type[M]) -> M:
    """Validate a YAML file against ``model``; unknown keys are errors, so typos never pass silently."""

    with path.open(encoding="utf-8") as handle:
        return model.model_validate(yaml.safe_load(handle))


def default_config(name: str) -> Path:
    """Path of a config shipped in ``configs/data/`` (e.g. ``default_config("collection.yaml")``)."""

    return CONFIG_DIR / name
