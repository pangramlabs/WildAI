"""Look up the paper's models by public name in `results/models.csv` and turn them into `RunSpec`s.

Groups `gNNN` hold one human-only control (`<size>-gNNN-control`) and the runs that add AI text, fresh human text or
repeated human text on top of exactly that control's human documents; their `human_steps` is the control's step
count. Filtering pairs `fNN` hold a web-mix run (`<size>-fNN-web-mix`, arm `natural`) and its AI-removed partner
(`<size>-fNN-ai-removed`, arm `filtered`). The partner's step count is the web mix's `human_steps`, so the web mix's
AI share is its extra steps over its total: 0.222 to 0.224 for the published pairs, whose rate-matched design targeted
0.2232 (models.csv lists the published runs' realized token counts).
"""

from __future__ import annotations

import csv
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from wildai.training.mixture import Arm, RunSpec

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODELS_CSV = REPO_ROOT / "results" / "models.csv"


class ModelRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    group: str
    arm: Arm
    size: str
    depth: int
    n_params: int
    seed: int
    added_ratio: float | None
    human_tokens: int
    ai_tokens: int
    total_tokens: int
    steps: int
    split: str


def load_models(path: Path = DEFAULT_MODELS_CSV) -> dict[str, ModelRow]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [ModelRow.model_validate({k: (v if v != "" else None) for k, v in row.items()}) for row in csv.DictReader(handle)]
    return {row.name: row for row in rows}


def run_spec(name: str, models: dict[str, ModelRow]) -> RunSpec:
    if name not in models:
        raise KeyError(f"unknown model {name!r}; see results/models.csv")
    row = models[name]
    if row.arm in (Arm.NATURAL, Arm.FILTERED):
        human_steps = models[f"{row.size}-{row.group}-ai-removed"].steps
    else:
        human_steps = models[f"{row.size}-{row.group}-control"].steps
    return RunSpec(name=name, depth=row.depth, arm=row.arm, human_steps=human_steps, steps=row.steps, seed=row.seed)
