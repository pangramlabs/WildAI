"""Output records of the law analyses and their CSV/JSON writers (schemas are listed in the package docstring)."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from wildai.laws.fit import LawFit
from wildai.laws.forms import Law
from wildai.laws.score import Predictions


class FitRecord(BaseModel):
    """One fit, with its coefficients in the paper's form (E, A, alpha, B, beta, eta, K, ...)."""

    law: str
    label: str
    target: str
    fold: str
    k: int
    runs: int
    starts: int
    objective: float
    converged: bool
    active_bounds: list[str]
    parameters: dict[str, float]  # fitted values; log-coefficients where the law fits logarithms
    coefficients: dict[str, float]  # the paper's coefficients

    @classmethod
    def of(cls, law: Law, fit: LawFit) -> FitRecord:
        return cls(law=fit.law, label=law.label, target=fit.target, fold=fit.fold, k=law.k, runs=fit.runs, starts=fit.starts, objective=fit.objective,
                   converged=fit.converged, active_bounds=fit.active_bounds, parameters=fit.parameters, coefficients=law.coefficients(fit.parameters))

    def fit(self) -> LawFit:
        return LawFit(law=self.law, target=self.target, fold=self.fold, parameters=self.parameters, objective=self.objective, converged=self.converged,
                      active_bounds=self.active_bounds, runs=self.runs, starts=self.starts)


class PredictionRecord(BaseModel):
    """A law's prediction for one run; changes are log(L / L_control)."""

    law: str
    target: str
    fold: str
    name: str
    split: str  # the run's split: "fit" (in-sample) or "held_out"
    control: str
    arm: str
    depth: int
    ratio: float
    human_tpp: float
    observed_bpb: float
    predicted_bpb: float
    observed_change: float
    predicted_change: float

    @classmethod
    def rows(cls, fit: LawFit, predictions: Predictions) -> list[PredictionRecord]:
        obs = predictions.obs
        observed, predicted = predictions.observed_change, predictions.predicted_change
        return [cls(law=fit.law, target=fit.target, fold=fit.fold, name=run.name, split=run.split, control=obs.runs[obs.control[i]].name, arm=run.arm,
                    depth=run.depth,
                    ratio=run.ratio, human_tpp=run.human_tpp, observed_bpb=float(obs.bpb[i]), predicted_bpb=float(predictions.predicted_bpb[i]),
                    observed_change=float(observed[i]), predicted_change=float(predicted[i]))
                for i, run in enumerate(obs.runs)]


def _cell(value: object) -> object:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "" if value is None else str(value)
    if isinstance(value, (list, tuple)):
        return ";".join(map(str, value))
    return value


def write_csv(path: Path, records: Sequence[BaseModel]) -> None:
    """Flat records as CSV; empty cells for None, lists joined with ';'. No records, no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.unlink(missing_ok=True)
        return
    rows = [r.model_dump() for r in records]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: _cell(v) for k, v in row.items()} for row in rows)


def _finite(value: object) -> object:
    """JSON has no infinity: non-finite numbers become null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


def write_json(path: Path, payload: BaseModel | Sequence[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = payload.model_dump(mode="python") if isinstance(payload, BaseModel) else [p.model_dump(mode="python") for p in payload]
    path.write_text(json.dumps(_finite(data), indent=1, allow_nan=False) + "\n")


def read_fits(path: Path) -> list[FitRecord]:
    return [FitRecord.model_validate(x) for x in json.loads(path.read_text())]
