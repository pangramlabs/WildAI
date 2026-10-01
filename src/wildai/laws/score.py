"""Scoring a fitted law on held-out runs.

Paired error (the paper's main metric): the RMSE over the AI runs of the predicted minus the observed change in log loss
against each run's own human-only control, log(L_hat_i / L_hat_c) - log(L_i / L_c). Absolute error: the RMSE of the
predicted minus the observed loss (BPB) over the same AI runs, or over every run (``absolute_all_rmse``).
Cutoffs keep the runs whose actual AI ratio r is below the cutoff (controls and fresh-human runs have r = 0).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from wildai.laws.data import Observations
from wildai.laws.fit import LawFit
from wildai.laws.forms import Law

CUTOFFS: tuple[float | None, ...] = (None, 1.0)  # None: every ratio; 1.0: r < 1, the range a web crawl can reach


@dataclass(frozen=True)
class Predictions:
    """A law's predictions for a set of paired runs (arrays aligned with ``obs.runs``)."""

    obs: Observations
    predicted_bpb: np.ndarray

    @property
    def observed_change(self) -> np.ndarray:
        return np.log(self.obs.bpb / self.obs.bpb[self.obs.control])

    @property
    def predicted_change(self) -> np.ndarray:
        return np.log(self.predicted_bpb / self.predicted_bpb[self.obs.control])

    @property
    def paired_residual(self) -> np.ndarray:
        return self.predicted_change - self.observed_change

    @property
    def absolute_residual(self) -> np.ndarray:
        return self.predicted_bpb - self.obs.bpb


def predict(law: Law, law_fit: LawFit, obs: Observations) -> Predictions:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        predicted = np.asarray(law.predict(law_fit.vector(law), obs.coords))
    if not (np.all(np.isfinite(predicted)) and np.all(predicted > 0)):
        raise FloatingPointError(f"{law.key} predicts a non-finite or non-positive loss on {obs.target}")
    return Predictions(obs, predicted)


class Score(BaseModel):
    """Held-out errors of one fit, for one slice of the held-out runs."""

    law: str
    target: str
    fold: str
    depth: int | None  # None: every held-out size
    cutoff: float | None  # None: every AI ratio; otherwise runs with r < cutoff
    runs: int  # held-out runs in the slice, controls and fresh-human runs included
    ai_runs: int
    human_runs: int  # fresh-human additions
    paired_rmse: float | None  # over the AI runs: the paper's paired error
    paired_human_rmse: float | None  # over the fresh-human additions
    absolute_ai_rmse: float | None
    absolute_all_rmse: float | None


def rmse(values: np.ndarray) -> float | None:
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else None


def score(law_fit: LawFit, predictions: Predictions, depth: int | None = None, cutoff: float | None = None) -> Score:
    obs = predictions.obs
    keep = np.ones(len(obs), bool)
    if depth is not None:
        keep &= obs.depths == depth
    if cutoff is not None:
        keep &= obs.ratios < cutoff - 1e-10
    ai, human = keep & (obs.arms == "ai"), keep & (obs.arms == "human")
    return Score(law=law_fit.law, target=law_fit.target, fold=law_fit.fold, depth=depth, cutoff=cutoff, runs=int(keep.sum()), ai_runs=int(ai.sum()),
                 human_runs=int(human.sum()), paired_rmse=rmse(predictions.paired_residual[ai]), paired_human_rmse=rmse(predictions.paired_residual[human]),
                 absolute_ai_rmse=rmse(predictions.absolute_residual[ai]), absolute_all_rmse=rmse(predictions.absolute_residual[keep]))


def score_slices(law_fit: LawFit, predictions: Predictions, depths: Iterable[int | None] = (None,), cutoffs: Iterable[float | None] = CUTOFFS) -> list[Score]:
    """Scores for every combination of held-out size and ratio cutoff that has runs."""
    out = []
    for depth in depths:
        for cutoff in cutoffs:
            s = score(law_fit, predictions, depth, cutoff)
            if s.runs:
                out.append(s)
    return out
