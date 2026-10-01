"""The law-form ablation (appendix table): our law on C4 with its credit or its harm changed one at a time.

Each variant is fitted on every fitted size and scored on the held-out 477M and 973M runs, and fitted five more times
with one fitted size left out and scored on that size (the leave-one-size-out error pools those five sets of runs).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from pydantic import BaseModel

from wildai.laws.bank import FOLDS, Case, fold_depth, resolve_law
from wildai.laws.data import Study
from wildai.laws.fit import LawFit
from wildai.laws.ours import ABLATION_FORMS
from wildai.laws.score import Predictions, predict, rmse

TARGET = "c4"


class AblationRecord(BaseModel):
    name: str
    group: str  # "benefit", "harm" or "both" (our law)
    description: str
    k: int
    objective: float
    active_bounds: list[str]
    held_out_ai_runs: int
    held_out_paired_rmse: float  # 477M and 973M together: the table's column
    held_out_paired_rmse_477m: float
    held_out_paired_rmse_973m: float
    held_out_paired_rmse_low: float  # r < 1
    leave_one_size_out_ai_runs: int
    leave_one_size_out_paired_rmse: float
    leave_one_size_out_paired_rmse_low: float
    leave_one_size_out_paired_rmse_high: float  # r >= 1


def cases() -> list[Case]:
    return [Case(name, TARGET, fold) for name in ABLATION_FORMS for fold in FOLDS]


def _ai_residuals(predictions: Predictions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = predictions.obs
    ai = obs.arms == "ai"
    return predictions.paired_residual[ai], obs.depths[ai], obs.ratios[ai]


def analyse(study: Study, fits: Mapping[Case, LawFit]) -> list[AblationRecord]:
    records = []
    for name, (group, description, _form) in ABLATION_FORMS.items():
        law = resolve_law(name)
        full = fits[Case(name, TARGET, "all")]
        held, depth, ratio = _ai_residuals(predict(law, full, study.observations(TARGET, study.split("held_out"))))
        pooled: list[tuple[np.ndarray, np.ndarray]] = []
        for fold in FOLDS[1:]:
            left = fold_depth(fold)
            obs = study.observations(TARGET, [x for x in study.split("fit") if x.depth == left])
            e, _d, r = _ai_residuals(predict(law, fits[Case(name, TARGET, fold)], obs))
            pooled.append((e, r))
        cv, cv_r = np.concatenate([e for e, _ in pooled]), np.concatenate([r for _, r in pooled])
        records.append(AblationRecord(
            name=name, group=group, description=description, k=law.k, objective=full.objective, active_bounds=full.active_bounds,
            held_out_ai_runs=len(held), held_out_paired_rmse=rmse(held), held_out_paired_rmse_477m=rmse(held[depth == 20]),
            held_out_paired_rmse_973m=rmse(held[depth == 26]), held_out_paired_rmse_low=rmse(held[ratio < 1]),
            leave_one_size_out_ai_runs=len(cv), leave_one_size_out_paired_rmse=rmse(cv), leave_one_size_out_paired_rmse_low=rmse(cv[cv_r < 1]),
            leave_one_size_out_paired_rmse_high=rmse(cv[cv_r >= 1])))
    return records
