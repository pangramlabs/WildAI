"""Leave-one-size-out fits on C4 (our law and the joint law of Shukor et al.): how much the fit and its projections to 8B
move when one fitted size is left out. The fits ``all`` and ``without_268M`` (fitted through 135M) are the two fits of the
extrapolation figure; each fold's left-out size is scored as held-out data."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from wildai.laws.bank import FOLDS, Case, fold_depth, resolve_law
from wildai.laws.catalog import PAPER_LAW
from wildai.laws.ceg import WEB_2026_AI_SHARE, ai_token_value, cost_of_not_filtering
from wildai.laws.data import Study
from wildai.laws.fit import LawFit
from wildai.laws.forms import FittedLaw
from wildai.laws.records import PredictionRecord
from wildai.laws.score import predict, score

LAWS = (PAPER_LAW, "shukor_joint")
TARGET = "c4"
PROJECTION_PARAMS = 8e9  # the size the paper projects to
PROJECTION_TPP = 20.0


class StabilityRecord(BaseModel):
    law: str
    target: str
    fold: str
    runs: int
    objective: float
    active_bounds: list[str]
    left_out_ai_runs: int  # AI runs of the left-out size (0 for the fold with every size)
    left_out_paired_rmse: float | None
    cost_of_not_filtering_8b: float  # compute an unfiltered 2026 mix needs at 8B and 20 TPP_h, over human-only training
    ai_token_value_8b: float  # value of the AI tokens of the 2026 mix at 8B and 20 TPP_h, in human tokens


def cases() -> list[Case]:
    return [Case(law, TARGET, fold) for law in LAWS for fold in FOLDS]


def projection(model: FittedLaw) -> tuple[float, float]:
    """Cost of not filtering and AI-token value of the 2026 mix at 8B parameters and 20 human tokens per parameter."""
    human = PROJECTION_TPP * PROJECTION_PARAMS
    ratio = WEB_2026_AI_SHARE / (1 - WEB_2026_AI_SHARE)
    ceg = cost_of_not_filtering(model, PROJECTION_PARAMS, PROJECTION_TPP, WEB_2026_AI_SHARE)
    return ceg, float(ai_token_value(model, PROJECTION_PARAMS, human, ratio)[0])


def analyse(study: Study, fits: Mapping[Case, LawFit]) -> tuple[list[StabilityRecord], list[PredictionRecord]]:
    records, rows = [], []
    for case, fit in fits.items():
        law = resolve_law(case.law)
        depth = fold_depth(case.fold)
        paired, n_ai = None, 0
        if depth is not None:
            left_out = study.observations(case.target, [x for x in study.split("fit") if x.depth == depth])
            predictions = predict(law, fit, left_out)
            s = score(fit, predictions)
            paired, n_ai = s.paired_rmse, s.ai_runs
            rows += PredictionRecord.rows(fit, predictions)
        ceg, value = projection(FittedLaw(law, fit.parameters))
        records.append(StabilityRecord(law=case.law, target=case.target, fold=case.fold, runs=fit.runs, objective=fit.objective,
                                       active_bounds=fit.active_bounds, left_out_ai_runs=n_ai, left_out_paired_rmse=paired,
                                       cost_of_not_filtering_8b=ceg, ai_token_value_8b=value))
    return records, rows
