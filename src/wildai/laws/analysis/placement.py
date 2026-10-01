"""Where the harm goes (appendix, "Choosing the penalty"): our law against the same law with the harm added to the loss
instead of scaling the data term, on every target. The two are fitted with the same protocol; the gap in held-out paired
error is bootstrapped by control group as in :mod:`wildai.laws.analysis.significance`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel

from wildai.laws.analysis.significance import SEED, bootstrap
from wildai.laws.analysis.stability import projection
from wildai.laws.bank import Case, resolve_law
from wildai.laws.catalog import PAPER_LAW
from wildai.laws.data import Study
from wildai.laws.fit import LawFit
from wildai.laws.forms import FittedLaw
from wildai.laws.records import PredictionRecord
from wildai.laws.score import predict, score

ADDITIVE = "harm_logshare_additive"


class PlacementRecord(BaseModel):
    target: str
    ours: float  # held-out paired RMSE, every AI ratio
    additive: float
    ours_973m: float
    additive_973m: float
    gap: float  # additive minus ours
    gap_interval: tuple[float, float]
    p_ours_lower: float
    cost_of_not_filtering_8b: dict[str, float]  # law -> cost at 8B and 20 TPP_h of the 2026 mix


def cases(targets: Sequence[str]) -> list[Case]:
    return [Case(ADDITIVE, target) for target in targets]


def analyse(study: Study, fits: Mapping[Case, LawFit], targets: Sequence[str]) -> list[PlacementRecord]:
    rng = np.random.default_rng(SEED)
    out = []
    held_out = study.split("held_out")
    for target in targets:
        rows, errors, errors_973m, costs = [], {}, {}, {}
        for key in (PAPER_LAW, ADDITIVE):
            law, fit = resolve_law(key), fits[Case(key, target)]
            predictions = predict(law, fit, study.observations(target, held_out))
            rows += [r for r in PredictionRecord.rows(fit, predictions) if r.arm == "ai"]
            errors[key], errors_973m[key] = score(fit, predictions).paired_rmse, score(fit, predictions, depth=26).paired_rmse
            costs[key] = projection(FittedLaw(law, fit.parameters))[0]
        b = bootstrap(target, "all", rows, [PAPER_LAW, ADDITIVE], rng, rival=ADDITIVE)
        out.append(PlacementRecord(target=target, ours=errors[PAPER_LAW], additive=errors[ADDITIVE], ours_973m=errors_973m[PAPER_LAW],
                                   additive_973m=errors_973m[ADDITIVE], gap=b.gap, gap_interval=b.gap_interval, p_ours_lower=b.p_ours_lower[ADDITIVE],
                                   cost_of_not_filtering_8b=costs))
    return out
