"""The law-derived numbers behind the paper's recommendations (Sections 4 and 5, Figures 1 and 6): the cost of not
filtering a web mix, the value of an AI token and the loss-minimising AI share, all from our law's fit of each target."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel

from wildai.laws.analysis.stability import PROJECTION_PARAMS, PROJECTION_TPP
from wildai.laws.bank import Case, resolve_law
from wildai.laws.catalog import PAPER_LAW
from wildai.laws.ceg import WEB_2026_AI_SHARE, ai_token_value, cost_of_not_filtering, largest_loss_reduction, optimal_ratio
from wildai.laws.data import Study
from wildai.laws.fit import LawFit
from wildai.laws.forms import FittedLaw

# AI shares of web tokens the paper quotes: our 2026 crawl, August 2026, and the forecasts for the ends of 2027 and 2028.
PAPER_SHARES: dict[str, float] = {"2026 crawl": WEB_2026_AI_SHARE, "August 2026": 0.311, "end of 2027": 0.423, "end of 2028": 0.507}
REFERENCE_TPP = 20.0
BUDGETS = (5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0)


class CostOfNotFiltering(BaseModel):
    target: str
    n_params: float
    human_tpp: float
    ai_share: float
    label: str  # which share, or "" for a point of the share grid
    cost: float  # compute of the unfiltered mix over human-only training at the same loss (inf: never reaches it)


class OptimalShare(BaseModel):
    target: str
    n_params: float
    human_tpp: float
    optimal_ratio: float
    optimal_share: float
    largest_loss_reduction: float


class TokenValue(BaseModel):
    target: str
    n_params: float
    human_tpp: float
    ratio: float
    value: float  # average value of the added AI tokens, in human tokens


def reference_size(study: Study) -> float:
    """N of the 268M models, the size the paper's recommendations use."""
    return float(next(x.n_params for x in study.split("fit") if x.depth == 16))


def analyse(study: Study, fits: Mapping[Case, LawFit], targets: Sequence[str]) -> tuple[list[CostOfNotFiltering], list[OptimalShare], list[TokenValue]]:
    n = reference_size(study)
    costs, shares, values = [], [], []
    for target in targets:
        model = FittedLaw(resolve_law(PAPER_LAW), fits[Case(PAPER_LAW, target)].parameters)
        for label, share in [*PAPER_SHARES.items(), *(("", float(s)) for s in np.linspace(0.0, 0.6, 61))]:
            costs.append(CostOfNotFiltering(target=target, n_params=n, human_tpp=REFERENCE_TPP, ai_share=share, label=label,
                                            cost=cost_of_not_filtering(model, n, REFERENCE_TPP, share)))
        costs.append(CostOfNotFiltering(target=target, n_params=PROJECTION_PARAMS, human_tpp=PROJECTION_TPP, ai_share=WEB_2026_AI_SHARE, label="2026 crawl",
                                        cost=cost_of_not_filtering(model, PROJECTION_PARAMS, PROJECTION_TPP, WEB_2026_AI_SHARE)))
        for tpp in sorted({*BUDGETS, *np.geomspace(5.0, 100.0, 80)}):
            r = optimal_ratio(model, n, tpp * n)
            shares.append(OptimalShare(target=target, n_params=n, human_tpp=float(tpp), optimal_ratio=r, optimal_share=r / (1 + r),
                                       largest_loss_reduction=largest_loss_reduction(model, n, tpp * n)))
        ratios = np.geomspace(1e-2, 1e2, 161)
        for tpp in BUDGETS:
            for r, v in zip(ratios, ai_token_value(model, n, tpp * n, ratios)):
                values.append(TokenValue(target=target, n_params=n, human_tpp=tpp, ratio=float(r), value=float(v)))
    return costs, shares, values
