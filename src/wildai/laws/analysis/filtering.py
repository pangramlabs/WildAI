"""Filtering the 2026 web mix (Section 4): the measured change in loss from removing the AI documents of a mix with
22.3 % AI tokens (adding nothing in their place), against what a law fitted on the scaling-law runs predicts.

A pair is a run on the mix and the same run with its AI-labelled documents removed. The law's prediction keeps the mix's
human tokens and drops its AI tokens: log L(N, D_H, 0) - log L(N, D_H, D_AI). These runs are never fitted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel

from wildai.laws.bank import Case, resolve_law
from wildai.laws.data import Run, Study
from wildai.laws.fit import LawFit
from wildai.laws.forms import FittedLaw


class FilteringRecord(BaseModel):
    law: str
    target: str
    group: str
    depth: int
    n_params: int
    human_tpp: float  # human tokens of the mix per parameter
    ratio: float  # AI over human tokens in the mix
    observed_change: float  # log(L_filtered / L_mix)
    predicted_change: float


class FilteringSummary(BaseModel):
    law: str
    target: str
    pairs: int
    paired_rmse: float
    direction_right: int  # pairs where the predicted and the measured change have the same sign
    filtering_helps: int  # pairs where removing the AI documents lowers the loss
    predicted_harm_where_it_helps: int  # of those, pairs where the law predicts a higher loss


def pairs(study: Study) -> list[tuple[Run, Run]]:
    """(mix, filtered) runs of every filtering pair."""
    runs = study.split("filtering")
    by_group: dict[str, dict[str, Run]] = {}
    for x in runs:
        by_group.setdefault(x.group, {})[x.arm] = x
    return [(g["natural"], g["filtered"]) for _group, g in sorted(by_group.items())]


def analyse(study: Study, fits: Mapping[Case, LawFit], laws: Sequence[str], targets: Sequence[str]) -> tuple[list[FilteringRecord], list[FilteringSummary]]:
    records, summaries = [], []
    for target in targets:
        for key in laws:
            model = FittedLaw(resolve_law(key), fits[Case(key, target)].parameters)
            rows = []
            for mix, filtered in pairs(study):
                observed = float(np.log(study.bpb(filtered.name, target) / study.bpb(mix.name, target)))
                predicted = float(np.log(model.loss1(mix.n_params, mix.human_tokens, 0.0) / model.loss1(mix.n_params, mix.human_tokens, mix.ai_tokens)))
                rows.append(FilteringRecord(law=key, target=target, group=mix.group, depth=mix.depth, n_params=mix.n_params, human_tpp=mix.human_tpp,
                                            ratio=mix.ratio, observed_change=observed, predicted_change=predicted))
            o = np.array([r.observed_change for r in rows])
            p = np.array([r.predicted_change for r in rows])
            summaries.append(FilteringSummary(law=key, target=target, pairs=len(rows), paired_rmse=float(np.sqrt(np.mean((p - o) ** 2))),
                                              direction_right=int(np.sum(np.sign(p) == np.sign(o))), filtering_helps=int(np.sum(o < 0)),
                                              predicted_harm_where_it_helps=int(np.sum((o < 0) & (p > 0)))))
            records += rows
    return records, summaries
