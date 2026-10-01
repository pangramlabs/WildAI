"""Compute-equivalent gain over the 2026 web mix at a fixed model size (appendix figure on CEG).

For a run that reaches loss L with D training tokens, CEG = D_ref(L) / D, where D_ref(L) is the number of tokens a model of
the same size trained on the 2026 mix (22.3 % AI) needs to reach L under our law. Both the measured loss of every run
(``observed_ceg``) and our law's prediction of it (``predicted_ceg``) are converted. The reference is searched only over the
fitted human budgets, so a loss outside that range has no CEG and keeps a status instead.

The figure's ladder is the 268M control at 20 TPP_h whose AI additions reach r = 1, as the AI share of its tokens varies.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel

from wildai.laws.bank import Case, resolve_law
from wildai.laws.catalog import PAPER_LAW
from wildai.laws.ceg import WEB_2026_AI_SHARE, InversionStatus, ReferenceMix
from wildai.laws.data import Study
from wildai.laws.fit import LawFit
from wildai.laws.forms import FittedLaw

LADDER_CONTROL = "268m-g081-control"
LADDER_TARGETS = ("c4", "cosmopedia")
LADDER_MAX_RATIO = 1.05


class CegRecord(BaseModel):
    target: str
    name: str
    control: str
    split: str
    arm: str
    depth: int
    n_params: int
    human_tokens: float
    ai_tokens: float
    ratio: float
    ai_share: float
    observed_bpb: float
    predicted_bpb: float
    observed_ceg: float | None
    predicted_ceg: float | None
    observed_status: InversionStatus
    predicted_status: InversionStatus


class CegCurvePoint(BaseModel):
    target: str
    control: str
    ai_share: float
    ceg: float | None
    status: InversionStatus


class CegSummary(BaseModel):
    target: str
    human_tpp_range: tuple[float, float]
    observed_status_counts: dict[str, int]
    predicted_status_counts: dict[str, int]
    ladder_control_predicted_ceg: float | None  # the human-only control of the ladder, under our law
    ladder_control_observed_ceg: float | None


def analyse(study: Study, fits: Mapping[Case, LawFit], targets: Sequence[str]) -> tuple[list[CegRecord], list[CegCurvePoint], list[CegSummary]]:
    records, curve, summaries = [], [], []
    cohort = study.split("fit", "held_out")
    fitted = study.split("fit")
    for target in targets:
        model = FittedLaw(resolve_law(PAPER_LAW), fits[Case(PAPER_LAW, target)].parameters)
        tpp_range = (min(x.human_tpp for x in fitted), max(x.human_tpp for x in fitted))
        references: dict[int, ReferenceMix] = {}
        obs = study.observations(target, cohort)
        predicted = model.loss(np.array([x.n_params for x in cohort], float), np.array([x.human_tokens for x in cohort]), np.array([x.ai_tokens for x in cohort]))
        rows = []
        for i, run in enumerate(obs.runs):
            ref = references.setdefault(run.n_params, ReferenceMix(model, run.n_params, tpp_range))
            total = run.human_tokens + run.ai_tokens
            observed_ceg, observed_status = ref.ceg(float(obs.bpb[i]), total)
            predicted_ceg, predicted_status = ref.ceg(float(predicted[i]), total)
            rows.append(CegRecord(target=target, name=run.name, control=obs.runs[obs.control[i]].name, split=run.split, arm=run.arm, depth=run.depth,
                                  n_params=run.n_params, human_tokens=run.human_tokens, ai_tokens=run.ai_tokens, ratio=run.ratio, ai_share=run.ai_tokens / total,
                                  observed_bpb=float(obs.bpb[i]), predicted_bpb=float(predicted[i]), observed_ceg=observed_ceg, predicted_ceg=predicted_ceg,
                                  observed_status=observed_status, predicted_status=predicted_status))
        control = next(r for r in rows if r.name == LADDER_CONTROL)
        if target in LADDER_TARGETS:
            ladder = [r for r in rows if r.control == LADDER_CONTROL and r.arm in ("control", "ai") and r.ratio <= LADDER_MAX_RATIO]
            ref = references[control.n_params]
            for share in np.unique(np.r_[np.linspace(0, max(r.ai_share for r in ladder), 241), WEB_2026_AI_SHARE]):
                total = control.human_tokens / (1 - share)
                ceg, status = ref.ceg(model.loss1(control.n_params, control.human_tokens, total * share), total)
                curve.append(CegCurvePoint(target=target, control=LADDER_CONTROL, ai_share=float(share), ceg=ceg, status=status))
        summaries.append(CegSummary(target=target, human_tpp_range=tpp_range, observed_status_counts=dict(Counter(r.observed_status for r in rows)),
                                    predicted_status_counts=dict(Counter(r.predicted_status for r in rows)),
                                    ladder_control_predicted_ceg=control.predicted_ceg, ladder_control_observed_ceg=control.observed_ceg))
        records += rows
    return records, curve, summaries
