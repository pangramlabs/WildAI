"""Mixed validation data hides harm to human text (appendix figure, and Figure 6 right). Measured losses only; no law.

For every AI addition, dh and da are its change in loss (BPB) against its control on the human-labelled (FW26-H) and the
AI-labelled (FW26-AI) partition of the 2026 crawl. A validation set whose scored bytes are a fraction q AI-labelled reports
the change ((1 - q) dh + q da) / ((1 - q) h_c + q a_c), with h_c and a_c the control's losses. A run that raises the
human loss (dh > 0) but lowers the AI loss (da < 0) looks like an improvement once q exceeds q* = dh / (dh - da).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from wildai.laws.ceg import WEB_2026_AI_SHARE
from wildai.laws.data import Study

HUMAN, AI = "fw26_human", "fw26_ai"
SHARES = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2, WEB_2026_AI_SHARE, 0.4)


class MaskingContrast(BaseModel):
    name: str
    control: str
    split: str
    ratio: float
    human_tpp: float
    delta_human_bpb: float
    delta_ai_bpb: float
    critical_share: float | None  # q*: defined when the human loss rises and the AI loss falls


class MaskingAtShare(BaseModel):
    ai_share: float  # AI-labelled share of the scored validation bytes
    masked: int  # human-harming runs the mixed set reports as improvements
    masked_share: float
    median_reported_change: float  # median relative change the mixed set reports for the human-harming runs


class MaskingSummary(BaseModel):
    ai_runs: int
    human_harmed: int  # AI additions that raise loss on human-labelled text
    ai_improved: int  # AI additions that lower loss on AI-labelled text
    control_ai_loss_reduction_range: tuple[float, float]  # human-only models: relative loss reduction on AI- vs human-labelled text
    median_flip_share: float  # AI share at which the median reported change of the harmed runs crosses zero
    at_share: list[MaskingAtShare]
    by_split: dict[str, dict[str, int]]  # split -> harmed runs and those masked at the 2026 share


def contrasts(study: Study) -> list[MaskingContrast]:
    human, ai = study.cohort(HUMAN), study.cohort(AI)
    out = []
    for i, run in enumerate(human.runs):
        if run.arm != "ai":
            continue
        c = human.control[i]
        dh, da = float(human.bpb[i] - human.bpb[c]), float(ai.bpb[i] - ai.bpb[c])
        out.append(MaskingContrast(name=run.name, control=human.runs[c].name, split=run.split, ratio=run.ratio, human_tpp=run.human_tpp,
                                   delta_human_bpb=dh, delta_ai_bpb=da, critical_share=dh / (dh - da) if dh > 0 > da else None))
    return out


def summarise(study: Study, rows: list[MaskingContrast]) -> MaskingSummary:
    human, ai = study.cohort(HUMAN), study.cohort(AI)
    loss_h = dict(zip(human.names, human.bpb))
    loss_a = dict(zip(ai.names, ai.bpb))
    harmed = [r for r in rows if r.delta_human_bpb > 0]
    dh = np.array([r.delta_human_bpb for r in harmed])
    da = np.array([r.delta_ai_bpb for r in harmed])
    hc = np.array([loss_h[r.control] for r in harmed])
    ac = np.array([loss_a[r.control] for r in harmed])

    def reported(q: float) -> np.ndarray:
        return ((1 - q) * dh + q * da) / ((1 - q) * hc + q * ac)

    lo, hi = 0.0, 1.0  # the median reported change falls monotonically with q (every harmed run improves on AI text)
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if np.median(reported(mid)) > 0 else (lo, mid)
    at_share = [MaskingAtShare(ai_share=q, masked=int(np.sum(reported(q) < 0)), masked_share=float(np.mean(reported(q) < 0)),
                               median_reported_change=float(np.median(reported(q)))) for q in SHARES]
    controls = [x.name for x in human.runs if x.arm == "control"]
    reduction = [1 - loss_a[c] / loss_h[c] for c in controls]
    by_split = {}
    for split in ("fit", "held_out"):
        s = [r for r in harmed if r.split == split]
        by_split[split] = {"ai_runs": sum(r.split == split for r in rows), "harmed": len(s),
                           "masked_at_2026_share": sum(r.critical_share is not None and r.critical_share < WEB_2026_AI_SHARE for r in s)}
    return MaskingSummary(ai_runs=len(rows), human_harmed=len(harmed), ai_improved=sum(r.delta_ai_bpb < 0 for r in rows),
                          control_ai_loss_reduction_range=(min(reduction), max(reduction)), median_flip_share=(lo + hi) / 2,
                          at_share=at_share, by_split=by_split)
