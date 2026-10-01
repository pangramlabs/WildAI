"""Is our law's lead on the held-out sizes significant? (appendix table)

Every fit is held fixed. The held-out AI runs are resampled by whole control group (a cluster bootstrap: 10,000 draws of
the groups with replacement) and each law's paired RMSE is recomputed on every draw. One generator, seeded once, draws the
targets and cuts in table order.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel

from wildai.laws.catalog import PAPER_LAW
from wildai.laws.data import HUMAN_TARGETS
from wildai.laws.records import PredictionRecord

RUNNER_UP = "shukor_joint"  # the best existing law on human text
DRAWS = 10_000
SEED = 20260923
CUTS: tuple[tuple[str, float | None], ...] = (("all", None), ("r<1", 1.0))


class SignificanceRecord(BaseModel):
    """Paired RMSEs (log units) on the held-out AI runs of one target, with bootstrap intervals and win rates."""

    target: str
    cut: str  # "all" or "r<1"
    runs: int
    groups: int
    rival: str  # the law the gap is measured against
    point: dict[str, float]  # law -> paired RMSE
    interval: dict[str, tuple[float, float]]  # law -> 95 % percentile interval
    gap: float  # rival minus ours
    gap_interval: tuple[float, float]
    p_ours_lower: dict[str, float]  # law -> share of draws in which ours has the lower error
    p_ours_best: float  # share of draws in which ours beats every other law
    leave_one_group_out_gap: tuple[float, float]  # smallest and largest gap with one control group dropped
    groups_ours_lower: int  # control groups on which ours has the lower error than the rival


def _rmse(sum_squares: np.ndarray, count: np.ndarray) -> np.ndarray:
    return np.sqrt(sum_squares / count)


def bootstrap(target: str, cut: str, rows: Sequence[PredictionRecord], laws: Sequence[str], rng: np.random.Generator,
              rival: str = RUNNER_UP) -> SignificanceRecord:
    """``rows``: held-out AI-run predictions of every law on one target (the same runs for each law)."""
    residual = {law: {r.name: r.predicted_change - r.observed_change for r in rows if r.law == law} for law in laws}
    runs = sorted({(r.control, r.name) for r in rows})
    groups = sorted({control for control, _ in runs})
    members = {g: [name for control, name in runs if control == g] for g in groups}
    size = np.array([len(members[g]) for g in groups], float)
    sum_squares = {law: np.array([sum(residual[law][t] ** 2 for t in members[g]) for g in groups]) for law in laws}
    counts = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=DRAWS).astype(float)
    draws = {law: _rmse(counts @ sum_squares[law], counts @ size) for law in laws}
    point = {law: float(_rmse(sum_squares[law].sum(), size.sum())) for law in laws}
    others = [law for law in laws if law != PAPER_LAW]

    def interval(x: np.ndarray) -> tuple[float, float]:
        lo, hi = np.percentile(x, [2.5, 97.5])
        return float(lo), float(hi)

    loo = []
    for g in groups:
        keep = np.array([h != g for h in groups])
        loo.append(float(_rmse(sum_squares[rival][keep].sum(), size[keep].sum()) - _rmse(sum_squares[PAPER_LAW][keep].sum(), size[keep].sum())))
    wins = sum(sum_squares[PAPER_LAW][i] < sum_squares[rival][i] for i in range(len(groups)))
    return SignificanceRecord(
        target=target, cut=cut, runs=len(runs), groups=len(groups), rival=rival, point=point, interval={law: interval(draws[law]) for law in laws},
        gap=point[rival] - point[PAPER_LAW], gap_interval=interval(draws[rival] - draws[PAPER_LAW]),
        p_ours_lower={law: float(np.mean(draws[PAPER_LAW] < draws[law])) for law in others},
        p_ours_best=float(np.mean(draws[PAPER_LAW] < np.min([draws[law] for law in others], axis=0))),
        leave_one_group_out_gap=(min(loo), max(loo)), groups_ours_lower=int(wins))


def analyse(predictions: Sequence[PredictionRecord], laws: Sequence[str], targets: Sequence[str] = HUMAN_TARGETS, rival: str = RUNNER_UP,
            cuts: Sequence[tuple[str, float | None]] = CUTS) -> list[SignificanceRecord]:
    """Bootstrap every target over all AI ratios and r < 1, from the held-out predictions of the given laws."""
    rng = np.random.default_rng(SEED)
    out = []
    for target in targets:
        for cut, cutoff in cuts:
            rows = [r for r in predictions if r.target == target and r.fold == "all" and r.split == "held_out" and r.arm == "ai" and r.law in laws
                    and (cutoff is None or r.ratio < cutoff - 1e-10)]
            out.append(bootstrap(target, cut, rows, laws, rng, rival))
    return out
