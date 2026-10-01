"""The benchmark tables (paired error on human, mixed and AI text, absolute error, r < 1): every law fitted on the 726 runs
at 19.9M to 268M and scored on the 74 held-out runs at 477M and 973M, per evaluation target."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from wildai.laws.bank import Case, resolve_law
from wildai.laws.catalog import PAPER_LAW, benchmark_laws
from wildai.laws.data import Study
from wildai.laws.fit import LawFit
from wildai.laws.records import PredictionRecord
from wildai.laws.score import CUTOFFS, Score, predict, score_slices

HELD_OUT_DEPTHS: tuple[int | None, ...] = (None, 20, 26)


def cases(targets: Sequence[str]) -> list[Case]:
    return [Case(law, target) for target in targets for law in benchmark_laws()]


def score_fits(study: Study, fits: Mapping[Case, LawFit]) -> tuple[list[Score], list[PredictionRecord]]:
    """Held-out scores (both held-out sizes and each one, every ratio and r < 1) and per-run predictions: the held-out runs
    of every law, and for our law the fitted runs too."""
    scores, rows = [], []
    for case, fit in fits.items():
        law = resolve_law(case.law)
        predictions = predict(law, fit, study.observations(case.target, study.split("held_out")))
        scores += score_slices(fit, predictions, HELD_OUT_DEPTHS, CUTOFFS)
        rows += PredictionRecord.rows(fit, predictions)
        if case.law == PAPER_LAW:
            rows += PredictionRecord.rows(fit, predict(law, fit, study.observations(case.target, study.split("fit"))))
    return scores, rows
