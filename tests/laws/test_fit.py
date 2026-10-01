"""Fitting: our law's C4 coefficients (Table law-coefficients), Chinchilla's C4 errors, and determinism across workers."""

from __future__ import annotations

import pytest

from wildai.laws.bank import Case, fit_cases
from wildai.laws.catalog import law
from wildai.laws.data import Study
from wildai.laws.fit import LawFit, fit, objective
from wildai.laws.ours import paper_law
from wildai.laws.score import predict, score


def printed(value: float) -> str:
    """Four significant figures, as the coefficient table prints them."""
    return f"{value:#.4g}"


def test_paper_law_c4_coefficients(c4_fit: LawFit, golden: dict) -> None:
    coefficients = paper_law().coefficients(c4_fit.parameters)
    assert {k: printed(v) for k, v in coefficients.items()} == golden["ours_coefficients"]["c4"]
    assert c4_fit.converged and not c4_fit.active_bounds


def test_paper_law_c4_held_out(study: Study, c4_fit: LawFit, golden: dict) -> None:
    predictions = predict(paper_law(), c4_fit, study.observations("c4", study.split("held_out")))
    expected = golden["scores"]["c4"]["ours_logshare_11"]
    assert f"{1e3 * score(c4_fit, predictions).paired_rmse:.2f}" == expected["paired"] == "0.83"
    assert f"{1e3 * score(c4_fit, predictions, cutoff=1.0).paired_rmse:.2f}" == expected["paired_r_below_1"]
    assert f"{1e3 * score(c4_fit, predictions).absolute_ai_rmse:.2f}" == expected["absolute"]


def test_chinchilla_fresh_human_and_ai_errors(study: Study) -> None:
    """Section 3: Chinchilla predicts the fresh-human additions to 1.32e-3 on C4 but the AI additions only to 4.32e-3."""
    chinchilla = law("chinchilla_5")
    f = fit(chinchilla, study.observations("c4", study.split("fit")))
    s = score(f, predict(chinchilla, f, study.observations("c4", study.split("held_out"))))
    assert (f"{1e3 * s.paired_human_rmse:.2f}", f"{1e3 * s.paired_rmse:.2f}") == ("1.32", "4.32")


def test_fit_does_not_depend_on_workers(study: Study, c4_fit: LawFit) -> None:
    case = Case("ours_logshare_11", "c4")
    pooled = fit_cases([case], workers=2)[case]
    assert pooled.parameters == c4_fit.parameters and pooled.objective == c4_fit.objective
    assert objective(paper_law(), study.observations("c4", study.split("fit")), pooled.parameters) == pytest.approx(c4_fit.objective, rel=1e-12)
