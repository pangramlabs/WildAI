"""The paper's law tables recomputed from scratch (slow: every law on every target, the leave-one-size-out fits and the
law-form ablation; about 40 CPU-minutes, spread over every core): every printed number of the benchmark tables, our law's
coefficients, the ablation and the bootstrap intervals.
"""

from __future__ import annotations

import os

import pytest

from wildai.laws.analysis import ablation, benchmark, placement, significance, stability
from wildai.laws.bank import Case, fit_cases, resolve_law
from wildai.laws.catalog import PAPER_LAW, benchmark_laws
from wildai.laws.data import HUMAN_TARGETS, PAPER_TARGETS, Study
from wildai.laws.fit import LawFit
from wildai.laws.score import Score

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def fits() -> dict[Case, LawFit]:
    cases = benchmark.cases(list(PAPER_TARGETS)) + stability.cases() + ablation.cases() + placement.cases(list(PAPER_TARGETS))
    return fit_cases(list(dict.fromkeys(cases)), workers=os.cpu_count() or 1)


@pytest.fixture(scope="module")
def benchmark_results(study: Study, fits: dict[Case, LawFit]) -> tuple[list[Score], list]:
    return benchmark.score_fits(study, {c: fits[c] for c in benchmark.cases(list(PAPER_TARGETS))})


def printed_scores(scores: list[Score]) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for s in scores:
        if s.depth is not None:
            continue
        row = out.setdefault((s.target, s.law), {})
        if s.cutoff is None:
            row.update(paired=f"{1e3 * s.paired_rmse:.2f}", absolute=f"{1e3 * s.absolute_ai_rmse:.2f}")
        else:
            row["paired_r_below_1"] = f"{1e3 * s.paired_rmse:.2f}"
    return out


def test_benchmark_tables(study: Study, fits: dict[Case, LawFit], benchmark_results: tuple, golden: dict) -> None:
    ours = printed_scores(benchmark_results[0])
    for target in PAPER_TARGETS:
        for law in benchmark_laws():
            assert ours[(target, law)] == golden["scores"][target][law], (target, law)


def test_paper_law_coefficients(study: Study, fits: dict[Case, LawFit], golden: dict) -> None:
    for target in PAPER_TARGETS:
        fit = fits[Case(PAPER_LAW, target)]
        printed = {k: f"{v:#.4g}" for k, v in resolve_law(PAPER_LAW).coefficients(fit.parameters).items()}
        assert printed == golden["ours_coefficients"][target], target


def test_paloma_comparators_predict_no_change(benchmark_results: tuple) -> None:
    """Appendix: five comparators fit a negligible data term on Paloma and predict almost no change from added AI text
    (|change| at most 2.1e-5)."""
    predictions = benchmark_results[1]
    for law in ("chinchilla_5", "muennighoff_7", "cd_8", "atlas_no_repeat", "hamidieh_first_order"):
        changes = [abs(r.predicted_change) for r in predictions if r.target == "paloma" and r.law == law and r.arm == "ai"]
        assert max(changes) < 2.1e-5, law


def test_ablation_table(study: Study, fits: dict[Case, LawFit], golden: dict) -> None:
    rows = ablation.analyse(study, {c: fits[c] for c in ablation.cases()})
    assert {r.name: f"{1e3 * r.held_out_paired_rmse:.2f}" for r in rows} == golden["ablation"]


def test_significance_of_our_lead(benchmark_results: tuple, golden: dict) -> None:
    """Our law's point errors are exact; its bootstrap intervals agree within Monte Carlo error (the draws depend on the
    order of the control groups; Paloma's heavy-tailed residuals move its upper ends by up to 1 %)."""
    records = significance.analyse(benchmark_results[1], list(benchmark_laws()), HUMAN_TARGETS)
    for r in records:
        expected = golden["significance"][r.target][r.cut]
        assert f"{1e3 * r.point[PAPER_LAW]:.2f}" == f"{expected['ours']:.2f}"
        for mine, reference in zip(r.interval[PAPER_LAW], expected["ours_interval"]):
            assert abs(1e3 * mine - reference) < 0.02 + 0.015 * reference, (r.target, r.cut)
        if r.cut == "all" and r.target != "paloma":
            assert r.p_ours_lower["shukor_joint"] > 0.999


def test_stability_and_harm_placement(study: Study, fits: dict[Case, LawFit]) -> None:
    """Appendix: the harm added to the loss scores 0.80 on C4 against our 0.83, the gap's 95 % interval spans zero, and the two
    give the same cost of not filtering at 8B (1.58 and 1.59x)."""
    records, _ = stability.analyse(study, {c: fits[c] for c in stability.cases()})
    ours_all = next(r for r in records if r.law == PAPER_LAW and r.fold == "all")
    assert f"{ours_all.cost_of_not_filtering_8b:.2f}" == "1.59"
    c4 = next(r for r in placement.analyse(study, fits, ["c4"]))
    assert (f"{1e3 * c4.additive:.2f}", f"{1e3 * c4.ours:.2f}") == ("0.80", "0.83")
    assert c4.gap_interval[0] < 0 < c4.gap_interval[1]
    assert f"{c4.cost_of_not_filtering_8b['harm_logshare_additive']:.2f}" == "1.58"
