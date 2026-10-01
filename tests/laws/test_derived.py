"""Derived quantities quoted in the paper: cost of not filtering, optimal AI share, token value, filtering and masking."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wildai.laws.analysis import filtering, masking
from wildai.laws.analysis.recommendations import reference_size
from wildai.laws.analysis.stability import projection
from wildai.laws.bank import Case
from wildai.laws.catalog import law
from wildai.laws.ceg import WEB_2026_AI_SHARE, ReferenceMix, ai_token_value, cost_of_not_filtering, optimal_share
from wildai.laws.data import Study
from wildai.laws.fit import LawFit, fit
from wildai.laws.forms import FittedLaw
from wildai.laws.ours import paper_law


@pytest.fixture(scope="module")
def c4(c4_fit: LawFit) -> FittedLaw:
    return FittedLaw(paper_law(), c4_fit.parameters)


def test_cost_of_not_filtering(study: Study, c4: FittedLaw) -> None:
    """Sections 1 and 4: 1.3x at our 2026 crawl's share, 1.6x at August 2026's, 2.1x and 3.0x at the 2027 and 2028 forecasts
    (268M, 20 TPP_h); 1.59x at 8B (appendix)."""
    n = reference_size(study)
    costs = [cost_of_not_filtering(c4, n, 20.0, share) for share in (WEB_2026_AI_SHARE, 0.311, 0.423, 0.507)]
    assert [f"{c:.1f}" for c in costs] == ["1.3", "1.6", "2.1", "3.0"]
    assert f"{projection(c4)[0]:.2f}" == "1.59"
    assert cost_of_not_filtering(c4, n, 20.0, 0.0) == pytest.approx(1.0, abs=1e-9)


def test_optimal_share(study: Study, c4: FittedLaw) -> None:
    """Section 5: the loss-minimising AI share on C4 is 37 % at 5 TPP_h and 1.0 % at 20 TPP_h."""
    n = reference_size(study)
    assert f"{100 * optimal_share(c4, n, 5 * n):.0f}" == "37"
    assert f"{100 * optimal_share(c4, n, 20 * n):.1f}" == "1.0"


def test_token_value_turns_negative(study: Study, c4: FittedLaw) -> None:
    """Figure 1: at 20 TPP_h the value of an AI token falls below zero once r exceeds about 0.08."""
    n = reference_size(study)
    ratios = np.geomspace(0.01, 1.0, 400)
    first_negative = ratios[np.argmax(ai_token_value(c4, n, 20 * n, ratios) < 0)]
    assert 0.07 < first_negative < 0.085
    assert np.all(ai_token_value(c4, n, 5 * n, np.array([0.01, 0.1])) > 0)


def test_reference_mix_round_trip(study: Study, c4: FittedLaw) -> None:
    n = reference_size(study)
    reference = ReferenceMix(c4, n, (2.9, 169.0))
    tokens = 20 * n / (1 - WEB_2026_AI_SHARE)
    loss = c4.loss1(n, (1 - WEB_2026_AI_SHARE) * tokens, WEB_2026_AI_SHARE * tokens)
    ceg, status = reference.ceg(loss, tokens)
    assert status == "ok" and ceg == pytest.approx(1.0, rel=1e-9)
    assert reference.tokens_for(0.1) == (None, "below_range")


def test_filtering_direction(study: Study, c4_fit: LawFit) -> None:
    """Section 4: our law gets the direction of filtering right in 18 of 19 pairs (paired error 3.28e-3); Chinchilla predicts
    a higher loss in all 13 pairs where removing the AI text lowers it."""
    chinchilla = law("chinchilla_5")
    fits = {Case("ours_logshare_11", "c4"): c4_fit, Case("chinchilla_5", "c4"): fit(chinchilla, study.observations("c4", study.split("fit")))}
    _records, summaries = filtering.analyse(study, fits, ["ours_logshare_11", "chinchilla_5"], ["c4"])
    ours, chin = summaries
    assert (ours.pairs, ours.direction_right, f"{1e3 * ours.paired_rmse:.2f}") == (19, 18, "3.28")
    assert (chin.filtering_helps, chin.predicted_harm_where_it_helps) == (13, 13)


def test_masking(study: Study) -> None:
    """Appendix: 243 AI additions raise loss on human-labelled text; a mixed validation set reports 49.4 % of them as
    improvements at 5 % AI and 95.5 % at 22.3 %; the median flips sign at 5.1 % and is -1.9 % at 22.3 %."""
    summary = masking.summarise(study, masking.contrasts(study))
    assert (summary.ai_runs, summary.human_harmed, summary.ai_improved) == (553, 243, 553)
    at = {round(a.ai_share, 3): a for a in summary.at_share}
    assert f"{100 * at[0.05].masked_share:.1f}" == "49.4" and f"{100 * at[0.223].masked_share:.1f}" == "95.5"
    assert f"{100 * summary.median_flip_share:.1f}" == "5.1" and f"{100 * at[0.223].median_reported_change:.1f}" == "-1.9"
    assert [f"{100 * x:.0f}" for x in summary.control_ai_loss_reduction_range] == ["12", "27"]
    assert math.isclose(summary.at_share[0].median_reported_change, 0.005784516, rel_tol=1e-6)
