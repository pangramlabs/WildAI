"""Law formulas: parameter counts, analytic (complex-step) derivatives, and the limits the paper relies on."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from wildai.laws.bank import resolve_law
from wildai.laws.catalog import benchmark_laws
from wildai.laws.comparators import Chinchilla
from wildai.laws.data import Coordinates
from wildai.laws.ours import ABLATION_FORMS, OursForm, ablation_law, paper_law

# Published k of Table 1 and of the law-form ablation table.
BENCHMARK_K = {"chinchilla_5": 5, "muennighoff_7": 7, "cd_8": 8, "lovelace_eq8_9": 9, "atlas_no_repeat": 6, "he_human_share": 6,
               "hamidieh_first_order": 7, "shukor_additive": 9, "shukor_joint": 13, "sedova_ai_share_coupled": 8, "jain_token_weighted": 7,
               "ours_logshare_11": 11}
ABLATION_K = {"benefit_none": 8, "benefit_linear": 9, "benefit_exp_unit": 10, "benefit_rational": 11, "benefit_tanh": 11, "benefit_exp_constant": 10,
              "benefit_exp_size": 12, "ours": 11, "harm_none": 8, "harm_logshare_constant": 9, "harm_logshare_budget": 10, "harm_logshare_size": 10,
              "harm_logshare_additive": 11, "harm_log": 11, "harm_boxcox": 12, "harm_power": 12}
X = Coordinates.from_tokens(np.array([2e7, 1e8, 2.7e8, 9.7e8, 2e7]), np.array([1e8, 2e9, 5e9, 4e10, 5e8]), np.array([0.0, 5e7, 5e9, 1e10, 3.2e10]))


def test_parameter_counts() -> None:
    assert {key: law.k for key, law in benchmark_laws().items()} == BENCHMARK_K
    assert list(benchmark_laws()) == list(BENCHMARK_K)  # Table 1 order
    assert {name: ablation_law(name).k for name in ABLATION_FORMS} == ABLATION_K


@pytest.mark.parametrize("key", [*BENCHMARK_K, *ABLATION_K])
def test_complex_step_derivatives(key: str) -> None:
    law = resolve_law(key)
    theta = np.array([p.initial for p in law.parameters]) + 0.01
    z = theta[None, :].astype(complex) + 1e-28j * np.eye(len(theta))
    exact = law.predict(z, X).imag.T / 1e-28
    numeric = approx_derivative(lambda th: law.predict(th, X), theta, method="3-point")
    np.testing.assert_allclose(exact, numeric, rtol=1e-5, atol=1e-8)
    assert np.all(np.isfinite(law.predict(theta, X))) and np.all(law.predict(theta, X) > 0)


def test_paper_law_reduces_to_chinchilla_without_ai() -> None:
    ours, chinchilla = paper_law(), Chinchilla()
    values = {p.name: p.initial + 0.1 for p in ours.parameters}
    human = Coordinates.from_tokens(X.n * 1e8, X.d * 1e9, 0.0)
    np.testing.assert_allclose(ours.loss(values, human), chinchilla.loss(values, human), rtol=1e-14)


def test_paper_law_matches_its_equation() -> None:
    """Equations (law), (law-credit) and (law-harm) of the paper, written out independently."""
    ours = paper_law()
    p = {"logE": -0.3, "logA": -1.4, "alpha": 0.3, "logB": -2.5, "beta": 0.47, "logeta": 0.55, "logK": -6.2, "rho": -3.3, "gamma": 0.4, "u": 1.2, "v": 0.44}
    n, d, r, t = X.n, X.d, X.r, X.t
    window = np.exp(p["logK"]) * t ** p["rho"]
    d_eff = d * (1 + np.exp(p["logeta"]) * window * (1 - np.exp(-r / window)))
    harm = p["gamma"] * t ** p["u"] * n ** p["v"] * (np.log(1 + r) - r / (1 + r))
    expected = np.exp(p["logE"]) + np.exp(p["logA"]) / n ** p["alpha"] + np.exp(p["logB"]) / d_eff ** p["beta"] * (1 + harm)
    np.testing.assert_allclose(ours.loss(p, X), expected, rtol=1e-13)


def test_harm_shape() -> None:
    """log(1 + r) - r / (1 + r) grows as r^2 / 2 for small r and as log r - 1 for large r."""
    law = paper_law()
    p = law.unpack(np.zeros(law.k))
    assert law.harm_shape(p, np.array([1e-4]))[0] == pytest.approx(0.5e-8, rel=1e-3)
    assert law.harm_shape(p, np.array([1e6]))[0] == pytest.approx(np.log(1e6) - 1, rel=1e-5)


def test_ablation_forms_differ_only_as_named() -> None:
    assert OursForm() == ABLATION_FORMS["ours"][2]
    for name, (_group, _description, form) in ABLATION_FORMS.items():
        changed = {f for f in OursForm.__dataclass_fields__ if getattr(form, f) != getattr(OursForm(), f)}
        assert len(changed) <= 3, (name, changed)
