"""The paper's law and the one-ingredient variants of its law-form ablation.

    L = E + A n^-alpha + B D_eff^-beta (1 + H)
    D_eff = d (1 + eta g(r)),   g(r) = R* (1 - exp(-r / R*)),   R* = K t^rho
    H = gamma t^u n^v [log(1 + r) - r / (1 + r)]

:class:`OursForm` describes the credit AI tokens earn and the harm they do; the paper's law is ``OursForm()``. The
ablation swaps one ingredient at a time (credit window shape, eta, the window's dependences; harm shape, its dependences
and whether it scales the data term or is added to the loss).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from wildai.laws.data import Coordinates
from wildai.laws.forms import Law, Parameter, backbone_parameters, power0

Credit = Literal["none", "linear", "exp", "rational", "tanh"]
HarmShape = Literal["none", "logshare", "log", "boxcox", "power"]

# Grids of the multi-start for the credit and harm coefficients.
LOGETA_GRID = (-2.0, 0.0, 2.0, 4.0, 6.0)
LOGK_GRID = (-3.0, -1.0, 1.0, 3.0)
RHO_GRID = (-2.0, -1.0, 0.0)
GAMMA_GRID = (0.001, 0.01, 0.1)
U_GRID = (-1.0, 0.0, 1.0)
V_GRID = (-0.5, 0.0, 0.5)
POWER_GRID = (0.3, 1.0, 2.0)


@dataclass(frozen=True)
class OursForm:
    """Which credit and which harm; the defaults are the paper's law."""

    credit: Credit = "exp"
    free_eta: bool = True  # eta fitted; False fixes eta = 1 (the CD credit)
    budget_window: bool = True  # R* depends on t (rho)
    size_window: bool = False  # R* also depends on n (sigma)
    harm: HarmShape = "logshare"
    harm_on_data: bool = True  # H multiplies the data term; False adds it to the loss
    harm_budget: bool = True  # H depends on t (u)
    harm_size: bool = True  # H depends on n (v)

    @property
    def windowed(self) -> bool:
        return self.credit in ("exp", "rational", "tanh")


PAPER_FORM = OursForm()


def box_cox_log1p(r: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """((1 + r)^lambda - 1) / lambda, the Box-Cox transform of log(1 + r), with its series near lambda = 0."""
    x = np.log1p(r)
    small = np.abs(np.real(lam)) < 1e-5
    safe = np.where(small, 1.0, lam)
    series = x + lam * x * x / 2 + lam**2 * x**3 / 6 + lam**3 * x**4 / 24 + lam**4 * x**5 / 120
    return np.where(small, series, np.expm1(safe * x) / safe)


class Ours(Law):
    """Chinchilla in an effective human token count, with a saturating credit and a decelerating harm."""

    citation = None

    def __init__(self, key: str, label: str, form: OursForm = PAPER_FORM) -> None:
        self.key, self.label, self.form = key, label, form
        f = form
        params = list(backbone_parameters())
        if f.credit != "none" and f.free_eta:
            params.append(Parameter("logeta", -8.0, 7.0, -0.5, LOGETA_GRID, symbol="eta", log=True))
        if f.windowed:
            params.append(Parameter("logK", -8.0, 5.0, -1.0, LOGK_GRID, symbol="K", log=True))
            if f.budget_window:
                params.append(Parameter("rho", -6.0, 3.0, -1.0, RHO_GRID, symbol="rho"))
            if f.size_window:
                params.append(Parameter("sigma", -3.0, 3.0, 0.0, symbol="sigma"))
        if f.harm != "none":
            params.append(Parameter("gamma", 0.0, 5.0, 0.05, GAMMA_GRID, symbol="gamma"))
            if f.harm == "power":
                params.append(Parameter("power", 0.1, 4.0, 1.0, POWER_GRID, symbol="p"))
            if f.harm_budget:
                params.append(Parameter("u", -3.0, 3.0, 0.0, U_GRID, symbol="u"))
            if f.harm_size:
                params.append(Parameter("v", -3.0, 3.0, 0.0, V_GRID, symbol="v"))
            if f.harm == "boxcox":
                params.append(Parameter("lam", -2.0, 1.0, 0.0, symbol="lambda"))
        self.parameters = tuple(params)

    def window(self, p: dict[str, np.ndarray], x: Coordinates) -> np.ndarray:
        """The saturation scale R* = K t^rho (n^sigma)."""
        return np.exp(p["logK"] + p.get("rho", 0.0) * np.log(x.t) + p.get("sigma", 0.0) * np.log(x.n))

    def credit(self, p: dict[str, np.ndarray], x: Coordinates) -> np.ndarray:
        """D_eff / D_H - 1: the human-token credit of the AI tokens, per human token."""
        f, r = self.form, x.r
        eta = np.exp(p["logeta"]) if "logeta" in p else 1.0
        if f.credit == "none":
            return np.zeros_like(r)
        if f.credit == "linear":
            return eta * r
        window = self.window(p, x)
        z = r / window
        if f.credit == "exp":
            g = -window * np.expm1(-z)
        elif f.credit == "rational":
            g = r / (1 + z)
        else:
            g = window * np.tanh(z)
        return eta * g

    def harm_shape(self, p: dict[str, np.ndarray], r: np.ndarray) -> np.ndarray:
        shape = self.form.harm
        if shape == "logshare":
            return np.log1p(r) - r / (1 + r)
        if shape == "log":
            return np.log1p(r)
        if shape == "boxcox":
            return box_cox_log1p(r, p["lam"])
        return power0(r, p["power"], r > 0)

    def harm(self, p: dict[str, np.ndarray], x: Coordinates) -> np.ndarray:
        """H: the relative harm of the AI tokens (an additive loss term when the harm is not on the data term)."""
        if self.form.harm == "none":
            return np.zeros_like(x.r)
        return p["gamma"] * x.t ** p.get("u", 0.0) * x.n ** p.get("v", 0.0) * self.harm_shape(p, x.r)

    def predict(self, theta: np.ndarray, x: Coordinates) -> np.ndarray:
        p = self.unpack(theta)
        model = np.exp(p["logA"]) * x.n ** (-p["alpha"])
        data = np.exp(p["logB"]) * (x.d * (1 + self.credit(p, x))) ** (-p["beta"])
        harm = self.harm(p, x)
        if self.form.harm_on_data:
            return np.exp(p["logE"]) + model + data * (1 + harm)
        return np.exp(p["logE"]) + model + data + harm


def paper_law() -> Ours:
    return Ours("ours_logshare_11", "Ours")


# The law-form ablation (appendix): the paper's law with one ingredient changed, keyed as in the published table.
ABLATION_FORMS: dict[str, tuple[str, str, OursForm]] = {
    # name: (group, description, form)
    "benefit_none": ("benefit", "No credit: AI tokens add no data", OursForm(credit="none")),
    "benefit_linear": ("benefit", r"Linear, $\eta r$ (no saturation)", OursForm(credit="linear")),
    "benefit_exp_unit": ("benefit", r"Exponential window, $\eta=1$ (CD's credit)", OursForm(free_eta=False)),
    "benefit_rational": ("benefit", r"Rational window, free $\eta$", OursForm(credit="rational")),
    "benefit_tanh": ("benefit", r"Tanh window, free $\eta$", OursForm(credit="tanh")),
    "benefit_exp_constant": ("benefit", r"Exponential window, no budget dependence ($\rho=0$)", OursForm(budget_window=False)),
    "benefit_exp_size": ("benefit", r"Exponential window, also size-dependent ($R^\star=K t^\rho n^\sigma$)", OursForm(size_window=True)),
    "ours": ("both", "Exponential window, free $\\eta$, $R^\\star=K t^\\rho$, harm $\\gamma t^{u}n^{v}[\\log(1+r)-r/(1+r)]$ on the data term (ours)", OursForm()),
    "harm_none": ("harm", "No harm (credit only)", OursForm(harm="none", harm_budget=False, harm_size=False)),
    "harm_logshare_constant": ("harm", r"$\gamma\,[\log(1+r)-r/(1+r)]$ on the data term, no budget or size dependence", OursForm(harm_budget=False, harm_size=False)),
    "harm_logshare_budget": ("harm", r"$\gamma t^{u}[\log(1+r)-r/(1+r)]$ on the data term, no size dependence", OursForm(harm_size=False)),
    "harm_logshare_size": ("harm", r"$\gamma n^{v}[\log(1+r)-r/(1+r)]$ on the data term, no budget dependence", OursForm(harm_budget=False)),
    "harm_logshare_additive": ("harm", r"$\gamma t^{u}n^{v}[\log(1+r)-r/(1+r)]$ added to the loss, not tied to the data term", OursForm(harm_on_data=False)),
    "harm_log": ("harm", r"$\gamma t^{u}n^{v}\log(1+r)$ on the data term (harm starts at the first AI token)", OursForm(harm="log")),
    "harm_boxcox": ("harm", r"Box--Cox of $\log(1+r)$ with free curvature $\lambda$, on the data term", OursForm(harm="boxcox")),
    "harm_power": ("harm", r"Power law $\gamma r^{p} t^{u} n^{v}$ added to the loss \citep{lovelace2026prescriptive}", OursForm(harm="power", harm_on_data=False)),
}


def ablation_law(name: str) -> Ours:
    _group, description, form = ABLATION_FORMS[name]
    return Ours(name, description, form)
