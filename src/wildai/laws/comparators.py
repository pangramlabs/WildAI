"""The published laws the paper benchmarks, written in the normalised coordinates of :mod:`wildai.laws.data`.

With M = E + A n^-alpha, T = d (1 + r), h = 1 / (1 + r) (human share) and f = r / (1 + r) (AI share):

    Chinchilla (Hoffmann et al.)        M + B T^-beta                              every AI token is a human token
    Muennighoff et al.                  E + A N_eff^-alpha + B D_eff^-beta         AI tokens as repeated data
    CD (Qin et al.)                     M + B [d (1 + R* (1 - e^{-r/R*}))]^-beta,  R* = K t^rho n^sigma
    Lovelace et al. (Eq. 8)             M + B T^-beta + gamma r^p t^u n^v
    ATLAS (Longpre et al.)              M + B [d (1 + q r)]^-beta
    He et al.                           (M + B T^-beta) h^-zeta
    Hamidieh et al. (first order)       M + B T^-beta exp(-a_H h log D_H - a_A f log D_A)
    Shukor et al., additive             M + B T^-beta + 1 / (c_H h^q_H + c_A f^q_A)
    Shukor et al., joint                E + (a_H h + a_A f)^p_A n^-alpha + (b_H h + b_A f)^p_B T^-beta + 1 / (c_H h^q_H + c_A f^q_A)
    Sedova et al.                       M + B n^c [d (1 + q r)]^-beta + gamma f
    Jain et al.                         M + B [(1 + q r) / (d (1 + r)^2)]^beta + gamma f^2

Coefficients are fitted on our runs; nothing is taken from the original papers.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from wildai.laws.data import Coordinates
from wildai.laws.forms import Law, Parameter, Params, backbone_parameters, power0

TRANSFER_GRID = (-4.0, -1.0, 0.0, 2.0, 4.0, 6.0)
MIX_GRID = (1.0, 3.0, 5.0, 8.0)
MIX_POWER_GRID = (0.2, 0.6, 1.0, 2.0)
AMPLITUDE_POWER_GRID = (0.3, 1.0, 2.0)


class _Backbone(Law):
    """Laws built on M = E + A n^-alpha with a data term the subclass defines."""

    extra: tuple[Parameter, ...] = ()

    def __init__(self) -> None:
        self.parameters = backbone_parameters() + self.extra

    @staticmethod
    def model_term(p: Params, x: Coordinates) -> np.ndarray:
        return np.exp(p["logE"]) + np.exp(p["logA"]) * x.n ** (-p["alpha"])

    @staticmethod
    def total(x: Coordinates) -> np.ndarray:
        """T = d (1 + r), all training tokens in billions."""
        return x.d * (1 + x.r)


class Chinchilla(_Backbone):
    key, label, citation = "chinchilla_5", "Chinchilla", "hoffmann2022chinchilla"

    def predict(self, theta: np.ndarray, x: Coordinates) -> np.ndarray:
        p = self.unpack(theta)
        return self.model_term(p, x) + np.exp(p["logB"]) * self.total(x) ** (-p["beta"])


class Muennighoff(_Backbone):
    """Human tokens are the unique data (U_D = D_H) and AI tokens the repetitions (R_D = r).

    N_eff = U_N (1 + R*_N (1 - exp(-R_N / R*_N))) with U_N = min(N, N_opt(D_H)), R_N = N / U_N - 1, and N_opt the
    compute-optimal size for D_H under the law's own coefficients.
    """

    key, label, citation = "muennighoff_7", "Muennighoff et al.", "muennighoff2023scaling"
    extra = (
        Parameter("logK", -8.0, 5.0, -1.0, (-3.0, -1.0, 1.0, 3.0), symbol="R*_D", log=True),
        # The upper bound lets N_eff reach N, so the law with N_eff = N is (almost exactly) nested.
        Parameter("log_rn_star", -8.0, 14.0, 1.0, symbol="R*_N", log=True),
    )

    def predict(self, theta: np.ndarray, x: Coordinates) -> np.ndarray:
        p = self.unpack(theta)
        a, b = np.exp(p["logA"]), np.exp(p["logB"])
        rd, rn = np.exp(p["logK"]), np.exp(p["log_rn_star"])
        d_eff = x.d * (1.0 - rd * np.expm1(-x.r / rd))
        n_opt = (p["alpha"] * a / (p["beta"] * b)) ** (1.0 / p["alpha"]) * x.d ** (p["beta"] / p["alpha"])
        u_n = np.where(np.real(n_opt) < x.n, n_opt, x.n)
        n_eff = u_n * (1.0 - rn * np.expm1(-(x.n / u_n - 1.0) / rn))
        return np.exp(p["logE"]) + a * n_eff ** (-p["alpha"]) + b * d_eff ** (-p["beta"])

    def extra_starts(self, backbone: Mapping[str, float]) -> list[dict[str, float]]:
        # Every pair of windows R*_D and R*_N, R*_N at its upper bound included (N_eff = N: no parameter term).
        start = self.start(backbone)
        return [dict(start, logK=k, log_rn_star=m) for k in (-3.0, -1.0, 1.0, 3.0) for m in (-3.0, -1.0, 1.0, 3.0, 14.0 - 1e-6)]


class CD(_Backbone):
    key, label, citation = "cd_8", "CD", "qin2026bridging"
    extra = (
        Parameter("logK", -8.0, 5.0, -1.0, (-3.0, -1.0, 1.0, 3.0), symbol="K", log=True),
        Parameter("rho", -6.0, 3.0, -1.0, (-2.0, -1.0, 0.0), symbol="rho"),
        Parameter("sigma", -3.0, 3.0, 0.0, symbol="sigma"),
    )

    def predict(self, theta: np.ndarray, x: Coordinates) -> np.ndarray:
        p = self.unpack(theta)
        window = np.exp(p["logK"] + p["rho"] * np.log(x.t) + p["sigma"] * np.log(x.n))
        effective = x.d * (1 - window * np.expm1(-x.r / window))
        return self.model_term(p, x) + np.exp(p["logB"]) * effective ** (-p["beta"])


class Lovelace(_Backbone):
    key, label, citation = "lovelace_eq8_9", "Lovelace et al.", "lovelace2026prescriptive"
    extra = (
        Parameter("gamma", 0.0, 5.0, 0.05, (0.001, 0.01, 0.1), symbol="gamma"),
        Parameter("power", 0.1, 4.0, 1.0, (0.3, 1.0, 2.0), symbol="p"),
        Parameter("u", -3.0, 3.0, 0.0, (-1.0, 0.0, 1.0), symbol="u"),
        Parameter("v", -3.0, 3.0, 0.0, (-0.5, 0.0, 0.5), symbol="v"),
    )

    def predict(self, theta: np.ndarray, x: Coordinates) -> np.ndarray:
        p = self.unpack(theta)
        penalty = p["gamma"] * power0(x.r, p["power"], x.r > 0) * x.t ** p["u"] * x.n ** p["v"]
        return self.model_term(p, x) + np.exp(p["logB"]) * self.total(x) ** (-p["beta"]) + penalty


class Atlas(_Backbone):
    key, label, citation = "atlas_no_repeat", "ATLAS", "longpre2026atlas"
    extra = (Parameter("log_transfer", -16.0, 16.0, 0.0, TRANSFER_GRID, symbol="q", log=True),)

    def predict(self, theta: np.ndarray, x: Coordinates) -> np.ndarray:
        p = self.unpack(theta)
        return self.model_term(p, x) + np.exp(p["logB"]) * (x.d * (1 + np.exp(p["log_transfer"]) * x.r)) ** (-p["beta"])


class He(_Backbone):
    key, label, citation = "he_human_share", "He et al.", "he2025scaling"
    extra = (Parameter("family_power", 0.0, 3.0, 0.03, (0.0, 0.005, 0.03, 0.15), symbol="zeta"),)

    def predict(self, theta: np.ndarray, x: Coordinates) -> np.ndarray:
        p = self.unpack(theta)
        human_share = 1 / (1 + x.r)
        return (self.model_term(p, x) + np.exp(p["logB"]) * self.total(x) ** (-p["beta"])) * np.exp(-p["family_power"] * np.log(human_share))


class Hamidieh(_Backbone):
    """First-order domain adjustment; the token counts inside the logarithms are raw counts, not billions."""

    key, label, citation = "hamidieh_first_order", "Hamidieh et al.", "hamidieh2025domainaware"
    extra = (
        Parameter("domain_h_rate", -2.5, 2.5, 0.0, symbol="a_H"),
        Parameter("domain_ai_rate", -2.5, 2.5, 0.0, symbol="a_A"),
    )

    def __init__(self) -> None:
        super().__init__()
        # B absorbs the first-order adjustment D_H^-a_H of the human-only runs, so it needs a much wider range.
        self.parameters = tuple(Parameter("logB", -80.0, 80.0, -0.7, symbol="B", log=True) if q.name == "logB" else q for q in self.parameters)

    def predict(self, theta: np.ndarray, x: Coordinates) -> np.ndarray:
        p = self.unpack(theta)
        human_share, ai_share = 1 / (1 + x.r), x.r / (1 + x.r)
        log_human = np.log(x.d) + np.log(1e9)
        log_ai = log_human + np.log(np.where(x.r > 0, x.r, 1.0))
        adjustment = -p["domain_h_rate"] * human_share * log_human - p["domain_ai_rate"] * ai_share * log_ai
        return self.model_term(p, x) + np.exp(p["logB"]) * np.exp(-p["beta"] * np.log(self.total(x)) + adjustment)

    def perturb(self, start: Mapping[str, float], rng: np.random.Generator) -> dict[str, float]:
        # Start the domain rates near zero and move B and beta so the human-only data term stays near the Chinchilla one.
        values = super().perturb(start, rng)
        rate = float(rng.choice([-0.05, 0.0, 0.05, 0.15]))
        values.update(domain_h_rate=rate, domain_ai_rate=rate + float(rng.choice([-0.05, 0.0, 0.05])),
                      logB=start["logB"] + np.log(1e9) * rate, beta=start["beta"] - rate)
        return values


def _mixture_floor(p: Params, x: Coordinates) -> np.ndarray:
    """1 / (c_H h^q_H + c_A f^q_A), Shukor et al.'s mixture term."""
    human_share, ai_share = 1 / (1 + x.r), x.r / (1 + x.r)
    ai_power = power0(ai_share, p["mix_ai_power"], ai_share > 0)
    return 1 / (np.exp(p["log_mix_h"]) * human_share ** p["mix_h_power"] + np.exp(p["log_mix_ai"]) * ai_power)


class ShukorAdditive(_Backbone):
    key, label, citation = "shukor_additive", "Shukor additive", "shukor2025scaling"
    extra = (
        Parameter("log_mix_h", -8.0, 16.0, 3.0, MIX_GRID, symbol="c_H", log=True),
        Parameter("log_mix_ai", -8.0, 16.0, 3.0, MIX_GRID, symbol="c_A", log=True),
        Parameter("mix_h_power", 0.02, 4.0, 1.0, MIX_POWER_GRID, symbol="q_H"),
        Parameter("mix_ai_power", 0.02, 4.0, 1.0, MIX_POWER_GRID, symbol="q_A"),
    )

    def predict(self, theta: np.ndarray, x: Coordinates) -> np.ndarray:
        p = self.unpack(theta)
        return self.model_term(p, x) + np.exp(p["logB"]) * self.total(x) ** (-p["beta"]) + _mixture_floor(p, x)


class ShukorJoint(Law):
    """The shares rescale both amplitudes: A = (a_H h + a_A f)^p_A and B = (b_H h + b_A f)^p_B."""

    key, label, citation = "shukor_joint", "Shukor joint", "shukor2025scaling"
    # Thirteen coefficients and an objective with many nearly equal minima: 200 random starts reach the lowest minimum
    # we know of (from 1,000 starts) on every paper target; 15 do not.
    random_starts = 200

    def __init__(self) -> None:
        self.parameters = (
            Parameter("logE", -14.0, 2.0, -0.7, symbol="E", log=True),
            Parameter("alpha", 0.02, 2.5, 0.35, symbol="alpha"),
            Parameter("beta", 0.02, 2.5, 0.21, symbol="beta"),
            Parameter("log_mix_h", -8.0, 16.0, 3.0, MIX_GRID, symbol="c_H", log=True),
            Parameter("log_mix_ai", -8.0, 16.0, 3.0, MIX_GRID, symbol="c_A", log=True),
            Parameter("mix_h_power", 0.02, 8.0, 1.0, MIX_POWER_GRID, symbol="q_H"),
            Parameter("mix_ai_power", 0.02, 8.0, 1.0, MIX_POWER_GRID, symbol="q_A"),
            Parameter("log_model_h", -40.0, 40.0, -1.5, symbol="a_H", log=True),
            Parameter("log_model_ai", -40.0, 40.0, -1.5, symbol="a_A", log=True),
            Parameter("model_mix_power", -6.0, 6.0, 1.0, AMPLITUDE_POWER_GRID, symbol="p_A"),
            Parameter("log_data_h", -40.0, 40.0, -0.7, symbol="b_H", log=True),
            Parameter("log_data_ai", -40.0, 40.0, -0.7, symbol="b_A", log=True),
            Parameter("data_mix_power", -6.0, 6.0, 1.0, AMPLITUDE_POWER_GRID, symbol="p_B"),
        )

    def predict(self, theta: np.ndarray, x: Coordinates) -> np.ndarray:
        p = self.unpack(theta)
        human_share, ai_share = 1 / (1 + x.r), x.r / (1 + x.r)
        model = (np.exp(p["log_model_h"]) * human_share + np.exp(p["log_model_ai"]) * ai_share) ** p["model_mix_power"]
        data = (np.exp(p["log_data_h"]) * human_share + np.exp(p["log_data_ai"]) * ai_share) ** p["data_mix_power"]
        total = x.d * (1 + x.r)
        return np.exp(p["logE"]) + model * x.n ** (-p["alpha"]) + data * total ** (-p["beta"]) + _mixture_floor(p, x)

    def start(self, backbone: Mapping[str, float]) -> dict[str, float]:
        # Both sources start at the Chinchilla amplitudes.
        values = super().start(backbone)
        values.update(log_model_h=backbone.get("logA", -1.0), log_model_ai=backbone.get("logA", -1.0),
                      log_data_h=backbone.get("logB", -1.0), log_data_ai=backbone.get("logB", -1.0))
        return values

    def perturb(self, start: Mapping[str, float], rng: np.random.Generator) -> dict[str, float]:
        values = super().perturb(start, rng)
        for name in ("log_model_ai", "log_data_ai"):
            values[name] += float(rng.choice([-2.0, 0.0, 2.0]))
        return values


class Sedova(_Backbone):
    """The AI share is the penalised source; the size coupling n^c of the data term is free."""

    key, label, citation = "sedova_ai_share_coupled", "Sedova et al.", "sedova2026scalinglawsmixturepretraining"
    extra = (
        Parameter("log_transfer", -16.0, 16.0, 0.0, TRANSFER_GRID, symbol="q", log=True),
        Parameter("share_tax", 0.0, 5.0, 0.03, symbol="gamma"),
        Parameter("size_coupling", 0.0, 2.5, 0.1, symbol="c"),
    )

    def predict(self, theta: np.ndarray, x: Coordinates) -> np.ndarray:
        p = self.unpack(theta)
        effective = x.d * (1 + np.exp(p["log_transfer"]) * x.r)
        ai_share = x.r / (1 + x.r)
        return self.model_term(p, x) + np.exp(p["logB"]) * x.n ** p["size_coupling"] * effective ** (-p["beta"]) + p["share_tax"] * ai_share


class Jain(_Backbone):
    """Token-proportional source weight, a common data exponent and the capacity term of their Section 5."""

    key, label, citation = "jain_token_weighted", "Jain et al.", "jain2024scaling"
    extra = (
        Parameter("log_surrogate_noise", -16.0, 16.0, 0.0, symbol="q", log=True),
        Parameter("mismatch", 0.0, 5.0, 0.03, symbol="gamma"),
    )

    def predict(self, theta: np.ndarray, x: Coordinates) -> np.ndarray:
        p = self.unpack(theta)
        data_factor = (1 + np.exp(p["log_surrogate_noise"]) * x.r) / (1 + x.r) ** 2
        ai_share = x.r / (1 + x.r)
        return self.model_term(p, x) + np.exp(p["logB"]) * (data_factor / x.d) ** p["beta"] + p["mismatch"] * ai_share**2

    def extra_starts(self, backbone: Mapping[str, float]) -> list[dict[str, float]]:
        # The AI-token noise ratio q spans many orders of magnitude.
        return [dict(self.start(backbone), log_surrogate_noise=q) for q in (-6.0, -2.0, 2.0, 6.0, 10.0)]
