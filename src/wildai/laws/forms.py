"""The ``Law`` base class: named, bounded coefficients and a vectorised, complex-step-safe prediction.

A law predicts loss (bits per byte) from the normalised coordinates of :mod:`wildai.laws.data`. Coefficients that must be
positive are fitted as logarithms (``logE`` for E, ``logeta`` for eta, ...); :meth:`Law.coefficients` maps them back.

``predict`` accepts a parameter vector with optional leading batch dimensions, so that one call evaluates every
complex-step perturbation of the Jacobian; every operation in a law must therefore be analytic (no ``abs``, no clipping).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from wildai.laws.data import Coordinates

Params = dict[str, np.ndarray]


@dataclass(frozen=True)
class Parameter:
    """One fitted coefficient: name, bounds, default start and the values the multi-start tries."""

    name: str
    lower: float
    upper: float
    initial: float
    grid: tuple[float, ...] = ()
    symbol: str = ""  # the coefficient's name in the paper; the exponential of the fitted value when ``log`` is set
    log: bool = False


def backbone_parameters() -> tuple[Parameter, ...]:
    """E + A n^-alpha + B (...)^-beta: the Chinchilla coefficients every law but Shukor's joint form shares."""
    return (
        Parameter("logE", -14.0, 2.0, -0.7, symbol="E", log=True),
        Parameter("logA", -10.0, 5.0, -1.5, symbol="A", log=True),
        Parameter("alpha", 0.02, 2.5, 0.35, symbol="alpha"),
        Parameter("logB", -10.0, 5.0, -0.7, symbol="B", log=True),
        Parameter("beta", 0.02, 2.5, 0.21, symbol="beta"),
    )


# Start values tried for the loss floor, whatever the law: the Chinchilla fit's own value, a low and a typical floor.
FLOOR_STARTS: tuple[float, ...] = (-3.0, -0.7)


def power0(base: np.ndarray, exponent: np.ndarray, positive: np.ndarray) -> np.ndarray:
    """``base ** exponent`` where ``positive`` holds and exactly 0 elsewhere, without log(0) in the derivatives."""
    safe = np.where(positive, base, 1.0)
    return np.where(positive, np.exp(exponent * np.log(safe)), 0.0)


class Law(ABC):
    """A scaling law with a fixed set of named coefficients."""

    key: str  # the published name, e.g. "ours_logshare_11"
    label: str  # the name the tables print
    citation: str | None = None  # bibtex key of the paper the form comes from
    parameters: tuple[Parameter, ...]
    random_starts: int = 15  # random starts of the multi-start (see wildai.laws.fit)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    @property
    def k(self) -> int:
        return len(self.parameters)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([p.lower for p in self.parameters]), np.array([p.upper for p in self.parameters])

    def unpack(self, theta: np.ndarray) -> Params:
        """Coefficient arrays that broadcast against the per-run coordinates."""
        theta = np.asarray(theta)
        return {name: theta[..., i, None] for i, name in enumerate(self.names)}

    def vector(self, values: Mapping[str, float]) -> np.ndarray:
        return np.array([values[name] for name in self.names], dtype=float)

    @abstractmethod
    def predict(self, theta: np.ndarray, x: Coordinates) -> np.ndarray:
        """Predicted loss (BPB) for every run; shape ``theta.shape[:-1] + (runs,)``."""

    def loss(self, values: Mapping[str, float], x: Coordinates) -> np.ndarray:
        """Predicted loss for one set of fitted values."""
        return np.asarray(self.predict(self.vector(values), x))

    def coefficients(self, values: Mapping[str, float]) -> dict[str, float]:
        """Fitted values in the paper's form: exponentials of log-coefficients, keyed by the paper's symbol."""
        out = {}
        for p in self.parameters:
            out[p.symbol or p.name] = float(np.exp(values[p.name])) if p.log else float(values[p.name])
        return out

    # ---- multi-start -------------------------------------------------------------------------------------------------
    def start(self, backbone: Mapping[str, float]) -> dict[str, float]:
        """Default start: every coefficient at its initial value, the backbone at the Chinchilla fit of the human runs."""
        values = {p.name: p.initial for p in self.parameters}
        values.update({k: v for k, v in backbone.items() if k in values})
        return values

    def perturb(self, start: Mapping[str, float], rng: np.random.Generator) -> dict[str, float]:
        """One random start: the floor and every coefficient with a grid drawn afresh, the rest from ``start``."""
        values = dict(start)
        values["logE"] = float(rng.choice([start["logE"], *FLOOR_STARTS]))
        for p in self.parameters:
            if p.grid:
                values[p.name] = float(rng.choice(p.grid))
        return values

    def extra_starts(self, backbone: Mapping[str, float]) -> list[dict[str, float]]:
        """Law-specific starts beyond the default and the random ones (none by default)."""
        return []


@dataclass(frozen=True)
class FittedLaw:
    """A law with fitted values, evaluated on raw token counts."""

    law: Law
    values: Mapping[str, float]

    def loss(self, n_params: np.ndarray | float, human_tokens: np.ndarray | float, ai_tokens: np.ndarray | float) -> np.ndarray:
        """Predicted loss (BPB) for N parameters trained on D_H human and D_AI AI tokens (broadcast arrays)."""
        x = Coordinates.from_tokens(n_params, human_tokens, ai_tokens)
        return np.ravel(self.law.loss(self.values, x))

    def loss1(self, n_params: float, human_tokens: float, ai_tokens: float) -> float:
        return float(self.loss(n_params, human_tokens, ai_tokens)[0])
