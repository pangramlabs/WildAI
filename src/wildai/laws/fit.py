"""Fitting a law: the paired objective, bounded robust least squares and a deterministic multi-start.

Objective. Every human-only control contributes the residual of its log loss, log L_hat - log L; every addition run (AI or
fresh human) contributes the residual of its change in log loss against its own control,
(log L_hat_i - log L_i) - (log L_hat_c - log L_c). Coefficients minimise the Huber loss (delta 0.1) of these residuals
with scipy's bounded trust-region least squares and an exact complex-step Jacobian. The loss floor E is bounded above by
the lowest fitted loss.

Multi-start. The backbone (E, A, alpha, B, beta) first comes from a Chinchilla fit of the human runs alone (controls and
fresh-human additions). Every law then starts from (1) that backbone with its other coefficients at their defaults,
(2) ``law.random_starts`` draws that keep the backbone and redraw the loss floor and each law-specific coefficient from its
grid (a fixed seed, so fits are reproducible), (3) the law's own extra starts and (4) any warm starts the caller passes.
Each start is solved independently and the lowest objective wins (the first start on ties), so the result does not depend
on how the starts are split across processes. The best solution is then re-solved from its own optimum ("polished").
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel
from scipy.optimize import OptimizeResult, least_squares

from wildai.laws.comparators import Chinchilla
from wildai.laws.data import Observations
from wildai.laws.forms import FLOOR_STARTS, Law

HUBER_DELTA = 0.1
ACTIVE_BOUND_TOLERANCE = 1e-4
POLISH_ROUNDS = 3
Start = Mapping[str, float]


@dataclass(frozen=True)
class FitOptions:
    random_starts: int | None = None  # None: the law's own number
    max_nfev: int = 4000
    tolerance: float = 2e-10
    seed: int = 20260912


DEFAULT_OPTIONS = FitOptions()

class LawFit(BaseModel):
    """The fitted coefficients of one law on one set of runs."""

    law: str
    target: str
    fold: str  # which runs were fitted: "all" (every fitted run) or "without_<size>"
    parameters: dict[str, float]  # fitted values (log-coefficients where the law fits logarithms)
    objective: float  # Huber cost at the optimum
    converged: bool
    active_bounds: list[str]
    runs: int
    starts: int  # starts that reached a finite optimum

    def vector(self, law: Law) -> np.ndarray:
        return law.vector(self.parameters)


@dataclass(frozen=True)
class Solution:
    """The optimum reached from one start."""

    index: int
    cost: float
    x: tuple[float, ...]
    converged: bool


def residuals(law: Law, obs: Observations) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """The paired residual vector and its complex-step Jacobian."""
    x, observed = obs.coords, np.log(obs.bpb)
    control, is_control = obs.control, obs.is_control

    def fun(theta: np.ndarray) -> np.ndarray:
        raw = np.log(law.predict(theta, x)) - observed
        return np.where(is_control, raw, raw - raw[..., control])

    def jac(theta: np.ndarray) -> np.ndarray:
        step = 1e-28
        z = theta.astype(complex)[None, :] + step * 1j * np.eye(len(theta))
        return fun(z).imag.T / step

    return fun, jac


def huber_cost(r: np.ndarray) -> float:
    """scipy's cost for loss='huber' with f_scale = HUBER_DELTA."""
    a = np.abs(r)
    return float(np.sum(np.where(a <= HUBER_DELTA, 0.5 * r * r, HUBER_DELTA * (a - HUBER_DELTA / 2))))


def objective(law: Law, obs: Observations, parameters: Start) -> float:
    fun, _ = residuals(law, obs)
    return huber_cost(fun(law.vector(parameters)))


def bounds_for(law: Law, obs: Observations) -> tuple[np.ndarray, np.ndarray]:
    """The law's bounds, with the loss floor E below every fitted loss."""
    lower, upper = law.bounds
    upper = upper.copy()
    upper[law.names.index("logE")] = np.log(obs.bpb.min())
    return lower, upper


def _least_squares(law: Law, obs: Observations, x0: np.ndarray, max_nfev: int, tolerance: float) -> OptimizeResult:
    fun, jac = residuals(law, obs)
    lower, upper = bounds_for(law, obs)
    x0 = np.clip(x0, lower + 1e-10, upper - 1e-10)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return least_squares(fun, x0, jac=jac, bounds=(lower, upper), loss="huber", f_scale=HUBER_DELTA, x_scale="jac",
                             max_nfev=max_nfev, ftol=tolerance, xtol=tolerance, gtol=tolerance)


def solve(law: Law, obs: Observations, starts: Sequence[Start], options: FitOptions = DEFAULT_OPTIONS, offset: int = 0) -> list[Solution]:
    """Solve from every start; starts whose prediction is not finite are skipped. ``offset`` numbers the starts."""
    out = []
    for i, start in enumerate(starts):
        try:
            result = _least_squares(law, obs, law.vector(start), options.max_nfev, options.tolerance)
        except (ValueError, FloatingPointError):
            continue
        if np.isfinite(result.cost):
            out.append(Solution(offset + i, float(result.cost), tuple(map(float, result.x)), bool(result.success)))
    return out


def select(law: Law, obs: Observations, solutions: Sequence[Solution], options: FitOptions = DEFAULT_OPTIONS, fold: str = "all") -> LawFit:
    """The lowest-cost solution (the first start on ties), then re-solved from its own optimum while that still lowers
    the objective: in the flat valleys of some fits the solver stops on its tolerance before the bottom."""
    if not solutions:
        raise RuntimeError(f"{law.key} on {obs.target}: no start reached a finite objective")
    best = min(solutions, key=lambda s: (s.cost, s.index))
    x, cost, converged = np.array(best.x), best.cost, best.converged
    for _ in range(POLISH_ROUNDS):
        refined = _least_squares(law, obs, x, 5 * options.max_nfev, options.tolerance)
        if not refined.cost < cost - 1e-15:
            break
        x, cost, converged = refined.x, float(refined.cost), bool(refined.success)
    lower, upper = bounds_for(law, obs)
    active = [n for n, v, lo, hi in zip(law.names, x, lower, upper) if min(abs(v - lo), abs(v - hi)) < ACTIVE_BOUND_TOLERANCE]
    return LawFit(law=law.key, target=obs.target, fold=fold, parameters=dict(zip(law.names, map(float, x))), objective=cost,
                  converged=converged, active_bounds=active, runs=len(obs), starts=len(solutions))


def chinchilla_backbone(obs: Observations, options: FitOptions = DEFAULT_OPTIONS) -> dict[str, float]:
    """Chinchilla fitted on the human runs only (controls and fresh-human additions): every law's backbone start."""
    human = obs.select(lambda run: run.arm != "ai")
    law = Chinchilla()
    starts = [dict(law.start({}), logE=e) for e in FLOOR_STARTS]
    return select(law, human, solve(law, human, starts, options), options).parameters


def starts_for(law: Law, backbone: Start, warm_starts: Sequence[Start] = (), options: FitOptions = DEFAULT_OPTIONS) -> list[dict[str, float]]:
    """Every start of the multi-start, in a fixed order."""
    default = law.start(backbone)
    count = law.random_starts if options.random_starts is None else options.random_starts
    rng = np.random.default_rng(options.seed)
    warm = [{**default, **{k: v for k, v in s.items() if k in default}} for s in warm_starts]
    return [default, *(law.perturb(default, rng) for _ in range(count)), *law.extra_starts(backbone), *warm]


def fit(law: Law, obs: Observations, warm_starts: Sequence[Start] = (), options: FitOptions = DEFAULT_OPTIONS, fold: str = "all") -> LawFit:
    """Fit ``law`` to ``obs`` in this process."""
    starts = starts_for(law, chinchilla_backbone(obs, options), warm_starts, options)
    return select(law, obs, solve(law, obs, starts, options), options, fold)
