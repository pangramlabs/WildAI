"""Shared pieces of the scaling-law figures: fitted laws from results/laws/fits.json, typed readers for the other files of
results/laws/, the law's predicted change in loss for a control group, target labels and styles, and the colour scales of
human budgets and model sizes."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from functools import cache
from typing import TypeVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from pydantic import BaseModel

from paper.groups import FITTED_DEPTHS
from paper.results import RESULTS, Model, models
from paper.style import PALETTE, TPP_H_LABEL
from wildai.laws.bank import resolve_law
from wildai.laws.catalog import PAPER_LAW
from wildai.laws.data import TARGET_LABELS, Coordinates
from wildai.laws.forms import FittedLaw
from wildai.laws.ours import Ours
from wildai.laws.records import FitRecord, read_fits

LAWS = RESULTS / "laws"
REFERENCE_DEPTH = 16  # the 268M models: the size the paper's recommendations are drawn at

VALUE_LABEL = "value of an AI token\n(in human tokens)"
ADD_AI_LABEL, ADD_HUMAN_LABEL = "add AI text", "add fresh human text"
RATIOS = np.logspace(-3, 2, 400)  # AI ratios of the law curves

SIZE_COLORS = dict(zip(FITTED_DEPTHS, PALETTE.ai_sizes))  # AI runs, 19.9M (light) to 268M (dark)
HUMAN_SIZE_COLORS = dict(zip(FITTED_DEPTHS, PALETTE.human_sizes))
HELD_OUT_COLORS = dict(zip((20, 26), PALETTE.held_out))
BUDGET_CMAP = LinearSegmentedColormap.from_list("budget", list(PALETTE.budget_ramp))
BUDGET_NORM = LogNorm(vmin=5.0, vmax=100.0)
DIVERGING_CMAP = LinearSegmentedColormap.from_list("diverging", list(PALETTE.diverging))
RATIO_CMAP = LinearSegmentedColormap.from_list("ratio", list(PALETTE.ratio_ramp))
# Line colour and style of each evaluation set in the per-target panels: the human-text sets in teals, FW26 in ochre, Cosmo pink.
TARGET_STYLE: dict[str, tuple[str, str | tuple[float, tuple[float, float]]]] = {
    "c4": (PALETTE.human, "-"),
    "fw22": (PALETTE.human_light, "--"),
    "paloma": (PALETTE.human_sizes[3], (0, (1, 1.3))),
    "fw26": (PALETTE.ochre, "-"),
    "cosmopedia": (PALETTE.ai_line, "-"),
}

Record = TypeVar("Record", bound=BaseModel)


def label(target: str) -> str:
    """The paper's short name of an evaluation set (C4, FW22, FW26-H, Paloma, FW26, FW26-AI, Cosmo)."""

    return TARGET_LABELS[target]


@cache
def _fits() -> dict[tuple[str, str, str], FitRecord]:
    return {(r.law, r.target, r.fold): r for r in read_fits(LAWS / "fits.json")}


def fitted(target: str, law: str = PAPER_LAW, fold: str = "all") -> FittedLaw:
    """A law's fit on one target: every fitted run ("all") or all but one fitted size ("without_<size>")."""

    return FittedLaw(resolve_law(law), _fits()[(law, target, fold)].parameters)


def read_records(name: str, record: type[Record], **where: str) -> list[Record]:
    """The rows of results/laws/<name> whose columns equal `where`, as `record`s (empty cells are None)."""

    with (LAWS / name).open(encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if all(r[k] == v for k, v in where.items())]
    return [record.model_validate({k: (v if v != "" else None) for k, v in r.items()}) for r in rows]


def reference_n() -> float:
    """N of the 268M models."""

    return float(next(m.n_params for m in models().values() if m.depth == REFERENCE_DEPTH))


def ai_change_pct(model: FittedLaw, control: Model, ratios: np.ndarray) -> np.ndarray:
    """Predicted change in loss (%) against `control` when AI tokens at ratio r are added to its human tokens."""

    return change_at(model, control.n_params, control.human_tokens, ratios)


def change_at(model: FittedLaw, n_params: float, human_tokens: float, ratios: np.ndarray) -> np.ndarray:
    """Predicted change in loss (%) when AI tokens at ratio r are added to `human_tokens` at N = `n_params`."""

    return 100.0 * (model.loss(n_params, human_tokens, ratios * human_tokens) / model.loss1(n_params, human_tokens, 0.0) - 1.0)


def human_change_pct(model: FittedLaw, control: Model, added: np.ndarray) -> np.ndarray:
    """Predicted change in loss (%) against `control` when fresh human tokens, `added` per human token, are added. For the
    AI ratio this is Chinchilla's prediction: every AI token counted as a human token."""

    n, human = control.n_params, control.human_tokens
    return 100.0 * (model.loss(n, human * (1.0 + added), 0.0) / model.loss1(n, human, 0.0) - 1.0)


@dataclass(frozen=True)
class OursTerms:
    """The pieces of our law at given (N, D_H, r): L = E + A n^-alpha + B [D_H (1 + credit)]^-beta (1 + harm)."""

    credit: np.ndarray  # human-token credit of the AI tokens, per human token
    harm: np.ndarray  # relative inflation of the data term
    window: np.ndarray  # R*, the AI ratio at which the credit saturates
    beta: float

    @property
    def effective_data(self) -> np.ndarray:
        """Effective human tokens per human token: the human-only data that gives the same data term."""

        return (1.0 + self.credit) * (1.0 + self.harm) ** (-1.0 / self.beta)


def ours_terms(model: FittedLaw, n_params: np.ndarray | float, human_tokens: np.ndarray | float, ratios: np.ndarray | float) -> OursTerms:
    """Our law's terms for N parameters trained on D_H human tokens and r D_H AI tokens (broadcast arrays)."""

    law = model.law
    assert isinstance(law, Ours)
    p = law.unpack(law.vector(model.values))
    x = Coordinates.from_tokens(n_params, human_tokens, np.asarray(ratios) * human_tokens)
    return OursTerms(np.ravel(law.credit(p, x)), np.ravel(law.harm(p, x)), np.ravel(law.window(p, x)), float(model.values["beta"]))


def backbone(model: FittedLaw, n_params: np.ndarray | float) -> np.ndarray:
    """E + A n^-alpha: the loss with unlimited data, which the data term adds to."""

    c = model.law.coefficients(model.values)
    return c["E"] + c["A"] * Coordinates.from_tokens(n_params, 1.0, 0.0).n ** (-c["alpha"])


def design_budget(human_tpp: float) -> float:
    """A human budget rounded to the nearest 5 tokens per parameter (half up), the value the figures label it with."""

    return 5.0 * math.floor(round(human_tpp, 1) / 5.0 + 0.5)


def tpp_axis(ax: plt.Axes, lo: float = 5.0, hi: float = 100.0, ticks: tuple[float, ...] = (5, 10, 20, 50, 100)) -> None:
    """A log axis of the human budget in tokens per parameter."""

    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def unit_lines(ax: plt.Axes) -> None:
    """Reference lines of a token-value axis: one human token and worthless."""

    ax.axhline(1.0, color=PALETTE.rule, lw=0.7, zorder=1)
    ax.axhline(0.0, color=PALETTE.rule, lw=0.7, zorder=1)


def budget_colorbar(fig: plt.Figure, ax: plt.Axes | list[plt.Axes], **kwargs: float) -> Colorbar:
    """A horizontal colour bar of the human budget, 5 to 100 tokens per parameter, below `ax`."""

    bar = fig.colorbar(ScalarMappable(norm=BUDGET_NORM, cmap=BUDGET_CMAP), ax=ax, location="bottom", **kwargs)
    bar.set_label(TPP_H_LABEL, fontsize=7)
    bar.set_ticks([5, 10, 20, 40, 100])
    bar.set_ticklabels(["5", "10", "20", "40", "100"])
    bar.ax.tick_params(labelsize=6.5)
    bar.ax.minorticks_off()
    bar.outline.set_visible(False)
    return bar
