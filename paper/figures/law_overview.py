"""Figure 1 (law_overview): how much of the web is AI-generated, what an added AI token is worth under our law at 268M from
a data-starved budget (5 TPP_h) to far past Chinchilla-optimal (100 TPP_h), and what leaving the AI text in a web crawl
costs in compute as the web's AI share grows (measured each month, then forecast)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, PercentFormatter

from paper.figures.law_common import BUDGET_CMAP, BUDGET_NORM, REFERENCE_DEPTH, VALUE_LABEL, budget_colorbar, fitted, reference_n, unit_lines
from paper.figures.web_ai_share import forecast
from paper.results import monthly_ai_share
from paper.style import AI, INK, MIXED, MUTED, R_LABEL, RULE, SIZE_LABEL, TEXT_WIDTH, ratio_axis, save_figure
from wildai.laws.ceg import ai_token_value, cost_of_not_filtering
from wildai.laws.forms import FittedLaw

VALUE_BUDGETS = (5.0, 20.0, 40.0, 60.0, 80.0, 100.0)
COST_TPP = 20.0
MARKED_JUNES = ("2024-06", "2025-06", "2026-06")
MARKED_FORECASTS = ("2027-12", "2028-12")
COST_FROM = "2023-01"
COST_TOP = 5.2  # room for the legend above the December 2028 label
LABEL_BOX = {"boxstyle": "square,pad=0.1", "fc": "white", "ec": "none"}


def _date(month: str) -> datetime:
    return datetime(int(month[:4]), int(month[5:7]), 15)


def share_panel(ax: plt.Axes) -> None:
    """The monthly share of web tokens in AI- and Mixed-labeled documents, with the June values and the latest month marked."""

    series = monthly_ai_share()
    months = [m.month for m in series]
    dates = [_date(m) for m in months]
    ai = [100 * m.ai_share for m in series]
    ax.fill_between(dates, 0, ai, color=AI, alpha=0.9, lw=0)
    ax.fill_between(dates, ai, [100 * m.ai_or_mixed_share for m in series], color=MIXED, alpha=0.9, lw=0)
    for month in MARKED_JUNES:
        k = months.index(month)
        ax.scatter([dates[k]], [ai[k]], s=12, color=INK, zorder=5)
        ax.annotate(f"{ai[k]:.0f}%", xy=(dates[k], ai[k]), xytext=(-3, 5), textcoords="offset points", fontsize=6.3, color=INK, ha="right", va="bottom")
    ax.scatter([dates[-1]], [ai[-1]], s=12, color=INK, zorder=5)
    ax.annotate(f"{ai[-1]:.0f}%", xy=(dates[-1], ai[-1]), xytext=(4, 0), textcoords="offset points", fontsize=6.3, color=INK, ha="left", va="center")
    ax.legend(handles=[Patch(color=MIXED, label="Mixed"), Patch(color=AI, label="AI")], loc="upper left", fontsize=6.3,
              handlelength=1.1, handletextpad=0.4, borderaxespad=0.3, labelspacing=0.3)
    ax.set_ylim(0, 40)
    ax.set_xlim(datetime(2021, 1, 1), datetime(2027, 1, 15))  # room for the latest month's label
    ax.xaxis.set_major_locator(mdates.YearLocator(2, month=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_title("AI Share of Web Text")
    ax.set_ylabel("share of web tokens")
    ax.set_xlabel("crawl date")


def token_value_panel(ax: plt.Axes, model: FittedLaw, ratios: np.ndarray) -> None:
    """Value of an added AI token in human tokens against r at 268M, one curve per human budget (5 to 100 TPP_h)."""

    n = reference_n()
    for tpp in VALUE_BUDGETS:
        ax.plot(ratios, ai_token_value(model, n, tpp * n, ratios), color=BUDGET_CMAP(BUDGET_NORM(tpp)), lw=1.3)
    unit_lines(ax)
    ax.text(90, 1.0, "= one human token", fontsize=6.3, color=MUTED, ha="right", va="bottom", bbox=LABEL_BOX)
    ax.text(0.012, 0.0, "worthless", fontsize=6.3, color=MUTED, ha="left", va="bottom")
    ax.set_ylim(-3.6, 2.4)
    ratio_axis(ax, 0.01, 100.0)


def cost_panel(ax: plt.Axes, model: FittedLaw) -> None:
    """Compute an unfiltered crawl needs to match training on its human subset (268M, 20 TPP_h), by crawl date: each
    month's measured AI share, then the forecast with its 80 % interval."""

    n = reference_n()

    def cost(share: float) -> float:
        return cost_of_not_filtering(model, n, COST_TPP, share)

    observed = [(m.month, m.ai_share) for m in monthly_ai_share() if m.month >= COST_FROM]
    ahead = sorted(forecast().values(), key=lambda f: f.month)
    ax.axhline(1.0, color=RULE, lw=0.7, zorder=1)
    ax.plot([_date(m) for m, _ in observed], [cost(s) for _, s in observed], color=AI, lw=1.3, marker="o", ms=1.8, zorder=4)
    last_month, last_share = observed[-1]
    dates = [_date(last_month)] + [_date(f.month) for f in ahead]
    ax.fill_between(dates, [cost(last_share)] + [cost(f.lo80) for f in ahead], [cost(last_share)] + [min(cost(f.hi80), COST_TOP) for f in ahead],
                    color=AI, alpha=0.16, lw=0, zorder=2)
    ax.plot(dates, [cost(last_share)] + [cost(f.share) for f in ahead], color=AI, lw=1.2, ls=(0, (3, 2)), zorder=3)
    marks = [(last_month, last_share, (-6, 9))] + [(f.month, f.share, (-5, 8)) for f in ahead if f.month in MARKED_FORECASTS]
    for month, share, offset in marks:
        ax.scatter([_date(month)], [cost(share)], s=13, color=INK, zorder=5)
        ax.annotate(f"{100 * share:.0f}% AI\n{cost(share):.1f}×", xy=(_date(month), cost(share)), xytext=offset, textcoords="offset points",
                    fontsize=6.0, color=INK, ha="right", va="bottom", linespacing=0.95)
    ax.set_ylim(0.9, COST_TOP)
    ax.set_xlim(_date(COST_FROM), _date("2029-02"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 3, 4, 5]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}×"))
    ax.text(0.97, 0.05, f"{SIZE_LABEL[REFERENCE_DEPTH]}, {COST_TPP:g} $TPP_h$", transform=ax.transAxes, fontsize=6.0, color=MUTED, ha="right", va="bottom")
    ax.set_xlabel("crawl date")
    ax.set_ylabel("compute to match\ntraining without AI text")
    ax.set_title("Cost of Not Filtering")
    handles = [Line2D([], [], color=AI, lw=1.3, marker="o", ms=1.8, label="measured AI share"),
               Line2D([], [], color=AI, lw=1.2, ls=(0, (3, 2)), label="forecast (80% band)")]
    ax.legend(handles=handles, loc="upper left", fontsize=6.0, handlelength=2.0, handletextpad=0.5, borderaxespad=0.3, labelspacing=0.3)


def overview(out: Path) -> Path:
    model = fitted("c4")
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.45), layout="constrained")
    share_panel(axes[0])
    token_value_panel(axes[1], model, np.logspace(-2, 2, 160))
    axes[1].set_title("Value of an AI Token")
    axes[1].set_ylabel(VALUE_LABEL)
    axes[1].set_xlabel(R_LABEL)
    cost_panel(axes[2], model)
    budget_colorbar(fig, axes[1], shrink=0.9, aspect=25, pad=0.02)
    return save_figure(fig, out, "law_overview")
