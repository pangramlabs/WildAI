"""Our law against the trained models.

- law_validation (Figure 4): the 268M runs at three budgets with the law through them; the held-out 477M and 973M runs
  against our law and Chinchilla, both fitted on 19.9M to 268M; predicted against observed change for every AI run.
- law_token_value_calibration (appendix): the law's value of an AI token against the value measured from paired AI and
  fresh-human additions to the same control, at 5 and 20 TPP_h.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

from paper.figures.law_common import (
    BUDGET_CMAP,
    BUDGET_NORM,
    HELD_OUT_COLORS,
    SIZE_COLORS,
    VALUE_LABEL,
    ai_change_pct,
    design_budget,
    fitted,
    human_change_pct,
    label,
    read_records,
    unit_lines,
)
from paper.groups import FITTED_DEPTHS, added_fraction, change_pct, fresh_human_groups, matched_group
from paper.style import HUMAN, INK, MUTED, PALETTE, R_LABEL, RULE, SIZE_LABEL, TEXT_WIDTH, budget_label, ratio_axis, save_figure, zero_line
from wildai.laws.catalog import PAPER_LAW
from wildai.laws.ceg import ai_token_value
from wildai.laws.records import PredictionRecord

TARGET = "c4"
FIT_BUDGETS = (4.72, 18.88, 37.76)
HELD_OUT_BUDGET = 18.88
HELD_OUT_MARKERS = {20: "o", 26: "D"}
CALIBRATION_BUDGETS = (4.72, 18.88)
PAIR_TOLERANCE = 0.15  # an AI and a fresh-human addition pair up when their amounts differ by less than this in log
MIN_HUMAN_GAIN_PCT = 0.1  # below this loss reduction from fresh human text, the measured value is noise over noise


def _fitted_runs_panel(ax: plt.Axes, depth: int = 16) -> None:
    """Trained models of one size at several budgets with our law over the ratios trained."""

    model = fitted(TARGET)
    zero_line(ax)
    handles = []
    for budget in FIT_BUDGETS:
        group = matched_group(depth, budget)
        if group is None:
            continue
        color = BUDGET_CMAP(BUDGET_NORM(design_budget(budget)))
        ratios = [m.ratio for m in group.ai]
        grid = np.logspace(np.log10(min(ratios)), np.log10(max(ratios)), 200)
        ax.plot(grid, ai_change_pct(model, group.control, grid), color=color, lw=1.3, zorder=3)
        ax.scatter(ratios, [change_pct(m, group.control, TARGET) for m in group.ai], s=15, color=color, edgecolor="white", lw=0.4, zorder=4)
        handles.append(Line2D([], [], color=color, marker="o", ms=3.4, lw=1.3, label=budget_label(budget)))
    ratio_axis(ax, 1.5e-3, 10)
    ax.legend(handles=handles, loc="upper left", fontsize=6.0, handlelength=1.6, handletextpad=0.4, borderaxespad=0.3, labelspacing=0.3)
    ax.set_title(f"Fits the Trained Runs, {SIZE_LABEL[depth]}", fontsize=8.5)
    ax.set_ylabel(f"change in {label(TARGET)} loss (%)")
    ax.set_xlabel(R_LABEL)


def _held_out_panel(ax: plt.Axes) -> None:
    """The held-out 477M and 973M runs at 20 TPP_h with our law and Chinchilla (every AI token counted as a human token)."""

    model = fitted(TARGET)
    zero_line(ax)
    ratios = np.logspace(np.log10(2e-3), np.log10(1.3), 160)
    for depth, marker in HELD_OUT_MARKERS.items():
        group = matched_group(depth, HELD_OUT_BUDGET)
        if group is None:
            continue
        ax.plot(ratios, human_change_pct(model, group.control, ratios), color=INK, lw=1.0, ls=(0, (1, 1.4)), zorder=2)
        ax.plot(ratios, ai_change_pct(model, group.control, ratios), color=INK, lw=1.3, zorder=3)
        ax.scatter([m.ratio for m in group.ai], [change_pct(m, group.control, TARGET) for m in group.ai], s=17 if marker == "o" else 14,
                   marker=marker, color=HELD_OUT_COLORS[depth], edgecolor="white", lw=0.4, zorder=4)
    ratio_axis(ax, 2e-3, 1.3)
    ax.xaxis.set_major_locator(FixedLocator([1e-2, 1e-1, 1.0]))
    ax.set_ylim(-1.6, 1.0)
    ax.set_xlabel(R_LABEL)
    ax.set_ylabel(f"change in {label(TARGET)} loss (%)")
    ax.set_title("Predicts Held-Out Sizes", fontsize=8.5)
    handles = [Line2D([], [], color=HELD_OUT_COLORS[20], marker="o", ms=3.6, lw=0, label=SIZE_LABEL[20]),
               Line2D([], [], color=HELD_OUT_COLORS[26], marker="D", ms=3.2, lw=0, label=SIZE_LABEL[26]),
               Line2D([], [], color=INK, lw=1.3, label="Ours"),
               Line2D([], [], color=INK, lw=1.0, ls=(0, (1, 1.4)), label="Chinchilla")]
    ax.legend(handles=handles, loc="lower left", ncol=2, fontsize=6.0, handlelength=1.6, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.2,
              labelspacing=0.3)
    ax.text(0.04, 0.96, budget_label(HELD_OUT_BUDGET), transform=ax.transAxes, fontsize=6.4, color=MUTED, ha="left", va="top")


def _predicted_observed_panel(ax: plt.Axes) -> None:
    """Predicted against observed change for every AI run under the fit on 19.9M to 268M."""

    rows = [r for r in read_records("predictions.csv", PredictionRecord, law=PAPER_LAW, target=TARGET, fold="all") if r.arm == "ai"]
    for split, held_out in (("fit", False), ("held_out", True)):
        chosen = [r for r in rows if r.split == split]
        ax.scatter([100 * math.expm1(r.observed_change) for r in chosen], [100 * math.expm1(r.predicted_change) for r in chosen],
                   s=14 if held_out else 9, marker="D" if held_out else "o", color=PALETTE.held_out[1] if held_out else HUMAN, edgecolor="white",
                   lw=0.3, alpha=0.95 if held_out else 0.6, zorder=4 if held_out else 3,
                   label="477M, 973M (held out)" if held_out else "19.9M–268M (fitted)")
    lo, hi = -4.0, 8.0
    ax.plot([lo, hi], [lo, hi], color=RULE, lw=0.8, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("observed change (%)")
    ax.set_ylabel(f"predicted change in {label(TARGET)} loss (%)")
    ax.set_title("Tracks Every Run", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=6.0, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.3)


def validation(out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 1.9), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.06)
    _fitted_runs_panel(axes[0])
    _held_out_panel(axes[1])
    _predicted_observed_panel(axes[2])
    for ax in axes:
        ax.tick_params(labelsize=6.5)
    return save_figure(fig, out, "law_validation")


def paired_token_values(budget: float) -> list[tuple[int, float, float]]:
    """(depth, r, value) at one budget: for every AI addition with a fresh-human addition of the same size and nearly the
    same amount, the loss reduction per AI token over the loss reduction per fresh human token."""

    out = []
    for depth in FITTED_DEPTHS:
        group = matched_group(depth, budget)
        fresh = [(added_fraction(m, g.control), m, g.control) for g in fresh_human_groups(depth, budget) for m in g.human]
        if group is None or not fresh:
            continue
        for run in group.ai:
            amount, human, control = min(fresh, key=lambda item: abs(math.log(item[0] / run.ratio)))
            human_gain = -change_pct(human, control, TARGET)
            if abs(math.log(amount / run.ratio)) > PAIR_TOLERANCE or human_gain < MIN_HUMAN_GAIN_PCT:
                continue
            out.append((depth, run.ratio, -change_pct(run, group.control, TARGET) / run.ratio / (human_gain / amount)))
    return out


def token_value_calibration(out: Path) -> Path:
    model = fitted(TARGET)
    ratios = np.logspace(-2, 1.3, 140)
    fig, axes = plt.subplots(1, 2, figsize=(0.8 * TEXT_WIDTH, 2.4), sharey=True, layout="constrained")
    for ax, budget in zip(axes, CALIBRATION_BUDGETS):
        unit_lines(ax)
        measured = paired_token_values(budget)
        for depth in sorted({d for d, _r, _v in measured}):
            control = matched_group(depth, budget).control
            ax.plot(ratios, ai_token_value(model, control.n_params, control.human_tokens, ratios), color=SIZE_COLORS[depth], lw=1.0, zorder=2)
            ax.scatter([r for d, r, _v in measured if d == depth], [v for d, _r, v in measured if d == depth], s=16, color=SIZE_COLORS[depth],
                       edgecolor=INK, lw=0.5, zorder=4)
        ratio_axis(ax, 0.01, 20.0)
        ax.set_title(budget_label(matched_group(16, budget).control.human_tpp))
        ax.set_xlabel(R_LABEL)
        ax.tick_params(labelsize=6.5)
    axes[0].set_ylim(-1.2, 2.6)
    axes[0].set_ylabel(VALUE_LABEL)
    axes[0].text(18, 1.0, "= one human token", fontsize=6.0, color=MUTED, ha="right", va="bottom")
    handles = [Line2D([], [], color=SIZE_COLORS[d], marker="o", ms=3.6, mec=INK, mew=0.5, lw=1.0, label=SIZE_LABEL[d]) for d in FITTED_DEPTHS]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=6.2, columnspacing=1.0, handletextpad=0.4,
               title="points: paired runs; curves: Ours", title_fontsize=6.2)
    return save_figure(fig, out, "law_token_value_calibration")
