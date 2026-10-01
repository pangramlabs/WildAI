"""Change in loss against the AI ratio r, measured against each run's human-only control and under our law.

- law_dose_response (Figure 3) and law_dose_response_cosmopedia (appendix): every fitted size at 5, 20 and 40 TPP_h,
  AI text against the same number of fresh human tokens, on C4 and on Cosmopedia.
- law_dose_ladder_all (appendix): every matched-budget group with our law's curves, one panel per size and budget.
- law_harm_per_doubling (appendix): from r = 1 to 64 the harm grows by the same amount per doubling of AI text.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from paper.figures.law_common import ADD_AI_LABEL, ADD_HUMAN_LABEL, HUMAN_SIZE_COLORS, RATIOS, SIZE_COLORS, ai_change_pct, fitted, human_change_pct, label
from paper.groups import FITTED_DEPTHS, MATCHED_BUDGETS, Group, added_fraction, change_pct, fresh_human_groups, matched_group
from paper.results import Model
from paper.style import AI, HUMAN, INK, PALETTE, R_LABEL, SIZE_LABEL, TEXT_WIDTH, budget_label, format_ratio, ratio_axis, save_figure, zero_line
from wildai.laws.forms import FittedLaw

RESPONSE_BUDGETS = (4.72, 18.88, 37.76)
DOUBLING_BUDGET = 18.88
DOUBLING_DEPTHS = (4, 6, 9)
ACCELERATING_POWER = 1.8  # the illustrative power-law penalty the doubling figure contrasts with


def _changes(pairs: list[tuple[Model, Model]], target: str) -> tuple[list[float], list[float]]:
    """(tokens added per human token, change in loss %) of each (run, control) pair, by amount added."""

    points = sorted((added_fraction(m, c), change_pct(m, c, target)) for m, c in pairs)
    return [x for x, _ in points], [y for _, y in points]


def _response(out: Path, target: str, stem: str) -> Path:
    fig, axes = plt.subplots(1, len(RESPONSE_BUDGETS), figsize=(TEXT_WIDTH, 1.75), sharey=True)
    for ax, budget in zip(axes, RESPONSE_BUDGETS):
        zero_line(ax)
        for depth in FITTED_DEPTHS:
            group = matched_group(depth, budget)
            if group is None:
                continue
            fresh = [(m, g.control) for g in fresh_human_groups(depth, budget) for m in g.human]
            for pairs, color, hollow in (([(m, group.control) for m in group.ai], SIZE_COLORS[depth], False), (fresh, HUMAN_SIZE_COLORS[depth], True)):
                if not pairs:
                    continue
                xs, ys = _changes(pairs, target)
                ax.plot(xs, ys, color=color, lw=0.8, alpha=0.35 if hollow else 0.55, ls=(0, (1, 1.5)) if hollow else "-", zorder=2)
                if hollow:
                    ax.scatter(xs, ys, s=13, facecolor="white", edgecolor=color, lw=0.9, zorder=4)
                else:
                    ax.scatter(xs, ys, s=13, color=color, edgecolor="white", lw=0.4, zorder=4)
        ratio_axis(ax)
        ax.set_title(budget_label(budget))
    axes[1].set_xlabel(R_LABEL)
    axes[0].set_ylabel(f"change in\n{label(target)} loss (%)")
    handles = [Line2D([], [], color=SIZE_COLORS[d], marker="o", ms=3.6, lw=0, label=SIZE_LABEL[d]) for d in FITTED_DEPTHS]
    handles += [Line2D([], [], color=AI, marker="o", ms=3.6, lw=0, mfc=AI, label=ADD_AI_LABEL),
                Line2D([], [], color=HUMAN, marker="o", ms=3.6, lw=0, mfc="white", mew=0.9, label=ADD_HUMAN_LABEL)]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, 0.0), columnspacing=1.1, handletextpad=0.4, fontsize=6.4)
    fig.tight_layout(w_pad=0.8, rect=(0, 0.13, 1, 1))
    return save_figure(fig, out, stem)


def dose_response(out: Path) -> Path:
    return _response(out, "c4", "law_dose_response")


def dose_response_cosmopedia(out: Path) -> Path:
    return _response(out, "cosmopedia", "law_dose_response_cosmopedia")


def _span(amounts: list[float]) -> np.ndarray:
    """Law curves run over the amounts trained, or over the whole axis when there are none."""

    return np.logspace(np.log10(min(amounts)), np.log10(max(amounts)), 200) if amounts else RATIOS


def _ladder_panel(ax: plt.Axes, group: Group, model: FittedLaw, target: str) -> None:
    """One group's AI and fresh-human additions with our law's curves over the amounts trained."""

    control = group.control
    ratios, added = _span([m.ratio for m in group.ai]), _span([added_fraction(m, control) for m in group.human])
    ax.plot(ratios, ai_change_pct(model, control, ratios), color=PALETTE.ai_line, lw=1.2, zorder=3)
    ax.plot(added, human_change_pct(model, control, added), color=HUMAN, lw=0.8, ls=(0, (1, 1.5)), alpha=0.8, zorder=2)
    ax.scatter([added_fraction(m, control) for m in group.human], [change_pct(m, control, target) for m in group.human],
               s=9, facecolor="white", edgecolor=HUMAN, lw=0.7, zorder=4)
    ax.scatter([m.ratio for m in group.ai], [change_pct(m, control, target) for m in group.ai], s=9, color=AI, edgecolor="white", lw=0.4, zorder=5)


def dose_ladder_all(out: Path) -> Path:
    target = "c4"
    model = fitted(target)
    fig, axes = plt.subplots(len(FITTED_DEPTHS), len(MATCHED_BUDGETS), figsize=(7.2, 1.2 * len(FITTED_DEPTHS)), sharex=True, sharey="row")
    for row, depth in enumerate(FITTED_DEPTHS):
        for col, budget in enumerate(MATCHED_BUDGETS):
            ax = axes[row, col]
            group = matched_group(depth, budget)
            if group is None:
                ax.set_axis_off()
                continue
            zero_line(ax)
            _ladder_panel(ax, group, model, target)
            ratio_axis(ax, sparse=True)
            ax.set_title(f"{SIZE_LABEL[depth]} · {budget_label(group.control.human_tpp)}", fontsize=6.8, pad=3)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel(f"change in {label(target)} loss (%)", fontsize=6.5)
            if row == len(FITTED_DEPTHS) - 1 or all(matched_group(d, budget) is None for d in FITTED_DEPTHS[row + 1:]):
                ax.set_xlabel(R_LABEL, fontsize=6.5)
                ax.tick_params(labelbottom=True)
    handles = [Line2D([], [], color=PALETTE.ai_line, mfc=AI, mec=AI, marker="o", ms=3.2, lw=1.2, label="add AI text (line: Ours)"),
               Line2D([], [], mfc="white", mec=HUMAN, color=HUMAN, marker="o", ms=3.2, lw=0.9, ls=(0, (1, 1.5)), label="add fresh human text (line: Ours)")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.01), handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(h_pad=0.6, w_pad=0.5, rect=(0, 0.02, 1, 1))
    return save_figure(fig, out, "law_dose_ladder_all")


def harm_per_doubling(out: Path) -> Path:
    """Our law against an accelerating penalty r^1.8 that agrees with it at r = 1 and 2, at 20 TPP_h."""

    target = "c4"
    model = fitted(target)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.6, 2.6))
    zero_line(ax)
    ratios = np.logspace(0, np.log10(70), 200)
    groups = [g for d in DOUBLING_DEPTHS if (g := matched_group(d, DOUBLING_BUDGET)) is not None]
    for group in groups:
        color = SIZE_COLORS[group.depth]
        law = ai_change_pct(model, group.control, ratios)
        at_1, at_2 = ai_change_pct(model, group.control, np.array([1.0, 2.0]))
        accelerating = at_1 + (at_2 - at_1) * (ratios**ACCELERATING_POWER - 1.0) / (2.0**ACCELERATING_POWER - 1.0)
        ax.plot(ratios, accelerating, color=color, lw=0.9, ls=(0, (3, 2)), alpha=0.75, zorder=2)
        ax.plot(ratios, law, color=color, lw=1.3, zorder=3)
        points = [m for m in group.ai if m.ratio >= 0.9]
        ax.scatter([m.ratio for m in points], [change_pct(m, group.control, target) for m in points], s=15, color=color, edgecolor="white", lw=0.4, zorder=4)
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.9, 70)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16, 32, 64]))
    ax.xaxis.set_major_formatter(FuncFormatter(format_ratio))
    ax.xaxis.set_minor_formatter(NullFormatter())
    observed_top = max(change_pct(m, g.control, target) for g in groups for m in g.ai)
    ax.set_ylim(-1.0, 2.2 * observed_top)
    ax.set_xlabel(R_LABEL)
    ax.set_ylabel(f"change in {label(target)} loss (%)")
    ax.set_title(budget_label(DOUBLING_BUDGET))
    sizes = [Line2D([], [], color=SIZE_COLORS[g.depth], marker="o", ms=3.4, lw=0, label=SIZE_LABEL[g.depth]) for g in groups]
    laws = [Line2D([], [], color=INK, lw=1.3, label="Ours"),
            Line2D([], [], color=INK, lw=0.9, ls=(0, (3, 2)), label=f"accelerating harm, $r^{{{ACCELERATING_POWER:g}}}$")]
    handles = [sizes[0], laws[0], sizes[1], laws[1], sizes[2]]  # the legend fills column by column: sizes on top, laws below
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), columnspacing=1.0, handletextpad=0.5, labelspacing=0.3)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    return save_figure(fig, out, "law_harm_per_doubling")
