"""Filtering, repeating and the compute-equivalent gain of a mix (Section 4 and appendix).

- law_ceg_filtering (Figure 5): the change in loss from removing the AI documents of the 2026 web mix, predicted by our
  law over model size and human budget and measured on the filtering pairs; and repeating the human corpus against adding
  the same number of AI tokens at 20 TPP_h.
- law_repetition_targets (appendix): repeating human text against adding AI or fresh human text, on six evaluation sets.
- law_ceg_ladder (appendix): compute-equivalent gain over the 2026 mix of the 268M models at 20 TPP_h as their AI share
  varies, measured and under our law, on C4 and Cosmopedia.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import QuadMesh
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from paper.figures.law_common import DIVERGING_CMAP, HUMAN_SIZE_COLORS, LAWS, SIZE_COLORS, TARGET_STYLE, fitted, label, read_records
from paper.groups import added_fraction, change_pct, fresh_human_groups
from paper.results import Model, controls, models
from paper.style import AI, HUMAN, INK, PALETTE, RULE, SIZE_LABEL, TEXT_WIDTH, TPP_H_LABEL, budget_label, format_params, format_ratio, save_figure, zero_line
from wildai.laws.analysis.ceg_ladder import LADDER_CONTROL, LADDER_MAX_RATIO, CegCurvePoint, CegRecord, CegSummary
from wildai.laws.analysis.filtering import FilteringRecord
from wildai.laws.catalog import PAPER_LAW
from wildai.laws.ceg import WEB_2026_AI_SHARE, ReferenceMix

TARGET = "c4"
FILTERING_LIMIT = 1.2  # colour scale of the filtering map, percent change in loss
LABELED_DEPTH = 26  # the filtering pairs whose measured change the map writes out
REPETITION_BUDGET = 18.88
REPETITION_TARGETS = ("c4", "fw22", "fw26_human", "fw26", "fw26_ai", "cosmopedia")
LADDER_TARGETS = ("c4", "cosmopedia")


def _filtering_map(ax: plt.Axes) -> QuadMesh:
    """Our law's change in loss from removing the AI documents of the 2026 mix (adding nothing) over size and human budget,
    with every measured filtering pair on the same colour scale."""

    model = fitted(TARGET)
    ratio = WEB_2026_AI_SHARE / (1.0 - WEB_2026_AI_SHARE)
    sizes, budgets = np.meshgrid(np.logspace(np.log10(1.5e7), np.log10(1.2e9), 90), np.logspace(np.log10(2), np.log10(100), 90))
    human = (sizes * budgets).ravel()
    change = 100.0 * (model.loss(sizes.ravel(), human, 0.0) / model.loss(sizes.ravel(), human, ratio * human) - 1.0)
    change = change.reshape(sizes.shape)
    mesh = ax.pcolormesh(sizes, budgets, change, cmap=DIVERGING_CMAP, vmin=-FILTERING_LIMIT, vmax=FILTERING_LIMIT, shading="auto", rasterized=True)
    ax.contour(sizes, budgets, change, levels=[0.0], colors=INK, linewidths=0.7)
    pairs = read_records("filtering.csv", FilteringRecord, law=PAPER_LAW, target=TARGET)
    measured = [100.0 * math.expm1(p.observed_change) for p in pairs]
    ax.scatter([p.n_params for p in pairs], [p.human_tpp for p in pairs], c=measured, cmap=DIVERGING_CMAP, vmin=-FILTERING_LIMIT, vmax=FILTERING_LIMIT,
               s=16, edgecolor=INK, lw=0.5, zorder=4)
    for p, value in zip(pairs, measured):
        if p.depth == LABELED_DEPTH:
            ax.text(p.n_params * 0.84, p.human_tpp, f"{value:+.2f}%".replace("-", "−"), fontsize=6.0, color=INK, ha="right", va="center")
    ax.text(2.4e8, 75, "filtering helps", fontsize=6.4, color=PALETTE.held_out[1], ha="center", va="center", fontweight="bold")
    ax.text(2.4e8, 3.0, "filtering hurts", fontsize=6.4, color="white", ha="center", va="center", fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator([2e7, 1e8, 1e9]))
    ax.xaxis.set_major_formatter(FuncFormatter(format_params))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50, 100]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("model parameters")
    ax.set_ylabel(TPP_H_LABEL)
    ax.set_title("When Filtering Wins", fontsize=8.5)
    return mesh


def _tokens_added(model: Model) -> float:
    """Training tokens beyond one pass over the human tokens, per human token: the AI ratio, or the repeated passes."""

    return (model.total_tokens - model.human_tokens) / model.human_tokens


def _repetition_panel(ax: plt.Axes, target: str, *, labels: bool, fresh: bool) -> None:
    """At 20 TPP_h, the change in loss from spending extra tokens by repeating the human text, by adding the same number
    of AI tokens, or (optionally) by adding fresh human text."""

    repeats = [m for m in models().values() if m.arm == "repeat"]
    matched = {(m.group, m.added_ratio): m for m in models().values() if m.arm == "ai"}
    zero_line(ax)
    ends: dict[str, tuple[float, float]] = {}
    for depth in sorted({m.depth for m in repeats}):
        rows = sorted((m for m in repeats if m.depth == depth), key=lambda m: m.added_ratio or 0.0)
        control = controls()[rows[0].group]
        ai = [matched[(m.group, m.added_ratio)] for m in rows]
        ax.plot([m.ratio for m in ai], [change_pct(m, control, target) for m in ai], color=SIZE_COLORS[depth], lw=1.2, marker="o", ms=3.2, mec="white",
                mew=0.4, zorder=4)
        ax.plot([_tokens_added(m) for m in rows], [change_pct(m, control, target) for m in rows], color=HUMAN_SIZE_COLORS[depth], lw=1.2, marker="o",
                ms=3.2, mec="white", mew=0.4, zorder=4)
        if fresh:
            added = sorted((added_fraction(m, g.control), change_pct(m, g.control, target)) for g in fresh_human_groups(depth, REPETITION_BUDGET)
                           for m in g.human if added_fraction(m, g.control) >= 0.9)
            ax.plot([x for x, _ in added], [y for _, y in added], color=HUMAN_SIZE_COLORS[depth], lw=0.9, ls=(0, (1, 1.5)), marker="o", ms=3.2,
                    mfc="white", mew=0.9, zorder=3)
        ends["add AI text"] = (ai[-1].ratio, change_pct(ai[-1], control, target))
        ends["repeat human text"] = (_tokens_added(rows[0]), change_pct(rows[0], control, target))  # labeled at its start, where there is room
    ax.set_xscale("log")
    ax.set_xlim(0.8, 11)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8]))
    ax.xaxis.set_major_formatter(FuncFormatter(format_ratio))
    ax.xaxis.set_minor_formatter(NullFormatter())
    if labels:  # direct labels: beside the end of the AI line, above the start of the repetition line
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.10 * (hi - lo))
        placement = {"add AI text": (AI, (-5, 3), "right"), "repeat human text": (HUMAN, (-2, 7), "left")}
        for text, (x, y) in ends.items():
            color, offset, ha = placement[text]
            ax.annotate(text, xy=(x, y), xytext=offset, textcoords="offset points", fontsize=6.0, color=color, ha=ha, va="bottom")
    ax.set_xlabel("tokens added / human tokens")
    ax.set_ylabel(f"change in {label(target)} loss (%)")


def ceg_filtering(out: Path) -> Path:
    fig = plt.figure(figsize=(TEXT_WIDTH, 2.3), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.05])
    left, right = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    mesh = _filtering_map(left)
    bar = fig.colorbar(mesh, ax=left, location="bottom", shrink=0.9, aspect=30, pad=0.02)
    bar.set_label(f"change in {label(TARGET)} loss from filtering (%)", fontsize=6.5)
    bar.ax.tick_params(labelsize=6.5)
    bar.outline.set_visible(False)
    _repetition_panel(right, TARGET, labels=True, fresh=False)
    right.set_title("Repeat Human or Add AI?", fontsize=8.5)
    depths = sorted({m.depth for m in models().values() if m.arm == "repeat"})
    sizes = [Line2D([], [], color=c, marker="o", ms=3.2, lw=0, label=SIZE_LABEL[d]) for c, d in zip((PALETTE.grey_light, PALETTE.grey), depths)]
    right.legend(handles=sizes, loc="upper left", fontsize=6.0, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25)
    for ax in (left, right):
        ax.tick_params(labelsize=6.5)
    return save_figure(fig, out, "law_ceg_filtering")


def repetition_targets(out: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 3.5), sharex=True, layout="constrained")
    for ax, target in zip(axes.ravel(), REPETITION_TARGETS):
        _repetition_panel(ax, target, labels=False, fresh=True)
        ax.set_title(label(target), fontsize=8.5)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(labelsize=6.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("change in loss (%)")
    axes[1, 1].set_xlabel("tokens added / human tokens")
    depths = sorted({m.depth for m in models().values() if m.arm == "repeat"})
    handles = [Line2D([], [], color=HUMAN, lw=1.2, marker="o", ms=3.2, label="repeat human text"),
               Line2D([], [], color=HUMAN, lw=0.9, ls=(0, (1, 1.5)), marker="o", ms=3.2, mfc="white", mew=0.9, label="add fresh human text"),
               Line2D([], [], color=AI, lw=1.2, marker="o", ms=3.2, label="add AI text"),
               Line2D([], [], color=PALETTE.grey_light, marker="o", ms=3.2, lw=0, label=f"{SIZE_LABEL[depths[0]]} (lighter)"),
               Line2D([], [], color=PALETTE.grey, marker="o", ms=3.2, lw=0, label=f"{SIZE_LABEL[depths[1]]} (darker)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=6.4, columnspacing=1.2, handletextpad=0.4)
    return save_figure(fig, out, "law_repetition_targets")


def _ladder_panel(ax: plt.Axes, target: str, summary: CegSummary) -> None:
    color = TARGET_STYLE[target][0]
    control = models()[LADDER_CONTROL]
    ladder = sorted((r for r in read_records("ceg_ladder.csv", CegRecord, target=target, control=LADDER_CONTROL)
                     if r.arm in ("control", "ai") and r.ratio <= LADDER_MAX_RATIO), key=lambda r: r.ai_share)
    curve = read_records("ceg_ladder_curve.csv", CegCurvePoint, target=target, control=LADDER_CONTROL)
    ceg = [p.ceg if p.ceg is not None else math.nan for p in curve]
    ax.plot([100 * p.ai_share for p in curve], ceg, color=color, lw=1.1, zorder=2)
    if math.isnan(ceg[0]):  # the human-only control lies below the reference budgets: mark the bound, not a value
        reference = ReferenceMix(fitted(target), control.n_params, summary.human_tpp_range)
        ax.scatter([0], [math.exp(reference.grid[0][0]) / control.total_tokens], marker="v", s=24, color=color, zorder=3)
    ax.scatter([100 * r.ai_share for r in ladder], [r.observed_ceg for r in ladder], s=18, color=color, edgecolor="white", lw=0.5, zorder=4)
    ax.axhline(1, color=RULE, lw=0.7, zorder=1)
    ax.axvline(100 * WEB_2026_AI_SHARE, color=PALETTE.grey_light, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.text(100 * WEB_2026_AI_SHARE + 1.5, 0.04, "2026 web mix", fontsize=6.3, color=PALETTE.muted, ha="left", va="bottom")
    ax.set_title(label(target))
    ax.set_xlim(-3, 103)
    ax.set_ylim(0, 2)
    ax.xaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
    ax.yaxis.set_major_locator(FixedLocator([0, 0.5, 1, 1.5, 2]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}×"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("AI share of training tokens (%)")


def ceg_ladder(out: Path) -> Path:
    summaries = {s.target: s for s in map(CegSummary.model_validate, json.loads((LAWS / "ceg_ladder_summary.json").read_text(encoding="utf-8")))}
    control = models()[LADDER_CONTROL]
    fig, axes = plt.subplots(1, len(LADDER_TARGETS), figsize=(TEXT_WIDTH, 2.3), sharey=True)
    for ax, target in zip(axes, LADDER_TARGETS):
        _ladder_panel(ax, target, summaries[target])
        ax.tick_params(labelleft=True)
    axes[0].set_ylabel("compute-equivalent gain\nover the 2026 web mix")
    handles = [Line2D([], [], color=INK, marker="o", lw=0, ms=3.6, label=f"measured, {SIZE_LABEL[control.depth]} at {budget_label(control.human_tpp)}"),
               Line2D([], [], color=INK, lw=1.1, label="Ours")]
    fig.legend(handles=handles, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.03), handletextpad=0.5, labelspacing=0.3)
    fig.tight_layout(w_pad=1, rect=(0, 0.15, 1, 1))
    return save_figure(fig, out, "law_ceg_ladder")
