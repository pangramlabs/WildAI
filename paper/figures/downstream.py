"""Downstream accuracy (appendix): change in CORE against each model's own human-only control as AI text or fresh human
text is added, one panel per human budget, every fitted size."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paper.groups import FITTED_DEPTHS, added_fraction, fresh_human_groups, matched_group
from paper.results import Model, downstream
from paper.style import AI, HUMAN, PALETTE, R_LABEL, SIZE_LABEL, TEXT_WIDTH, budget_label, ratio_axis, save_figure, zero_line

BUDGETS = (4.72, 18.88, 37.76)
AI_SIZE_COLORS = dict(zip(FITTED_DEPTHS, PALETTE.ai_sizes))
HUMAN_SIZE_COLORS = dict(zip(FITTED_DEPTHS, PALETTE.human_sizes))


def core_changes(out: Path) -> Path:
    core = {name: 100.0 * score for name, score in downstream().items()}  # points of chance-centred accuracy
    fig, axes = plt.subplots(1, len(BUDGETS), figsize=(TEXT_WIDTH, 2.55), sharey=True, sharex=True)
    for ax, budget in zip(axes, BUDGETS):
        zero_line(ax)
        controls: list[Model] = []
        for depth in FITTED_DEPTHS:
            group = matched_group(depth, budget)
            if group is None:
                continue
            controls.append(group.control)
            fresh = [(m, g.control) for g in fresh_human_groups(depth, budget) for m in g.human]
            for pairs, color, hollow in (([(m, group.control) for m in group.ai], AI_SIZE_COLORS[depth], False), (fresh, HUMAN_SIZE_COLORS[depth], True)):
                points = sorted((added_fraction(m, c), core[m.name] - core[c.name]) for m, c in pairs if m.name in core and c.name in core)
                if not points:
                    continue
                xs, ys = zip(*points)
                ax.plot(xs, ys, color=color, lw=0.8, alpha=0.35 if hollow else 0.55, ls=(0, (1, 1.5)) if hollow else "-", zorder=2)
                if hollow:
                    ax.scatter(xs, ys, s=12, facecolor="white", edgecolor=color, lw=0.9, zorder=4)
                else:
                    ax.scatter(xs, ys, s=12, color=color, edgecolor="white", lw=0.4, zorder=4)
        ratio_axis(ax)
        ax.set_title(budget_label(budget))
    axes[0].set_ylabel("change in CORE\n(points)")
    axes[1].set_xlabel(R_LABEL)
    handles = [Line2D([], [], color=AI_SIZE_COLORS[d], marker="o", ms=3.6, lw=0, label=SIZE_LABEL[d]) for d in FITTED_DEPTHS]
    handles += [Line2D([], [], color=AI, marker="o", ms=3.6, lw=0, mfc=AI, label="add AI text"),
                Line2D([], [], color=HUMAN, marker="o", ms=3.6, lw=0, mfc="white", mew=0.9, label="add fresh human text")]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, 0.0), columnspacing=1.1, handletextpad=0.4, fontsize=6.4)
    fig.tight_layout(w_pad=0.8, h_pad=0.6, rect=(0, 0.09, 1, 1))
    return save_figure(fig, out, "downstream_core")
