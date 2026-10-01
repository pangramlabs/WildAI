"""The terms of our law (appendix).

- law_anatomy: the credit, the harm and the net value of an AI token on C4 at 268M, from 5 to 100 TPP_h.
- style_cd_window: the credit window R* against the human budget for each evaluation set, and the largest loss reduction
  added AI text achieves, per fitted group and under our law (in the style of Qin et al.'s Figure 4).
- style_cd_effective_data_collapse: the data-reducible loss of every fitted run against its human tokens and against our
  law's effective human tokens (in the style of Qin et al.'s Figure 3).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from paper.figures.law_common import (
    BUDGET_CMAP,
    BUDGET_NORM,
    RATIO_CMAP,
    RATIOS,
    SIZE_COLORS,
    TARGET_STYLE,
    VALUE_LABEL,
    backbone,
    budget_colorbar,
    fitted,
    label,
    ours_terms,
    reference_n,
    tpp_axis,
)
from paper.figures.law_overview import VALUE_BUDGETS, token_value_panel
from paper.groups import FITTED_DEPTHS, change_pct, groups
from paper.results import losses, models
from paper.style import AI, HUMAN, INK, R_LABEL, SIZE_LABEL, TEXT_WIDTH, TPP_H_LABEL, ratio_axis, save_figure
from wildai.laws.ceg import largest_loss_reduction
from wildai.laws.forms import FittedLaw

TARGET = "c4"
RATIO_NORM = LogNorm(vmin=2.5e-3, vmax=64.0)


def anatomy(out: Path) -> Path:
    model = fitted(TARGET)
    n = reference_n()
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.35), layout="constrained")
    for tpp in VALUE_BUDGETS:
        terms = ours_terms(model, n, tpp * n, RATIOS)
        color = BUDGET_CMAP(BUDGET_NORM(tpp))
        axes[0].plot(RATIOS, 1.0 + terms.credit, color=color, lw=1.3)
        axes[1].plot(RATIOS, 1.0 + terms.harm, color=color, lw=1.3)
    token_value_panel(axes[2], model, RATIOS)
    axes[0].set_title("Credit")
    axes[0].set_ylabel("effective human data (×)")
    axes[1].set_title("Harm")
    axes[1].set_ylabel("inflation of the data term (×)")
    axes[2].set_title("Net Value of an AI Token")
    axes[2].set_ylabel(VALUE_LABEL)
    for ax in axes:
        ratio_axis(ax, 0.01, 100.0)
    axes[1].set_xlabel(R_LABEL)
    budget_colorbar(fig, list(axes), shrink=0.4, aspect=40, pad=0.08)
    return save_figure(fig, out, "law_anatomy")


def cd_window(out: Path) -> Path:
    budgets = np.logspace(np.log10(4), np.log10(100), 100)
    n = reference_n()
    fig, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.3))
    for target, (color, ls) in TARGET_STYLE.items():
        left.plot(budgets, ours_terms(fitted(target), n, budgets * n, 0.0).window, color=color, ls=ls, lw=1.3, label=label(target))
    model = fitted(TARGET)
    best = [100.0 * largest_loss_reduction(model, n, tpp * n) for tpp in budgets]
    right.plot(budgets, best, color=INK, lw=1.2, ls=(0, (3, 2)), zorder=2, label="Ours")
    for group in groups():
        if group.depth in FITTED_DEPTHS and group.budget is not None and group.ai:
            observed = max(-change_pct(m, group.control, TARGET) for m in group.ai)
            right.scatter([group.control.human_tpp], [max(observed, 0.0)], s=16, color=SIZE_COLORS[group.depth], edgecolor="white", lw=0.4, zorder=4)
    for ax in (left, right):
        tpp_axis(ax, 4, 100)
        ax.set_xlabel(TPP_H_LABEL)
        ax.tick_params(labelsize=6.5)
    left.set_yscale("log")
    left.set_ylabel("credit window $R^{\\star}$")
    left.set_title("Window Closes as the Budget Grows")
    right.set_ylabel(f"largest {label(TARGET)} loss reduction (%)")
    right.set_title("Most That AI Text Can Help")
    right.set_ylim(-0.1, 3.2)
    target_handles = [Line2D([], [], color=c, ls=ls, lw=1.3, label=label(t)) for t, (c, ls) in TARGET_STYLE.items()]
    size_handles = [Line2D([], [], color=SIZE_COLORS[d], marker="o", ms=3.4, lw=0, label=SIZE_LABEL[d]) for d in FITTED_DEPTHS]
    size_handles.append(Line2D([], [], color=INK, lw=1.2, ls=(0, (3, 2)), label="Ours"))
    fig.legend(handles=target_handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.045), columnspacing=1.2, handletextpad=0.5)
    fig.legend(handles=size_handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02), columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(w_pad=1.2, rect=(0, 0.15, 1, 1))
    return save_figure(fig, out, "style_cd_window")


def _data_term(model: FittedLaw, human_tokens: np.ndarray) -> np.ndarray:
    """B D^-beta: our law's loss at r = 0 above its backbone, the same at every model size."""

    n = reference_n()
    return model.loss(n, human_tokens, 0.0) - backbone(model, n)


def cd_effective_data_collapse(out: Path) -> Path:
    model = fitted(TARGET)
    runs = [m for m in models().values() if m.split == "fit"]
    human_runs = [m for m in runs if m.arm != "ai"]
    ai_runs = [m for m in runs if m.arm == "ai"]
    reducible = {m.name: losses()[(m.name, TARGET)] - float(backbone(model, m.n_params)[0]) for m in runs}
    effective = {m.name: m.human_tokens * float(ours_terms(model, m.n_params, m.human_tokens, m.ratio).effective_data[0]) for m in runs}
    fig, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.4), sharey=True)
    tokens = np.logspace(7.6, 10.8, 100)
    for ax, x in ((left, lambda m: m.human_tokens), (right, lambda m: effective[m.name])):
        ax.plot(tokens, _data_term(model, tokens), color=INK, lw=1.0, zorder=2)
        ax.scatter([x(m) for m in human_runs], [reducible[m.name] for m in human_runs], s=13, color=HUMAN, edgecolor="white", lw=0.4, zorder=5)
        ax.scatter([x(m) for m in ai_runs], [reducible[m.name] for m in ai_runs], c=[m.ratio for m in ai_runs], cmap=RATIO_CMAP, norm=RATIO_NORM,
                   s=11, edgecolor="none", alpha=0.85, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator([1e8, 1e9, 1e10]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v / 1e9:g}B"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelsize=6.5)
    left.set_xlabel("human tokens")
    right.set_xlabel("effective human tokens (Ours)")
    left.set_ylabel("data-reducible loss (bits per byte)")
    left.set_title("Chinchilla")
    right.set_title("Ours")
    handles = [Line2D([], [], color=HUMAN, marker="o", ms=3.6, lw=0, label="human text only"),
               Line2D([], [], color=AI, marker="o", ms=3.6, lw=0, label="add AI text (shaded by $r$)"),
               Line2D([], [], color=INK, lw=1.0, label="Chinchilla data term")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.36, -0.02), columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.12, 1, 1))
    # the colour bar sits in the legend row, placed after the layout so the layout leaves it alone
    bar = fig.colorbar(ScalarMappable(norm=RATIO_NORM, cmap=RATIO_CMAP), cax=fig.add_axes((0.72, 0.045, 0.2, 0.025)), orientation="horizontal")
    bar.set_ticks([0.01, 0.1, 1, 10])
    bar.set_ticklabels(["0.01×", "0.1×", "1×", "10×"])
    bar.ax.tick_params(labelsize=6, length=2)
    bar.ax.minorticks_off()
    bar.outline.set_visible(False)
    bar.set_label(R_LABEL, fontsize=6.5, labelpad=2)
    return save_figure(fig, out, "style_cd_effective_data_collapse")
