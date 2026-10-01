"""Evaluating on AI text, from every model's loss on the human- and AI-labeled partitions of FW26 (no law is fitted).

- law_when_ai (Figure 6): the AI share of training tokens that minimizes our law's predicted loss per evaluation set, and
  what a validation set reports for the AI additions that raise loss on human text, as its AI-labeled share grows.
- ai_text_evaluation (appendix): every model's loss on AI- against human-labeled text, and the share of the human-harming
  AI additions that a mixed validation set reports as improvements.

A validation set whose scored bytes are a share q AI-labeled reports the change ((1 - q) dh + q da) / ((1 - q) h_c + q a_c)
for a run whose losses change by dh and da against a control with losses h_c and a_c (see wildai.laws.analysis.masking).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from paper.figures.law_common import LAWS, TARGET_STYLE, label, read_records, tpp_axis
from paper.results import losses, models
from paper.style import AI, HUMAN, INK, MUTED, PALETTE, RULE, TEXT_WIDTH, TPP_H_LABEL, save_figure
from wildai.laws.analysis.masking import AI as AI_PARTITION
from wildai.laws.analysis.masking import HUMAN as HUMAN_PARTITION
from wildai.laws.analysis.masking import MaskingContrast, MaskingSummary
from wildai.laws.analysis.recommendations import OptimalShare
from wildai.laws.ceg import WEB_2026_AI_SHARE

MASKED_SHARES = np.linspace(0.0, 0.30, 121)
MARKED_SHARES = (0.05, WEB_2026_AI_SHARE)
SPLITS = (("fit", HUMAN, "-", "19.9M–268M"), ("held_out", PALETTE.held_out[1], (0, (3, 1.5)), "477M, 973M"))
CURVE_LABELS = {"cosmopedia": (40, 88), "fw26": (11, 72)}  # the AI-leaning sets are labeled on their curves, clear of the lines


@dataclass(frozen=True)
class Harmed:
    """An AI addition that raises loss on human-labeled text, with its control's losses on both partitions."""

    contrast: MaskingContrast
    human_control: float
    ai_control: float

    def reported_pct(self, q: np.ndarray | float) -> np.ndarray:
        """Change in loss (%) that a validation set with AI-labeled share q reports for this run."""

        c = self.contrast
        return 100.0 * ((1 - q) * c.delta_human_bpb + q * c.delta_ai_bpb) / ((1 - q) * self.human_control + q * self.ai_control)

    @property
    def critical_share(self) -> float:
        """The AI-labeled share above which the run looks like an improvement (every such run improves on AI text)."""

        assert self.contrast.critical_share is not None
        return self.contrast.critical_share


def harmed() -> list[Harmed]:
    loss = losses()
    return [Harmed(c, loss[(c.control, HUMAN_PARTITION)], loss[(c.control, AI_PARTITION)])
            for c in read_records("masking_contrasts.csv", MaskingContrast) if c.delta_human_bpb > 0]


def summary() -> MaskingSummary:
    return MaskingSummary.model_validate(json.loads((LAWS / "masking.json").read_text(encoding="utf-8")))


def optimal_share_panel(ax: plt.Axes) -> None:
    """The AI share that minimizes our law's predicted loss at 268M with the human corpus fixed, per evaluation set."""

    handles = {}
    for target, (color, ls) in TARGET_STYLE.items():
        rows = sorted(read_records("optimal_share.csv", OptimalShare, target=target), key=lambda r: r.human_tpp)
        handles[target], = ax.plot([r.human_tpp for r in rows], [100 * r.optimal_share for r in rows], color=color, ls=ls, lw=1.3, label=label(target))
    for target, xy in CURVE_LABELS.items():
        ax.text(*xy, label(target), fontsize=6.4, color=TARGET_STYLE[target][0], ha="left", va="center")
    ax.legend(handles=[h for t, h in handles.items() if t not in CURVE_LABELS], loc="lower right", bbox_to_anchor=(1.0, 0.04), fontsize=6.2,
              handlelength=1.8, handletextpad=0.5, borderaxespad=0.2, labelspacing=0.3)
    tpp_axis(ax)
    ax.set_ylim(-3, 103)
    ax.set_title("How Much AI to Add")
    ax.set_ylabel("optimal AI share (%)")
    ax.set_xlabel(TPP_H_LABEL)


def _arrow(color: str) -> dict[str, str | float]:
    return {"arrowstyle": "-", "color": color, "lw": 0.5, "shrinkA": 0, "shrinkB": 1}


def reported_panel(ax: plt.Axes, runs: list[Harmed]) -> None:
    """The median change in loss (with its interquartile band) a validation set reports for the human-harming runs, as its
    AI-labeled share grows: worse on human text alone, better once a few percent of the set is AI text."""

    stats = summary()
    q = np.linspace(0.0, 0.40, 161)
    lo, mid, hi = np.percentile(np.array([run.reported_pct(q) for run in runs]), [25, 50, 75], axis=0)
    at_web = next(s for s in stats.at_share if s.ai_share == WEB_2026_AI_SHARE)
    flip = 100 * stats.median_flip_share
    (worse, worse_alpha), (better, better_alpha), (band, band_alpha) = PALETTE.worse_fill, PALETTE.better_fill, PALETTE.band_fill
    ax.axhspan(0.0, 10.0, color=worse, alpha=worse_alpha, lw=0, zorder=0)
    ax.axhspan(-10.0, 0.0, color=better, alpha=better_alpha, lw=0, zorder=0)
    ax.fill_between(100 * q, lo, hi, color=band, alpha=band_alpha, lw=0, zorder=2)
    ax.plot(100 * q, mid, color=INK, lw=1.6, zorder=3)
    ax.axhline(0.0, color=RULE, lw=0.8, zorder=1)
    ax.axvline(100 * WEB_2026_AI_SHARE, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.annotate(f"human text only:\n+{mid[0]:.1f}% (worse)", xy=(0.0, mid[0]), xytext=(3.0, 1.45), fontsize=6.0, color=PALETTE.ai_dark, ha="left",
                va="bottom", linespacing=1.0, arrowprops=_arrow(PALETTE.ai_dark))
    at_web_pct = 100 * at_web.median_reported_change
    ax.annotate(f"{100 * WEB_2026_AI_SHARE:.1f}% AI, as on the 2026 web:\n{at_web_pct:.1f}% (looks better)".replace("-", "−"),
                xy=(100 * WEB_2026_AI_SHARE, at_web_pct), xytext=(2.0, -4.3), fontsize=6.0, color=HUMAN, ha="left", va="top", linespacing=1.0,
                arrowprops=_arrow(HUMAN))
    ax.annotate(f"flips at {flip:.1f}% AI", xy=(flip, 0.0), xytext=(flip + 6.0, 0.95), fontsize=6.0, color=MUTED, ha="left", va="bottom",
                arrowprops=_arrow(MUTED))
    ax.set_xlim(-1.0, 40.0)
    ax.set_ylim(-5.6, 2.6)
    ax.set_xlabel("AI-labeled share of the validation set (%)")
    ax.set_ylabel("change in validation loss (%)")
    ax.set_title("What the Validation Set Reports")


def when_ai(out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 1.95), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.08)
    optimal_share_panel(axes[0])
    reported_panel(axes[1], harmed())
    axes[1].xaxis.label.set_fontsize(7.4)
    axes[1].yaxis.label.set_fontsize(7.4)
    for ax in axes:
        ax.tick_params(labelsize=6.5)
    return save_figure(fig, out, "law_when_ai")


def _partition_panel(ax: plt.Axes) -> None:
    """Loss on AI-labeled against human-labeled FW26 text for every model of the scaling-law study."""

    loss = losses()
    cohort = [m for m in models().values() if m.split in ("fit", "held_out")]
    lo, hi = 0.5, 1.45  # the same range on both axes, so equal loss runs corner to corner
    ax.plot([lo, hi], [lo, hi], color=RULE, lw=0.8, zorder=1)
    ax.text(0.98, 1.0, "equal loss", fontsize=6.0, color=MUTED, rotation=45, ha="center", va="bottom", rotation_mode="anchor")
    for ai, color, size, alpha, zorder in ((True, AI, 7, 0.75, 2), (False, HUMAN, 9, 0.9, 3)):
        chosen = [m for m in cohort if (m.arm == "ai") == ai]
        ax.scatter([loss[(m.name, HUMAN_PARTITION)] for m in chosen], [loss[(m.name, AI_PARTITION)] for m in chosen], s=size, color=color,
                   edgecolor="none", alpha=alpha, zorder=zorder)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4])
    ax.set_yticks([0.6, 0.8, 1.0, 1.2, 1.4])
    ax.set_xlabel("loss on human-labeled\nFW26 (bpb)", fontsize=7.4)
    ax.set_ylabel("loss on AI-labeled\nFW26 (bpb)", fontsize=7.4)
    ax.set_title("AI vs. Human Text")
    handles = [Line2D([], [], marker="o", ms=3.6, lw=0, color=HUMAN, label="human text only"),
               Line2D([], [], marker="o", ms=3.4, lw=0, color=AI, label="add AI text")]
    ax.legend(handles=handles, loc="upper left", fontsize=6.0, handletextpad=0.3, borderaxespad=0.2, labelspacing=0.3)


def _masked_panel(ax: plt.Axes, runs: list[Harmed]) -> None:
    """Share of the human-harming AI additions that a mixed validation set reports as improvements, by its AI share."""

    for split, color, ls, sizes in SPLITS:
        chosen = [run.critical_share for run in runs if run.contrast.split == split]
        masked = [100.0 * np.mean([s < q for s in chosen]) for q in MASKED_SHARES]
        ax.plot(MASKED_SHARES, masked, color=color, lw=1.4, ls=ls, zorder=3, label=f"{sizes}, {len(chosen)} runs")
    for q, above in zip(MARKED_SHARES, (False, True)):
        y = 100.0 * np.mean([run.critical_share < q for run in runs])
        ax.scatter([q], [y], s=16, color=INK, zorder=5)
        ax.annotate(f"{y:.1f}% at {100 * q:.3g}%", xy=(q, y), xytext=(q + 0.008, y + (4 if above else -6)), fontsize=6.0, color=INK,
                    va="bottom" if above else "top", ha="left")
    ax.set_xlim(0.0, 0.30)
    ax.set_ylim(0, 104)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlabel("AI-labeled share of the validation set")
    ax.set_ylabel("harmful runs that look better")
    ax.set_title("Runs That Look Like Improvements")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.04), fontsize=6.0, handlelength=1.6, handletextpad=0.4, borderaxespad=0.2, labelspacing=0.3)


def ai_text_evaluation(out: Path) -> Path:
    fig = plt.figure(figsize=(0.78 * TEXT_WIDTH, 2.35), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05])
    axes = [fig.add_subplot(grid[k]) for k in range(2)]
    _partition_panel(axes[0])
    _masked_panel(axes[1], harmed())
    for ax in axes:
        ax.tick_params(labelsize=6.5)
    return save_figure(fig, out, "ai_text_evaluation")
