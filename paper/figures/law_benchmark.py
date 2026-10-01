"""Our law against the published laws (appendix).

- law_benchmark: which benchmarked laws meet the five criteria of Section 3, beside their paired error on the held-out
  477M and 973M runs for every evaluation set.
- law_extrapolation: our law and the joint law of Shukor et al. on C4 at 20 TPP_h, at 268M and projected to 8B, for the
  fit on every fitted size and a fit that leaves one size out (the 268M runs, i.e. fitted through 135M), with the range
  over all leave-one-size-out fits as a band; and the loss floor each joint-law fit implies for a training mix.
- law_extrapolation_alt: the same figure with the fit that leaves out the 19.9M runs as the dashed comparison.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from paper.figures.law_common import change_at, fitted, label, read_records, reference_n
from paper.groups import FITTED_DEPTHS, change_pct, matched_group
from paper.style import AI, HUMAN, INK, MUTED, PALETTE, R_LABEL, RULE, SIZE_LABEL, TEXT_WIDTH, budget_label, ratio_axis, save_figure, zero_line
from wildai.laws.analysis.stability import PROJECTION_PARAMS, PROJECTION_TPP
from wildai.laws.bank import FOLDS, fold_depth
from wildai.laws.catalog import PAPER_LAW, law
from wildai.laws.forms import FittedLaw
from wildai.laws.score import Score

CRITERIA = ("C1 Flexible\ntoken value", "C2 Help\nand harm", "C3 Separate\nbenefit and harm", "C4 Finite first-\ntoken value", "C5 Chinchilla\nat r = 0")
# Whether each law's form meets each criterion: y through free coefficients, p partly or only under a condition the fit may
# not satisfy (a first-token value that is finite only when a fitted exponent reaches one, or fixed at one), n for no
# coefficient values. Rows in the figure's order.
MARKS: dict[str, str] = {
    "chinchilla_5": "nnnpy",
    "muennighoff_7": "pnnpy",
    "cd_8": "pnnpy",
    "lovelace_eq8_9": "pyyny",
    "atlas_no_repeat": "pnnyy",
    "he_human_share": "pynyy",
    "hamidieh_first_order": "pynny",
    "shukor_additive": "yynpp",
    "shukor_joint": "yynpp",
    "jain_token_weighted": "yypyy",
    "sedova_ai_share_coupled": "yyyyy",
    PAPER_LAW: "yyyyy",
}
GLYPHS = {"y": ("✓", HUMAN, "met"), "p": ("◐", PALETTE.ochre, "partly or conditional"), "n": ("✗", PALETTE.grey_light, "not met")}
TARGET_MARKERS = {
    "c4": {"marker": "o", "color": HUMAN, "mfc": HUMAN},
    "fw22": {"marker": "o", "color": HUMAN, "mfc": "white"},
    "paloma": {"marker": "^", "color": HUMAN, "mfc": HUMAN},
    "fw26": {"marker": "D", "color": PALETTE.ochre, "mfc": PALETTE.ochre},
    "fw26_human": {"marker": "D", "color": PALETTE.ochre, "mfc": "white"},
    "fw26_ai": {"marker": "s", "color": AI, "mfc": "white"},
    "cosmopedia": {"marker": "s", "color": AI, "mfc": AI},
}
JOINT = "shukor_joint"
TARGET = "c4"
RATIOS = np.logspace(np.log10(2.5e-3), np.log10(64), 160)
SHARES = np.linspace(0.0, 0.985, 300)  # the runs reach r = 64, an AI share of 98.5 %


def benchmark(out: Path) -> Path:
    errors = {(s.law, s.target): 1e3 * s.paired_rmse for s in read_records("scores.csv", Score, fold="all", depth="", cutoff="")
              if s.paired_rmse is not None}
    fig, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.35), gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.06})
    ys = np.arange(len(MARKS))[::-1]
    glyph_font = {"fontfamily": "DejaVu Sans"}  # the check, half circle and cross
    for y, (key, cells) in zip(ys, MARKS.items()):
        if set(cells) == {"y"}:
            for ax in (left, right):
                ax.axhspan(y - 0.5, y + 0.5, color=PALETTE.highlight, zorder=0, lw=0)
        for x, cell in enumerate(cells):
            text, color, _ = GLYPHS[cell]
            left.text(x, y, text, ha="center", va="center", fontsize=9, color=color, fontweight="bold", **glyph_font)
        for target, marker in TARGET_MARKERS.items():
            if (key, target) in errors:
                right.plot([errors[(key, target)]], [y], lw=0, ms=4.2, mew=0.8, **marker, zorder=3)
    left.set_xlim(-0.5, len(CRITERIA) - 0.5)
    left.set_ylim(-0.5, len(MARKS) - 0.5)
    left.set_yticks(ys)
    left.set_yticklabels([f"{law(key).label} ({law(key).k})" for key in MARKS], fontsize=7.5)
    left.get_yticklabels()[-1].set_fontweight("bold")  # our law, the last row
    left.set_xticks(range(len(CRITERIA)))
    left.set_xticklabels(CRITERIA, rotation=50, ha="left", va="bottom", fontsize=6.2, rotation_mode="anchor")
    left.xaxis.tick_top()
    left.tick_params(axis="both", length=0)
    for spine in left.spines.values():
        spine.set_visible(False)
    for y in ys[:-1]:
        left.axhline(y - 0.5, color=RULE, lw=0.4, zorder=0)
        right.axhline(y - 0.5, color=RULE, lw=0.4, zorder=0)
    key_x = -0.5
    for text, color, meaning in GLYPHS.values():
        left.text(key_x, -1.0, text, fontsize=7.5, color=color, va="top", fontweight="bold", **glyph_font)
        left.text(key_x + 0.28, -1.0, meaning, fontsize=6.2, color=MUTED, va="top")
        key_x += 0.55 + 0.085 * len(meaning)
    right.set_xscale("log")
    right.set_xlim(0.5, 130)
    right.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20, 50, 100]))
    right.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    right.xaxis.set_minor_formatter(NullFormatter())
    right.set_ylim(-0.5, len(MARKS) - 0.5)
    right.set_yticks([])
    right.spines["left"].set_visible(False)
    right.set_xlabel(r"Reserved-size RMSE ($\times 10^{-3}$)")
    right.set_title("Error on the reserved 477M and 973M")
    handles = [Line2D([], [], lw=0, ms=4.2, mew=0.8, label=label(t), **m) for t, m in TARGET_MARKERS.items()]
    right.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.07), ncol=4, columnspacing=0.8, handletextpad=0.3, fontsize=6.2)
    left.set_title("Does the law meet the criterion?", pad=12)
    return save_figure(fig, out, "law_benchmark")


def fold_label(fold: str) -> str:
    """How the figure names a fit: the largest size it saw, or the size it left out."""

    left_out = fold_depth(fold)
    if left_out in (None, FITTED_DEPTHS[-1]):
        return f"fitted through {SIZE_LABEL[max(d for d in FITTED_DEPTHS if d != left_out)]}"
    return f"fitted without {SIZE_LABEL[left_out]}"


def joint_floor(model: FittedLaw, share: np.ndarray) -> np.ndarray:
    """The joint law's loss with unlimited parameters and data at AI share f: E + 1 / (c_H (1 - f)^q_H + c_A f^q_A)."""

    c = model.law.coefficients(model.values)
    ai = np.where(share > 0, share, 1.0) ** c["q_A"] * (share > 0)
    return c["E"] + 1.0 / (c["c_H"] * (1.0 - share) ** c["q_H"] + c["c_A"] * ai)


def _change_panel(ax: plt.Axes, n_params: float, dashed: str) -> None:
    """Both laws' predicted change in loss against r at size N: the fit on every size (solid), the `dashed` fit, and the
    range over every leave-one-size-out fit (band)."""

    zero_line(ax)
    human = PROJECTION_TPP * n_params
    for key, color, fill, alpha in ((PAPER_LAW, INK, PALETTE.grey_light, 0.55), (JOINT, PALETTE.shukor, PALETTE.shukor, 0.16)):
        curves = {fold: change_at(fitted(TARGET, key, fold), n_params, human, RATIOS) for fold in FOLDS}
        band = np.array(list(curves.values()))
        ax.fill_between(RATIOS, band.min(axis=0), band.max(axis=0), color=fill, alpha=alpha, lw=0, zorder=2)
        ax.plot(RATIOS, curves[dashed], color=color, lw=1.0, ls=(0, (3, 1.6)), zorder=4)
        ax.plot(RATIOS, curves["all"], color=color, lw=1.6, zorder=5)
    ratio_axis(ax, 2e-3, 90, sparse=True)
    ax.set_xlabel(R_LABEL)
    ax.text(0.04, 0.95, budget_label(PROJECTION_TPP), transform=ax.transAxes, fontsize=6.4, color=MUTED, ha="left", va="top")


def _floor_panel(ax: plt.Axes, dashed: str) -> None:
    """The loss floor on the target each joint-law fit implies for a training mix, relative to human-only training."""

    shukor = PALETTE.shukor
    ax.axhspan(-13, 0, color=shukor, alpha=0.05, lw=0, zorder=0)
    for fold in FOLDS:
        model = fitted(TARGET, JOINT, fold)
        relative = 100.0 * (joint_floor(model, SHARES) / joint_floor(model, np.zeros(1))[0] - 1.0)
        if fold == "all":
            ax.plot(100 * SHARES, relative, color=shukor, lw=1.6, zorder=5)
        else:
            ax.plot(100 * SHARES, relative, color=shukor, lw=1.0 if fold == dashed else 0.7, ls=(0, (3, 1.6)), alpha=1.0 if fold == dashed else 0.45, zorder=4)
    ax.axhline(0.0, color=INK, lw=1.6, zorder=6)
    everything = fitted(TARGET, JOINT)
    ai_only = 100.0 * (1.0 - joint_floor(everything, np.ones(1))[0] / joint_floor(everything, np.zeros(1))[0])
    ax.text(2, 0.45, "Ours: one floor for every mix", fontsize=6.2, color=INK, ha="left", va="bottom")
    ax.text(97, -10.6, f"Shukor: AI-only training\nreaches a floor {ai_only:.0f}% lower", fontsize=6.2, color=shukor, ha="right", va="top", linespacing=1.0)
    ax.set_ylim(-13, 5)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
    ax.set_xlabel("AI share of training tokens (%)")
    ax.set_ylabel("loss floor vs. human-only (%)")
    ax.set_title(f"Loss Floor on {label(TARGET)}", fontsize=8.5)


def _extrapolation(out: Path, dashed: str, stem: str) -> Path:
    fig = plt.figure(figsize=(TEXT_WIDTH, 2.3), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.08])
    trained = fig.add_subplot(grid[0])
    projected = fig.add_subplot(grid[1], sharey=trained)
    floor = fig.add_subplot(grid[2])
    fig.get_layout_engine().set(w_pad=0.04)
    _change_panel(trained, reference_n(), dashed)
    _change_panel(projected, PROJECTION_PARAMS, dashed)
    group = matched_group(16, 18.88)
    trained.scatter([m.ratio for m in group.ai], [change_pct(m, group.control, TARGET) for m in group.ai], s=16, color=PALETTE.human_mid,
                    edgecolor="white", lw=0.5, zorder=6)
    trained.set_ylim(-4.6, 4.2)
    trained.set_ylabel(f"change in {label(TARGET)} loss (%)")
    trained.set_title(f"{SIZE_LABEL[16]} (Trained)", fontsize=8.5)
    projected.set_title(f"{PROJECTION_PARAMS / 1e9:g}B (Extrapolated)", fontsize=8.5)
    projected.tick_params(labelleft=False)
    projected.set_facecolor(to_rgba(RULE, 0.1))  # no model of this size was trained
    _floor_panel(floor, dashed)
    for ax in (trained, projected, floor):
        ax.tick_params(labelsize=6.5)
    handles = [Line2D([], [], color=INK, lw=1.6, label="Ours"), Line2D([], [], color=PALETTE.shukor, lw=1.6, label="Shukor joint"),  # filled by column
               Line2D([], [], color=MUTED, lw=1.6, label=fold_label("all")), Line2D([], [], color=MUTED, lw=1.0, ls=(0, (3, 1.6)), label=fold_label(dashed)),
               Patch(color=MUTED, alpha=0.28, lw=0, label="one fitted size left out"),
               Line2D([], [], color=PALETTE.human_mid, marker="o", lw=0, ms=3.8, label="trained models")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=6.4, frameon=False, columnspacing=1.8, handlelength=2.2, labelspacing=0.35)
    return save_figure(fig, out, stem)


def extrapolation(out: Path) -> Path:
    return _extrapolation(out, "without_268M", "law_extrapolation")


def extrapolation_alt(out: Path) -> Path:
    return _extrapolation(out, "without_19.9M", "law_extrapolation_alt")
