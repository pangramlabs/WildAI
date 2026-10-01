"""Shared look of the paper's figures: palettes, matplotlib style, sizes, axis helpers and saving.

Two palettes: `tropical` (the conference version: pink = AI, teal = human, golds for Mixed) and `pangram` (the preprint:
orange = AI, forest green = human, the same golds). Select one with the environment variable PAPER_PALETTE before the
figure modules are imported; `python -m paper.make --palette pangram` does this for you.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter


@dataclass(frozen=True)
class Palette:
    ai: str
    ai_line: str  # thin law curves
    human: str
    human_light: str
    human_mid: str
    mixed_fill: str  # Mixed label fills
    ochre: str  # secondary gold lines
    ink: str
    muted: str
    rule: str
    grey: str
    grey_light: str
    shukor: str  # the joint law's comparison curve
    ai_sizes: tuple[str, str, str, str, str]  # 19.9M ... 268M, light to dark
    human_sizes: tuple[str, str, str, str, str]
    budget_ramp: tuple[str, ...]  # TPP_h colour scale, light to dark
    diverging: tuple[str, ...]  # filtering helps (human end) ... hurts (AI end)
    loss_ramp: tuple[str, ...]
    ai_share_ramp: tuple[str, ...]  # 0 to 100 % AI
    held_out: tuple[str, str]  # 477M, 973M
    highlight: str  # background band behind our law's row
    ai_dark: str  # annotations about AI-heavy results
    ratio_ramp: tuple[str, ...]  # AI ratio colour scale, light to dark
    phrase_budgets: tuple[str, str]  # AI-phrase lines at 20 and 40 TPP_h
    categorical: tuple[str, ...]  # topic and format lines
    table_shade: tuple[str, str]  # benchmark-table cells for the human-text columns: best (dark) to worst (light)
    worse_fill: tuple[str, float]  # (colour, opacity) behind a change that reads as worse, e.g. higher validation loss
    better_fill: tuple[str, float]  # ... and one that reads as better
    band_fill: tuple[str, float]  # interquartile bands around a median line


TROPICAL = Palette(
    ai="#E85993", ai_line="#D4548A", human="#1F898A", human_light="#43C4C0", human_mid="#2FA7A5",
    mixed_fill="#E0A526", ochre="#C48A12", ink="#222222", muted="#6B6763", rule="#D1CDCA", grey="#8F8A86",
    grey_light="#C9C4C0", shukor="#6E4FB3",
    ai_sizes=("#F6A9CB", "#EE7FB0", "#E85993", "#BD3A75", "#7E2352"),
    human_sizes=("#9FE3E0", "#5FD0CC", "#2FA7A5", "#1F898A", "#125C5D"),
    budget_ramp=("#B8ECE9", "#43C4C0", "#1F898A", "#0F4A4B"),
    diverging=("#0F4A4B", "#43C4C0", "#FFFFFF", "#F08BB5", "#8E245C"),
    loss_ramp=("#F4F2F0", "#EBCF7A", "#43C4C0", "#1F898A", "#0F4A4B"),
    ai_share_ramp=("#FFFFFF", "#F9C6DC", "#E85993", "#8E245C"),
    held_out=("#1F898A", "#0F4A4B"),
    highlight="#E3F6F5", ai_dark="#B23A70",
    ratio_ramp=("#F9C6DC", "#E85993", "#9E2E65", "#4E1533"),
    phrase_budgets=("#F08BB5", "#A82C66"),
    categorical=("#E85993", "#1F898A", "#C48A12", "#7E2352", "#2E5A88", "#43C4C0", "#E07B39", "#C2185B", "#5C7A29", "#8A6BBE"),
    table_shade=("#43B5B2", "#F0F9F8"),
    worse_fill=("#E85993", 0.06), better_fill=("#1F898A", 0.06), band_fill=("#C9C4C0", 0.7),
)

# Brand colours: orange #FF6106, peach #FECAB9, light gray #E2E2E2, off-white #F8F7F5, soft yellow #FFE782, mustard #D17A01,
# dark brown #612700, lavender #F4BEFF, sky blue #AAD8F9, forest green #15502E, charcoal green #14201E.
PANGRAM = Palette(
    ai="#FF6106", ai_line="#E8580A", human="#15502E", human_light="#6FA383", human_mid="#2E7048",
    mixed_fill="#E0A526", ochre="#C48A12", ink="#14201E", muted="#5B6A64", rule="#D6D1C8", grey="#8F8A86",
    grey_light="#C9C4C0", shukor="#1D547C",  # the joint law in a deep sky blue, apart from our black curve
    ai_sizes=("#FFB08A", "#FF8A4C", "#FF6106", "#C24A06", "#612700"),
    human_sizes=("#9FCFAE", "#6AAA80", "#3F8A5C", "#15502E", "#0B2E1A"),
    budget_ramp=("#9FCFAE", "#5FA57A", "#2E7048", "#0B2E1A"),
    diverging=("#15502E", "#86B395", "#FFFFFF", "#AAD8F9", "#1D547C"),  # forest to sky blue (and a deeper shade of it)
    loss_ramp=("#F8F7F5", "#EBCF7A", "#86B395", "#2E7048", "#0B2E1A"),
    ai_share_ramp=("#FFFFFF", "#FECAB9", "#FF8A4C", "#E8580A"),
    held_out=("#2E7048", "#0B2E1A"),
    highlight="#E6F0E9", ai_dark="#E8580A",
    ratio_ramp=("#FECAB9", "#FF6106", "#B8470A", "#612700"),
    phrase_budgets=("#FF6106", "#612700"),
    categorical=("#FF6106", "#15502E", "#2F7BB5", "#D17A01", "#612700", "#9B6BC4", "#6AAA80", "#7FBCE6", "#F4A07A", "#8F8A86"),
    table_shade=("#7DB08E", "#F1F7F3"),
    worse_fill=("#FECAB9", 0.45), better_fill=("#AAD8F9", 0.35), band_fill=("#14201E", 0.10),
)

PALETTES = {"tropical": TROPICAL, "pangram": PANGRAM}
PALETTE = PALETTES[os.environ.get("PAPER_PALETTE", "tropical")]

AI, HUMAN, MIXED, INK, MUTED, RULE = PALETTE.ai, PALETTE.human, PALETTE.mixed_fill, PALETTE.ink, PALETTE.muted, PALETTE.rule
LABEL_COLORS = {"Human": HUMAN, "Mixed": MIXED, "AI": AI}
AI_SHARE_CMAP = LinearSegmentedColormap.from_list("ai_share", list(PALETTE.ai_share_ramp))

TEXT_WIDTH = 5.5  # inches: the conference template's text width

SIZE_LABEL = {4: "19.9M", 6: "35.8M", 9: "86.2M", 12: "135M", 16: "268M", 20: "477M", 26: "973M"}
R_LABEL = "$r$ (AI / human tokens)"
TPP_H_LABEL = "$TPP_h$ (human tokens / parameter)"


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig: plt.Figure, out: Path, stem: str) -> Path:
    """Write `<stem>.pdf` (for LaTeX) and a `<stem>.png` preview into `out`."""

    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{stem}.png", dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out / f"{stem}.pdf"


def budget_label(human_tpp: float) -> str:
    """A human budget as the figures label it: tokens per parameter rounded to the nearest 5, half up (18.9 -> 20)."""

    return f"{5 * math.floor(round(human_tpp, 1) / 5 + 0.5):g} $TPP_h$"


def format_ratio(value: float, _pos: int | None = None) -> str:
    return f"{value:g}×"


def format_params(value: float, _pos: int | None = None) -> str:
    return f"{value / 1e9:g}B" if value >= 1e9 else f"{value / 1e6:g}M"


def ratio_axis(ax: plt.Axes, lo: float = 1.5e-3, hi: float = 100.0, *, sparse: bool = False) -> None:
    """A log axis of the AI ratio r, labelled as multiples of the human corpus."""

    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    candidates = (1e-2, 1.0, 100.0) if sparse else (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    ax.xaxis.set_major_locator(FixedLocator([t for t in candidates if lo <= t <= hi]))
    ax.xaxis.set_major_formatter(FuncFormatter(format_ratio))
    ax.xaxis.set_minor_formatter(NullFormatter())


def zero_line(ax: plt.Axes) -> None:
    ax.axhline(0.0, color=RULE, lw=0.7, zorder=1)
