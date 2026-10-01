"""How AI-labeled documents fare in quality filters, from the filter audit of one 2026 crawl.

- web_formats_filters (Figure 2): formats over-represented in AI text (left) and survival through FineWeb's filters (right).
- web_filter_sankey (appendix): survival through the FineWeb and DCLM pipelines, with Mixed documents.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

from paper.figures.web_topics import format_mismatch_panel
from paper.results import Stage, filter_audit
from paper.style import INK, LABEL_COLORS, MUTED, TEXT_WIDTH, save_figure

GAP = 0.12  # vertical gap between the ribbons of two labels


def _ribbon(ax: plt.Axes, x0: float, x1: float, lo0: float, hi0: float, lo1: float, hi1: float, color: str) -> None:
    """A smooth band from the interval [lo0, hi0] at x0 to [lo1, hi1] at x1."""

    dx = 0.42 * (x1 - x0)
    verts = [(x0, hi0), (x0 + dx, hi0), (x1 - dx, hi1), (x1, hi1), (x1, lo1), (x1 - dx, lo1), (x0 + dx, lo0), (x0, lo0), (x0, hi0)]
    codes = [MplPath.MOVETO, *[MplPath.CURVE4] * 3, MplPath.LINETO, *[MplPath.CURVE4] * 3, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=0.85, zorder=2))


def sankey_panel(ax: plt.Axes, stages: list[Stage], totals: dict[str, int], order: tuple[str, ...], *, fontsize: float = 6.3,
                 rotation: float = 25, end_label: str = "{lab}: {pct:.1f}% survive", end_room: float = 1.9, gap: float = GAP) -> None:
    """Each label's documents as a ribbon through one pipeline, every ribbon starting at full height so survival rates
    compare directly; documents kept are written above each stage and survival at the end."""

    alive = [[getattr(s, lab) / 100.0 for lab in order] for s in stages]
    xs = np.arange(len(stages), dtype=float)
    for i in range(len(stages) - 1):
        for j, lab in enumerate(order):
            base = j * (1.0 + gap)
            _ribbon(ax, xs[i], xs[i + 1], base, base + alive[i][j], base, base + alive[i + 1][j], LABEL_COLORS[lab])
    top = len(order) * (1.0 + gap)
    for i, stage in enumerate(stages):
        for j in range(len(order)):
            base = j * (1.0 + gap)
            ax.plot([xs[i], xs[i]], [base, base + alive[i][j]], color=INK, lw=2.0, solid_capstyle="butt", zorder=3)
        ax.text(xs[i], top - gap + 0.06, f"{stage.kept:,}", fontsize=fontsize - 0.1, color=MUTED, ha="center", va="bottom")
    for j, lab in enumerate(order):
        middle = j * (1.0 + gap) + 0.5
        ax.text(xs[-1] + 0.15, middle, end_label.format(lab=lab, pct=getattr(stages[-1], lab)), fontsize=fontsize + 0.1,
                color=LABEL_COLORS[lab], ha="left", va="center", fontweight="bold")
        ax.text(-0.15, middle, f"{lab}\n{totals[lab]:,}", fontsize=fontsize - 0.1, color=LABEL_COLORS[lab], ha="right", va="center")
    ax.set_xlim(-0.9, len(stages) - 1 + end_room)
    ax.set_ylim(-0.05, top - gap + 0.35)
    ax.set_xticks(xs)
    if rotation:
        ax.set_xticklabels([s.stage for s in stages], rotation=rotation, ha="right", va="top", fontsize=fontsize, rotation_mode="anchor")
    else:  # upright, two-line stage names
        ax.set_xticklabels([s.stage.replace(" ", "\n", 1) for s in stages], ha="center", va="top", fontsize=fontsize, linespacing=1.1)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def formats_and_filters(out: Path) -> Path:
    fineweb = filter_audit()["FineWeb"]
    fig, (formats, filters) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.25), layout="constrained", gridspec_kw={"width_ratios": [1, 1.9]})
    format_mismatch_panel(formats)
    formats.set_xlabel("share in AI ÷ share in human text")
    formats.set_title("Formats of AI Text")
    sankey_panel(filters, fineweb.stages, fineweb.labels, ("AI", "Human"), rotation=0, end_label="{pct:.1f}%\nsurvive", end_room=1.3, gap=0.3)
    filters.set_title("Surviving FineWeb's Filters")
    return save_figure(fig, out, "web_formats_filters")


def filter_sankey(out: Path) -> Path:
    pipelines = filter_audit()
    fig, axes = plt.subplots(len(pipelines), 1, figsize=(TEXT_WIDTH, 2.0 * len(pipelines) + 0.4), gridspec_kw={"hspace": 0.75})
    for ax, (name, pipeline) in zip(np.atleast_1d(axes), pipelines.items()):
        sankey_panel(ax, pipeline.stages, pipeline.labels, ("AI", "Mixed", "Human"))
        ax.set_title(f"{name} pipeline", loc="left")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return save_figure(fig, out, "web_filter_sankey")
