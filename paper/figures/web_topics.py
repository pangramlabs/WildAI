"""What AI-labeled web text is about and how it is written (WebOrganizer topics and formats).

- web_ai_topic_format_pangram: share of documents labeled AI in each topic x format cell of the training pool.
- web_topic_format_over_time_pangram: each format's and topic's share of the monthly sample's tokens by calendar
  quarter, and the share of those tokens labeled AI.
- format_mismatch_panel: formats over- and under-represented in AI-labeled tokens (left panel of Figure 2).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from paper.results import TopicFormatCount, topic_format_counts
from paper.style import AI, AI_SHARE_CMAP, HUMAN, INK, PALETTE, RULE, TEXT_WIDTH, save_figure

POOL_MIN_CELL = 200  # documents a topic x format cell needs to be drawn
UNLABELED = {"", "None", "unlabeled"}
MISMATCH_WINDOW = ("2026-01", "2026-06")  # months the format mix of Figure 2 is measured over
MISMATCH_ROWS = 5  # over- and under-represented categories drawn each way
MISMATCH_MIN_SHARE = 0.01  # a category must hold this share of AI- or of human-labeled tokens to be ranked
TREND_HIGHLIGHT_BY_VOLUME, TREND_HIGHLIGHT_BY_AI = 6, 4
TREND_MIN_DOCUMENTS = 150  # a category needs this many documents in the last quarter to be ranked by AI share


def _labeled(counts: list[TopicFormatCount]) -> list[TopicFormatCount]:
    return [c for c in counts if c.topic not in UNLABELED and c.format not in UNLABELED]


def pool_heatmap(out: Path) -> Path:
    counts = _labeled(topic_format_counts("pool"))
    docs: dict[tuple[str, str], int] = defaultdict(int)
    ai: dict[tuple[str, str], int] = defaultdict(int)
    for c in counts:
        docs[(c.topic, c.format)] += c.documents
        if c.label == "AI":
            ai[(c.topic, c.format)] += c.documents

    def ai_share_of(index: int, category: str) -> float:
        keys = [k for k in docs if k[index] == category]
        return sum(ai[k] for k in keys) / sum(docs[k] for k in keys)

    topics = sorted({t for t, _ in docs}, key=lambda t: -ai_share_of(0, t))
    formats = sorted({f for _, f in docs}, key=lambda f: -ai_share_of(1, f))
    grid = np.full((len(formats), len(topics)), np.nan)  # formats down the side, topics across
    for (t, f), n in docs.items():
        if n >= POOL_MIN_CELL:
            grid[formats.index(f), topics.index(t)] = 100.0 * ai[(t, f)] / n
    vmax = float(np.nanpercentile(grid, 98))
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 3.55))
    mesh = ax.pcolormesh(np.ma.masked_invalid(grid), cmap=AI_SHARE_CMAP, vmin=0, vmax=vmax, edgecolors="white", linewidth=0.5)
    ax.set_facecolor("#F1EFED")
    for i in range(len(formats)):
        for j in range(len(topics)):
            if not np.isnan(grid[i, j]):
                ax.text(j + 0.5, i + 0.5, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=4.3, color="white" if grid[i, j] > 0.55 * vmax else INK)
    ax.set_xticks(np.arange(len(topics)) + 0.5)
    ax.set_xticklabels(topics, rotation=50, ha="right", va="top", fontsize=5.6, rotation_mode="anchor")
    ax.set_yticks(np.arange(len(formats)) + 0.5)
    ax.set_yticklabels(formats, fontsize=5.6)
    ax.tick_params(length=0, pad=1.5)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Topic", fontsize=7)
    ax.set_ylabel("Format", fontsize=7)
    ax.set_title("AI Share by Topic and Format", fontsize=7.5)
    bar = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.015)
    bar.set_label("AI (%)", fontsize=6.5)
    bar.ax.tick_params(labelsize=6)
    bar.outline.set_visible(False)
    fig.tight_layout()
    return save_figure(fig, out, "web_ai_topic_format_pangram")


def _quarter(month: str) -> str:
    return f"{month[:4]}Q{(int(month[5:7]) - 1) // 3 + 1}"


def _quarter_position(quarter: str) -> float:
    return int(quarter[:4]) + (int(quarter[5]) - 0.5) / 4


def over_time(out: Path) -> Path:
    counts = _labeled(topic_format_counts("monthly_sample"))
    quarters = sorted({_quarter(c.period) for c in counts})
    x = [_quarter_position(q) for q in quarters]
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 5.6))
    for row, axis in enumerate(("format", "topic")):
        tokens: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        ai_tokens: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        documents: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for c in counts:
            category, quarter = getattr(c, axis), _quarter(c.period)
            tokens[category][quarter] += c.tokens
            documents[category][quarter] += c.documents
            if c.label == "AI":
                ai_tokens[category][quarter] += c.tokens
        totals = {q: sum(tokens[c][q] for c in tokens) for q in quarters}
        share = {c: [100.0 * tokens[c][q] / totals[q] for q in quarters] for c in tokens}
        ai_share = {c: [100.0 * ai_tokens[c][q] / max(tokens[c][q], 1e-9) for q in quarters] for c in tokens}
        by_volume = sorted(tokens, key=lambda c: -share[c][-1])[:TREND_HIGHLIGHT_BY_VOLUME]
        big_enough = [c for c in tokens if documents[c][quarters[-1]] >= TREND_MIN_DOCUMENTS]
        by_ai = sorted(big_enough, key=lambda c: -ai_share[c][-1])[:TREND_HIGHLIGHT_BY_AI]
        highlight = list(dict.fromkeys(by_volume + by_ai))
        colors = {c: PALETTE.categorical[i % len(PALETTE.categorical)] for i, c in enumerate(highlight)}
        panels = ((share, "share of the quarter's tokens (%)", f"Each {axis}'s share of the web"),
                  (ai_share, "tokens labeled AI (%)", f"AI share within each {axis}"))
        for col, (series, ylabel, title) in enumerate(panels):
            ax = axes[row, col]
            for c in tokens:
                if c not in colors:
                    ax.plot(x, series[c], color=PALETTE.grey_light, lw=0.7, zorder=1)
            for c in highlight:
                ax.plot(x, series[c], color=colors[c], lw=1.5, zorder=3)
                ax.scatter(x, series[c], s=5, color=colors[c], zorder=4)
            _end_labels(ax, x, [(series[c][-1], c, colors[c]) for c in highlight])
            ax.set_xticks(range(int(x[0]), int(x[-1]) + 1))
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=8.5, loc="left")
            ax.grid(axis="y", color=RULE, lw=0.4)
    fig.tight_layout(h_pad=1.6, w_pad=1.2)
    return save_figure(fig, out, "web_topic_format_over_time_pangram")


def _end_labels(ax: plt.Axes, x: list[float], ends: list[tuple[float, str, str]]) -> None:
    """Name each highlighted line at its right end, nudging labels apart so they do not overlap."""

    dx = (x[-1] - x[0]) / max(len(x) - 1, 1)
    ymin, ymax = ax.get_ylim()
    ymax += 0.06 * (ymax - ymin)
    ax.set_ylim(ymin, ymax)
    gap = 0.065 * (ymax - ymin)  # about one label height at this panel size
    ends = sorted(ends)
    placed: list[float] = []
    for value, _name, _color in ends:
        placed.append(value if not placed or value - placed[-1] >= gap else placed[-1] + gap)
    overflow = placed[-1] + 0.5 * gap - ymax
    if overflow > 0:
        placed = [y - overflow for y in placed]
    for (value, name, color), y in zip(ends, placed):
        arrow = {"arrowstyle": "-", "color": color, "lw": 0.4, "shrinkA": 0, "shrinkB": 1} if abs(y - value) > 0.02 * (ymax - ymin) else None
        ax.annotate(name, xy=(x[-1], value), xytext=(x[-1] + 0.36 * dx, y), fontsize=6.0, color=color, va="center", arrowprops=arrow)
    ax.set_xlim(x[0] - 0.6 * dx, x[-1] + 7.5 * dx)


def _ratio_text(ratio: float) -> str:
    if ratio >= 1:
        return f"{ratio:.1f}×"
    inverse = 1 / ratio
    return f"1/{inverse:.0f}" if inverse >= 10 else f"1/{inverse:.1f}"


def format_shares() -> dict[str, tuple[float, float]]:
    """Format -> (share of human-labeled tokens, share of AI-labeled tokens) over the mismatch window."""

    tokens: dict[str, dict[str, int]] = {"Human": defaultdict(int), "AI": defaultdict(int)}
    for c in topic_format_counts("monthly_sample"):
        if MISMATCH_WINDOW[0] <= c.period <= MISMATCH_WINDOW[1] and c.label in tokens:
            tokens[c.label][c.format] += c.tokens
    totals = {label: sum(v.values()) for label, v in tokens.items()}
    return {f: (tokens["Human"][f] / totals["Human"], tokens["AI"][f] / totals["AI"]) for f in set(tokens["Human"]) | set(tokens["AI"])}


def format_mismatch_panel(ax: plt.Axes, fontsize: float = 6.3) -> None:
    """Log-scale bars of each format's AI-labeled share over its human-labeled share, the most over-represented on top."""

    kept = {f: (h, a) for f, (h, a) in format_shares().items() if max(h, a) >= MISMATCH_MIN_SHARE and min(h, a) > 0}
    ranked = sorted(kept, key=lambda f: kept[f][1] / kept[f][0])
    under = [f for f in ranked if kept[f][1] < kept[f][0]][:MISMATCH_ROWS]
    over = [f for f in ranked if kept[f][1] > kept[f][0]][-MISMATCH_ROWS:]
    rows = under + over
    for y, name in enumerate(rows):
        human, ai = kept[name]
        ratio = ai / human
        ax.barh(y, np.log2(ratio), color=AI if ratio > 1 else HUMAN, height=0.68, lw=0)
        ax.text(np.log2(ratio) + (0.12 if ratio > 1 else -0.12), y, _ratio_text(ratio), fontsize=fontsize - 0.5, color=INK,
                ha="left" if ratio > 1 else "right", va="center")
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=fontsize)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    reach = max(abs(np.log2(kept[f][1] / kept[f][0])) for f in rows) + 1.1
    ax.set_xlim(-reach - 0.5, reach)
    powers = range(-3, 4) if reach <= 5.5 else range(-6, 7, 2)  # every other power of two when the bars span a wide range
    ax.xaxis.set_major_locator(FixedLocator([t for t in powers if abs(t) < reach]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: "1" if v == 0 else (f"{2 ** v:g}×" if v > 0 else f"1/{2 ** -v:g}")))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelsize=fontsize)
