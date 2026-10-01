"""The text the models write (appendix): AI-typical phrase rates of the held-out 973M models against r (figure), and
phrase rates and Pangram labels of the 268M generations (table)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from paper.groups import matched_budget
from paper.results import PhraseRate, ai_phrase_rates, controls, generation_labels, models
from paper.style import MUTED, PALETTE, R_LABEL, TEXT_WIDTH, budget_label, format_ratio, save_figure
from paper.tables.layout import write_table

PANELS = (("writingprompts", "Story Continuations (WritingPrompts)"), ("webtext", "Web-Article Continuations (WebText)"))
FIGURE_BUDGETS = dict(zip((18.88, 37.76), PALETTE.phrase_budgets))  # 973M groups drawn, by human budget
TABLE_DEPTH = 16


def _ladders(depth: int, prompts: str) -> dict[float, list[tuple[float, PhraseRate]]]:
    """Human budget -> [(design r, rate)] for the groups of one size, control first."""

    ladders: dict[float, list[tuple[float, PhraseRate]]] = defaultdict(list)
    for rate in ai_phrase_rates():
        model = models()[rate.name]
        budget = matched_budget(controls()[model.group])
        if model.depth == depth and rate.prompts == prompts and budget is not None:
            ladders[budget].append((model.added_ratio or 0.0, rate))
    return {b: sorted(v, key=lambda t: t[0]) for b, v in ladders.items()}


def phrase_figure(out: Path) -> Path:
    """AI phrases per thousand generated words against r with 95 % prompt-bootstrap intervals; each budget's human-only
    model is a dashed line with its interval shaded."""

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.3), layout="constrained", sharey=True)
    for ax, (prompts, title) in zip(axes, PANELS):
        ladders = _ladders(26, prompts)
        for budget, color in FIGURE_BUDGETS.items():
            (_, control), *treated = ladders[budget]
            ax.axhspan(control.low, control.high, color=color, alpha=0.13, lw=0)
            ax.axhline(control.rate, color=color, lw=0.9, ls=(0, (4, 2)))
            x, y = [r for r, _ in treated], [p.rate for _, p in treated]
            errors = [[p.rate - p.low for _, p in treated], [p.high - p.rate for _, p in treated]]
            ax.errorbar(x, y, yerr=errors, color=color, lw=1.2, elinewidth=0.8, capsize=1.6, marker="o", ms=3.6, mec="white", mew=0.5, label=budget_label(budget))
        ax.set_xscale("log")
        ax.set_xlim(0.04, 1.25)
        ax.xaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.25, 0.5, 1.0]))
        ax.xaxis.set_major_formatter(FuncFormatter(format_ratio))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", length=0)
        ax.set_ylim(0, None)
        ax.set_xlabel(R_LABEL)
        ax.set_title(title)
    axes[0].set_ylabel("AI phrases per 1,000 words")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=MUTED, lw=0.9, ls=(0, (4, 2))))
    labels.append("same budget, $r = 0$ (band: 95 % interval)")
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=6.5, columnspacing=1.4)
    return save_figure(fig, out, "ai_phrase_dose_response")


def generation_table(out: Path) -> Path:
    low, high = sorted(_ladders(TABLE_DEPTH, "writingprompts"))
    rate = {(budget, r): p.rate for budget in (low, high) for r, p in _ladders(TABLE_DEPTH, "writingprompts")[budget]}
    labeled = {models()[g.name].added_ratio or 0.0: g for g in generation_labels() if matched_budget(controls()[models()[g.name].group]) == high}
    rows = []
    for r in sorted({r for _budget, r in rate}):
        cells = [f"{rate[(budget, r)]:.2f}" for budget in (low, high)]
        g = labeled.get(r)
        cells.append(f"{g.ai_pct:.1f} [{g.ai_pct_low:.1f}, {g.ai_pct_high:.1f}]" if g else "--")
        rows.append(f"{r:g} & {100 * r / (1 + r):.1f} & " + " & ".join(cells) + r" \\")
    lo_label, hi_label = budget_label(low), budget_label(high)
    tex = "\n".join([
        r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{6pt}", r"\begin{tabular}{rrccc}", r"\toprule",
        r" & & \multicolumn{2}{c}{AI phrases per 1,000 words} & Stories labeled AI (\%) \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-5}",
        f"$r$ & AI share (\\%) & {lo_label} & {hi_label} & {hi_label} \\\\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
        r"\caption{\textbf{The 268M models trained with more AI text write more like AI.} Story continuations of 500 WritingPrompts prompts "
        r"from the 268M models at 5 and 20 $TPP_h$, one model per $r$. AI-typical phrases per 1,000 words over eight continuations per prompt, "
        r"and the share of one continuation per prompt that Pangram~3.3.2 labels AI, with 95\% intervals from a bootstrap over prompts.}",
        r"\label{tab:generation-behavior}", r"\end{table}", ""])
    return write_table(out, "generation_behavior_table", tex)
