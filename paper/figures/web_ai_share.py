"""The AI share of web tokens, measured each crawl month and forecast to 2030 (appendix figure and table).

The forecast (a random walk with drift fitted from December 2022, with Student-t prediction intervals) is computed by
`wildai.data.forecast` into results/web/ai_share_forecast.csv; this module only draws it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from paper.results import RESULTS, monthly_ai_share
from paper.style import AI, INK, MUTED, RULE, TEXT_WIDTH, save_figure
from paper.tables.layout import write_table

FIT_START = "2022-12"
LABELED_DECEMBERS = ("2028-12", "2030-12")


@dataclass(frozen=True)
class Forecast:
    month: str
    share: float
    lo80: float
    hi80: float
    lo95: float
    hi95: float


def forecast() -> dict[str, Forecast]:
    with (RESULTS / "web/ai_share_forecast.csv").open(encoding="utf-8") as handle:
        return {r["month"]: Forecast(r["month"], *(float(r[k]) for k in ("share", "lo80", "hi80", "lo95", "hi95")))
                for r in csv.DictReader(handle) if r["kind"] == "forecast"}


def _date(month: str) -> datetime:
    return datetime(int(month[:4]), int(month[5:7]), 15)


def forecast_figure(out: Path) -> Path:
    series = [(m.month, m.ai_share) for m in monthly_ai_share()]
    ahead = sorted(forecast().values(), key=lambda f: f.month)
    fig, ax = plt.subplots(figsize=(0.72 * TEXT_WIDTH, 2.4), layout="constrained")
    ax.plot([_date(m) for m, _ in series], [100 * s for _, s in series], color=AI, lw=1.3, marker="o", ms=1.8, zorder=4)
    future = [_date(series[-1][0])] + [_date(f.month) for f in ahead]
    last = 100 * series[-1][1]
    ax.fill_between(future, [last] + [100 * f.lo95 for f in ahead], [last] + [100 * f.hi95 for f in ahead], color=AI, alpha=0.10, lw=0, zorder=1)
    ax.fill_between(future, [last] + [100 * f.lo80 for f in ahead], [last] + [100 * f.hi80 for f in ahead], color=AI, alpha=0.18, lw=0, zorder=2)
    ax.plot(future, [last] + [100 * f.share for f in ahead], color=AI, lw=1.2, ls=(0, (3, 2)), zorder=3)
    ax.axvline(_date(FIT_START), color=RULE, lw=0.7, zorder=0)
    ax.text(_date(FIT_START), 97, " fit starts\n December 2022", fontsize=6.0, color=MUTED, ha="left", va="top")
    by_month = {f.month: f for f in ahead}
    for month in LABELED_DECEMBERS:
        f = by_month[month]
        ax.annotate(f"{100 * f.share:.0f}%", xy=(_date(month), 100 * f.share), xytext=(0, 5), textcoords="offset points", fontsize=6.2, color=INK, ha="center", va="bottom")
        ax.scatter([_date(month)], [100 * f.share], s=11, color=INK, zorder=6)
    ax.set_ylim(0, 100)
    ax.set_xlim(datetime(2021, 1, 1), datetime(2031, 3, 1))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlabel("crawl date")
    ax.set_ylabel("AI share of web tokens")
    ax.set_title("AI Share of the Web, Measured and Forecast")
    ax.tick_params(labelsize=6.5)
    handles = [Line2D([], [], color=AI, lw=1.3, marker="o", ms=1.8, label="measured, 5,000 documents a month"),
               Line2D([], [], color=AI, lw=1.2, ls=(0, (3, 2)), label="random walk with drift"),
               Patch(color=AI, alpha=0.28, lw=0, label="80% interval"), Patch(color=AI, alpha=0.10, lw=0, label="95% interval")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.80), fontsize=6.0, handlelength=2.0, handletextpad=0.5, borderaxespad=0.3, labelspacing=0.3)
    return save_figure(fig, out, "web_ai_share_forecast")


def forecast_table(out: Path) -> Path:
    by_month = forecast()
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{6pt}", r"\begin{tabular}{lrrr}", r"\toprule",
             r"December & Forecast & 80\% interval & 95\% interval \\", r"\midrule"]
    for year in range(2026, 2031):
        f = by_month[f"{year}-12"]
        lines.append(f"{year} & {100 * f.share:.1f}\\% & {100 * f.lo80:.1f} to {100 * f.hi80:.1f}\\% & {100 * f.lo95:.1f} to {100 * f.hi95:.1f}\\%" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Forecast AI share of web tokens each December. The intervals are Student-$t$ prediction intervals that include the uncertainty of the drift.}",
              r"\label{tab:ai-share-forecast}", r"\end{table}", ""]
    return write_table(out, "ai_share_forecast_table", "\n".join(lines))
