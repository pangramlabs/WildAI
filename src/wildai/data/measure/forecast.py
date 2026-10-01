"""Forecast the AI share of web tokens with a random walk with drift.

The model is fitted to the monthly shares from December 2022 (ChatGPT's first full month) on. Months without a crawl stay
missing: the series is unevenly spaced, so each step is compared with drift x its gap in months and scaled by sqrt(gap).

* drift = (last share - first share) / months elapsed;
* residual of a step over ``gap`` months = (change - drift x gap) / sqrt(gap), variance with len(residuals) - 1 dof;
* at horizon h after the last month: mean = last + drift x h, sd = sqrt(var x (h + h^2 / elapsed)) (the second term is
  the uncertainty of the drift), with Student-t 80% and 95% intervals, clipped to [0, 1].

    python -m wildai.data.measure.forecast            # reads and writes results/web/

Output: ``ai_share_forecast.csv`` with ``month, kind (observed | forecast), share, lo80, hi80, lo95, hi95``.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from scipy import stats

from wildai.data.measure.ai_share import DEFAULT_OUTPUT

FIT_START = "2022-12"
LAST_FORECAST = "2030-12"


class ForecastRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    month: str
    kind: str
    share: float
    lo80: float | None = None
    hi80: float | None = None
    lo95: float | None = None
    hi95: float | None = None


def month_index(month: str) -> int:
    return int(month[:4]) * 12 + int(month[5:7]) - 1


def month_label(index: int) -> str:
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def drift_forecast(observed: list[tuple[str, float]], last_month: str = LAST_FORECAST) -> list[ForecastRow]:
    """Monthly forecasts from the month after the last observation through ``last_month``."""

    times = [month_index(m) for m, _share in observed]
    shares = [share for _m, share in observed]
    elapsed = times[-1] - times[0]
    drift = (shares[-1] - shares[0]) / elapsed
    residuals = [((shares[i] - shares[i - 1]) - drift * (times[i] - times[i - 1])) / math.sqrt(times[i] - times[i - 1])
                 for i in range(1, len(shares))]
    dof = len(residuals) - 1
    variance = sum(e * e for e in residuals) / dof
    q80, q95 = stats.t.ppf(0.90, dof), stats.t.ppf(0.975, dof)

    def clip(value: float) -> float:
        return min(max(value, 0.0), 1.0)

    rows = []
    for index in range(times[-1] + 1, month_index(last_month) + 1):
        h = index - times[-1]
        mean, sd = shares[-1] + drift * h, math.sqrt(variance * (h + h * h / elapsed))
        rows.append(ForecastRow(month=month_label(index), kind="forecast", share=clip(mean), lo80=clip(mean - q80 * sd),
                                hi80=clip(mean + q80 * sd), lo95=clip(mean - q95 * sd), hi95=clip(mean + q95 * sd)))
    return rows


def forecast_table(monthly: list[tuple[str, float]], fit_start: str = FIT_START) -> list[ForecastRow]:
    fitted = [(m, s) for m, s in sorted(monthly) if m >= fit_start]
    return [ForecastRow(month=m, kind="observed", share=s) for m, s in fitted] + drift_forecast(fitted)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--monthly-csv", type=Path, default=DEFAULT_OUTPUT / "monthly_ai_share.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    with args.monthly_csv.open(encoding="utf-8") as handle:
        monthly = [(r["month"], float(r["ai_share"])) for r in csv.DictReader(handle)]
    rows = forecast_table(monthly)
    path = args.output_dir / "ai_share_forecast.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(ForecastRow.model_fields))
        writer.writerows([r.month, r.kind, r.share, *("" if v is None else v for v in (r.lo80, r.hi80, r.lo95, r.hi95))]
                         for r in rows)
    december = {r.month: r for r in rows if r.month.endswith("-12") and r.kind == "forecast"}
    for month, row in december.items():
        print(f"{month}: {100 * row.share:.1f}% (95% {100 * row.lo95:.1f} to {100 * row.hi95:.1f})")


if __name__ == "__main__":
    main()
