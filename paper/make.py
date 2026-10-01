"""Render every figure and table of the paper from `results/`.

    python -m paper.make                         # conference palette, into paper/out/
    python -m paper.make --palette pangram --out paper/out/preprint
    python -m paper.make --only web_filter_sankey downstream_core

Figures are written as PDF (plus a PNG preview) under <out>/figures, tables as .tex under <out>/tables, with the file
names the LaTeX source includes.
"""

from __future__ import annotations

import argparse
import importlib
import os
import time
from pathlib import Path

# name -> (module, function, kind); every function takes the output directory and returns the written path
OUTPUTS: dict[str, tuple[str, str, str]] = {
    "web_ai_share_forecast": ("paper.figures.web_ai_share", "forecast_figure", "figures"),
    "ai_share_forecast_table": ("paper.figures.web_ai_share", "forecast_table", "tables"),
    "web_formats_filters": ("paper.figures.web_filters", "formats_and_filters", "figures"),
    "web_filter_sankey": ("paper.figures.web_filters", "filter_sankey", "figures"),
    "web_ai_topic_format_pangram": ("paper.figures.web_topics", "pool_heatmap", "figures"),
    "web_topic_format_over_time_pangram": ("paper.figures.web_topics", "over_time", "figures"),
    "downstream_core": ("paper.figures.downstream", "core_changes", "figures"),
    "ai_phrase_dose_response": ("paper.figures.generation", "phrase_figure", "figures"),
    "generation_behavior_table": ("paper.figures.generation", "generation_table", "tables"),
    "budget_convention_table": ("paper.tables.runs", "budget_table", "tables"),
    "runs_table": ("paper.tables.runs", "runs_table", "tables"),
    "run_inventory_table": ("paper.tables.runs", "inventory_table", "tables"),
    "architecture_table": ("paper.tables.runs", "architecture_table", "tables"),
    "law_overview": ("paper.figures.law_overview", "overview", "figures"),
    "law_dose_response": ("paper.figures.law_ratio", "dose_response", "figures"),
    "law_dose_response_cosmopedia": ("paper.figures.law_ratio", "dose_response_cosmopedia", "figures"),
    "law_dose_ladder_all": ("paper.figures.law_ratio", "dose_ladder_all", "figures"),
    "law_harm_per_doubling": ("paper.figures.law_ratio", "harm_per_doubling", "figures"),
    "law_validation": ("paper.figures.law_validation", "validation", "figures"),
    "law_token_value_calibration": ("paper.figures.law_validation", "token_value_calibration", "figures"),
    "law_anatomy": ("paper.figures.law_anatomy", "anatomy", "figures"),
    "style_cd_window": ("paper.figures.law_anatomy", "cd_window", "figures"),
    "style_cd_effective_data_collapse": ("paper.figures.law_anatomy", "cd_effective_data_collapse", "figures"),
    "law_benchmark": ("paper.figures.law_benchmark", "benchmark", "figures"),
    "law_extrapolation": ("paper.figures.law_benchmark", "extrapolation", "figures"),
    "law_extrapolation_alt": ("paper.figures.law_benchmark", "extrapolation_alt", "figures"),
    "law_ceg_filtering": ("paper.figures.law_filtering", "ceg_filtering", "figures"),
    "law_repetition_targets": ("paper.figures.law_filtering", "repetition_targets", "figures"),
    "law_ceg_ladder": ("paper.figures.law_filtering", "ceg_ladder", "figures"),
    "law_when_ai": ("paper.figures.evaluation_masking", "when_ai", "figures"),
    "ai_text_evaluation": ("paper.figures.evaluation_masking", "ai_text_evaluation", "figures"),
    "law_results_table": ("paper.tables.laws", "results_table", "tables"),
    "law_results_table_float": ("paper.tables.laws", "results_table_wide", "tables"),
    "law_results_ai_targets_table": ("paper.tables.laws", "ai_targets_table", "tables"),
    "law_benchmark_absolute_table": ("paper.tables.laws", "absolute_table", "tables"),
    "law_results_lowdose_table": ("paper.tables.laws", "low_ratio_table", "tables"),
    "law_coefficient_table": ("paper.tables.laws", "coefficient_table", "tables"),
    "law_ablation_table": ("paper.tables.laws", "ablation_table", "tables"),
    "law_significance_table": ("paper.tables.laws", "significance_table", "tables"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--palette", choices=("tropical", "pangram"), default="tropical")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    parser.add_argument("--only", nargs="*", choices=sorted(OUTPUTS), help="render only these outputs")
    args = parser.parse_args()
    os.environ["PAPER_PALETTE"] = args.palette  # read by paper.style on first import
    from paper.style import apply_style

    apply_style()
    for name in args.only or OUTPUTS:
        module, function, kind = OUTPUTS[name]
        start = time.monotonic()
        path = getattr(importlib.import_module(module), function)(args.out / kind)
        print(f"{name:40s} {time.monotonic() - start:5.1f}s  {path}")


if __name__ == "__main__":
    main()
