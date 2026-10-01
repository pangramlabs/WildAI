"""Share of each Pangram label's documents still alive after each filter stage (``filter_audit.json``).

FineWeb: the audit's ``fineweb_stages.parquet`` (:mod:`wildai.data.measure.filter_audit`) joined with the Pangram labels
of its sample. DCLM: an optional per-document table with a ``label`` column and DCLM's cumulative stage flags (the
DCLM pipeline itself is not part of this repository; its per-document results are released with the audit).
Documents Pangram could not label count towards ``kept`` but towards no label.

    python -m wildai.data.measure.filter_survival --fineweb-dir data/filter_audit \
        --pangram-dir data/labels/pangram/filter_audit [--dclm-table data/filter_audit/dclm_stages.parquet]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from wildai.data.config import default_config, load_config
from wildai.data.measure.ai_share import DEFAULT_OUTPUT
from wildai.data.measure.filter_audit import FilterAuditConfig
from wildai.data.parquet_io import read_joined

LABELS = ("Human", "Mixed", "AI")
# (stage, the per-document columns that must all be true for a document to be alive after it)
FINEWEB_STAGES: list[tuple[str, list[str]]] = [
    ("English", ["english"]), ("Gopher repetition", ["gopher_repetition"]), ("Gopher quality", ["gopher_quality"]),
    ("C4 rules", ["c4"]), ("FineWeb quality", ["fineweb_quality"]),
]
DCLM_STAGES: list[tuple[str, list[str]]] = [
    ("URL", ["refinedweb_url_stage_pass"]),
    ("English", ["refinedweb_english_stage_pass"]),
    ("Length", ["refinedweb_page_length_stage_pass"]),
    ("Other heuristics", [f"refinedweb_{s}_stage_pass" for s in ("mean_word_length", "symbol_ratio", "bullet_ratio",
                                                                  "ellipsis_ratio", "alphabetic_word_ratio", "stop_words")]),
    ("Repetition", ["refinedweb_repetition_stage_pass"]),
    ("Line cleanup", ["refinedweb_line_cleanup_stage_pass"]),
    ("DCLM classifier", ["pass_dclm_baseline_no_global_dedup"]),
]


class StageSurvival(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    stage: str
    kept: int
    human: float | None = Field(alias="Human")
    mixed: float | None = Field(alias="Mixed")
    ai: float | None = Field(alias="AI")
    """Percent of the label's documents alive after the stage; ``None`` when the sample holds none of the label."""


class PipelineSurvival(BaseModel):
    model_config = ConfigDict(frozen=True)

    extraction: str
    labels: dict[str, int]
    stages: list[StageSurvival]


class FilterAudit(BaseModel):
    model_config = ConfigDict(frozen=True)

    crawl: str
    warc_files: int
    documents_per_pipeline: int
    pipelines: dict[str, PipelineSurvival]


def survival(table: pa.Table, stages: list[tuple[str, list[str]]], extraction: str) -> PipelineSurvival:
    """Cumulative survival by label; ``table`` has a ``label`` column (None = unlabeled) and the stages' columns."""

    labels = np.array([lab if lab is not None else "" for lab in table["label"].to_pylist()])
    totals = {lab: int((labels == lab).sum()) for lab in LABELS}
    alive = np.ones(len(table), dtype=bool)
    rows = [StageSurvival(stage="Sample", kept=len(table), Human=100.0, Mixed=100.0, AI=100.0)]
    for stage, columns in stages:
        for column in columns:
            alive &= np.array([bool(v) for v in table[column].to_pylist()])  # a missing flag counts as removed
        shares = {lab: 100.0 * float((alive & (labels == lab)).sum()) / totals[lab] if totals[lab] else None for lab in LABELS}
        rows.append(StageSurvival(stage=stage, kept=int(alive.sum()), **shares))
    return PipelineSurvival(extraction=extraction, labels=totals, stages=rows)


def fineweb_table(audit_dir: Path, pangram_dir: Path) -> pa.Table:
    stages = pq.read_table(audit_dir / "fineweb_stages.parquet")
    labels = read_joined(audit_dir / "sample" / "part-00000.parquet", {"pangram": pangram_dir / "sample"}, columns=["id"])
    if labels["id"].to_pylist() != stages["id"].to_pylist():
        raise RuntimeError("fineweb_stages.parquet is not aligned with the audit sample")
    return stages.append_column("label", labels["pangram_label"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=default_config("filter_audit.yaml"))
    parser.add_argument("--fineweb-dir", type=Path, required=True, help="output directory of filter_audit")
    parser.add_argument("--pangram-dir", type=Path, required=True, help="Pangram sidecars of the audit (with sample/)")
    parser.add_argument("--dclm-table", type=Path, help="per-document DCLM stage flags with a label column")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    config = load_config(args.config, FilterAuditConfig)
    pipelines = {"FineWeb": survival(fineweb_table(args.fineweb_dir, args.pangram_dir), FINEWEB_STAGES, "trafilatura")}
    if args.dclm_table:
        pipelines["DCLM"] = survival(pq.read_table(args.dclm_table), DCLM_STAGES, "resiliparse")
    audit = FilterAudit(crawl=config.dump, warc_files=config.warc_files, documents_per_pipeline=config.documents,
                        pipelines=pipelines)
    path = args.output_dir / "filter_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit.model_dump(by_alias=True), indent=1) + "\n")
    for name, pipeline in pipelines.items():
        final = pipeline.stages[-1]
        print(f"{name}: AI {final.ai:.1f}% and human {final.human:.1f}% survive ({final.kept:,} documents kept)")


if __name__ == "__main__":
    main()
