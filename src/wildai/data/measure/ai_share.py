"""Monthly AI share of web tokens: GPT-2 tokens of documents Pangram labels AI over tokens of all labeled documents.

Documents are weighted by their GPT-2 token count and carry their whole-document label (AI and Mixed are reported
separately). The 95% interval resamples the month's documents with replacement 2,000 times, seeded by the month's seed
(frame seed + ``YYYYMM``, see :mod:`wildai.data.monthly_sample`); it reflects sampling error only, not detector error.
Draws index documents in file order, so the same month file gives the same interval.

    python -m wildai.data.measure.ai_share --sample-dir data/monthly_sample --pangram-dir data/labels/pangram/monthly_sample

Output: ``monthly_ai_share.csv`` in ``--output-dir`` (default ``results/web``).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from wildai.data.config import REPO_ROOT, default_config, load_config
from wildai.data.monthly_sample import MonthlySampleConfig, month_seed
from wildai.data.parquet_io import read_joined

LABELS = ("Human", "Mixed", "AI")
BOOTSTRAP_DRAWS = 2000
DEFAULT_OUTPUT = REPO_ROOT / "results" / "web"


class MonthlyShare(BaseModel):
    """One row of ``monthly_ai_share.csv``."""

    model_config = ConfigDict(frozen=True)

    month: str
    dumps: str
    source: str
    documents: int
    human_documents: int
    mixed_documents: int
    ai_documents: int
    tokens: int
    human_tokens: int
    mixed_tokens: int
    ai_tokens: int
    ai_share: float
    ai_or_mixed_share: float
    ai_share_lo95: float
    ai_share_hi95: float


def bootstrap_interval(numerator: np.ndarray, denominator: np.ndarray, seed: int, draws: int = BOOTSTRAP_DRAWS) -> tuple[float, float]:
    """95% percentile interval of sum(numerator) / sum(denominator) over documents resampled with replacement."""

    rng = np.random.default_rng(seed)
    n = len(numerator)
    ratios = np.empty(draws)
    for i in range(draws):
        index = rng.integers(n, size=n)
        ratios[i] = numerator[index].sum() / denominator[index].sum()
    low, high = np.quantile(ratios, [0.025, 0.975])
    return float(low), float(high)


def month_share(month: str, source: str, dumps: list[str], labels: list[str], tokens: np.ndarray, seed: int) -> MonthlyShare:
    labels_array = np.array(labels)
    docs = {lab: int((labels_array == lab).sum()) for lab in LABELS}
    toks = {lab: int(tokens[labels_array == lab].sum()) for lab in LABELS}
    total = int(tokens.sum())
    ai = tokens * (labels_array == "AI")
    low, high = bootstrap_interval(ai.astype(float), tokens.astype(float), seed)
    return MonthlyShare(month=month, dumps=";".join(sorted(set(dumps))), source=source, documents=len(labels),
                        human_documents=docs["Human"], mixed_documents=docs["Mixed"], ai_documents=docs["AI"], tokens=total,
                        human_tokens=toks["Human"], mixed_tokens=toks["Mixed"], ai_tokens=toks["AI"],
                        ai_share=toks["AI"] / total, ai_or_mixed_share=(toks["AI"] + toks["Mixed"]) / total,
                        ai_share_lo95=low, ai_share_hi95=high)


def monthly_shares(sample_dir: Path, pangram_dir: Path, config: MonthlySampleConfig) -> list[MonthlyShare]:
    rows = []
    for month in config.all_months():
        path = sample_dir / f"{month}.parquet"
        if not path.exists():
            continue
        frame, _dumps = config.month(month)
        table = read_joined(path, {"pangram": pangram_dir}, columns=["id", "dump", "token_count"])
        labels = table["pangram_label"].to_pylist()
        keep = [i for i, lab in enumerate(labels) if lab is not None]
        if len(keep) < len(labels):
            print(f"{month}: {len(labels) - len(keep)} documents without a Pangram label are left out")
        table = table.take(keep)
        rows.append(month_share(month, frame.source, table["dump"].to_pylist(), table["pangram_label"].to_pylist(),
                                table["token_count"].to_numpy(), month_seed(frame.seed, month)))
    return rows


def write_csv(rows: list[BaseModel], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(type(rows[0]).model_fields))
        writer.writeheader()
        writer.writerows(row.model_dump() for row in rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=default_config("monthly_sample.yaml"))
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--pangram-dir", type=Path, required=True, help="Pangram sidecars of the monthly sample")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    rows = monthly_shares(args.sample_dir, args.pangram_dir, load_config(args.config, MonthlySampleConfig))
    write_csv(rows, args.output_dir / "monthly_ai_share.csv")
    for row in rows:
        print(f"{row.month}: AI {100 * row.ai_share:.2f}% [{100 * row.ai_share_lo95:.2f}, {100 * row.ai_share_hi95:.2f}]")


if __name__ == "__main__":
    main()
