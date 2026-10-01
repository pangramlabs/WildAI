"""Evaluation sets, AI-share measurement, forecast, topic/format counts and filter survival."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.data.builders import WordCounter, write
from wildai.data.config import REPO_ROOT
from wildai.data.evalsets.common import MANIFEST, SHARD, take_chars, write_eval_set
from wildai.data.evalsets.fw22 import dump_budgets
from wildai.data.evalsets.fw26 import select_tail
from wildai.data.measure.ai_share import bootstrap_interval, month_share, monthly_shares
from wildai.data.measure.filter_survival import FINEWEB_STAGES, survival
from wildai.data.measure.forecast import drift_forecast, forecast_table
from wildai.data.measure.topic_format import count
from wildai.data.monthly_sample import MonthlyFrame, MonthlySampleConfig

WEB_RESULTS = REPO_ROOT / "results" / "web"


def test_take_chars_and_budgets() -> None:
    texts = ["x" * 50, "y" * 5, "z" * 60, "w" * 70]
    assert list(take_chars(texts, target_chars=100, min_chars=10)) == ["x" * 50, "z" * 60]
    assert dump_budgets(["b", "a", "c"], 10) == {"a": 4, "b": 3, "c": 3}


def test_write_eval_set(tmp_path: Path) -> None:
    manifest = write_eval_set(tmp_path, "paloma_x", ["one two", "three"], source="s", revision="r", selection="all",
                              domains=["d1", "d1"], counter=WordCounter())
    assert (manifest.documents, manifest.characters, manifest.gpt2_tokens, manifest.domains) == (2, 12, 3, {"d1": 2})
    assert pq.read_table(tmp_path / "paloma_x" / SHARD).column_names == ["text", "domain"]
    assert json.loads((tmp_path / "paloma_x" / MANIFEST).read_text())["name"] == "paloma_x"
    with pytest.raises(FileExistsError):
        write_eval_set(tmp_path, "paloma_x", ["again"], source="s", revision=None, selection="", counter=WordCounter())


def test_fw26_tail_skips_training_documents(tmp_path: Path) -> None:
    natural = pa.table({"id": [f"n{i}" for i in range(10)], "text": ["t"] * 10, "label": ["human", "ai"] * 5,
                        "token_count": [10] * 10, "sampling_hash": pa.array(range(10), pa.uint64())})
    write(pa.table({"id": ["n9", "n7"]}), tmp_path / "pool" / "part-00000.parquet")  # in training
    chosen = select_tail(natural, target_tokens=25, pools=[tmp_path / "pool"])
    assert chosen["id"].to_pylist() == ["n5", "n6", "n8"]  # the tail minus n7 and n9, back in pool order
    with pytest.raises(ValueError):
        select_tail(natural, target_tokens=1_000, pools=[tmp_path / "pool"])


def test_bootstrap_interval_is_seeded() -> None:
    rng = np.random.default_rng(0)
    tokens = rng.integers(1, 1000, 500).astype(float)
    ai = tokens * (rng.random(500) < 0.2)
    low, high = bootstrap_interval(ai, tokens, seed=5)
    assert (low, high) == bootstrap_interval(ai, tokens, seed=5) != bootstrap_interval(ai, tokens, seed=6)
    assert low < ai.sum() / tokens.sum() < high


def test_month_share_counts() -> None:
    row = month_share("2026-01", "fineweb", ["A", "B", "A"], ["AI", "Human", "Mixed"], np.array([10, 30, 60]), seed=1)
    assert (row.documents, row.ai_documents, row.human_tokens, row.dumps) == (3, 1, 30, "A;B")
    assert row.ai_share == pytest.approx(0.1) and row.ai_or_mixed_share == pytest.approx(0.7)


def test_monthly_shares_reads_month_files(tmp_path: Path) -> None:
    ids = [f"d{i}" for i in range(20)]
    write(pa.table({"id": ids, "dump": ["CC-MAIN-2026-30"] * 20, "token_count": list(range(1, 21))}), tmp_path / "s" / "2026-07.parquet")
    labels = ["AI" if i % 4 == 0 else "Human" for i in range(20)]
    labels[1] = None  # an unlabeled document is left out
    write(pa.table({"id": ids, "pangram_label": pa.array(labels, pa.string())}), tmp_path / "p" / "2026-07.parquet")
    config = MonthlySampleConfig(frames=[MonthlyFrame(source="common_crawl_random_warc", seed=20260918,
                                                      months={"2026-07": ["CC-MAIN-2026-30"], "2026-08": ["CC-MAIN-2026-34"]})])
    (row,) = monthly_shares(tmp_path / "s", tmp_path / "p", config)
    assert row.documents == 19 and row.source == "common_crawl_random_warc"
    assert row.ai_tokens == sum(i + 1 for i in range(0, 20, 4))


def test_drift_forecast_on_a_straight_line() -> None:
    observed = [("2023-01", 0.1), ("2023-02", 0.2), ("2023-04", 0.4), ("2023-05", 0.5)]
    rows = drift_forecast(observed, last_month="2023-07")
    assert [r.month for r in rows] == ["2023-06", "2023-07"]
    assert rows[0].share == pytest.approx(0.6) and rows[0].lo95 == pytest.approx(0.6)  # no noise, no width


@pytest.mark.skipif(not (WEB_RESULTS / "ai_share_forecast.csv").exists(), reason="published web results not present")
def test_forecast_reproduces_the_published_file() -> None:
    with (WEB_RESULTS / "monthly_ai_share.csv").open() as handle:
        monthly = [(r["month"], float(r["ai_share"])) for r in csv.DictReader(handle)]
    with (WEB_RESULTS / "ai_share_forecast.csv").open() as handle:
        published = list(csv.DictReader(handle))
    ours = forecast_table(monthly)
    assert [(r.month, r.kind) for r in ours] == [(r["month"], r["kind"]) for r in published]
    for mine, theirs in zip(ours, published):
        for field in ("share", "lo80", "hi80", "lo95", "hi95"):
            value = getattr(mine, field)
            assert (value is None and theirs[field] == "") or abs(value - float(theirs[field])) < 1e-12


def test_topic_format_count_groups_and_skips_unlabeled() -> None:
    table = pa.table({"period": ["2026-01"] * 3, "topic": ["Health"] * 3, "format": ["Tutorial"] * 3,
                      "label": ["AI", "AI", None], "token_count": [5, 7, 9]})
    (cell,) = count([table])
    assert (cell.label, cell.documents, cell.tokens) == ("AI", 2, 12)


def test_survival_is_cumulative_per_label() -> None:
    table = pa.table({"label": ["AI", "AI", "Human", "Human", None],
                      "english": [True, True, True, False, True], "gopher_repetition": [True, False, True, True, True],
                      "gopher_quality": [True] * 5, "c4": [True] * 5, "fineweb_quality": [True] * 5})
    result = survival(table, FINEWEB_STAGES, "trafilatura")
    assert result.labels == {"Human": 2, "Mixed": 0, "AI": 2}
    stages = {s.stage: s for s in result.stages}
    assert stages["Sample"].kept == 5 and stages["English"].kept == 4 and stages["Gopher repetition"].kept == 3
    assert stages["Gopher repetition"].ai == 50.0 and stages["Gopher repetition"].human == 50.0
    assert stages["English"].mixed is None  # no Mixed documents in this sample
