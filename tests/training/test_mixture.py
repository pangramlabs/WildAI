"""The additive design: exact human-document identity across arms, exact budgets, determinism, no repeats."""

from __future__ import annotations

import numpy as np
import pytest

from tests.training.helpers import SMALL_RECIPE
from wildai.training.mixture import AI, HUMAN, WEB_AI_SHARE_2026, Arm, DocumentPlan, Pools, RunSpec, order_keys, plan_run

BASE_STEPS = 60
ROW_LEN = SMALL_RECIPE.sequence_len + 1
STEP_TOKENS = SMALL_RECIPE.rows_per_step(4) * ROW_LEN


def plan(pools: Pools, arm: Arm, steps: int = BASE_STEPS, seed: int = 1337, human_steps: int = BASE_STEPS) -> DocumentPlan:
    return plan_run(RunSpec(name=f"{arm.value}-{steps}", depth=4, arm=arm, human_steps=human_steps, steps=steps, seed=seed), pools, SMALL_RECIPE)


def docs(p: DocumentPlan, pool: int) -> dict[int, int]:
    mask = p.pool == pool
    return dict(zip(p.row[mask].tolist(), p.length[mask].tolist()))


def test_ai_run_trains_on_exactly_the_controls_human_documents(pools: Pools) -> None:
    control = plan(pools, Arm.CONTROL)
    ai = plan(pools, Arm.AI, steps=90)
    assert docs(ai, HUMAN) == docs(control, HUMAN)
    # ... in the same relative order
    assert np.array_equal(ai.row[ai.pool == HUMAN], control.row)


def test_budgets_are_exact(pools: Pools) -> None:
    control = plan(pools, Arm.CONTROL)
    ai = plan(pools, Arm.AI, steps=90)
    assert control.length.sum() == BASE_STEPS * STEP_TOKENS
    assert ai.length[ai.pool == HUMAN].sum() == BASE_STEPS * STEP_TOKENS
    assert ai.length[ai.pool == AI].sum() == 30 * STEP_TOKENS  # r = 0.5
    assert ai.length.max() <= ROW_LEN  # a document contributes at most one row
    summary = ai.summary(SMALL_RECIPE.sequence_len)
    assert summary.ai_tokens * 2 == summary.human_tokens
    assert summary.trained_tokens == 90 * SMALL_RECIPE.batch_tokens(4)


def test_selection_is_a_prefix_of_the_pool_order(pools: Pools) -> None:
    ai = plan(pools, Arm.AI, steps=90)
    for pool in (HUMAN, AI):
        rows = np.sort(ai.row[ai.pool == pool])
        assert np.array_equal(rows, np.arange(len(rows)))
    keys = pools.human.keys
    assert np.all(keys[1:] > keys[:-1])  # stores are sorted by key


def test_fresh_human_run_adds_the_next_human_documents(pools: Pools) -> None:
    control = plan(pools, Arm.CONTROL)
    fresh = plan(pools, Arm.HUMAN, steps=90)
    assert (fresh.pool == HUMAN).all()
    control_rows = set(control.row.tolist())
    added = [r for r in fresh.row.tolist() if r not in control_rows]
    assert min(added) == max(control_rows) + 1
    assert fresh.length.sum() == 90 * STEP_TOKENS
    assert {r: n for r, n in docs(fresh, HUMAN).items() if r in control_rows} == docs(control, HUMAN)


def test_no_document_is_used_twice(pools: Pools) -> None:
    for arm, steps in [(Arm.CONTROL, BASE_STEPS), (Arm.AI, 120), (Arm.HUMAN, 75)]:
        p = plan(pools, arm, steps)
        pairs = p.pool.astype(np.int64) * 10**9 + p.row
        assert len(np.unique(pairs)) == len(pairs)


def test_plans_are_deterministic_and_the_seed_only_changes_the_order(pools: Pools) -> None:
    first, again = plan(pools, Arm.AI, steps=90), plan(pools, Arm.AI, steps=90)
    for name in ("pool", "row", "length", "ai"):
        assert np.array_equal(getattr(first, name), getattr(again, name))
    other = plan(pools, Arm.AI, steps=90, seed=7)
    assert not np.array_equal(first.row, other.row)
    assert sorted(zip(first.pool.tolist(), first.row.tolist())) == sorted(zip(other.pool.tolist(), other.row.tolist()))


def test_order_keys_depend_on_seed_and_epoch() -> None:
    keys = np.arange(1000, dtype=np.uint64)
    assert not np.array_equal(order_keys(keys, 1), order_keys(keys, 2))
    assert not np.array_equal(order_keys(keys, 1, 0), order_keys(keys, 1, 1))
    assert np.array_equal(order_keys(keys, 1), order_keys(keys, 1, 0))


def test_repeat_run_repeats_the_controls_documents(pools: Pools) -> None:
    control = plan(pools, Arm.CONTROL)
    repeat = plan(pools, Arm.REPEAT, steps=150)  # 2.5 passes
    assert repeat.length.sum() == 150 * STEP_TOKENS
    assert set(repeat.row.tolist()) == set(control.row.tolist())
    first_pass = repeat.row[: len(control.row)]
    assert np.array_equal(first_pass, control.row)  # the first pass is the control's order
    counts = np.bincount(repeat.row)
    assert set(counts[control.row].tolist()) <= {2, 3}
    assert repeat.summary(SMALL_RECIPE.sequence_len).passes == 2.5


def test_filtering_pair_is_a_rate_matched_mix_and_its_human_part(pools: Pools) -> None:
    tpp = 90 * SMALL_RECIPE.batch_tokens(4) / SMALL_RECIPE.param_counts(4).paper_n  # a 90-step web mix
    web_mix = plan_run(RunSpec.from_budget(4, Arm.NATURAL, tpp, 1337, SMALL_RECIPE), pools, SMALL_RECIPE)
    filtered = plan_run(RunSpec.from_budget(4, Arm.FILTERED, tpp, 1337, SMALL_RECIPE), pools, SMALL_RECIPE)
    control = plan(pools, Arm.CONTROL, steps=filtered.steps, human_steps=filtered.steps)
    total = web_mix.steps
    assert web_mix.spec.human_steps == filtered.steps == round((1 - WEB_AI_SHARE_2026) * total)
    summary = web_mix.summary(SMALL_RECIPE.sequence_len)
    assert summary.ai_token_share == (total - filtered.steps) / total
    assert abs(summary.ai_token_share - WEB_AI_SHARE_2026) <= 0.5 / total
    # the filtered run trains on exactly the web mix's human documents, in the same order, and nothing else
    assert not filtered.ai.any() and docs(filtered, HUMAN) == docs(web_mix, HUMAN)
    assert np.array_equal(web_mix.row[web_mix.pool == HUMAN], filtered.row)
    # which is the human-only control of the same length
    assert np.array_equal(filtered.row, control.row) and np.array_equal(filtered.length, control.length)


def test_not_enough_data_is_an_error(pools: Pools) -> None:
    with pytest.raises(ValueError, match="pool 'ai'"):
        plan(pools, Arm.AI, steps=BASE_STEPS + 10_000)


def test_run_spec_validation() -> None:
    with pytest.raises(ValueError):
        RunSpec(name="x", depth=4, arm=Arm.AI, human_steps=10, steps=10)
    with pytest.raises(ValueError):
        RunSpec(name="x", depth=4, arm=Arm.FILTERED, human_steps=8, steps=10)
    with pytest.raises(ValueError):
        RunSpec(name="x", depth=4, arm=Arm.NATURAL, human_steps=10, steps=10)
    spec = RunSpec.from_budget(depth=4, arm=Arm.AI, tpp=3.0, seed=1, recipe=SMALL_RECIPE, ratio=0.5)
    n = SMALL_RECIPE.param_counts(4).paper_n
    assert spec.human_steps == round(3.0 * n / 512)
    assert spec.steps == int(np.floor(1.5 * spec.human_steps + 0.5))
