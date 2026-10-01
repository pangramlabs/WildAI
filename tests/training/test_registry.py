"""Paper model names map to runs of the right group."""

from __future__ import annotations

from wildai.training.mixture import WEB_AI_SHARE_2026, Arm
from wildai.training.registry import load_models, run_spec


def test_every_model_maps_to_a_run() -> None:
    models = load_models()
    specs = {name: run_spec(name, models) for name in models}
    assert len(specs) == len(models)
    for name, spec in specs.items():
        row = models[name]
        assert spec.depth == row.depth and spec.seed == row.seed
        assert spec.steps == row.steps


def test_group_members_share_the_controls_budget() -> None:
    models = load_models()
    spec = run_spec("268m-g072-ai-r0.5", models)
    assert spec.arm == Arm.AI and spec.human_steps == models["268m-g072-control"].steps == 2240 and spec.steps == 3360
    repeat = run_spec("19.9m-g010-repeat-r4", models)
    assert repeat.arm == Arm.REPEAT and repeat.human_steps == models["19.9m-g010-control"].steps


def test_filtering_pairs_are_rate_matched_mixtures() -> None:
    models = load_models()
    for row in models.values():
        if row.arm != Arm.NATURAL:
            continue
        web_mix = run_spec(row.name, models)
        filtered = run_spec(row.name.replace("-web-mix", "-ai-removed"), models)
        assert filtered.arm == Arm.FILTERED and filtered.steps == filtered.human_steps == web_mix.human_steps
        # the web mix's AI steps give the 2026 share; models.csv's token counts are the published runs' realized ones
        assert abs((web_mix.steps - web_mix.human_steps) / web_mix.steps - WEB_AI_SHARE_2026) < 0.002
        assert abs(row.ai_tokens / row.total_tokens - WEB_AI_SHARE_2026) < 0.01
