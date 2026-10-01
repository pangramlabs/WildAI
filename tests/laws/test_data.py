"""The released runs: cohort sizes, control pairing and coordinates."""

from __future__ import annotations

import numpy as np

from wildai.laws.data import Coordinates, Study


def test_cohort(study: Study) -> None:
    fitted, held_out = study.split("fit"), study.split("held_out")
    assert len(fitted) == 726 and len(held_out) == 74
    assert {x.depth for x in fitted} == {4, 6, 9, 12, 16} and {x.depth for x in held_out} == {20, 26}
    ai = [x for x in held_out if x.arm == "ai"]
    assert len(ai) == 51 and sum(x.ratio < 1 for x in ai) == 43
    assert len({x.group for x in ai}) == 10


def test_pairing(study: Study) -> None:
    obs = study.cohort("c4")
    for i, run in enumerate(obs.runs):
        control = obs.runs[obs.control[i]]
        assert control.arm == "control" and control.group == run.group
        assert (control.depth, control.seed, control.n_params) == (run.depth, run.seed, run.n_params)
        assert control.ai_tokens == 0
    assert np.all(obs.control[obs.is_control] == np.flatnonzero(obs.is_control))


def test_coordinates(study: Study) -> None:
    run = study.runs["268m-g076-ai-r0.5"]
    x = Coordinates.of_runs([run])
    assert x.n[0] == run.n_params / 1e8 and x.d[0] == run.human_tokens / 1e9
    assert x.r[0] == run.ai_tokens / run.human_tokens and x.t[0] == run.human_tokens / run.n_params / 20
