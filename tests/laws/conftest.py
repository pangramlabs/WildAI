"""Shared fixtures of the scaling-law tests: the released runs, the expected printed numbers and our law's C4 fit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wildai.laws.data import Study
from wildai.laws.fit import LawFit, fit
from wildai.laws.ours import paper_law

GOLDEN = Path(__file__).with_name("golden.json")


@pytest.fixture(scope="session")
def study() -> Study:
    return Study.load()


@pytest.fixture(scope="session")
def golden() -> dict:
    """The expected printed numbers of the paper's law tables."""
    return json.loads(GOLDEN.read_text())


@pytest.fixture(scope="session")
def c4_fit(study: Study) -> LawFit:
    return fit(paper_law(), study.observations("c4", study.split("fit")))
