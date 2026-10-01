"""Fixtures for the data-pipeline tests."""

from __future__ import annotations

import pytest

from tests.data.builders import WordCounter


@pytest.fixture
def counter() -> WordCounter:
    return WordCounter()
