"""GPU tests run only when asked for with ``-m gpu``, so the default run never touches a GPU."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if "gpu" in (config.getoption("markexpr") or ""):
        return
    skip = pytest.mark.skip(reason="GPU test; run with -m gpu")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
