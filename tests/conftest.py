"""Configuration for pytest."""

from __future__ import annotations

from typing import List

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add pytest options.

    Args:
        parser: The pytest option parser.
    """
    parser.addoption("--runslow", action="store_true", default=False, help="Run slow tests")


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    """Modify pytest tests (add skip conditions).

    Args:
        config: The pytest configuration.
        items: A list of pytest items.
    """
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="Need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
