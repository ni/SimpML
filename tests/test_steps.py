"""Steps tests."""

from __future__ import annotations

import os

from simpml.tabular.steps_pool import SmartImpute

TEST_DATA_DIR: str = os.path.join("tests", "data")


def test_smart_impute() -> None:
    """Test the `SmartImpute` class."""
    assert hasattr(SmartImpute, "fit")
    assert hasattr(SmartImpute, "transform")
