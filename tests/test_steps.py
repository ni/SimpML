"""Steps tests."""

from __future__ import annotations

import os
import sys

ROOT_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from simpml.tabular.steps_pool import SmartImpute

TEST_DATA_DIR: str = os.path.join(ROOT_PATH, "tests", "data")


def test_smart_impute() -> None:
    """Test the `SmartImpute` class."""
    assert hasattr(SmartImpute, "fit")
    assert hasattr(SmartImpute, "transform")
