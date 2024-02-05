"""Adapters tests."""

from __future__ import annotations

import os

from simpml.tabular.adapters_pool import ManipulateAdapter

TEST_DATA_DIR: str = os.path.join("tests", "data")


def test_manipulate_adapter() -> None:
    """Test the `ManipulateAdapter` class."""
    assert hasattr(ManipulateAdapter, "manipulate")
