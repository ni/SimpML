"""Data fetcher tests."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

ROOT_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

TEST_DATA_DIR: str = os.path.join("tests", "data")

from simpml.tabular.data_fetcher_pool import TabularDataFetcher


def test_tabular_data_fetcher() -> None:
    """Test the `TabularDataFetcher` class."""
    kwargs_load_data: Dict[str, Any] = {
        "path": os.path.join(TEST_DATA_DIR, "Titanic.csv"),
        "target": "Survived",
        "drop_cols": ["PassengerId", "Name"],
    }
    my_data_fetcher = TabularDataFetcher(kwargs_load_data["path"])
    data = my_data_fetcher.get_items()
    # data.rename(columns={kwargs_load_data["target"]: "target"}, inplace=True)
    data.drop(columns=kwargs_load_data["drop_cols"], inplace=True)

    assert "Name" not in data.columns and "PassengerId" not in data.columns
