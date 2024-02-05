"""Data fetcher tests."""

from __future__ import annotations

import os
from typing import Any, Dict

from simpml.tabular.data_fetcher_pool import TabularDataFetcher

TEST_DATA_DIR: str = os.path.join("tests", "data")


def test_tabular_data_fetcher() -> None:
    """Test the `TabularDataFetcher` class."""
    kwargs_load_data: Dict[str, Any] = {
        "path": os.path("docs/examples/datasets/binary/Titanic.csv"),
        "target": "Survived",
        "drop_cols": ["PassengerId", "Name"],
    }
    my_data_fetcher = TabularDataFetcher(kwargs_load_data["path"])
    data = my_data_fetcher.get_items()
    # data.rename(columns={kwargs_load_data["target"]: "target"}, inplace=True)
    data.drop(columns=kwargs_load_data["drop_cols"], inplace=True)

    assert "Name" not in data.columns and "PassengerId" not in data.columns
