"""Data splitter tests."""
# mypy: ignore-errors
# flake8: noqa
from __future__ import annotations

import os
import sys
from typing import Any, Dict

ROOT_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from simpml.core.base import Dataset
from simpml.tabular.data_fetcher_pool import TabularDataFetcher
from simpml.tabular.splitter_pool import (
    DateTimeSplitter,
    GroupSplitter,
    IndexSplitter,
    RandomSplitter,
)

TEST_DATA_DIR: str = os.path.join(ROOT_PATH, "tests", "data")


def test_random_splitter() -> None:
    """Test the `RandomSplitter` class."""
    kwargs_load_data: Dict[str, Any] = {
        "path": os.path.join(ROOT_PATH, "docs/examples/datasets/binary/Titanic.csv"),
        "target": "Survived",
        "drop_cols": ["PassengerId", "Name"],
    }
    my_data_fetcher = TabularDataFetcher(kwargs_load_data["path"])
    data = my_data_fetcher.get_items()
    # data.rename(columns={kwargs_load_data["target"]: "target"}, inplace=True)
    data.drop(columns=kwargs_load_data["drop_cols"], inplace=True)
    kwargs_split_data: Dict[str, Any] = {
        "target": "Survived",
        "split_sets": {Dataset.Train: 0.6, Dataset.Valid: 0.2, Dataset.Test: 0.2},
        "random_state": 42,
    }
    my_splitter = RandomSplitter(**kwargs_split_data)
    indents = my_splitter.split(data)

    assert list(indents.keys()) == [Dataset.Train, Dataset.Valid, Dataset.Test]
    assert len(indents[Dataset.Train]) > len(indents[Dataset.Valid])


def test_group_splitter() -> None:
    """Test the `GroupSplitter` class."""
    kwargs_load_data: Dict[str, Any] = {
        "path": os.path.join(ROOT_PATH, "docs/examples/datasets/binary/Titanic.csv"),
        "target": "Survived",
        "drop_cols": ["PassengerId", "Name"],
    }
    my_data_fetcher = TabularDataFetcher(kwargs_load_data["path"])
    data = my_data_fetcher.get_items()
    # data.rename(columns={kwargs_load_data["target"]: "target"}, inplace=True)
    data.drop(columns=kwargs_load_data["drop_cols"], inplace=True)
    kwargs_split_data: Dict[str, Any] = {
        "group_columns": "Sex",
        "split_sets": {Dataset.Train: 0.6, Dataset.Valid: 0.4},
        "random_state": 42,
    }
    my_splitter = GroupSplitter(**kwargs_split_data)
    indents = my_splitter.split(data)

    assert list(indents.keys()) == [Dataset.Train, Dataset.Valid]
    # Check the unique groups in each split
    train_groups = data.loc[indents[Dataset.Train], 'Sex'].unique()
    valid_groups = data.loc[indents[Dataset.Valid], 'Sex'].unique()
    # Verify that the same group is not in both train and valid sets
    assert not set(train_groups).intersection(valid_groups)

def test_index_splitter() -> None:
    """Test the `IndexSplitter` class."""
    kwargs_load_data: Dict[str, Any] = {
        "path": os.path.join("docs/examples/datasets/binary/Titanic.csv"),
        "target": "Survived",
        "drop_cols": ["PassengerId", "Name"],
    }
    my_data_fetcher = TabularDataFetcher(kwargs_load_data["path"])
    data = my_data_fetcher.get_items()
    data.drop(columns=kwargs_load_data["drop_cols"], inplace=True)
    kwargs_split_data: Dict[str, Any] = {
        "split_sets": {
            Dataset.Train: list(range(0, len(data), 2)),
            Dataset.Valid: list(range(1, len(data), 2)),
        }
    }
    my_splitter = IndexSplitter(**kwargs_split_data)
    indents = my_splitter.split(data)

    assert list(indents.keys()) == [Dataset.Train, Dataset.Valid]


def test_datetime_splitter() -> None:
    """Test the `DateTimeSplitter` class."""
    kwargs_load_data: Dict[str, Any] = {
        "path": os.path.join(
            ROOT_PATH, "docs/examples/datasets/multiclass/Shelter Animal Outcomes.csv"
        ),
        "target": "OutcomeType",
        "drop_cols": [],
    }
    my_data_fetcher = TabularDataFetcher(kwargs_load_data["path"])
    data = my_data_fetcher.get_items()
    data.drop(columns=kwargs_load_data["drop_cols"], inplace=True)
    kwargs_split_data: Dict[str, Any] = {
        "target": "OutcomeType",
        "split_sets": {Dataset.Train: 0.6, Dataset.Valid: 0.2, Dataset.Test: 0.2},
        "time_column": "DateTime",
    }
    my_splitter = DateTimeSplitter(**kwargs_split_data)
    indents = my_splitter.split(data)

    train = len(indents[Dataset.Train])
    valid = len(indents[Dataset.Valid])
    test = len(indents[Dataset.Test])

    assert list(indents.keys()) == [Dataset.Train, Dataset.Valid, Dataset.Test]
    assert train == int(len(data.index) * 0.6)
    assert valid == int(len(data.index) * 0.2)
    assert test == len(data.index) - train - valid

    for i in range(len(indents[Dataset.Train]) - 1):
        assert (
            data.loc[indents[Dataset.Train][i]]["DateTime"]
            <= data.loc[indents[Dataset.Train][i + 1]]["DateTime"]
        )

    assert (
        data.loc[indents[Dataset.Train][len(indents[Dataset.Train]) - 1]]["DateTime"]
        <= data.loc[indents[Dataset.Valid][0]]["DateTime"]
    )

    for i in range(len(indents[Dataset.Valid]) - 1):
        assert (
            data.loc[indents[Dataset.Valid][i]]["DateTime"]
            <= data.loc[indents[Dataset.Valid][i + 1]]["DateTime"]
        )

    assert (
        data.loc[indents[Dataset.Valid][len(indents[Dataset.Valid]) - 1]]["DateTime"]
        <= data.loc[indents[Dataset.Test][0]]["DateTime"]
    )

    for i in range(len(indents[Dataset.Test]) - 1):
        assert (
            data.loc[indents[Dataset.Test][i]]["DateTime"]
            <= data.loc[indents[Dataset.Test][i + 1]]["DateTime"]
        )
