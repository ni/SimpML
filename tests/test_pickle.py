"""Pickle related tests."""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
from typing import Dict, Optional, Tuple

import dill
import joblib
import numpy as np
import pandas as pd

ROOT_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from simpml.core.base import Dataset, PredictionType
from simpml.tabular.tabular_data_manager import SupervisedTabularDataManager

TEST_DATA_DIR: str = os.path.join("tests", "data")


def get_dataset() -> Dict[Dataset, Tuple[pd.DataFrame, Optional[pd.Series]]]:
    """Get a data set."""
    data_manager = SupervisedTabularDataManager(
        os.path.join(TEST_DATA_DIR, "train_titanic.csv"),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )
    pipeline = data_manager.build_pipeline()
    data_manager.load_pipeline(pipeline)
    data_manager.load_and_split_data()
    X_train, y_train = data_manager.get_training_data()
    X_valid, y_valid = data_manager.get_validation_data()
    X_test = data_manager.get_x(Dataset.Test)
    y_test = data_manager.get_y(Dataset.Test)
    dataset: Dict[Dataset, Tuple[pd.DataFrame, Optional[pd.Series]]] = {
        Dataset.Train: (X_train, y_train),
        Dataset.Valid: (X_valid, y_valid),
        Dataset.Test: (X_test, y_test),
    }

    assert list(dataset.keys()) == [Dataset.Train, Dataset.Valid, Dataset.Test]
    assert len(dataset[Dataset.Train][0]) > len(dataset[Dataset.Valid][0])
    return dataset


def ensure_datasets_equal(
    dataset1: Dict[Dataset, Tuple[pd.DataFrame, Optional[pd.Series]]],
    dataset2: Dict[Dataset, Tuple[pd.DataFrame, Optional[pd.Series]]],
) -> None:
    """Ensure that 2 datasets are equal."""
    assert list(dataset1.keys()) == [Dataset.Train, Dataset.Valid, Dataset.Test]
    assert list(dataset2.keys()) == [Dataset.Train, Dataset.Valid, Dataset.Test]
    for dataset_key in [Dataset.Train, Dataset.Valid, Dataset.Test]:
        X1 = dataset1[dataset_key][0]
        y1 = dataset1[dataset_key][1]
        X2 = dataset2[dataset_key][0]
        y2 = dataset2[dataset_key][1]
        assert y1 is not None
        assert y2 is not None
        assert isinstance(X1, pd.DataFrame)
        assert isinstance(X2, pd.DataFrame)
        assert X1.shape == X2.shape
        assert len(y1) == len(y2)
        assert np.allclose(X1, X2)
        assert np.allclose(y1, y2)


def test_dataset_pickle() -> None:
    """Test pickle on datasets."""
    orig_dataset = get_dataset()
    with tempfile.TemporaryDirectory() as tempdir:
        filepath = os.path.join(tempdir, "test_dataset_pickle.pkl")
        with open(filepath, "wb") as outfile:
            pickle.dump(orig_dataset, outfile)
        with open(filepath, "rb") as infile:
            pickle_dataset = pickle.load(infile)
    ensure_datasets_equal(orig_dataset, pickle_dataset)


def test_dataset_joblib() -> None:
    """Test joblib on datasets."""
    orig_dataset = get_dataset()
    with tempfile.TemporaryDirectory() as tempdir:
        filepath = os.path.join(tempdir, "test_dataset_joblib.pkl")
        with open(filepath, "wb") as outfile:
            joblib.dump(orig_dataset, outfile)
        with open(filepath, "rb") as infile:
            joblib_dataset = joblib.load(infile)
    ensure_datasets_equal(orig_dataset, joblib_dataset)


def test_dataset_dill() -> None:
    """Test dill on datasets."""
    orig_dataset = get_dataset()
    with tempfile.TemporaryDirectory() as tempdir:
        filepath = os.path.join(tempdir, "test_dataset_dill.pkl")
        with open(filepath, "wb") as outfile:
            dill.dump(orig_dataset, outfile)
        with open(filepath, "rb") as infile:
            dill_dataset = dill.load(infile)
    ensure_datasets_equal(orig_dataset, dill_dataset)
