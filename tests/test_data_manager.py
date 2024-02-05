"""Model data helper tests."""

from __future__ import annotations

import os
from typing import List

import numpy as np
from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC
from imblearn.under_sampling import ClusterCentroids

from simpml.core.base import Dataset, PredictionType
from simpml.tabular.adapters_pool import ManipulateAdapter
from simpml.tabular.splitter_pool import RandomSplitter
from simpml.tabular.tabular_data_manager import SupervisedTabularDataManager

TEST_DATA_DIR: str = os.path.join("tests", "data")

def test_get_data() -> None:
    """Test getting data."""
    data_manager = SupervisedTabularDataManager(
        os.path.join(TEST_DATA_DIR, "train_titanic.csv"),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )
    assert (
        data_manager.get_training_data()[0].shape[1]
        == data_manager.get_validation_data()[0].shape[1]
    )


def test_random_state_same() -> None:
    """Test when the random state is the same between data managers."""
    data_manager = SupervisedTabularDataManager(
        os.path.join(TEST_DATA_DIR, "train_titanic.csv"),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )

    data_manager_2 = SupervisedTabularDataManager(
        os.path.join(TEST_DATA_DIR, "train_titanic.csv"),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )

    assert (
        data_manager.get_training_data()[0]["Age"].iloc[0]
        == data_manager_2.get_training_data()[0]["Age"].iloc[0]
    )


def test_random_state_different() -> None:
    """Test when the random state is different between data managers."""
    splitter = RandomSplitter(target="Survived", random_state=5)
    data_manager = SupervisedTabularDataManager(
        os.path.join(TEST_DATA_DIR, "train_titanic.csv"),
        target="Survived",
        splitter=splitter,
        prediction_type=PredictionType.BinaryClassification,
    )

    data_manager_2 = SupervisedTabularDataManager(
        os.path.join(TEST_DATA_DIR, "train_titanic.csv"),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )

    assert (
        data_manager.get_training_data()[0]["Age"].iloc[0]
        != data_manager_2.get_training_data()[0]["Age"].iloc[0]
    )


def test_description() -> None:
    """Test the description functionality."""
    data_manager = SupervisedTabularDataManager(
        os.path.join(TEST_DATA_DIR, "train_titanic.csv"),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )
    data_manager.set_description("Test")

    assert data_manager.description == "Test"


def test_clone() -> None:
    """Test the clone functionality."""
    splitter = RandomSplitter(target="Survived", random_state=5)
    data_manager = SupervisedTabularDataManager(
        os.path.join(TEST_DATA_DIR, "train_titanic.csv"),
        target="Survived",
        splitter=splitter,
        prediction_type=PredictionType.BinaryClassification,
    )
    data_manager_2 = data_manager.clone()
    assert data_manager_2.id != data_manager.id


def test_undersampling_balance_method() -> None:
    """Test the undersampling balance method."""
    data_manager = SupervisedTabularDataManager(
        os.path.join(TEST_DATA_DIR, "train_titanic.csv"),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )
    pipeline = data_manager.build_pipeline()
    data_manager.load_pipeline(pipeline)
    data_manager.load_and_split_data()
    X_train, _ = data_manager.get_training_data()
    original_size = X_train.shape[0]

    data_manager = SupervisedTabularDataManager(
        os.path.join(TEST_DATA_DIR, "train_titanic.csv"),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )
    pipeline = data_manager.build_pipeline()
    ros_wrapped = ManipulateAdapter(ClusterCentroids(), "fit_resample")  # imblearn step

    pipeline.add_train_step(("sampler", ros_wrapped))

    data_manager.load_pipeline(pipeline)
    data_manager.load_and_split_data()

    X_train, _ = data_manager.get_training_data()

    size_after_balance = X_train.shape[0]

    assert original_size > size_after_balance


def test_oversampling_balance_method_smote() -> None:
    """Test data set that has only numerical features."""
    data_manager = SupervisedTabularDataManager(
        os.path.join(TEST_DATA_DIR, "train_titanic.csv"),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )
    pipeline = data_manager.build_pipeline()
    data_manager.load_pipeline(pipeline)
    data_manager.load_and_split_data()
    X_train, _ = data_manager.get_training_data()
    original_size = X_train.shape[0]

    data_manager = SupervisedTabularDataManager(
        os.path.join(TEST_DATA_DIR, "train_titanic.csv"),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )
    pipeline = data_manager.build_pipeline()
    ros_wrapped = ManipulateAdapter(SMOTE(), "fit_resample")  # imblearn step

    pipeline.add_train_step(("sampler", ros_wrapped))

    data_manager.load_pipeline(pipeline)
    data_manager.load_and_split_data()

    X_train, _ = data_manager.get_training_data()

    size_after_balance = X_train.shape[0]

    assert original_size < size_after_balance


def test_oversampling_balance_method_smotenc() -> None:
    """Test data set that has both categorical and numerical features."""
    data_manager = SupervisedTabularDataManager(
        os.path.join(TEST_DATA_DIR, "train_titanic.csv"),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )
    pipeline = data_manager.build_pipeline()
    data_manager.load_pipeline(pipeline)
    data_manager.load_and_split_data()
    X_orig, y_orig = data_manager.get_dataset_from_indices(Dataset.Train)
    cat_cols = []
    for colname in X_orig.columns:
        dtype = X_orig[colname].dtype.name
        if dtype in ("category", "object", "int64"):
            cat_cols.append(colname)

    X_train, _ = data_manager.get_training_data()
    original_size = X_train.shape[0]

    cat_cols_indices: List[int] = []
    for idx, colname in enumerate(cat_cols):
        if colname in X_train.columns:
            cat_cols_indices.append(idx)

    cat_cols_indices_arr = np.array(cat_cols_indices, dtype=int)

    data_manager = SupervisedTabularDataManager(
        os.path.join(TEST_DATA_DIR, "train_titanic.csv"),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )
    pipeline = data_manager.build_pipeline()
    ros_wrapped = ManipulateAdapter(SMOTENC(cat_cols_indices_arr), "fit_resample")  # imblearn step

    pipeline.add_train_step(("sampler", ros_wrapped))

    data_manager.load_pipeline(pipeline)
    data_manager.load_and_split_data()

    X_train, _ = data_manager.get_training_data()

    size_after_balance = X_train.shape[0]

    assert original_size < size_after_balance


def test_oversampling_balance_method_smoten() -> None:
    """Test data set that has only categorical features."""
    data_manager = SupervisedTabularDataManager(
        os.path.join(TEST_DATA_DIR, "train_titanic.csv"),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )
    pipeline = data_manager.build_pipeline()
    data_manager.load_pipeline(pipeline)
    data_manager.load_and_split_data()
    X_train, _ = data_manager.get_training_data()
    original_size = X_train.shape[0]

    data_manager = SupervisedTabularDataManager(
        os.path.join(TEST_DATA_DIR, "train_titanic.csv"),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )
    pipeline = data_manager.build_pipeline()
    ros_wrapped = ManipulateAdapter(SMOTEN(), "fit_resample")  # imblearn step

    pipeline.add_train_step(("sampler", ros_wrapped))

    data_manager.load_pipeline(pipeline)
    data_manager.load_and_split_data()

    X_train, _ = data_manager.get_training_data()

    size_after_balance = X_train.shape[0]

    assert original_size < size_after_balance
