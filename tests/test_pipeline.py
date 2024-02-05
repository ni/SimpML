"""Pipeline tests."""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import pandas as pd
from feature_engine.encoding import OneHotEncoder
from feature_engine.wrappers import SklearnTransformerWrapper
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import MinMaxScaler

TEST_DATA_DIR: str = os.path.join("tests", "data")

from simpml.core.base import Dataset, PredictionType
from simpml.tabular.adapters_pool import ManipulateAdapter
from simpml.tabular.data_fetcher_pool import TabularDataFetcher
from simpml.tabular.pipeline import Pipeline
from simpml.tabular.splitter_pool import RandomSplitter
from simpml.tabular.tabular_data_manager import TabularDataManager


def test_pipeline() -> None:
    """Test the `Pipeline` class."""
    kwargs_load_data: Dict[str, Any] = {
        "path": os.path("docs/examples/datasets/binary/Titanic.csv"),
        "target": "Survived",
        "drop_cols": ["PassengerId", "Name"],
    }
    my_data_fetcher = TabularDataFetcher(data=kwargs_load_data["path"])
    data = my_data_fetcher.get_items()
    data.drop(columns=kwargs_load_data["drop_cols"], inplace=True)
    kwargs_split_data: Dict[str, Any] = {
        "target": "Survived",
        "split_sets": {Dataset.Train: 0.6, Dataset.Valid: 0.2, Dataset.Test: 0.2},
        "random_state": 42,
    }
    my_splitter = RandomSplitter(**kwargs_split_data)
    indices = my_splitter.split(data)
    train_data = my_data_fetcher.get_items().loc[indices[Dataset.Train]]
    X_train, y_train = train_data.drop(my_splitter.target, axis=1), train_data[my_splitter.target]
    valid_data = my_data_fetcher.get_items().loc[indices[Dataset.Valid]]
    X_valid, y_valid = valid_data.drop(my_splitter.target, axis=1), valid_data[my_splitter.target]
    assert X_train.shape[0] == y_train.shape[0]
    assert X_valid.shape[0] == y_valid.shape[0]
    assert X_train.shape[1] == X_valid.shape[1]
    imp_mean = SklearnTransformerWrapper(
        transformer=SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    )
    one_hot_encoder = OneHotEncoder()  # feature engine step
    min_max_scalar = SklearnTransformerWrapper(
        transformer=MinMaxScaler()
    )  # sklearn step wrapped with feature engine
    ros_wrapped = ManipulateAdapter(SMOTE(), "fit_resample")  # imblearn step

    sklearn_pipeline = SklearnPipeline(steps=[("impute", imp_mean)])
    my_pipeline = Pipeline(sklearn_pipeline=sklearn_pipeline)

    assert not pd.DataFrame(my_pipeline.fit_transform(X=X_train, y=y_train)).isnull().any().any()
    assert my_pipeline.sklearn_pipeline.steps[-1][0] == "impute"

    my_pipeline.add_sklearn_step(("encoder", one_hot_encoder))
    my_pipeline.add_sklearn_step(("MinMaxScaler", min_max_scalar))
    my_pipeline.add_train_step(("sampler", ros_wrapped))

    assert my_pipeline.train_pipeline is not None
    assert my_pipeline.train_pipeline.steps[-1][0] == "sampler"
    assert my_pipeline.sklearn_pipeline.steps[2][0] == "MinMaxScaler"
    assert len(
        my_pipeline.fit_transform(
            X=X_train,
            y=y_train,
        )
    ) == len(X_train)


def test_normalization() -> None:
    """Test that normalization step (MinMaxScaler) does not introduce None into dataset."""
    kwargs_load_data: Dict[str, Any] = {
        "path": os.path("docs/examples/datasets/binary/Titanic.csv"),
        "target": "Survived",
        "drop_cols": ["PassengerId", "Name"],
    }
    my_data_fetcher = TabularDataFetcher(data=kwargs_load_data["path"])
    data = my_data_fetcher.get_items()
    data.drop(columns=kwargs_load_data["drop_cols"], inplace=True)

    kwargs_split_data: Dict[str, Any] = {
        "target": "Survived",
        "split_sets": {Dataset.Train: 0.6, Dataset.Valid: 0.2, Dataset.Test: 0.2},
        "random_state": 42,
    }
    my_splitter = RandomSplitter(**kwargs_split_data)
    indices = my_splitter.split(data)
    train_data = my_data_fetcher.get_items().loc[indices[Dataset.Train]]
    initial_no_of_none_train = train_data.isna().sum().sum()
    X_train, y_train = train_data.drop(my_splitter.target, axis=1), train_data[my_splitter.target]
    valid_data = my_data_fetcher.get_items().loc[indices[Dataset.Valid]]
    initial_no_of_none_valid = valid_data.isna().sum().sum()
    X_valid, y_valid = valid_data.drop(my_splitter.target, axis=1), valid_data[my_splitter.target]

    min_max_scaler = SklearnTransformerWrapper(transformer=MinMaxScaler())

    sklearn_pipeline = SklearnPipeline(steps=[("MinMaxScaler", min_max_scaler)])
    my_pipeline = Pipeline(sklearn_pipeline=sklearn_pipeline)
    assert my_pipeline.sklearn_pipeline.steps[0][0] == "MinMaxScaler"

    no_of_none_after_normalization_train = (
        pd.DataFrame(my_pipeline.fit_transform(X=X_train, y=y_train)).isna().sum().sum()
    )
    no_of_none_after_normalization_valid = (
        pd.DataFrame(my_pipeline.fit_transform(X=X_valid, y=y_valid)).isna().sum().sum()
    )

    assert initial_no_of_none_train == no_of_none_after_normalization_train
    assert initial_no_of_none_valid == no_of_none_after_normalization_valid


def test_preserving_of_indexes_after_normalization() -> None:
    """Test that normalization step does not modify string indexes."""
    kwargs_load_data: Dict[str, Any] = {
        "path": os.path("docs/examples/datasets/binary/Titanic.csv"),
        "target": "Survived",
    }
    my_data_fetcher = TabularDataFetcher(data=kwargs_load_data["path"])
    data = my_data_fetcher.get_items()
    data.set_index("Name", inplace=True)
    assert isinstance(data.index[0], str)

    kwargs_split_data: Dict[str, Any] = {
        "target": "Survived",
        "split_sets": {Dataset.Train: 0.6, Dataset.Valid: 0.2, Dataset.Test: 0.2},
        "random_state": 42,
    }
    my_splitter = RandomSplitter(**kwargs_split_data)

    min_max_scaler = SklearnTransformerWrapper(transformer=MinMaxScaler())

    sklearn_pipeline = SklearnPipeline(
        steps=[
            ("MinMaxScaler", min_max_scaler),
        ]
    )

    my_pipeline = Pipeline(sklearn_pipeline=sklearn_pipeline)
    my_data_manager = TabularDataManager(
        data_fetcher=my_data_fetcher,
        splitter=my_splitter,
        pipeline=my_pipeline,
        prediction_type=PredictionType.BinaryClassification,
    )

    my_data_manager.load_and_split_data()
    indices = my_splitter.split(data)
    initial_train_indices = indices[Dataset.Train]
    X_train = my_data_manager.get_training_data()[0]
    train_indices_after_preprocessing = X_train.index

    assert all(initial_train_indices == train_indices_after_preprocessing)
