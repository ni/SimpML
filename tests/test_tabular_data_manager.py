"""Tabular data manager tests."""

from __future__ import annotations

import os
from typing import Any, Dict

from feature_engine.encoding import OneHotEncoder
from feature_engine.wrappers import SklearnTransformerWrapper
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import MinMaxScaler

TEST_DATA_DIR: str = os.path.join("tests", "data")

from simpml.core.base import Dataset, PredictionType
from simpml.tabular.adapters_pool import ManipulateAdapter
from simpml.tabular.data_fetcher_pool import TabularDataFetcher
from simpml.tabular.pipeline import Pipeline, TrainPipeline
from simpml.tabular.splitter_pool import RandomSplitter
from simpml.tabular.steps_pool import SmartImpute
from simpml.tabular.tabular_data_manager import TabularDataManager


def test_tabular_data_manager() -> None:
    """Test the `TabularDataManager` class."""
    kwargs_load_data: Dict[str, Any] = {
        "path": "docs/examples/datasets/binary/Titanic.csv",
        "target": "Survived",
        "drop_cols": ["PassengerId", "Name"],
    }
    my_data_fetcher = TabularDataFetcher(kwargs_load_data["path"])
    data = my_data_fetcher.get_items()
    data.drop(columns=kwargs_load_data["drop_cols"], inplace=True)
    kwargs_split_data: Dict[str, Any] = {
        "target": "Survived",
        "split_sets": {Dataset.Train: 0.6, Dataset.Valid: 0.2, Dataset.Test: 0.2},
        "random_state": 42,
    }
    my_splitter = RandomSplitter(**kwargs_split_data)

    smart_impute = SmartImpute()  # customized step
    one_hot_encoder = OneHotEncoder()  # feature engine step
    min_max_scalar = SklearnTransformerWrapper(
        transformer=MinMaxScaler()
    )  # sklearn step wrapped with feature engine
    ros_wrapped = ManipulateAdapter(SMOTE(), "fit_resample")  # imblearn step

    sklearn_pipeline = SklearnPipeline(
        steps=[
            ("impute", smart_impute),
            ("encoder", one_hot_encoder),
            ("MinMaxScaler", min_max_scalar),
        ]
    )
    train_pipeline = TrainPipeline([("sampler", ros_wrapped)])

    my_pipeline = Pipeline(sklearn_pipeline=sklearn_pipeline, train_pipeline=train_pipeline)
    my_data_manager = TabularDataManager(
        data_fetcher=my_data_fetcher,
        splitter=my_splitter,
        pipeline=my_pipeline,
        prediction_type=PredictionType.BinaryClassification,
    )

    my_data_manager.load_and_split_data()
    indents = my_splitter.split(data)

    X_train, y_train = my_data_manager.get_training_data()

    assert y_train is not None
    assert len(indents) == 3
    assert len(y_train) == len(X_train)
    assert len(indents[Dataset.Valid]) < len(indents[Dataset.Train])
    assert len(indents[Dataset.Test]) < len(indents[Dataset.Train])

    X_valid, y_valid = my_data_manager.get_validation_data()

    assert y_valid is not None
    assert len(X_valid) == len(y_valid)
