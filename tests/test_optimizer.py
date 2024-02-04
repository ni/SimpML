"""Hyper-parameter optimizer tests."""

from __future__ import annotations

import os
import sys

import pandas as pd

ROOT_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

TEST_DATA_DIR: str = os.path.join(ROOT_PATH, "tests", "data")

from simpml.core.base import HyperParamsOptimizationLevel, MetricName, PredictionType
from simpml.core.data_set import DataSet
from simpml.core.experiment_manager import ExperimentManager
from simpml.tabular.hyperparameters_optimizer import SupervisedTabularOptimizer
from simpml.tabular.tabular_data_manager import SupervisedTabularDataManager


def test_tabular_optimizer() -> None:
    """Test the supervised tabular optimizer case."""
    data_manager = SupervisedTabularDataManager(
        data=DataSet.load_titanic_dataset(),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )
    data_manager.build_pipeline(drop_cols=["PassengerId"])
    hyper_parameters_optimizer = SupervisedTabularOptimizer(
        iters=10, cv=2, optimization_level=HyperParamsOptimizationLevel.Fast
    )
    optimizer_models = hyper_parameters_optimizer.get_optimizer_models()
    assert isinstance(optimizer_models, list)
    assert isinstance(optimizer_models[0], str)
    df = hyper_parameters_optimizer.get_model_params_df("LGBMClassifier")
    assert isinstance(df, pd.DataFrame)
    df.loc[
        (df["optimization_level"] == "Fast")
        & (df["hyperparameter_name"] == "learning_rate")
        & (df["param"] == "high"),
        "value",
    ] = 0.6
    hyper_parameters_optimizer.set_params(df, model_name="LGBMClassifier")
    df = hyper_parameters_optimizer.get_params_df()
    assert isinstance(df, pd.DataFrame)
    hyper_parameters_optimizer.set_params(df)
    hyper_parameters_optimizer.restore_params()
    exp_manager = ExperimentManager(
        data_manager,
        optimize_metric=MetricName.AUC,
        hyper_parameters_optimizer=hyper_parameters_optimizer,
    )
    exp_manager.run_experiment()
