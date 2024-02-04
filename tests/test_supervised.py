"""Supervised tests."""

from __future__ import annotations

import os
import sys
from typing import cast

import pandas as pd
import plotly.graph_objects as go
import pytest

ROOT_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

TEST_DATA_DIR: str = os.path.join(ROOT_PATH, "tests", "data")

from simpml.core.base import Dataset, MetricName, PredictionType
from simpml.core.data_set import DataSet
from simpml.core.experiment_manager import ExperimentManager
from simpml.tabular.interpreter import (
    FeatureImportanceMethod,
    TabularInterpreterBinaryClassification,
    TabularInterpreterClassification,
    TabularInterpreterRegression,
)
from simpml.tabular.splitter_pool import RandomSplitter
from simpml.tabular.tabular_data_manager import SupervisedTabularDataManager


def test_binary_classification() -> None:
    """Test the supervised binary classification case."""
    data_manager = SupervisedTabularDataManager(
        data=DataSet.load_titanic_dataset(),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
        splitter=RandomSplitter(
            split_sets={Dataset.Train: 0.8, Dataset.Valid: 0.2}, target="Survived"
        ),
    )

    assert len(cast(RandomSplitter, data_manager.splitter).split_sets) == 2

    data_manager.build_pipeline(drop_cols=["PassengerId"])

    exp_manager = ExperimentManager(data_manager, optimize_metric=MetricName.AUC)
    exp_manager.display_models_pool()
    exp_manager.remove_models("Gradient Boosting")
    exp_manager.get_available_models_df()
    exp_manager.reset_models()
    exp_manager.display_metrics_pool()
    exp_manager.run_experiment(metrics_kwargs={"pos_label": 1})
    best_model = exp_manager.get_best_model()
    assert best_model
    exp_id = exp_manager.get_current_experiment_id()
    assert exp_id
    model = exp_manager.get_model(model_name="LightGBM", experiment_id=exp_id)
    assert model

    assert exp_manager.opt_metric is not None
    interpreter = TabularInterpreterBinaryClassification(
        model=best_model,
        data_manager=data_manager,
        opt_metric=exp_manager.opt_metric,
        pos_class={"pos_class": 1},
    )
    fig = interpreter.main_fig()
    assert isinstance(fig, go.Figure)


@pytest.mark.slow
def test_binary_classification_extended() -> None:
    """Test the supervised binary classification extended case."""
    data_manager = SupervisedTabularDataManager(
        data=DataSet.load_titanic_dataset(),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
        splitter=RandomSplitter(
            split_sets={Dataset.Train: 0.8, Dataset.Valid: 0.2}, target="Survived"
        ),
    )

    assert len(cast(RandomSplitter, data_manager.splitter).split_sets) == 2

    data_manager.build_pipeline(drop_cols=["PassengerId"])
    data_manager.build_pipeline(drop_cols=["PassengerId"], smote=True)

    exp_manager = ExperimentManager(data_manager, optimize_metric=MetricName.AUC)
    exp_manager.display_models_pool()
    exp_manager.remove_models("Gradient Boosting")
    exp_manager.get_available_models_df()
    exp_manager.reset_models()
    exp_manager.display_metrics_pool()
    exp_manager.run_experiment(metrics_kwargs={"pos_label": 1})
    best_model = exp_manager.get_best_model()
    assert best_model
    exp_id = exp_manager.get_current_experiment_id()
    assert exp_id
    model = exp_manager.get_model(model_name="LightGBM", experiment_id=exp_id)
    assert model

    assert exp_manager.opt_metric is not None
    interpreter = TabularInterpreterBinaryClassification(
        model=best_model,
        data_manager=data_manager,
        opt_metric=exp_manager.opt_metric,
        pos_class={"pos_class": 1},
    )

    # This will create a blocking pop-up, so skip for now.
    # interpreter.plot_summary_shap()
    X, y = data_manager.get_validation_data()
    assert X is not None
    assert y is not None
    # This will create a blocking pop-up, so skip for now.
    # interpreter.plot_summary_shap(X)
    # This will create a blocking pop-up, so skip for now.
    # interpreter.plot_shap_dependence("Age")
    fig = interpreter.plot_confusion_matrix()
    assert isinstance(fig, go.Figure)
    fig = interpreter.find_best_threshold()
    assert isinstance(fig, go.Figure)
    fig = interpreter.get_probability_plot()
    assert isinstance(fig, go.Figure)
    fig = interpreter.plot_roc_curve()
    assert isinstance(fig, go.Figure)
    fig = interpreter.plot_fpr_tpr_curve()
    assert isinstance(fig, go.Figure)
    fig = interpreter.plot_feature_importance(method=FeatureImportanceMethod.Shap)
    assert isinstance(fig, go.Figure)
    fig = interpreter.plot_feature_importance(method=FeatureImportanceMethod.Permutation)
    assert isinstance(fig, go.Figure)
    fig = interpreter.main_fig()
    assert isinstance(fig, go.Figure)

    df = interpreter.get_feature_importance(method=FeatureImportanceMethod.Shap)
    assert isinstance(df, pd.DataFrame)
    noisy_features = interpreter.get_noisy_features()
    assert isinstance(noisy_features, list)
    assert isinstance(noisy_features[0], str)
    is_leakage = interpreter.leakage_detector()
    assert is_leakage is True
    new_data_manager_without_noisy_features = data_manager.clone()
    new_data_manager_without_noisy_features.set_description("Noisy Features Dropped")
    new_data_manager_without_noisy_features.build_pipeline(
        drop_cols=["PassengerId"] + noisy_features, smote=False
    )
    exp_manager.set_new_data(new_data_manager_without_noisy_features)
    exp_manager.run_experiment()


def test_multiclass_classification() -> None:
    """Test the supervised multi-class classification case."""
    splitter = RandomSplitter(
        split_sets={Dataset.Train: 0.6, Dataset.Valid: 0.2, Dataset.Test: 0.2}, target="target"
    )
    data_manager = SupervisedTabularDataManager(
        data=DataSet.load_wine_dataset(),
        target="target",
        prediction_type=PredictionType.MulticlassClassification,
        splitter=splitter,
    )
    df = data_manager.data_fetcher.get_items()
    assert isinstance(df, pd.DataFrame)
    data_manager.build_pipeline()
    exp_manager = ExperimentManager(data_manager, optimize_metric=MetricName.Accuracy)
    exp_manager.display_models_pool()
    exp_manager.run_experiment()
    best_model = exp_manager.get_best_model()
    assert best_model
    assert exp_manager.opt_metric is not None
    interpreter = TabularInterpreterClassification(
        model=exp_manager.get_model(
            "Decision Tree", experiment_id=exp_manager.get_current_experiment_id()
        ),
        data_manager=data_manager,
        opt_metric=exp_manager.opt_metric,
        pos_class={"pos_class": 1},
    )
    fig = interpreter.get_label_density_plot()
    assert isinstance(fig, go.Figure)
    fig = interpreter.main_fig()
    assert isinstance(fig, go.Figure)


def test_regression() -> None:
    """Test the supervised regression case."""
    data_manager = SupervisedTabularDataManager(
        data=DataSet.load_fetch_california_housing_dataset(),
        target="MedHouseVal",
        splitter=RandomSplitter(
            split_sets={Dataset.Train: 0.8, Dataset.Valid: 0.2},
            target="MedHouseVal",
            stratify=False,
        ),
        prediction_type=PredictionType.Regression,
    )
    data_manager.build_pipeline()

    exp_manager = ExperimentManager(data_manager, optimize_metric=MetricName.MSE)
    exp_manager.run_experiment()

    assert exp_manager.opt_metric is not None
    interpreter = TabularInterpreterRegression(
        exp_manager.get_model("XGBoost", exp_manager.get_current_experiment_id()),
        data_manager,
        exp_manager.opt_metric,
    )
    fig = interpreter.main_fig()
    assert isinstance(fig, go.Figure)


def test_rca() -> None:
    """Test the supervised RCA (root cause analysis) case."""
    data_manager = SupervisedTabularDataManager(
        data=DataSet.load_titanic_dataset(),
        target="Survived",
        splitter="RCA",
        prediction_type=PredictionType.BinaryClassification,
    )
    data_manager.build_pipeline(drop_cols=["PassengerId"])
    exp_manager = ExperimentManager(data_manager, optimize_metric=MetricName.Accuracy)
    exp_manager.remove_models(
        ["Gradient Boosting", "LightGBM", "BaselineClassification", "XGBoost"]
    )
    exp_manager.run_experiment()
    assert exp_manager.opt_metric is not None
    interpreter = TabularInterpreterBinaryClassification(
        model=exp_manager.get_best_model(),
        data_manager=data_manager,
        opt_metric=exp_manager.opt_metric,
        pos_class={"pos_class": 1},
    )
    df = interpreter.get_feature_importance()
    assert isinstance(df, pd.DataFrame)
