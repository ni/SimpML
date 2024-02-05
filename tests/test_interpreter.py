"""Interpreter tests."""

from __future__ import annotations

import os
import sys

ROOT_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from simpml.core.base import MetricName, PredictionType
from simpml.core.experiment_manager import ExperimentManager
from simpml.tabular.interpreter import TabularInterpreterClassification
from simpml.tabular.tabular_data_manager import SupervisedTabularDataManager

TEST_DATA_DIR: str = os.path.join("tests", "data")


def test_noisy_features() -> None:
    """Test getting noisy features."""
    data_manager = SupervisedTabularDataManager(
        os.path.join(TEST_DATA_DIR, "train_titanic.csv"),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )
    data_manager.load_pipeline(data_manager.build_pipeline())
    exp_manager = ExperimentManager(data_manager, MetricName.AUC)
    exp_manager.run_experiment()
    assert exp_manager.opt_metric is not None
    interp = TabularInterpreterClassification(
        exp_manager.get_model("XGBoost", exp_manager.get_current_experiment_id()),
        exp_manager.get_data_model_of_best_model(),
        exp_manager.opt_metric,
    )
    assert len(interp.get_noisy_features()) > -1


def test_no_shap() -> None:
    """Test getting noisy features."""
    data_manager = SupervisedTabularDataManager(
        os.path.join(TEST_DATA_DIR, "train_titanic.csv"),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )
    data_manager.load_pipeline(data_manager.build_pipeline())
    exp_manager = ExperimentManager(data_manager, MetricName.AUC)
    exp_manager.run_experiment()
    assert exp_manager.opt_metric is not None
    interp = TabularInterpreterClassification(
        exp_manager.get_model("XGBoost", exp_manager.get_current_experiment_id()),
        exp_manager.get_data_model_of_best_model(),
        exp_manager.opt_metric,
        enable_shap=False,
    )
    assert interp.get_noisy_features() == []
