"""Unsupervised tests."""

from __future__ import annotations

import os
import sys

ROOT_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

TEST_DATA_DIR: str = os.path.join("tests", "data")

from simpml.core.base import MetricName, PredictionType
from simpml.core.experiment_manager import ExperimentManager
from simpml.tabular.splitter_pool import ADSplitter
from simpml.tabular.tabular_data_manager import UnsupervisedTabularDataManager


def test_unsupervised() -> None:
    """Test the unsupervised case."""
    data_path = "docs/examples/datasets/binary/Titanic.csv"

    data_manager = UnsupervisedTabularDataManager(
        data=data_path,
        prediction_type=PredictionType.AnomalyDetection,
        target="Survived",
        splitter=ADSplitter(target="Survived"),
    )
    data_manager.build_pipeline()

    X_train, y_train = data_manager.get_training_data()
    X_valid, y_valid = data_manager.get_validation_data()

    assert y_train is not None
    assert len(y_train) == len(X_train)
    assert y_valid is not None
    assert len(X_valid) == len(y_valid)
    assert X_train.shape[1] == X_valid.shape[1]

    exp_manager = ExperimentManager(data_manager, optimize_metric=MetricName.AUC)
    exp_manager.display_models_pool()
    exp_manager.display_metrics_pool()

    def find_contamination(data_manager: UnsupervisedTabularDataManager) -> float:
        """Find contamination.

        Args:
            data_manager: The data manager.

        Returns:
            The contamination.
        """
        y_valid = data_manager.get_validation_data()[1]
        assert y_valid is not None
        value_counts = y_valid.value_counts()
        total_samples = value_counts.sum()
        return (value_counts / total_samples)[1]

    contamination = find_contamination(data_manager)
    exp_manager.run_experiment(models_kwargs={"contamination": contamination})
    best_model = exp_manager.get_best_model()
    assert best_model


def test_unsupervised_clustering() -> None:
    """Test the unsupervised clustering case."""
    data_path = "docs/examples/datasets/binary/Titanic.csv"

    data_manager = UnsupervisedTabularDataManager(
        data=data_path,
        prediction_type=PredictionType.Clustering,
        splitter="Random",
    )
    data_manager.build_pipeline()

    X_train, y_train = data_manager.get_training_data()
    X_valid, y_valid = data_manager.get_validation_data()

    assert y_train is None
    assert y_valid is None
    assert X_train.shape[1] == X_valid.shape[1]

    exp_manager = ExperimentManager(data_manager, optimize_metric=MetricName.CalinskiHarabasz)
    exp_manager.display_models_pool()
    exp_manager.display_metrics_pool()
    exp_manager.run_experiment()
    best_model = exp_manager.get_best_model()
    assert best_model
