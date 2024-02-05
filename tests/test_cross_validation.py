"""Cross-validation tests."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt

ROOT_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

TEST_DATA_DIR: str = os.path.join("tests", "data")

from simpml.core.base import MetricName, PredictionType
from simpml.core.data_set import DataSet
from simpml.core.experiment_manager import ExperimentManager
from simpml.core.trainers_pool import CVAggregation, CVSelectedModel, CVTrainer
from simpml.tabular.tabular_data_manager import CrossValidationSupervisedTabularDataManager


def test_cross_validation() -> None:
    """Test the cross validation case."""
    data_manager = CrossValidationSupervisedTabularDataManager(
        data=DataSet.load_titanic_dataset(),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
        n_folds=5,
    )
    data_manager.build_pipeline(drop_cols=["PassengerId"])
    splitter = data_manager.splitter
    assert hasattr(splitter, "plot_cross_validation")
    fig = splitter.plot_cross_validation(data_manager.data)
    assert isinstance(fig, plt.Figure)
    exp_manager = ExperimentManager(
        data_manager,
        optimize_metric=MetricName.AUC,
        trainer=CVTrainer(aggregation=CVAggregation.MEAN, selected_model=CVSelectedModel.BEST),
    )
    exp_manager.run_experiment()
    assert isinstance(exp_manager.trainer, CVTrainer)
    try:
        fig = exp_manager.trainer.plot_cv_res()
        assert isinstance(fig, plt.Figure)
    except AttributeError:
        # TODO: Fix the pipeline machine environment (inconsistent matplotlib version) to
        # avoid this error:
        # AttributeError: 'Axes' object has no attribute 'is_first_col'
        # https://stackoverflow.com/questions/75644384
        pass
