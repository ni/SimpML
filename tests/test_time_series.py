"""Time series tests."""
# mypy: ignore-errors
# flake8: noqa
from __future__ import annotations

import os
import sys
import uuid
from typing import cast, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import pytest
import seaborn as sns

ROOT_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

TEST_DATA_DIR: str = os.path.join(ROOT_PATH, "tests", "data")

from simpml.core.base import Dataset, MetricName, PredictionType
from simpml.core.data_set import DataSet
from simpml.core.experiment_manager import ExperimentManager
from simpml.tabular.data_fetcher_pool import TabularDataFetcher
from simpml.tabular.interpreter import TabularInterpreterBinaryClassification
from simpml.tabular.pipeline import get_data_types, pipeline_decorator, PipelineBuilder
from simpml.tabular.splitter_pool import GroupSplitter
from simpml.tabular.tabular_data_manager import SupervisedTabularDataManager


@pytest.mark.slow
def test_time_series_train() -> None:
    """Test the time series case (with training)."""
    df = DataSet.load_time_series_classification_dataset()[::100]
    data_manager = SupervisedTabularDataManager(
        data=df,
        target="target",
        prediction_type=PredictionType.BinaryClassification,
        splitter=GroupSplitter(
            split_sets={Dataset.Train: 0.8, Dataset.Valid: 0.2}, group_columns="ID"
        ),
    )
    data_manager.build_pipeline(
        id_label_encoder=True,
        waveforms_feature_extractor=True,
        step_params={
            "column_id": "ID",
        },
    )
    exp_manager = ExperimentManager(data_manager, optimize_metric=MetricName.AUC)
    exp_manager.run_experiment()
    best_model = exp_manager.get_best_model()
    assert best_model
    assert exp_manager.opt_metric is not None
    interpreter = TabularInterpreterBinaryClassification(
        model=exp_manager.get_model(
            model_name="XGBoost", experiment_id=exp_manager.get_current_experiment_id()
        ),
        data_manager=data_manager,
        opt_metric=exp_manager.opt_metric,
        pos_class={"pos_class": 1},
    )
    fig = interpreter.main_fig()
    assert isinstance(fig, go.Figure)
