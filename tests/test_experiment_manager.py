"""Experiment manager tests."""


# mypy: ignore-errors
# flake8: noqa

from __future__ import annotations

import os
import sys

ROOT_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from simpml.core.base import MetricName, PredictionType
from simpml.core.experiment_manager import ExperimentManager
from simpml.tabular.tabular_data_manager import SupervisedTabularDataManager
from simpml.vision.data import VisionDataManager
from fastai.vision.all import ImageDataLoaders, untar_data, URLs, get_image_files, Resize

TEST_DATA_DIR: str = os.path.join("tests", "data")

def test_best_opt_metric() -> None:
    """Test finding the best optimization metric."""
    data_manager = SupervisedTabularDataManager(
        os.path.join(TEST_DATA_DIR, "train_titanic.csv"),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )
    data_manager.load_pipeline(data_manager.build_pipeline())
    exp_manager = ExperimentManager(data_manager, MetricName.AUC)
    exp_manager.run_experiment()

    assert exp_manager.get_best_opt_metric() > 0.5

def test_vision_custom_type() -> None:
    """Test vision with custom use case of prediction type & data type."""
    
    def is_first_char_upper(f): return f[0].isupper()

    data_manager = VisionDataManager(ImageDataLoaders.from_name_func(
        untar_data(URLs.PETS),
        get_image_files(untar_data(URLs.PETS)/"images"),
        is_first_char_upper,
        item_tfms=Resize(224)),
        prediction_type = 'kuku1',
        data_type ='kuku2'
    )

    exp_mang = ExperimentManager(data_manager, MetricName.AUC)
    assert exp_mang.prediction_type == 'kuku1'
    assert exp_mang.data_type == 'kuku2'

