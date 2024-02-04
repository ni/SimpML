"""Vision tests."""

from __future__ import annotations

import os
import sys

import pytest
from fastai.vision.all import get_image_files, ImageDataLoaders, Resize, untar_data, URLs

ROOT_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

TEST_DATA_DIR: str = os.path.join(ROOT_PATH, "tests", "data")

from simpml.core.base import DataType, MetricName, PredictionType
from simpml.core.experiment_manager import ExperimentManager
from simpml.vision.data import VisionDataManager


def is_first_char_upper(val: str) -> bool:
    """Test if the first character of a string is upper case.

    Args:
        val: The string to test.

    Returns:
        Whether the first character of the string is upper case.
    """
    return val[0].isupper()


def test_vision_basic() -> None:
    """Test the basic vision case."""
    data_manager = VisionDataManager(
        ImageDataLoaders.from_name_func(
            untar_data(URLs.PETS),
            get_image_files(untar_data(URLs.PETS) / "images"),
            is_first_char_upper,
            item_tfms=Resize(224),
        ),
        prediction_type=PredictionType.BinaryClassification.value,
        data_type=DataType.Vision.value,
    )
    data_manager.dls.show_batch()
    exp_manager = ExperimentManager(data_manager, MetricName.AUC)
    exp_manager.display_models_pool()
    exp_manager.display_metrics_pool()


@pytest.mark.slow
def test_vision_train() -> None:
    """Test the training vision case."""
    data_manager = VisionDataManager(
        ImageDataLoaders.from_name_func(
            untar_data(URLs.PETS),
            get_image_files(untar_data(URLs.PETS) / "images"),
            is_first_char_upper,
            item_tfms=Resize(224),
        ),
        prediction_type=PredictionType.BinaryClassification,
        data_type=DataType.Vision,
    )
    exp_manager = ExperimentManager(data_manager, MetricName.AUC)
    exp_manager.run_experiment(models_kwargs={"num_epocs": 1})
    best_model = exp_manager.get_best_model()
    assert best_model
