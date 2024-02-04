"""Adapters tests."""

from __future__ import annotations

import os
import sys

import pytest

ROOT_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from simpml.core.base import (
    Dataset,
    DataType,
    FeatureImportanceMethod,
    HyperParamsOptimizationLevel,
    MetricName,
    MinOrMax,
    PredictionType,
    VariableType,
)

TEST_DATA_DIR: str = os.path.join(ROOT_PATH, "tests", "data")


def test_enums() -> None:
    """Test the various Enums in simpml.core.base."""
    # Dataset
    assert Dataset.from_value("Train") == Dataset.Train
    assert Dataset.from_value("Valid") == Dataset.Valid
    assert Dataset.from_value("Test") == Dataset.Test
    # name and value are the same in most cases throughout the string-based enums.
    assert Dataset.from_name("Train") == Dataset.Train
    assert Dataset.from_name("Valid") == Dataset.Valid
    assert Dataset.from_name("Test") == Dataset.Test

    # Test some not exist cases.
    with pytest.raises(ValueError):
        Dataset.from_name("NotExists")
    assert Dataset.from_name("NotExists", default=Dataset.Test) == Dataset.Test
    assert Dataset.from_name("NotExists", default=None) is None

    with pytest.raises(ValueError):
        Dataset.from_value("NotExists")
    assert Dataset.from_value("NotExists", default=Dataset.Test) == Dataset.Test
    assert Dataset.from_value("NotExists", default=None) is None

    # PredictionType
    assert PredictionType.from_value("Regression") == PredictionType.Regression
    assert PredictionType.from_value("BinaryClassification") == PredictionType.BinaryClassification
    assert (
        PredictionType.from_value("MulticlassClassification")
        == PredictionType.MulticlassClassification
    )
    assert PredictionType.from_value("Clustering") == PredictionType.Clustering
    assert PredictionType.from_value("AnomalyDetection") == PredictionType.AnomalyDetection

    # FeatureImportanceMethod
    assert FeatureImportanceMethod.from_value("Shap") == FeatureImportanceMethod.Shap
    assert FeatureImportanceMethod.from_value("Permutation") == FeatureImportanceMethod.Permutation

    # DataType
    assert DataType.from_value("Tabular") == DataType.Tabular
    assert DataType.from_value("Vision") == DataType.Vision

    # MinOrMax
    assert MinOrMax.from_value("Min") == MinOrMax.Min
    assert MinOrMax.from_value("Max") == MinOrMax.Max

    # MetricName
    assert MetricName.from_value("MSE") == MetricName.MSE
    assert MetricName.from_value("RMSE") == MetricName.RMSE
    assert MetricName.from_value("MAPE") == MetricName.MAPE
    assert MetricName.from_value("R2") == MetricName.R2
    assert MetricName.from_value("Accuracy") == MetricName.Accuracy
    assert MetricName.from_value("AUC") == MetricName.AUC
    assert MetricName.from_value("Recall") == MetricName.Recall
    assert MetricName.from_value("Precision") == MetricName.Precision
    assert MetricName.from_value("Balanced Accuracy") == MetricName.BalancedAccuracy
    assert MetricName.from_value("Kappa") == MetricName.Kappa
    assert MetricName.from_value("F1") == MetricName.F1
    assert MetricName.from_value("Silhouette Score") == MetricName.Silhouette
    assert MetricName.from_value("Davies-Bouldin Score") == MetricName.DaviesBouldin
    assert MetricName.from_value("Calinski-Harabasz Score") == MetricName.CalinskiHarabasz
    # Cases where name != value
    assert MetricName.from_name("BalancedAccuracy") == MetricName.BalancedAccuracy
    assert MetricName.from_name("Silhouette") == MetricName.Silhouette
    assert MetricName.from_name("DaviesBouldin") == MetricName.DaviesBouldin
    assert MetricName.from_name("CalinskiHarabasz") == MetricName.CalinskiHarabasz

    # VariableType
    assert VariableType.from_value("Generic") == VariableType.Generic
    assert VariableType.from_value("Categorical") == VariableType.Categorical
    assert VariableType.from_value("Numerical") == VariableType.Numerical
    assert VariableType.from_value("DateTime") == VariableType.DateTime
    assert VariableType.from_value("Imbalanced") == VariableType.Imbalanced
    assert VariableType.from_value("IsTarget") == VariableType.IsTarget

    # HyperParamsOptimizationLevel
    default_level = HyperParamsOptimizationLevel.Default
    assert HyperParamsOptimizationLevel.from_value("Default") == default_level
    assert HyperParamsOptimizationLevel.from_value("Fast") == HyperParamsOptimizationLevel.Fast
    assert HyperParamsOptimizationLevel.from_value("Slow") == HyperParamsOptimizationLevel.Slow
    assert HyperParamsOptimizationLevel.from_name("Default") == HyperParamsOptimizationLevel.Default
    assert HyperParamsOptimizationLevel.from_name("Fast") == HyperParamsOptimizationLevel.Fast
    assert HyperParamsOptimizationLevel.from_name("Slow") == HyperParamsOptimizationLevel.Slow

    # Test some not exist cases.
    with pytest.raises(ValueError):
        HyperParamsOptimizationLevel.from_name("NotExists")
    assert (
        HyperParamsOptimizationLevel.from_name(
            "NotExists", default=HyperParamsOptimizationLevel.Fast
        )
        == HyperParamsOptimizationLevel.Fast
    )
