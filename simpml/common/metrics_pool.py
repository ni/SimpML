"""Metrics pool."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn import metrics
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from simpml.core.base import DataType, MetricManagerBase, MetricName, MinOrMax, PredictionType
from simpml.tabular.model import (
    root_mean_squared_error,
    SupervisedMetricManager,
    UnsupervisedMetricManager,
)

METRICS_POOL: Dict[str, Dict[str, List[MetricManagerBase]]] = {}


def safe_silhouette_score(
    X: Union[np.ndarray, list],
    labels: Union[np.ndarray, list],
    *,
    default_value: Optional[float] = None,
    metric: str = "euclidean",
    sample_size: Optional[int] = None,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    **kwds: Any,
) -> Optional[float]:
    """A safe version of silhouette_score that checks the number of unique labels
    and returns a default value if the number of unique labels is not between 2 and n_samples - 1.

    Args:
        X: {array-like, sparse matrix} of shape (n_samples_a, n_samples_a). If metric ==
            "precomputed" or (n_samples_a, n_features) otherwise an array of pairwise distances
            between samples, or a feature array.
        labels: array-like of shape (n_samples,). Predicted labels for each sample.
        default_value: The default value to return if the number of unique labels is not valid for
            silhouette score.
        metric: The metric to use when calculating distance between instances in a feature array.
            If metric is a string, it must be one of the options allowed by
            metrics.pairwise.pairwise_distances. If X is the distance array itself, use
            metric="precomputed".
        sample_size: The size of the sample to use when computing the Silhouette Coefficient on a
            random subset of the data. If sample_size is None, no sampling is used.
        random_state: Determines random number generation for selecting a subset of samples. Used
            when sample_size is not None. Pass an int for reproducible results across multiple
            function calls. See Glossary.
        **kwds: Optional keyword parameters. Any further parameters are passed directly to the
            distance function. If using a scipy.spatial.distance metric, the parameters are still
            metric dependent. See the scipy docs for usage examples.

    Returns:
        Mean Silhouette Coefficient for all samples.
    """
    n_samples = len(labels)
    unique_labels = np.unique(labels)

    if 2 <= len(unique_labels) <= n_samples - 1:
        return silhouette_score(
            X, labels, metric=metric, sample_size=sample_size, random_state=random_state, **kwds
        )

    # If the number of unique labels is not valid for silhouette score, return the default value
    return default_value


METRICS_POOL[DataType.Tabular.value] = {
    PredictionType.Regression.value: [
        SupervisedMetricManager(
            metric=metrics.mean_squared_error,
            name=MetricName.MSE.value,
            optimization_direction=MinOrMax.Min.value,
        ),
        SupervisedMetricManager(
            metric=root_mean_squared_error,
            name=MetricName.RMSE.value,
            optimization_direction=MinOrMax.Min.value,
        ),
        SupervisedMetricManager(
            metric=metrics.r2_score,
            name=MetricName.R2.value,
            optimization_direction=MinOrMax.Max.value,
        ),
    ],
    PredictionType.BinaryClassification.value: [
        SupervisedMetricManager(
            metric=metrics.accuracy_score,
            name=MetricName.Accuracy.value,
            optimization_direction=MinOrMax.Max.value,
        ),
        SupervisedMetricManager(
            metric=metrics.roc_auc_score,
            name=MetricName.AUC.value,
            optimization_direction=MinOrMax.Max.value,
        ),
        SupervisedMetricManager(
            metric=metrics.recall_score,
            name=MetricName.Recall.value,
            optimization_direction=MinOrMax.Max.value,
        ),
        SupervisedMetricManager(
            metric=metrics.precision_score,
            name=MetricName.Precision.value,
            optimization_direction=MinOrMax.Max.value,
        ),
        SupervisedMetricManager(
            metric=metrics.balanced_accuracy_score,
            name=MetricName.BalancedAccuracy.value,
            optimization_direction=MinOrMax.Max.value,
        ),
        SupervisedMetricManager(
            metric=metrics.f1_score,
            name=MetricName.F1.value,
            optimization_direction=MinOrMax.Max.value,
        ),
    ],
    PredictionType.MulticlassClassification.value: [
        SupervisedMetricManager(
            metric=metrics.accuracy_score,
            name=MetricName.Accuracy.value,
            optimization_direction=MinOrMax.Max.value,
        ),
        SupervisedMetricManager(
            metric=metrics.cohen_kappa_score,
            name=MetricName.Kappa.value,
            optimization_direction=MinOrMax.Max.value,
        ),
    ],
    PredictionType.Clustering.value: [
        UnsupervisedMetricManager(
            metric=safe_silhouette_score,
            name=MetricName.Silhouette.value,
            optimization_direction=MinOrMax.Max.value,
        ),
        UnsupervisedMetricManager(
            metric=davies_bouldin_score,
            name=MetricName.DaviesBouldin.value,
            optimization_direction=MinOrMax.Min.value,
        ),
        UnsupervisedMetricManager(
            metric=calinski_harabasz_score,
            name=MetricName.CalinskiHarabasz.value,
            optimization_direction=MinOrMax.Max.value,
        ),
    ],
}

METRICS_POOL[DataType.Tabular.value][PredictionType.AnomalyDetection.value] = (
    METRICS_POOL[DataType.Tabular.value][PredictionType.BinaryClassification.value])

METRICS_POOL[DataType.Vision.value] = METRICS_POOL[DataType.Tabular.value]


def register_metrics_to_pool(data_type: Union[str, DataType],
                             prediction_type: Union[str, PredictionType],
                             metrics: List[MetricManagerBase]) -> None:
    """Register a list of metrics to the METRICS_POOL under the specified data and prediction types.

    :param data_type: The type of data (e.g., 'Tabular', 'Text', 'Vision').
    :param prediction_type: The type of prediction task (e.g., 'Regression', 'Classification').
    :param metrics: A list of instances of ModelManagerBase or its subclasses.
    """
    data_type = data_type if isinstance(data_type, str) else data_type.value
    prediction_type = (
        prediction_type
        if isinstance(prediction_type, str)
        else prediction_type.value)

    if data_type not in METRICS_POOL:
        METRICS_POOL[data_type] = {}

    if prediction_type not in METRICS_POOL[data_type]:
        METRICS_POOL[data_type][prediction_type] = []

    current_metrics = METRICS_POOL[data_type][prediction_type]

    for metric in metrics:
        if metric.name not in [metric.name for metric in current_metrics]:
            current_metrics.append(metric)
