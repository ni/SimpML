"""Tabular model based classes."""

from __future__ import annotations

import copy
import pickle
from os import PathLike
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn import metrics
from typing_extensions import Self

from simpml.core.base import (
    MinOrMax,
    PandasModelManagerBase,
    SupervisedMetricManagerBase,
    UnsupervisedMetricManagerBase,
)
from simpml.tabular.inference import TabularInferenceManager
from simpml.tabular.tabular_data_manager import TabularDataManager


class SupervisedMetricManager(SupervisedMetricManagerBase):
    """Model for presentation in the table

    Input:
        metric (metric-func): metric
        name (str): name of the model-view
    """

    def __init__(
        self, metric: Any, name: str, optimization_direction: Union[str, MinOrMax], desc: str = ""
    ) -> None:
        """Initializes the SupervisedMetricManager class.

        Args:
            metric: The object that will perform the metric.
            name: The name of the metric.
            optimization_direction: The optimization direction as a `MinOrMax` Enum or string value.
            desc: The description of the metric.
        """
        super().__init__(metric, name, desc)
        self.optimization_direction: MinOrMax = (
            optimization_direction
            if isinstance(optimization_direction, MinOrMax)
            else MinOrMax.from_value(optimization_direction)
        )

    def get_optimization_direction(self) -> MinOrMax:
        """Get the optimization direction.

        Returns:
            The optimization direction as a `MinOrMax` enum.
        """
        return self.optimization_direction


class UnsupervisedMetricManager(UnsupervisedMetricManagerBase):
    """Model for presentation in the table

    Input:
        metric (metric-func): metric
        name (str): name of the model-view
    """

    def __init__(
        self, metric: Any, name: str, optimization_direction: Union[str, MinOrMax], desc: str = ""
    ) -> None:
        """Initializes the UnsupervisedMetricManager class.

        Args:
            metric: The object that will perform the metric.
            name: The name of the metric.
            optimization_direction: The optimization direction as a `MinOrMax` Enum or string value.
            desc: The description of the metric.
        """
        super().__init__(metric, name, desc)
        self.optimization_direction: MinOrMax = (
            optimization_direction
            if isinstance(optimization_direction, MinOrMax)
            else MinOrMax.from_value(optimization_direction)
        )

    def get_optimization_direction(self) -> MinOrMax:
        """Get the optimization direction.

        Returns:
            The optimization direction as a `MinOrMax` enum.
        """
        return self.optimization_direction


def root_mean_squared_error(
    y_actual: Union[np.ndarray, pd.Series], y_predicted: Union[np.ndarray, pd.Series]
) -> float:
    """Get the root mean squared error.

    Args:
        y_actual: The actual values.
        y_predicted: The values predicted by the model.

    Returns:
        The root mean squared error.
    """
    return metrics.mean_squared_error(y_actual, y_predicted, squared=False)


class TabularModelManager(PandasModelManagerBase):
    """Model for presentation in the table

    Input:
        model (model-type): model with hyper-parameters
        name (str): name of the model-view
        desc (str): Description of the model
        random_state (int): random state
        prediction_type (str): The type of models trained
        opt_metric (MetricWrapper): the metric you want to optimized

    Attributes:
        fit(X_train, y_train)
    """

    def __init__(self, model: Any, name: str, desc: str) -> None:
        """Initializes the TabularModelManager class."""
        super().__init__(model, name, desc)

    def __repr__(self) -> str:
        """Represent object instance as string.

        Returns:
            String representation.
        """
        return f"Model: {self.model}, Description: {self.desc}"

    def get_model_pipeline(self, data_manager: TabularDataManager) -> Any:
        """Get the model pipeline.

        Args:
            data_manager: The data manager object.

        Returns:
            The model pipeline.
        """
        inference_manager = TabularInferenceManager(data_manager, self)
        return inference_manager

    def fit(self, data: Tuple[pd.DataFrame, Optional[pd.Series]], **kwargs: Any) -> Self:
        """Fit the model.

        Args:
            data: The training data.
            **kwargs: For compatibility with the base class.

        Returns:
            This class instance.
        """
        if kwargs:
            raise RuntimeError(f"Unrecognized kwargs: {kwargs}")
        return self.model.fit(*data)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Use the model to make a prediction.

        Args:
            X: The feature data.

        Returns:
            The prediction results.
        """
        return self.model.predict(X)

    def export(self, path: Union[str, PathLike] = "my_model.pkl", **kwargs: Any) -> None:
        """Export model.

        Args:
            path: String or PathLike of file path to export the model into.
            **kwargs: For compatibility with the base class.
        """
        if kwargs:
            raise RuntimeError(f"Unrecognized kwargs: {kwargs}")
        with open(path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def clone(self) -> Self:
        """Creates a copy of this class instance.

        Returns:
            A copy of this class instance.
        """
        return copy.deepcopy(self)


class BaselineRegression:
    """Baseline for regression solution where we simply guess the mean."""

    def __init__(self) -> None:
        """Initializes the BaselineRegression class."""
        self.target: Optional[float] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> BaselineRegression:
        """Fit the model.

        Args:
            X_train: The training feature data.
            y_train: The training target data.

        Returns:
            This class instance.
        """
        del X_train

        self.target = y_train.mean()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Use the model to make a prediction.

        Args:
            X: The feature data.

        Returns:
            The prediction results.
        """
        return np.full((len(X),), self.target)


class BaselineClassification:
    """Baseline for modeling where we try to predict according to classes' distribution."""

    def __init__(self) -> None:
        """Initializes the BaselineClassification class."""
        self.target: Optional[Dict[str, float]] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> BaselineClassification:
        """Fit the model.

        Args:
            X_train: The training feature data.
            y_train: The training target data.

        Returns:
            This class instance.
        """
        del X_train

        y_train = pd.Series(y_train)
        self.target = dict(y_train.value_counts() / len(y_train))
        self.classes_ = y_train.unique()
        self.classes_ = y_train.unique()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Use the model to make a prediction.

        Args:
            X: The feature data.

        Returns:
            The prediction results.
        """
        assert self.target is not None
        items = self.target.items()
        classes = [item[0] for item in items]
        ps = [item[1] for item in items]
        return np.random.choice(classes, len(X), p=ps)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Use the model to get prediction probabilities.

        Args:
            X: The feature data.

        Returns:
            The prediction probabilities.
        """
        assert self.classes_ is not None
        return np.full((len(X), len(self.classes_)), 0.5, dtype=np.float_)
