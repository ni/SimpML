"""File to implement the ShapManager class"""

from __future__ import annotations

import threading
from functools import wraps
from typing import Any, Callable, cast, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from simpml.core.base import PandasModelManagerBase


def default_data_fill(f: Callable) -> Callable:
    """Decorator function to automatically fill X (feature data) and y (target data) variables
    in a wrapped API if those are set to None.

    If "y" is not present in a wrapped API, it will skip filling the target data.

    Args:
        f: The wrapped function.

    Returns:
        The decorator.
    """

    @wraps(f)
    def wrapper(
        self: ShapManager,
        X: Optional[pd.DataFrame] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """The wrapper function for the decorator.

        Args:
            self: Since this is meant to wrap class instance methods, this is the "self" argument
               to a class instance that is derived from `TabularInterpreterBase`.
            X: The argument for the feature data.
            y: The argument for the target data.
            *args: Any other arguments for the wrapped function.
            **kwargs: Any other keyword arguments for the wrapped function.

        Returns:
            Whatever the wrapped function returns.
        """
        if X is None:
            X = self.X_valid
        return f(self, X, *args, **kwargs)

    return wrapper


class FunctionTimeoutError(Exception):
    """Timeout Error"""

    pass


def timeout_decorator(func: Callable) -> Callable:
    """Decorator meant to cancel a functions execution if it runs for too long"""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        seconds = args[0].timeout if hasattr(args[0], "timeout") else None
        if seconds is None:
            raise ValueError("Timeout not provided")

        result_container: Dict[str, Any] = {"result": None}
        exc_container: Dict[str, Any] = {"exception": None}

        def target() -> None:
            try:
                result_container["result"] = func(*args, **kwargs)
            except Exception as e:
                exc_container["exception"] = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=seconds)

        if thread.is_alive():
            thread.join()
            raise FunctionTimeoutError(
                f"Function '{func.__name__}' timed out after {seconds} seconds"
            )
        else:
            if exc_container["exception"]:
                raise exc_container["exception"]
            return result_container["result"]

    return wrapper


# Calcualte only values and send them to other functions
class ShapManager:
    """ShapManager class
    This class is instantiated by the interpreter and handles
    all SHAP related calculations and funcionality
    """

    def __init__(
        self,
        model: PandasModelManagerBase,
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        check_additivity: bool,
        timeout: int = 60,
    ):
        """Init method for ShapManager class
        model: The machine learning model recieved from the interpreter
        X_train: The training data recieved from the interpreter
        X_valid: The validation data recieved from the interpreter
        check_additivity: Controls the check_additivity parameter in SHAP, recieved from interpreter
        timeout: Controls the maximum amount of time to let the SHAP values calculate
        """
        self.X_train = X_train
        self.X_valid = X_valid
        try:
            self.explainer = shap.Explainer(model.model, self.X_train)
        except Exception as e:
            print(f"Failed to find a shap.Explainer with the following error {e}")
        self.model = model.model
        self.check_additivity = check_additivity
        self.timeout = timeout
        self.cache = {
            "X_train": self.get_shap_values(self.X_train, cache=False),
            "X_valid": self.get_shap_values(self.X_valid, cache=False),
        }

    @timeout_decorator
    def get_shap_values(
        self,
        x: Optional[pd.DataFrame] = None,
        single_mat: bool = True,
        as_df: bool = True,
        cache: bool = True,
    ) -> Optional[Union[shap.Explanation, pd.DataFrame]]:
        """Returns the SHAP values for the given data based
        on the interpreters model and explainer
        """
        explainer = self.explainer
        if x is None:
            x = self.X_valid
        if x.equals(self.X_train) and cache:
            return self.cache["X_train"]
        elif x.equals(self.X_valid) and cache:
            return self.cache["X_valid"]
        else:
            try:
                vals = explainer(x, check_additivity=self.check_additivity)
            except Exception as e:
                if "ExplainerError" in str(type(e)):
                    print("Caught an ExplainerError:", e)  #
                    print(
                        "Running without check_additivity which might lead to inaccurate results!"
                    )
                    self.check_additivity = False
                    vals = explainer(x, check_additivity=False)
                else:
                    raise

            if (
                isinstance(self.model, LGBMClassifier)
                or isinstance(self.model, LGBMRegressor)
                or isinstance(self.model, LogisticRegression)
                or isinstance(self.model, LinearRegression)
            ):
                return vals
            if len(vals.shape) == 3 and vals.shape[-1] == 2:
                return vals[:, :, 0]
            # if len(vals) == 2:
            #     return vals[0]
            if as_df:
                return pd.DataFrame(vals.values, columns=x.columns)
            else:
                return vals

    @default_data_fill
    def plot_summary_shap(
        self, x: Optional[pd.DataFrame] = None, size: Tuple[int, int] = (8, 6)
    ) -> None:
        """Create a SHAP beeswarm plot, colored by feature values when they are provided.

        Args:
            X: Input Dataset.
            size: Tuple of width, height of the plot.

        Returns:
            A plot of the shap values for each feature.
        """
        if x is None:
            x = self.X_valid
        shap_values = self.get_shap_values(x)
        shap.summary_plot(shap_values, feature_names=x.columns)

    @default_data_fill
    def plot_shap_on_row(
        self,
        X: Optional[Union[pd.DataFrame, shap.Explainer]] = None,
        row_number: int = 0,
        plot_type: str = "force_plot",
    ) -> Union[shap.force_plot, shap.decision_plot]:
        """Visualize the given SHAP values with an additive force layout or Visualize model
        decisions using cumulative SHAP values.

        Args:
            X: Input Dataset.
            row_number: Index of row to display shap values for.
            plot_type: Type of plot to display, either "force_plot" (default) or "decision_plot".

        Returns:
            A plot of each features shap value on a single row of data
        """
        if X is None:
            X = self.X_valid
        assert X is not None
        shap_values = self.get_shap_values(X)
        if isinstance(self.explainer.expected_value, (int, float, np.float32)):
            expected_value = self.explainer.expected_value
        else:
            expected_value = self.explainer.expected_value[0]

        if plot_type == "force_plot":
            return shap.force_plot(
                base_value=expected_value,
                # shap_values=np.array(shap_values.iloc[row_number]),
                shap_values=np.array(shap_values.data[row_number]),
                features=X.iloc[row_number],
            )
        elif plot_type == "decision_plot":
            return shap.decision_plot(
                base_value=expected_value,
                shap_values=np.array(shap_values.data[row_number]),
                feature_names=X.columns.to_list(),
                link="logit",
            )
        else:
            raise ValueError(f"Invalid plot_type: {plot_type}")

    @default_data_fill
    def plot_shap_dependence(
        self, feature_name: str, X: Optional[pd.DataFrame] = None
    ) -> shap.dependence_plot:
        """Create a SHAP dependence plot, colored by an interaction feature.

        Plots the value of the feature on the x-axis and the SHAP value of the same feature
        on the y-axis. This shows how the model depends on the given feature, and is like a
        richer extenstion of the classical parital dependence plots. Vertical dispersion of the
        data points represents interaction effects. Grey ticks along the y-axis are data
        points where the feature's value was NaN.

        Args:
            feature_name: the column name (in a pandas dataframe) of the feature you want to
                inspect.
            X: Input Dataset.

        Returns:
            Shap dependance plot of a single feature in the given dataset.
        """
        # Can't use data fill decorator because of default argument order
        if X is None:
            X = self.X_valid
        vals = self.get_shap_values(X)
        return shap.dependence_plot(ind=feature_name, shap_values=vals.values, features=X)

    @default_data_fill
    def calculate_shap_fi(
        self, X: pd.DataFrame, shap_values: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Produces an importance table based on SHAP (much less biased than native feature
        importance implementations that exist for individual algorithms).

        Args:
            x: The local population relevant to the calculation.
            shap_values: The function may have the Shap values to save running time, otherwise it
                will perform the calculation.

        Returns:
            A data frame with the importance table.
        """
        if shap_values is None:
            shap_values = self.get_shap_values(X)
        # vals: np.ndarray = np.abs(cast(np.ndarray, np.array(shap_values)))
        vals: np.ndarray = np.abs(shap_values.values)
        feature_importance = pd.DataFrame(
            list(zip(X.columns, cast(np.ndarray, sum(vals)))),
            columns=["col_name", "feature_importance_vals"],
        )
        feature_importance.sort_values(
            by=["feature_importance_vals"], ascending=False, inplace=True
        )
        feature_importance["feature_importance_vals"] = (
            feature_importance["feature_importance_vals"]
            / feature_importance["feature_importance_vals"].sum()
        )
        return feature_importance.reset_index(drop=True)
