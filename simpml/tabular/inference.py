"""Implementation of TabularInferenceManager."""

from __future__ import annotations

import copy
import pickle
from os import PathLike
from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline as SklearnPipeline

from simpml.core.base import InferenceManagerBase, ModelManagerBase
from simpml.tabular.pipeline import TargetPipeline
from simpml.tabular.tabular_data_manager import TabularDataManager


class TabularInferenceManager(InferenceManagerBase):
    """A class used to manage the inference process for tabular data.

    The class is initialized with a data manager and a model, and provides methods for
    making predictions with the model
    pipeline on the given data. The model pipeline is a deep copy of the data manager's pipeline
    with the model appended to it.

    Attributes
    ----------
    model_pipeline : Pipeline
        A deep copy of the data manager's pipeline with the model appended to it.
    target_pipeline : Pipeline
        A deep copy of the data manager's target pipeline.

    Methods
    -------
    predict(data: Iterable) -> Iterable
        Makes predictions with the model pipeline on the given data.
    """

    def __init__(self, data_manager: TabularDataManager, model: ModelManagerBase):
        """Initializes the TabularInferenceManager with a given data manager and model.

        It deep copies the data manager's pipeline and appends the model to it.

        Args:
            data_manager: The data manager to be used for inference.
            model: The model to be used for inference.
        """
        if data_manager.pipeline is None:
            self.model_pipeline: SklearnPipeline = SklearnPipeline(steps=[("model", model)])
            self.target_pipeline: Optional[TargetPipeline] = None
        else:
            self.model_pipeline = copy.deepcopy(data_manager.pipeline.sklearn_pipeline)
            self.model_pipeline.steps.append(("model", model))
            if data_manager.pipeline.target_pipeline is not None:
                self.target_pipeline = copy.deepcopy(data_manager.pipeline.target_pipeline)
            else:
                self.target_pipeline = None
        self.model = model

    def predict(self, data: Iterable, with_input: bool = False) -> Union[np.ndarray, pd.DataFrame]:
        """Makes predictions with the model pipeline on the given data.

        Args:
           data: Data to make predictions on.

        Returns:
            The model's predictions as either a NumPy array or a DataFrame.
        """
        y = self.model_pipeline.predict(data)
        if self.target_pipeline is not None:
            y = self.target_pipeline.inverse_transform(data, y)
        if with_input:
            data_df = pd.DataFrame(data)
            if isinstance(y, pd.Series):
                y.index = data_df.index
            return pd.concat([data_df, pd.Series(y, index=data_df.index, name="Pred")], axis=1)
        else:
            return y

    def export(
        self, path: Union[str, PathLike] = "my_inference_pipline.pkl", **kwargs: Any
    ) -> None:
        """Export inference pipline.

        Args:
            path: String or PathLike of file path to export the inference pipline into.
            **kwargs: For compatibility with the base class.
        """
        if kwargs:
            raise RuntimeError(f"Unrecognized kwargs: {kwargs}")
        with open(path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
