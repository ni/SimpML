"""Base definitions for vision."""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
from fastai.data.core import DataLoaders
from typing_extensions import Self

from simpml.core.base import ModelManagerBase


class FastaiModelManagerBase(ModelManagerBase):
    """This class is an interface for managing models that use Fast AI based data.

    The user must implement the abstract methods.
    """

    @abc.abstractmethod
    def fit(self, data: DataLoaders, num_epocs: int = 5, **kwargs: Any) -> Self:
        """Fit the model.

        Args:
            data: The training data.
            num_epocs: The number of epocs to train.
            **kwargs: For compatibility with the base class.

        Returns:
            This class instance.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Use the model to make a prediction.

        Args:
            X: The feature data.

        Returns:
            The prediction results.
        """
        raise NotImplementedError
