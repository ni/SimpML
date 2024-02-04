"""Tabular adapters pool."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from simpml.core.base import ManipulateAdapterBase, TransformerAdapterBase


class ManipulateAdapter(ManipulateAdapterBase):
    """Adapter to manipulate data (no training required)."""

    def __init__(self, manipulator: Any, func_name: str) -> None:
        """Initializes the ManipulateAdapter class.

        Args:
            manipulator: A manipulator object with function to be used as step in train pipeline.
            func_name: A string representing the name of the function.
        """
        super().__init__(manipulator, func_name)

    def manipulate(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Manipulate function adapted from original object.

        Args:
            X: Feature data.
            y: Optional target data.

        Returns:
            Manipulated data.
        """
        method = getattr(self.manipulator, self.func_name)
        return method(X, y)


class EncodeTargetAdapter(TransformerAdapterBase):
    """Adapter to encode the independent (target) variable (training required)."""

    def __init__(self, transformer: Any) -> None:
        """Initializes the EncodeTargetAdapter class.

        Args:
            transformer: A transformer object to be used as step in pipeline.
        """
        super().__init__(transformer)

    def fit(self, y: pd.Series) -> EncodeTargetAdapter:
        """'fit' function adapted from original object.

        Args:
            y: Training data.

        Returns:
            This object instance.
        """
        self.transformer.fit(y)
        return self

    def transform(self, y: pd.Series) -> pd.Series:
        """'transform' function adapted from original object.

        Args:
            y: Training data.

        Returns:
            Transformed data.
        """
        return self.transformer.transform(y)

    def fit_transform(self, y: pd.Series) -> pd.Series:
        """'fit_transform' function adapted from original object.

        Args:
            y: Training data.

        Returns:
            Transformed data.
        """
        self.transformer.fit(y)
        return self.transformer.transform(y)
