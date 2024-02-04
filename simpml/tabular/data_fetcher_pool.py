"""Tabular data fetcher pool."""

from __future__ import annotations

from os import PathLike
from typing import Union

import pandas as pd

from simpml.core.base import DataFetcherBase


class TabularDataFetcher(DataFetcherBase):
    """A data fetcher class for supervised tabular data.

    Inherits from the `DataFetcherBase` class.
    """

    def __init__(self, data: Union[str, PathLike, pd.DataFrame]) -> None:
        """Initializes the TabularDataFetcher class.

        Args:
            data: If a string or PathLike, this is a file path to a CSV file that contains
                the data. Otherwise, this is the data frame of the data itself.
        """
        self.data = data
        if isinstance(self.data, (str, PathLike)):
            self.loaded_data: pd.DataFrame = pd.read_csv(self.data)
        else:
            self.loaded_data = self.data

    def get_items(self) -> pd.DataFrame:
        """Fetches the data items (from a CSV file or preloaded data).

        Returns:
            A single data frame containing all of the data (input features and target if available).
        """
        return self.loaded_data
