"""DataSet class for retrieving public data sets."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_wine


class DataSet:
    """Retrieves public data sets."""

    @staticmethod
    def load_titanic_dataset() -> pd.DataFrame:
        """Retrieves the Titanic data set.

        Returns:
            The data set as a data frame.
        """
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)
        return df

    @staticmethod
    def load_fetch_california_housing_dataset() -> pd.DataFrame:
        """Retrieves the California housing data set.

        Returns:
            The data set as a data frame.
        """
        dataset = fetch_california_housing(as_frame=True)
        return dataset["frame"]

    @staticmethod
    def load_clustering_dataset() -> pd.DataFrame:
        """Retrieves the Iris clustering data set.

        Returns:
            The data set as a data frame.
        """
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
        names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
        df = pd.read_csv(url, names=names)
        return df

    @staticmethod
    def load_time_series_classification_dataset() -> pd.DataFrame:
        """Retrieves the time series classification data set.

        Returns:
            The data set as a data frame.
        """

        def get_df(array: np.ndarray) -> pd.DataFrame:
            array_flat = array.flatten()
            id_array = np.repeat(range(len(array)), len(array[0]))
            data = {"measure": array_flat, "ID": id_array}
            df = pd.DataFrame(data)
            df["ID"] = df["ID"].astype("int32")
            return df

        def readucr(filename: str) -> Tuple[np.ndarray, np.ndarray]:
            data = np.loadtxt(filename, delimiter="\t")
            y = data[:, 0]
            x = data[:, 1:]
            return x, y.astype(int)

        root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

        x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
        x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

        # Standardize the data
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        idx = np.random.permutation(len(x_train))
        x_train = x_train[idx]
        y_train = y_train[idx]
        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0
        x_train = get_df(x_train)
        y_train = pd.Series(y_train, name="traget")
        x_test = get_df(x_test)
        y_test = pd.Series(y_test, name="traget")
        x_test["ID"] += x_train["ID"].max() + 1  # Corrected line
        df = pd.concat([x_train, x_test]).reset_index(drop=True)
        df["target"] = df["ID"].map(pd.concat([y_train, y_test]).reset_index(drop=True))
        return df

    @staticmethod
    def load_wine_dataset() -> pd.DataFrame:
        """Retrieves the wine data set.

        Returns:
            The data set as a data frame.
        """
        dataset = load_wine(as_frame=True)
        return dataset["frame"]
