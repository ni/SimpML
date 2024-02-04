"""Tabular steps pool."""

from __future__ import annotations

import re
import unittest.mock
from typing import Any, cast, Dict, List, Optional, Sequence, Type, Union

import numpy as np
import pandas as pd
from feature_engine.dataframe_checks import check_X
from feature_engine.selection import DropFeatures
from feature_engine.selection.base_selector import BaseSelector
from numba import cuda
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute._base import _BaseImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted


class SmartImpute(_BaseImputer):
    """SmartImpute imputes missing values in a dataset with an appropriate strategy.

    Inherits from `_BaseImputer` class.
    """

    def __init__(
        self,
        *,
        missing_values: float = np.nan,
        numeric_strategy: str = "mean",
        verbose: int = 0,
        copy: bool = True,
        add_indicator: bool = False,
    ) -> None:
        """Initializes a SmartImpute instance.

        Args:
            missing_values: The placeholder for the missing values.
            numeric_strategy: The imputation strategy for numeric columns. May be "mean" or
                "median".
            verbose: The verbosity level of the imputation process. 0 means silent, 1 means verbose.
            copy: Whether to copy the imputed dataset or modify it in place.
            add_indicator: Whether to add a binary indicator column for imputed values.
        """
        super().__init__(missing_values=missing_values, add_indicator=add_indicator)
        self.verbose: int = verbose
        self.copy: bool = copy
        self.numeric_strategy: str = numeric_strategy

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> SmartImpute:
        """Fits a SmartImpute instance to the input dataset.

        Args:
            X: The input dataset.
            y: Ignored. Present for API consistency by convention.

        Returns:
            The fitted SmartImpute instance.
        """
        imputed_values = {col: None for col in X.columns}

        for col in X.columns:
            dtype = X[col].dtype.name
            if dtype == "object" or dtype == "category":
                imputed_values[col] = X[col].mode()[0]
            elif np.issubdtype(dtype, np.number):
                if self.numeric_strategy == "mean":
                    imputed_values[col] = X[col].mean()
                elif self.numeric_strategy == "median":
                    imputed_values[col] = X[col].median()

        self.imputed_values = {k: v for k, v in imputed_values.items() if v is not None}

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms the input dataset by imputing missing values.

        Args:
            X: The input dataset.

        Returns:
            The transformed dataset with imputed missing values.
        """
        return X.fillna(value=self.imputed_values)


class RemoveSpecialJSONCharacters(BaseSelector):
    """Removes special characters of column names that may interfere with tools such as XGBoost."""

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> RemoveSpecialJSONCharacters:
        """Fit the pipeline step.

        Args:
            X: The training feature data. Unused.
            y: The training target data. Unused.

        Returns:
            This class instance.
        """
        del X
        del y

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Perform the transformation.

        Args:
            X: The feature data.

        Returns:
            The transformed data.
        """
        return X.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "___", str(x)))

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Perform both a fit and a transformation.

        Args:
            X: The training feature data.
            y: The training target data. Unused.

        Returns:
            The transformed data.
        """
        return self.transform(X)


class SafeDropFeatures(DropFeatures):
    """Drop specified features (safe version)."""

    def __init__(self, features_to_drop: Sequence[Union[str, int]]) -> None:
        """Initializes the SafeDropFeatures class.

        Args:
            features_to_drop: List of tuple of feature names or indices to drop.
        """
        super().__init__(features_to_drop)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> SafeDropFeatures:
        """Fit the pipeline step.

        Args:
            X: The training feature data.
            y: The training target data. Unused.

        Returns:
            This class instance.
        """
        del y

        X = check_X(X)

        # Only include features that exist in the DataFrame
        self.features_to_drop_ = [
            feature for feature in self.features_to_drop if feature in X.columns
        ]

        # Check if user is removing all columns in the dataframe
        if len(self.features_to_drop_) == len(X.columns):
            raise ValueError(
                "The resulting dataframe will have no columns after dropping all "
                "existing variables"
            )

        self._get_feature_names_in(X)

        return self


class HighCardinalityDropper(BaseSelector):
    """Drop features with high cardinality."""

    def __init__(self, cardinality_percentage_threshold: float = 0.2) -> None:
        """Inititalizes the HighCardinalityDropper class.

        Args:
            cardinality_percentage_threshold: The cardinality percentage threshold.
        """
        self.cardinality_percentage_threshold: float = cardinality_percentage_threshold

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> HighCardinalityDropper:
        """Fit the pipeline step.

        Args:
            X: The training feature data.
            y: The training target data. Unused.

        Returns:
            This class instance.
        """
        # check if X is a pandas DataFrame
        X = self._check_input(X)

        # find the variables with high cardinality
        self.features_to_drop_ = [
            col
            for col in X.columns
            if (X[col].dtype in ["object", "category"])
            and (X[col].nunique() / len(X)) > self.cardinality_percentage_threshold
        ]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Perform the transformation.

        Args:
            X: The feature data.

        Returns:
            The transformed data.
        """
        # check if fit was performed
        check_is_fitted(self)

        # check if X is a pandas DataFrame
        X = self._check_input(X)

        # drop high cardinality features
        X = X.drop(columns=self.features_to_drop_)

        return X

    def _check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        return X


def safe_extract_features(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Safely extract tsfresh features.

    Args:
        *args: The arguments to pass to tsfresh's "extract_features".
        **kwargs: The keyword arguments to pass to tsfresh's "extract_features".
    """
    with unittest.mock.patch.object(cuda, "is_available", return_value=False):
        import multiprocessing

        import tsfresh
        from tsfresh.feature_extraction.settings import from_columns

        multiprocessing.set_start_method("spawn", True)
        X_tsfresh = tsfresh.extract_features(*args, **kwargs)
        return X_tsfresh, from_columns(X_tsfresh.columns)


class WaveformsFeatureExtractor(BaseSelector):
    """Feature extractor for Waveforms using tsfresh."""

    def __init__(
        self,
        column_id: Optional[str] = None,
        column_sort: Optional[str] = None,
        kind_to_fc_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes the WaveformsFeatureExtractor class.

        Args:
            column_id: The "column_id" argument to pass to tsfresh's "extract_features".
            column_sort: The "column_sort" argument to pass to tsfresh's "extract_features".
            kind_to_fc_parameters: The "kind_to_fc_parameters" argument to pass to tsfresh's
                "extract_features".
        """
        self.column_id: Optional[str] = column_id
        self.column_sort: Optional[str] = column_sort
        self.kind_to_fc_parameters: Optional[Dict[str, Any]] = kind_to_fc_parameters

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> WaveformsFeatureExtractor:
        """Fit the pipeline step.

        Args:
            X: The training feature data. Unused.
            y: The training target data. Unused.

        Returns:
            This class instance.
        """
        del X
        del y

        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Perform the transformation.

        Args:
            X: The feature data.

        Returns:
            The transformed data.
        """
        idx = X[self.column_id].unique()
        X_tsfresh, _ = safe_extract_features(
            X,
            column_id=self.column_id,
            column_sort=self.column_sort,
            kind_to_fc_parameters=self.kind_to_fc_parameters,
        )
        return pd.DataFrame(X_tsfresh).loc[idx]

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Perform fit and transform in a single step for the WaveformsFeatureExtractor.

        This method first applies the 'transform' method to extract features
        from the input DataFrame X.
        It then uses tsfresh's 'from_columns' to determine the extracted
        feature configurations and sets these as the default feature
        calculators for future transformations.
        This ensures that the same features extracted during the
        training are used for validation and testing.

        To make it compatible with environments without CUDA,
        it mocks the 'cuda.is_available' function to return False.
        This is necessary if tsfresh's behavior changes based on
        CUDA availability.

        Args:
            X: The input DataFrame containing the feature data.
            y: The target data, which is optional and unused. Included for
               compatibility with scikit-learn's fit_transform method signature.

        Returns:
            A DataFrame containing the extracted features from tsfresh.
        """
        idx = X[self.column_id].unique()
        X_tsfresh, kind_to_fc_parameters = safe_extract_features(
            X,
            column_id=self.column_id,
            column_sort=self.column_sort,
            kind_to_fc_parameters=self.kind_to_fc_parameters,
        )
        self.kind_to_fc_parameters = kind_to_fc_parameters
        return pd.DataFrame(X_tsfresh).loc[idx]


class MinMaxScalerWithColumnNames(BaseEstimator):
    """Min/max scaler for Pandas data frames (supports column names)."""

    class FlexibleMinMaxScaler(MinMaxScaler):
        """Min/max scaler for numpy arrays."""

        def _check_n_features(self, X: np.ndarray, reset: bool) -> np.ndarray:
            """Check the number of features.

            If the number of features in 'X' is different than that of training data,
            this method will attempt to adjust 'X' to match the number of features of training data.

            Args:
                X: The feature data to check the number of features on.
                reset: Ignored.

            Returns:
                The modified feature data with matching number of features as with the training
                data.
            """
            n_features = X.shape[1]

            if hasattr(self, "n_features_in_"):
                n_features_in: int = cast(int, self.n_features_in_)  # type: ignore [has-type]
                if n_features_in != n_features:
                    # If X has more features than training data, drop the extra features
                    if n_features > n_features_in:
                        X = X[:, :n_features_in]

                    # If X has fewer features than training data, add zeros
                    elif n_features < n_features_in:
                        zeros = np.zeros((X.shape[0], n_features_in - n_features))
                        X = np.concatenate([X, zeros], axis=1)
            else:
                self.n_features_in_ = n_features

            return X

        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            """Perform the transformation.

            Args:
                X: The feature data.

            Returns:
                The transformed data.
            """
            X = self._check_n_features(X, reset=False)
            return super().transform(X)

    def __init__(self) -> None:
        """Initializes the MinMaxScalerWithColumnNames class."""
        self.scaler: MinMaxScalerWithColumnNames.FlexibleMinMaxScaler = (
            MinMaxScalerWithColumnNames.FlexibleMinMaxScaler()
        )
        self.column_names: Optional[Sequence[str]] = None

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> MinMaxScalerWithColumnNames:
        """Fit the pipeline step.

        Args:
            df: The training feature data.
            y: The training target data. Unused.

        Returns:
            This class instance.
        """
        # Select only the numeric columns
        df = df.select_dtypes(include=["int64", "float64"])

        # If no numeric column is present, return self
        if df.shape[1] == 0:
            return self

        self.column_names = df.columns
        self.scaler.fit(df.values)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform the transformation.

        Args:
            df: The feature data.

        Returns:
            The transformed data.
        """
        # Select only the numeric columns
        df_numeric = df.select_dtypes(include=["int64", "float64"])[self.column_names]

        # If no numeric column is present, return the original dataframe
        if df_numeric.shape[1] == 0:
            return df

        df_numeric.reset_index(drop=True, inplace=True)
        scaled_data = self.scaler.transform(df_numeric.values)
        scaled_df = pd.DataFrame(scaled_data, columns=self.column_names)
        scaled_df.index = df.index

        # Return original dataframe with updated values in numeric columns
        df[self.column_names] = scaled_df
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform an inverse transformation.

        Args:
            df: The transformed feature data.

        Returns:
            The untransformed data.
        """
        # Select only the numeric columns
        df_numeric = df.select_dtypes(include=["int64", "float64"])

        # If no numeric column is present, return the original dataframe
        if df_numeric.shape[1] == 0:
            return df

        original_data = self.scaler.inverse_transform(df_numeric.values)
        original_df = pd.DataFrame(original_data, columns=self.column_names)

        # Return original dataframe with updated values in numeric columns
        df[self.column_names] = original_df
        return df


class Infinity2Nan(BaseSelector):
    """Converts infinity values to NaN."""

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Infinity2Nan:
        """Fit the pipeline step.

        Args:
            X: The training feature data. Unused.
            y: The training target data. Unused.

        Returns:
            This class instance.
        """
        del X
        del y

        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Perform the transformation.

        Args:
            X: The feature data.
            y: The target data. Unused.

        Returns:
            The transformed data.
        """
        del y

        return X.replace([np.inf, -np.inf], np.nan)


class NanColumnDropper(BaseEstimator, TransformerMixin):
    """Drops columns that have a specified percentage of NaNs."""

    def __init__(self) -> None:
        """Initializes the NanColumnDropper class."""
        self.columns_to_drop_: Optional[Sequence[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> NanColumnDropper:
        """Fit the pipeline step.

        Args:
            X: The training feature data.
            y: The training target data. Unused.

        Returns:
            This class instance.
        """
        # Check if X is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Find columns with all NaN values
        self.columns_to_drop_ = X.columns[X.isnull().all()]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Perform the transformation.

        Args:
            X: The feature data.

        Returns:
            The transformed data.
        """
        # Check if fit was performed
        if self.columns_to_drop_ is None:
            raise RuntimeError("fit method must be called before transform")

        # Check if X is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Drop columns with all NaN values
        X = X.drop(columns=self.columns_to_drop_)

        return X


class DictLabelEncoder(BaseSelector):
    """Label encoder with configurable encoding values.

    This class allows you to convert categorical variables into a format
    that's easier to understand for machine learning algorithms. It works
    by encoding each label with a distinct integer value.

    Args:
        encoding_dict (dict, optional): A dictionary mapping labels to integers.
            If this is not provided, the class will generate a mapping
            based on the unique labels it encounters during fit.

    Attributes:
        encoding_dict (dict): The dictionary mapping labels to integers.
        inverse_dict (dict): The dictionary mapping integers back to their original labels.
    """

    def __init__(self, encoding_dict: Optional[Dict[Union[str, int], int]] = None) -> None:
        """Intializes the DictLabelEncoder class.

        Args:
            encoding_dict: A dictionary mapping labels to integers.
                If this is not provided, the class will generate a mapping
                based on the unique labels it encounters during fit.
        """
        self.encoding_dict: Optional[Dict[Union[str, int], int]] = encoding_dict
        self.inverse_dict: Optional[Dict[int, Union[str, int]]] = None
        if self.encoding_dict is not None:
            self.inverse_dict = {v: k for k, v in self.encoding_dict.items()}

    def fit(self, input_data: pd.DataFrame, labels: Sequence[Union[str, int]]) -> DictLabelEncoder:
        """Fit the label encoder based on the provided labels.

        Args:
            labels: A list of labels.

        Returns:
            This class instance.

        Raises:
            Exception: If `fit` is called after the encoder has already been fitted.
        """
        if self.encoding_dict is None:
            unique_labels = sorted(set(labels))
            self.encoding_dict = {label: i for i, label in enumerate(unique_labels)}
            self.inverse_dict = dict(enumerate(unique_labels))
        return self

    def transform(
        self, input_data: pd.DataFrame, labels: Sequence[Union[str, int]]
    ) -> List[Union[str, int]]:
        """Transform the provided labels using the encoding dictionary.

        Args:
            labels: A list of labels.

        Returns:
            A list of encoded labels.

        Raises:
            Exception: If `transform` is called before the encoder has been fitted.
        """
        if self.encoding_dict is None:
            raise Exception("The encoder has not been fitted yet. Call the 'fit' method first.")
        return [
            self.encoding_dict[label] if label in self.encoding_dict else label for label in labels
        ]

    def fit_transform(
        self, input_data: pd.DataFrame, labels: Sequence[Union[str, int]]
    ) -> List[Union[str, int]]:
        """Fit the label encoder based on the provided labels, then transform the labels.

        Args:
            labels: A list of labels.

        Returns:
            A list of encoded labels.
        """
        self.fit(input_data, labels)
        return self.transform(input_data, labels)

    def inverse_transform(
        self, input_data: pd.DataFrame, encoded_labels: Sequence[int]
    ) -> List[Union[str, int]]:
        """Transform encoded labels back into their original form using the inverse dictionary.

        Args:
            encoded_labels: A list of encoded labels.

        Returns:
            A list of original labels.

        Raises:
            Exception: If `inverse_transform` is called before the encoder has been fitted.
        """
        if self.inverse_dict is None:
            raise Exception("The encoder has not been fitted yet. Call the 'fit' method first.")
        return [
            self.inverse_dict[label] if label in self.inverse_dict else label
            for label in encoded_labels
        ]


class UniqueIDLabelEncoder(BaseSelector):
    """Encoder that keeps a single y value for each unique ID.

    This class is useful in time series scenarios where each unit (ID) has multiple y values,
    and only one value needs to be retained.

    Args:
        column_id (str): Name of the column in the input_data DataFrame that contains the IDs.

    Attributes:
        column_id (str): The name of the ID column.
    """

    def __init__(self, column_id: str) -> None:
        """Initializes the UniqueIDLabelEncoder class.

        Args:
            column_id: Name of the column in the input_data DataFrame that contains the IDs.
        """
        self.column_id = column_id

    def fit(self, input_data: pd.DataFrame, labels: pd.Series) -> UniqueIDLabelEncoder:
        """Fit method for the encoder. Does nothing in this implementation.

        Args:
            input_data: DataFrame containing the data.
            labels: Series containing the labels.

        Returns:
            This class instance.
        """
        return self

    def transform(self, input_data: pd.DataFrame, labels: pd.Series) -> pd.Series:
        """Transform method that keeps a single y value for each unique ID.

        Args:
            input_data: DataFrame containing the data.
            labels: Series containing the labels.

        Returns:
            A Series of transformed labels.

        Raises:
            Exception: If there are multiple y values for a single ID.
        """
        grouped = input_data.join(labels, how="inner").groupby(self.column_id)
        if any(grouped.nunique()[labels.name] > 1):
            raise Exception("Multiple y values found for a single ID.")
        return grouped.first()[labels.name].loc[input_data[self.column_id].unique()]

    def fit_transform(self, input_data: pd.DataFrame, labels: pd.Series) -> pd.Series:
        """Fit and transform method. Fits the encoder and then transforms the labels.

        Args:
            input_data: DataFrame containing the data.
            labels: Series containing the labels.

        Returns:
            A Series of transformed labels.
        """
        return self.fit(input_data, labels).transform(input_data, labels)

    def inverse_transform(
        self, input_data: pd.DataFrame, encoded_labels: Union[pd.Series, List]
    ) -> pd.Series:
        """Inverse transform method that expands single y values
        to match all occurrences of each ID.

        Args:
            input_data: DataFrame containing the data.
            encoded_labels: List or Series containing the encoded labels.

        Returns:
            A Series of original labels expanded to match the input data.
        """
        # Convert encoded_labels to a Series if it's not already one
        if not isinstance(encoded_labels, pd.Series):
            encoded_labels = pd.Series(encoded_labels)

        # Ensure that the series has a name, which is necessary for the join operation
        if encoded_labels.name is None:
            encoded_labels.name = "encoded_label"

        # Creating a DataFrame from the encoded_labels for joining
        encoded_labels_df = encoded_labels.to_frame()
        encoded_labels_df[self.column_id] = input_data[self.column_id].unique()
        # Joining the input_data DataFrame with the encoded_labels_df
        joined_df = joined_df = (
            input_data.reset_index()
            .set_index(self.column_id)
            .join(encoded_labels_df.set_index(self.column_id), how="left")
            .reset_index()
            .set_index("index")
        )
        joined_df.index.name = input_data.index.name
        return joined_df.loc[input_data.index].reset_index(drop=True)[encoded_labels.name].to_list()


class SafeCategoricalTransformer(BaseEstimator, TransformerMixin):
    """A safe wrapper for categorical transformers from feature_engine.

    This class applies the transformation only if categorical variables are present.
    It allows using any transformer class from feature_engine by providing flexibility
    in handling datasets with mixed types of variables.

    Args:
        transformer_cls: The feature_engine transformer class to be used.
        kwargs: Arguments to be passed to the transformer_cls.
    """

    def __init__(self, transformer_cls: Type[BaseEstimator], **kwargs: Any) -> None:
        """Initialize the SafeCategoricalTransformer."""
        self.transformer_cls = transformer_cls
        self.kwargs = kwargs
        self.transformer: Optional[BaseEstimator] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "SafeCategoricalTransformer":
        """Fit the transformer on the categorical variables of X, if they exist.

        Args:
            X: DataFrame of shape [n_samples, n_features].
            y: Target variable. Default is None.

        Returns:
            self: The fitted transformer instance.
        """
        # Identify categorical columns
        categorical_cols = X.select_dtypes(include=["category", "object"]).columns

        if len(categorical_cols) > 0:
            # Initialize and fit the actual transformer if categorical columns are present
            self.transformer = self.transformer_cls(**self.kwargs)
            self.transformer.fit(X, y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transformation to the categorical variables of X, if they exist.

        Args:
            X: DataFrame of shape [n_samples, n_features].

        Returns:
            Transformed X: DataFrame with transformed categorical variables.
        """
        if self.transformer is not None:
            X = self.transformer.transform(X)

        return X
