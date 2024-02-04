"""Tabular data manager."""

from __future__ import annotations

import copy
import uuid
from typing import Any, Callable, cast, Dict, List, Optional, Protocol, Tuple, Union

import methodtools
import pandas as pd
from typing_extensions import Self

from simpml.core.base import (
    DataFetcherBase,
    DataManagerBase,
    Dataset,
    DataType,
    PredictionType,
    SplitterBase,
)
from simpml.tabular.data_fetcher_pool import TabularDataFetcher
from simpml.tabular.pipeline import get_data_types, Pipeline, pipeline_decorator, PipelineBuilder
from simpml.tabular.splitter_pool import CrossValidationSplitter, splitter_pool


class BuildPipelineFuncType(Protocol):
    """Class to define the API for the build pipeline function."""

    def __call__(self, **kwargs: Any) -> Pipeline:
        """The actual API for the build pipeline function.

        Args:
            **kwargs: Allow any keyword arguments.

        Returns: The constructed pipeline object.
        """
        ...


class TabularDataManager(DataManagerBase):
    """A class representing a data manager for tabular data.

    Parameters:
        data_fetcher (DataFetcherBase): An object that fetches raw data.
        splitter (SplitterBase): An object that splits data.
        pipeline (TabularPipeline): A pipeline object for processing tabular data.
        prediction_type (str): A string indicating the type of prediction, e.g. "classification" or
            "regression".
        data_type (str): A string indicating the type of data, e.g. "tabular" or "vision".
        **pipeline_kwargs (optional): Additional keyword arguments to be passed to the pipeline.

    Attributes:
        prediction_type (str): An Enum indicating the type of prediction, e.g. "classification" or
            "regression".
        data_type (str): An Enum indicating the type of data, e.g. "tabular" or "text".
        data_fetcher (DataFetcherBase): An object that fetches raw data.
        splitter (SplitterBase): An object that splits data.
        pipeline (Pipeline): A pipeline object for processing tabular data.
    """

    def __init__(
        self,
        data_fetcher: DataFetcherBase,
        splitter: SplitterBase,
        pipeline: Optional[Pipeline],
        prediction_type: Union[str, PredictionType],
        description: str = "",
    ) -> None:
        """Initializes the TabularDataManager class.

        Args:
            data_fetcher: An object that fetches raw data.
            splitter: An object that splits data.
            pipeline: A pipeline object for processing tabular data.
            prediction_type: Indicates the type of prediction, e.g. "classification" or
                "regression".
            description: The description of the data manager.
        """
        super().__init__(description)
        self.data_type: str = DataType.Tabular.value
        self.prediction_type: str = (
            prediction_type if isinstance(prediction_type, str) else prediction_type.value
        )
        self.data_fetcher: DataFetcherBase = data_fetcher
        self.splitter: SplitterBase = splitter
        self.pipeline: Optional[Pipeline] = pipeline
        self.data: pd.DataFrame = pd.DataFrame()
        self.load_and_split_data()

    def load_and_split_data(self) -> None:
        """Load and split the data using the initialized data fetcher and splitter."""
        self.data = self.data_fetcher.get_items()
        self.indices = self.splitter.split(data=self.data)

    @methodtools.lru_cache()  # type: ignore [misc]
    def _get_data(
        self, pipeline: Optional[Pipeline], dataset_name: Union[str, Dataset] = Dataset.Train
    ) -> Union[
        Tuple[pd.DataFrame, Optional[pd.Series]], Tuple[List[pd.DataFrame], List[pd.Series]]
    ]:
        """Returns the data (X and y) after applying the pipeline transformations on specific
        dataset.

        Args:
            pipeline: Pipeline to use for data preprocessing.
            dataset_name: Name of the dataset to get target values for. Default is
                `Dataset.Train`.

        Returns:
            Target values for the specified dataset.
        """
        # Convert dataset_name to string if it's a member of the Dataset enum
        dataset: Dataset = (
            Dataset.from_value(dataset_name) if isinstance(dataset_name, str) else dataset_name
        )

        # Check if dataset_name is valid
        if hasattr(self.splitter, "split_sets"):
            if dataset not in self.splitter.split_sets.keys():
                raise Exception(
                    f"Dataset '{dataset}' does not exist. Available datasets are: "
                    f"{list(self.splitter.split_sets.keys())}"
                )

        X, y = self.get_dataset_from_indices(dataset)

        if dataset == Dataset.Train:
            if pipeline:
                X, y = pipeline.fit_transform_manipulate(X=X, y=y)
        else:
            if pipeline:
                X, y = pipeline.transform(X=X, y=y)

        return X, y

    def get_data(
        self, dataset_name: Union[str, Dataset] = Dataset.Train
    ) -> Union[
        Tuple[pd.DataFrame, Optional[pd.Series]], Tuple[List[pd.DataFrame], List[pd.Series]]
    ]:
        """Returns the data (X and y) after applying the pipeline transformations on specific
        dataset.

        Args:
            dataset_name: The data set to use (e.g. `Dataset.Train`).

        Returns:
            2-tuple containing processed dataset X and target variable y.
        """
        return self._get_data(self.pipeline, dataset_name)

    def get_x(
        self, dataset_name: Union[str, Dataset] = Dataset.Train
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Returns the features (X) after applying the pipeline transformations on specific dataset.

        Args:
            dataset_name: The data set to use (e.g. `Dataset.Train`).

        Returns:
            The processed dataset X.
        """
        return self._get_data(self.pipeline, dataset_name)[0]

    def get_y(
        self, dataset_name: Union[str, Dataset] = Dataset.Train
    ) -> Union[Optional[pd.Series], List[pd.Series]]:
        """Returns the target variable (y) after applying the pipeline transformations on a
        specified dataset.

        Args:
            dataset_name: The data set to use (e.g. `Dataset.Train`).

        Returns:
            The target variable (y).
        """
        return self._get_data(self.pipeline, dataset_name)[1]

    def get_training_data(self) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Returns the training data (X and y) after applying the pipeline transformations.

        Returns:
            2-tuple containing processed training dataset X and target variable y.
        """
        return self.get_x(), self.get_y()

    def get_validation_data(self) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Returns the Validation data (X and y) after applying the pipeline transformations.

        Returns:
            2-tuple containing processed training dataset X and target variable y
        """
        return self.get_x(Dataset.Valid.name), self.get_y(Dataset.Valid.name)

    def get_prediction_type(self) -> str:
        """Returns the prediction type for this TabularDataManager instance.

        Returns:
            A string representing the prediction type, such as "binary" or "multiclass".
        """
        return self.prediction_type

    def get_data_type(self) -> str:
        """Returns the data type for this TabularDataManager instance.

        Returns:
            A string representing the data type, such as "Tabular" or "Vision".
        """
        return self.data_type

    def set_description(self, description: str) -> None:
        """Set the description of the data manager.

        Args:
            description: The description of the data manager.
        """
        self.description = description

    def clone(self) -> Self:
        """Creates a copy of this class instance.

        Returns:
            A copy of this class instance.
        """
        new_data_manage = copy.deepcopy(self)
        new_data_manage.id = str(uuid.uuid4())[:8]
        return new_data_manage

    def _get_dataset_from_indices(
        self, dataset_name: Union[str, Dataset], target: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Get a dataset from the data set name.

        Args:
            dataset_name: The data set name.
            target: The name of the independent (target) variable.

        Returns:
            A 2-tuple of the feature (X) and target (y) data for the data set.
        """
        dataset: Dataset = (
            Dataset.from_value(dataset_name) if isinstance(dataset_name, str) else dataset_name
        )
        if dataset not in self.indices:
            raise ValueError(f"No data split named '{dataset}' exists.")
        if hasattr(self.splitter, "target"):
            if self.splitter.target is not None:
                target = self.splitter.target
        indices = self.indices[dataset]

        if isinstance(indices, tuple):
            indices = self.indices[dataset][0].index

        if target is not None:
            return self.data.loc[indices].drop([target], axis=1), self.data.loc[indices, target]
        else:
            return self.data.loc[indices], None

    def get_dataset_from_indices(
        self, dataset_name: Union[str, Dataset], target: Optional[str] = None
    ) -> Union[
        Tuple[pd.DataFrame, Optional[pd.Series]], Tuple[List[pd.DataFrame], List[pd.Series]]
    ]:
        """Get a dataset from the data set name.

        Args:
            dataset_name: The data set name.

        Returns:
            A 2-tuple of the feature (X) and target (y) data for the data set.
        """
        return self._get_dataset_from_indices(dataset_name)


class SupervisedTabularDataManager(TabularDataManager):
    """Data manager for supervised tabular data."""

    def __init__(
        self,
        data: Union[str, pd.DataFrame],
        target: Optional[str],
        prediction_type: PredictionType,
        splitter: Union[str, SplitterBase] = "Random",
        data_types: Optional[Dict[str, bool]] = None,
    ) -> None:
        """Initializes the SupervisedTabularDataManager class.

        Args:
            data: If a string or PathLike, this is a file path to a CSV file that contains
                the data. Otherwise, this is the data frame of the data itself.
            target: The name of the independent (target) variable.
            prediction_type: Indicates the type of prediction, e.g. "classification" or
                "regression".
            splitter: An object that splits data or the object name from the splitter pool.
            data_types: Optional dictionary with key names from the data types (from the
                VariableType Enum) and boolean values as to whether they are used.
        """
        if isinstance(splitter, str):
            splitter = splitter_pool[splitter](target=target)
        super().__init__(
            data_fetcher=TabularDataFetcher(data=data),
            splitter=splitter,
            pipeline=None,
            prediction_type=prediction_type,
        )
        self.target = target
        self.data_types: Dict[str, bool] = (
            get_data_types(self.data_fetcher.get_items(), target=target)
            if data_types is None
            else data_types
        )
        self.build_pipeline: BuildPipelineFuncType = pipeline_decorator(self)(
            PipelineBuilder().get_pipeline_func_based_on_prediction_type(
                self.get_prediction_type(),
                target,
                self.data_types,
                cols_to_drop=getattr(splitter, "cols_to_drop", None),
            )
        )

    def load_pipeline(self, pipeline: Pipeline) -> None:
        """Use the specified pipeline.

        Args:
            pipeline: The pipeline to use.

        Raises:
            ValueError: If `pipeline` is not a valid `Pipeline` object.
        """
        if not isinstance(pipeline, Pipeline):
            raise ValueError(
                f"pipeline {type(pipeline)} not supported. Please use a pipeline of type "
                "simpml.tabular.pipeline.Pipeline"
            )
        self.pipeline = pipeline

    def get_dataset_from_indices(  # type: ignore [override]
        self, dataset_name: Union[str, Dataset]
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Get a dataset from the data set name.

        Args:
            dataset_name: The data set name.

        Returns:
            A 2-tuple of the feature (X) and target (y) data for the data set.
        """
        return self._get_dataset_from_indices(dataset_name, self.target)


class UnsupervisedTabularDataManager(TabularDataManager):
    """Data manager for unsupervised tabular data."""

    def __init__(
        self,
        data: Union[str, pd.DataFrame],
        prediction_type: PredictionType,
        target: Optional[str] = None,
        splitter: Union[str, SplitterBase] = "Random",
        data_types: Optional[Dict[str, bool]] = None,
    ) -> None:
        """Initializes the UnsupervisedTabularDataManager class.

        Args:
            data: If a string or PathLike, this is a file path to a CSV file that contains
                the data. Otherwise, this is the data frame of the data itself.
            prediction_type: Indicates the type of prediction, e.g. "classification" or
                "regression".
            target: The name of the independent (target) variable.
            splitter: An object that splits data or the object name from the splitter pool.
            data_types: Optional dictionary with key names from the data types (from the
                VariableType Enum) and boolean values as to whether they are used.
        """
        if isinstance(splitter, str):
            splitter = splitter_pool[splitter](target=target)
        super().__init__(
            data_fetcher=TabularDataFetcher(data=data),
            splitter=splitter,
            pipeline=None,
            prediction_type=prediction_type,
        )
        self.data_types: Dict[str, bool] = (
            get_data_types(self.data_fetcher.get_items(), target=target)
            if data_types is None
            else data_types
        )
        self.build_pipeline: Callable[[], Pipeline] = pipeline_decorator(self)(
            PipelineBuilder().get_pipeline_func_based_on_prediction_type(
                self.get_prediction_type(),
                None,
                self.data_types,
                cols_to_drop=getattr(splitter, "cols_to_drop", None),
            )
        )

    def load_pipeline(self, pipeline: Pipeline) -> None:
        """Use the specified pipeline.

        Args:
            pipeline: The pipeline to use.

        Raises:
            ValueError: If `pipeline` is not a valid `Pipeline` object.
        """
        if not isinstance(pipeline, Pipeline):
            raise ValueError(
                f"pipeline {type(pipeline)} not supported. Please use a pipeline of type "
                "simpml.tabular.pipeline.Pipeline"
            )
        self.pipeline = pipeline

    def get_dataset_from_indices(  # type: ignore [override]
        self, dataset_name: Union[str, Dataset]
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Get a dataset from the data set name.

        Args:
            dataset_name: The data set name.

        Returns:
            A 2-tuple of the feature (X) and target (y) data for the data set.
        """
        return self._get_dataset_from_indices(dataset_name)


class CrossValidationSupervisedTabularDataManager(SupervisedTabularDataManager):
    """Data manager for supervised tabular data with cross-validation."""

    def __init__(
        self,
        data: Union[str, pd.DataFrame],
        target: Optional[str],
        prediction_type: PredictionType,
        splitter: Union[str, SplitterBase] = "KFold",
        data_types: Optional[Dict[str, bool]] = None,
        n_folds: int = 5,
        stratify: bool = False,
    ) -> None:
        """Initializes the CrossValidationSupervisedTabularDataManager class.

        Args:
            data: If a string or PathLike, this is a file path to a CSV file that contains
                the data. Otherwise, this is the data frame of the data itself.
            target: The name of the independent (target) variable.
            prediction_type: Indicates the type of prediction, e.g. "classification" or
                "regression".
            splitter: An object that splits data or the object name from the splitter pool.
            data_types: Optional dictionary with key names from the data types (from the
                VariableType Enum) and boolean values as to whether they are used.
            n_folds: Number of folds for cross-validation.
            stratify: make splits stratify.
        """
        if isinstance(splitter, str):
            splitter_class = splitter_pool[splitter]
            if issubclass(splitter_class, CrossValidationSplitter):
                splitter = splitter_class(
                    target=target,
                    n_folds=n_folds,
                    prediction_type=prediction_type,
                    stratify=stratify,
                )
            else:
                splitter = splitter_class(target=target)
        super().__init__(
            data=data,
            target=target,
            splitter=splitter,
            prediction_type=prediction_type,
            data_types=data_types,
        )
        self.target = target
        self.data_types: Dict[str, bool] = (
            get_data_types(self.data_fetcher.get_items(), target=target)
            if data_types is None
            else data_types
        )
        self.build_pipeline: BuildPipelineFuncType = pipeline_decorator(self)(
            PipelineBuilder().get_pipeline_func_based_on_prediction_type(
                self.get_prediction_type(),
                target,
                self.data_types,
                cols_to_drop=getattr(splitter, "cols_to_drop", None),
            )
        )
        self.selected_fold: Optional[int] = None

    @methodtools.lru_cache()  # type: ignore [misc]
    def _get_data(  # type: ignore [override]
        self, pipeline: Optional[Pipeline], dataset_name: Union[str, Dataset] = Dataset.Train
    ) -> Tuple[List[pd.DataFrame], List[pd.Series]]:
        """Returns the data (X and y) after applying the pipeline transformations on specific
        dataset.

        Args:
            pipeline: Pipeline to use for data preprocessing.
            dataset_name: Name of the dataset to get target values for. Default is
                `Dataset.Train`.

        Returns:
            Two lists of data for the specified dataset, one for X and one for y.
        """
        if not hasattr(self, "pipeline_folds"):
            if not hasattr(self.splitter, "n_folds"):
                raise ValueError(f"splitter of type {type(self.splitter)} has no 'n_folds' member")
            self.pipeline_folds = [
                copy.deepcopy(self.pipeline) for i in range(self.splitter.n_folds)
            ]
        # Convert dataset_name to string if it's a member of the Dataset enum
        dataset: Dataset = (
            Dataset.from_value(dataset_name) if isinstance(dataset_name, str) else dataset_name
        )

        # Check if dataset_name is valid
        if hasattr(self.splitter, "split_sets"):
            if dataset not in self.splitter.split_sets.keys():
                raise Exception(
                    f"Dataset '{dataset}' does not exist. Available datasets are: "
                    f"{list(self.splitter.split_sets.keys())}"
                )

        X, y = self.get_dataset_from_indices(dataset)

        transformed_data_X = []
        transformed_data_y = []
        if self.pipeline_folds:
            assert y is not None
            for X_fold, y_fold, pipeline in zip(X, y, self.pipeline_folds):
                if dataset == Dataset.Train:
                    if pipeline:
                        X_transformed, y_transformed = pipeline.fit_transform_manipulate(
                            X=X_fold, y=y_fold
                        )
                    else:
                        X_transformed = X_fold
                        y_transformed = y_fold
                else:
                    if pipeline:
                        X_transformed, y_transformed = pipeline.transform(X=X_fold, y=y_fold)
                    else:
                        X_transformed = X_fold
                        y_transformed = y_fold

                transformed_data_X.append(X_transformed)
                transformed_data_y.append(y_transformed)
        else:
            assert y is not None
            for X_fold, y_fold in zip(X, y):
                X_transformed, y_transformed = X_fold, y_fold

                transformed_data_X.append(X_transformed)
                transformed_data_y.append(y_transformed)

        return transformed_data_X, transformed_data_y

    def get_dataset_from_indices(  # type: ignore [override]
        self, dataset_name: Union[str, Dataset]
    ) -> Tuple[List[pd.DataFrame], List[pd.Series]]:
        """Get a dataset from the data set name.

        Args:
            dataset_name: The data set name.

        Returns:
            A 2-tuple of the feature (X) and target (y) data for the data set.
        """
        dataset: Dataset = (
            Dataset.from_value(dataset_name) if isinstance(dataset_name, str) else dataset_name
        )
        if dataset not in self.indices:
            raise ValueError(f"No data split named '{dataset}' exists.")

        indices = self.indices[dataset]
        return [self.data.loc[indice].drop([self.target], axis=1) for indice in indices], [
            self.data.loc[indice, self.target] for indice in indices
        ]

    def get_training_data(
        self, fold: Optional[int] = None, all_: bool = False
    ) -> Union[
        Tuple[pd.DataFrame, Optional[pd.Series]], Tuple[List[pd.DataFrame], List[pd.Series]]
    ]:
        """Returns the training data (X and y) after applying the pipeline transformations.

        Returns:
            2-tuple containing processed training dataset X and target variable y.
        """
        if fold is None:
            fold = self.get_selected_fold()
        x = cast(List[pd.DataFrame], self.get_x())
        y = cast(List[pd.Series], self.get_y())
        if all_:
            return x, y
        else:
            return x[fold], y[fold]

    def get_validation_data(
        self, fold: Optional[int] = None, all_: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Returns the Validation data (X and y) after applying the pipeline transformations.

        Returns:
            2-tuple containing processed training dataset X and target variable y
        """
        if fold is None:
            fold = self.get_selected_fold()
        x = cast(List[pd.DataFrame], self.get_x(Dataset.Valid))
        y = cast(List[pd.Series], self.get_y(Dataset.Valid))
        if all_:
            return x, y
        else:
            return x[fold], y[fold]

    def get_selected_fold(self) -> int:
        """Get the currently selected fold.

        Returns:
            The currently selected fold or 0 if no fold is selected.
        """
        if self.selected_fold is None:
            return 0
        else:
            return self.selected_fold

    def set_selected_fold(self, fold: Optional[int]) -> None:
        """Set the currently selected fold.

        Args:
            fold: The fold to select.
        """
        self.selected_fold = fold
