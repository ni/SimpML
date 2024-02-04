"""Tabular splitter pool."""

from __future__ import annotations

from typing import Any, cast, Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from simpml.core.base import Dataset, PredictionType, SplitterBase

DEFAULT_RANDOM_SPLIT: Dict[Dataset, float] = {
    Dataset.Train: 0.6,
    Dataset.Valid: 0.2,
    Dataset.Test: 0.2,
}
DEFAULT_TRAIN_ONLY_SPLIT: Dict[Dataset, float] = {Dataset.Train: 1.0, Dataset.Valid: 1.0}


class RandomSplitter(SplitterBase):
    """A class to split a dataset randomly into specified proportions using train_test_split from
    sklearn.

    Attributes:
        split_sets (dict): A dictionary specifying the proportions for each dataset split
            (e.g., {Dataset.Train: 0.6, Dataset.Valid: 0.2, Dataset.Test: 0.2}).
        random_state (int): A random seed for reproducibility. Default is 42.
        target (str, optional): The name of the target column in the dataset. If provided, the
            stratify option will be used in train_test_split to ensure proportional distribution
            of target classes. Default is None.
    """

    def __init__(
        self,
        target: Optional[str] = None,
        split_sets: Dict[Dataset, float] = DEFAULT_RANDOM_SPLIT,
        stratify: bool = True,
        random_state: int = 42,
    ) -> None:
        """Initializes the RandomSplitter class.

        Args:
            split_sets: A dictionary specifying the proportions for each dataset split.
            target: The name of the target column in the dataset. Default is None. Need for
                stratify split, if None will be fully random.
            stratify: The stratify option ensures that the distribution of target classes in each
                split is approximately the same as the overall dataset.  When target is None, the
                data will be split randomly without considering the target class distribution.
                This can be useful when the target column is not known, or when stratification is
                not necessary, such as in unsupervised learning tasks.
            random_state: A random seed for reproducibility. Default is 42.
        """
        self.split_sets: Dict[Dataset, float] = split_sets
        self.random_state: int = random_state
        self.target: Optional[str] = target
        self.stratify: bool = stratify

    def split(self, data: pd.DataFrame) -> Dict[Dataset, pd.Index]:
        """Splits the input data into the specified proportions.

        Args:
            data: The input dataset to be split.

        Returns:
            A dictionary containing the split datasets. The keys are Dataset enum values
            (e.g., Dataset.Train, Dataset.Valid, Dataset.Test) and the values are the
            indexes corresponding to data subsets.
        """
        indices = {}
        remaining_data = data.copy()

        for dataset, proportion in self.split_sets.items():
            if dataset == list(self.split_sets.keys())[-1]:
                # For the last split, assign all remaining data
                indices[dataset] = remaining_data.index
            else:
                # For other splits, calculate split size as a proportion of total data size
                split_size = int(proportion * len(data))

                stratify = remaining_data[self.target] if self.target and self.stratify else None
                split_data, remaining_data = train_test_split(
                    remaining_data,
                    train_size=split_size,
                    stratify=stratify,
                    random_state=self.random_state,
                )
                indices[dataset] = split_data.index

        return indices


class CrossValidationSplitter(SplitterBase):
    """Splits input dataset by cross validation folds."""

    def __init__(
        self,
        prediction_type: Union[str, PredictionType],
        target: Optional[str] = None,
        test_size: float = 0.2,
        stratify: bool = True,
        random_state: int = 42,
        n_folds: int = 1,
    ) -> None:
        """Initializes the CrossValidationSplitter class.

        Args:
            prediction_type: the type of prediction, its ENUM.
            split_sets: A dictionary specifying the proportions for each dataset split.
            target: The name of the target column in the dataset. Default is None. Need for
                stratify split, if None will be fully random.
            test_size: The size of the test dataset.
            stratify: The stratify option ensures that the distribution of target classes in each
                split is approximately the same as the overall dataset.  When target is None, the
                data will be split randomly without considering the target class distribution.
                This can be useful when the target column is not known, or when stratification is
                not necessary, such as in unsupervised learning tasks.
            random_state: A random seed for reproducibility. Default is 42.
            n_folds: The number of folds in the cross validation dataset.
        """
        self.prediction_type: str = (
            prediction_type if isinstance(prediction_type, str) else prediction_type.value
        )
        self.is_classification = (
            self.prediction_type == PredictionType.MulticlassClassification.value
            or self.prediction_type == PredictionType.BinaryClassification.value
        )
        self.target: Optional[str] = target
        self.stratify: bool = stratify
        self.random_state: int = random_state
        self.test_size: float = test_size
        self.n_folds: int = n_folds
        remaining: float = 1 - self.test_size
        self.split_sets: Dict[Dataset, float] = {
            Dataset.Train: round(
                remaining * ((self.n_folds - 1) / self.n_folds), 2
            ),  # The remaining data times the proportion of folds for training
            Dataset.Valid: round(
                remaining / self.n_folds, 2
            ),  # The remaining data divided by the number of folds
            Dataset.Test: self.test_size,  # The given part for testing
        }

    def _split_n_folds(self, data: pd.DataFrame) -> List[Tuple[pd.Index, Optional[pd.Index]]]:
        """Splits the input data into the configured number of folds.

        Args:
            data: The input dataset to be split.

        Returns:
            A list containing the split datasets. Each is a 2-tuple with the first
            being the indexes for the Training Set and the second is corresponding
            to the Valid Set indexes.
        """
        folds = []
        if self.is_classification:
            kf = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.random_state
            )
        else:
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for train_index, valid_index in kf.split(data, data[self.target]):
            folds.append(
                (
                    data.iloc[train_index].index,
                    data.iloc[valid_index].index if self.target else None,
                )
            )

        return folds

    def split(self, data: pd.DataFrame) -> Dict[Dataset, pd.Index]:
        """Splits the input data into the specified proportions.

        Args:
            data: The input dataset to be split.

        Returns:
            A dictionary containing the split datasets. The keys are Dataset enum values
            (e.g., Dataset.Train, Dataset.Valid, Dataset.Test) and the values are the
            indexes corresponding to data subsets.
        """
        indices = {}

        if self.n_folds > 1:
            train_valid_data, test_data = train_test_split(
                data,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=data[self.target] if self.stratify and self.target else None,
            )
            folds = self._split_n_folds(train_valid_data)
            indices[Dataset.Train] = [fold[0] for fold in folds]
            indices[Dataset.Valid] = [fold[1] for fold in folds]
            indices[Dataset.Test] = test_data.index
        else:
            raise ValueError(
                "Number of folds must be greater than 1 for k-fold cross-validation. "
                "Consider using RandomSplitter for random splitting."
            )

        return indices

    def plot_cross_validation(self, data: pd.DataFrame) -> plt.Figure:
        """Plot the cross validation.

        Args:
            data: The input dataset to plot.

        Returns:
            A matplotlib Figure object containing the plots.
        """
        indices = self.split(data)
        # Concatenate the indices into a single Series
        values_series = pd.Series(
            [item for sublist in indices[Dataset.Valid] for item in sublist.tolist()]
        )

        # Create the fold column using a list comprehension
        fold_series = pd.Series(
            [i for i, index in enumerate(indices[Dataset.Valid]) for _ in range(len(index))]
        )

        # Create the DataFrame by combining the value and fold Series
        df = pd.DataFrame({"value": values_series, "fold": fold_series})

        # Create plot
        fig, ax = plt.subplots(figsize=(12, self.n_folds))

        # Define color for validation and training
        validation_color = "b"
        training_color = "r"
        folds = df["fold"].unique()
        for fold in folds:
            # For each fold, get validation and training indices
            validation_indices = df[df["fold"] == fold].index
            training_indices = df[df["fold"] != fold].index

            # Plot validation indices
            ax.scatter(
                validation_indices,
                [fold] * len(validation_indices),
                color=validation_color,
                label="validation",
                s=250,
                marker="s",
            )

            # Plot training indices as squares

            # Plot training indices
            ax.scatter(
                training_indices,
                [fold] * len(training_indices),
                color=training_color,
                label="training" if fold == 0 else "",
                s=250,
                marker="s",
            )

            # Add text to describe fold
            ax.text(
                min(df.index) - 60, fold, f"Fold {fold}", fontsize=12, verticalalignment="center"
            )

        # Avoid duplication in legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        # Place the legend outside of the plot area at the upper right
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", bbox_to_anchor=(1.15, 1))

        ax.set_title(f"Cross-Validation Visualization: {str(self.n_folds)} folds")
        ax.set_xlabel("Indices")
        ax.set_ylabel("Fold")
        ax.axis("off")
        return fig


class DateTimeSplitter(SplitterBase):
    """Splits input dataset by time series."""

    def __init__(
        self, target: Optional[str], split_sets: Dict[Dataset, float], time_column: str
    ) -> None:
        """Initializes the DateTimeSplitter class.

        Args:
            target: The name of the independent (target) variable.
            split_sets: Dictionary specifying the number of samples to be allocated for each split
            time_column: Name of the time column in the dataset
        """
        self.target: Optional[str] = target
        self.split_sets: Dict[Dataset, float] = split_sets
        self.time_column: str = time_column

    def split(self, data: pd.DataFrame) -> Dict[Dataset, pd.Index]:
        """This method splits input dataset by time series.

        The size of each split is determined by the percentage split specified in the "split_sets"
        dictionary. The input dataset must have a time column specified by "time_column", which is
        used to sort the data in ascending order, such the datasets don't overlapping each other.

        Args:
            data: Input dataset to be split.

        Returns:
            A dictionary containing the split datasets. The keys are Dataset enum values
            (e.g., Dataset.Train, Dataset.Valid, Dataset.Test) and the values are the
            indexes corresponding to data subsets.
        """
        # Create an empty dictionary to store the output splits
        data_sets = {}
        split_abs_values = np.floor(
            np.array(list(self.split_sets.values())) * data.shape[0]
        ).astype(int)
        split_abs_values[split_abs_values == 0] = 1

        # Sort the input dataset by the time column in ascending order
        dataset_second = data.copy()
        dataset_second = dataset_second.sort_values(self.time_column)

        for i in range(len(self.split_sets) - 1):
            # Select the first split of the data
            dataset_first = dataset_second.iloc[: split_abs_values[i],]

            # Update the second split to start after the end of the first split
            dataset_second = dataset_second.iloc[split_abs_values[i] :,]

            # Add the first split to the "data_sets" dictionary
            data_sets[list(self.split_sets.keys())[i]] = dataset_first.index

        # Add the remaining data to the "data_sets" dictionary
        data_sets[
            list(self.split_sets.keys())[len(list(self.split_sets.values())) - 1]
        ] = dataset_second.index

        return data_sets


class GroupSplitter:
    """Splits a dataset into multiple subsets based on categorical
    columns, with consideration for subset sizes.
    """

    def __init__(
        self,
        split_sets: Dict[str, float],
        group_columns: str,
        random_state: int = 42,
        copy: bool = False,
        drop_group_cols: bool = True,
    ):
        """Initializes the GroupSplitter class.

        Args:
            split_sets: A dictionary with subset names as keys and
            the ratios of the dataset to allocate to each subset as values.
            group_columns: The name of the column containing the grouping information for the split.
            random_state: The random seed for reproducibility.
            copy: make copy from the data
            drop_group_cols: drop the group columns
        """
        self.split_sets = split_sets
        self.group_columns = (
            [group_columns] if not isinstance(group_columns, list) else group_columns
        )
        self.random_state = random_state
        self.copy = copy
        self.drop_group_cols = drop_group_cols

    def split(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Splits the given DataFrame into subsets based on predefined split ratios.

        This method uses a group-based approach to ensure that data associated with
        the same group (defined by 'group_columns') goes into the same subset.
        It handles grouping, random shuffling, and allocation of groups to different subsets.

        Args:
            data: A pandas DataFrame to be split into subsets.

        Returns:
            A dictionary where keys are subset names and values are pandas Series
            containing the indices of rows in 'data' that belong to each subset.
        """
        if not np.isclose(sum(self.split_sets.values()), 1.0):
            raise ValueError("The sum of the split ratios must be 1.")

        if self.copy:
            data = data.copy()

        # Create a group key column using vectorized operations
        data["_group_key"] = data[self.group_columns[0]].astype(str)
        for col in self.group_columns[1:]:
            data["_group_key"] = data["_group_key"] + "-" + data[col].astype(str)
        unique_groups = data["_group_key"].unique()

        np.random.seed(self.random_state)
        if len(unique_groups) < len(self.split_sets):
            raise ValueError("Not enough unique groups for the requested number of sets.")

        data["_group_key"] = data["_group_key"].fillna("None")

        # Sort subsets by requested size
        sorted_split_sets = sorted(self.split_sets.items(), key=lambda x: x[1], reverse=True)

        # Count group sizes using vectorized operations
        group_sizes = data.groupby("_group_key").size()

        # Sort groups by size
        sorted_groups = group_sizes.sort_values(ascending=False).index

        data_sets: Dict[str, List[Any]] = {set_name: [] for set_name in self.split_sets}
        self.allocated_sizes: Dict[str, float] = {set_name: 0 for set_name in self.split_sets}
        self.group_allocation: Dict[str, List] = {set_name: [] for set_name in self.split_sets}

        # Allocate one group to each subset based on size
        for (set_name, _), group in zip(sorted_split_sets, sorted_groups):
            group_size = group_sizes[group]
            self.allocated_sizes[set_name] += group_size
            self.group_allocation[set_name].append(group)

        # Remaining groups
        remaining_groups = set(sorted_groups[len(sorted_split_sets) :])
        np.random.shuffle(list(remaining_groups))

        # Allocate remaining groups
        total_size = len(data)
        for group in remaining_groups:
            group_size = group_sizes[group]

            # Vectorized allocation
            allocation_metric = {
                set_name: self.split_sets[set_name]
                - (self.allocated_sizes[set_name] + group_size) / total_size
                for set_name in data_sets
            }
            best_set = max(allocation_metric, key=lambda x: allocation_metric[x])
            self.allocated_sizes[best_set] += group_size
            self.group_allocation[best_set].append(group)

        # Construct the final datasets
        for set_name, groups in self.group_allocation.items():
            data_sets[set_name] = data[data["_group_key"].isin(groups)].drop("_group_key", axis=1)
        data.drop("_group_key", axis=1, inplace=True)
        if self.drop_group_cols:
            self.cols_to_drop = self.group_columns
        return {set_name: df.index for set_name, df in data_sets.items()}


class IndexSplitter(SplitterBase):
    """Splits a dataset into multiple subsets based on specified indices."""

    def __init__(self, split_sets: Dict[Dataset, Any]) -> None:
        """Initializes the IndexSplitter class.

        Args:
            split_sets: A dictionary containing the names of the resulting subsets as keys and
                        the indices of the dataset to be allocated to each subset as values.
                        The indices should be provided as lists of integers.
        """
        self.split_sets = split_sets

    def split(self, data: pd.DataFrame) -> Dict[Dataset, Any]:
        """Splits a dataset into multiple subsets based on specified indices.

        Args:
            data: The dataset to be split.

        Returns:
            A dictionary containing the resulting split subsets, with the names specified in
            split_sets as keys.
        """
        return self.split_sets


class ADSplitter(SplitterBase):
    """A class to split a dataset randomly into specified proportions using train_test_split from
    sklearn.

    Attributes:
        split_sets (dict): A dictionary specifying the proportions for each dataset split
            (e.g., {Dataset.Train: 0.6, Dataset.Valid: 0.2, Dataset.Test: 0.2}).
        random_state (int): A random seed for reproducibility. Default is 42.
        target (str, optional): The name of the target column in the dataset. If provided, the
            stratify option will be used in train_test_split to ensure proportional distribution of
            target classes. Default is None.
    """

    def __init__(
        self,
        target: Optional[str] = None,
        split_sets: Dict[Dataset, float] = DEFAULT_RANDOM_SPLIT,
        stratify: bool = True,
        train_set_name: str = "Train",
        random_state: int = 42,
    ) -> None:
        """Initializes the RandomSplitter class.

        Args:
            split_sets: A dictionary specifying the proportions for each dataset split.
            target: The name of the target column in the dataset. Default is None. need for
                stratify split, if None use fully random.
            stratify: The stratify option ensures that the distribution of target classes in each
                split is approximately the same as the overall dataset. When target is None, the
                data will be split randomly without considering the target class distribution.
                This can be useful when the target column is not known, or when stratification is
                not necessary, such as in unsupervised learning tasks.
            train_set_name: The name of the training set.
            random_state: A random seed for reproducibility. Default is 42.
        """
        self.split_sets: Dict[Dataset, float] = split_sets
        self.random_state: int = random_state
        self.target: Optional[str] = target
        self.stratify: bool = stratify
        self.train_set_name: str = train_set_name

    def split(self, data: pd.DataFrame) -> Dict[Dataset, Tuple[pd.DataFrame, Optional[pd.Series]]]:
        """Splits the input data into the specified proportions.

        Args:
            data: The input dataset to be split.

        Returns:
            A dictionary containing the split datasets. The keys are Dataset enum values
            (e.g., Dataset.Train, Dataset.Valid, Dataset.Test) and the values are the corresponding
            data subsets.
        """
        data_sets = {}
        remaining_data = data
        total_splits = sum(self.split_sets.values())

        for dataset, proportion in self.split_sets.items():
            if (
                dataset == list(self.split_sets.keys())[-1]
            ):  # If it's the last dataset, assign remaining data to it.
                if self.target is not None:
                    data_sets[dataset] = (
                        remaining_data.drop([self.target], axis=1),
                        remaining_data[self.target],
                    )
                else:
                    data_sets[dataset] = (remaining_data, None)
                continue

            train_size = proportion / total_splits
            if self.target and self.stratify:
                train_data, remaining_data = train_test_split(
                    remaining_data,
                    train_size=train_size,
                    random_state=self.random_state,
                    stratify=remaining_data[self.target],
                )
            else:
                train_data, remaining_data = train_test_split(
                    remaining_data, train_size=train_size, random_state=self.random_state
                )
            if dataset.value == self.train_set_name:
                data_sets[dataset] = (train_data, None)
            elif self.target is not None:
                data_sets[dataset] = (
                    train_data.drop([self.target], axis=1),
                    train_data[self.target],
                )
            else:
                data_sets[dataset] = (train_data, None)
            total_splits -= proportion

        return data_sets


class TrainOnlySplitter(SplitterBase):
    """Split into a training set that contains all of the data."""

    def __init__(
        self,
        split_sets: Dict[Dataset, float] = DEFAULT_TRAIN_ONLY_SPLIT,
        target: Optional[str] = None,
    ) -> None:
        """Initializes the TrainOnlySplitter class.

        Args:
            split_sets: A dictionary specifying the proportions for each dataset split.
            target: The name of the target column in the dataset. Default is None. Need for
                stratify split, if None will be fully random.
        """
        self.split_sets = split_sets
        self.target: Optional[str] = target

    def split(self, data: pd.DataFrame) -> Dict[Dataset, pd.Index]:
        """Splits the input data into the specified proportions.

        Args:
            data: The input dataset to be split.

        Returns:
            A dictionary containing the indices for each split. The keys are Dataset enum values
            (e.g., Dataset.Train, Dataset.Valid, Dataset.Test) and the values are the corresponding
            data indices.
        """
        indices: Dict[Dataset, pd.Index] = {}
        for key in self.split_sets.keys():
            indices[key] = data.index

        return indices


class SetsSplitter(SplitterBase):
    """Split into data sets based on column names and column data."""

    def split(self, data: pd.DataFrame) -> Dict[Dataset, Tuple[pd.DataFrame, Optional[pd.Series]]]:
        """Splits the input data into the specified proportions.

        Args:
            data: The input dataset to be split.

        Returns:
            A dictionary containing the split datasets. The keys are Dataset enum values
            (e.g., Dataset.Train, Dataset.Valid, Dataset.Test) and the values are the corresponding
            data subsets.
        """
        data_sets = {}
        for key, value in data.items():
            data_sets[key] = value
        return data_sets


class SplitterPool:
    """Splitter pool."""

    def __init__(self) -> None:
        """Initializes the SplitterPool class."""
        self.RandomSplitter: Type[RandomSplitter] = RandomSplitter
        self.DateTimeSplitter: Type[DateTimeSplitter] = DateTimeSplitter
        self.GroupSplitter: Type[GroupSplitter] = GroupSplitter
        self.IndexSplitter: Type[IndexSplitter] = IndexSplitter


splitter_pool: Dict[str, Type[SplitterBase]] = {
    "Random": RandomSplitter,
    "RCA": TrainOnlySplitter,
    "KFold": cast(Type[SplitterBase], CrossValidationSplitter),
}
