"""Vision splitter pool."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastcore.foundation import L

from simpml.core.base import Dataset, PredictionType
from simpml.tabular.splitter_pool import (
    CrossValidationSplitter,
    DEFAULT_RANDOM_SPLIT,
    RandomSplitter,
)


class FastaiCrossValidationSplitterAdapter:
    """Adapter for Fastai cross-validation splitter.

    This class adapts the Fastai cross-validation splitter to be used with custom data splits.
    """

    def __init__(
        self,
        get_y: Optional[Callable] = None,
        test_size: float = 0.2,
        stratify: bool = True,
        random_state: int = 42,
        n_folds: int = 5,
    ):
        """Initialize the FastaiCrossValidationSplitterAdapter.

        Args:
            get_y (Optional[Callable], optional): Function to extract target
            variable. Defaults to None.
            test_size (float, optional): Proportion of the dataset to include
            in the test split. Defaults to 0.2.
            stratify (bool, optional): Whether to stratify the split. Defaults to True.
            random_state (int, optional): Seed for random number generator. Defaults to 42.
            n_folds (int, optional): Number of folds for cross-validation. Defaults to 5.
        """
        self.get_y = get_y
        self.splitter = CrossValidationSplitter(
            target="target" if get_y else None,
            test_size=test_size,
            stratify=stratify,
            random_state=random_state,
            n_folds=n_folds,
            prediction_type=PredictionType.BinaryClassification
        )
        self.current_fold = 0

    def split_data(self, items: list) -> List[Tuple[list, list]]:
        """Split the data into training and validation sets based on the provided items.

        Args:
            items (list): List of items to be split.

        Returns:
            List[Tuple[list, list]]: List of tuples containing
            training and validation indices for each fold.
        """
        # Convert items to dataframe
        df = (
            pd.DataFrame({"items": np.array(items)})
            if self.get_y is None
            else pd.DataFrame(
                {"items": np.array(items), "target": [self.get_y(item) for item in items]}
            )
        )

        # Split using the CrossValidationSplitter
        splits = self.splitter.split(df)

        # Get the train and validation indices for each fold.
        fold_splits = [
            (L(train_idx.tolist()), L(valid_idx.tolist()))
            for train_idx, valid_idx in zip(splits[Dataset.Train], splits[Dataset.Valid])
        ]

        return fold_splits

    def __call__(self, items: list) -> Tuple[list, list]:
        """Call the splitter to obtain train and validation
        indices for the current fold.

        Args:
            items (list): List of items to be split.

        Returns:
            Tuple[list, list]: Tuple containing training and
            validation indices for the current fold.
        """
        all_splits = self.split_data(items)

        # Return indices for the current fold and increment the fold counter.
        result = all_splits[self.current_fold]
        self.current_fold = (self.current_fold + 1) % len(all_splits)

        return result


class FastaiRandomSplitterAdapter:
    """Adapter for Fastai random splitter.

    This class adapts the Fastai random splitter for custom data splits.
    """

    def __init__(
        self,
        get_y: Optional[Callable] = None,
        split_sets: Dict[Dataset, float] = DEFAULT_RANDOM_SPLIT,  # Set the default value
        stratify: bool = True,
        random_state: int = 42,
        splitter_cls: Optional[
            Callable
        ] = None,  # Default is None, will initialize RandomSplitter if not provided
    ):
        """Initialize the FastaiRandomSplitterAdapter.

        Args:
            get_y (Optional[Callable], optional): Function to
            extract the target variable. Defaults to None.
            split_sets (Dict[Dataset, float], optional): Proportions
            for splitting datasets. Defaults to DEFAULT_RANDOM_SPLIT.
            stratify (bool, optional): Whether to stratify the split. Defaults to True.
            random_state (int, optional): Seed for random number generator. Defaults to 42.
            splitter_cls (Optional[Callable], optional): Custom splitter class. Defaults to None.
        """
        self.get_y = get_y
        if splitter_cls is not None:
            self.splitter = splitter_cls(
                target="target" if self.get_y else None,
                split_sets=split_sets,
                stratify=stratify,
                random_state=random_state,
            )
        else:
            self.splitter = RandomSplitter(
                target="target" if self.get_y else None,
                split_sets=split_sets,  # Already defaults to DEFAULT_RANDOM_SPLIT
                stratify=stratify,
                random_state=random_state,
            )

    def __call__(self, items: list) -> Tuple[list, list]:
        """Call the splitter to obtain train and validation indices.

        Args:
            items (list): List of items to be split.

        Returns:
            Tuple[list, list]: Tuple containing training and validation indices.
        """
        # Convert items to dataframe
        df = (
            pd.DataFrame({"items": np.array(items)})
            if self.get_y is None
            else pd.DataFrame(
                {"items": np.array(items), "target": [self.get_y(item) for item in items]}
            )
        )

        # Split using the RandomSplitter
        splits = self.splitter.split(df)

        # Get the train and validation indices.
        train_idx = L(splits[Dataset.Train].tolist())
        valid_idx = L(splits[Dataset.Valid].tolist())

        return train_idx, valid_idx
