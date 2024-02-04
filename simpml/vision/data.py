"""Vision data."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
from fastai.data.core import DataLoaders, Datasets

from simpml.core.base import DataManagerBase, DataType, PredictionType
from simpml.vision.splitter_pool import FastaiCrossValidationSplitterAdapter


class VisionDataManager(DataManagerBase):
    """Data manager for vision."""

    def __init__(
        self,
        dls: DataLoaders,
        prediction_type: Union[str, PredictionType],
        data_type: Union[str, DataType],
    ) -> None:
        """Inititalizes the VisionDataManager class.

        Args:
            dls: A FastAI `DataLoaders` instance.
            prediction_type: The type of prediction.
            data_type: The type of data.
        """
        super().__init__()
        self.data_type: str = data_type if isinstance(data_type, str) else data_type.value
        self.prediction_type: str = (
            prediction_type if isinstance(prediction_type, str) else prediction_type.value
        )
        self.dls = dls

    def get_training_data(self) -> Any:
        """Return the data for model training.

        Returns:
            The data for model training.
        """
        return self.dls

    def get_validation_data(self) -> Any:
        """Return the data for model validation.

        Returns:
            The data for model validation.
        """
        return [i[0] for i in self.dls.valid_ds], np.array([i[1] for i in self.dls.valid_ds])

    def get_prediction_type(self) -> str:
        """Return the prediction type.

        Returns:
            The prediction type.
        """
        if isinstance(self.prediction_type, str):
            return self.prediction_type
        else:
            return self.prediction_type.value

    def get_data_type(self) -> str:
        """Return the data type.

        Returns:
            The data type.
        """
        if isinstance(self.data_type, str):
            return self.data_type
        else:
            return self.data_type.value


class CrossValidationVisionDataManager(VisionDataManager):
    """Data manager for cross validation vision."""

    def __init__(
        self,
        dls: DataLoaders,
        cross_validation_splitter: FastaiCrossValidationSplitterAdapter,
        prediction_type: Union[str, PredictionType],
        data_type: Union[str, DataType],
    ) -> None:
        """Inititalizes the CrossValidationVisionDataManager class.

        Args:
            dls: A FastAI `DataLoaders` instance.
            prediction_type: The type of prediction.
            cross_validation_splitter: splitter for cross validation
            data_type: The type of data.
        """

        def modify_split(
            dls: DataLoaders, splitter: FastaiCrossValidationSplitterAdapter
        ) -> DataLoaders:
            # Apply the new splitter on the existing items
            splits = splitter(dls.items)
            # Recreate the Datasets object with the same attributes, using new splits
            dsets = Datasets(
                items=dls.items, tfms=dls.tfms, splits=splits, types=dls.types[0], n_inp=dls.n_inp
            )

            return DataLoaders(dls.train.new(dsets.train), dls.valid.new(dsets.valid))

        super().__init__(dls, prediction_type, data_type)
        self.selected_fold: Optional[int] = None
        self.splitter = cross_validation_splitter
        self.dls_folds_list = [
            modify_split(dls, self.splitter) for i in range(self.splitter.splitter.n_folds)
        ]

    def get_training_data(self, fold: Optional[int] = None, all_: bool = False) -> Any:
        """Return the data for model training.

        Returns:
            The data for model training.
        """
        if fold is None:
            fold = self.get_selected_fold()
        if all_:
            return self.dls_folds_list
        else:
            return self.dls_folds_list[fold]

    def get_validation_data(self, fold: Optional[int] = None, all_: bool = False) -> Any:
        """Return the data for model validation."""
        if fold is None:
            fold = self.get_selected_fold()

        if all_:
            return (
                [[i[0] for i in dls.valid_ds] for dls in self.dls_folds_list],
                [np.array([i[1] for i in dls.valid_ds]) for dls in self.dls_folds_list],
            )
        else:
            return (
                [i[0] for i in self.dls_folds_list[fold].valid_ds],
                np.array([i[1] for i in self.dls_folds_list[fold].valid_ds]),
            )

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
