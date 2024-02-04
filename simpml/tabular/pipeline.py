"""Tabular pipeline classes."""

from __future__ import annotations

import functools
import inspect
from functools import partial
from itertools import islice
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union

import pandas as pd
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import AddMissingIndicator, CategoricalImputer, MeanMedianImputer
from feature_engine.preprocessing import MatchVariables
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.utils import compute_class_weight

from simpml.core.base import DataManagerBase, PredictionType, VariableType
from simpml.tabular.adapters_pool import ManipulateAdapter
from simpml.tabular.steps_pool import (
    DictLabelEncoder,
    HighCardinalityDropper,
    Infinity2Nan,
    MinMaxScalerWithColumnNames,
    NanColumnDropper,
    RemoveSpecialJSONCharacters,
    SafeCategoricalTransformer,
    SafeDropFeatures,
    UniqueIDLabelEncoder,
    WaveformsFeatureExtractor,
)


class TrainPipeline:
    """Training data data-processing pipeline class."""

    def __init__(self, steps: Sequence[Tuple[str, Any]]) -> None:
        """Initializes a TrainPilepine class.

        Args:
            steps: A list of steps. Each step is a 2-tuple with the step name and step object.
        """
        self.steps: List[Tuple[str, Any]] = list(steps)

    def add_step(self, step: Tuple[str, Any]) -> None:
        """Add a step.

        Args:
            step: The step to add. It is a 2-tuple with the step name and step object.
        """
        self.steps.append(step)

    def _validate_names(self, names: Sequence[str]) -> None:
        """Validate the step names.

        Args:
            names: A list or tuple of step names.

        Raises:
            ValueError: If the step name is invalid.
        """
        if len(set(names)) != len(names):
            raise ValueError("Names provided are not unique: {0!r}".format(list(names)))
        invalid_names = [name for name in names if "__" in name]
        if invalid_names:
            raise ValueError("Estimator names must not contain __: got {0!r}".format(invalid_names))

    def _validate_steps(self) -> None:
        """Validate the steps.

        Raises:
            TypeError: If the step type is invalid.
        """
        names, manipulates = zip(*self.steps)

        # validate names
        self._validate_names(names)

        # validate manipulates
        for t in manipulates:
            if not (hasattr(t, "manipulate")):
                raise TypeError(
                    "All intermediate steps should be manipulate, '%s' (type %s) doesn't"
                    % (t, type(t))
                )

    def _iter(self, filter_passthrough: bool = True) -> Generator[Tuple[int, Any, Any], Any, None]:
        """Generate (idx, (name, trans)) tuples from `self.steps`.

        Args:
            filter_passthrough: When filter_passthrough is True, 'passthrough' and None
                transformers are filtered out.

        Returns:
            The generator.
        """
        stop = len(self.steps)

        for idx, (name, trans) in enumerate(islice(self.steps, 0, stop)):
            if not filter_passthrough:
                yield idx, name, trans
            elif trans is not None and trans != "passthrough":
                yield idx, name, trans

    def __len__(self) -> int:
        """Returns the length of the pipeline.

        Returns:
            The length of the pipeline.
        """
        return len(self.steps)

    def __getitem__(self, ind: Union[int, slice]) -> Any:
        """Returns a sub-pipeline or a single estimator in the pipeline.

        Args:
            ind: Indexing with an integer will return an estimator; using a slice
                returns another Pipeline instance which copies a slice of this
                Pipeline. This copy is shallow: modifying (or fitting) estimators in
                the sub-pipeline will affect the larger pipeline and vice-versa.
                However, replacing a value in `step` will not affect a copy.

        Returns:
            A sub-pipeline or a single estimator in the pipeline.
        """
        if isinstance(ind, slice):
            if ind.step not in (1, None):
                raise ValueError("Pipeline slicing only supports a step of 1")
            return self.__class__(self.steps[ind])
        _, est = self.steps[ind]
        return est

    # Estimator interface
    def manipulate(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Run the "manipulate" method of each step.

        Args:
            X: The feature data.
            y: The target data.

        Returns:
            A 2-tuple with the manipulated feature data (X) and target data (y).
        """
        _, manipulates = zip(*self.steps)
        self._validate_steps()
        for manipulates_item in list(manipulates):
            X, y = manipulates_item.manipulate(X, y)
        return X, y

    def __hash__(self) -> int:
        """Return the hash value of the object instance.

        Returns:
            The hash value of the object instance.
        """
        return hash(tuple((name, hash(trans)) for name, trans in self.steps))


class TargetPipeline:
    """Independent (target) variable data-processing pipeline class."""

    def __init__(self, steps: Sequence[Tuple[str, Any]]) -> None:
        """Initializes a TargetPipeline class.

        Args:
            steps: A list of steps. Each step is a 2-tuple with the step name and step object.
        """
        self.steps: List[Tuple[str, Any]] = list(steps)

    def add_step(self, step: Tuple[str, Any]) -> None:
        """Add a step.

        Args:
            step: The step to add. It is a 2-tuple with the step name and step object.
        """
        self.steps.append(step)

    def _validate_names(self, names: Sequence[str]) -> None:
        """Validate the step names.

        Args:
            names: A list or tuple of step names.

        Raises:
            ValueError: If the step name is invalid.
        """
        if len(set(names)) != len(names):
            raise ValueError("Names provided are not unique: {0!r}".format(list(names)))
        invalid_names = [name for name in names if "__" in name]
        if invalid_names:
            raise ValueError("Estimator names must not contain __: got {0!r}".format(invalid_names))

    def _validate_steps(self) -> None:
        """Validate the steps.

        Raises:
            TypeError: If the step type is invalid.
        """
        names, manipulates = zip(*self.steps)

        # validate names
        self._validate_names(names)

        # validate manipulates
        for t in manipulates:
            if (
                (not hasattr(t, "fit"))
                or (not hasattr(t, "transform"))
                or (not hasattr(t, "fit_transform"))
            ):
                raise TypeError(
                    "All intermediate steps should contain fit, transform and fit_transform, "
                    "'{t}' (type {type(t)}) doesn't"
                )

    def __hash__(self) -> int:
        """Return the hash value of the object instance.

        Returns:
            The hash value of the object instance.
        """
        return hash(tuple((name, hash(trans)) for name, trans in self.steps))

    def _iter(self, filter_passthrough: bool = True) -> Generator[Tuple[int, Any, Any], Any, None]:
        """Generate (idx, (name, trans)) tuples from `self.steps`.

        Args:
            filter_passthrough: When filter_passthrough is True, 'passthrough' and None
                transformers are filtered out.

        Returns:
            The generator.
        """
        stop = len(self.steps)

        for idx, (name, trans) in enumerate(islice(self.steps, 0, stop)):
            if not filter_passthrough:
                yield idx, name, trans
            elif trans is not None and trans != "passthrough":
                yield idx, name, trans

    def __len__(self) -> int:
        """Returns the length of the pipeline.

        Returns: The length of the pipeline.
        """
        return len(self.steps)

    def __getitem__(self, ind: Union[int, slice]) -> Any:
        """Returns a sub-pipeline or a single estimator in the pipeline.

        Args:
            ind: Indexing with an integer will return an estimator; using a slice
                returns another Pipeline instance which copies a slice of this
                Pipeline. This copy is shallow: modifying (or fitting) estimators in
                the sub-pipeline will affect the larger pipeline and vice-versa.
                However, replacing a value in `step` will not affect a copy.

        Returns:
            A sub-pipeline or a single estimator in the pipeline.
        """
        if isinstance(ind, slice):
            if ind.step not in (1, None):
                raise ValueError("Pipeline slicing only supports a step of 1")
            return self.__class__(self.steps[ind])
        _, est = self.steps[ind]
        return est

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TargetPipeline:
        """Fit the target pipeline steps.

        Args:
            y: The training target data.

        Returns:
            This class instance.
        """
        _, transforms = zip(*self.steps)
        self._validate_steps()
        for transform in list(transforms):
            if hasattr(transform, "fit"):
                transform.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Perform the transformations.

        Args:
            y: The target data.

        Returns:
            The transformed data.
        """
        _, transforms = zip(*self.steps)
        self._validate_steps()
        for transform in list(transforms):
            if hasattr(transform, "transform"):
                y = transform.transform(X, y)
        return y

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Perform both the fits and the transformations.

        Args:
            y: The training target data.

        Returns:
            The transformed data.
        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Apply `inverse_transform` for each step in a reverse order.

        All estimators in the pipeline must support `inverse_transform`.

        Args:
            y: The training target data.

        Returns:
            The transformed data.
        """
        reverse_iter = reversed(list(self._iter()))
        for _, _, transform in reverse_iter:
            y = transform.inverse_transform(X, y)
        return y


class Pipeline:
    """Top-level data-processing pipeline class."""

    def __init__(
        self,
        sklearn_pipeline: SklearnPipeline,
        train_pipeline: Optional[TrainPipeline] = None,
        target_pipeline: Optional[TargetPipeline] = None,
    ) -> None:
        """Initializes the Pipeline class.

        Args:
            sklearn_pipeline: The scikit-learn pipeline.
            train_pipeline: The optional `TrainPipeline` object.
            target_pipeline: The optional `TargetPipeline` object.
        """
        self.sklearn_pipeline: SklearnPipeline = sklearn_pipeline
        self.train_pipeline: Optional[TrainPipeline] = train_pipeline
        self.target_pipeline: Optional[TargetPipeline] = target_pipeline

    def add_sklearn_step(self, step: Any) -> None:
        """Add a scikit-learn step.

        Args:
            step: The step to add. It is a 2-tuple with the step name and step object.
        """
        self.sklearn_pipeline.steps.append(step)

    def add_train_step(self, step: Any) -> None:
        """Add a `TrainPipeline` step.

        Args:
            step: The step to add. It is a 2-tuple with the step name and step object.
        """
        if self.train_pipeline:
            self.train_pipeline.add_step(step)
        else:
            self.train_pipeline = TrainPipeline(steps=[step])

    def add_target_step(self, step: Any) -> None:
        """Add a `TargetPipeline` step.

        Args:
            step: The step to add. It is a 2-tuple with the step name and step object.
        """
        if self.target_pipeline:
            self.target_pipeline.add_step(step)
        else:
            self.target_pipeline = TargetPipeline(steps=[step])

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Pipeline:
        """Fit the pipeline steps.

        Args:
            X: The training feature data.
            y: The training target data.

        Returns:
            This class instance.
        """
        self.sklearn_pipeline.fit(X, y)
        return self

    def fit_transform_manipulate(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Perform the fits, transformations, and manipulations.

        Args:
            X: The feature data.
            y: The target data.

        Returns:
            The transformed data.

        target_pipeline run first on the original data
        """
        if self.target_pipeline and y is not None:
            y = self.target_pipeline.fit_transform(X, y)
        X = self.sklearn_pipeline.fit_transform(X)
        if self.train_pipeline:
            X, y = self.train_pipeline.manipulate(X, y)
        return X, y

    def transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Perform the transformations.

        Args:
            X: The feature data.
            y: The target data.

        Returns:
            The transformed data.

        target_pipeline run first on the original data
        """
        if y is not None:
            if self.target_pipeline:
                y = self.target_pipeline.transform(X, y)
        X = self.sklearn_pipeline.transform(X)
        return X, y

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Perform both the fits and the transformations.

        Args:
            X: The training feature data.
            y: The training target data. Unused.

        Returns:
            The transformed data.
        """
        return self.sklearn_pipeline.fit_transform(X, y)

    def __str__(self) -> str:
        """Describe object instance as string.

        Returns:
            String description.
        """
        output = "Sklearn Pipeline:\n"
        for i, step in enumerate(self.sklearn_pipeline.steps):
            output += f"{step[0]} ({step[1]})"
            if i < len(self.sklearn_pipeline.steps) - 1:
                output += " ->\n"
        output += "\n"

        if self.train_pipeline:
            output += "Train Pipeline:\n"
            for i, step in enumerate(self.train_pipeline.steps):
                output += f"{step[0]} ({step[1]})"
                if i < len(self.train_pipeline.steps) - 1:
                    output += " ->\n"

        if self.target_pipeline:
            output += "\n"
            output += "Target Pipeline:\n"
            for i, step in enumerate(self.target_pipeline.steps):
                output += f"{step[0]} ({step[1]})"
                if i < len(self.target_pipeline.steps) - 1:
                    output += " ->\n"

        return output

    def __repr__(self) -> str:
        """Represent object instance as string.

        Returns:
            String representation.
        """
        return self.__str__()

    def __hash__(self) -> int:
        """Return the hash value of the object instance.

        Returns:
            The hash value of the object instance.
        """
        return hash(
            (
                hash(tuple((name, hash(trans)) for name, trans in self.sklearn_pipeline.steps)),
                self.train_pipeline,
                self.target_pipeline,
            )
        )


class PipelineBuilder:
    """Class to construct a machine learning pipeline based on given parameters."""

    def __init__(self) -> None:
        """Initialize pipeline parameters."""
        self.steps_pool: Dict[str, List[str]] = {
            VariableType.Generic.value: [
                "add_missing_indicator",
                "match_variables",
                "remove_special_json_characters",
                "nan_column_dropper",
            ],
            VariableType.Numerical.value: [
                "mean_median_imputer",
                "min_max_scaler",
                "infinity_to_nan",
            ],
            VariableType.Categorical.value: [
                "one_hot_encoder",
                "categorical_imputer",
                "high_cardinality_dropper",
            ],
            VariableType.Imbalanced.value: ["smote"],
            VariableType.DateTime.value: ["waveforms_feature_extractor", "id_label_encoder"],
            VariableType.IsTarget.value: ["ordinal_encode_target"],
        }
        self.pipeline_dict_params: Dict[str, List[str]] = {
            PredictionType.BinaryClassification.value: [
                "remove_special_json_characters",
                "add_missing_indicator",
                "mean_median_imputer",
                "categorical_imputer",
                "one_hot_encoder",
                "smote",
                "infinity_to_nan",
                "nan_column_dropper",
                "ordinal_encode_target",
                "high_cardinality_dropper",
                "match_variables",
                "min_max_scaler",
            ],
            PredictionType.MulticlassClassification.value: [
                "add_missing_indicator",
                "mean_median_imputer",
                "min_max_scaler",
                "infinity_to_nan",
                "nan_column_dropper",
                "categorical_imputer",
                "one_hot_encoder",
                "ordinal_encode_target",
                "high_cardinality_dropper",
                "match_variables",
            ],
            PredictionType.Regression.value: [
                "add_missing_indicator",
                "mean_median_imputer",
                "min_max_scaler",
                "infinity_to_nan",
                "nan_column_dropper",
                "categorical_imputer",
                "one_hot_encoder",
                "high_cardinality_dropper",
                "match_variables",
            ],
            PredictionType.AnomalyDetection.value: [
                "add_missing_indicator",
                "mean_median_imputer",
                "min_max_scaler",
                "infinity_to_nan",
                "nan_column_dropper",
                "categorical_imputer",
                "one_hot_encoder",
                "high_cardinality_dropper",
                "match_variables",
            ],
            PredictionType.Clustering.value: [
                "add_missing_indicator",
                "mean_median_imputer",
                "min_max_scaler",
                "infinity_to_nan",
                "nan_column_dropper",
                "categorical_imputer",
                "one_hot_encoder",
                "high_cardinality_dropper",
                "match_variables",
            ],
        }

    def get_pipeline(
        self,
        target: str,
        add_missing_indicator: bool,
        infinity_to_nan: bool,
        mean_median_imputer: bool,
        categorical_imputer: bool,
        nan_column_dropper: bool,
        one_hot_encoder: bool,
        smote: bool,
        ordinal_encode_target: bool,
        high_cardinality_dropper: bool,
        min_max_scaler: bool,
        waveforms_feature_extractor: bool,
        match_variables: bool,
        remove_special_json_characters: bool,
        id_label_encoder: bool,
        drop_cols: Optional[Sequence[str]] = None,
        inernal_drop_cols: Optional[Sequence[str]] = None,
        step_params: Optional[Dict[str, Any]] = None,
    ) -> Pipeline:
        """Return a constructed pipeline based on the given parameters.

        Args:
            target: The independent (target) variable name
            add_missing_indicator: Whether to add missing indicators.
            infinity_to_nan: Whether to convert infinity values to NaN.
            mean_median_imputer: Whether to impute missing numeric values as mean or median of
                existing values.
            categorical_imputer: Whether to impute missing categorical values.
            nan_column_dropper: Whether to drop columns that contain NaN values above a certain
                threshold.
            one_hot_encoder: Whether to encode categorical columns as one-hot columns.
            smote: Wheter to balance the data with the SMOTE algorithm.
            ordinal_encode_target: Whether to encode an ordinal target.
            high_cardinality_dropper: Whether to drop high cardinality features.
            min_max_scaler: Whether to scale the data using the min/max scaler.
            waveforms_feature_extractor: Whether to extract waveforms features.
            match_variables: Whether to ensure that the validation/test set variables match the
                training set variables.
            remove_special_json_characters: Whether to remove special characters from column names.
            column_id: The "column_id" argument for the `WaveformsFeatureExtractor` step.
            drop_cols: A list or tuple of which columns to drop from the user.
            inernal_drop_cols: A list or tuple of which columns to drop from the class.
            encoding_dict: A dictionary mapping labels to integers.
                If this is not provided, the class will generate a mapping
                based on the unique labels it encounters during fit.

        Returns:
            A constructed pipeline based on the given parameters.
        """
        # Initialize pipeline steps
        sklearn_pipeline_steps: List[Tuple[str, Any]] = []
        train_pipeline_steps: List[Tuple[str, Any]] = []
        target_pipeline_steps: List[Tuple[str, Any]] = []

        def get_step_params(
            step_name: str, step_params: Optional[Dict[str, Any]], step_function: Any
        ) -> Dict[str, Any]:
            """Retrieves and filters parameters for a specific step in a pipeline.
            Treats parameters with the same name as the step function as local parameters.

            Args:
                step_name (str): Name of the step.
                step_params (Optional[Dict[str, Any]]): Global and specific parameters.
                step_function (Any): The function associated with the step.

            Returns:
                Dict[str, Any]: Filtered parameters valid for the step function.
            """
            if step_params is None:
                return {}

            # Treat a parameter as specific if its name matches step_name, even if it's a dictionary
            specific_params = step_params.get(step_name, {})
            # All other parameters are considered global
            global_params = {k: v for k, v in step_params.items() if k != step_name}

            combined_params = {**global_params, **specific_params}

            if not callable(step_function):
                raise ValueError("step_function must be callable")

            valid_params = {
                k: combined_params[k]
                for k in combined_params
                if k in inspect.signature(step_function).parameters
            }
            return valid_params

        if drop_cols is None:
            drop_cols = []
        if inernal_drop_cols is None:
            inernal_drop_cols = []
        if match_variables:
            sklearn_pipeline_steps.append(
                (
                    "MatchVariablesBefore",
                    MatchVariables(
                        missing_values="ignore",
                        **get_step_params("MatchVariablesBefore", step_params, MatchVariables),
                    ),
                )
            )
        if drop_cols:
            sklearn_pipeline_steps.append(
                (
                    "SafeDropFeaturesBefore",
                    SafeDropFeatures(features_to_drop=drop_cols),
                )
            )
        if waveforms_feature_extractor:
            sklearn_pipeline_steps.append(
                (
                    "WaveformsFeatureExtractor",
                    WaveformsFeatureExtractor(
                        **get_step_params(
                            "WaveformsFeatureExtractor", step_params, WaveformsFeatureExtractor
                        )
                    ),
                )
            )
        if inernal_drop_cols:
            sklearn_pipeline_steps.append(
                (
                    "SafeInernalDropFeaturesBefore",
                    SafeDropFeatures(features_to_drop=inernal_drop_cols),
                )
            )
        if nan_column_dropper:
            sklearn_pipeline_steps.append(
                (
                    "NanColumnDropper",
                    NanColumnDropper(
                        **get_step_params("NanColumnDropper", step_params, NanColumnDropper)
                    ),
                )
            )
        if infinity_to_nan:
            sklearn_pipeline_steps.append(
                (
                    "Infinity2Nan",
                    Infinity2Nan(**get_step_params("Infinity2Nan", step_params, Infinity2Nan)),
                )
            )
        if min_max_scaler:
            sklearn_pipeline_steps.append(
                (
                    "MinMaxScaler",
                    MinMaxScalerWithColumnNames(
                        **get_step_params("MinMaxScaler", step_params, MinMaxScalerWithColumnNames)
                    ),
                )
            )
        if high_cardinality_dropper:
            sklearn_pipeline_steps.append(
                (
                    "HighCardinalityDropper",
                    HighCardinalityDropper(
                        **get_step_params(
                            "HighCardinalityDropper", step_params, HighCardinalityDropper
                        )
                    ),
                )
            )
        if add_missing_indicator:
            sklearn_pipeline_steps.append(
                (
                    "AddMissingIndicator",
                    AddMissingIndicator(
                        **get_step_params("AddMissingIndicator", step_params, AddMissingIndicator)
                    ),
                )
            )
        if mean_median_imputer:
            sklearn_pipeline_steps.append(
                (
                    "NumericalImputer",
                    MeanMedianImputer(
                        **get_step_params("NumericalImputer", step_params, MeanMedianImputer)
                    ),
                )
            )
        if categorical_imputer:
            sklearn_pipeline_steps.append(
                (
                    "SafeCategoricalImputer",
                    SafeCategoricalTransformer(
                        transformer_cls=CategoricalImputer,
                        **get_step_params(
                            "SafeCategoricalImputer", step_params, CategoricalImputer
                        ),
                    ),
                )
            )
        if one_hot_encoder:
            sklearn_pipeline_steps.append(
                (
                    "SafeOneHotEncoder",
                    SafeCategoricalTransformer(
                        transformer_cls=OneHotEncoder,
                        **get_step_params("SafeOneHotEncoder", step_params, OneHotEncoder),
                    ),
                )
            )
        if smote:
            train_pipeline_steps.append(
                (
                    "SMOTE",
                    ManipulateAdapter(
                        SMOTE(),
                        "fit_resample",
                        **get_step_params("SMOTE", step_params, ManipulateAdapter),
                    ),
                )
            )
        if id_label_encoder:
            target_pipeline_steps.append(
                (
                    "UniqueIDLabelEncoder",
                    UniqueIDLabelEncoder(
                        **get_step_params("UniqueIDLabelEncoder", step_params, UniqueIDLabelEncoder)
                    ),
                )
            )
        if ordinal_encode_target:
            target_pipeline_steps.append(
                (
                    "LabelEncoder",
                    DictLabelEncoder(
                        **get_step_params("LabelEncoder", step_params, DictLabelEncoder)
                    ),
                )
            )
        if remove_special_json_characters:
            sklearn_pipeline_steps.append(
                (
                    "RemoveSpecialJSONCharacters",
                    RemoveSpecialJSONCharacters(
                        **get_step_params(
                            "RemoveSpecialJSONCharacters", step_params, RemoveSpecialJSONCharacters
                        )
                    ),
                )
            )
        if drop_cols or inernal_drop_cols:
            sklearn_pipeline_steps.append(
                (
                    "SafeDropFeaturesAfter",
                    SafeDropFeatures(
                        features_to_drop=list(set(drop_cols) | set(inernal_drop_cols))
                    ),
                )
            )
        if match_variables:
            sklearn_pipeline_steps.append(
                (
                    "MatchVariablesAfter",
                    MatchVariables(
                        missing_values="ignore",
                        **get_step_params("MatchVariablesAfter", step_params, MatchVariables),
                    ),
                )
            )
        return Pipeline(
            sklearn_pipeline=SklearnPipeline(sklearn_pipeline_steps),
            train_pipeline=TrainPipeline(train_pipeline_steps),
            target_pipeline=TargetPipeline((target_pipeline_steps)),
        )

    def get_pipeline_func_based_on_prediction_type(
        self,
        prediction_type: Union[str, PredictionType],
        target: Optional[str] = None,
        data_schema: Optional[Dict[str, Any]] = None,
        cols_to_drop: Optional[List] = None,
    ) -> Callable[[], Pipeline]:
        """Return a function that will contruct a `Pipeline` instance based on the given
        prediction type.

        Args:
            prediction_type: The prediction type.
            target: The name of the independent (target) variable.
            data_schema: A dictionary with key being the step name (same name as used in the step
                pool) and value being another dictionary of keyword arguments to use when
                constructing that step.
            cols_to_drop: columns to drop.

        Returns:
            A function that will contruct a `Pipeline` instance.
        """
        if isinstance(prediction_type, PredictionType):
            prediction_type_str = prediction_type.name
        else:
            prediction_type_str = prediction_type

        if prediction_type_str not in self.pipeline_dict_params:
            raise ValueError(f"Prediction type {prediction_type_str} not supported.")
        if data_schema is None:
            raise ValueError("'data_schema' must be set.")
        params_dict: Dict[str, Any] = {}
        for key, val in data_schema.items():
            for param in self.steps_pool[key]:
                params_dict[param] = param in self.pipeline_dict_params[prediction_type_str] and val
        if cols_to_drop:
            params_dict["inernal_drop_cols"] = cols_to_drop
        return partial(self.get_pipeline, target=target, **params_dict)


def pipeline_decorator(manager_instance: DataManagerBase) -> Callable[[Callable], Callable]:
    """Constructs a decorator function that will be used in a data manager's "build_pipeline"
    member.

    Args:
        manager_instance: The data manager object.

    Returns:
        The decorator function.
    """

    def actual_decorator(func: Callable) -> Callable:
        """The actual decorator function.

        Args:
            func: The function to wrap.

        Returns:
            The decorator.
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """The wrapper for the decorator.

            Args:
                *args: The arguments of the wrapped function.
                **kwargs: The keyword arguments of the wrapped function.

            Returns:
                Whatever the wrapped function returns.
            """
            result = func(*args, **kwargs)
            manager_instance.pipeline = result  # type: ignore [attr-defined]
            return result

        return wrapper

    return actual_decorator


def get_data_types(X: pd.DataFrame, target: Optional[str] = None) -> Dict[str, bool]:
    """Get the data types dictionary.

    This is a dictionary with key names from the data types (from the `VariableType` Enum) and
    boolean values as to whether they are used.

    Args:
        X: The feature data.
        target: The name of the independent (target) variable.

    Returns:
        Dictionary with key names from the data types (from the `VariableType` Enum) and boolean
        values as to whether they are used.
    """
    if target:
        y: Optional[pd.Series] = X[target]
        X = X.drop(target, axis=1)
    else:
        y = None
    types = {
        VariableType.Numerical.value: len(
            list(X.select_dtypes(include=["int64", "float64"]).columns)
        )
        > 0,
        VariableType.Categorical.value: len(
            list(X.select_dtypes(include=["object", "bool", "category"]).columns)
        )
        > 0,
        VariableType.DateTime.value: len(
            list(X.select_dtypes(include=["datetime64", "datetime", "timedelta"]).columns)
        )
        > 0,
    }
    types[VariableType.Generic.value] = True
    # Check if target is provided and exists in the dataframe
    if y is not None:
        if target is not None and y.name == target:
            class_weights = compute_class_weight("balanced", classes=y.unique(), y=y)
            max_weight = max(class_weights)
            min_weight = min(class_weights)

            # If the weight difference is significant, data is imbalanced
            types[VariableType.Imbalanced.value] = True if max_weight / min_weight > 2 else False
        else:
            types[VariableType.Imbalanced.value] = False
        types[VariableType.IsTarget.value] = True
    else:
        types[VariableType.IsTarget.value] = False
        types[VariableType.Imbalanced.value] = False
    return types
