"""Implementation of Full Life Cycle."""

from __future__ import annotations

import abc
import copy
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from simpml.core.base import DataManagerBase, DataType, MetricName, ModelManagerBase, PredictionType
from simpml.core.experiment_manager import ExperimentManager
from simpml.tabular.inference import TabularInferenceManager
from simpml.tabular.tabular_data_manager import TabularDataManager


class SafetyFilterBase:
    """This class is an interface for managing safety filters.

    The user must implement the abstract methods.
    """

    @classmethod
    def __subclasshook__(cls, subclass: Any) -> bool:
        """Return True if subclass should be considered a (direct or indirect) subclass of this
        class.

        Args:
            subclass: The class to check if it is a (direct or indirect) subclass of this class.

        Returns:
            True if subclass should be considered a (direct or indirect) subclass of this
            class. Otherwise, returns `NotImplemented` (the type checker counts this as a bool).
        """
        return (
            hasattr(subclass, "predict")
            and callable(subclass.predict)
            and hasattr(subclass, "fit")
            and callable(subclass.fit)
            or NotImplemented
        )

    @abc.abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predicts the safety filter outcome for given data.

        Args:
            data: The data on which safety filter is to be applied.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, data_manager: DataManagerBase, verbose: bool = True) -> None:
        """Fits the safety filter with a data manager.

        Args:
            data_manager: The data manager used to manage data.
            verbose: Verbosity for the fitting process.
        """
        raise NotImplementedError


class AnomalyGeneratorBase:
    """This class is an interface for managing anomaly generator.

    The user must implement the abstract methods.
    """

    @classmethod
    def __subclasshook__(cls, subclass: Any) -> bool:
        """Return True if subclass should be considered a (direct or indirect) subclass of this
        class.

        Args:
            subclass: The class to check if it is a (direct or indirect) subclass of this class.

        Returns:
            True if subclass should be considered a (direct or indirect) subclass of this
            class. Otherwise, returns `NotImplemented` (the type checker counts this as a bool).
        """
        return (
            hasattr(subclass, "generate_anomalies")
            and callable(subclass.generate_anomalies)
            or NotImplemented
        )

    @abc.abstractmethod
    def generate_anomalies(self, normal_data: Iterable) -> Iterable:
        """Generates anomalies from the provided normal data.

        Args:
            normal_data: The normal data from which anomalies are to be generated.
        """
        raise NotImplementedError


class TabularSafetyFilter(SafetyFilterBase):
    """This class is for applying a safety filter on tabular data.

    This class is intended to detect anomalies in the data
    that the model wasn't trained on.
    """

    def __init__(self) -> None:
        """Initializes the TabularSafetyFilter."""
        self.model: Optional[Any] = None

    def fit(
        self,
        data_manager: DataManagerBase,
        verbose: bool = True,
        opt_metric: MetricName = MetricName.F1,
        contamination: float = 0.0000005,
    ) -> None:
        """Fits the TabularSafetyFilter with a TabularDataManager.

        Args:
            data_manager (TabularDataManager): The data manager used to manage tabular data.
            verbose (bool): Verbosity for the ExperimentManager.
            opt_metric (MetricName) : metric name to optimize.
            contamination (float) : contamination factor for training model.
        """
        exp_mang = ExperimentManager(data_manager, optimize_metric=opt_metric, verbose=verbose)
        exp_mang.run_experiment(models_kwargs={"contamination": contamination})
        self.model = exp_mang.get_best_model()

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Makes predictions with the model pipeline on the given data.

        Args:
           data (Iterable): Data to make predictions on.

        Returns:
            Iterable: The model's predictions.
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet. Please call 'fit' before using 'predict'.")
        return self.model.predict(data)


class TabularUnivarietAnomalyGenerator(AnomalyGeneratorBase):
    """This class generates univariate anomalies in tabular data."""

    def __init__(
        self,
        anomalies_number: int = 100,
        std_multiplier_range: Tuple[Union[float, int], Union[float, int]] = (3, 10),
    ) -> None:
        """Initializes the anomaly generator.

        Args:
            anomalies_number (int): The number of anomalies to generate. Default is 100.
            std_multiplier_range (Tuple[int, int]): The range of standard deviations from
                                                    the mean for generating anomalies.
                                                    Default is (3, 10).
        """
        self.anomalies_number: int = anomalies_number
        self.std_multiplier_range: Tuple[
            Union[float, int], Union[float, int]
        ] = std_multiplier_range

    def generate_anomalies(self, normal_data: pd.DataFrame) -> pd.DataFrame:
        """Generates anomalies from the provided normal data.

        Args:
            normal_data (pd.DataFrame): The normal data from which anomalies are to be generated.

        Returns:
            pd.DataFrame: DataFrame containing the generated anomalies.
        """
        anomalies = []
        for _ in range(self.anomalies_number):
            sample_index = np.random.randint(normal_data.shape[0])
            sample = normal_data.iloc[sample_index].copy()
            feature_index = np.random.randint(normal_data.shape[1])
            feature = normal_data.columns[feature_index]
            mean = normal_data[feature].mean()
            std = normal_data[feature].std()
            std_multiplier = np.random.uniform(*self.std_multiplier_range)
            if np.random.choice([True, False]):
                sample[feature] = mean + std_multiplier * std
            else:
                sample[feature] = mean - std_multiplier * std
            anomalies.append(sample)
        anomalies = pd.concat(anomalies, axis=1).T
        return anomalies


class TabularMultivarietAnomalyGenerator(AnomalyGeneratorBase):
    """This class generates multivariate anomalies in tabular data."""

    def __init__(
        self,
        anomalies_number: int = 100,
        std_multiplier_range: Tuple[Union[float, int], Union[float, int]] = (3, 10),
        features_to_alter: int = 2,
    ) -> None:
        """Initializes the anomaly generator.

        Args:
            anomalies_number (int): The number of anomalies to generate. Default is 100.
            std_multiplier_range (Tuple[int, int]): The range of standard deviations from
                                                    the mean for generating anomalies.
                                                    Default is (3, 10).
            features_to_alter (int): The number of features to alter while generating anomalies.
                                     Default is 2.
        """
        self.anomalies_number: int = anomalies_number
        self.std_multiplier_range: Tuple[
            Union[float, int], Union[float, int]
        ] = std_multiplier_range
        self.features_to_alter: int = features_to_alter

    def generate_anomalies(self, normal_data: pd.DataFrame) -> pd.DataFrame:
        """Generates anomalies from the provided normal data.

        Args:
            normal_data (pd.DataFrame): The normal data from which anomalies are to be generated.

        Returns:
            pd.DataFrame: DataFrame containing the generated anomalies.
        """
        anomalies = []
        for _ in range(self.anomalies_number):
            sample_index = np.random.randint(normal_data.shape[0])
            sample = normal_data.iloc[sample_index].copy()
            feature_indices = np.random.choice(normal_data.columns, self.features_to_alter)
            for feature in feature_indices:
                mean = normal_data[feature].mean()
                std = normal_data[feature].std()
                std_multiplier = np.random.uniform(*self.std_multiplier_range)
                if np.random.choice([True, False]):
                    sample[feature] = mean + std_multiplier * std
                else:
                    sample[feature] = mean - std_multiplier * std
            anomalies.append(sample)
        anomalies = pd.concat(anomalies, axis=1).T
        return anomalies


class TabularAnomalyGenerator:
    """This class generates anomalies in tabular data using a list of anomaly generators."""

    def __init__(self, generators: Sequence[AnomalyGeneratorBase]) -> None:
        """Initializes the TabularAnomalyGenerator with a list of anomaly generators.

        Args:
            generators (List[AnomalyGeneratorBase]): The list of anomaly generators to be used
            for generating anomalies.
        """
        self.generators: Sequence[AnomalyGeneratorBase] = generators

    def create_anomalies(self, normal_data: pd.DataFrame) -> pd.DataFrame:
        """Creates anomalies from the provided normal data using the list of anomaly generators.

        Args:
            normal_data (pd.DataFrame): The normal data from which anomalies are to be generated.

        Returns:
            pd.DataFrame: DataFrame containing the generated anomalies.
        """
        all_anomalies = []
        for generator in self.generators:
            anomalies = generator.generate_anomalies(normal_data)
            all_anomalies.append(anomalies)

        all_anomalies = pd.concat(all_anomalies, ignore_index=True)

        return all_anomalies


class SafetyFilterTabularDataManager(DataManagerBase):
    """Implementation of the DataManagerBase abstract base class."""

    def __init__(
        self,
        anomaly_data: pd.DataFrame,
        normal_data: pd.DataFrame,
    ) -> None:
        """Initializes the SafetyFilterDataManager class.

        Args:
            anomaly_data: The anomaly data.
            normal_data: The normal data.
            description: The description of the data manager.
        """
        super().__init__()

        anomaly_data["target"] = 1
        normal_data["target"] = 0

        normal_train, normal_val = train_test_split(normal_data, test_size=len(anomaly_data))
        self.validation_data: pd.DataFrame = pd.concat([anomaly_data, normal_val])
        self.train_data: pd.DataFrame = normal_train.drop("target", axis=1)
        self.target_col: str = "target"

    def __repr__(self) -> str:
        """Represent object instance as string.

        Returns:
            String representation.
        """
        return str(self.id)

    def get_training_data(self) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Return the data for model training.

        Returns:
            A 2-tuple in which the first element is X and the second is y.
        """
        X = self.train_data
        y = None
        return X, y

    def get_validation_data(self) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Return the data for model validation.

        Returns:
            A 2-tuple in which the first element is X and the second is y.
        """
        X = self.validation_data.drop(self.target_col, axis=1)
        y = self.validation_data[self.target_col]
        return X, y

    def get_prediction_type(self) -> str:
        """Return the type of the prediction that you want to make.

        Returns:
            The type of the prediction.
        """
        return PredictionType.AnomalyDetection.value

    def get_data_type(self) -> str:
        """Return the type of the data.

        Returns:
            The type of the data.
        """
        return DataType.Tabular.value


class SafeTabularInferenceManager(TabularInferenceManager):
    """This class manages the inference process on tabular data ensuring safety filter."""

    def __init__(
        self,
        data_manager: TabularDataManager,
        model: ModelManagerBase,
        safety_filter: Optional[SafetyFilterBase] = None,
        verbose: bool = True,
        anomalies_generators: Optional[Sequence[AnomalyGeneratorBase]] = None,
        anomaly_label: str = "Unknown",
    ) -> None:
        """Initializes the SafeTabularInferenceManager with a data manager, model, safety filter,
                                                            verbise, and anomaly generators.

        Args:
            data_manager (TabularDataManager): The data manager used to manage tabular data.
            model (ModelManagerBase): The model used for inference.
            safety_filter (SafetyFilterBase): The safety filter used for anomaly detection.
                                              Default is TabularSafetyFilter.
            verbose (bool): Verbosity for the safety filter fitting process. Default is True.
            anomaly_label (str) : Anomaly label for noamlies detected by safety filter.
            anomalies_generators (List[AnomalyGeneratorBase]):
              The list of anomaly generators used for generating anomalies.
              Default is [TabularUnivarietAnomalyGenerator(), TabularMultivarietAnomalyGenerator()].
        """
        super().__init__(data_manager, model)
        if anomalies_generators is None:
            anomalies_generators = [
                TabularUnivarietAnomalyGenerator(),
                TabularMultivarietAnomalyGenerator(),
            ]
        anomaly_generator: TabularAnomalyGenerator = TabularAnomalyGenerator(anomalies_generators)
        anomalies: pd.DataFrame = anomaly_generator.create_anomalies(
            data_manager.get_training_data()[0]
        )
        safety_filter_data_manager: SafetyFilterTabularDataManager = SafetyFilterTabularDataManager(
            anomalies, data_manager.get_training_data()[0]
        )
        if safety_filter is None:
            safety_filter = TabularSafetyFilter()
        self.safety_filter: SafetyFilterBase = safety_filter
        self.safety_filter.fit(safety_filter_data_manager, verbose)
        self.anomaly_label: str = anomaly_label

    def predict(self, data: Iterable, with_input: bool = False) -> Union[np.ndarray, pd.DataFrame]:
        """Predicts the outcome on given data after ensuring safety through anomaly detection.

        Args:
            data (pd.DataFrame): The data on which predictions are to be made.

        Returns:
            np.ndarray: The predictions made by the model.
                       For the instances that are not detected as anomalies by the safety filter,
                       it returns the original model's prediction.
                       Otherwise, it returns anoamly label.
        """
        preprocessing_pipeline = copy.deepcopy(self.model_pipeline)
        model = preprocessing_pipeline.steps[-1][1]
        preprocessing_pipeline.steps = preprocessing_pipeline.steps[:-1]
        preprocessed_data = preprocessing_pipeline.transform(data)

        anomaly_prediction = self.safety_filter.predict(preprocessed_data)
        original_prediction = model.predict(preprocessed_data)
        combined_prediction = [
            self.anomaly_label if anomaly == 1 else original
            for anomaly, original in zip(anomaly_prediction, original_prediction)
        ]

        if self.target_pipeline is not None:
            y = self.target_pipeline.inverse_transform(data, combined_prediction)
        if with_input:
            data_df = pd.DataFrame(data)
            if isinstance(y, pd.Series):
                y.index = data_df.index
            return pd.concat([data_df, pd.Series(y, index=data_df.index, name="Pred")], axis=1)
        else:
            return y
