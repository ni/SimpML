"""Base definitions."""

from __future__ import annotations

import abc
import inspect
import uuid
from enum import Enum
from os import PathLike
from typing import Any, Callable, Dict, Iterable, Optional, overload, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
from typing_extensions import Self


class EnumFromName(Enum):
    """Enum helper that will add a 'from_name' class method to Enums that inherit from this
    class.
    """

    @overload
    @classmethod
    def from_name(cls, name: str) -> Self:
        ...

    @overload
    @classmethod
    def from_name(cls, name: str, default: Type[Exception]) -> Self:
        ...

    @overload
    @classmethod
    def from_name(cls, name: str, default: Self) -> Self:
        ...

    @overload
    @classmethod
    def from_name(cls, name: str, default: None) -> Optional[Self]:
        ...

    @classmethod
    def from_name(
        cls, name: str, default: Optional[Union[Type[Exception], Self]] = ValueError
    ) -> Optional[Self]:
        """Get the enumeration instance based on the name of an enumeration key.

        Args:
            name: The name of the enumeration key.
            default: The value to use if not found. If this is an exception type, it will raise
                an exception in this case.

        Returns:
            The corresponding enumeration instance. When not found, if `default` is an exception
            type, will raise that exception. Otherwise will return `default`.
        """
        for key, val in cls.__members__.items():
            if key == name:
                return val
        if isinstance(default, type) and issubclass(default, Exception):
            raise default(f"'{cls.__name__}' enum not found for '{name}'")
        return default


class EnumStringValues(EnumFromName):
    """Enum helper that will add a 'from_value' class method to Enums that inherit from this
    class.

    Also inherits from `EnumFromName`.

    This is for Enums with string values.
    """

    @overload
    @classmethod
    def from_value(cls, value: str) -> Self:
        ...

    @overload
    @classmethod
    def from_value(cls, value: str, default: Type[Exception]) -> Self:
        ...

    @overload
    @classmethod
    def from_value(cls, value: str, default: Self) -> Self:
        ...

    @overload
    @classmethod
    def from_value(cls, value: str, default: None) -> Optional[Self]:
        ...

    @classmethod
    def from_value(
        cls, value: str, default: Optional[Union[Type[Exception], Self]] = ValueError
    ) -> Optional[Self]:
        """Get the enumeration instance based on a string value of the enumeration.

        Args:
            value: A string value of the enumeration.
            default: The value to use if not found. If this is an exception type, it will raise
                an exception in this case.

        Returns:
            The corresponding enumeration instance. When not found, if `default` is an exception
            type, will raise that exception. Otherwise will return `default`.
        """
        for val in cls.__members__.values():
            if val.value == value:
                return val
        if isinstance(default, type) and issubclass(default, Exception):
            raise default(f"'{cls.__name__}' enum not found for '{value}'")
        return default


class EnumIntValues(EnumFromName):
    """Enum helper that will add a 'from_value' class method to Enums that inherit from this
    class.

    Also inherits from `EnumFromName`.

    This is for Enums with int values.
    """

    @overload
    @classmethod
    def from_value(cls, value: int) -> Self:
        ...

    @overload
    @classmethod
    def from_value(cls, value: int, default: Type[Exception]) -> Self:
        ...

    @overload
    @classmethod
    def from_value(cls, value: int, default: Self) -> Self:
        ...

    @overload
    @classmethod
    def from_value(cls, value: int, default: None) -> Optional[Self]:
        ...

    @classmethod
    def from_value(
        cls, value: int, default: Optional[Union[Type[Exception], Self]] = ValueError
    ) -> Optional[Self]:
        """Get the enumeration instance based on an integer value of the enumeration.

        Args:
            value: An integer value of the enumeration.
            default: The value to use if not found. If this is an exception type, it will raise
                an exception in this case.

        Returns:
            The corresponding enumeration instance. When not found, if `default` is an exception
            type, will raise that exception. Otherwise will return `default`.
        """
        for val in cls.__members__.values():
            if val.value == value:
                return val
        if isinstance(default, type) and issubclass(default, Exception):
            raise default(f"'{cls.__name__}' enum not found for '{value}'")
        return default


class Dataset(EnumStringValues):
    """Enum for categories of data sets."""

    Train = "Train"
    Valid = "Valid"
    Test = "Test"
    Inference = "Inference"


class PredictionType(EnumStringValues):
    """Enum for types of predictions."""

    Regression = "Regression"
    BinaryClassification = "BinaryClassification"
    MulticlassClassification = "MulticlassClassification"
    Clustering = "Clustering"
    AnomalyDetection = "AnomalyDetection"


class FeatureImportanceMethod(EnumStringValues):
    """Enum for the method to ascertain feature importance."""

    Shap = "Shap"
    Permutation = "Permutation"


class HyperParamsOptimizationLevel(EnumStringValues):
    """Enum for hyperparameter optimization level."""

    Default = "Default"
    Fast = "Fast"
    Slow = "Slow"


class DataType(EnumStringValues):
    """Enum for type of data."""

    Tabular = "Tabular"
    Vision = "Vision"


class MinOrMax(EnumStringValues):
    """Enum for optimization direction (min or max)."""

    Min = "Min"
    Max = "Max"


class MetricName(EnumStringValues):
    """Enum for metric names."""

    MSE = "MSE"
    RMSE = "RMSE"
    MAPE = "MAPE"
    R2 = "R2"
    Accuracy = "Accuracy"
    AUC = "AUC"
    Recall = "Recall"
    Precision = "Precision"
    BalancedAccuracy = "Balanced Accuracy"
    Kappa = "Kappa"
    F1 = "F1"
    Silhouette = "Silhouette Score"
    DaviesBouldin = "Davies-Bouldin Score"
    CalinskiHarabasz = "Calinski-Harabasz Score"


class VariableType(EnumStringValues):
    """Enum for variable types."""

    Generic = "Generic"
    Categorical = "Categorical"
    Numerical = "Numerical"
    DateTime = "DateTime"
    Imbalanced = "Imbalanced"
    IsTarget = "IsTarget"


class DataManagerBase(metaclass=abc.ABCMeta):
    """The class is an interface for DataManager instances, which are objects that manage the data.

    The user must implement the abstract methods.
    """

    def __init__(self, description: str = "") -> None:
        """Initializes the DataManageBase class.

        Args:
            description: The description of the data manager.
        """
        self.id: str = str(uuid.uuid4())[:8]
        self.description: str = description

    def __repr__(self) -> str:
        """Represent object instance as string.

        Returns:
            String representation.
        """
        return str(self.id)

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
            hasattr(subclass, "get_training_data")
            and callable(subclass.get_training_data)
            and hasattr(subclass, "get_validation_data")
            and callable(subclass.get_validation_data)
            or NotImplemented
        )

    @abc.abstractmethod
    def get_training_data(self) -> Any:
        """Return the data for model training.

        Returns:
            A 2-tuple in which the first element is X and the second is y.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_validation_data(self) -> Any:
        """Return the data for model validation

        Returns:
            A 2-tuple in which the first element is X and the second is y.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_prediction_type(self) -> str:
        """Return the type of the prediction that you want to make.

        Returns:
            The type of the prediction.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_data_type(self) -> str:
        """Return the type of the data.

        Returns:
            The type of the data.
        """
        raise NotImplementedError


class ModelManagerBase(metaclass=abc.ABCMeta):
    """This class is an interface for managing models.

    The user must implement the abstract methods.
    """

    def __init__(self, model: Any, name: str, desc: str = "") -> None:
        """Initializes the ModelManagerBase class.

        Args:
            model: A model object representing the model to be fitted.
            name: A string representing the name of the model.
            desc: A string representing a description of the model. Default is an empty string.
        """
        self.model: Any = model
        self.name: str = name
        self.desc: str = desc

    def fit_with_kwargs(self, data: Any, kwargs: Dict[str, Any]) -> None:
        """Fits the model using the data provided and any keyword arguments specified.

        Only keyword arguments that are compatible with the fit method of the model will be used.

        Args:
            data: A pandas DataFrame or numpy array containing the data to fit the model.
            kwargs: A dictionary containing keyword arguments to be passed to the fit method of
                the model.
        """
        if hasattr(self, "fit"):
            fit: Callable = self.fit
            # This is a bound method, so `self` is not required in invocation.
            fit(
                data,
                **{k: kwargs[k] for k in kwargs if k in list(inspect.signature(fit).parameters)},
            )
        else:
            raise TypeError(f"{self.__class__.name} has no 'fit' function.")

    def __repr__(self) -> str:
        """Represent object instance as string.

        Returns:
            String representation.
        """
        return f"Model: {self.name}. Description: {self.desc}"

    def get_model_info(self) -> Tuple[str, str]:
        """Get the name and description of the model object.

        Returns:
            The name and description of the model object as a 2-tuple.
        """
        return self.name, self.desc

    def get_model(self) -> Any:
        """Get the model object.

        Returns:
            The model object.
        """
        return self.model

    def get_model_pipeline(self, data_manager: Any) -> Any:
        """Get the model object with full pipeline.

        Returns:
            The model object.
        """
        return self.model

    def set_model(self, model: Any) -> None:
        """Set the model object.

        Args:
            model: A model object representing the model to be fitted.
        """
        self.model = model

    def set_name(self, name: str) -> None:
        """Set the model name.

        Args:
           name: A string of the model name.
        """
        self.name = name

    def set_desc(self, desc: str) -> None:
        """Set the model description.

        Args:
            desc: A string of the model description.
        """
        self.desc = desc

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
            hasattr(subclass, "fit")
            and callable(subclass.fit)
            and hasattr(subclass, "predict")
            and callable(subclass.predict)
            and hasattr(subclass, "export")
            and callable(subclass.export)
            or NotImplemented
        )

    @abc.abstractmethod
    def predict(self, X: Any) -> Any:
        """Predict the model.

        Args:
            X: The inference data.

        Returns:
            Predictions.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, data: Any, **kwargs: Any) -> Self:
        """Fit the model.

        Args:
            data: The training data.
            **kwargs: Additional sub-class specific keyword arguments.

        Returns:
            This class instance.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def export(self, path: Union[str, PathLike], **kwargs: Any) -> None:
        """Export model.

        Args:
            path: String or PathLike of file path to export the model into.
            **kwargs: Additional sub-class specific keyword arguments.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def clone(self) -> Self:
        """Creates a copy of this class instance.

        Returns:
            A copy of this class instance.
        """
        raise NotImplementedError


class PandasModelManagerBase(ModelManagerBase):
    """This class is an interface for managing models that use Pandas based data.

    The user must implement the abstract methods.
    """

    @abc.abstractmethod
    def fit(self, data: Tuple[pd.DataFrame, Optional[pd.Series]], **kwargs: Any) -> Self:
        """Fit the model.

        Args:
            data: The training data.
            **kwargs: For compatibility with the base class.

        Returns:
            This class instance.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Use the model to make a prediction.

        Args:
            data: The feature data.

        Returns:
            The prediction results.
        """
        raise NotImplementedError


class MetricManagerBase(metaclass=abc.ABCMeta):
    """This Class is an interface for managing metrics.

    The user must implement the abstract methods.
    """

    def __init__(self, metric: Any, name: str, desc: str = "") -> None:
        """Initializes the MetricManagerBase class.

        Args:
            metric: The object that will perform the metric.
            name: The name of the metric.
            desc: The description of the metric.
        """
        self.metric: Any = metric
        self.name: str = name
        self.desc: str = desc

    def __repr__(self) -> str:
        """Represent object instance as string.

        Returns:
            String representation.
        """
        return f"Metric: {self.name}. Description: {self.desc}"

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
            hasattr(subclass, "calculate")
            and callable(subclass.calculate)
            and hasattr(subclass, "get_optimization_direction")
            and callable(subclass.get_optimization_direction)
            or NotImplemented
        )

    @abc.abstractmethod
    def get_optimization_direction(self) -> MinOrMax:
        """Get the optimization direction.

        Returns:
            `Enum` of `MinOrMax` to point if the optimization is minimum opt or maximum opt.
        """
        raise NotImplementedError


class SupervisedMetricManagerBase(MetricManagerBase):
    """This Class is an interface for managing supervised metrics.

    The user must implement the abstract methods.
    """

    def calculate(
        self,
        true: Union[np.ndarray, pd.DataFrame],
        pred: Union[np.ndarray, pd.DataFrame],
        kwargs: Dict[str, Any],
    ) -> float:
        """Compute the metric score.

        Args:
            true: The true labels.
            pred: The predicted labels.

        Returns:
            The metric score.
        """
        return self.metric(
            true,
            pred,
            **{
                k: kwargs[k] for k in kwargs if k in list(inspect.signature(self.metric).parameters)
            },
        )


class UnsupervisedMetricManagerBase(MetricManagerBase):
    """This Class is an interface for managing unsupervised metrics.

    The user must implement the abstract methods.
    """

    def calculate(
        self, X: Union[np.ndarray, pd.DataFrame], labels: Sequence[str], kwargs: Dict[str, Any]
    ) -> float:
        """Compute the metric score.

        Args:
            X: The feature data.
            labels: The class labels.
            kwargs: Additional arguments to pass to the metric.

        Returns:
            The metric score.
        """
        return self.metric(
            X,
            labels,
            **{
                k: kwargs[k] for k in kwargs if k in list(inspect.signature(self.metric).parameters)
            },
        )


class DataFetcherBase(metaclass=abc.ABCMeta):
    """This class is an interface for classes that handle loading data.

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
        return hasattr(subclass, "get_items") and callable(subclass.get_items) or NotImplemented

    @abc.abstractmethod
    def get_items(self) -> pd.DataFrame:
        """Return the raw data.

        Returns:
            The loaded data.
        """
        raise NotImplementedError


class SplitterBase(metaclass=abc.ABCMeta):
    """This class is an interace for classes that implement splitting data by some algorithm.

    The user must implement the abstract methods.
    """

    def __init__(self, target: Optional[str] = None) -> None:
        """Initialize the SplitterBase class.

        Args:
            target: The name of the independent (target) variable.
        """
        del target

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
        return hasattr(subclass, "split") and callable(subclass.split) or NotImplemented

    @abc.abstractmethod
    def split(self, data: pd.DataFrame) -> Dict[Dataset, pd.Index]:
        """Split the data into datasets.

        Args:
            data: The data you want to split.

        Returns:
            A dictionary where the keys are names of the splitted dataset and the values are
            the indexes for the data subsets.
        """
        raise NotImplementedError


class ManipulateAdapterBase(metaclass=abc.ABCMeta):
    """This class is an interface for manipulate step adapter.

    The user must implement the abstract methods.
    """

    def __init__(self, manipulator: Any, func_name: str) -> None:
        """Initializes the ManipulateAdapterBase class.

        Args:
            manipulator: A manipulator object with function to be used as step in train pipeline.
            func_name: A string representing the name of the function.
        """
        self.manipulator: Any = manipulator
        self.func_name: str = func_name

    def __str__(self) -> str:
        """Describe object instance as string.

        Returns:
            String description.
        """
        return f"{self.func_name}"

    def __hash__(self) -> int:
        """Return the hash value of the object instance.

        Returns:
            The hash value of the object instance.
        """
        return hash((self.manipulator, self.func_name))

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
        return hasattr(subclass, "manipulate") and callable(subclass.manipulate) or NotImplemented

    @abc.abstractmethod
    def manipulate(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Manipulate function adapted from original object.

        Args:
            X: Feature data.
            y: Optional target data.

        Returns:
            Manipulated data.
        """
        raise NotImplementedError


class TransformerAdapterBase(metaclass=abc.ABCMeta):
    """This class is an interface for transformer step adapter.

    The user must implement the abstract methods.
    """

    def __init__(self, transformer: Any) -> None:
        """Initializes the TransformerAdapterBase class.

        Args:
            transformer: A transformer object to be used as step in pipeline.
        """
        self.transformer: Any = transformer

    def __str__(self) -> str:
        """Describe object instance as string.

        Returns:
            String description.
        """
        return f"{self.transformer}"

    def __hash__(self) -> int:
        """Return the hash value of the object instance.

        Returns:
            The hash value of the object instance.
        """
        return hash((self.transformer))

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
            hasattr(subclass, "fit")
            and callable(subclass.fit)
            or hasattr(subclass, "transform")
            and callable(subclass.transform)
            or hasattr(subclass, "fit_transform")
            and callable(subclass.fit_transform)
            or NotImplemented
        )

    @abc.abstractmethod
    def fit(self, y: pd.Series) -> TransformerAdapterBase:
        """'fit' function adapted from original object.

        Args:
            y: Training data.

        Returns:
            This object instance.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, y: pd.Series) -> pd.Series:
        """'transform' function adapted from original object.

        Args:
            y: Training data.

        Returns:
            Transformed data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit_transform(self, y: pd.Series) -> pd.Series:
        """'fit_transform' function adapted from original object.

        Args:
            y: Training data.

        Returns:
            Transformed data.
        """
        raise NotImplementedError


class LoggerBase(metaclass=abc.ABCMeta):
    """This class is an interface for managing logging.

    The user must implement the abstract methods.
    """

    @abc.abstractmethod
    def start_run_log(
        self,
        run_name: str,
        model: ModelManagerBase,
        metrics: Sequence[MetricManagerBase],
        data_manager: DataManagerBase,
        models_kwargs: Dict[str, Any],
        metrics_kwargs: Dict[str, Any],
        opt_metric: Optional[MetricManagerBase],
        experiment_description: str,
    ) -> None:
        """Start run log.

        Args:
            run_name: The run name to start.
            model: The model.
            metrics: The metrics.
            data_manager: The data manager that provides the training data.
            models_kwargs: Additional keyword arguments for the model.
            metrics_kwargs: Additional keyword arguments for the metrics.
            opt_metric: The optimization metric manager.
            experiment_description: The experiment description.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log_run_results(
        self,
        run_name: str,
        model: ModelManagerBase,
        metrics: Dict[str, Any],
        data_manager: DataManagerBase,
        models_kwargs: Dict[str, Any],
        metrics_kwargs: Dict[str, Any],
        opt_metric: Optional[MetricManagerBase],
        experiment_description: str,
    ) -> None:
        """Log a run results.

        Args:
            run_name: The run name to log model results.
            model: The model that was trained.
            metrics: The metrics reulst of the model.
            data_manager: The data manager that provides the training data.
            models_kwargs: Additional keyword arguments for the model.
            metrics_kwargs: Additional keyword arguments for the metrics.
            opt_metric: The optimization metric manager.
            experiment_description: The experiment description.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def end_run_log(
        self,
        run_name: str,
        model: ModelManagerBase,
        metrics: Dict[str, Any],
        data_manager: DataManagerBase,
        models_kwargs: Dict[str, Any],
        metrics_kwargs: Dict[str, Any],
        opt_metric: Optional[MetricManagerBase],
        experiment_description: str,
    ) -> None:
        """End run log.

        Args:
            run_name: The run name to log best model results.
            model: The  model that was trained.
            metrics: The metrics reulst of the best model.
            data_manager: The data manager that provides the training data.
            models_kwargs: Additional keyword arguments for the model.
            metrics_kwargs: Additional keyword arguments for the metrics.
            opt_metric: The optimization metric manager.
            experiment_description: The experiment description.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log_summary_results(
        self,
        run_name: str,
        model: ModelManagerBase,
        metrics: Dict[str, Any],
        data_manager: DataManagerBase,
        models_kwargs: Dict[str, Any],
        metrics_kwargs: Dict[str, Any],
        opt_metric: Optional[MetricManagerBase],
        experiment_description: str,
    ) -> None:
        """Log summary results.

        Args:
            run_name: The run name to log model results.
            model: The best model that was trained.
            metrics: The metrics reulst of the model.
            data_manager: The data manager that provides the training data.
            models_kwargs: Additional keyword arguments for the model.
            metrics_kwargs: Additional keyword arguments for the metrics.
            opt_metric: The optimization metric manager.
            experiment_description: The experiment description.
        """
        raise NotImplementedError


class OptimizerBase(metaclass=abc.ABCMeta):
    """This class is an interface for managing optimizers.

    The user must implement the abstract methods.
    """

    @abc.abstractmethod
    def set_data_metric(
        self, data_manager: DataManagerBase, optimize_metric: MetricManagerBase
    ) -> None:
        """Set the data metric.

        Args:
            data_manager: Any type of data manager object.
            optimize_metric: Metric wrapper of the optimized metric.
        """
        raise NotImplementedError

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
        return hasattr(subclass, "optimized") and callable(subclass.optimized) or NotImplemented

    @abc.abstractmethod
    def optimize(self, model: Any) -> Any:
        """Optimize model.

        Args:
           model: Model wrapper.

        Returns:
            The model wrapper.
        """
        raise NotImplementedError


class TrainerBase(metaclass=abc.ABCMeta):
    """Base class for trainers."""

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
            hasattr(subclass, "fit_and_evaluate")
            and callable(subclass.fit_and_evaluate)
            or NotImplemented
        )

    @abc.abstractmethod
    def fit_and_evaluate(
        self,
        model: ModelManagerBase,
        data_manager: DataManagerBase,
        models_kwargs: Dict[str, Any],
        metrics_kwargs: Dict[str, Any],
        metrics_to_train: Sequence[MetricManagerBase],
        opt_metric: MetricManagerBase,
        current_experiment_id: Optional[str] = None,
        save_models_to_disk: bool = False,
        checkpoints_path: Optional[str] = None,
        empty_cache_func: Optional[Callable] = None,
        load_checkpoints_func: Optional[Callable] = None,
    ) -> Tuple[ModelManagerBase, Dict[str, Any]]:
        """Fit the model with the given data manager and model arguments, and then evaluate it.

        Args:
            model: The model to train and evaluate.
            data_manager: The data manager that provides the training data.
            models_kwargs: Additional keyword arguments for the model.
            metrics_kwargs: Additional keyword arguments for the metrics.
            metrics_to_train: List of metrics to be used during training.
            opt_metric: The optimization metric manager.

        Returns:
            A 2-tuple with the model manager and evaluation dictionary.
        """
        raise NotImplementedError


class InferenceManagerBase(metaclass=abc.ABCMeta):
    """This class is an interface for managing inference.

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
            or hasattr(subclass, "__init__")
            and callable(subclass.__init__)
            or hasattr(subclass, "export")
            and callable(subclass.export)
            or NotImplemented
        )

    @abc.abstractmethod
    def __init__(self, data_manager: DataManagerBase, model: ModelManagerBase):
        """Initialize the InferenceManagerBase.

        Args:
            data_manager: The data manager to be used for inference.
            model: The model to be used for inference.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, data: Iterable) -> Iterable:
        """Make predictions with the model.

        Args:
           data: Data to make predictions on.

        Returns:
            The model's predictions.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def export(self, path: Union[str, PathLike], **kwargs: Any) -> None:
        """Export class.

        Args:
            path: String or PathLike of file path to export the class into.
            **kwargs: Additional sub-class specific keyword arguments.
        """
        raise NotImplementedError
