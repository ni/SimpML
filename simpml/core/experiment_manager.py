"""Experiment Manager."""

from __future__ import annotations

import gc
import os
import pickle
import shutil
import uuid
import warnings
from copy import deepcopy
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from IPython import display

from simpml.common.metrics_pool import METRICS_POOL
from simpml.common.models_pool import MODELS_POOL
from simpml.core.base import (
    DataManagerBase,
    DataType,
    LoggerBase,
    MetricManagerBase,
    MetricName,
    MinOrMax,
    ModelManagerBase,
    OptimizerBase,
    PredictionType,
    TrainerBase,
)
from simpml.core.trainers_pool import StandardTrainer


warnings.filterwarnings("ignore", message="All message displayed in console.")


class ExperimentManager:
    """The Class initializes a number of models depending on the type of data and makes predictions
    on a number of metrics.

    Input:
        data_manager: Object that contains preprocessed data.
        random_state: Seed for the distribution of the majority population in each subset.

    Attributes:
        get_df_metrics(): Trains the relevant data models and presents their results.
        prediction_type: The type of models trained.
        df_metrics: Data frame with results on all models.
        best_model: The model with the best score in the selected metric.
    """

    def __init__(
        self,
        data_manager: DataManagerBase,
        optimize_metric: Optional[Union[str, MetricName]],
        random_state: int = 0,
        n_jobs: int = 1,
        trainer: Optional[TrainerBase] = None,
        hyper_parameters_optimizer: Optional[OptimizerBase] = None,
        logger: Optional[LoggerBase] = None,
        verbose: bool = True,
        save_models_to_disk: bool = False,
        checkpoints_path: Optional[str] = ".checkpoints",
        load_checkpoints_func: Optional[Callable[[str], Any]] = None,
        empty_cache_func: Optional[Callable[[], None]] = None,
    ) -> None:
        """Initializes the ExperimentManager class.

        Args:
            data_manager: Object that contains preprocessed data.
            optimize_metric: The name of the metric to optimize. If None, the metrics must be
                specified later (e.g. with `self.set_opt_metric` or `self.add_metrics`).
            random_state: Seed for the distribution of the majority population in each subset.
            n_jobs: The number of parallel jobs.
            trainer: TrainerBase = StandardTrainer(),
            hyper_parameters_optimizer: Optional hyper-parameter optimization object.
            verbose: Whether to output more information.
            save_models_to_disk: Whether to save trained models to disk.
            checkpoints_path: The path to the checkpoints.
            load_checkpoints_func: The function to use to load the checkpoints. If `None`, will
                use the default function (loads pickle files).

        Raises:
            ValueError: If `checkpoints_path` is None or empty string and `save_models_to_disk` is
                True.
        """
        if not checkpoints_path and save_models_to_disk:
            raise ValueError("'checkpoints_path' must be set if 'save_models_to_disk' is True")
        self.n_jobs: int = n_jobs
        self.seed: int = random_state
        self.verbose: bool = verbose
        self.save_models_to_disk: bool = save_models_to_disk
        self.checkpoints_path: Optional[str] = checkpoints_path
        self.load_checkpoints_func: Callable[[str], Any] = (
            load_checkpoints_func
            if load_checkpoints_func is not None
            else self.__default_load_checkpoints_func
        )
        self.empty_cache_func: Callable[[], None] = (
            empty_cache_func if empty_cache_func is not None else self.__default_empty_cache_func
        )
        if trainer is None:
            trainer = StandardTrainer()
        self.trainer: TrainerBase = trainer
        self.__initialize_checkpoints_folder()
        self.list_experiment_id: List[str] = []
        if isinstance(data_manager.get_prediction_type(), str):
            if data_manager.get_prediction_type() in (item.value for item in PredictionType):
                self.prediction_type: Union[str, PredictionType] = PredictionType.from_value(
                    data_manager.get_prediction_type()
                )
            else:
                self.prediction_type = data_manager.get_prediction_type()
        else:
            self.prediction_type = cast(PredictionType, data_manager.get_prediction_type())
        if isinstance(data_manager.get_data_type(), str):
            if data_manager.get_data_type() in (item.value for item in DataType):
                self.data_type: Union[str, DataType] = DataType.from_value(
                    data_manager.get_data_type()
                )
            else:
                self.data_type = data_manager.get_data_type()
        else:
            self.data_type = cast(DataType, data_manager.get_data_type())

        self.metrics_pool: List[MetricManagerBase] = self.__get_assets("metrics")
        self.metrics_to_train: List[MetricManagerBase] = self.metrics_pool.copy()
        self.columns_df_metrics: List[str] = self.__initialize_df_metrics_columns()
        self.opt_metric: Optional[MetricManagerBase] = self.__get_opt_metric(optimize_metric)
        self.models_pool: List[ModelManagerBase] = self.__get_assets("models")
        self.models_to_train: List[ModelManagerBase] = self.models_pool.copy()
        self.data: List[DataManagerBase] = [data_manager]
        self.trained_models: Dict[str, Dict[str, Any]] = {}
        self.best_model: Optional[Any] = None
        self.df_metrics: Optional[pd.DataFrame] = None
        self.hyper_parameters_optimizer: Optional[OptimizerBase] = hyper_parameters_optimizer
        if self.hyper_parameters_optimizer and self.opt_metric:
            self.hyper_parameters_optimizer.set_data_metric(data_manager, self.opt_metric)
        self.logger: Optional[LoggerBase] = logger
        self.start_time: datetime = datetime.now()

    # --------------#
    # private methods
    # --------------#

    def __initialize_checkpoints_folder(self) -> None:
        """Initialize the checkpoints folder."""
        if self.checkpoints_path:
            if os.path.exists(self.checkpoints_path):
                shutil.rmtree(self.checkpoints_path)
            if self.save_models_to_disk:
                os.makedirs(self.checkpoints_path, exist_ok=True)

    def __default_load_checkpoints_func(self, path: Union[str, PathLike]) -> Any:
        """The default load checkpoints function. Uses pickle.

        Args:
            path: The path to the pickle file.

        Returns:
            The model object.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def __default_empty_cache_func(self) -> None:
        gc.collect()

    def __initialize_df_metrics_columns(self) -> List[str]:
        """Initilalize the data frame metrics columns.

        Returns:
            A list of data frame metrics column names.
        """
        return (
            [
                "Experiment ID",
                "Experiment Description",
                "Model",
                "Model Description",
                "Data Version",
                "Data Description",
                "Model Params",
                "Metric Params",
            ]
            + [metric.name for metric in self.metrics_to_train]
            + ["Run Time"]
        )

    def __highlight_opt_metric(
        self, series: Union[np.ndarray, pd.Series], is_flipped: bool = False
    ) -> List[str]:
        """Highlight in green the opt-metric in a series.

        Args:
            series: The series to highlight.
            is_flipped: If True, use minimum values. Otherwise use maximum values.

        Returns:
            A list of color strings for use with Plotly.
        """
        opt = series.min() if is_flipped else series.max()
        is_opt = series == opt
        return ["background-color: green" if v else "" for v in is_opt]

    def __highlight_min_max(
        self, series: Union[np.ndarray, pd.Series], is_flipped: bool = False
    ) -> List[str]:
        """Highlight minimum and maximum values in a series.

        Args:
            series: The series to highlight.
            is_flipped: If True, use minimum values. Otherwise use maximum values.

        Returns:
            A list of color strings for use with Plotly.
        """
        style_template = "background-color: %s"
        min_color, max_color = "red", "yellow"
        if is_flipped:
            min_color, max_color = max_color, min_color

        is_min = series == series.min()
        is_max = series == series.max()
        series_list: List[str] = []

        for i in range(len(series)):
            style = ""
            if is_min[i]:
                style = style_template % min_color
            elif is_max[i]:
                style = style_template % max_color
            series_list.append(style)

        return series_list

    def __repr__(self) -> str:
        """Represent object instance as string.

        Returns:
            String representation.
        """
        return (
            f"Prediction Type: {self.prediction_type}, Metric: {self.opt_metric}, "
            f"Random State: {self.seed}"
        )

    def __get_assets(self, assets: str) -> List[Any]:
        """Get assets based on type.

        Args:
            assets: The asset type. Either "metrics" or "models".

        Returns:
            The list of specified assets.
        """
        data_type = self.data_type if isinstance(self.data_type, str) else self.data_type.value
        prediction_type = (
            self.prediction_type
            if isinstance(self.prediction_type, str)
            else self.prediction_type.value
        )

        if assets == "models":
            if data_type not in MODELS_POOL:
                print(
                    f"There is no models pool for data_type: {data_type}."
                    ' Please add models using: "add_models" function'
                )
                return []

            if prediction_type not in MODELS_POOL[data_type]:
                print(
                    f"There is no models pool for data_type: {data_type} and prediction_type"
                    f' {prediction_type}. Please add models using: "add_models" function'
                )
                return []

            return MODELS_POOL[data_type][prediction_type]

        if assets == "metrics":
            if data_type not in METRICS_POOL:
                print(
                    f"There is no metrics pool for data_type: {data_type}."
                    ' Please add metrics using: "add_metrics" function'
                )
                return []

            if prediction_type not in METRICS_POOL[data_type]:
                print(
                    f"There is no metrics pool for data_type: {data_type} and prediction_type"
                    f' {prediction_type}. Please add metrics using: "add_metrics" function'
                )
                return []

        return METRICS_POOL[data_type][prediction_type]

    def __get_opt_metric(
        self, opt_metric: Union[str, MetricName, None]
    ) -> Optional[MetricManagerBase]:
        """Get a metric manager object based on metric name.

        Args:
            opt_metrics: The metric name.

        Returns:
            The corresponding metric manager object.
        """
        assert opt_metric is not None
        if isinstance(opt_metric, str) and opt_metric in (item.value for item in MetricName):
            opt_metric = MetricName.from_value(opt_metric).value
        if isinstance(opt_metric, MetricName):
            opt_metric = opt_metric.value
        metric = [metric for metric in self.metrics_to_train if metric.name == opt_metric]
        if metric:
            return metric[0]
        else:
            print(
                f"Metric {opt_metric} Not found in your list of metrics that includes: "
                f"{self.metrics_to_train} Please add the metric or choose another"
            )
            return None

    def __get_best_row(self, opt_metric: Optional[Union[str, MetricName]]) -> pd.DataFrame:
        """Get the row with the best result for the selected metric.

        Args:
            opt_metric: The metric name.

        Returns:
            A data frame view of the row with the best result for the selected metric.
        """
        if opt_metric is not None:
            opt_metric_obj = self.__get_opt_metric(opt_metric)
        else:
            opt_metric_obj = self.opt_metric
        assert opt_metric_obj is not None
        if opt_metric_obj.get_optimization_direction() == MinOrMax.Max:
            assert self.df_metrics is not None
            best_model_row = self.df_metrics.loc[
                self.df_metrics[opt_metric_obj.name] == self.df_metrics[opt_metric_obj.name].max()
            ]

        elif opt_metric_obj.get_optimization_direction() == MinOrMax.Min:
            assert self.df_metrics is not None
            best_model_row = self.df_metrics.loc[
                self.df_metrics[opt_metric_obj.name] == self.df_metrics[opt_metric_obj.name].min()
            ]
        else:
            raise RuntimeError("Could not find best model row.")
        return best_model_row

    def __get_models_list_names_in_df(self, models: Sequence[ModelManagerBase]) -> pd.DataFrame:
        """Create an empty data frame with column names as model names.

        Args:
            models: A list or tuple of model manager objects.

        Returns:
            The empty data frame with column names as model names.
        """
        return pd.DataFrame({model.name: model.desc for model in models}, index=[0]).T.rename(
            columns={0: "Description"}
        )

    def __highlight_isactive(self, series: Union[np.ndarray, pd.Series]) -> List[str]:
        """Highlight which entries are active.

        Args:
            series: The series to highlight.

        Returns:
            A list of color strings for use with Plotly.
        """
        # A Pandas series is a special case where `series == True` is different from
        # `series is True`. Ignore this flake8 warning:
        # E712 "comparison to True should be 'if cond is True:' or 'if cond:'"
        is_active = series == True  # noqa: E712
        return ["background-color: yellow" if v else "background-color: red" for v in is_active]

    def __reset_entities(self, entities: str) -> None:
        """Reset entities with their default state.

        Args:
            entities: The entity category, either "models" or "metrics".
        """
        if entities == "models":
            self.models_to_train = self.__get_assets("models").copy()
        elif entities == "metrics":
            self.metrics_to_train = self.__get_assets("metrics").copy()
        else:
            raise ValueError(
                f"Unknown entities: '{entities}'. Please use either 'models' or 'metrics'"
            )
        return self.__display_entities_pool(entities)

    def __remove_entities(self, entities: str, entity_names: Union[str, Sequence[str]]) -> None:
        """Remove specified entities.

        Args:
            entities: The entity category, either "models" or "metrics".
            entity_names: The name of the entity to remove from its specified category.
        """
        if isinstance(entity_names, str):
            entity_names = [entity_names]
        entity_list: Union[List[ModelManagerBase], List[MetricManagerBase]] = (
            self.models_to_train if entities == "models" else self.metrics_to_train
        )
        entity_pool: Union[List[ModelManagerBase], List[MetricManagerBase]] = (
            self.models_pool if entities == "models" else self.metrics_pool
        )
        for entity in entity_list.copy():
            if entity.name in entity_names:
                if entity not in entity_pool:
                    print(
                        f"{entities.capitalize()} {entity.name} was added manually so it will "
                        "not be possible to restore it"
                    )
                entity_list.remove(cast(Any, entity))
        if entities == "metrics":
            self.columns_df_metrics = self.__initialize_df_metrics_columns()
        self.__display_entities_pool(entities)

    def __remove_all_entities(self, entities: str) -> None:
        """Remove all entities from a specified category.

        Args:
            entities: The entity category, either "models" or "metrics".
        """
        if entities == "models":
            self.models_to_train = []
        elif entities == "metrics":
            self.metrics_to_train = []
        else:
            raise ValueError(
                f"Unknown entities: '{entities}'. Please use either 'models' or 'metrics'"
            )
        self.__display_entities_pool(entities)

    def __add_entities(
        self,
        entities_to_add: Union[
            MetricManagerBase, ModelManagerBase, List[MetricManagerBase], List[ModelManagerBase]
        ],
        entities: str,
    ) -> None:
        """Add specified entities.

        Args:
            entities_to_add: The entity/entities to add to the specified category.
            entities: The entity category, either "models" or "metrics".
        """
        if not isinstance(entities_to_add, list):
            entities_to_add = cast(
                Union[List[MetricManagerBase], List[ModelManagerBase]], [entities_to_add]
            )
        if entities == "models":
            self.models_to_train += cast(List[ModelManagerBase], entities_to_add)
        elif entities == "metrics":
            self.metrics_to_train += cast(List[MetricManagerBase], entities_to_add)
            self.columns_df_metrics = self.__initialize_df_metrics_columns()
        else:
            raise ValueError(
                f"Unknown entities: '{entities}'. Please use either 'models' or 'metrics'"
            )
        return self.__display_entities_pool(entities)

    def __display_entities_pool(self, entities: str) -> None:
        """Display all entities from a specified category.

        Args:
            entities: The entity category, either "models" or "metrics".
        """
        if entities == "models":
            entity_list: Union[
                List[ModelManagerBase], List[MetricManagerBase]
            ] = self.models_to_train
            entity_pool: Union[List[ModelManagerBase], List[MetricManagerBase]] = self.models_pool
            columns: List[str] = ["Name", "Description", "Source", "Is Available"]
            sort_columns = ["Is Available"]
        elif entities == "metrics":
            entity_list = self.metrics_to_train
            entity_pool = self.metrics_pool
            columns = ["Name", "Description", "Source", "Is Available", "Is Optimal"]
            sort_columns = ["Is Available", "Is Optimal"]
        else:
            raise ValueError(
                f"Unknown entities: '{entities}'. Please use either 'models' or 'metrics'."
            )

        entities_table = pd.DataFrame(columns=columns)
        all_entities_list = list(set(entity_list + entity_pool))

        for entity in all_entities_list:
            entity_source = "Pool" if entity in entity_pool else "Custom"
            row_data: List[Any] = [entity.name, entity.desc, entity_source]
            if entities == "metrics":
                is_available = entity in self.metrics_to_train
                is_optimal = entity == self.opt_metric
                row_data.extend([is_available, is_optimal])
            elif entities == "models":
                is_available = entity in self.models_to_train
                row_data.extend([is_available])
            entities_table.loc[len(entities_table)] = row_data

        entities_table = entities_table.sort_values(by=sort_columns, ascending=False)
        entities_table = entities_table.reset_index(drop=True)
        html = entities_table.style.apply(
            self.__highlight_isactive, subset=sort_columns
        ).set_properties(subset=["Description"])
        display.display(html)  # type: ignore [no-untyped-call]

    def __generate_experiment_id(self) -> str:
        """Generate a new experiment ID.

        Returns:
            The generated experiment ID.
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_uuid = str(uuid.uuid4())[:4]
        return timestamp + "_" + random_uuid

    # --------------#
    # public methods
    # --------------#

    def get_current_data_manager(self) -> DataManagerBase:
        """Get the current data manager.

        Returns:
            The current data manager.
        """
        return self.data[-1]

    def get_current_experiment_id(self) -> str:
        """Get the current experiment ID.

        Returns:
            The current experiment ID.
        """
        if self.list_experiment_id:
            return self.list_experiment_id[-1]
        else:
            return "You have no runs yet"

    def save_model(self, model: ModelManagerBase) -> None:
        """Save the model to disk or store it in memory."""
        if self.save_models_to_disk:
            assert self.checkpoints_path
            path_to_save = (
                Path(self.checkpoints_path) / f"{model.name}_{self.get_current_experiment_id()}.pkl"
            )
            model.export(path_to_save)
            self.trained_models[self.get_current_experiment_id()][model.name] = path_to_save
            del model
            self.empty_cache_func()
        else:
            self.trained_models[self.get_current_experiment_id()][model.name] = deepcopy(model)

    def run_experiment(
        self,
        models_kwargs: Optional[Dict[str, Any]] = None,
        metrics_kwargs: Optional[Dict[str, Any]] = None,
        experiment_description: str = "",
    ) -> None:
        """Run the experiment."""

        def initialize_df_metrics() -> pd.DataFrame:
            """Initialize the metrics data frame.

            Returns:
                 Initialized metrics data frame.
            """
            return pd.DataFrame(columns=self.columns_df_metrics)

        models_kwargs = models_kwargs or {}
        metrics_kwargs = metrics_kwargs or {}
        data_manager = self.get_current_data_manager()
        self.list_experiment_id.append(self.__generate_experiment_id())
        self.trained_models[self.get_current_experiment_id()] = {}
        if self.df_metrics is None:
            self.df_metrics = initialize_df_metrics()
        for model in self.models_to_train:
            model = model.clone()
            self.start_time = datetime.now()
            self.start_run_log(
                model,
                self.metrics_to_train,
                data_manager,
                models_kwargs,
                metrics_kwargs,
                self.opt_metric,
                experiment_description,
            )
            self.optimize_hyperparameters(model)
            if self.opt_metric is None:
                raise ValueError("self.opt_metric must be set before calling fit_and_evaluate.")

            models_kwargs["experiment_manager"] = self
            model, metrics = self.trainer.fit_and_evaluate(
                model,
                data_manager,
                models_kwargs,
                metrics_kwargs,
                self.metrics_to_train,
                self.opt_metric,
                self.get_current_experiment_id(),
                self.save_models_to_disk,
                self.checkpoints_path,
                self.empty_cache_func,
                self.load_checkpoints_func,
            )
            self.log_run_results(
                model,
                metrics,
                data_manager,
                models_kwargs,
                metrics_kwargs,
                self.opt_metric,
                experiment_description,
            )
            self.update_metrics_table(
                model, data_manager, metrics, models_kwargs, metrics_kwargs, experiment_description
            )
            self.save_model(model)  # save the model after it has been trained and evaluated
            self.end_run_log(
                model,
                metrics,
                data_manager,
                models_kwargs,
                metrics_kwargs,
                self.opt_metric,
                experiment_description,
            )
            self.display_experiment_results()
        self.finalize_experiment()
        self.log_summary_results(
            self.get_best_model_pipeline(),
            self.__get_best_row(None).iloc[0][[metric.name for metric in self.metrics_to_train]],
            data_manager,
            models_kwargs,
            metrics_kwargs,
            self.opt_metric,
            experiment_description,
        )

    def start_run_log(
        self,
        model: ModelManagerBase,
        metrics: List[MetricManagerBase],
        data_manager: DataManagerBase,
        models_kwargs: Dict[str, Any],
        metrics_kwargs: Dict[str, Any],
        opt_metric: Optional[MetricManagerBase],
        experiment_description: str,
    ) -> None:
        """Start the run logger."""
        if self.logger is not None:
            self.logger.start_run_log(
                self.get_current_experiment_id(),
                model,
                metrics,
                data_manager,
                models_kwargs,
                metrics_kwargs,
                opt_metric,
                experiment_description,
            )

    def end_run_log(
        self,
        model: ModelManagerBase,
        metrics: Dict[str, Any],
        data_manager: DataManagerBase,
        models_kwargs: Dict[str, Any],
        metrics_kwargs: Dict[str, Any],
        opt_metric: Optional[MetricManagerBase],
        experiment_description: str,
    ) -> None:
        """End the logger."""
        if self.logger is not None:
            self.logger.end_run_log(
                self.get_current_experiment_id(),
                model,
                metrics,
                data_manager,
                models_kwargs,
                metrics_kwargs,
                opt_metric,
                experiment_description,
            )

    def log_run_results(
        self,
        model: ModelManagerBase,
        metrics: Dict[str, Any],
        data_manager: DataManagerBase,
        models_kwargs: Dict[str, Any],
        metrics_kwargs: Dict[str, Any],
        opt_metric: Optional[MetricManagerBase],
        experiment_description: str,
    ) -> None:
        """Log run results"""
        if self.logger is not None:
            self.logger.log_run_results(
                self.get_current_experiment_id(),
                model,
                metrics,
                data_manager,
                models_kwargs,
                metrics_kwargs,
                opt_metric,
                experiment_description,
            )

    def log_summary_results(
        self,
        model: ModelManagerBase,
        metrics: Dict[str, Any],
        data_manager: DataManagerBase,
        models_kwargs: Dict[str, Any],
        metrics_kwargs: Dict[str, Any],
        opt_metric: Optional[MetricManagerBase],
        experiment_description: str,
    ) -> None:
        """Log summary results"""
        if self.logger is not None:
            self.logger.log_summary_results(
                self.get_current_experiment_id(),
                model,
                metrics,
                data_manager,
                models_kwargs,
                metrics_kwargs,
                opt_metric,
                experiment_description,
            )

    def optimize_hyperparameters(self, model: ModelManagerBase) -> None:
        """Optimize the hyperparameters of the model, if a hyperparameters optimizer is provided."""
        if self.hyper_parameters_optimizer:
            model = self.hyper_parameters_optimizer.optimize(model)

    def update_metrics_table(
        self,
        model: ModelManagerBase,
        data_manager: DataManagerBase,
        metrics: Dict[str, Any],
        models_kwargs: Dict[str, Any],
        metrics_kwargs: Dict[str, Any],
        experiment_description: str,
    ) -> None:
        """Update the metrics table with the results of the model evaluation."""
        values = [
            self.list_experiment_id[-1],
            experiment_description,
            model.name,
            model.desc,
            data_manager,
            data_manager.description,
            models_kwargs,
            metrics_kwargs,
        ] + list(metrics.values())
        values.append(":".join(str(datetime.now() - self.start_time).split(".")[:1]))
        assert self.df_metrics is not None
        self.df_metrics.loc[len(self.df_metrics)] = dict(zip(self.columns_df_metrics, values))

    def display_experiment_results(self) -> None:
        """Display the experiment results, if verbosity is turned on."""
        if self.verbose:
            min_metrics_names = [
                m.name
                for m in self.metrics_to_train
                if m.get_optimization_direction() == MinOrMax.Min
            ]
            max_metrics_names = [
                m.name
                for m in self.metrics_to_train
                if m.get_optimization_direction() == MinOrMax.Max
            ]
            assert self.df_metrics is not None
            assert self.opt_metric is not None
            html = (
                self.df_metrics.style.apply(
                    self.__highlight_min_max, subset=min_metrics_names, is_flipped=True
                )
                .apply(self.__highlight_min_max, subset=max_metrics_names, is_flipped=False)
                .apply(
                    self.__highlight_opt_metric,
                    subset=[self.opt_metric.name],
                    is_flipped=self.opt_metric.get_optimization_direction() == MinOrMax.Min,
                )
            )
            display.clear_output(wait=True)  # type: ignore [no-untyped-call]
            display.display(html)  # type: ignore [no-untyped-call]

    def finalize_experiment(self) -> None:
        """Perform final steps after all models have been trained and evaluated."""
        if not self.models_to_train:
            raise ValueError(
                'No model were found in your models list. Please add models using "add_models" list'
            )
        self.best_model = self.get_best_model()

    def get_best_opt_metric(self, opt_metric: Optional[Union[str, MetricName]] = None) -> float:
        """Get the value of with the best metric result for the selected metric.

        Args:
            opt_metric: The metric name.

        Returns:
            The value of with the best metric result for the selected metric.
        """
        if opt_metric is not None:
            opt_metric_obj = self.__get_opt_metric(opt_metric)
        else:
            opt_metric_obj = self.opt_metric
        assert opt_metric_obj is not None
        return self.__get_best_row(opt_metric)[opt_metric_obj.name].iloc[0]

    def get_best_model_pipeline(
        self, opt_metric: Optional[Union[str, MetricName]] = None,
        data_manager: Any = None
    ) -> Any:
        """Get the trained model pipeline with the best result for the selected metric.

        Args:
            opt_metric: The metric name.
            data_manager: data_manager to get preprocess data from.

        Returns:
            The inferece manager with the best model pipeline for the selected metric.
        """
        if data_manager is None:
            data_manager = self.get_current_data_manager()
        return self.get_best_model(opt_metric).get_model_pipeline(data_manager)

    def get_best_model(
        self, opt_metric: Optional[Union[str, MetricName]] = None, load_model: bool = True
    ) -> Any:
        """Get the trained model with the best result for the selected metric.

        Args:
            opt_metric: The metric name.
            load_model: Whether to load the model from disk.

        Returns:
            The trained model with the best result for the selected metric.
        """
        best_model_row = self.__get_best_row(opt_metric)
        return self.get_model(
            best_model_row.iloc[0]["Model"],
            best_model_row.iloc[0]["Experiment ID"],
            load_model=load_model,
        )

    def get_model(self, model_name: str, experiment_id: str, load_model: bool = True) -> Any:
        """Get a specific model.

        Args:
            model_name: The name of the model.
            experiment_id: The experiment ID used for training the model.
            load_model: Whether to load the model from disk.

        Returns:
            The specified trained model.
        """
        run_models = self.trained_models.get(experiment_id)
        if run_models is None:
            raise ValueError(
                f"The experiment_id name you entered '{experiment_id}' does not exist in the "
                "experiment_id list"
            )

        selected_model = run_models.get(model_name)
        if selected_model is None:
            raise ValueError(
                f"The model name you entered '{model_name}' does not exist in the model list"
            )

        if self.save_models_to_disk and load_model:
            if os.path.exists(selected_model):
                return self.load_checkpoints_func(selected_model)
            else:
                raise FileNotFoundError(f"No saved model found at '{selected_model}'")
        return selected_model

    def get_data_model_of_best_model(
        self, opt_metric: Optional[Union[str, MetricName]] = None
    ) -> DataManagerBase:
        """Get the data manager of the model with the best result for the selected metric.

        Args:
            opt_metric: The metric name.

        Returns:
            The data manager of the model with the best result for the selected metric.
        """
        best_model_row = self.__get_best_row(opt_metric)
        return best_model_row.iloc[0]["Data Version"]

    def get_models_pool_names(self) -> List[str]:
        """Get the names of the models in the models pool.

        Returns:
            A list of names of the models in the models pool.
        """
        return [model.name for model in self.models_pool]

    def get_models_to_train_names(self) -> List[str]:
        """Get the names of the models to train.

        Returns:
            A list of names of the models to train.
        """
        return [model.name for model in self.models_to_train]

    def get_metrics_to_train_names(self) -> List[str]:
        """Get the names of the metrics use to evaluate models.

        Returns:
            A list of names of the metrics to use to evaluate models.
        """
        return [metric.name for metric in self.metrics_to_train]

    def get_available_models_df(self) -> pd.DataFrame:
        """Create an empty data frame with column names as available model names.

        Returns:
            The empty data frame with column names as available model names.
        """
        return self.__get_models_list_names_in_df(self.models_to_train)

    def get_metric(self, metric_name: str) -> MetricManagerBase:
        """Get a specific metric manager from the list of metrics to use.

        Args:
            metric_name: The name of the metric.

        Returns:
            The specified metric manager object.

        Raises:
            ValueError: If it cannot find the metric manager object.
        """
        for metric in self.metrics_to_train:
            if metric.name == metric_name:
                return metric
        raise ValueError(
            f"The metric name you entered '{metric_name}' does not exist in the metric list"
        )

    def set_subset_models(self, subset_models_list: Sequence[str]) -> None:
        """Remove all models not in the subset list.

        Args:
            subset_models_list: A list or tuple of model names to keep.
        """
        self.remove_models(
            [model for model in self.get_models_to_train_names() if model not in subset_models_list]
        )

    def set_subset_metrics(self, subset_metrics_list: Sequence[str]) -> None:
        """Remove all metric not in the subset list.

        Args:
            subset_metrics_list: A list or tuple of metric names to keep.
        """
        self.remove_metrics(
            [
                metric
                for metric in self.get_metrics_to_train_names()
                if metric not in subset_metrics_list
            ]
        )

    def display_models_pool(self) -> None:
        """Display the models pool."""
        return self.__display_entities_pool("models")

    def display_metrics_pool(self) -> None:
        """Display the metrics pool."""
        return self.__display_entities_pool("metrics")

    def reset_models(self) -> None:
        """Reset the models pool to its default state."""
        return self.__reset_entities("models")

    def reset_metrics(self) -> None:
        """Reset the metrics pool to its default state."""
        self.__reset_entities("metrics")

    def remove_models(self, model_names: Union[str, Sequence[str]]) -> None:
        """Remove specified models from the models pool.

        Args:
            model_names: The names of the model(s) to remove.
        """
        self.__remove_entities("models", model_names)

    def remove_metrics(self, metric_names: Union[str, Sequence[str]]) -> None:
        """Remove specified metrics from the metrics pool.

        Args:
            metric_names: The names of the metric(s) to remove.
        """
        self.__remove_entities("metrics", metric_names)

    def remove_all_models(self) -> None:
        """Remove all models from the models pool."""
        self.__remove_all_entities("models")

    def remove_all_metrics(self) -> None:
        """Remove all metrics from the metrics pool."""
        self.__remove_all_entities("metrics")

    def add_models(self, model: Union[ModelManagerBase, List[ModelManagerBase]]) -> None:
        """Add specified models to the models pool.

        Args:
            model: The model manager(s) to add.
        """
        self.__add_entities(model, "models")

    def add_metrics(
        self, metric: Union[MetricManagerBase, List[MetricManagerBase]], make_opt: bool = False
    ) -> None:
        """Add specified metrics to the metrics pool.

        Args:
            metric: The metric manager(s) to add.
            make_opt: Whether to set the current optimization metric.
        """
        if self.df_metrics is not None:
            if isinstance(metric, (list, tuple)):
                for metric_item in metric:
                    self.df_metrics[metric_item.name] = None
            else:
                self.df_metrics[metric.name] = None
            # Ensure 'Run Time' is the last column
            cols = list(self.df_metrics.columns)
            cols.remove("Run Time")
            cols.append("Run Time")
            self.df_metrics = self.df_metrics[cols]
        if make_opt:
            if isinstance(metric, (list, tuple)):
                self.set_opt_metric(metric[-1].name)
            else:
                self.set_opt_metric(metric.name)
        self.__add_entities(metric, "metrics")

    def set_opt_metric(self, metric_name: str) -> MetricManagerBase:
        """Set the current metric manager.

        Args:
            metric_name: The metric name to use.

        Returns:
            The specified metric manager.
        """
        self.opt_metric = self.get_metric(metric_name)
        return self.opt_metric

    def get_data_description(self) -> pd.DataFrame:
        """Get a description of the data as a data frame.

        Returns:
            A description of the data as a data frame.
        """
        if len(self.data) > 0 and hasattr(self.data[0], "datasets"):
            data = pd.DataFrame(columns=["Data Version", "Data Description", "Datasets"])
            for data_model in self.data:
                if not hasattr(data_model, "datasets"):
                    raise RuntimeError(
                        f"Concrete class of {data_model} does not contain a `datasets` member."
                    )
                datasets = data_model.datasets
                data.loc[len(data)] = [
                    data_model.id,
                    data_model.description,
                    str(list(datasets.keys())),
                ]
        else:
            data = pd.DataFrame(columns=["Data Version", "Data Description"])
            for data_model in self.data:
                data.loc[len(data)] = [
                    data_model.id,
                    data_model.description,
                ]

        return data

    def set_new_data(self, data_manager: DataManagerBase) -> DataManagerBase:
        """Add a new data manager.

        Args:
            data_manager: The data manager object to add.

        Returns:
            The specified data manager object.
        """
        self.data += [data_manager]
        return data_manager
