"""Trainers pool."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, cast, Dict, Generator, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from simpml.core.base import (
    DataManagerBase,
    EnumStringValues,
    MetricManagerBase,
    MinOrMax,
    ModelManagerBase,
    PandasModelManagerBase,
    SupervisedMetricManagerBase,
    TrainerBase,
    UnsupervisedMetricManagerBase,
)
from simpml.tabular.tabular_data_manager import CrossValidationSupervisedTabularDataManager


class StandardTrainer(TrainerBase):
    """Trainer that fits and evaluates a model using standard (non-cross-validated) training."""

    def fit_model(
        self, model: ModelManagerBase, data_manager: DataManagerBase, models_kwargs: Dict[str, Any]
    ) -> ModelManagerBase:
        """Fit the model with the given data manager and model arguments."""
        model.fit_with_kwargs(data_manager.get_training_data(), models_kwargs)
        return model

    def evaluate_model(
        self,
        model: ModelManagerBase,
        data_manager: DataManagerBase,
        metrics_kwargs: Dict[str, Any],
        metrics_to_train: Sequence[MetricManagerBase],
    ) -> Dict[str, Any]:
        """Evaluate the model with the given data manager and metrics arguments."""
        valid_data, valid_target_data = data_manager.get_validation_data()
        predicted = cast(PandasModelManagerBase, model).predict(valid_data)
        if valid_target_data is None:
            return {
                metric.name: cast(
                    Union[SupervisedMetricManagerBase, UnsupervisedMetricManagerBase], metric
                ).calculate(valid_data, cast(Any, predicted), metrics_kwargs)
                for metric in metrics_to_train
            }
        else:
            return {
                metric.name: cast(
                    Union[SupervisedMetricManagerBase, UnsupervisedMetricManagerBase], metric
                ).calculate(valid_target_data, cast(Any, predicted), metrics_kwargs)
                for metric in metrics_to_train
            }

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
        return self.fit_model(model, data_manager, models_kwargs), self.evaluate_model(
            model, data_manager, metrics_kwargs, metrics_to_train
        )


class CVAggregation(EnumStringValues):
    """Enum for aggregation of cross validation results."""

    MEAN = "Mean"
    MIN = "Min"
    MAX = "Max"


class CVSelectedModel(EnumStringValues):
    """Enum for model selection of cross validation results."""

    WORST = "Worst"
    BEST = "Best"


class CVTrainer(TrainerBase):
    """Trainer that fits and evaluates a model using cross-validated training."""

    def __init__(
        self,
        aggregation: Optional[Union[str, CVAggregation]] = None,
        selected_model: Optional[Union[str, CVSelectedModel]] = None,
    ) -> None:
        """Initializes the CVTrainer class.

        Args:
            aggregation: How to aggregate cross validation results.
            selected_model: How to select a model based on cross validation results.
        """
        self.trained_models: Dict[str, Dict[str, List[Union[ModelManagerBase, Path]]]] = {}
        self.metrics_cv: Dict[str, Dict[str, float]] = {}
        self.metrics_per_model: Dict[str, Dict[int, Dict[str, Any]]] = {}
        if aggregation is None:
            self.aggregation: CVAggregation = CVAggregation.MEAN
        elif isinstance(aggregation, str):
            self.aggregation = CVAggregation.from_value(aggregation)
        else:
            self.aggregation = aggregation
        if selected_model is None:
            self.selected_model: CVSelectedModel = CVSelectedModel.BEST
        elif isinstance(selected_model, str):
            self.selected_model = CVSelectedModel.from_value(selected_model)
        else:
            self.selected_model = selected_model

    def fit_model(
        self,
        model: ModelManagerBase,
        data_manager: DataManagerBase,
        models_kwargs: Dict[str, Any],
        current_experiment_id: Optional[str],
        save_models_to_disk: bool,
        checkpoints_path: Optional[str],
        empty_cache_func: Optional[Callable],
    ) -> None:
        """Fit the model with the given data manager and model arguments."""
        # Assuming data_manager.get_training_data() returns a list of (X, y) tuples for each fold
        if current_experiment_id is not None:
            if self.trained_models.get(current_experiment_id) is None:
                self.trained_models[current_experiment_id] = {}
            self.trained_models[current_experiment_id][model.name] = []
        else:
            # Handle the case where current_experiment_id is None, if needed
            raise ValueError("current_experiment_id cannot be None")
        if not isinstance(data_manager, CrossValidationSupervisedTabularDataManager):
            raise TypeError(
                f"Data Manager is {type(data_manager)} instead of "
                "CrossValidationSupervisedTabularDataManager"
            )
        data_folds = data_manager.get_training_data(all_=True)
        data_iterator: Union[zip, Generator[Union[Any, List[Any]], None, None]]
        if isinstance(data_folds, tuple) and len(data_folds) == 2:
            data_iterator = zip(*data_folds)
        else:
            data_iterator = (data for data in data_folds)

        for idx, args_to_fit in enumerate(data_iterator):
            model_copy = model.clone()  # Create a copy of the model for each fold
            model_copy.fit_with_kwargs(args_to_fit, models_kwargs)
            self.save_model(
                model_copy,
                current_experiment_id,
                save_models_to_disk,
                checkpoints_path,
                empty_cache_func,
                idx,
            )
        if save_models_to_disk:
            del model_copy
            if empty_cache_func is not None:
                empty_cache_func()

    def evaluate_model(
        self,
        model: ModelManagerBase,
        data_manager: DataManagerBase,
        metrics_kwargs: Dict[str, Any],
        metrics_to_train: Sequence[SupervisedMetricManagerBase],
        current_experiment_id: Optional[str],
        save_models_to_disk: bool,
        load_checkpoints_func: Optional[Callable],
        empty_cache_func: Optional[Callable],
    ) -> Dict[str, Any]:
        """Evaluate the model with the given data manager and metrics arguments."""
        metric_values: Dict[str, List[float]] = {metric.name: [] for metric in metrics_to_train}
        if model.name not in list(self.metrics_cv.keys()):
            self.metrics_cv[model.name] = {}
            self.metrics_per_model[model.name] = {}
            # Assuming data_manager.get_validation_data() returns a list of (X, y) tuples for
            # each fold
            if not isinstance(data_manager, CrossValidationSupervisedTabularDataManager):
                raise TypeError(
                    f"Data Manager is {type(data_manager)} instead of "
                    "CrossValidationSupervisedTabularDataManager"
                )
            X_folds, y_folds = data_manager.get_validation_data(all_=True)
            assert y_folds is not None
            assert current_experiment_id is not None
            assert load_checkpoints_func is not None
            for i, (X, y, matching_model) in enumerate(
                zip(
                    X_folds,
                    y_folds,
                    self.get_models_generator(
                        model.name,
                        current_experiment_id,
                        save_models_to_disk,
                        load_checkpoints_func,
                    ),
                )
            ):
                predicted = cast(PandasModelManagerBase, matching_model).predict(X)
                metric_values_for_this_model = {}
                for metric in metrics_to_train:
                    metric_value = metric.calculate(y, predicted, metrics_kwargs)
                    metric_values[metric.name].append(metric_value)
                    metric_values_for_this_model[metric.name] = metric_value
                self.metrics_per_model[matching_model.name][i] = metric_values_for_this_model
                model_name = matching_model.name
                if save_models_to_disk:
                    del matching_model
                    if empty_cache_func is not None:
                        empty_cache_func()
            # Calculate average, standard deviation, minimum and maximum for each metric
            for metric_name, values in metric_values.items():
                self.metrics_cv[model_name][f"{metric_name}_mean"] = cast(float, np.mean(values))
                self.metrics_cv[model_name][f"{metric_name}_std"] = cast(float, np.std(values))
                self.metrics_cv[model_name][f"{metric_name}_min"] = cast(float, np.min(values))
                self.metrics_cv[model_name][f"{metric_name}_max"] = cast(float, np.max(values))
        str_split = (
            "_mean"
            if self.aggregation == CVAggregation.MEAN
            else "_max"
            if self.aggregation == CVAggregation.MAX
            else "_min"
        )
        return {
            metric_name.replace(str_split, ""): value
            for metric_name, value in self.metrics_cv[model_name].items()
            if str_split in metric_name
        }

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
        # The base class of `fit_and_evaluate` uses MetricManagerBase but this subclass
        # requires SupervisedMetricManagerBase.
        if not isinstance(opt_metric, SupervisedMetricManagerBase):
            raise TypeError(
                f"`opt_metric` has metric of type '{type(opt_metric)}'. "
                "Expected a subclass of 'SupervisedMetricManagerBase'"
            )

        for metric in metrics_to_train:
            if not isinstance(metric, SupervisedMetricManagerBase):
                raise TypeError(
                    f"`metrics_to_train` has metric of type '{type(metric)}'. "
                    "Expected a subclass of 'SupervisedMetricManagerBase'"
                )

        supervised_metrics_to_train = cast(Sequence[SupervisedMetricManagerBase], metrics_to_train)

        self.fit_model(
            model,
            data_manager,
            models_kwargs,
            current_experiment_id,
            save_models_to_disk,
            checkpoints_path,
            empty_cache_func,
        )
        metrics = self.evaluate_model(
            model,
            data_manager,
            metrics_kwargs,
            supervised_metrics_to_train,
            current_experiment_id,
            save_models_to_disk,
            load_checkpoints_func,
            empty_cache_func,
        )
        if self.selected_model == CVSelectedModel.WORST:
            model = self.get_worst_model(
                opt_metric,
                model,
                metrics_kwargs,
                supervised_metrics_to_train,
                current_experiment_id,
                save_models_to_disk,
                load_checkpoints_func,
            )
        else:
            model = self.get_best_model(
                opt_metric,
                model,
                metrics_kwargs,
                supervised_metrics_to_train,
                current_experiment_id,
                save_models_to_disk,
                load_checkpoints_func,
            )
        return model, metrics

    def get_best_model(
        self,
        metric: SupervisedMetricManagerBase,
        model: ModelManagerBase,
        metrics_kwargs: Dict[str, Any],
        metrics_to_train: Sequence[SupervisedMetricManagerBase],
        current_experiment_id: Optional[str] = None,
        save_models_to_disk: bool = False,
        load_checkpoints_func: Optional[Callable] = None,
    ) -> ModelManagerBase:
        """Get the best model according to a given metric.

        Args:
            metric: The metric manager to use.
            model: The model to train and evaluate.
            metrics_kwargs: Additional keyword arguments for the metrics.
            metrics_to_train: List of metrics to be used during training.

        Returns:
            The model manager of the selected model.
        """
        best_model = None
        best_metric_value = (
            float("-inf") if metric.get_optimization_direction() == MinOrMax.Max else float("inf")
        )
        if current_experiment_id is None:
            raise ValueError("current_experiment_id cannot be None")
        assert load_checkpoints_func is not None
        for i, current_model in enumerate(
            self.get_models_generator(
                model.name, current_experiment_id, save_models_to_disk, load_checkpoints_func
            )
        ):
            current_metric_value = self.metrics_per_model[model.name][i][metric.name]

            if (
                metric.get_optimization_direction() == MinOrMax.Max
                and current_metric_value > best_metric_value
            ) or (
                metric.get_optimization_direction() == MinOrMax.Min
                and current_metric_value < best_metric_value
            ):
                best_metric_value = current_metric_value
                best_model = current_model
        if best_model is None:
            raise ValueError("Could not find the best model.")
        return best_model

    def get_worst_model(
        self,
        metric: SupervisedMetricManagerBase,
        model: ModelManagerBase,
        metrics_kwargs: Dict[str, Any],
        metrics_to_train: Sequence[SupervisedMetricManagerBase],
        current_experiment_id: Optional[str],
        save_models_to_disk: bool,
        load_checkpoints_func: Optional[Callable],
    ) -> ModelManagerBase:
        """Get the worst model according to a given metric.

        Args:
            metric: The metric manager to use.
            model: The model to train and evaluate.
            metrics_kwargs: Additional keyword arguments for the metrics.
            metrics_to_train: List of metrics to be used during training.

        Returns:
            The model manager of the selected model.
        """
        worst_model = None
        worst_metric_value = (
            float("inf") if metric.get_optimization_direction() == MinOrMax.Max else float("-inf")
        )
        if current_experiment_id is None:
            raise ValueError("current_experiment_id cannot be None")
        assert load_checkpoints_func is not None
        for i, current_model in enumerate(
            self.get_models_generator(
                model.name, current_experiment_id, save_models_to_disk, load_checkpoints_func
            )
        ):
            current_metric_value = self.metrics_per_model[model.name][i][metric.name]

            if (
                metric.get_optimization_direction() == MinOrMax.Max
                and current_metric_value < worst_metric_value
            ) or (
                metric.get_optimization_direction() == MinOrMax.Min
                and current_metric_value > worst_metric_value
            ):
                worst_metric_value = current_metric_value
                worst_model = current_model
        if worst_model is None:
            raise ValueError("Could not find the worst model.")
        return worst_model

    def save_model(
        self,
        model: ModelManagerBase,
        current_experiment_id: Optional[str],
        save_models_to_disk: bool,
        checkpoints_path: Optional[str] = None,
        empty_cache_func: Optional[Callable] = None,
        idx: Optional[int] = None,
    ) -> None:
        """Save the model to disk or store it in memory."""
        if current_experiment_id is None:
            raise ValueError("current_experiment_id cannot be None")

        if save_models_to_disk:
            if checkpoints_path is None or idx is None:
                raise ValueError(
                    "checkpoints_path and idx cannot be None when saving models to disk"
                )

            path_to_save = (
                Path(checkpoints_path) / f"{model.name}_fold_{str(idx)}_{current_experiment_id}.pkl"
            )
            model.export(path_to_save)

            if model.name not in self.trained_models[current_experiment_id]:
                self.trained_models[current_experiment_id][model.name] = []

            self.trained_models[current_experiment_id][model.name].append(path_to_save)

            del model
            if empty_cache_func is not None:
                empty_cache_func()
        else:
            if model.name not in self.trained_models[current_experiment_id]:
                self.trained_models[current_experiment_id][model.name] = []

            self.trained_models[current_experiment_id][model.name].append(
                model.clone()
            )  # Store model instance

    def get_models_generator(
        self,
        model_name: str,
        current_experiment_id: str,
        save_models_to_disk: bool,
        load_checkpoints_func: Callable[[str], Any],
        load_model: bool = True,
    ) -> Generator[Any, None, None]:
        """Get a generator that yields a model.

        Args:
            model_name: The name of the model.
            current_experiment_id: The current experiment ID.
            save_models_to_disk: Whether to save models to disk.
            load_checkpoints_func: A callable that given a path to a checkpoint will return the
                corresponding model.
            load_model: Whether to load the model.

        Returns:
            A generator that yields a model.
        """
        run_models = self.trained_models.get(current_experiment_id)
        if run_models is None:
            raise ValueError(f"No models found for experiment ID '{current_experiment_id}'")

        selected_model_paths = run_models.get(model_name)
        if selected_model_paths is None:
            raise ValueError(f"No model paths found for model name '{model_name}'")

        if save_models_to_disk and load_model:
            for path in selected_model_paths:
                if isinstance(path, Path) or isinstance(path, str):
                    if os.path.exists(path):
                        yield load_checkpoints_func(str(path))  # Yielding one model at a time
                    else:
                        raise FileNotFoundError(f"No saved model found at '{path}'")
                else:
                    raise TypeError(f"Unexpected type for path: {type(path)}")
        else:
            for model in selected_model_paths:
                yield model  # Yield the model

    def get_cv_log(self) -> pd.DataFrame:
        """Get the cross-validation log.

        Returns:
            The log as a data frame.
        """
        return pd.DataFrame.from_dict(self.metrics_cv, orient="index")

    def plot_cv_res(self) -> plt.Figure:
        """Plot the cross-validation results.

        Returns:
            A matplotlib Figure object containing the plots.
        """
        df = self.get_cv_log()
        metrics = [col.split("_")[0] for col in df.columns if "_mean" in col]

        fig, axs = plt.subplots(
            len(metrics) // 2 + len(metrics) % 2,
            2,
            figsize=(20, (len(metrics) // 2 + len(metrics) % 2) * 7),
        )  # Create subplots and increase the figure size

        # Flatten the axs array for easy indexing
        axs = axs.flatten()

        for i, metric in enumerate(metrics):
            df[[f"{metric}_min", f"{metric}_mean", f"{metric}_max"]].rename(
                columns={f"{metric}_min": "Min", f"{metric}_mean": "Mean", f"{metric}_max": "Max"}
            ).plot(kind="bar", ax=axs[i])
            axs[i].set_title(metric)
            axs[i].set_ylabel("Value")
            axs[i].grid(True)
            axs[i].set_ylim(0, 1)

            # Add std as text above the mean bar
            for model in df.index:
                axs[i].text(
                    df.index.get_loc(model),
                    df.loc[model, f"{metric}_mean"],
                    f"± {df.loc[model, f'{metric}_std']:.3f}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        return fig

    def box_plot_per_model_and_metric(self, model: str, metric: str) -> plt.Figure:
        """Create a box plot per the specified model and metric.

        Args:
            model: The model name.
            metric: The metric name.

        Returns:
            A matplotlib Figure object containing the plots.
        """
        # Extract the specific metric values for the model
        data = self.metrics_per_model
        metric_values = [fold[metric] for fold in data[model].values()]
        # Create figure and axis
        fig, ax = plt.subplots()

        # Create boxplot
        ax.boxplot(metric_values, vert=False)

        # Add dots for individual metric values
        y = np.ones_like(metric_values)
        ax.scatter(metric_values, y, color="red", alpha=0.5)

        # Add title and labels
        ax.set_title(f"Distribution of {metric} for {model} across folds")
        ax.set_xlabel(metric)
        ax.set_yticks([])

        return fig

    def display_best_models(self, metric_name: str = "Accuracy") -> plt.Figure:
        """Display the comparison of the models across different folds.

        Args:
            metric_name: The name of the metric to display (default is 'Accuracy').

        Returns:
            A matplotlib Figure object containing the plots.
        """
        # Getting the model names
        metrics = self.metrics_per_model
        model_names = list(metrics.keys())
        # Number of folds
        num_folds = len(next(iter(metrics.values())))
        # Calculate the number of rows based on two folds per row
        num_rows = (num_folds + 1) // 2

        # Create a figure with separate subplots for each fold
        fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=[16, 5 * num_rows])
        axes = axes.flatten()
        fig.suptitle("Model Comparison Across Folds")

        # Iterate through the folds
        for fold in range(num_folds):
            ax = axes[fold]
            # Values and model names for the current fold
            values = [
                metrics[model][fold][metric_name]
                for model in model_names  # Split the line to avoid E501
            ]
            best_model_idx = values.index(max(values))

            # Plotting each model's metric for the current fold
            for idx, (model_name, value) in enumerate(zip(model_names, values)):
                color = sns.color_palette("deep")[0] if idx == best_model_idx else "gray"
                ax.bar(model_name, value, color=color, width=0.4)

            # Adding labels and title
            ax.set_ylabel(metric_name)
            ax.set_title(f"Fold {fold}")
            ax.set_ylim(0, 1)
            ax.set_xticklabels(
                model_names, rotation=45, ha="right", style="italic"
            )  # Split the line to avoid E501

        # Remove any unused subplots if there's an odd number of folds
        if num_folds % 2:
            axes[-1].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        return fig

    def display_model_cross_metrics(self, model_name: str) -> plt.Figure:
        """Display the comparison of the metrics across different folds for a specific model.

        Args:
            model_name: The name of the model to display.

        Returns:
            A matplotlib Figure object containing the plots.
        """
        # Get the metrics for the specified model
        model_metrics = self.metrics_per_model[model_name]
        # Get the number of folds
        num_folds = len(model_metrics)
        # List of metrics to display
        metrics_to_display = list(model_metrics[0].keys())
        # Create subplots for each fold
        fig, axes = plt.subplots(nrows=num_folds, figsize=[8, 4 * num_folds])
        # Set the figure title
        fig.suptitle(f"{model_name} Metrics Across Folds")
        # Iterate through the folds
        for fold in range(num_folds):
            ax = axes[fold]
            # Get the metrics for the current fold
            fold_metrics = model_metrics[fold]
            # Get the metric values and metric names
            metric_values = [fold_metrics[metric_name] for metric_name in metrics_to_display]
            metric_names = metrics_to_display
            # Set the x-axis positions for bars
            x_pos = range(len(metric_names))
            # Plot the metric values as bars
            ax.bar(x_pos, metric_values, color=sns.color_palette("deep"), width=0.4)
            # Adding labels and title
            ax.set_ylabel('Value')
            ax.set_title(f"Fold {fold}")
            ax.set_ylim(0, 1)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metric_names, rotation=45, ha="right", style="italic")
            # Add metric values on top of bars with 2 digits after the dot
            for i, v in enumerate(metric_values):
                ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()
        return fig
