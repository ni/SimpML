"""Loggers pool."""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, Optional, Sequence

import dill
import mlflow
import mlflow.pyfunc

from simpml.core.base import (
    DataManagerBase,
    LoggerBase,
    MetricManagerBase,
    MetricName,
    ModelManagerBase
)
from simpml.utils.MlFlow import MlFlowRunHandler


class ModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper class for log model to mlflow."""

    def __init__(self, model: ModelManagerBase):
        """Initialize the ModelWrapper.

        Args:
            model: The model to be wrapped.
        """
        self.model = model

    def predict(self, context: Any, model_input: Any) -> Any:
        """Predicts the output using the given model input.

        Args:
            context: The context for the prediction.
            model_input: The input data for the prediction.

        Returns:
            The predicted output.
        """
        return self.model.predict(model_input)


class MlflowLogger(LoggerBase):
    """MlflowLogger class for logging."""

    def __init__(
        self,
        tracking_uri: str = "http://0.0.0.0:5000",
        artifact_location: str = "./artficats",
        experiment_name: str = "default",
        summary_suffix: str = "_summary",
        log_all_models: bool = False,
        register_best_model: bool = False,
    ):
        """Initialize the MlflowLogger.

        Args:
            tracking_uri (str): The URI of the MLflow tracking server.
            artifact_location (str): The location where artifacts will be stored.
            experiment_name (str): The name of the experiment.
            summary_suffix (str): The suffix to be added to summary metrics.
            log_all_models (bool): Flag indicating whether to log all models.
            register_best_model (bool): Flag indicating whether to register the best model.
        """
        instance_id = os.getenv("MLFLOW_EC2_INSTANCE_ID")
        if instance_id:
            import boto3
            ec2 = boto3.client("ec2")
            response = ec2.describe_instances(InstanceIds=[instance_id])
            state = response["Reservations"][0]["Instances"][0]["State"]["Name"]

            if state != "running":
                ec2.start_instances(InstanceIds=[instance_id])
                ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])

        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name, artifact_location)
        else:
            self.experiment_id = experiment.experiment_id

        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.summaty_suffix = summary_suffix
        self.log_all_models = log_all_models
        self.register_best_model = register_best_model
        self.run_context = None

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
        """Start the logging of a run.

        Args:
            run_name (str): The name of the run.
            model: The model being logged.
            metrics: The metrics being logged.
            data_manager: The data manager being used.
            models_kwargs: The keyword arguments for the models.
            metrics_kwargs: The keyword arguments for the metrics.
            opt_metric: The optimization metric.
            experiment_description: The description of the experiment.
        """
        self.run_context = mlflow.start_run(
            run_name=run_name, experiment_id=self.experiment_id, description=experiment_description
        )

    def __log_run_results(
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
        """Log the results of a model run to MLflow.

        Args:
        - run_name (str): The name of the model run.
        - model: The trained model object.
        - metrics: The metrics obtained from the model run.
        - data_manager: The data manager object used for the model run.
        - models_kwargs: The keyword arguments used for the model.
        - metrics_kwargs: The keyword arguments used for the metrics.
        - opt_metric: The optimal metric for the model run.
        - experiment_description: The description of the experiment.

        Returns:
        None
        """
        mlflow.log_params(models_kwargs)
        mlflow.log_params(metrics_kwargs)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_name", model.name)
        mlflow.set_tag("model_description", model.desc)
        mlflow.set_tag("data_manager_version", data_manager.id)
        mlflow.set_tag("data_manager_description", data_manager.description)

        if isinstance(opt_metric, MetricName):
            mlflow.set_tag("opt_metric", opt_metric.name)
        elif opt_metric is not None:
            mlflow.set_tag("opt_metric", str(opt_metric))
        else:
            mlflow.set_tag("opt_metric", "")

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
        """Logs the results of a run.

        Args:
            run_name (str): The name of the run.
            model: The trained model.
            metrics: The calculated metrics.
            data_manager: The data manager object.
            models_kwargs: The keyword arguments used for model training.
            metrics_kwargs: The keyword arguments used for metric calculation.
            opt_metric: The optimized metric.
            experiment_description: The description of the experiment.

        Returns:
            None
        """
        if self.run_context is not None:
            with self.run_context:
                self.__log_run_results(
                    run_name,
                    model,
                    metrics,
                    data_manager,
                    models_kwargs,
                    metrics_kwargs,
                    opt_metric,
                    experiment_description,
                )

                if self.log_all_models:
                    wrapped_model = ModelWrapper(model)
                    mlflow.pyfunc.log_model("model", python_model=wrapped_model)

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
        """Ends the MLflow run and logs the final metrics and model information.

        Args:
            run_name (str): The name of the MLflow run.
            model: The trained model object.
            metrics: The dictionary of metrics.
            data_manager: The data manager object.
            models_kwargs: The keyword arguments used for model training.
            metrics_kwargs: The keyword arguments used for metric calculation.
            opt_metric: The name of the optimal metric.
            experiment_description: The description of the experiment.

        Returns:
            None
        """
        if self.run_context is not None:
            mlflow.end_run()

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
        """Logs the summary results of a model run.

        Args:
            run_name (str): The name of the model run.
            model: The trained model object.
            metrics: The metrics calculated for the model.
            data_manager: The data manager object used for the model run.
            models_kwargs: Additional keyword arguments for the model.
            metrics_kwargs: Additional keyword arguments for the metrics.
            opt_metric: The optimized metric for the model.
            experiment_description: The description of the experiment.

        Returns:
            None
        """
        with mlflow.start_run(
            run_name=run_name + self.summaty_suffix,
            experiment_id=self.experiment_id,
            description=experiment_description,
        ):
            self.__log_run_results(
                run_name,
                model.model,
                metrics,
                data_manager,
                models_kwargs,
                metrics_kwargs,
                opt_metric,
                experiment_description,
            )

            wrapped_model = ModelWrapper(model)
            if self.register_best_model:
                mlflow.pyfunc.log_model(
                    "model",
                    python_model=wrapped_model,
                    registered_model_name=self.experiment_name
                    + "_"
                    + run_name
                    + "_"
                    + model.model.name,
                )
            else:
                mlflow.pyfunc.log_model("model", python_model=wrapped_model)

            temp_file_name = str(uuid.uuid4()) + ".pkl"
            try:
                with open(temp_file_name, "wb") as temp_file:
                    dill.dump(data_manager, temp_file)
                    temp_file.flush()
                mlflow.log_artifact(temp_file_name, "data manager")
            finally:
                os.remove(temp_file_name)

    def get_run_handler(
        self, run_name: Optional[str] = None,
        tags: Optional[dict[str, str]] = None
    ) -> MlFlowRunHandler:
        """Returns an instance of MlFlowRunHandler for logging runs in MLflow.

        Args:
            run_name (str, optional): Name of the run.
            tags (dict, optional): Tags to associate with the run.

        Returns:
            MlFlowRunHandler: An instance of MlFlowRunHandler for logging runs in MLflow.
        """
        if run_name is None:
            if self.run_context is not None:
                run_name = self.run_context.info.run_name + self.summaty_suffix
            else:
                run_name = self.summaty_suffix
        return MlFlowRunHandler(self.experiment_name, run_name, self.tracking_uri, tags)
