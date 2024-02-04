"""Mlflow utils."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import dill
import mlflow

from ..core.experiment_manager import ExperimentManager


class MlFlowRunHandler:
    """Class for handling MLflow runs."""

    def __init__(self,
                 experiment_name: str,
                 run_name: str,
                 tracking_uri: str = "http://0.0.0.0:5000",
                 tags: Optional[Dict[str , str]] = None):
        """Initialize the MlFlowRunHandler.

        Args:
        experiment_name (str): The name of the experiment.
        run_name (str): The name of the run.
        tracking_uri (str): The URI of the MLflow tracking server.
        tags (dict): Optional tags to associate with the run.
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.client = mlflow.tracking.MlflowClient(tracking_uri)
        self.tags = tags
        self.run_id = self.__find_run_id()

    def __find_run_id(self) -> str:
        """Finds the run ID for a specific run in the MLflow experiment.

        Returns:
            str: The run ID of the specified run.

        Raises:
            ValueError: If the experiment or run is not found.
        """
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{self.experiment_name}' not found")

        filter_string = f"tags.mlflow.runName = '{self.run_name}'"
        if self.tags:
            for key, value in self.tags.items():
                filter_string += f" and tags.{key} = '{value}'"

        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id], filter_string=filter_string
        )

        if not runs:
            raise ValueError(
                f"Run '{self.run_name}' not found in experiment '{self.experiment_name}'"
            )

        if len(runs) > 1:
            raise ValueError("Multiple runs found. Try adding tags to get to a unique run.")

        return runs[0].info.run_id

    def log_metric(self, key: str, value: float) -> None:
        """Logs a metric with the specified key and value to the active MLflow run.

        Args:
        - key (str): The name of the metric.
        - value (float): The value of the metric.

        Returns:
        None
        """
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_metric(key, value)

    def log_figure(self, fig: Any, name: str) -> None:
        """Logs a figure to the active MLflow run.

        Args:
        - fig: The figure object to be logged.
        - name: The name of the figure.

        Returns:
        None
        """
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_figure(fig, name)

    def log_image(self, image: Any, name: str) -> None:
        """Logs an image artifact to the active MLflow run.

        Args:
        - image: The image artifact to be logged.
        - name: The name of the image.

        Returns:
        None
        """
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_artifact(image, name)

    def log_tag(self, key: str, value: str) -> None:
        """Logs a tag with the specified key and value to the active MLflow run.

        Args:
            key (str): The key of the tag.
            value: The value of the tag.

        Returns:
            None
        """
        with mlflow.start_run(run_id=self.run_id):
            mlflow.set_tag(key, value)

    def log_params(self, params: Dict[str , Any]) -> None:
        """Logs the given parameters to the active MLflow run.

        Args:
            params (dict): A dictionary containing the parameter names and values.
        """
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_params(params)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Logs an artifact to the active MLflow run.

        Args:
            local_path (str): The local path of the artifact to be logged.
            artifact_path (str, optional): The desired location in the artifact repository.

        Returns:
            None
        """
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_artifact(local_path, artifact_path)

    def register_model(self,
                       registered_model_name: str,
                       model_artifact_path: Optional[str] = "model") -> None:
        """Registers a model with the specified name and artifact path to MLflow.

        Args:
        - registered_model_name (str): The name of the registered model.
        - model_artifact_path (str, optional): The artifact path of the model. Default is "model".

        Returns:
        None
        """
        model_uri = f"runs:/{self.run_id}/{model_artifact_path}"
        mlflow.register_model(model_uri, registered_model_name)

    def load_model_pipeline(self, model_artifact_path: Optional[str] = "model") -> Any:
        """Load the model pipeline from the specified artifact path.

        Args:
            model_artifact_path (str): The path to the model artifact within the MLflow run.

        Returns:
            model: The loaded model pipeline.
        """
        model_uri = f"runs:/{self.run_id}/{model_artifact_path}"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        return loaded_model.unwrap_python_model().model

    def load_model(self, model_artifact_path: Optional[str] = "model") -> Any:
        """Loads the trained model from the specified artifact path.

        Args:
            model_artifact_path (str): The path to the model artifact. Defaults to "model".

        Returns:
            The loaded trained model.
        """
        return self.load_model_pipeline(model_artifact_path).model

    def load_data_manager(self,
                          artifact_path: Optional[str] = "data manager",
                          local_temp_dir: Optional[str] = None) -> Any:
        """Load the data manager artifact from the given artifact path.

        Args:
        artifact_path (str): The artifact path where the data manager is saved.
        local_temp_dir (str): Temporary local directory to store downloaded artifacts.

        Returns:
        The loaded data manager object.
        """
        if local_temp_dir is None:
            local_temp_dir = os.getcwd()

        if not os.path.exists(local_temp_dir):
            os.makedirs(local_temp_dir)

        local_artifact_path = self.client.download_artifacts(
            self.run_id, artifact_path, local_temp_dir
        )

        files = os.listdir(local_artifact_path)
        if files:
            file_path = os.path.join(local_artifact_path, files[0])
            with open(file_path, "rb") as file:
                data_manager = dill.load(file)
            os.remove(file_path)

        return data_manager

    def load_experiment_manager(self) -> ExperimentManager:
        """Loads the ExperimentManager object with the data manager and optimization metric.

        Returns:
            ExperimentManager: The ExperimentManager object.
        """
        data_manager = self.load_data_manager()
        opt_metric = self.client.get_run(self.run_id).data.tags["opt_metric"]

        return ExperimentManager(data_manager, opt_metric)

    def load_artifact(self,
                      artifact_path: str,
                      local_temp_dir: Optional[str] = None) -> str:
        """Download and load a general artifact from the given artifact path.

        Args:
        artifact_path (str): The artifact path to download from.
        local_temp_dir (str): Local directory to store downloaded artifacts.

        Returns:
        The local path to the downloaded artifact.
        """
        if local_temp_dir is None:
            local_temp_dir = os.getcwd()

        if not os.path.exists(local_temp_dir):
            os.makedirs(local_temp_dir)

        local_artifact_path = self.client.download_artifacts(
            self.run_id, artifact_path, local_temp_dir
        )
        return local_artifact_path
