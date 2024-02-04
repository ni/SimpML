"""Models pool."""

from __future__ import annotations

from typing import Dict, List, Union

from fastai.vision.all import resnet18, resnet34, resnet50
from lightgbm import LGBMClassifier, LGBMRegressor
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.iforest import IForest
from pyod.models.loda import LODA
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LassoLarsCV, LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from simpml.core.base import DataType, ModelManagerBase, PredictionType
from simpml.tabular.model import BaselineClassification, TabularModelManager
from simpml.vision.model import (
    ClassicalImageClassifier,
    FastaiModelClassificationManager,
    VisionBaselineClassification,
)

seed = 42
n_jobs = -1

MODELS_POOL: Dict[str, Dict[str, List[ModelManagerBase]]] = {}

MODELS_POOL[DataType.Tabular.value] = {
    PredictionType.Regression.value: [
        TabularModelManager(
            model=ElasticNet(random_state=seed),
            name="Elastic Net",
            desc="Default settings",
        ),
        TabularModelManager(
            model=LassoLarsCV(),
            name="Lasso Lars CV",
            desc="Default settings",
        ),
        TabularModelManager(
            model=DecisionTreeRegressor(random_state=seed),
            name="Decision Tree",
            desc="Default settings",
        ),
        TabularModelManager(
            model=RandomForestRegressor(random_state=seed, n_jobs=n_jobs),
            name="Random Forest",
            desc="Default settings",
        ),
        TabularModelManager(
            model=GradientBoostingRegressor(random_state=seed),
            name="Gradient Boosting",
            desc="Default settings",
        ),
        TabularModelManager(
            model=XGBRegressor(random_state=seed, verbosity=0, n_jobs=n_jobs),
            name="XGBoost",
            desc="Default settings",
        ),
        TabularModelManager(
            model=LGBMRegressor(random_state=seed, n_jobs=n_jobs, verbose=-1),
            name="LightGBM",
            desc="Default settings",
        ),
    ],
    PredictionType.BinaryClassification.value: [
        TabularModelManager(
            model=BaselineClassification(),
            name="Baseline Classification",
            desc="Default settings",
        ),
        TabularModelManager(
            model=LogisticRegression(random_state=seed, n_jobs=n_jobs),
            name="Logistic Regression",
            desc="Default settings",
        ),
        TabularModelManager(
            model=SVC(random_state=seed, probability=True),
            name="Support Vector Classifier",
            desc="Default settings",
        ),
        TabularModelManager(
            model=AdaBoostClassifier(random_state=seed),
            name="AdaBoost Classifier",
            desc="Default settings",
        ),
        TabularModelManager(
            model=DecisionTreeClassifier(random_state=seed),
            name="Decision Tree",
            desc="Default settings",
        ),
        TabularModelManager(
            model=RandomForestClassifier(random_state=seed, n_jobs=n_jobs),
            name="Random Forest",
            desc="Default settings",
        ),
        TabularModelManager(
            model=GradientBoostingClassifier(random_state=seed),
            name="Gradient Boosting",
            desc="Default settings",
        ),
        TabularModelManager(
            model=XGBClassifier(random_state=seed, verbosity=0, n_jobs=n_jobs),
            name="XGBoost",
            desc="Default settings",
        ),
        TabularModelManager(
            model=LGBMClassifier(random_state=seed, n_jobs=n_jobs, verbose=-1),
            name="LightGBM",
            desc="Default settings",
        ),
    ],
    PredictionType.AnomalyDetection.value: [
        TabularModelManager(
            model=FeatureBagging(),
            name="Feature Bagging",
            desc="Default settings",
        ),
        TabularModelManager(
            model=LODA(),
            name="LODA",
            desc="Default settings",
        ),
        TabularModelManager(
            model=IForest(),
            name="Isolation Forest",
            desc="Default settings",
        ),
    ],
    PredictionType.Clustering.value: [
        TabularModelManager(
            model=KMeans(random_state=seed),
            name="K-Means",
            desc="Default settings",
        ),
        TabularModelManager(
            model=GaussianMixture(n_components=2),
            name="Gaussian Mixture",
            desc="Default settings",
        ),
    ],
}

MODELS_POOL[DataType.Tabular.value][PredictionType.MulticlassClassification.value] = MODELS_POOL[
    DataType.Tabular.value
][PredictionType.BinaryClassification.value]


MODELS_POOL[DataType.Vision.value] = {
    PredictionType.Regression.value: [],
    PredictionType.BinaryClassification.value: [
        VisionBaselineClassification(
            name="Random Baseline Classification",
            desc="Default settings",
        ),
        ClassicalImageClassifier(
            name="Naive Baseline Classification",
            desc="Default settings",
        ),
        FastaiModelClassificationManager(
            arch=resnet50, name="Resnet-50", desc="Resnet-50 Pretrained Model ImageNet"
        ),
        FastaiModelClassificationManager(
            arch=resnet34, name="Resnet-34", desc="Resnet-34 Pretrained Model ImageNet"
        ),
        FastaiModelClassificationManager(
            arch=resnet18, name="Resnet-18", desc="Resnet-18 Pretrained Model ImageNet"
        ),
    ],
}

MODELS_POOL[DataType.Vision.value][PredictionType.MulticlassClassification.value] = MODELS_POOL[
    DataType.Vision.value
][PredictionType.BinaryClassification.value]


def register_models_to_pool(
    data_type: Union[str, DataType],
    prediction_type: Union[str, PredictionType],
    models: List[ModelManagerBase],
) -> None:
    """Register a list of models to the MODELS_POOL under the specified data and prediction types.

    :param data_type: The type of data (e.g., 'Tabular', 'Text', 'Vision').
    :param prediction_type: The type of prediction task (e.g., 'Regression', 'Classification').
    :param models: A list of instances of ModelManagerBase or its subclasses.
    """
    data_type = data_type if isinstance(data_type, str) else data_type.value
    prediction_type = prediction_type if isinstance(prediction_type, str) else prediction_type.value

    if data_type not in MODELS_POOL:
        MODELS_POOL[data_type] = {}

    if prediction_type not in MODELS_POOL[data_type]:
        MODELS_POOL[data_type][prediction_type] = []

    current_models = MODELS_POOL[data_type][prediction_type]

    for model in models:
        if model.name not in [model.name for model in current_models]:
            current_models.append(model)
