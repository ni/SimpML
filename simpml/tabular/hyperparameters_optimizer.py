"""Hyper-parameters optimizer."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import optuna
import pandas as pd
import sklearn.ensemble
from optuna.trial import Trial
from sklearn.metrics import make_scorer

from simpml.core.base import (
    DataManagerBase,
    HyperParamsOptimizationLevel,
    MetricManagerBase,
    MinOrMax,
    ModelManagerBase,
    OptimizerBase,
)

warnings.filterwarnings("ignore", message="All message displayed in console.")

optuna.logging.set_verbosity(optuna.logging.WARNING)


class SupervisedTabularOptimizer(OptimizerBase):
    """The Class initializes a search space of hyperparameters depending on the type of model and
    finds optimizaed set of hyperparameters.

    Input:
        iters: Number of iterations for each model optimization.
        cv: Number of cross validation folds.

    Attributes:
        optimized(model): Finds optimized set of hyper-parameters for a given model and updates the
            model wrapper accordingly.
        get_params_dict(): Prints and returns the dict of hyper parameters.
        set_params_dict(): Set the class hyper parameters dict according to given input.
        restore_params_dict(): Restores the hyper parameters dict to the default settings.
    """

    def __init__(
        self,
        iters: int = 30,
        cv: int = 2,
        optimization_level: HyperParamsOptimizationLevel = HyperParamsOptimizationLevel.Default,
    ) -> None:
        """Initializes the SupervisedTabularOptimizer class.

        Args:
            iters: The number of iterations (trials).
            cv: The number of folds in a (Stratified)KFold.
            optimization_level: The optimizations level as a `HyperParamsOptimizationLevel` Enum,
                its int value or its string name.
        """
        self.params_dict: Dict[str, Dict[str, Any]] = self.init_dict()
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.X_valid: Optional[pd.DataFrame] = None
        self.y_valid: Optional[pd.Series] = None
        self.metric: Optional[MetricManagerBase] = None
        self.iters: int = iters
        self.cv: int = cv
        self.optimization_level: HyperParamsOptimizationLevel = optimization_level

    def optimize(self, model: ModelManagerBase, verbose: bool = False) -> ModelManagerBase:
        """Perform the optimization.

        Args:
            model: A model manager.
            verbose: Whether to output detailed information.

        Returns:
            The input model manager instance.
        """
        stored_model = model.get_model()
        if type(stored_model).__name__ != "BaselineClassification":
            if self.optimization_level != HyperParamsOptimizationLevel.Default:
                if type(stored_model).__name__ not in self.params_dict.keys():
                    warnings.warn(
                        "Hyper parameters search space doesnt exist for model "
                        f"{type(stored_model).__name__}, this model was not optimized",
                        stacklevel=2,
                    )
                elif (
                    self.optimization_level.name
                    not in self.params_dict[type(stored_model).__name__]
                ):
                    warnings.warn(
                        "Hyper parameters optimization level "
                        f"{HyperParamsOptimizationLevel(self.optimization_level.name).name} "
                        f"doesnt exist for model {type(stored_model).__name__}, "
                        "this model was not optimized",
                        stacklevel=2,
                    )
                else:
                    best_params = self.get_best_params(model, verbose)
                    model_class = stored_model.__class__
                    model.set_model(model_class(**best_params))
                    best_params_str = ""
                    for key in best_params:
                        best_param = best_params[key]
                        if isinstance(best_param, float):
                            best_param = f"{best_param:.3f}"
                        else:
                            best_param = str(best_param)
                        best_params_str = best_params_str + ", " + key + "=" + best_param
                    model.set_desc("Optimized " + best_params_str)
        return model

    def get_optimizer_models(self) -> List[str]:
        """Get optimizer model names.

        Returns:
            A list of optizer model names.
        """
        return list(self.params_dict.keys())

    def get_model_params_df(self, model_name: str) -> pd.DataFrame:
        """Get a data frame with the model parameters used based on model name.

        Args:
            model_name: The model name to get the parameters of.

        Returns:
            A data frame with the model parameters used.
        """
        df = self.get_params_df()
        model_df = df[df["model"] == model_name].copy()
        return model_df

    def get_params_df(self) -> pd.DataFrame:
        """Get a data frame with the model parameters used.

        Returns:
            A data frame with the model parameters used.
        """
        params = self.get_params_dict()
        flattened_data = self.flatten_dict(params)
        columns = [
            "model",
            "optimization_level",
            "hyperparameter_type",
            "hyperparameter_name",
            "param",
            "value",
        ]
        df = pd.DataFrame(columns=columns)
        for key in flattened_data.keys():
            row = key.split("__")
            row.append(flattened_data[key])
            df.loc[len(df)] = row
        return df

    def set_params(self, df: pd.DataFrame, model_name: Optional[str] = None) -> None:
        """Set parameters for a specific model name.

        Args:
            df: A data frame with the parameters.
            model_name: The model name to set the parameters for.
        """
        flattened_data = {}
        for index, _ in df.iterrows():
            concatenated_string = (
                f'{df.loc[index, "model"]}__'
                f'{df.loc[index, "optimization_level"]}__'
                f'{df.loc[index, "hyperparameter_type"]}__'
                f'{df.loc[index, "hyperparameter_name"]}__'
                f'{df.loc[index, "param"]}'
            )
            flattened_data[concatenated_string] = df.loc[index, "value"]
        params = self.unflatten_dict(flattened_data)
        if model_name is None:
            self.params_dict = params
        else:
            del self.params_dict[model_name]
            self.params_dict[model_name] = params[model_name]

    def restore_params(self) -> None:
        """Restore initial parameters."""
        self.params_dict = self.init_dict()

    def set_data_metric(
        self, data_manager: DataManagerBase, optimize_metric: MetricManagerBase
    ) -> None:
        """Set the data metric.

        Args:
            data_manager: any type of data manager object.
            optimize_metric: metric wrapper of the optimized metric.
        """
        self.X_train, self.y_train = data_manager.get_training_data()
        self.X_valid, self.y_valid = data_manager.get_validation_data()
        self.metric = optimize_metric

    # --------------#

    # private methods

    # --------------#

    def flatten_dict(
        self, dictionary: Dict[str, Any], parent_key: str = "", sep: str = "__"
    ) -> Dict[str, Any]:
        """Flatten a dictionary.

        Args:
            dictionary: The dictionary to flatten.
            parent_key: The parent key.
            sep: The separater to use for multi-level key names.

        Returns:
            The flattened dictionary.
        """
        flattened_dict: Dict[str, Any] = {}
        for key, value in dictionary.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                flattened_dict.update(self.flatten_dict(value, new_key, sep=sep))
            else:
                flattened_dict[new_key] = value
        return flattened_dict

    def unflatten_dict(self, flattened_dict: Dict[str, Any], sep: str = "__") -> Dict[str, Any]:
        """Unflatten a dictionary.

        Args:
            flattened_dict: The flattened dictionary to unflatten.
            sep: The separater to use for multi-level key names.

        Returns:
            The unflattened dictionary.
        """
        unflattened_dict: Dict[str, Any] = {}
        for key, value in flattened_dict.items():
            keys = key.split(sep)
            current_dict = unflattened_dict
            for sub_key in keys[:-1]:
                if sub_key not in current_dict:
                    current_dict[sub_key] = {}
                current_dict = current_dict[sub_key]
            current_dict[keys[-1]] = value
        return unflattened_dict

    def get_params_dict(self, print_params: bool = False) -> Dict[str, Dict[str, Any]]:
        """Get a dictionary of parameters.

        Args:
            print_params: Whether to print the parameters.

        Returns:
            A dictionary of parameters.

        """
        if print_params:
            for model in self.params_dict:
                print(f"Showing hyperparameters for model {model}:")
                for depth in self.params_dict[model]:
                    print(f"\t optimization_level value {depth}:")
                    for category in self.params_dict[model][depth]:
                        print(f"\t\t for {category} type of category:")
                        for param in self.params_dict[model][depth][category]:
                            print(f"\t\t\t{param} with settings:")
                            for setting in self.params_dict[model][depth][category][param]:
                                print(
                                    f"\t\t\t\t {setting} = "
                                    f"{self.params_dict[model][depth][category][param][setting]}"
                                )
        return self.params_dict

    def get_best_params(self, model_wrapper: ModelManagerBase, verbose: bool) -> Dict[str, Any]:
        """Get a best parameters.

        Args:
            model_wrapper: A model manager object.
            verbose: Whether to output detailed information.

        Returns:
            The best parameters.
        """

        def objective(trial: Trial) -> float:
            """Objective function.

            Args:
                trial: The current Optuna trial (set of parameters).

            Returns:
                The metric for this set of parameters.
            """
            assert self.metric is not None

            params_float: Dict[str, Any] = {}
            params_int: Dict[str, Any] = {}
            params_categorical: Dict[str, Any] = {}
            params_uniform: Dict[str, Any] = {}
            if "float" in self.params_model:
                params_float = self.params_model["float"]
            if "int" in self.params_model:
                params_int = self.params_model["int"]
            if "categorical" in self.params_model:
                params_categorical = self.params_model["categorical"]
            if "uniform" in self.params_model:
                params_uniform = self.params_model["uniform"]
            to_optimize_dict: Dict[str, Any] = {}
            for key in params_float:
                to_optimize_dict[key] = trial.suggest_float(**params_float[key])
            for key in params_int:
                to_optimize_dict[key] = trial.suggest_int(**params_int[key])
            for key in params_categorical:
                to_optimize_dict[key] = trial.suggest_categorical(**params_categorical[key])
            for key in params_uniform:
                to_optimize_dict[key] = trial.suggest_uniform(**params_uniform[key])
            model_class = self.model.__class__
            model_instance = model_class(**to_optimize_dict)
            if self.cv == 1:
                model_instance.fit(self.X_train, self.y_train)
                retval = self.metric.metric(self.y_valid, model_instance.predict(self.X_valid))
            else:
                scorer = make_scorer(self.metric.metric)
                retval = sklearn.model_selection.cross_val_score(
                    model_instance,
                    self.X_train,
                    self.y_train,
                    n_jobs=-1,
                    cv=self.cv,
                    scoring=scorer,
                ).mean()
            del model_instance
            return retval

        if self.optimization_level == HyperParamsOptimizationLevel.Default:
            return model_wrapper.model.get_params()
        self.optimizer_configuration(model_wrapper)
        assert self.metric is not None
        direction = (
            "maximize" if self.metric.get_optimization_direction() == MinOrMax.Max else "minimize"
        )
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=self.iters, show_progress_bar=verbose)
        return study.best_params

    def init_dict(self) -> Dict[str, Dict[str, Any]]:
        """Initialize the parameters dictionary.

        Returns:
            The initialized parameters dictionary.
        """
        params: Dict[str, Dict[str, Any]] = {
            "LGBMClassifier": {},
            "XGBClassifier": {},
            "GradientBoostingClassifier": {},
            "RandomForestClassifier": {},
            "DecisionTreeClassifier": {},
            "AdaBoostClassifier": {},
            "SVC": {},
            "LogisticRegression": {},
            "LGBMRegressor": {},
            "XGBRegressor": {},
            "LightGBM": {},
            "XGBoost": {},
            "GradientBoostingRegressor": {},
            "RandomForestRegressor": {},
            "ElasticNet": {},
            "LassoLarsCV": {},
            "DecisionTreeRegressor": {},
        }

        params["LGBMClassifier"] = {
            HyperParamsOptimizationLevel.Fast.name: {
                "int": {"max_depth": {"name": "max_depth", "low": 5, "high": 7}},
                "categorical": {
                    "class_weight": {"name": "class_weight", "choices": [None, "balanced"]},
                    "n_estimators": {"name": "n_estimators", "choices": [100, 150, 200, 250]},
                    "reg_alpha": {"name": "reg_alpha", "choices": [0.001, 0.01, 0.1, 1, 10, 100]},
                    "reg_lambda": {"name": "reg_lambda", "choices": [0.001, 0.01, 0.1, 1, 10, 100]},
                    "learning_rate": {"name": "learning_rate", "choices": [0.01, 0.001, 0.0001]},
                },
            },
            HyperParamsOptimizationLevel.Slow.name: {
                "int": {"max_depth": {"name": "max_depth", "low": 2, "high": 12}},
                "categorical": {
                    "class_weight": {"name": "class_weight", "choices": [None, "balanced"]},
                    "n_estimators": {
                        "name": "n_estimators",
                        "choices": [100, 150, 200, 250, 300, 350, 400, 450],
                    },
                    "reg_alpha": {"name": "reg_alpha", "choices": [0.001, 0.01, 0.1, 1, 10, 100]},
                    "reg_lambda": {"name": "reg_lambda", "choices": [0.001, 0.01, 0.1, 1, 10, 100]},
                    "learning_rate": {
                        "name": "learning_rate",
                        "choices": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
                    },
                },
            },
        }

        params["XGBClassifier"] = {
            HyperParamsOptimizationLevel.Fast.name: {
                "int": {"max_depth": {"name": "max_depth", "low": 5, "high": 7}},
                "categorical": {
                    "class_weight": {"name": "class_weight", "choices": [None, "balanced"]},
                    "n_estimators": {"name": "n_estimators", "choices": [100, 150, 200, 250]},
                    "reg_alpha": {"name": "reg_alpha", "choices": [0.001, 0.01, 0.1, 1, 10, 100]},
                    "reg_lambda": {"name": "reg_lambda", "choices": [0.001, 0.01, 0.1, 1, 10, 100]},
                    "learning_rate": {"name": "learning_rate", "choices": [0.01, 0.001, 0.0001]},
                },
            },
            HyperParamsOptimizationLevel.Slow.name: {
                "int": {
                    "max_depth": {"name": "max_depth", "low": 2, "high": 12},
                    "min_child_weight": {"name": "min_child_weight", "low": 1, "high": 8},
                },
                "categorical": {
                    "class_weight": {"name": "class_weight", "choices": [None, "balanced"]},
                    "n_estimators": {
                        "name": "n_estimators",
                        "choices": [100, 150, 200, 250, 300, 350, 400, 450, 700, 1000, 1500],
                    },
                    "reg_alpha": {"name": "reg_alpha", "choices": [0.001, 0.01, 0.1, 1, 10, 100]},
                    "reg_lambda": {"name": "reg_lambda", "choices": [0.001, 0.01, 0.1, 1, 10, 100]},
                    "learning_rate": {
                        "name": "learning_rate",
                        "choices": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
                    },
                    "subsample": {"name": "subsample", "choices": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
                    "gamma": {"name": "gamma", "choices": [0, 0.01, 0.1, 1, 10]},
                    "colsample_bytree": {
                        "name": "colsample_bytree",
                        "choices": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    },
                    "eval_metric": {
                        "name": "eval_metric",
                        "choices": ["error", "logloss", "auc", "aucpr", "mlogloss"],
                    },
                },
            },
        }

        params["GradientBoostingClassifier"] = {
            HyperParamsOptimizationLevel.Fast.name: {
                "float": {
                    "learning_rate": {
                        "name": "learning_rate",
                        "low": 0.2,
                        "high": 0.35,
                        "step": 0.005,
                    }
                },
                "int": {"max_depth": {"name": "max_depth", "low": 5, "high": 7}},
                "categorical": {
                    "n_estimators": {"name": "n_estimators", "choices": [100, 150, 200, 250]}
                },
            },
            HyperParamsOptimizationLevel.Slow.name: {
                "float": {
                    "learning_rate": {
                        "name": "learning_rate",
                        "low": 0.05,
                        "high": 0.35,
                        "step": 0.005,
                    }
                },
                "int": {"max_depth": {"name": "max_depth", "low": 3, "high": 8}},
                "categorical": {
                    "n_estimators": {
                        "name": "n_estimators",
                        "choices": [100, 150, 200, 250, 300, 350, 400, 450],
                    }
                },
            },
        }

        params["RandomForestClassifier"] = {
            HyperParamsOptimizationLevel.Fast.name: {
                "float": {
                    "max_features": {"name": "max_features", "low": 0.8, "high": 1.0, "step": 0.005}
                },
                "categorical": {
                    "n_estimators": {"name": "n_estimators", "choices": [100, 150, 200, 250]},
                    "min_samples_leaf": {"name": "min_samples_leaf", "choices": [1, 2, 3]},
                },
            },
            HyperParamsOptimizationLevel.Slow.name: {
                "float": {
                    "max_features": {"name": "max_features", "low": 0.7, "high": 1.0, "step": 0.005}
                },
                "categorical": {
                    "n_estimators": {
                        "name": "n_estimators",
                        "choices": [100, 150, 200, 250, 300, 350, 400, 450],
                    },
                    "min_samples_leaf": {"name": "min_samples_leaf", "choices": [1, 2, 3]},
                },
            },
        }

        params["DecisionTreeClassifier"] = {
            HyperParamsOptimizationLevel.Fast.name: {
                "float": {
                    "max_features": {"name": "max_features", "low": 0.8, "high": 1.0, "step": 0.005}
                },
                "categorical": {
                    "min_samples_leaf": {"name": "min_samples_leaf", "choices": [1, 2, 3]}
                },
            },
            HyperParamsOptimizationLevel.Slow.name: {
                "float": {
                    "max_features": {"name": "max_features", "low": 0.7, "high": 1.0, "step": 0.005}
                },
                "categorical": {
                    "min_samples_leaf": {"name": "min_samples_leaf", "choices": [1, 2, 3]}
                },
            },
        }

        params["AdaBoostClassifier"] = {
            HyperParamsOptimizationLevel.Fast.name: {
                "float": {
                    "learning_rate": {
                        "name": "learning_rate",
                        "low": 0.2,
                        "high": 0.35,
                        "step": 0.005,
                    }
                },
                "categorical": {
                    "n_estimators": {"name": "n_estimators", "choices": [100, 150, 200, 250]}
                },
            },
            HyperParamsOptimizationLevel.Slow.name: {
                "float": {
                    "learning_rate": {
                        "name": "learning_rate",
                        "low": 0.05,
                        "high": 0.35,
                        "step": 0.005,
                    }
                },
                "categorical": {
                    "n_estimators": {
                        "name": "n_estimators",
                        "choices": [100, 150, 200, 250, 300, 350, 400, 450],
                    }
                },
            },
        }

        params["LogisticRegression"] = {
            HyperParamsOptimizationLevel.Fast.name: {
                "categorical": {
                    "C": {"name": "C", "choices": [0.001, 0.01, 0.1, 1, 10, 100]},
                    "multi_class": {"name": "multi_class", "choices": ["ovr", "multinomial"]},
                    "class_weight": {"name": "class_weight", "choices": [None, "balanced"]},
                    "tol": {"name": "tol", "choices": [1e-4, 1e-3, 1e-2, 1e-1]},
                    "max_iter": {"name": "max_iter", "choices": [100, 500, 1000]},
                },
            },
            HyperParamsOptimizationLevel.Slow.name: {
                "categorical": {
                    "C": {"name": "C", "choices": [0.001, 0.01, 0.1, 1, 10, 100]},
                    "multi_class": {"name": "multi_class", "choices": ["ovr", "multinomial"]},
                    "class_weight": {"name": "class_weight", "choices": [None, "balanced"]},
                    "tol": {"name": "tol", "choices": [1e-4, 1e-3, 1e-2, 1e-1]},
                    "max_iter": {"name": "max_iter", "choices": [100, 500, 1000]},
                },
            },
        }

        params["SVC"] = {
            HyperParamsOptimizationLevel.Fast.name: {
                "categorical": {
                    "C": {"name": "C", "choices": [0.001, 0.01, 0.1, 1, 10, 100]},
                    "kernel": {"name": "kernel", "choices": ["linear", "poly", "rbf", "sigmoid"]},
                    "class_weight": {"name": "class_weight", "choices": [None, "balanced"]},
                    "tol": {"name": "tol", "choices": [1e-4, 1e-3, 1e-2, 1e-1]},
                    "max_iter": {"name": "max_iter", "choices": [100, 500, 1000]},
                    "degree": {"name": "degree", "choices": [2, 3, 4]},
                    "gamma": {"name": "gamma", "choices": ["scale", "auto"]},
                    "coef0": {"name": "coef0", "choices": [0.0, 1.0]},
                },
            },
            HyperParamsOptimizationLevel.Slow.name: {
                "categorical": {
                    "C": {"name": "C", "choices": [0.001, 0.01, 0.1, 1, 10, 100]},
                    "kernel": {"name": "kernel", "choices": ["linear", "poly", "rbf", "sigmoid"]},
                    "class_weight": {"name": "class_weight", "choices": [None, "balanced"]},
                    "tol": {"name": "tol", "choices": [1e-4, 1e-3, 1e-2, 1e-1]},
                    "max_iter": {"name": "max_iter", "choices": [100, 500, 1000]},
                    "degree": {"name": "degree", "choices": [2, 3, 4]},
                    "gamma": {"name": "gamma", "choices": ["scale", "auto"]},
                    "coef0": {"name": "coef0", "choices": [0.0, 1.0]},
                },
            },
        }

        params["ElasticNet"] = {
            HyperParamsOptimizationLevel.Fast.name: {
                "categorical": {
                    "alpha": {"name": "alpha", "choices": [0.0001, 0.001, 0.01, 0.1, 1.0]},
                    "l1_ratio": {"name": "l1_ratio", "choices": [0.0, 0.25, 0.5, 0.75, 1.0]},
                    "normalize": {"name": "normalize", "choices": [True, False]},
                    "tol": {"name": "tol", "choices": [1e-4, 1e-3, 1e-2, 1e-1]},
                    "max_iter": {"name": "max_iter", "choices": [100, 500, 1000]},
                    "positive": {"name": "positive", "choices": [False, True]},
                },
            },
            HyperParamsOptimizationLevel.Slow.name: {
                "categorical": {
                    "alpha": {"name": "alpha", "choices": [0.0001, 0.001, 0.01, 0.1, 1.0]},
                    "l1_ratio": {"name": "l1_ratio", "choices": [0.0, 0.25, 0.5, 0.75, 1.0]},
                    "normalize": {"name": "normalize", "choices": [True, False]},
                    "tol": {"name": "tol", "choices": [1e-4, 1e-3, 1e-2, 1e-1]},
                    "max_iter": {"name": "max_iter", "choices": [100, 500, 1000]},
                    "positive": {"name": "positive", "choices": [False, True]},
                },
            },
        }

        params["LassoLarsCV"] = {
            HyperParamsOptimizationLevel.Fast.name: {
                "categorical": {
                    "precompute": {"name": "precompute", "choices": ["auto", True, False]},
                    "eps": {"name": "eps", "choices": [2.220446049250313e-16, 1e-5, 1e-4]},
                    "normalize": {"name": "normalize", "choices": [True, False]},
                    "max_iter": {"name": "max_iter", "choices": [100, 500, 1000]},
                    "positive": {"name": "positive", "choices": [False, True]},
                },
            },
            HyperParamsOptimizationLevel.Slow.name: {
                "categorical": {
                    "precompute": {"name": "precompute", "choices": ["auto", True, False]},
                    "eps": {"name": "eps", "choices": [2.220446049250313e-16, 1e-5, 1e-4]},
                    "normalize": {"name": "normalize", "choices": [True, False]},
                    "max_iter": {"name": "max_iter", "choices": [100, 500, 1000]},
                    "positive": {"name": "positive", "choices": [False, True]},
                },
            },
        }

        params["LGBMRegressor"] = {
            HyperParamsOptimizationLevel.Fast.name: {
                "float": {
                    "learning_rate": {
                        "name": "learning_rate",
                        "low": 0.2,
                        "high": 0.35,
                        "step": 0.005,
                    }
                },
                "int": {"max_depth": {"name": "max_depth", "low": 5, "high": 7}},
                "categorical": {
                    "class_weight": {"name": "class_weight", "choices": [None, "balanced"]},
                    "n_estimators": {"name": "n_estimators", "choices": [100, 150, 200, 250]},
                },
            },
            HyperParamsOptimizationLevel.Slow.name: {
                "float": {
                    "learning_rate": {
                        "name": "learning_rate",
                        "low": 0.05,
                        "high": 0.35,
                        "step": 0.005,
                    }
                },
                "int": {"max_depth": {"name": "max_depth", "low": 3, "high": 8}},
                "categorical": {
                    "class_weight": {"name": "class_weight", "choices": [None, "balanced"]},
                    "n_estimators": {
                        "name": "n_estimators",
                        "choices": [100, 150, 200, 250, 300, 350, 400, 450],
                    },
                },
            },
        }

        params["XGBRegressor"] = {
            HyperParamsOptimizationLevel.Fast.name: {
                "float": {
                    "learning_rate": {
                        "name": "learning_rate",
                        "low": 0.2,
                        "high": 0.35,
                        "step": 0.005,
                    }
                },
                "int": {"max_depth": {"name": "max_depth", "low": 5, "high": 7}},
                "categorical": {
                    "n_estimators": {"name": "n_estimators", "choices": [100, 150, 200, 250]},
                    "reg_alpha": {"name": "reg_alpha", "choices": [0.001, 0.01, 0.1, 1, 10, 100]},
                    "reg_lambda": {"name": "reg_lambda", "choices": [0.001, 0.01, 0.1, 1, 10, 100]},
                },
            },
            HyperParamsOptimizationLevel.Slow.name: {
                "float": {
                    "learning_rate": {
                        "name": "learning_rate",
                        "low": 0.05,
                        "high": 0.35,
                        "step": 0.005,
                    }
                },
                "int": {"max_depth": {"name": "max_depth", "low": 3, "high": 8}},
                "categorical": {
                    "n_estimators": {
                        "name": "n_estimators",
                        "choices": [100, 150, 200, 250, 300, 350, 400, 450],
                    },
                    "reg_alpha": {"name": "reg_alpha", "choices": [0.001, 0.01, 0.1, 1, 10, 100]},
                    "reg_lambda": {"name": "reg_lambda", "choices": [0.001, 0.01, 0.1, 1, 10, 100]},
                },
            },
        }

        params["LightGBM"] = {
            HyperParamsOptimizationLevel.Fast.name: {
                "float": {
                    "learning_rate": {
                        "name": "learning_rate",
                        "low": 0.2,
                        "high": 0.35,
                        "step": 0.005,
                    }
                },
                "int": {"max_depth": {"name": "max_depth", "low": 5, "high": 7}},
                "categorical": {
                    "n_estimators": {"name": "n_estimators", "choices": [100, 150, 200, 250]},
                    "reg_alpha": {"name": "reg_alpha", "choices": [0.001, 0.01, 0.1, 1, 10, 100]},
                    "reg_lambda": {"name": "reg_lambda", "choices": [0.001, 0.01, 0.1, 1, 10, 100]},
                },
            },
            HyperParamsOptimizationLevel.Slow.name: {
                "float": {
                    "learning_rate": {
                        "name": "learning_rate",
                        "low": 0.05,
                        "high": 0.35,
                        "step": 0.005,
                    }
                },
                "int": {"max_depth": {"name": "max_depth", "low": 3, "high": 8}},
                "categorical": {
                    "n_estimators": {
                        "name": "n_estimators",
                        "choices": [100, 150, 200, 250, 300, 350, 400, 450],
                    },
                    "reg_alpha": {"name": "reg_alpha", "choices": [0.001, 0.01, 0.1, 1, 10, 100]},
                    "reg_lambda": {"name": "reg_lambda", "choices": [0.001, 0.01, 0.1, 1, 10, 100]},
                },
            },
        }

        params["XGBoost"] = {
            HyperParamsOptimizationLevel.Fast.name: {
                "float": {
                    "learning_rate": {
                        "name": "learning_rate",
                        "low": 0.2,
                        "high": 0.35,
                        "step": 0.005,
                    }
                }
            },
            HyperParamsOptimizationLevel.Slow.name: {
                "float": {
                    "learning_rate": {
                        "name": "learning_rate",
                        "low": 0.05,
                        "high": 0.35,
                        "step": 0.005,
                    }
                }
            },
        }

        params["GradientBoostingRegressor"] = {
            HyperParamsOptimizationLevel.Fast.name: {
                "float": {
                    "learning_rate": {
                        "name": "learning_rate",
                        "low": 0.2,
                        "high": 0.35,
                        "step": 0.005,
                    }
                },
                "int": {"max_depth": {"name": "max_depth", "low": 5, "high": 7}},
                "categorical": {
                    "n_estimators": {"name": "n_estimators", "choices": [100, 150, 200, 250]},
                },
            },
            HyperParamsOptimizationLevel.Slow.name: {
                "float": {
                    "learning_rate": {
                        "name": "learning_rate",
                        "low": 0.05,
                        "high": 0.35,
                        "step": 0.005,
                    }
                },
                "int": {"max_depth": {"name": "max_depth", "low": 3, "high": 8}},
                "categorical": {
                    "n_estimators": {
                        "name": "n_estimators",
                        "choices": [100, 150, 200, 250, 300, 350, 400, 450],
                    }
                },
            },
        }

        params["RandomForestRegressor"] = {
            HyperParamsOptimizationLevel.Fast.name: {
                "float": {
                    "max_features": {"name": "max_features", "low": 0.8, "high": 1.0, "step": 0.005}
                },
                "categorical": {
                    "n_estimators": {"name": "n_estimators", "choices": [100, 150, 200, 250]},
                    "min_samples_leaf": {"name": "min_samples_leaf", "choices": [1, 2, 3]},
                },
            },
            HyperParamsOptimizationLevel.Slow.name: {
                "float": {
                    "max_features": {"name": "max_features", "low": 0.7, "high": 1.0, "step": 0.005}
                },
                "categorical": {
                    "n_estimators": {
                        "name": "n_estimators",
                        "choices": [100, 150, 200, 250, 300, 350, 400, 450],
                    },
                    "min_samples_leaf": {"name": "min_samples_leaf", "choices": [1, 2, 3]},
                },
            },
        }

        params["DecisionTreeRegressor"] = {
            HyperParamsOptimizationLevel.Fast.name: {
                "float": {
                    "max_features": {"name": "max_features", "low": 0.8, "high": 1.0, "step": 0.005}
                },
                "categorical": {
                    "min_samples_leaf": {"name": "min_samples_leaf", "choices": [1, 2, 3]}
                },
            },
            HyperParamsOptimizationLevel.Slow.name: {
                "float": {
                    "max_features": {"name": "max_features", "low": 0.7, "high": 1.0, "step": 0.005}
                },
                "categorical": {
                    "min_samples_leaf": {"name": "min_samples_leaf", "choices": [1, 2, 3]}
                },
            },
        }

        return params

    def optimizer_configuration(self, model_wrapper: ModelManagerBase) -> bool:
        """Configure the optimizer.

        Args:
            model_wrapper: The model manager object.

        Returns:
            True on success.

        Raises:
            ValueError: If the configuration fails.
        """
        self.model = (model_wrapper.clone()).get_model()
        self.model_name = type(model_wrapper.model).__name__
        if self.optimization_level.name in self.params_dict[self.model_name]:
            self.params_model = self.params_dict[self.model_name][self.optimization_level.name]
            return True
        else:
            raise ValueError(
                f"Hyper parameters optimization level {self.optimization_level.name} "
                f"doesnt exist for model {self.model_name}"
            )
