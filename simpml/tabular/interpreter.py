"""Tabular interpreter."""

# Future imports
from __future__ import annotations

# Standard library imports
import inspect
import textwrap
from functools import wraps
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    Union,
)

# Third-party imports
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import shap
import sklearn.preprocessing as preprocessing
from lightgbm import LGBMClassifier
from plotly.subplots import make_subplots
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import export_text
from ydata_profiling import ProfileReport

# Conditional imports
try:
    # Version 2.x
    from dtreeviz import dtreeviz, DTreeVizAPI
except ImportError:
    # Version 1.x
    from dtreeviz.trees import DTreeViz as DTreeVizAPI
    from dtreeviz.trees import dtreeviz

from simpml.core.base import (
    DataManagerBase,
    FeatureImportanceMethod,
    MetricManagerBase,
    MinOrMax,
    PandasModelManagerBase,
    PredictionType,
)
from simpml.tabular.shap_manager import ShapManager

PLOTLY_DEFAULT_COLORS = px.colors.qualitative.Plotly

shap.initjs()
sns.set()


class ShapDatasetsDict(TypedDict):
    """Exact definition of the dictionary used to cache the SHAP datasets."""

    Train: Optional[pd.DataFrame]
    Valid: Optional[pd.DataFrame]
    Train_np: Optional[np.ndarray]
    Valid_np: Optional[np.ndarray]


def default_data_fill(f: Callable) -> Callable:
    """Decorator function to automatically fill X (feature data) and y (target data) variables
    in a wrapped API if those are set to None.

    If "y" is not present in a wrapped API, it will skip filling the target data.

    Args:
        f: The wrapped function.

    Returns:
        The decorator.
    """

    @wraps(f)
    def wrapper(
        self: TabularInterpreterBase,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """The wrapper function for the decorator.

        Args:
            self: Since this is meant to wrap class instance methods, this is the "self" argument
               to a class instance that is derived from `TabularInterpreterBase`.
            X: The argument for the feature data.
            y: The argument for the target data.
            *args: Any other arguments for the wrapped function.
            **kwargs: Any other keyword arguments for the wrapped function.

        Returns:
            Whatever the wrapped function returns.
        """
        if X is None:
            X = self.X_valid
        if "y" in inspect.getfullargspec(f)[0]:
            if y is None:
                y = self.y_valid
            return f(self, X, y, *args, **kwargs)
        return f(self, X, *args, **kwargs)

    return wrapper


class TabularInterpreterBase:
    """Base class for all other interpreters. Contains basic funcionality that is shared across all
    model and problem types.

    Input:
        model: Trained ModelWrapper object.
        data_manager: Object that contains data used for model.
        opt_metric: Optimization Metric.
        pos_class: Positive label for binary classification problem.

    Attributes:
        _estimator_type (str): ModelWrapper type
        model (BaseEstimator): from ModelWrapper.model
        predict_proba_support (bool): Whether the model supports predict proba.
        opt_metric (MetricWrapper): The metric the model optimized.
        model_data (TabularDataManager):
        pos_class (int or str or None): Positive class for classification problems.
        X_train (pd.DataFrame): Models training data.
        y_train (pd.DataFrame): Models training labels.
        X_valid (pd.DataFrame): Models validation data.
        y_valid (pd.DataFrame): Models validation labels.

        For SHAP-supported models:
        explainer (shap.TreeExplainer): on ModelWrapper.model

    Functions:
        get_explainer
        default_data_fill
        has_pred_proba
        generate_tooltip
        __calculate_permutation_importance
        __get_shap_values
        __calculate_shap_fi
        plot_summary_shap
        plot_shap_on_row
        plot_shap_dependence
        __display_tree
        __print_tree
        __calculate_feat_importance
        keep_impacting_feature_importance
        get_feature_importance
        plot_feature_importance
        get_bias_variance_decomposition_plot
        leakage_detector
        get_noisy_features
        pandas_profiler_feature_importances
    """

    def __init__(
        self,
        model: PandasModelManagerBase,
        data_manager: DataManagerBase,
        opt_metric: MetricManagerBase,
        pos_class: Optional[Dict[str, int]] = None,
        verbose: Optional[bool] = True,
        enable_shap: Optional[bool] = True,
        shap_timeout: int = 60,
    ) -> None:
        """Intializes the TabularInterpreterBase class.

        Args:
            model: The interface class instance for managing a model.
            data_manager: The interface class instance for managing data.
            opt_metric: The interface class instance for managing metrics.
            pos_class: Positive class for classification problems.
            verbose: Whether to output details.
            enable_shap: Dictates whether SHAP operations will be supported.
            shap_timeout: The timeout for getting the shap values in seconds.
        """
        self.model: PandasModelManagerBase = model
        self.predict_proba_support: bool = self.has_pred_proba()
        self.opt_metric: MetricManagerBase = opt_metric
        self.model_data: DataManagerBase = data_manager
        self.pos_class: Optional[Dict[str, int]] = pos_class
        self.shap_check_additivity: bool = True
        self.X_train, self.y_train = data_manager.get_training_data()
        self.X_valid, self.y_valid = data_manager.get_validation_data()
        self.verbose: Optional[bool] = verbose
        self.shap_timeout = shap_timeout
        self.shap_enabled = False
        self.enable_shap = enable_shap
        if self.enable_shap:
            try:
                self.shap_manager = ShapManager(
                    self.model,
                    self.X_train,
                    self.X_valid,
                    self.shap_check_additivity,
                    self.shap_timeout,
                )
                self.shap_enabled = True
            except Exception as e:
                print(f"Shap manager failed with error {e}, SHAP will not be functional")
                self.shap_enabled = False
        if model.name == "Decision Tree":
            self.display_tree = self.__display_tree
            self.print_tree = self.__print_tree

    def has_pred_proba(self) -> bool:
        """Check whether model has a "predict_proba" method.

        Returns:
            True if the model has a "predict_proba" method. False otherwise.
        """
        try:
            self.model.model.predict_proba
            return True
        except Exception:
            return False

    def generate_tooltip(self, coords: Sequence[int], text: str) -> go.Figure:
        """Generates 'tooltips' for the plots in the main figures.

        The function places a green asterisk (customized one point scatter plot), that by hovering
        over it displays the string in the text parameter.

        The function is meant to prevent some code repetition, however the coords parameter is
        highly dependent on the plot and its axis types.

        Args:
            coords: Coordinates where the tooltip should be placed. For categorical axes, should
                be a category name
            text: Text to display upon hover. For readability, text should formatted using
                textwrap (examples provided in main_fig plots).

        Returns:
            A plotly figure, should be overlayed / joined with the main figure where the tooltip
            should show.
        """
        tooltip_df = pd.DataFrame({"x": [coords[0]], "y": [coords[1]], "About": text})
        fig = px.scatter(
            tooltip_df,
            x="x",
            y="y",
            symbol="x",
            symbol_sequence=["asterisk"],
            hover_data={"x": False, "y": False, "About": True},
        )

        fig.update_traces(hovertemplate=f"About Plot:<br> {text}")
        fig.update_traces(marker=dict(size=12, line=dict(width=2, color="green")))
        return fig

    # Permutation importance
    def __calculate_permutation_importance(
        self, X: pd.DataFrame, y: Optional[pd.Series], n_repeats: int = 3
    ) -> pd.DataFrame:
        """Calculate the permutation importance used in the sklearn library

        Args:
            X: Input Dataset.
            y: Input target.
            n_repeats: Number of permutation importance iterations.

        Returns:
            The calculated permutation importance as a data frame.
        """

        def score(estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> float:
            """Score a model based on the metric `self.opt_metric`.

            Args:
                estimator: The model to score.
                X: The feature data.
                y: The true labels.

            Returns:
                The score.
            """
            y_pred = estimator.predict(X)
            return self.opt_metric.metric(y, y_pred)

        r = permutation_importance(self.model.model, X, y, n_repeats=n_repeats, scoring=score)

        feat_imp = pd.DataFrame(
            r["importances_mean"], index=X.columns, columns=["feature_importance_vals"]
        ).reset_index()
        feat_imp = feat_imp.rename({"index": "col_name"}, axis=1).sort_values(
            "feature_importance_vals", ascending=False
        )
        feat_imp["feature_importance_vals"] = (
            feat_imp["feature_importance_vals"] / feat_imp["feature_importance_vals"].sum()
        )

        return feat_imp

    @default_data_fill
    def __calculate_shap_fi(
        self, X: pd.DataFrame, shap_values: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Produces an importance table based on SHAP (much less biased than native feature
        importance implementations that exist for individual algorithms).

        Args:
            x: The local population relevant to the calculation.
            shap_values: The function may have the Shap values to save running time, otherwise it
                will perform the calculation.

        Returns:
            A data frame with the importance table.
        """
        if not self.shap_enabled:
            print("Shap not enabled in this interpreter instance, returning")
            return
        return self.shap_manager.calculate_shap_fi(X=X, shap_values=shap_values)

    # Display tree (not tested yet)
    @default_data_fill
    def __display_tree(
        self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None
    ) -> DTreeVizAPI:
        assert X is not None
        assert y is not None

        dt = self.model.model
        viz = dtreeviz(
            dt, X, y, target_name="placeholder", class_names=list(y.unique()), feature_names=list(X)
        )
        return viz

    @default_data_fill
    def __print_tree(self, X: Optional[pd.DataFrame] = None) -> None:
        assert X is not None

        dt = self.model.model
        r = export_text(dt, feature_names=list(X))
        print(r)

    # Feature importance
    def __calculate_feat_importance(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        method: FeatureImportanceMethod = FeatureImportanceMethod.Shap,
    ) -> pd.DataFrame:
        """Calculate the local feature importance.

        The function performs the calculation on the data it has received.
        The function returns the feature importance according to SHAP if available, otherwise the
        permutation importance.

        Args:
            X: Input Dataset.
            y: Input target.
            method: The method to calculate feature importance by.

        Returns:
            Data frame containing features and their importance.
        """
        if method == FeatureImportanceMethod.Shap and self.shap_enabled:
            feat_imp = self.shap_manager.calculate_shap_fi(
                X=X, shap_values=self.shap_manager.get_shap_values(X)
            )
        elif method == FeatureImportanceMethod.Permutation:
            feat_imp = self.__calculate_permutation_importance(X, y)
        elif not self.shap_enabled and method in [
            FeatureImportanceMethod.Shap,
            FeatureImportanceMethod.Permutation,
        ]:
            feat_imp = self.__calculate_permutation_importance(X, y)
        else:
            raise ValueError(
                "Please use in one of the following options only: "
                "[FeatureImportanceMethod.Shap, FeatureImportanceMethod.Permutation]"
            )
        return feat_imp

    @default_data_fill
    def keep_impacting_feature_importance(
        self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None, keep_imp: int = 0
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Returns the top `keep_imp` features ordered by feature importance.

        Args:
            X: Input Dataset.
            y: Input target.
            keep_imp: Number of features to keep

        Returns:
            The top `keep_imp` features ordered by feature importance.
        """
        assert X is not None

        if keep_imp >= len(X.columns):
            raise ValueError("keep_imp larger than number of features in X")
        fi = self.get_feature_importance(X, y)
        fi["cumsum"] = fi["feature_importance_vals"].cumsum()
        impacting: pd.DataFrame = fi.loc[fi["cumsum"] <= keep_imp].drop("cumsum", axis=1)
        if self.verbose:
            print(f"dropped {fi.shape[0] - len(impacting)} columns from feature importance table")
        return impacting, list(set(fi["col_name"]) - set(impacting["col_name"]))

    @default_data_fill
    def get_feature_importance(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        method: FeatureImportanceMethod = FeatureImportanceMethod.Shap,
    ) -> pd.DataFrame:
        """Get feature importance.

        Args:
            X: Input Dataset.
            y: Input target.
            method: The method to calculate feature importance by.

        Returns:
            Data frame containing features and their importance.
        """
        return self.__calculate_feat_importance(X, y, method=method)

    def plot_feature_importance(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        method: FeatureImportanceMethod = FeatureImportanceMethod.Shap,
        n_view: int = 10,
        size: Optional[Sequence[int]] = None,
        main_fig: bool = False,
    ) -> go.Figure:
        """Plot feature importance.

        Args:
            X: Input Dataset.
            y: Input target.
            method: The method to calculate feature importance by.
            n_view: Number of features to display.
            size: Tuple of width, height of the plot. None for auto-sizing.

        Returns:
            The generated plotly figure.
        """
        FEAT_IMPORTANCE_TOOLTIP = "<br>".join(
            textwrap.wrap(
                "Feature Importance Plot which displays the most important features for the "
                "models decision found using the method described in the method parameter()",
                width=40,
            )
        )
        feature_importance_df = self.get_feature_importance(X, y, method=method)
        importance = feature_importance_df["feature_importance_vals"][:n_view]
        names = feature_importance_df["col_name"][:n_view]
        feature_importance = np.array(importance)
        feature_names = np.array(names)
        last_feat_name = feature_names[-1]
        max_feat_importance = feature_importance[0]

        data = {"feature_names": feature_names, "feature_importance": feature_importance}
        fi_df = pd.DataFrame(data)

        fig = px.bar(fi_df, y="feature_importance", x="feature_names")
        tooltip = self.generate_tooltip(
            [last_feat_name, max_feat_importance], FEAT_IMPORTANCE_TOOLTIP
        )
        if main_fig:
            fig = go.Figure(fig.data + tooltip.data)
        fig.update_xaxes(title_text="Features")
        fig.update_yaxes(title_text="Score")
        fig.update_layout(title="Feature importance plot")
        if size is not None and not main_fig:
            fig.update_layout(autosize=False, width=size[0], height=size[1])
        labels = fig["data"][-1]["x"]

        new_labels = [
            "<br>".join(label[i : i + 30] for i in range(0, len(label), 30)) for label in labels
        ]

        fig["data"][-1]["x"] = new_labels

        return fig

    def get_bias_variance_decomposition_plot(
        self,
        target_perf: Optional[Union[float, int]] = None,
        hide_test_score: bool = False,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        opt_metric: Optional[MetricManagerBase] = None,
    ) -> go.Figure:
        """This method creates a waterfall plot by the dataset scores.

        The diffrences between the scores are presented in the plot.

        Args:
            target_perf: Adds target performance column for comparison.
            hide_test_score: Flag to show or hide test score
            X_test: Input Dataset
            y_test: Input target
            opt_metric: Metric to calculate scores for.

        Returns:
            Displays a plot of the bias variance decomposition.
        """
        if opt_metric is None:
            opt_metric = self.opt_metric
        train_score = opt_metric.metric(self.y_train, self.model.predict(self.X_train))
        valid_score = opt_metric.metric(self.y_valid, self.model.predict(self.X_valid))
        index: List[str] = []
        data: Dict[str, Any] = {"amount": []}
        base: List[Union[float, int]] = []
        if target_perf:
            if train_score < target_perf:
                index += ["Target Performance", "Underfitting"]
            else:
                index += ["Target Performance", "Target Train Difference"]
            data["amount"] += [target_perf, np.abs(target_perf - train_score)]
            base += [0, min(target_perf, train_score)]
        index += ["Train Score", "Overfitting", "Validation Score"]
        data["amount"] += [train_score, train_score - valid_score, valid_score]
        base += [0, valid_score, 0]
        if X_test is not None and y_test is not None:
            test_score = opt_metric.metric(y_test, self.model.predict(X_test))
            index += ["Validation Overfitting", "Test Score"]
            data["amount"] += [valid_score - test_score, test_score]
            base += [test_score, 0]

        colors = []
        amnts = data["amount"]
        c1, c2, c3 = PLOTLY_DEFAULT_COLORS[:3]
        for i in range(len(index)):
            if i % 2:
                if amnts[i + 1] > amnts[i - 1]:
                    colors.append(c3)
                else:
                    colors.append(c2)
            else:
                colors.append(c1)
        d = {"data": data["amount"], "colors": colors, "base": base, "index": index}
        df = pd.DataFrame(data=d)
        df["category"] = [str(i) for i in df.index]
        fig = px.bar(
            df,
            x="index",
            y="data",
            base="base",
            color="category",
            color_discrete_sequence=colors,
        )
        fig.update_layout(showlegend=False)
        return fig

    # Leakage detector
    # Depends on model.clone
    @default_data_fill
    def leakage_detector(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        epsilon: Optional[float] = 0.05,
    ) -> bool:
        """This function detects a suspected leakage and locates the suspected features.

        The function measures the difference in performance between a model with all the
        features and a model with only the most important feature. If the difference between
        the models is smaller than epsilon or the second model is better, we suspect that it is a
        leakage.

        Args:
            X: Input Dataset.
            y: Input target.
            epsilon: Difference criterion for deciding if there is a data leakage.

        Returns:
            Boolean for whether a data leakage was detected or not, the method also prints relevant
            information on suspected features.
        """
        assert X is not None

        old_score = self.opt_metric.metric(self.model.predict(X), y)
        new_model = self.model.clone()
        most_importance_feature = [self.__calculate_feat_importance(X, y).loc[0]["col_name"]]
        new_model.model.fit(self.X_train[most_importance_feature], self.y_train)
        new_score = self.opt_metric.metric(new_model.predict(X[most_importance_feature]), y)
        leakage_detected = bool(old_score - new_score < epsilon)
        if leakage_detected:
            if self.verbose:
                print(f"Suspected data leakage detected! feature {most_importance_feature[0]}")
                print(f"Model results with the all the data: {old_score} {self.opt_metric.name}")
                print(f"Model results with this feature only: {new_score} {self.opt_metric.name}")
                print(
                    f"The difference between the results of the models is {old_score - new_score} "
                    f"< smaller than {epsilon}"
                )
        return leakage_detected

    # Noisy features
    @default_data_fill
    def get_noisy_features(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        distance: float = 0.5,
        number_top_fi: int = 20,
        opt_metric: Optional[MetricManagerBase] = None,
    ) -> List[str]:
        """Feature reduction method based on SHAP.

        Full multi-dimensional support. The starting point is that the higher the correlation
        between SHAP values and the original values, the more significant this feature is, and the
        lower this correlation, the more likely it is to over-match.

        The function detects suspicious features by calculating the distances between the Shap
        values and the original values and tries to remove them, if after removal the score is
        higher, this function is added to the removal list and the procedure is repeated.

        Args:
            X: Input Dataset.
            y: Input target.
            distance: Represents between_shap_populations is a value between [0,1], and roughly
                translates to how "separate" we demand our SHAP populations to be.
            number_top_fi: The amount of features on which the calculation is performed in each
                iteration.
            opt_metric: Optimization metric. If None, will use the default set in the class
                instance.

        Returns:
            List of noisy features the algorithm found.
        """
        if not self.shap_enabled:
            print(
                "SHAP is not enabled since the ShapManager failed to start "
                "and SHAP is needed to run this function, returning"
            )
            return []

        def get_noisy_features_iteration(
            model: BaseEstimator,
            x: pd.DataFrame,
            number_top_fi: int = 20,
            distance: Union[float, int] = 1,
        ) -> pd.DataFrame:
            """Feature reduction method based on SHAP.

            Full multi-dimensional support.

            TODO: Add dimension name.

            Args:
                model: The model to get the noisy features from.
                x: The feature data.
                number_top_fi: The amount of features on which the calculation is performed in
                    each iteration.
                distance: Represents between_shap_populations is a value between [0,1], and
                    roughly translates to how "separate" we demand our SHAP populations to be.

            Returns:
                A data frame with the noisy features.
            """

            def get_noisy_features_per_dim(
                vals: shap.Explanation, x: pd.DataFrame, fi_list: List[str], number_top_fi: int
            ) -> List[np.ndarray]:
                def normalization_df(df: pd.DataFrame) -> pd.DataFrame:
                    min_max_scaler = preprocessing.MinMaxScaler()
                    x_scaled = min_max_scaler.fit_transform(df)
                    return pd.DataFrame(x_scaled, columns=df.columns)

                df_shap = pd.DataFrame(vals.values, columns=x.columns.values).drop(
                    fi_list[number_top_fi:], axis=1
                )

                # TEMPORARY SOLUTION - only work with numeric cols.
                # relevant for cases where we didn't one-hot encode the features, e.g. for lightGBM
                numeric_cols = [c for c in x.columns if str(x[c].dtype) != "category"]

                df_values = normalization_df(x[numeric_cols])
                noisy_features: List[np.ndarray] = []
                for col in fi_list[:number_top_fi]:
                    pos_shap_feat_values = (
                        df_values[col]
                        .loc[df_shap[col].loc[df_shap[col] > 0].index.to_list()]
                        .mean()
                    )
                    neg_shap_feat_values = (
                        df_values[col]
                        .loc[df_shap[col].loc[df_shap[col] < 0].index.to_list()]
                        .mean()
                    )
                    distance_between_populations = abs(pos_shap_feat_values - neg_shap_feat_values)
                    noisy_features.append(distance_between_populations)
                return noisy_features

            def get_noisy_features_by_distance(
                distance: Union[float, int],
                list_dimension: Sequence[str],
                noisy_features: pd.DataFrame,
            ) -> pd.DataFrame:
                suspicious = noisy_features.loc[
                    noisy_features[list_dimension[0]] <= distance
                ].index.to_list()
                for i in range(1, len(list_dimension)):
                    suspicious_dim = noisy_features.loc[
                        noisy_features[list_dimension[i]] <= distance
                    ].index.to_list()
                    for row in suspicious:
                        if row not in suspicious_dim:
                            suspicious.remove(row)
                return noisy_features.filter(items=suspicious, axis=0)

            vals: shap.Explanation = cast(
                shap.Explanation,
                self.shap_manager.get_shap_values(x, single_mat=False, as_df=False),
            )
            if isinstance(model, LGBMClassifier) and isinstance(vals, list):
                vals = vals[1]
            # The pylint warning seems to be a false positive since `pd.DataFrame` is subscriptable
            fi_list: List[str] = self.__calculate_shap_fi(
                X=x, shap_values=self.shap_manager.get_shap_values(x)
            )[  # pylint: disable=unsubscriptable-object
                "col_name"
            ].to_list()

            # TEMPORARY SOLUTION - ignore categorical features
            fi_list = [f for f in fi_list if str(x[f].dtype) != "category"]

            noisy_features = pd.DataFrame(fi_list[:number_top_fi], columns=["col_name"])
            if (len(np.array(vals).shape) == 2) or (
                (len(np.array(vals).shape) == 3) and isinstance(model, LGBMClassifier)
            ):  # THIS IS A TEMPORARY UGLY SOLUTION to support LGBMClassifier
                noisy_features["separate_vals"] = get_noisy_features_per_dim(
                    vals, x, fi_list, number_top_fi
                )
                noisy_features.sort_values(by="separate_vals", inplace=True)
                noisy_features = noisy_features.loc[noisy_features["separate_vals"] <= distance]

            else:
                list_dimension: List[str] = []
                for i in range(len(vals)):
                    list_dimension.append(f"separate_vals_dimension_{i + 1}")
                    noisy_features[list_dimension[i]] = get_noisy_features_per_dim(
                        vals[i], x, fi_list, number_top_fi
                    )
                noisy_features.sort_values(by="separate_vals_dimension_1", inplace=True)
                noisy_features = get_noisy_features_by_distance(
                    distance, list_dimension, noisy_features
                )
                if (
                    noisy_features["separate_vals_dimension_1"]
                    == noisy_features["separate_vals_dimension_2"]
                ).all() and len(noisy_features.columns) == 3:
                    noisy_features = noisy_features.drop("separate_vals_dimension_2", axis=1)
                    noisy_features = noisy_features.rename(
                        {"separate_vals_dimension_1": "separate_vals"}, axis=1
                    )
            return noisy_features.reset_index(drop=True)

        base_model: BaseEstimator = self.model
        x_train = self.X_train
        y_train = self.y_train
        x_valid = self.X_valid
        y_valid = self.y_valid
        if opt_metric is None:
            opt_metric = self.opt_metric
        metric = opt_metric.metric
        min_or_max_obj = self.opt_metric.get_optimization_direction()
        min_or_max = min_or_max_obj.value.lower()
        base_model = base_model.clone()
        base_model.model.fit(x_train, y_train)
        final_drop_list = []
        if not self.predict_proba_support:
            old_score = np.around(metric(y_valid, base_model.predict(x_valid)), 4)
        elif self.predict_proba_support:
            old_score = np.around(
                metric(y_valid, (base_model.model.predict_proba(x_valid) > 0.5)[:, 1].astype(int)),
                4,
            )
        else:
            print("Could not find if model supports predict proba, exiting")
            return

        while True:
            base_score = old_score
            new_score = base_score
            drop_list = get_noisy_features_iteration(
                base_model.model, x_train, distance=distance, number_top_fi=number_top_fi
            )["col_name"].to_list()
            for col in drop_list:
                x_train_new = x_train.drop(col, axis=1)
                x_valid_new = x_valid.drop(col, axis=1)

                new_model = base_model.clone()
                new_model.model.fit(x_train_new, y_train)

                if not self.predict_proba_support:
                    new_score = np.around(metric(y_valid, new_model.predict(x_valid_new)), 4)
                elif self.predict_proba_support:
                    new_score = np.around(
                        metric(
                            y_valid,
                            (new_model.model.predict_proba(x_valid_new)[:, 1] > 0.5).astype(int),
                        ),
                        4,
                    )
                flag = False

                if min_or_max == "max":
                    if new_score >= old_score:
                        flag = True
                elif min_or_max == "min":
                    if new_score <= old_score:
                        flag = True
                else:
                    print("min or max value not recognized")

                if flag:
                    print("Bad noisy feature found:", col)
                    print(
                        "old " + opt_metric.name + ":",
                        old_score,
                        "new " + opt_metric.name + ":",
                        new_score,
                    )
                    if isinstance(self, TabularInterpreterRegression):
                        y_true = y_valid
                        y_pred = new_model.predict(x_valid_new)
                        # mean_absolute_error = metrics.mean_absolute_error(y_pred, y_true)
                        mse = metrics.mean_squared_error(y_pred, y_true)
                        # median_absolute_error = metrics.median_absolute_error(y_pred, y_true)
                        r2 = metrics.r2_score(y_pred, y_true)

                        results = pd.DataFrame(
                            [
                                {
                                    "R2": round(r2, 4),
                                    "Mean Squared Error": round(mse, 4),
                                    "Root Mean Squared Error": round(np.sqrt(mse), 4),
                                }
                            ]
                        )

                        print(results)
                    else:
                        assert isinstance(
                            self,
                            (
                                TabularInterpreterBinaryClassification,
                                TabularInterpreterClassification,
                            ),
                        )
                        print(self.results(y_valid, new_model.predict(x_valid_new)))
                    base_model = new_model
                    old_score = new_score
                    x_train = x_train_new
                    x_valid = x_valid_new
                    final_drop_list.append(col)
                    break

            if min_or_max == "max":
                if base_score >= new_score:
                    break

            if min_or_max == "min":
                if base_score <= new_score:
                    break

        return final_drop_list

    @default_data_fill
    def pandas_profiler_feature_importances(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        keep_imp: float = 0.99,
    ) -> ProfileReport:
        """Run a profiler on the top features according to SHAP.

        Things to look at:
            - **Warnings section** - Diverse set of warnings regarding abnormalities in our data.
            - **Correlation section** - All kinds of correlations are comfortably calculated.
              between all variables. See additional explanation on correlations in the next section.
            - **Missing values section** - The interesting part here is viewing combination of
              missing values, i.e. feature_i and feature_j are frequently missed together. The
              dendrogram is a nice view for it.
        """
        assert X is not None

        feature_importance, drop_list = self.keep_impacting_feature_importance(
            keep_imp=0.99, X=X, y=y
        )
        df = X

        most_imp_profile = ProfileReport(
            df[feature_importance.loc[~feature_importance["col_name"].isin(drop_list)]["col_name"]],
            title="profiling report",
        )
        return most_imp_profile


class TabularInterpreterClassification(TabularInterpreterBase):
    """TabularInterpreter for classification problems (inherits from Base)

    Functions:
        get_label_density_plot
        print_report
        results
        get_label_density_plot_main_fig
        plot_confusion_matrix_main_fig
        plot_confusion_matrix
        main_fig
    """

    def __init__(
        self,
        model: PandasModelManagerBase,
        data_manager: DataManagerBase,
        opt_metric: MetricManagerBase,
        pos_class: Optional[Dict[str, int]] = None,
        verbose: Optional[bool] = True,
        enable_shap: Optional[bool] = True,
        shap_timeout: int = 120,
    ) -> None:
        """Intializes the TabularInterpreterClassification class.

        Args:
            model: The interface class instance for managing a model.
            data_manager: The interface class instance for managing data.
            opt_metric: The interface class instance for managing metrics.
            pos_class: Positive class for classification problems.
            verbose: Whether to output details.
            enable_shap: Whether shap operations are available in the interp instance.
            shap_timeout: The timeout for getting the shap values in seconds.
        """
        super().__init__(
            model, data_manager, opt_metric, pos_class, verbose, enable_shap, shap_timeout
        )

    @default_data_fill
    def print_report(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None) -> None:
        """Print a classification report (on the console).

        Args:
            X: Input Dataset.
            y: Input target.
        """
        assert X is not None
        assert y is not None

        if y.dtype == "O":
            print(self.results(y_true=y, y_pred=self.model.predict(X)))
        else:
            print(
                self.results(y_true=(y).astype(float), y_pred=self.model.predict(X).astype(float))
            )

    def results(
        self, y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
    ) -> str:
        """Returns an sklearn classification report.

        Args:
            y_true: True labels of the data.
            y_pred: Models predicted labels.

        Returns:
            The classification report.
        """
        return classification_report(y_true, y_pred)

    @default_data_fill
    def get_label_density_plot(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        size: Optional[Sequence[int]] = None,
    ) -> go.Figure:
        """Returns the label density plot, displaying the predicted label distribution for each
        class.

        Args:
            X: Input Dataset.
            y: Input target.
            size: Tuple of width, height of the plot. None for auto-sizing.

        Returns:
            The label density plot.
        """
        assert X is not None
        assert y is not None

        y_true = y
        y_proba = self.model.predict(X)
        df = pd.DataFrame({"Actual": y_true, "Probabilities": y_proba})
        temp_df = pd.DataFrame(
            df.groupby(["Actual", "Probabilities"])
            .size()
            .reset_index(level=["Actual", "Probabilities"])
            .rename(columns={"Probabilities": "Predicted", 0: "relative_count"})
        )

        temp_df["Total_count"] = temp_df.groupby("Actual")["relative_count"].transform("sum")
        pred_set = np.unique(temp_df["Predicted"])
        colors = PLOTLY_DEFAULT_COLORS[: len(pred_set)]
        color_match = {pred_set[i]: colors[i] for i in range(len(pred_set))}
        temp_df["Color"] = temp_df["Predicted"].map(color_match)
        bar = px.bar(temp_df, x="Actual", y="relative_count", color="Predicted")
        bar.update_xaxes(title_text="Ground Truth")
        bar.update_yaxes(title_text="Total Predictions")
        bar.update_layout(title="Predictions Distribution Among Classes")
        if size is not None:
            bar.update_layout(autosize=False, width=size[0], height=size[1])
        return bar

    @default_data_fill
    def get_label_density_plot_main_fig(
        self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None
    ) -> go.Figure:
        """Modified version of label density plot for the main figure, needs to return bar and
        tooltip seperately because of a bug when they're returned together.

        Args:
            X: Input Dataset.
            y: Input target.

        Returns:
            Label density plot with tooltip.
        """
        assert X is not None
        assert y is not None

        PROBA_TOOLTIP = "<br>".join(
            textwrap.wrap(
                "Predicted labels distribution plot which shows the distirbution of predicted "
                "labels for each class",
                width=40,
            )
        )
        y_true = y
        y_proba = self.model.predict(X)
        df = pd.DataFrame({"Actual": y_true, "Probabilities": y_proba})
        temp_df = pd.DataFrame(
            df.groupby(["Actual", "Probabilities"])
            .size()
            .reset_index(level=["Actual", "Probabilities"])
            .rename(columns={"Probabilities": "Predicted", 0: "relative_count"})
        )

        temp_df["Total_count"] = temp_df.groupby("Actual")["relative_count"].transform("sum")
        pred_set = np.unique(temp_df["Predicted"])
        colors = PLOTLY_DEFAULT_COLORS[: len(pred_set)]
        color_match = {pred_set[i]: colors[i] for i in range(len(pred_set))}
        temp_df["Color"] = temp_df["Predicted"].map(color_match)
        bar = go.Bar(
            x=temp_df["Actual"],
            y=temp_df["relative_count"],
            hovertext=temp_df["Predicted"],
            hovertemplate="Actual: %{x}" + "<br>Predicted: %{hovertext}" + "<br>Count: %{y}",
            marker=dict(color=temp_df["Color"]),
        )

        last_x_axis = bar.x[-1]
        max_count = temp_df["Total_count"].max()
        proba_tooltip = self.generate_tooltip([last_x_axis, max_count], PROBA_TOOLTIP)
        return bar, proba_tooltip

    @default_data_fill
    def plot_confusion_matrix_main_fig(
        self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None
    ) -> go.Figure:
        """Generates and plots a confusion matrix on the given data, in the graph objects format
        for the main fig.

        Args:
            X: Input Dataset.
            y: Input target.

        Returns:
            The plotly figure containing the confusion matrix plot.
        """
        assert X is not None
        assert y is not None

        z = metrics.confusion_matrix(y, self.model.predict(X))
        z_prop = ((metrics.confusion_matrix(y, self.model.predict(X)) / len(y)) * 100).astype(int)[
            ::-1
        ]
        cm_x = [str(x) for x in sorted(np.unique(y))]
        cm_y = cm_x.copy()
        z_text = [[str(y) for y in x] for x in z]
        z_prop_text = [[f"{str(y)}%" for y in x] for x in z_prop]
        final_z = [[x + "(" + y + ")" for x, y in zip(X, Y)] for X, Y in zip(z_text, z_prop_text)]
        heatmap = go.Heatmap(
            z=z,
            x=cm_x,
            y=cm_y,
            text=final_z,
            texttemplate="%{text}",
            colorscale="blues",
            showscale=False,
        )
        fig = go.Figure(data=[heatmap])

        return fig

    @default_data_fill
    def plot_confusion_matrix(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        size: Optional[Sequence[int]] = None,
    ) -> go.Figure:
        """Generates and plots a confusion matrix on the given data, in the graph objects format
        for the main fig.

        Args:
            X: Input Dataset
            y: Input target
            size: Tuple of width, height of the plot. None for auto-sizing.

        Returns:
            The plotly figure containing the confusion matrix plot.
        """
        assert X is not None
        assert y is not None

        z = metrics.confusion_matrix(y, self.model.predict(X))
        z_prop = ((metrics.confusion_matrix(y, self.model.predict(X)) / len(y)) * 100).astype(int)
        cm_x = [str(x) for x in sorted(np.unique(y))]
        cm_y = cm_x.copy()
        z_text = [[str(y) for y in x] for x in z]
        cm = px.imshow(
            z,
            x=cm_x,
            y=cm_y,
            text_auto=True,
            title="Confusion matrix",
            color_continuous_scale="blues",
        )
        cm.update_xaxes(title_text="Predicted")
        cm.update_yaxes(title_text="Ground Truth")
        cm.update_layout(title="Confusion Matrix")
        z_prop_text = [[f"{str(y)}%" for y in x] for x in z_prop]
        final_z = [[x + "(" + y + ")" for x, y in zip(X, Y)] for X, Y in zip(z_text, z_prop_text)]
        cm = go.Heatmap(
            z=z,
            x=cm_x,
            y=cm_y,
            text=final_z,
            texttemplate="%{text}",
            colorscale="blues",
            showscale=False,
        )
        fig = go.Figure(data=cm)
        if size is not None:
            fig.update_layout(autosize=False, width=size[0], height=size[1])
        fig.update_xaxes(title_text="Predicted")
        fig.update_yaxes(title_text="Ground Truth")
        fig.update_layout(title="Confusion Matrix")
        return fig

    @default_data_fill
    def main_fig(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        feat_imp_method: FeatureImportanceMethod = FeatureImportanceMethod.Shap,
        n_feats: int = 10,
        force_shap: bool = False,
        size: Optional[Tuple[int, int]] = (1100, 800),
    ) -> go.Figure:
        """Main figure plot for classification problems.

        Displays three plots:
        1: Label Density Plot
        2: Confusion Matrix
        3: Feature Importance

        Args:
            X: Input Dataset.
            y: Input target.
            feat_imp_method: Used to control the feature importance method used in the feature
                importance plot.
            n_feats: Number of features to display in the feature importance plot.
            force_shap: bool for whether to force shap based feature importance for models with
                slow SHAP runtimes.
            size: Tuple of width, height of the plot. None for auto-sizing.

        Returns:
            A plotly figure containing the final plot.
        """
        assert X is not None
        assert y is not None

        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=("Label Distribution Plot", "Confusion Matrix", "Feature Importance"),
            specs=[[{}], [{}], [{}]],
        )
        # Tooltip and graph need to be returned seperetaly and then overlayed
        # in order to be overlayed properly
        ret, tooltip = self.get_label_density_plot_main_fig(X, y)
        fig.add_trace(ret, row=1, col=1)
        for trace in tooltip.data:
            fig.add_trace(trace, row=1, col=1)

        fig.add_trace(self.plot_confusion_matrix_main_fig(X, y).data[0], row=2, col=1)

        for trace in self.plot_feature_importance(
            X, y, method=feat_imp_method, n_view=n_feats, main_fig=True
        ).data:
            fig.add_trace(trace, 3, 1)

        fig.update_xaxes(title_text="Label", row=1, col=1)
        fig.update_xaxes(title_text="Predicted", row=2, col=1)
        fig.update_xaxes(title_text="Feature Names", row=3, col=1)

        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Ground Truth", row=2, col=1)
        fig.update_yaxes(title_text="Importance", row=3, col=1)

        if size is not None:
            fig.update_layout(height=size[0], width=size[1])
        fig.update_coloraxes(showscale=False)
        fig.update_layout(showlegend=False)

        return fig


class TabularInterpreterBinaryClassification(TabularInterpreterClassification):
    """TabularInterpreter for binary classification problems."""

    def __init__(
        self,
        model: PandasModelManagerBase,
        data_manager: DataManagerBase,
        opt_metric: MetricManagerBase,
        pos_class: Optional[Dict[str, int]],
        verbose: Optional[bool] = True,
        enable_shap: Optional[bool] = True,
        shap_timeout: int = 120,
    ) -> None:
        """Intializes the TabularInterpreterBinaryClassification class.

        Args:
            model: The interface class instance for managing a model.
            data_manager: The interface class instance for managing data.
            opt_metric: The interface class instance for managing metrics.
            pos_class: Positive class for classification problems.
            verbose: Whether to output details.
            enable_shap: Whether shap operation are available in the interp instance.
            shap_timeout: The timeout for getting the shap values in seconds.
        """
        super().__init__(
            model, data_manager, opt_metric, pos_class, verbose, enable_shap, shap_timeout
        )
        self.predict_proba_support = self.has_pred_proba()
        if not self.predict_proba_support:
            if self.verbose:
                print(
                    "No predict proba function has been detected for your model so most "
                    "TabularInterpreterBinaryClassification functionality is not supported, "
                    "please use TabularInterpreterClassification instead"
                )

    def has_pred_proba(self) -> bool:
        """Check whether model has a "predict_proba" method.

        Returns:
            True if the model has a "predict_proba" method. False otherwise.
        """
        try:
            self.model.model.predict_proba
            return True
        except Exception:
            return False

    def __pred_from_pred_proba_by_threshold(
        self, y_pred_proba: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        return (y_pred_proba > threshold).astype(int)

    @default_data_fill
    def find_best_threshold(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        plot: bool = True,
        opt_metric: Optional[MetricManagerBase] = None,
        dataset_name: str = "",
    ) -> Union[Tuple[Union[int, float], pd.DataFrame], go.Figure]:
        """Finds the decision threshold which optimizes the models performance on a given metric
        (default metric is the optimazation metric defined in the experiment manager).

        Args:
            X: Input Dataset.
            y: Input target.
            plot: Whether to return a plot of thresholds vs metric.
            opt_metric: Metric which will be evaluated.
            dataset_name: Optional name to display in the title of the plot.

        Returns:
            If `plot` is False, the optimal threshold which maximizes the given metric (default
            is the experiment manager optimization metric). If `plot` is true, a plotly figure.
        """
        assert X is not None
        assert y is not None

        pred_proba = self.model.model.predict_proba(X)[:, 1].astype(float)
        y = np.array(y).astype(float)

        if opt_metric is None:
            opt_metric = self.opt_metric
        threshold = 0.0
        stop = 1.0
        increment = 0.05
        threshold_list = []
        scores_list_to_find_best = []
        while threshold <= stop:
            threshold += increment
            threshold_list.append(threshold)
            scores_list_to_find_best.append(
                opt_metric.metric(
                    y, self.__pred_from_pred_proba_by_threshold(pred_proba, threshold)
                )
            )

        res = pd.Series(
            index=threshold_list,
            data=scores_list_to_find_best,
            name=opt_metric.name + " Score on " + dataset_name,
        )
        threshold = round(res[res == max(res)].index[0], 2)
        if opt_metric.get_optimization_direction() == MinOrMax.Max:
            opt_val = max(res)
        elif opt_metric.get_optimization_direction() == MinOrMax.Min:
            opt_val = min(res)
        else:
            if self.verbose:
                print(
                    "Could not recognize metric.optimization_direction parameter "
                    "(should be Min or Max)"
                )
        palette_res = dict(
            threshold=res.index,
            score=scores_list_to_find_best,
            colors=["tomato" if val == opt_val else "cornflowerblue" for val in res],
        )

        df_res = pd.DataFrame(index=list(range(len(res))), data=palette_res)

        if plot:
            scatter = px.scatter(df_res, x="threshold", y="score", color="colors")
            line = px.line(df_res, x="threshold", y="score")
            fig = go.Figure(data=scatter.data + line.data)
            fig.update_layout(
                showlegend=False,
                title="Score vs Threshold",
                xaxis_title="Score",
                yaxis_title="Threshold",
            )
            return fig
        return threshold, df_res

    @default_data_fill
    def confusion_matrix_to_dataset_mapping(
        self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None
    ) -> Dict[str, pd.DataFrame]:
        """This method creates a map between each section in the confusion matrix (true-positive,
        tru-negative, false-positive, false-negative) to all its corresposding instances in the
        data.

        Args:
            X: Input Dataset.
            y: Input target.

        Returns:
            A dictionary mapping classification type (True Postive, False Positive, True Negative,
            False Negative) to a dataframe containing the matching rows.
        """
        assert X is not None
        assert y is not None

        classes = y.unique()

        df = pd.DataFrame({"True": y, "Pred": self.model.predict(X)}, index=y.index)
        # Encode different boxes by the true label and predicted label.
        df["Map"] = df["True"].astype(str) + df["Pred"].astype(str)
        # Calculate populations.
        my_dict: Dict[str, pd.DataFrame] = {}
        # Takes all the instances that their true labels are zero and their predicted labels are
        # zero and put them together as a dictionary value.
        my_dict["TN"] = X.loc[list(df[df["Map"] == (str(classes[0]) + str(classes[0]))].index)]
        my_dict["FP"] = X.loc[list(df[df["Map"] == (str(classes[0]) + str(classes[1]))].index)]
        my_dict["FN"] = X.loc[list(df[df["Map"] == (str(classes[1]) + str(classes[0]))].index)]
        my_dict["TP"] = X.loc[list(df[df["Map"] == (str(classes[1]) + str(classes[1]))].index)]

        return my_dict

    @default_data_fill
    def get_best_predictions(
        self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None, samples_num: int = 10
    ) -> pd.DataFrame:
        """This method returns the 'samples_num' samples on which the model predicts the closest
        scores to the true labels.

        Args:
            X: Input Dataset.
            y: Input target.
            samples_num: Number of samples to return.

        Returns:
            The top 'samples_num' best predictions, best being where the model was most
            confidently right (where confidence is the probability the model gave for the sample
            to belong to the predicted class).
        """
        assert X is not None
        assert y is not None

        if not self.predict_proba_support:
            raise ValueError(
                "This model doesn't predict by probabilities, so we don't have the confidence for "
                "each sample"
            )

        X_set_dict = self.confusion_matrix_to_dataset_mapping(X, y)
        samples_distances = self.compute_distances_by_probabilities(X, y)

        X = pd.concat(
            [X_set_dict["TP"], X_set_dict["FP"], X_set_dict["TN"], X_set_dict["FN"]]
        )  # create the original dataset
        best_confidence = X.loc[
            list(samples_distances.head(samples_num).index)
        ]  # extarct the samples with ambiguous confidence
        return best_confidence

    @default_data_fill
    def get_worst_predictions(
        self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None, samples_num: int = 10
    ) -> pd.DataFrame:
        """This method returns the 'samples_num' samples on which the model predicts the farthest
        scores from the true labels.

        Args:
            X: Input Dataset.
            y: Input target.
            samples_num: Number of samples to return.

        Returns:
            The top 'samples_num' worst predictions, worst being where the model was most
            confidently wrong (where confidence is the probability the model gave for the sample
            to belong to the predicted class).
        """
        assert X is not None
        assert y is not None

        if not self.predict_proba_support:
            raise ValueError(
                "This model doesn't predict by probabilities, so we don't have the confidence for "
                "each sample"
            )

        X_set_dict = self.confusion_matrix_to_dataset_mapping(X, y)

        X = pd.concat(
            [X_set_dict["TP"], X_set_dict["FP"], X_set_dict["TN"], X_set_dict["FN"]]
        )  # create the original dataset
        samples_distances = self.compute_distances_by_probabilities(X, y)
        worst_confidence = X.loc[
            list(samples_distances.tail(samples_num).index)
        ]  # extarct the samples with ambiguous confidence
        return worst_confidence

    @default_data_fill
    def get_ambiguous_predictions(
        self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None, epsilon: float = 0.1
    ) -> pd.DataFrame:
        """This method returns the samples on which the model predicts ambiguous scores, defined by
        the range of (0.5-epsilon) to (0.5+epsilon).

        Args:
            X: Input Dataset.
            y: Input target.
            epsilon: Ambiguity measure.

        Returns:
            The samples where the models prediction was ambiguous.
        """
        assert X is not None
        assert y is not None

        if not self.predict_proba_support:
            raise ValueError(
                "This model doesn't predict by probabilities, so we don't have the confidence for "
                "each sample"
            )
        X_set_dict = self.confusion_matrix_to_dataset_mapping(X, y)

        samples_distances = self.compute_distances_by_probabilities(X, y)

        X = pd.concat(
            [X_set_dict["TP"], X_set_dict["FP"], X_set_dict["TN"], X_set_dict["FN"]]
        )  # create the original dataset
        ambiguous_confidence = X.loc[
            list(
                (
                    samples_distances.loc[
                        (samples_distances[1] >= (0.5 - epsilon))
                        & (samples_distances[1] <= (0.5 + epsilon))
                    ]
                ).index
            )
        ]  # extarct the samples with ambiguous confidence
        return ambiguous_confidence

    @default_data_fill
    def compute_distances_by_probabilities(
        self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """This method computes, for each sample, the distance between the true label and the
        probability the model gave the sample to be 1 (positive), and returns the samples sorted
        in ascending order.

        Args:
            X: Input Dataset.
            y: Input target.

        Returns:
            A pandas DataFrame where the index of each row is the index of the same sample in the
            data, and the value is the computed distance.
        """
        assert X is not None
        assert y is not None

        if not self.predict_proba_support:
            raise ValueError(
                "This model doesn't predict by probabilities, so we don't have the confidence for "
                "each sample"
            )

        X_set_dict = self.confusion_matrix_to_dataset_mapping(X, y)

        TP_distances = 1 - pd.DataFrame(
            pd.DataFrame(
                self.model.model.predict_proba(X_set_dict["TP"]), index=X_set_dict["TP"].index
            )[1]
        )
        FP_distances = pd.DataFrame(
            pd.DataFrame(
                self.model.model.predict_proba(X_set_dict["FP"]), index=X_set_dict["FP"].index
            )[1]
        )
        TN_distances = pd.DataFrame(
            pd.DataFrame(
                self.model.model.predict_proba(X_set_dict["TN"]), index=X_set_dict["TN"].index
            )[1]
        )
        FN_distances = 1 - pd.DataFrame(
            pd.DataFrame(
                self.model.model.predict_proba(X_set_dict["FN"]), index=X_set_dict["FN"].index
            )[1]
        )
        samples_distances = pd.concat(
            [TP_distances, FP_distances, TN_distances, FN_distances], axis=0
        ).sort_values(by=[1])
        return samples_distances

    @default_data_fill
    def plot_confusion_matrix(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        size: Optional[Sequence[int]] = None,
    ) -> go.Figure:
        """Generates and plots a confusion matrix on the given data.

        Args:
            X: Input Dataset.
            y: Input target.
            size: Tuple of width, height of the plot. None for auto-sizing.

        Returns:
            The plotly figure containing the confusion matrix plot
        """
        assert X is not None
        assert y is not None

        z = metrics.confusion_matrix(y, self.model.predict(X))[::-1]
        z_prop = ((metrics.confusion_matrix(y, self.model.predict(X)) / len(y)) * 100).astype(int)[
            ::-1
        ]
        cm_x = [str(x) for x in sorted(np.unique(y))]
        cm_y = cm_x.copy()
        z_text = [[str(y) for y in x] for x in z]
        z_prop_text = [[f"{str(y)}%" for y in x] for x in z_prop]
        final_z = [[x + "(" + y + ")" for x, y in zip(X, Y)] for X, Y in zip(z_text, z_prop_text)]
        cm = go.Heatmap(
            z=z,
            x=cm_x,
            y=cm_y,
            text=final_z,
            texttemplate="%{text}",
            colorscale="blues",
            showscale=False,
        )
        fig = go.Figure(data=cm)
        if size is not None:
            fig.update_layout(autosize=False, width=size[0], height=size[1])
        fig.update_xaxes(title_text="Predicted")
        fig.update_yaxes(title_text="Ground Truth")
        fig.update_layout(title="Confusion Matrix")
        return fig

    @default_data_fill
    def plot_roc_curve(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        size: Optional[Sequence[int]] = None,
        main_fig: bool = False,
    ) -> go.Figure:
        """Plots the ROC AUC Curve of the given data.

        Args:
            X: Input Dataset.
            y: Input target.
            size: Tuple of width, height of the plot. None for auto-sizing.
            main_fig: Flag used to signify wether the plot is part of the main interpreter figure
                or not.

        Returns:
            A plotly figure of the roc auc curve
        """
        assert X is not None
        assert y is not None

        ROC_CURVE_TOOLTIP = "<br>".join(
            textwrap.wrap(
                """ROC-AUC Plot, shows the relationship between the true postive rate and false
                            positive rate along different thresholds.""",
                width=40,
            )
        )
        roc_curve_tooltip = self.generate_tooltip([0, 1], ROC_CURVE_TOOLTIP)
        y_true = y
        y_proba = self.model.model.predict_proba(X)[:, 1]

        fpr, tpr, _ = metrics.roc_curve(y_true, y_proba, pos_label=1)
        # df = pd.DataFrame({"Actual": y_true, "Probabilities": y_proba})
        area = px.area(
            x=fpr,
            y=tpr,
            title=f"ROC Curve (AUC={auc(fpr, tpr):.4f})",
            labels=dict(x="False Positive Rate", y="True Positive Rate"),
        )
        line = px.line(x=[0, 1], y=[0, 1])
        if not main_fig:
            fig = go.Figure(data=area.data + line.data)
            if size is not None:
                fig.update_layout(autosize=False, width=size[0], height=size[1])
        else:
            fig = go.Figure(data=area.data + line.data + roc_curve_tooltip.data)
        fig.update_yaxes(title_text="TPR", scaleanchor="x", scaleratio=1)
        fig.update_xaxes(title_text="FPR", constrain="domain")
        fig.update_layout(title=f"ROC Curve (AUC={auc(fpr, tpr):.4f})")
        return fig

    @default_data_fill
    def plot_fpr_tpr_curve(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        size: Optional[Tuple[int, int]] = None,
        main_fig: bool = False,
    ) -> go.Figure:
        """Plots the fpr tpr curve of the given data.

        Args:
            X: Input Dataset.
            y: Input target.
            size: Tuple of width, height of the plot. None for auto-sizing.
            main_fig: Flag used to signify wether the plot is part of the main interpreter figure
                or not.

        Returns:
            A plotly figure of the fpr tpr curve.
        """
        assert X is not None
        assert y is not None

        FPR_TPR_TOOLTIP = "<br>".join(
            textwrap.wrap(
                """FPR TPR plot, shows the True Positive Rate (red) and False Positive Rate(blue)
                    as the classifiers decision threshold increases""",
                width=40,
            )
        )

        fpr_tpr_tooltip = self.generate_tooltip([1, 1], FPR_TPR_TOOLTIP)
        y_true = y
        y_proba = self.model.model.predict_proba(X)[:, 1]

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_proba, pos_label=1)
        fpr = fpr[1:]
        tpr = tpr[1:]
        thresholds = thresholds[1:]
        df = pd.DataFrame({"FPR": fpr, "TPR": tpr}, index=thresholds)
        df.index.name = "Thresholds"
        df.columns.name = "Rate"
        fig = px.line(df, title="TPR and FPR at every threshold")
        fig.update_yaxes(title_text="Value")
        if size is not None:
            fig.update_layout(autosize=False, width=size[0], height=size[1])
        if main_fig:
            fig = go.Figure(fig.data + fpr_tpr_tooltip.data)
        return fig

    @default_data_fill
    def get_probability_plot(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        main_fig: bool = False,
    ) -> go.Figure:
        """Plots the models label probability distirbution of the given data.

        Args:
            X: Input Dataset.
            y: Input target.
            main_fig: Flag used to signify wether the plot is part of the main interpreter figure
                or not.

        Returns:
            A plotly figure of the label probability distribution.
        """
        assert X is not None
        assert y is not None

        PROBA_HIST_TOOLTIP = "<br>".join(
            textwrap.wrap(
                """Probability Density plot. Show the count of predictions from each class
                    with their respective probabilities the model probided. The more red
                    and blue are seperated the better the model classifies """,
                width=40,
            )
        )

        y_true = y
        y_proba = self.model.model.predict_proba(X)[:, 1]
        df = pd.DataFrame({"Actual": y_true, "Probabilities": y_proba})
        bins, edges = np.histogram(df["Probabilities"], bins=25)
        fig = px.histogram(
            df,
            "Probabilities",
            color="Actual",
            nbins=25,
            barmode="overlay",
            labels=dict(color="True Labels", x="Score"),
        )
        proba_hist_tooltip = self.generate_tooltip([edges[-1], bins.max() + 1], PROBA_HIST_TOOLTIP)
        if main_fig:
            fig = go.Figure(fig.data + proba_hist_tooltip.data)
        fig.update_yaxes(title_text="Density")
        fig.update_xaxes(title_text="Probability")
        fig.update_layout(title="Label distribution over classifier probabilities")
        return fig

    def __get_fpr_and_tpr(
        self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fpr = []
        tpr = []
        thresholds = np.around(np.arange(0, 1.01, 0.01), 2)
        pred_proba = model.predict_proba(X)[:, 1]
        for thresh in thresholds:
            tn, fp, fn, tp = metrics.confusion_matrix(y, pred_proba > thresh).ravel()
            fpr.append(fp / (fp + tn))
            tpr.append(tp / (tp + fn))
        return thresholds, np.array(fpr), np.array(tpr)

    def pad_thresholds(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        thresholds: np.ndarray,
        fpr: Sequence[float],
        tpr: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pad thresholds.

        Args:
            model: The model.
            X: The feature data.
            y: The target data.
            thresholds: Array of thresholds.
            fpr: Array of fpr values.
            tpr: Array of tpr values.

        Returns:
            A 3-tuple containing the thresholds, fpr, and tpr arrays.
        """
        base_thresh, base_fpr, base_tpr = self.__get_fpr_and_tpr(model, X, y)
        for i, thresh in enumerate(thresholds):
            insert_idx = np.searchsorted(base_thresh, thresh)
            base_thresh = np.insert(base_thresh, insert_idx, thresh)
            base_fpr = np.insert(base_fpr, insert_idx, fpr[i])
            base_tpr = np.insert(base_tpr, insert_idx, tpr[i])
        return base_thresh, base_fpr, base_tpr

    def get_meaningful_thresholds(
        self, fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray, k: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Used internally in main_fig to reduce the number of thresholds when it's too high.
        Gets the most "meaningful" thresholds to display in the main fig, meaningfulness
        being the difference between the fpr and tpr in consecutive threshold pairs.

        Args:
            fpr: Array of fpr values.
            tpr: Array of tpr values.
            thresholds: Array of thresholds.
            k: k + 100 thresholds will be returned by the function.

        Returns:
            A slice of thresholds, fpr and tpr representing the k most meaningful thresholds
        """
        largest_diffs = np.diff(fpr) + np.diff(tpr)
        if len(largest_diffs) < k:
            return fpr, tpr, thresholds
        else:
            top_jump_indices = np.argpartition(largest_diffs, len(largest_diffs) - k)[-k:]
            top_jump_indices = np.sort(top_jump_indices)

        meaningful_fpr = fpr[top_jump_indices]
        meaningful_tpr = tpr[top_jump_indices]
        meaningful_thresholds = thresholds[top_jump_indices]

        return meaningful_fpr, meaningful_tpr, meaningful_thresholds

    @default_data_fill
    def main_fig(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        feat_imp_method: FeatureImportanceMethod = FeatureImportanceMethod.Shap,
        n_feats: int = 10,
        force_shap: bool = False,
        size: Optional[Tuple[int, int]] = (900, 800),
    ) -> go.Figure:
        """Prepares and displays the main interpreter figure for the binary classification case.

        Includes:
        1. Confusion Matix
        2. ROC AUC Curve
        3. FPR TPR Plot
        4. Classifier prediction probability density plot
        5. Feature Importance Plot
        6. A slider for changing the classifiers threshold and updating the graphs accordingly

        The function is extensively documented in the comments.

        Args:
            X: Input Dataset.
            y: Input target.
            feat_imp_method: Used to control the feature importance method used in the feature
                importance plot.
            n_feats: Number of features to display in the feature importance plot.
            force_shap: bool for whether to force shap based feature importance for models with
                slow SHAP runtimes.
            size: Tuple of width, height of the plot. None for auto-sizing.

        Returns:
            A plotly figure containing the final plot
        """
        assert X is not None
        assert y is not None

        pred_proba = self.model.model.predict_proba(X)[:, 1]
        assert self.pos_class is not None
        fpr, tpr, thresholds = metrics.roc_curve(
            y, pred_proba, pos_label=self.pos_class["pos_class"]
        )

        fpr, tpr, thresholds = self.get_meaningful_thresholds(fpr, tpr, thresholds)
        thresholds, fpr, tpr = self.pad_thresholds(self.model.model, X, y, thresholds, fpr, tpr)

        fpr = fpr[:-1]
        tpr = tpr[:-1]
        thresholds = thresholds[:-1]

        # mid_point = np.searchsorted(thresholds, 0.500001)
        mid_point = np.searchsorted(thresholds, 0.49999999)
        # mid_point = len(thresholds) // 2
        # GENERALISE POS LABEL
        fig = make_subplots(
            rows=3,
            cols=3,
            vertical_spacing=0.5 / 3,
            subplot_titles=(
                "Confusion Matrix",
                f"ROC Curve (AUC = {np.around(roc_auc_score(y, pred_proba), 2)})",
                "FPR TPR Plot",
                "Probability Density Plot",
                "Feature Importance Plot",
            ),
            specs=[[{}, {}, {}], [{"colspan": 3}, None, None], [{"colspan": 3}, None, None]],
        )
        # Here we generate all states for the threshold slider, notice initially all the plots'
        # visibility is set to False, the slider works by selectively setting the visibility
        # of each graph to True of False
        for i, thresh in enumerate(thresholds):
            # Confusion matrix
            z = metrics.confusion_matrix(y, pred_proba > thresh)[::-1]
            z_prop = (
                (metrics.confusion_matrix(y, pred_proba > thresh)[::-1] / len(y)) * 100
            ).astype(int)
            cm_x = [str(x) for x in sorted(np.unique(y))]
            cm_y = cm_x.copy()[::-1]
            z_text = [[str(y) for y in x] for x in z]
            z_prop_text = [[f"{str(y)}%" for y in x] for x in z_prop]
            final_z = [
                [x + "(" + y + ")" for x, y in zip(X, Y)] for X, Y in zip(z_text, z_prop_text)
            ]
            fig.append_trace(
                go.Heatmap(
                    z=z,
                    x=cm_x,
                    y=cm_y,
                    visible=False,
                    text=final_z,
                    texttemplate="%{text}",
                    colorscale="blues",
                    showscale=False,
                ),
                row=1,
                col=1,
            )

            # Point on ROC AUC Curve
            fig.append_trace(
                go.Scatter(x=[fpr[i]], y=[tpr[i]], visible=False, marker=dict(color="red", size=5)),
                row=1,
                col=2,
            )

            # Vertical line on fpr tpr curve
            fig.append_trace(
                go.Scatter(
                    x=[thresh, thresh],
                    y=[0, 1],
                    mode="lines",
                    visible=False,
                    opacity=0.5,
                    line=dict(color="black", dash="dash"),
                ),
                row=1,
                col=3,
            )

        # Like before, define the initial slider points as visible
        fig.data[mid_point * 3].visible = True
        fig.data[(mid_point * 3) - 1].visible = True
        fig.data[(mid_point * 3) - 2].visible = True

        # Plot the bases for each plot, these don't change, the slider only contorls the plots
        # that overlay them. For example the dot on the ROC AUC Curve changes while the Curve
        # itself doesn't change
        for trace in self.plot_roc_curve(X, y, main_fig=True).data:
            fig.add_trace(trace, 1, 2)

        for trace in self.plot_fpr_tpr_curve(X, y, main_fig=True).data:
            fig.add_trace(trace, 1, 3)

        for trace in self.get_probability_plot(X, y, main_fig=True).data:
            fig.add_trace(trace, 2, 1)

        for trace in self.plot_feature_importance(
            X, y, method=feat_imp_method, n_view=n_feats, main_fig=True
        ).data:
            fig.add_trace(trace, 3, 1)

        # Here we generate all possible steps for each threshold
        steps = []
        thresh_idx = 0

        # Used to get the plot title with related metrics (needs to be in one line for plotting
        # reasons)
        def get_title(thresh: Union[float, int]) -> str:
            return (
                f"Accuracy: {str(np.around(accuracy_score(y, pred_proba > thresh), 2))} | "
                f"Precision: {str(np.around(precision_score(y, pred_proba > thresh), 2))} | "
                f"Recall: {str(np.around(recall_score(y, pred_proba > thresh), 2))} | "
                f"F1: {str(np.around(f1_score(y, pred_proba > thresh), 2))}"
            )

        # Final 13 are the extra plots (for example tooltips, ROC AUC Curve dots etc..)
        for i in range(len(fig.data) - 13):
            # Generate a step: Visibility of everything is set to false except the relevant plots
            step: Dict[str, Any] = dict(
                method="update",
                args=[
                    {"visible": [False] * len(fig.data)},
                    {"title": get_title(thresholds[thresh_idx])},
                ],
                # Info to show next to slider
                label=f"{np.around(thresholds[thresh_idx], 5):.5f}",
            )
            if not i % 3:
                thresh_idx += 1
            # everything after i is the 3 relevant plot (cm, roc auc dot, fpr tpr line).
            step["args"][0]["visible"][i] = True
            step["args"][0]["visible"][i + 1] = True
            step["args"][0]["visible"][i + 2] = True
            # Same explanation as -13: These have to be loaded for every thresh
            # (for example tooltips).
            step["args"][0]["visible"][-11] = True
            step["args"][0]["visible"][-10] = True
            step["args"][0]["visible"][-9] = True
            step["args"][0]["visible"][-8] = True
            step["args"][0]["visible"][-7] = True
            step["args"][0]["visible"][-6] = True
            step["args"][0]["visible"][-5] = True
            step["args"][0]["visible"][-4] = True
            step["args"][0]["visible"][-3] = True
            step["args"][0]["visible"][-2] = True
            step["args"][0]["visible"][-1] = True

            steps.append(step)

        # Defind property of slider
        # active = (len(fig.data) - 13) // 2,
        sliders = [
            dict(
                # Initial active state
                active=(mid_point * 3) - 1,
                steps=steps,
                currentvalue={"prefix": "Threshold: ", "offset": 2, "font": {"color": "black"}},
                y=0.77,
                tickcolor="black",
                font=dict(color="black"),
                ticklen=5,  # adjust this value
                minorticklen=0,  # adjust this value
                tickwidth=2,  # adjust this value
            )
        ]

        fig.update_layout(sliders=sliders)
        fig.update_layout(title=get_title(thresholds[mid_point]))
        fig.update_xaxes(title_text="Predicted", row=1, col=1)
        fig.update_xaxes(title_text="FPR", row=1, col=2)
        fig.update_xaxes(title_text="Threshold", row=1, col=3)
        fig.update_xaxes(title_text="Probability", row=2, col=1)
        fig.update_xaxes(title_text="Feature Names", row=3, col=1)

        fig.update_yaxes(title_text="Ground Truth", row=1, col=1)
        fig.update_yaxes(title_text="TPR", row=1, col=2)
        fig.update_yaxes(title_text="Value", row=1, col=3)
        fig.update_yaxes(title_text="Density", row=2, col=1)
        fig.update_yaxes(title_text="Importance", row=3, col=1)

        fig.update_yaxes(title_standoff=0.01)
        fig.update_xaxes(title_standoff=0.1)

        fig.update_layout(title_x=0.5)
        if size is not None:
            fig.update_layout(height=size[0], width=size[1])
        fig.update_coloraxes(showscale=False)
        fig.update_traces(showlegend=False)
        fig.update(layout_showlegend=False)
        fig.update_coloraxes(showscale=False)
        labels = fig["data"][-2]["x"]
        labels_2 = fig["data"][-1]["x"]

        new_labels = [
            "<br>".join(label[i : i + 30] for i in range(0, len(label), 30)) for label in labels
        ]
        new_labels_2 = [
            "<br>".join(label[i : i + 30] for i in range(0, len(label), 30)) for label in labels_2
        ]

        fig["data"][-2]["x"] = new_labels
        fig["data"][-1]["x"] = new_labels_2

        return fig


class TabularInterpreterRegression(TabularInterpreterBase):
    """TabularInterpreter for regression problems (inherits from Base)."""

    def __init__(
        self,
        model: PandasModelManagerBase,
        data_manager: DataManagerBase,
        opt_metric: MetricManagerBase,
        pos_class: Optional[Dict[str, Any]] = None,
        verbose: Optional[bool] = True,
        enable_shap: Optional[bool] = True,
        shap_timeout: int = 120,
    ) -> None:
        """Intializes the TabularInterpreterRegression class.

        Args:
            model: The interface class instance for managing a model.
            data_manager: The interface class instance for managing data.
            opt_metric: The interface class instance for managing metrics.
            pos_class: Positive class for classification problems.
            verbose: Whether to output details.
            enable_shap: Whether shap operations are available in the interp instance.
            shap_timeout: The timeout for getting the shap values in seconds.
        """
        super().__init__(
            model, data_manager, opt_metric, pos_class, verbose, enable_shap, shap_timeout
        )
        self.predict_proba_support: bool = False
        self.coeffs_support: bool = self.check_coeffs()

    def check_coeffs(self) -> bool:
        """Check whether model has a "coef_" method.

        Returns:
            True if the model has a "coef_" method. False otherwise.
        """
        try:
            _ = self.model.model.coef_
            return True
        except Exception:
            return False

    @default_data_fill
    def plot_regression_results(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> go.Figure:
        """Plots the actual vs predicted plot.

        Args:
            X: Input Dataset.
            y: Input target.
            size: Tuple of width, height of the plot. None for auto-sizing.

        Returns:
            A plotly figure containing the final plot.
        """
        assert X is not None
        assert y is not None

        df = pd.DataFrame({"Actual": y, "Predicted": self.model.predict(X)})
        fig = px.scatter(
            df,
            x="Actual",
            y="Predicted",
            marginal_x="histogram",
            marginal_y="histogram",
            title="Predicted Values vs Actual Results",
            trendline="ols",
        )
        fig.update_traces(histnorm="probability", selector={"type": "histogram"})
        fig.add_shape(
            type="line", line=dict(dash="dash"), x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max()
        )
        if size is not None:
            fig.update_layout(autosize=False, width=size[0], height=size[1])

        return fig

    @default_data_fill
    def plot_residuals(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> go.Figure:
        """Plots the residuals of the models prediction plot.

        Args:
            X: Input Dataset.
            y: Input target.
            size: Tuple of width, height of the plot. None for auto-sizing.

        Returns:
            A plotly figure containing the final plot.
        """
        assert X is not None
        assert y is not None

        df = pd.DataFrame({"Actual": y, "Predicted": self.model.predict(X)})
        df["residual"] = df["Predicted"] - df["Actual"]

        fig = px.scatter(
            df,
            x="Predicted",
            y="residual",
            marginal_y="violin",
            trendline="ols",
            title="Residual plot",
        )
        if size is not None:
            fig.update_layout(autosize=False, width=size[0], height=size[1])

        return fig

    @default_data_fill
    def plot_coeffs(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> go.Figure:
        """Plots a coefficient value plot.

        Args:
            X: Input Dataset.
            y: Input target.
            size: Tuple of width, height of the plot. None for auto-sizing.

        Returns:
            A plotly figure containing the final plot
        """
        assert X is not None
        assert y is not None

        try:
            coeffs = self.model.model.coef_
        except Exception:
            raise RuntimeError(
                f"Could not find coefficients for the model being used {self.model.model}"
            )
        cols = []
        vals = []
        for col, val in sorted(zip(X.columns, coeffs), key=lambda x: x[1]):
            if val == 0:
                continue
            cols.append(col)
            vals.append(val)
        df = pd.DataFrame(
            {"names": cols, "data": vals, "color": ["pos" if val > 0 else "neg" for val in vals]}
        )
        fig = px.bar(df, x="names", y="data", color="color", title="Coefficient value plot")
        if size is not None:
            fig.update_layout(autosize=False, width=size[0], height=size[1])
        return fig

    @default_data_fill
    def main_fig(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        ret_fig: bool = True,
        size: Optional[Tuple[int, int]] = (900, 800),
    ) -> Optional[go.Figure]:
        """Plots the main figure for Regressions problems.

        It contains:
            1. An Actual vs Predicted plot
            2. A predictions residuals plot

        Args:
            X: Input Dataset.
            y: Input target.
            ret_fig: Whether to return a figure object.
            size: Tuple of width, height of the plot. None for auto-sizing.

        Returns:
            If `ret_fig` is True, a plotly figure containing the final plot. Otherwise None.
        """
        assert X is not None
        assert y is not None

        if not ret_fig:
            fig1 = self.plot_regression_results(X, y)
            fig1.show()
            fig2 = self.plot_residuals()
            fig2.show()
            if self.coeffs_support:
                fig3 = self.plot_coeffs(X, y)
            else:
                fig3 = self.plot_feature_importance(X, y, main_fig=False)
            fig3.show()
            return None
        elif ret_fig:
            fig = make_subplots(
                rows=3,
                cols=1,
                subplot_titles=("Actual vs Predicted plot", "Residual Plot", "Feature Importance"),
                specs=[[{}], [{}], [{}]],
            )
            for trace in self.plot_regression_results(X, y).data:
                fig.add_trace(trace, row=1, col=1)

            for trace in self.plot_residuals(X, y).data:
                fig.add_trace(trace, row=2, col=1)

            if self.coeffs_support:
                for trace in self.plot_coeffs(X, y).data:
                    fig.add_trace(trace, row=3, col=1)
            else:
                for trace in self.plot_feature_importance(X, y, main_fig=True).data:
                    fig.add_trace(trace, row=3, col=1)

            fig.update_xaxes(title_text="Predicted", row=1, col=1)
            fig.update_xaxes(title_text="Predicted", row=2, col=1)
            fig.update_xaxes(title_text="Feature Names", row=3, col=1)

            fig.update_yaxes(title_text="Actual", row=1, col=1)
            fig.update_yaxes(title_text="Residual", row=2, col=1)
            fig.update_yaxes(title_text="Importance", row=3, col=1)

            if size is not None:
                fig.update_layout(height=size[0], width=size[1])
            fig.update_coloraxes(showscale=False)
            fig.update_layout(showlegend=False)

            return fig

    @default_data_fill
    def results(
        self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Returns a dataframe containing common metrics.

        For regression tasks:
        - Mean Absolute Error
        - Mean Squared Error
        - Median Absolute Error
        - R^2

        Args:
            X: Input Dataset.
            y: Input target.

        Returns:
            DataFrame containing common regression task metrics.
        """
        assert X is not None
        assert y is not None

        y_true = y
        y_pred = self.model.predict(X)

        # mean_absolute_error = metrics.mean_absolute_error(y_pred, y_true)
        mse = metrics.mean_squared_error(y_pred, y_true)
        # median_absolute_error = metrics.median_absolute_error(y_pred, y_true)
        r2 = metrics.r2_score(y_pred, y_true)

        results = pd.DataFrame(
            [
                {
                    "R2": round(r2, 4),
                    "Mean Squared Error": round(mse, 4),
                    "Root Mean Squared Error": round(np.sqrt(mse), 4),
                }
            ]
        )

        return results

    @default_data_fill
    def print_report(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None) -> None:
        """Prints common regression metrics to the console.

        Args:
            X: Input Dataset.
            y: Input target.
        """
        print(self.results(X=None, y=None))

    @default_data_fill
    def get_predictions_distance(
        self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Returns a DataFrame of each target value, it's predicted value and the model's error.

        Args:
            X: Input Dataset.
            y: Input target.

        Returns:
            DataFrame containing the target, predicted target and error.
        """
        assert X is not None
        assert y is not None

        y_pred = self.model.predict(X)

        predictions = pd.DataFrame(columns=["Actual Target", "Predicted Target", "Error"])
        predictions["Actual Target"] = y
        predictions["Predicted Target"] = y_pred
        predictions["Error"] = predictions["Predicted Target"] - predictions["Actual Target"]

        return predictions

    @default_data_fill
    def get_worst_predictions(
        self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None, samples_num: int = 10
    ) -> pd.DataFrame:
        """Returns the models prediction that were farthest away from the actual value.

        Args:
            X: Input Dataset.
            y: Input target.
            samples_num: The number of worst predictions to return.

        Returns:
            DataFrame with the model's worst `samples_num` predictions.
        """
        assert X is not None
        assert y is not None

        return self.get_predictions_distance(X, y)[:samples_num]

    @default_data_fill
    def get_best_predictions(
        self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None, samples_num: int = 10
    ) -> pd.DataFrame:
        """Returns the models prediction that were closest to the actual value.

        Args:
            X: Input Dataset.
            y: Input target.
            samples_num: The number of best predictions to return.

        Returns:
            DataFrame with the model's best `samples_num` predictions.
        """
        assert X is not None
        assert y is not None

        return self.get_predictions_distance(X, y)[-samples_num:]


class TabularInterpreter:
    """Selects the appropriate subclass of `TabularInterpreterBase` for instantiation."""

    # Work around mypy warning using 'type: ignore':
    # Incompatible return type for "__new__" (returns "TabularInterpreterBase", but must return
    # a subtype of "TabularInterpreter")  [misc]
    def __new__(  # type: ignore [misc]
        cls,
        prediction_type: Union[str, PredictionType],
        model: PandasModelManagerBase,
        model_data: DataManagerBase,
        opt_metric: MetricManagerBase,
        pos_class: Optional[Dict[str, int]] = None,
        verbose: Optional[bool] = True,
        enable_shap: Optional[bool] = True,
        shap_timeout: int = 120,
    ) -> TabularInterpreterBase:
        """This function returns the appropriate interpreter subclass based on prediction_type.

        Args:
            prediction_type: The type of prediction, e.g. "classification" or "regression".
            model: The interface class instance for managing a model.
            model_data: The interface class instance for managing data.
            opt_metric: The interface class instance for managing metrics.
            pos_class: Positive class for classification problems.

        Returns:
            An instance of the appropriate interpreter subclass.
        """
        if isinstance(prediction_type, str):
            prediction_type = PredictionType.from_value(prediction_type)

        if prediction_type == PredictionType.BinaryClassification:
            cls_: Type[TabularInterpreterBase] = TabularInterpreterBinaryClassification
        elif prediction_type == PredictionType.MulticlassClassification:
            cls_ = TabularInterpreterClassification
        elif prediction_type == PredictionType.Regression:
            cls_ = TabularInterpreterRegression

        instance = object.__new__(cls_)
        if not issubclass(cls_, cls):
            # Work around mypy warning using 'type: ignore':
            # Accessing "__init__" on an instance is unsound, since instance.__init__ could be
            # from an incompatible subclass  [misc]
            instance.__init__(  # type: ignore [misc]
                model,
                model_data,
                opt_metric,
                pos_class,
                verbose,
                enable_shap,
                shap_timeout,
            )

        return instance
