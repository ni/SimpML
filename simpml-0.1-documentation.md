# SimpML 0.1 documentation

```
%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path
cwd = Path.cwd()
ROOT_PATH = str(cwd.parent.parent.parent.parent)

sys.path.append(ROOT_PATH)

from simpml.tabular.all import *
np.random.seed(0)
%matplotlib inline
```

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAWCAYAAAA1vze2AAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAdxJREFUeNq0Vt1Rg0AQJjcpgBJiBWIFkgoMFYhPPAIVECogPuYpdJBYgXQQrMCUkA50V7+d2ZwXuXPGm9khHLu3f9+3l1nkWNvtNqfHLgpfQ1EUS3tz5nAQ0+NIsiAZSc6eDlI8M3J00B/mDuUKDk6kfOebAgW3pkdD0pFcODGW4gKKvOrAUm04MA4QDt1OEIXU9hDigfS5rC1eS5T90gltck1Xrizo257kgySZcNRzgCSxCvgiE9nckPJo2b/B2AcEkk2OwL8bD8gmOKR1GPbaCUqxEgTq0tLvgb6zfo7+DgYGkkWL2tqLDV4RSITfbHPPfJKIrWz4nJQTMPAWA7IbD6imcNaDeDfgk+4No+wZr40BL3g9eQJJCFqRQ54KiSt72lsLpE3o3MCBSxDuq4yOckU2hKXRuwBH3OyMR4g1UpyTYw6mlmBqNdUXRM1NfyF5EPI6JkcpIDBIX8jX6DR/6ckAZJ0wEAdLR8DEk6OfC1Pp8BKo6TQIwPJbvJ6toK5lmuvJoRtfK6Ym1iRYIarRo2UyYHvRN5qpakR3yoizWrouoyuXXQqI185LCw07op5ZyCRGL99h24InP0e9xdQukEKVmhzrqZuRIfwISB//cP3Wk3f8f/yR+BRgAHu00HjLcEQBAAAAAElFTkSuQmCC)

### Overview[](broken-reference)

This notebook presents a guide to using SimpML, with specific focus on tabular data. It demonstrates various use cases including binary classification, multi-class classification, regression, and root-cause analysis.

The key steps covered in this notebook include:

1. Data Manager: Preprocess and manipulate the data, including pipeline construction, data distribution for training and testing, and more.
2. Experiment Manager: Easily run experiments with multiple models and evaluate their performance.
3. Interpreter: Gain insights into models’ behavior, feature importance, and detect issues.
4. Inference: Make predictions using selected models during inference.

Let’s dive into each use case and see how SimpML simplifies the machine learning process.

### Binary Classification[](broken-reference)

#### Data Manager[](broken-reference)

We will demonstrate this use case using the Titanic binary classification dataset.

Here are the key steps:

* Data Loading: We create a SupervisedTabularDataManager instance, specifying the dataset file path, target variable (‘Survived’), and prediction type as binary classification.
* Preprocessing: SimpML’s data management capabilities are leveraged to preprocess the data. The build\_pipeline method is called to construct a pipeline for data preprocessing, including dropping unnecessary columns such as ‘PassengerId’.

Suppose we want other split values, we can easily initialize the splitter manually:

```
data_manager = SupervisedTabularDataManager(data = DataSet.load_titanic_dataset(),
                                            target = 'Survived',
                                            prediction_type = PredictionType.BinaryClassification,
                                            splitter = RandomSplitter(split_sets = {Dataset.Train: 0.8, Dataset.Valid: 0.2}, target = 'Survived'))
```

```
data_manager.splitter.split_sets
```

```
{<Dataset.Train: 'Train'>: 0.8, <Dataset.Valid: 'Valid'>: 0.2}
```

By using the build\_pipeline method, SimpML provides a convenient and customizable way to build a preprocessing pipeline tailored for binary classification tasks. This simplifies the preliminary data processing step and lays the foundation for training and evaluation of subsequent models Let’s see how the initialization of the pipeline looks like.

```
data_manager.build_pipeline(drop_cols = ['PassengerId'])
```

```
Sklearn Pipeline:
MatchVariablesBefore (MatchVariables(missing_values='ignore')) ->
SafeDropFeaturesBefore (SafeDropFeatures(features_to_drop=['PassengerId'])) ->
NanColumnDropper (NanColumnDropper()) ->
Infinity2Nan (Infinity2Nan()) ->
MinMaxScaler (MinMaxScalerWithColumnNames()) ->
HighCardinalityDropper (HighCardinalityDropper()) ->
AddMissingIndicator (AddMissingIndicator()) ->
NumericalImputer (MeanMedianImputer()) ->
SafeCategoricalImputer (SafeCategoricalTransformer(transformer_cls=<class 'feature_engine.imputation.categorical.CategoricalImputer'>)) ->
SafeOneHotEncoder (SafeCategoricalTransformer(transformer_cls=<class 'feature_engine.encoding.one_hot.OneHotEncoder'>)) ->
RemoveSpecialJSONCharacters (RemoveSpecialJSONCharacters()) ->
SafeDropFeaturesAfter (SafeDropFeatures(features_to_drop=['PassengerId'])) ->
MatchVariablesAfter (MatchVariables(missing_values='ignore'))

Target Pipeline:
LabelEncoder (DictLabelEncoder())
```

The build\_pipeline method in SimpML’s SupervisedTabularDataManager class is responsible for constructing a machine learning pipeline for preprocessing the data such as: missing value handling, categorical variable handling, data balancing, target encoding, features removal end more.

It offers a range of capabilities to handle various preprocessing tasks and provides default values for ease of use.

The build\_pipeline method adapts to different problems by adjusting its default settings based on the “prediction\_type” argument. This provides a fitting initial setup for users. However, for maximum personalization, users can override these defaults by passing in different argument values, offering both standardization and customization in an efficient package. Let’s see how it can be done easily:

```
data_manager.build_pipeline(drop_cols = ['PassengerId'], smote = False)
```

```
Sklearn Pipeline:
MatchVariablesBefore (MatchVariables(missing_values='ignore')) ->
SafeDropFeaturesBefore (SafeDropFeatures(features_to_drop=['PassengerId'])) ->
NanColumnDropper (NanColumnDropper()) ->
Infinity2Nan (Infinity2Nan()) ->
MinMaxScaler (MinMaxScalerWithColumnNames()) ->
HighCardinalityDropper (HighCardinalityDropper()) ->
AddMissingIndicator (AddMissingIndicator()) ->
NumericalImputer (MeanMedianImputer()) ->
SafeCategoricalImputer (SafeCategoricalTransformer(transformer_cls=<class 'feature_engine.imputation.categorical.CategoricalImputer'>)) ->
SafeOneHotEncoder (SafeCategoricalTransformer(transformer_cls=<class 'feature_engine.encoding.one_hot.OneHotEncoder'>)) ->
RemoveSpecialJSONCharacters (RemoveSpecialJSONCharacters()) ->
SafeDropFeaturesAfter (SafeDropFeatures(features_to_drop=['PassengerId'])) ->
MatchVariablesAfter (MatchVariables(missing_values='ignore'))

Target Pipeline:
LabelEncoder (DictLabelEncoder())
```

#### Experiment Manager[](broken-reference)

Next, we’ll leverage SimpML’s Experiment Manager to automatically build and compare multiple models, using common evaluation metrics such as accuracy, AUC, recall, and precision.

The ExperimentManager is an advanced tool of the SimpML library that allows the user to experiment with a wide range of machine learning models and evaluate their performance based on various metrics.

To initialize the ExperimentManager, it takes two parameters:

* data\_manager: The DataManager object which encapsulates your dataset and includes information about features and targets.
* optimize\_metric: This is the metric that you want to optimize. It is of the Enum type, MetricName, and can be Accuracy, Precision, Recall, AUC, F1, etc. By default, it is set to Accuracy.

```
exp_mang = ExperimentManager(data_manager, optimize_metric = MetricName.AUC)
```

After initializing the experiment manager we can see the models available for our experiment. These models are initialized according to the type of data and the type of problem defined in the data manager.

You can easily omit certain models from the list or add custom models.

The table below describes the main features of each model, including their names, descriptions, sources and availability statuses.

```
exp_mang.display_models_pool()
```

|   | Name                      | Description      | Source | Is Available |
| - | ------------------------- | ---------------- | ------ | ------------ |
| 0 | Gradient Boosting         | Default settings | Pool   | True         |
| 1 | AdaBoost Classifier       | Default settings | Pool   | True         |
| 2 | Baseline Classification   | Default settings | Pool   | True         |
| 3 | XGBoost                   | Default settings | Pool   | True         |
| 4 | Decision Tree             | Default settings | Pool   | True         |
| 5 | Logistic Regression       | Default settings | Pool   | True         |
| 6 | LightGBM                  | Default settings | Pool   | True         |
| 7 | Random Forest             | Default settings | Pool   | True         |
| 8 | Support Vector Classifier | Default settings | Pool   | True         |

Let’s see how we can omit a certain model or models (You can send a single model or a list of models):

```
exp_mang.remove_models('Gradient Boosting')
```

|   | Name                      | Description      | Source | Is Available |
| - | ------------------------- | ---------------- | ------ | ------------ |
| 0 | AdaBoost Classifier       | Default settings | Pool   | True         |
| 1 | Baseline Classification   | Default settings | Pool   | True         |
| 2 | XGBoost                   | Default settings | Pool   | True         |
| 3 | Decision Tree             | Default settings | Pool   | True         |
| 4 | Logistic Regression       | Default settings | Pool   | True         |
| 5 | LightGBM                  | Default settings | Pool   | True         |
| 6 | Random Forest             | Default settings | Pool   | True         |
| 7 | Support Vector Classifier | Default settings | Pool   | True         |
| 8 | Gradient Boosting         | Default settings | Pool   | False        |

We see that now only three models remain available:

```
exp_mang.get_available_models_df()
```

|                           | Description      |
| ------------------------- | ---------------- |
| Baseline Classification   | Default settings |
| Logistic Regression       | Default settings |
| Support Vector Classifier | Default settings |
| AdaBoost Classifier       | Default settings |
| Decision Tree             | Default settings |
| Random Forest             | Default settings |
| XGBoost                   | Default settings |
| LightGBM                  | Default settings |

Note: There is an option to add customized models to the list of available models. To see how to do this go to the “Experiment Manager Components” page under the “Add Custom Models” section

If necessary, we can reset the list of models back to the default. This action will undo the model removals that have been made and delete custom models that have been added:

|   | Name                      | Description      | Source | Is Available |
| - | ------------------------- | ---------------- | ------ | ------------ |
| 0 | Gradient Boosting         | Default settings | Pool   | True         |
| 1 | AdaBoost Classifier       | Default settings | Pool   | True         |
| 2 | Baseline Classification   | Default settings | Pool   | True         |
| 3 | XGBoost                   | Default settings | Pool   | True         |
| 4 | Decision Tree             | Default settings | Pool   | True         |
| 5 | Logistic Regression       | Default settings | Pool   | True         |
| 6 | LightGBM                  | Default settings | Pool   | True         |
| 7 | Random Forest             | Default settings | Pool   | True         |
| 8 | Support Vector Classifier | Default settings | Pool   | True         |

Similarly, you can review the metrics that will be started by default and see who is slamming it we want to optimize:

```
exp_mang.display_metrics_pool()
```

|   | Name              | Description | Source | Is Available | Is Optimal |
| - | ----------------- | ----------- | ------ | ------------ | ---------- |
| 0 | AUC               |             | Pool   | True         | True       |
| 1 | F1                |             | Pool   | True         | False      |
| 2 | Accuracy          |             | Pool   | True         | False      |
| 3 | Recall            |             | Pool   | True         | False      |
| 4 | Precision         |             | Pool   | True         | False      |
| 5 | Balanced Accuracy |             | Pool   | True         | False      |

Once the ExperimentManager is initialized, you can use the fit\_suite function to fit multiple models on the data encapsulated in the provided DataManager object.

You can see how you can send parameters to the metrics we are going to calculate using the metrics\_kwargs dictionary. If the metric has a parameter name of a key in our dictionary, the experiment manager will calculate the metric with this parameter. In the example below we send a variable “pos\_label” that will affect the metrics of a binary classification

```
exp_mang.run_experiment(metrics_kwargs = {'pos_label': 1})
```

|   | Experiment ID        | Experiment Description | Model                     | Model Description | Data Version | Data Description | Model Params                                                                                                                       | Metric Params     | Accuracy | AUC      | Recall   | Precision | Balanced Accuracy | F1       | Run Time |
| - | -------------------- | ---------------------- | ------------------------- | ----------------- | ------------ | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ----------------- | -------- | -------- | -------- | --------- | ----------------- | -------- | -------- |
| 0 | 20240320184107\_d6d3 |                        | Baseline Classification   | Default settings  | 19067569     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {'pos\_label': 1} | 0.530726 | 0.507444 | 0.405797 | 0.394366  | 0.507444          | 0.400000 | 0:00:00  |
| 1 | 20240320184107\_d6d3 |                        | Logistic Regression       | Default settings  | 19067569     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {'pos\_label': 1} | 0.810056 | 0.794137 | 0.724638 | 0.769231  | 0.794137          | 0.746269 | 0:00:01  |
| 2 | 20240320184107\_d6d3 |                        | Support Vector Classifier | Default settings  | 19067569     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {'pos\_label': 1} | 0.804469 | 0.773386 | 0.637681 | 0.814815  | 0.773386          | 0.715447 | 0:00:00  |
| 3 | 20240320184107\_d6d3 |                        | AdaBoost Classifier       | Default settings  | 19067569     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {'pos\_label': 1} | 0.765363 | 0.757773 | 0.724638 | 0.684932  | 0.757773          | 0.704225 | 0:00:00  |
| 4 | 20240320184107\_d6d3 |                        | Decision Tree             | Default settings  | 19067569     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {'pos\_label': 1} | 0.765363 | 0.755072 | 0.710145 | 0.690141  | 0.755072          | 0.700000 | 0:00:00  |
| 5 | 20240320184107\_d6d3 |                        | Random Forest             | Default settings  | 19067569     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {'pos\_label': 1} | 0.798883 | 0.776943 | 0.681159 | 0.770492  | 0.776943          | 0.723077 | 0:00:00  |
| 6 | 20240320184107\_d6d3 |                        | Gradient Boosting         | Default settings  | 19067569     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {'pos\_label': 1} | 0.815642 | 0.790580 | 0.681159 | 0.810345  | 0.790580          | 0.740157 | 0:00:00  |
| 7 | 20240320184107\_d6d3 |                        | XGBoost                   | Default settings  | 19067569     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {'pos\_label': 1} | 0.782123 | 0.763307 | 0.681159 | 0.734375  | 0.763307          | 0.706767 | 0:00:00  |
| 8 | 20240320184107\_d6d3 |                        | LightGBM                  | Default settings  | 19067569     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {'pos\_label': 1} | 0.782123 | 0.768709 | 0.710145 | 0.720588  | 0.768709          | 0.715328 | 0:00:00  |

The fit\_suite function will return a pandas DataFrame summarizing the experiments. It includes:

* Experiment ID: An unique identifier for each run.
* Experiment Description: Experiment description that was provided in run\_experiment
* Model Description: The name of the machine learning model used.
* Description: A short description about the model.
* Data Version: The version of the data used for the experiment.
* Data Description: A short description of the data used.
* Model Params: The parameters used by the model.
* Metric Params: The parameters used for the performance metrics.
* Various metrics: The values for different metrics like Accuracy, AUC, Recall, Precision, BalancedAccuracy, F1.
* Run Time: The total time taken by each model to train and make predictions.

The get\_best\_model function can be used to retrieve the model that performed the best according to the optimize\_metric.

```
best_model = exp_mang.get_best_model()
best_model
```

```
Model: LogisticRegression(n_jobs=-1, random_state=42), Description: Default settings
```

Alternatively, we can retrieve any other trained model from the experiment manager using get\_model\_by\_name function and sending the model name and its Experiment ID:

```
exp_id = exp_mang.get_current_experiment_id()
exp_id
```

```
light_gbm = exp_mang.get_model(model_name = 'LightGBM', experiment_id = exp_id)
```

**Logger**[****](broken-reference)

In the Experiment Manager, users have the option to utilize a logger, enabling the storage of experiments in a persistent manner. They can choose from a selection of pre-defined loggers or create a custom one according to their needs. Next, we will explore how to work with the MLflow logger.

As part of Experiment Manager init, we can send the logger

```
#exp_mang = ExperimentManager(data_manager, optimize_metric = MetricName.AUC, logger = MlflowLogger('my_projrct_name'))
```

```
#exp_mang.run_experiment(experiment_description='Mlflow Demo', metrics_kwargs = {'pos_label': 1})
```

```
#exp_mang.get_best_model()
```

Our experimnt is now saved in MLFlow with parameters, metrics & tags. We have summary row representing best model with our data manager & model pipeline saved as artifacts&#x20;

In order to reperduce run from MlFlow Experiment back to notebook, you can use the following util. By default you will get pointer to latest run but you can send run\_name & tags to get different one.

```
#mlflow_run_handler = exp_mang.logger.get_run_handler()
```

```
#mlflow_run_handler.run_name
```

```
#mlflow_run_handler.load_model()
```

```
#mlflow_run_handler.load_model_pipeline()
```

```
#mlflow_run_handler.load_data_manager()
```

```
#exp_mang = mlflow_run_handler.load_experiment_manager()
#exp_mang.run_experiment(metrics_kwargs = {'pos_label': 1})
```

#### Interpreter[](broken-reference)

In the world of machine learning and data science, model interpretability and understanding is critical. It’s not enough to have a model that simply makes accurate predictions; we also need to understand why it makes those decisions and if there are any inherent flaws or biases in the model. This is where the Interpreter comes in.

Once the models are trained, SimpML provides an intuitive way to interpret the results. We’ll create a Binary Classification Interpreter to analyze the model’s performance on the dataset.

```
interp = TabularInterpreterBinaryClassification(model = light_gbm,
                                                data_manager = data_manager,
                                                opt_metric = exp_mang.opt_metric,
                                                pos_class = {'pos_class' : 1})

```

```
Caught an ExplainerError: Additivity check failed in TreeExplainer! Please ensure the data matrix you passed to the explainer is the same shape that the model was trained on. If your data shape is correct then please report this on GitHub. This check failed because for one of the samples the sum of the SHAP values was -0.235552, while the model output was -1.009207. If this difference is acceptable you can set check_additivity=False to disable this check.
Running without check_additivity which might lead to inaccurate results!
```

**Shap**[****](broken-reference)

The interpreter provides an easy way to plot a variety of shape values:

```
interp.shap_manager.plot_summary_shap()
```

By default, the interpreter functions will run on the validation set, but we can easily override them like this:

```
X,y = data_manager.get_training_data()
interp.shap_manager.plot_summary_shap(X)
```

```
interp.shap_manager.plot_shap_dependence('Age')
```

**Analysis of the model predictions**[****](broken-reference)

First, we can plot the confusion matrix, also here the default is to display it on the validation set. To display it on another set we will have to send this set to the function

```
interp.plot_confusion_matrix()
```

The interp.find\_best\_threshold() method is used to find the optimal decision threshold for our model. In binary classification, this threshold is the cutoff for classifying an instance as positive (1) or negative (0).confusion matrix The optimal threshold maximizes our model’s performance based on the metric we specified in our Experiment Manager.

By default, this threshold is often set to 0.5, but it can vary depending on the data and task. Adjusting it can help balance precision and recall, or minimize false positives or negatives.

When calling this function without arguments, it uses the data and optimization metric specified when initiating our interpreter.

Now, let’s find the best threshold for our model:

```
interp.find_best_threshold()
```

We can also visually see the distribution of the probability of the model’s predictions

```
interp.get_probability_plot()
```

The plot\_roc\_curve() method is used to generate and display the Receiver Operating Characteristic (ROC) curve for your model. The ROC curve is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied. It is created by plotting the True Positive Rate (TPR) against the False Positive Rate (FPR).

The plot\_fpr\_tpr\_curve() method, which is similar to the plot\_roc\_curve() method, generates and displays the Receiver Operating Characteristic (ROC) curve for a model. The ROC curve is a graphical plot that demonstrates the diagnostic ability of a binary classifier system.

```
interp.plot_fpr_tpr_curve()
```

The plot\_feature\_importance() function is used to create a graphical representation of the importance of various features in your model. The interpreter supports two methods for calculating the importance of features: Shape and Permutation.

* Shap: The SHAP (SHapley Additive Explanations) method calculates the contribution of each feature to the prediction for each instance. It assigns each attribute an importance score for a particular prediction. A higher SHAP value means a higher contribution to the prediction. SHAP values ​​have a ‘game theoretic’ property where the sum of SHAP values ​​for all attributes equals the difference between the prediction and the baseline (expected) value.
* Permutation: Permutation feature importance is a technique used to estimate the importance of input features by calculating the increase in prediction error after changing the feature values, which breaks the relationship between the feature and the true result. The idea is that the importance of an attribute is proportional to the increase in the prediction error of the model after changing the attribute values.

Let’s see how we can get the feature importance according to each method:

```
interp.plot_feature_importance(method = FeatureImportanceMethod.Shap)
```

```
interp.plot_feature_importance(method = FeatureImportanceMethod.Permutation)
```

In the next step, we’ll assemble an interactive dashboard that includes all the plots related to our model’s prediction analysis. This platform allows you to adjust the threshold indicator, the pivot point distinguishing between normal and abnormal outcomes, and directly observe how this adjustment influences the predictions made by the model.

Let’s see how to do it:

**Mlflow logger**[****](broken-reference)

You can use mlflow run handler util to save interepret plots & figures to mlflow

```
#mlflow_run_handler.log_figure(interp.main_fig(), "main_fig.html")
```

**Insights**[****](broken-reference)

In machine learning, insights provide a deeper understanding of our models. They reveal relationships between features and outcomes, identify data issues like noise or leakage, and guide the improvement and debugging of the model. By leveraging these insights, we can enhance model performance and reliability.

First, we can get the feature importance as a data frame table to further explore it later. Let’s do it like this:

```
interp.get_feature_importance(method = FeatureImportanceMethod.Shap)
```

|     | col\_name                    | feature\_importance\_vals |
| --- | ---------------------------- | ------------------------- |
| 0   | Sex\_male                    | 0.332964                  |
| 1   | Fare                         | 0.166671                  |
| 2   | Pclass                       | 0.129707                  |
| 3   | Age                          | 0.129606                  |
| 4   | Cabin\_na                    | 0.074817                  |
| ... | ...                          | ...                       |
| 137 | Cabin\_B35                   | 0.000000                  |
| 138 | Cabin\_E67                   | 0.000000                  |
| 139 | Cabin\_C23\_\_\_C25\_\_\_C27 | 0.000000                  |
| 140 | Cabin\_E33                   | 0.000000                  |
| 141 | Embarked\_Missing            | 0.000000                  |

142 rows × 2 columns

The get\_noisy\_features() method uses SHAP to identify ‘noisy’ features - those contributing more to overfitting than to model accuracy.

This function computes correlation between SHAP values and feature values; high correlation signifies more important features while low correlation suggests overfitting potential.

The method iteratively analyzes and possibly drops noisy features if model performance improves, stopping when performance ceases to improve.

Let’s find noisy features in our data:

```
noisy_features = interp.get_noisy_features()
```

```
Bad noisy feature found: Age_na
old AUC: 0.7687 new AUC: 0.7687
              precision    recall  f1-score   support

           0       0.82      0.83      0.82       110
           1       0.72      0.71      0.72        69

    accuracy                           0.78       179
   macro avg       0.77      0.77      0.77       179
weighted avg       0.78      0.78      0.78       179

```

The leakage\_detector() method is designed to identify and locate features suspected of causing data leakage.

It achieves this by comparing the performance of a model trained on all features against one trained on only the most important feature. If the performance difference is smaller than a specified epsilon or if the second model performs better, data leakage is suspected.

```
is_leakage = interp.leakage_detector()
```

```
Suspected data leakage detected! feature Sex_male
Model results with the all the data: 0.7702040275569688 AUC
Model results with this feature only: 0.7671575437621562 AUC
The difference between the results of the models is 0.003046483794812649 < smaller than 0.05
```

**Actions**[****](broken-reference)

Finally, we can harness the insights derived from the interpreter to refine our data, subsequently initiating a new experiment to enhance our model!

Here’s how we can go about this:

In this example we discovered that we have a noisy feature, a feature that does not contribute to the performance of the model and may cause overfitting. Let’s create a new pipeline without this feature:

```
new_data_manager_without_noisy_features = data_manager.clone()
new_data_manager_without_noisy_features.set_description('Noisy Features Dropped')
new_data_manager_without_noisy_features.build_pipeline(drop_cols = ['PassengerId'] + noisy_features, smote = False)
```

```
Sklearn Pipeline:
MatchVariablesBefore (MatchVariables(missing_values='ignore')) ->
SafeDropFeaturesBefore (SafeDropFeatures(features_to_drop=['PassengerId', 'Age_na'])) ->
NanColumnDropper (NanColumnDropper()) ->
Infinity2Nan (Infinity2Nan()) ->
MinMaxScaler (MinMaxScalerWithColumnNames()) ->
HighCardinalityDropper (HighCardinalityDropper()) ->
AddMissingIndicator (AddMissingIndicator()) ->
NumericalImputer (MeanMedianImputer()) ->
SafeCategoricalImputer (SafeCategoricalTransformer(transformer_cls=<class 'feature_engine.imputation.categorical.CategoricalImputer'>)) ->
SafeOneHotEncoder (SafeCategoricalTransformer(transformer_cls=<class 'feature_engine.encoding.one_hot.OneHotEncoder'>)) ->
RemoveSpecialJSONCharacters (RemoveSpecialJSONCharacters()) ->
SafeDropFeaturesAfter (SafeDropFeatures(features_to_drop=['PassengerId', 'Age_na'])) ->
MatchVariablesAfter (MatchVariables(missing_values='ignore'))

Target Pipeline:
LabelEncoder (DictLabelEncoder())
```

We will load the new version of the data back to the experiment manager:

```
exp_mang.set_new_data(new_data_manager_without_noisy_features)
```

And we will run a new experiment on this data:

```
exp_mang.run_experiment()
```

|    | Experiment ID        | Experiment Description | Model                     | Model Description | Data Version | Data Description       | Model Params                                                                                                                       | Metric Params     | Accuracy | AUC      | Recall   | Precision | Balanced Accuracy | F1       | Run Time |
| -- | -------------------- | ---------------------- | ------------------------- | ----------------- | ------------ | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ----------------- | -------- | -------- | -------- | --------- | ----------------- | -------- | -------- |
| 0  | 20240320184107\_d6d3 |                        | Baseline Classification   | Default settings  | 19067569     |                        | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {'pos\_label': 1} | 0.530726 | 0.507444 | 0.405797 | 0.394366  | 0.507444          | 0.400000 | 0:00:00  |
| 1  | 20240320184107\_d6d3 |                        | Logistic Regression       | Default settings  | 19067569     |                        | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {'pos\_label': 1} | 0.810056 | 0.794137 | 0.724638 | 0.769231  | 0.794137          | 0.746269 | 0:00:01  |
| 2  | 20240320184107\_d6d3 |                        | Support Vector Classifier | Default settings  | 19067569     |                        | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {'pos\_label': 1} | 0.804469 | 0.773386 | 0.637681 | 0.814815  | 0.773386          | 0.715447 | 0:00:00  |
| 3  | 20240320184107\_d6d3 |                        | AdaBoost Classifier       | Default settings  | 19067569     |                        | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {'pos\_label': 1} | 0.765363 | 0.757773 | 0.724638 | 0.684932  | 0.757773          | 0.704225 | 0:00:00  |
| 4  | 20240320184107\_d6d3 |                        | Decision Tree             | Default settings  | 19067569     |                        | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {'pos\_label': 1} | 0.765363 | 0.755072 | 0.710145 | 0.690141  | 0.755072          | 0.700000 | 0:00:00  |
| 5  | 20240320184107\_d6d3 |                        | Random Forest             | Default settings  | 19067569     |                        | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {'pos\_label': 1} | 0.798883 | 0.776943 | 0.681159 | 0.770492  | 0.776943          | 0.723077 | 0:00:00  |
| 6  | 20240320184107\_d6d3 |                        | Gradient Boosting         | Default settings  | 19067569     |                        | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {'pos\_label': 1} | 0.815642 | 0.790580 | 0.681159 | 0.810345  | 0.790580          | 0.740157 | 0:00:00  |
| 7  | 20240320184107\_d6d3 |                        | XGBoost                   | Default settings  | 19067569     |                        | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {'pos\_label': 1} | 0.782123 | 0.763307 | 0.681159 | 0.734375  | 0.763307          | 0.706767 | 0:00:00  |
| 8  | 20240320184107\_d6d3 |                        | LightGBM                  | Default settings  | 19067569     |                        | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {'pos\_label': 1} | 0.782123 | 0.768709 | 0.710145 | 0.720588  | 0.768709          | 0.715328 | 0:00:00  |
| 9  | 20240320184243\_c3e4 |                        | Baseline Classification   | Default settings  | 6df8f01e     | Noisy Features Dropped | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}                | 0.458101 | 0.432148 | 0.318841 | 0.305556  | 0.432148          | 0.312057 | 0:00:00  |
| 10 | 20240320184243\_c3e4 |                        | Logistic Regression       | Default settings  | 6df8f01e     | Noisy Features Dropped | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}                | 0.810056 | 0.794137 | 0.724638 | 0.769231  | 0.794137          | 0.746269 | 0:00:00  |
| 11 | 20240320184243\_c3e4 |                        | Support Vector Classifier | Default settings  | 6df8f01e     | Noisy Features Dropped | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}                | 0.804469 | 0.773386 | 0.637681 | 0.814815  | 0.773386          | 0.715447 | 0:00:00  |
| 12 | 20240320184243\_c3e4 |                        | AdaBoost Classifier       | Default settings  | 6df8f01e     | Noisy Features Dropped | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}                | 0.765363 | 0.757773 | 0.724638 | 0.684932  | 0.757773          | 0.704225 | 0:00:00  |
| 13 | 20240320184243\_c3e4 |                        | Decision Tree             | Default settings  | 6df8f01e     | Noisy Features Dropped | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}                | 0.765363 | 0.755072 | 0.710145 | 0.690141  | 0.755072          | 0.700000 | 0:00:00  |
| 14 | 20240320184243\_c3e4 |                        | Random Forest             | Default settings  | 6df8f01e     | Noisy Features Dropped | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}                | 0.798883 | 0.776943 | 0.681159 | 0.770492  | 0.776943          | 0.723077 | 0:00:00  |
| 15 | 20240320184243\_c3e4 |                        | Gradient Boosting         | Default settings  | 6df8f01e     | Noisy Features Dropped | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}                | 0.815642 | 0.790580 | 0.681159 | 0.810345  | 0.790580          | 0.740157 | 0:00:00  |
| 16 | 20240320184243\_c3e4 |                        | XGBoost                   | Default settings  | 6df8f01e     | Noisy Features Dropped | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}                | 0.782123 | 0.763307 | 0.681159 | 0.734375  | 0.763307          | 0.706767 | 0:00:00  |
| 17 | 20240320184243\_c3e4 |                        | LightGBM                  | Default settings  | 6df8f01e     | Noisy Features Dropped | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}                | 0.782123 | 0.768709 | 0.710145 | 0.720588  | 0.768709          | 0.715328 | 0:00:00  |

In the new experiment we conducted we improved the performance of the model using the insights we received from the interpreter!

#### Inference[](broken-reference)

Will be supported soon

### Multi-Class Classification[](broken-reference)

In this use case, we will use the Shelter Animal Outcomes dataset for a multi-class classification task.

Here are the key steps:

* Data Loading: We instantiate a SupervisedTabularDataManager object, specifying the dataset file path, target variable (‘OutcomeType’), and prediction type as multi-class classification.
* Preprocessing: We leverage SimpML’s data management capabilities to preprocess the data. The build\_pipeline method is invoked to construct a pipeline for data preprocessing. This may include dropping unnecessary columns or transforming variables, depending on the specific requirements of the dataset. For instance, columns like ‘AnimalID’ or ‘Name’ which may not contribute to the prediction of the outcome can be dropped.

```
data_manager = SupervisedTabularDataManager(data = DataSet.load_wine_dataset(),
                                            target = 'target',
                                            prediction_type = PredictionType.MulticlassClassification,
                                           )
data_manager.build_pipeline()
```

```
Sklearn Pipeline:
MatchVariablesBefore (MatchVariables(missing_values='ignore')) ->
NanColumnDropper (NanColumnDropper()) ->
Infinity2Nan (Infinity2Nan()) ->
MinMaxScaler (MinMaxScalerWithColumnNames()) ->
AddMissingIndicator (AddMissingIndicator()) ->
NumericalImputer (MeanMedianImputer()) ->
MatchVariablesAfter (MatchVariables(missing_values='ignore'))

Target Pipeline:
LabelEncoder (DictLabelEncoder())
```

```
splitter = RandomSplitter(split_sets = {Dataset.Train: 0.6, Dataset.Valid: 0.2, Dataset.Test: 0.2}, target = 'target')
data_manager = SupervisedTabularDataManager(data = DataSet.load_wine_dataset(),
                                            target = 'target',
                                            prediction_type = PredictionType.MulticlassClassification,
                                            splitter = splitter)
```

```
data_manager.data_fetcher.get_items()
```

|     | alcohol | malic\_acid | ash  | alcalinity\_of\_ash | magnesium | total\_phenols | flavanoids | nonflavanoid\_phenols | proanthocyanins | color\_intensity | hue  | od280/od315\_of\_diluted\_wines | proline | target |
| --- | ------- | ----------- | ---- | ------------------- | --------- | -------------- | ---------- | --------------------- | --------------- | ---------------- | ---- | ------------------------------- | ------- | ------ |
| 0   | 14.23   | 1.71        | 2.43 | 15.6                | 127.0     | 2.80           | 3.06       | 0.28                  | 2.29            | 5.64             | 1.04 | 3.92                            | 1065.0  | 0      |
| 1   | 13.20   | 1.78        | 2.14 | 11.2                | 100.0     | 2.65           | 2.76       | 0.26                  | 1.28            | 4.38             | 1.05 | 3.40                            | 1050.0  | 0      |
| 2   | 13.16   | 2.36        | 2.67 | 18.6                | 101.0     | 2.80           | 3.24       | 0.30                  | 2.81            | 5.68             | 1.03 | 3.17                            | 1185.0  | 0      |
| 3   | 14.37   | 1.95        | 2.50 | 16.8                | 113.0     | 3.85           | 3.49       | 0.24                  | 2.18            | 7.80             | 0.86 | 3.45                            | 1480.0  | 0      |
| 4   | 13.24   | 2.59        | 2.87 | 21.0                | 118.0     | 2.80           | 2.69       | 0.39                  | 1.82            | 4.32             | 1.04 | 2.93                            | 735.0   | 0      |
| ... | ...     | ...         | ...  | ...                 | ...       | ...            | ...        | ...                   | ...             | ...              | ...  | ...                             | ...     | ...    |
| 173 | 13.71   | 5.65        | 2.45 | 20.5                | 95.0      | 1.68           | 0.61       | 0.52                  | 1.06            | 7.70             | 0.64 | 1.74                            | 740.0   | 2      |
| 174 | 13.40   | 3.91        | 2.48 | 23.0                | 102.0     | 1.80           | 0.75       | 0.43                  | 1.41            | 7.30             | 0.70 | 1.56                            | 750.0   | 2      |
| 175 | 13.27   | 4.28        | 2.26 | 20.0                | 120.0     | 1.59           | 0.69       | 0.43                  | 1.35            | 10.20            | 0.59 | 1.56                            | 835.0   | 2      |
| 176 | 13.17   | 2.59        | 2.37 | 20.0                | 120.0     | 1.65           | 0.68       | 0.53                  | 1.46            | 9.30             | 0.60 | 1.62                            | 840.0   | 2      |
| 177 | 14.13   | 4.10        | 2.74 | 24.5                | 96.0      | 2.05           | 0.76       | 0.56                  | 1.35            | 9.20             | 0.61 | 1.60                            | 560.0   | 2      |

178 rows × 14 columns

```
data_manager.build_pipeline()
```

```
Sklearn Pipeline:
MatchVariablesBefore (MatchVariables(missing_values='ignore')) ->
NanColumnDropper (NanColumnDropper()) ->
Infinity2Nan (Infinity2Nan()) ->
MinMaxScaler (MinMaxScalerWithColumnNames()) ->
AddMissingIndicator (AddMissingIndicator()) ->
NumericalImputer (MeanMedianImputer()) ->
MatchVariablesAfter (MatchVariables(missing_values='ignore'))

Target Pipeline:
LabelEncoder (DictLabelEncoder())
```

#### Experiment Manager[](broken-reference)

```
exp_mang = ExperimentManager(data_manager, optimize_metric = MetricName.Accuracy)
exp_mang.display_models_pool()
```

|   | Name                      | Description      | Source | Is Available |
| - | ------------------------- | ---------------- | ------ | ------------ |
| 0 | Gradient Boosting         | Default settings | Pool   | True         |
| 1 | AdaBoost Classifier       | Default settings | Pool   | True         |
| 2 | Baseline Classification   | Default settings | Pool   | True         |
| 3 | XGBoost                   | Default settings | Pool   | True         |
| 4 | Decision Tree             | Default settings | Pool   | True         |
| 5 | Logistic Regression       | Default settings | Pool   | True         |
| 6 | LightGBM                  | Default settings | Pool   | True         |
| 7 | Random Forest             | Default settings | Pool   | True         |
| 8 | Support Vector Classifier | Default settings | Pool   | True         |

```
exp_mang.run_experiment()
```

|   | Experiment ID        | Experiment Description | Model                     | Model Description | Data Version | Data Description | Model Params                                                                                                                                | Metric Params | Accuracy | Kappa    | Run Time |
| - | -------------------- | ---------------------- | ------------------------- | ----------------- | ------------ | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | -------- | -------- | -------- |
| 0 | 20240320184248\_5ddb |                        | Baseline Classification   | Default settings  | 6a54ddda     |                  | {'experiment\_manager': Prediction Type: PredictionType.MulticlassClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.485714 | 0.217391 | 0:00:00  |
| 1 | 20240320184248\_5ddb |                        | Logistic Regression       | Default settings  | 6a54ddda     |                  | {'experiment\_manager': Prediction Type: PredictionType.MulticlassClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.971429 | 0.956576 | 0:00:00  |
| 2 | 20240320184248\_5ddb |                        | Support Vector Classifier | Default settings  | 6a54ddda     |                  | {'experiment\_manager': Prediction Type: PredictionType.MulticlassClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 1.000000 | 1.000000 | 0:00:00  |
| 3 | 20240320184248\_5ddb |                        | AdaBoost Classifier       | Default settings  | 6a54ddda     |                  | {'experiment\_manager': Prediction Type: PredictionType.MulticlassClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.828571 | 0.734177 | 0:00:00  |
| 4 | 20240320184248\_5ddb |                        | Decision Tree             | Default settings  | 6a54ddda     |                  | {'experiment\_manager': Prediction Type: PredictionType.MulticlassClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.971429 | 0.956359 | 0:00:00  |
| 5 | 20240320184248\_5ddb |                        | Random Forest             | Default settings  | 6a54ddda     |                  | {'experiment\_manager': Prediction Type: PredictionType.MulticlassClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.971429 | 0.956576 | 0:00:00  |
| 6 | 20240320184248\_5ddb |                        | Gradient Boosting         | Default settings  | 6a54ddda     |                  | {'experiment\_manager': Prediction Type: PredictionType.MulticlassClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.828571 | 0.738806 | 0:00:00  |
| 7 | 20240320184248\_5ddb |                        | XGBoost                   | Default settings  | 6a54ddda     |                  | {'experiment\_manager': Prediction Type: PredictionType.MulticlassClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.942857 | 0.913473 | 0:00:00  |
| 8 | 20240320184248\_5ddb |                        | LightGBM                  | Default settings  | 6a54ddda     |                  | {'experiment\_manager': Prediction Type: PredictionType.MulticlassClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.971429 | 0.956576 | 0:00:00  |

```
best_model = exp_mang.get_best_model()
best_model
```

```
Model: SVC(probability=True, random_state=42), Description: Default settings
```

#### Interpreter[](broken-reference)

```
interp = TabularInterpreterClassification(model = exp_mang.get_model('Decision Tree', experiment_id = exp_mang.get_current_experiment_id()),
                                          data_manager = data_manager,
                                          opt_metric = exp_mang.opt_metric,
                                          pos_class = {'pos_class' : 1})
```

```
Shap manager failed with error Must pass 2-d input. shape=(106, 13, 3), SHAP will not be functional
```

```
interp.get_label_density_plot()
```

### Regression[](broken-reference)

#### Data Manager[](broken-reference)

```
data_manager = SupervisedTabularDataManager(data = DataSet.load_fetch_california_housing_dataset(),
                                            target = 'MedHouseVal',
                                            splitter = RandomSplitter(split_sets = {Dataset.Train: 0.8, Dataset.Valid: 0.2}, target = 'MedHouseVal', stratify = False),
                                            prediction_type = PredictionType.Regression)
data_manager.build_pipeline()
```

```
Sklearn Pipeline:
MatchVariablesBefore (MatchVariables(missing_values='ignore')) ->
NanColumnDropper (NanColumnDropper()) ->
Infinity2Nan (Infinity2Nan()) ->
MinMaxScaler (MinMaxScalerWithColumnNames()) ->
AddMissingIndicator (AddMissingIndicator()) ->
NumericalImputer (MeanMedianImputer()) ->
MatchVariablesAfter (MatchVariables(missing_values='ignore'))
```

#### Experiment Manager[](broken-reference)

```
exp_mang = ExperimentManager(data_manager, optimize_metric = MetricName.MSE)
exp_mang.run_experiment()
```

|   | Experiment ID        | Experiment Description | Model             | Model Description | Data Version | Data Description | Model Params                                                                                                             | Metric Params | MSE      | RMSE     | R2        | Run Time |
| - | -------------------- | ---------------------- | ----------------- | ----------------- | ------------ | ---------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------- | -------- | -------- | --------- | -------- |
| 0 | 20240320184253\_6563 |                        | Elastic Net       | Default settings  | 8e2b3505     |                  | {'experiment\_manager': Prediction Type: PredictionType.Regression, Metric: Metric: MSE. Description: , Random State: 0} | {}            | 1.310696 | 1.144856 | -0.000219 | 0:00:00  |
| 1 | 20240320184253\_6563 |                        | Lasso Lars CV     | Default settings  | 8e2b3505     |                  | {'experiment\_manager': Prediction Type: PredictionType.Regression, Metric: Metric: MSE. Description: , Random State: 0} | {}            | 0.554797 | 0.744847 | 0.576623  | 0:00:00  |
| 2 | 20240320184253\_6563 |                        | Decision Tree     | Default settings  | 8e2b3505     |                  | {'experiment\_manager': Prediction Type: PredictionType.Regression, Metric: Metric: MSE. Description: , Random State: 0} | {}            | 0.500155 | 0.707216 | 0.618322  | 0:00:00  |
| 3 | 20240320184253\_6563 |                        | Random Forest     | Default settings  | 8e2b3505     |                  | {'experiment\_manager': Prediction Type: PredictionType.Regression, Metric: Metric: MSE. Description: , Random State: 0} | {}            | 0.256335 | 0.506295 | 0.804386  | 0:00:12  |
| 4 | 20240320184253\_6563 |                        | Gradient Boosting | Default settings  | 8e2b3505     |                  | {'experiment\_manager': Prediction Type: PredictionType.Regression, Metric: Metric: MSE. Description: , Random State: 0} | {}            | 0.294538 | 0.542714 | 0.775232  | 0:00:08  |
| 5 | 20240320184253\_6563 |                        | XGBoost           | Default settings  | 8e2b3505     |                  | {'experiment\_manager': Prediction Type: PredictionType.Regression, Metric: Metric: MSE. Description: , Random State: 0} | {}            | 0.224460 | 0.473772 | 0.828710  | 0:00:01  |
| 6 | 20240320184253\_6563 |                        | LightGBM          | Default settings  | 8e2b3505     |                  | {'experiment\_manager': Prediction Type: PredictionType.Regression, Metric: Metric: MSE. Description: , Random State: 0} | {}            | 0.214206 | 0.462824 | 0.836535  | 0:00:00  |

#### Interpreter[](broken-reference)

```
interp = TabularInterpreterRegression(exp_mang.get_model('XGBoost', exp_mang.get_current_experiment_id()),
                                      data_manager, exp_mang.opt_metric)
```

```
 94%|=================== | 3898/4128 [00:16<00:00]
```

### RCA[](broken-reference)

#### Data Manager[](broken-reference)

```
data_manager = SupervisedTabularDataManager(data = DataSet.load_titanic_dataset(),
                                            target = 'Survived',
                                            splitter = 'RCA',\
                                            prediction_type = PredictionType.BinaryClassification)
data_manager.build_pipeline(drop_cols = ['PassengerId'])
```

```
Sklearn Pipeline:
MatchVariablesBefore (MatchVariables(missing_values='ignore')) ->
SafeDropFeaturesBefore (SafeDropFeatures(features_to_drop=['PassengerId'])) ->
NanColumnDropper (NanColumnDropper()) ->
Infinity2Nan (Infinity2Nan()) ->
MinMaxScaler (MinMaxScalerWithColumnNames()) ->
HighCardinalityDropper (HighCardinalityDropper()) ->
AddMissingIndicator (AddMissingIndicator()) ->
NumericalImputer (MeanMedianImputer()) ->
SafeCategoricalImputer (SafeCategoricalTransformer(transformer_cls=<class 'feature_engine.imputation.categorical.CategoricalImputer'>)) ->
SafeOneHotEncoder (SafeCategoricalTransformer(transformer_cls=<class 'feature_engine.encoding.one_hot.OneHotEncoder'>)) ->
RemoveSpecialJSONCharacters (RemoveSpecialJSONCharacters()) ->
SafeDropFeaturesAfter (SafeDropFeatures(features_to_drop=['PassengerId'])) ->
MatchVariablesAfter (MatchVariables(missing_values='ignore'))

Target Pipeline:
LabelEncoder (DictLabelEncoder())
```

#### Experiment Manager[](broken-reference)

```
exp_mang = ExperimentManager(data_manager, optimize_metric = MetricName.Accuracy)
exp_mang.remove_models(['Gradient Boosting', 'LightGBM', 'BaselineClassification', 'XGBoost'])
exp_mang.run_experiment()
```

|   | Experiment ID        | Experiment Description | Model                     | Model Description | Data Version | Data Description | Model Params                                                                                                                            | Metric Params | Accuracy | AUC      | Recall   | Precision | Balanced Accuracy | F1       | Run Time |
| - | -------------------- | ---------------------- | ------------------------- | ----------------- | ------------ | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------- | -------- | -------- | -------- | --------- | ----------------- | -------- | -------- |
| 0 | 20240320184443\_c5ae |                        | Baseline Classification   | Default settings  | 72927937     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.548822 | 0.521427 | 0.403509 | 0.410714  | 0.521427          | 0.407080 | 0:00:00  |
| 1 | 20240320184443\_c5ae |                        | Logistic Regression       | Default settings  | 72927937     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.832772 | 0.820751 | 0.769006 | 0.789790  | 0.820751          | 0.779259 | 0:00:01  |
| 2 | 20240320184443\_c5ae |                        | Support Vector Classifier | Default settings  | 72927937     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.818182 | 0.790720 | 0.672515 | 0.821429  | 0.790720          | 0.739550 | 0:00:00  |
| 3 | 20240320184443\_c5ae |                        | AdaBoost Classifier       | Default settings  | 72927937     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.845118 | 0.835730 | 0.795322 | 0.800000  | 0.835730          | 0.797654 | 0:00:00  |
| 4 | 20240320184443\_c5ae |                        | Decision Tree             | Default settings  | 72927937     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.986532 | 0.983559 | 0.970760 | 0.994012  | 0.983559          | 0.982249 | 0:00:00  |
| 5 | 20240320184443\_c5ae |                        | Random Forest             | Default settings  | 72927937     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.986532 | 0.983559 | 0.970760 | 0.994012  | 0.983559          | 0.982249 | 0:00:00  |

#### Interpreter[](broken-reference)

```
interp = TabularInterpreterBinaryClassification(model = exp_mang.get_best_model(),
                                                data_manager = data_manager,
                                                opt_metric = exp_mang.opt_metric,
                                                pos_class = {'pos_class' : 1})
```

```
interp.get_feature_importance()
```

|     | col\_name         | feature\_importance\_vals |
| --- | ----------------- | ------------------------- |
| 0   | Sex\_male         | 0.312146                  |
| 1   | Age               | 0.156952                  |
| 2   | Pclass            | 0.156683                  |
| 3   | Fare              | 0.117617                  |
| 4   | Cabin\_Missing    | 0.083905                  |
| ... | ...               | ...                       |
| 157 | Cabin\_A19        | 0.000000                  |
| 158 | Cabin\_B49        | 0.000000                  |
| 159 | Cabin\_D          | 0.000000                  |
| 160 | Cabin\_C65        | 0.000000                  |
| 161 | Embarked\_Missing | 0.000000                  |

162 rows × 2 columns

### Hyperparameters Optimization[](broken-reference)

#### Data Manager[](broken-reference)

```
data_manager = SupervisedTabularDataManager(data = 'datasets/binary/Titanic.csv',
                                            target = 'Survived',
                                            prediction_type = PredictionType.BinaryClassification,
                                            )
data_manager.build_pipeline(drop_cols = ['PassengerId'])
```

```
Sklearn Pipeline:
MatchVariablesBefore (MatchVariables(missing_values='ignore')) ->
SafeDropFeaturesBefore (SafeDropFeatures(features_to_drop=['PassengerId'])) ->
NanColumnDropper (NanColumnDropper()) ->
Infinity2Nan (Infinity2Nan()) ->
MinMaxScaler (MinMaxScalerWithColumnNames()) ->
HighCardinalityDropper (HighCardinalityDropper()) ->
AddMissingIndicator (AddMissingIndicator()) ->
NumericalImputer (MeanMedianImputer()) ->
SafeCategoricalImputer (SafeCategoricalTransformer(transformer_cls=<class 'feature_engine.imputation.categorical.CategoricalImputer'>)) ->
SafeOneHotEncoder (SafeCategoricalTransformer(transformer_cls=<class 'feature_engine.encoding.one_hot.OneHotEncoder'>)) ->
RemoveSpecialJSONCharacters (RemoveSpecialJSONCharacters()) ->
SafeDropFeaturesAfter (SafeDropFeatures(features_to_drop=['PassengerId'])) ->
MatchVariablesAfter (MatchVariables(missing_values='ignore'))

Target Pipeline:
LabelEncoder (DictLabelEncoder())
```

#### Supervised Tabular Optimizer[](broken-reference)

To start using an optimizer, you must first initialize the optimizer according to the type of problem - in this case its supervised and tabular.

Please note that optimizer is not available by default to every possible case, refer to the documentation for more information.

The supervised tabular optimizer require: - iters: number of hyperparameters trials for each model. - cv: number of cross validation folds. - optimization\_level: there are 2 possibilities by default, HyperParamsOptimizationLevel.Fast and HyperParamsOptimizationLevel.Slow .

```
hyper_parameters_optimizer = SupervisedTabularOptimizer(iters=10, cv=2, optimization_level=HyperParamsOptimizationLevel.Fast)
```

The tabular supervised optimizer supports the following actions:

* Display the current models that are supported by the optimizer:

```
hyper_parameters_optimizer.get_optimizer_models()
```

```
['LGBMClassifier',
 'XGBClassifier',
 'GradientBoostingClassifier',
 'RandomForestClassifier',
 'DecisionTreeClassifier',
 'AdaBoostClassifier',
 'SVC',
 'LogisticRegression',
 'LGBMRegressor',
 'XGBRegressor',
 'LightGBM',
 'XGBoost',
 'GradientBoostingRegressor',
 'RandomForestRegressor',
 'ElasticNet',
 'LassoLarsCV',
 'DecisionTreeRegressor']
```

* Get the current hyperparameter search space of a specific model as a dataframe

```
df = hyper_parameters_optimizer.get_model_params_df('LGBMClassifier')
df.head(10)
```

|   | model          | optimization\_level | hyperparameter\_type | hyperparameter\_name | param   | value                           |
| - | -------------- | ------------------- | -------------------- | -------------------- | ------- | ------------------------------- |
| 0 | LGBMClassifier | Fast                | int                  | max\_depth           | name    | max\_depth                      |
| 1 | LGBMClassifier | Fast                | int                  | max\_depth           | low     | 5                               |
| 2 | LGBMClassifier | Fast                | int                  | max\_depth           | high    | 7                               |
| 3 | LGBMClassifier | Fast                | categorical          | class\_weight        | name    | class\_weight                   |
| 4 | LGBMClassifier | Fast                | categorical          | class\_weight        | choices | \[None, balanced]               |
| 5 | LGBMClassifier | Fast                | categorical          | n\_estimators        | name    | n\_estimators                   |
| 6 | LGBMClassifier | Fast                | categorical          | n\_estimators        | choices | \[100, 150, 200, 250]           |
| 7 | LGBMClassifier | Fast                | categorical          | reg\_alpha           | name    | reg\_alpha                      |
| 8 | LGBMClassifier | Fast                | categorical          | reg\_alpha           | choices | \[0.001, 0.01, 0.1, 1, 10, 100] |
| 9 | LGBMClassifier | Fast                | categorical          | reg\_lambda          | name    | reg\_lambda                     |

* Update the search space for a specific model by using the same dataframe format

```
df.loc[(df['optimization_level']=='Fast') & (df['hyperparameter_name']=='learning_rate') & (df['param']=='high'), 'value'] = 0.6
hyper_parameters_optimizer.set_params(df, model_name='LGBMClassifier')
```

* Get the current hyperparameter search space of all models as a dataframe

```
df = hyper_parameters_optimizer.get_params_df()
df.head(10)
```

|   | model         | optimization\_level | hyperparameter\_type | hyperparameter\_name | param   | value                           |
| - | ------------- | ------------------- | -------------------- | -------------------- | ------- | ------------------------------- |
| 0 | XGBClassifier | Fast                | int                  | max\_depth           | name    | max\_depth                      |
| 1 | XGBClassifier | Fast                | int                  | max\_depth           | low     | 5                               |
| 2 | XGBClassifier | Fast                | int                  | max\_depth           | high    | 7                               |
| 3 | XGBClassifier | Fast                | categorical          | class\_weight        | name    | class\_weight                   |
| 4 | XGBClassifier | Fast                | categorical          | class\_weight        | choices | \[None, balanced]               |
| 5 | XGBClassifier | Fast                | categorical          | n\_estimators        | name    | n\_estimators                   |
| 6 | XGBClassifier | Fast                | categorical          | n\_estimators        | choices | \[100, 150, 200, 250]           |
| 7 | XGBClassifier | Fast                | categorical          | reg\_alpha           | name    | reg\_alpha                      |
| 8 | XGBClassifier | Fast                | categorical          | reg\_alpha           | choices | \[0.001, 0.01, 0.1, 1, 10, 100] |
| 9 | XGBClassifier | Fast                | categorical          | reg\_lambda          | name    | reg\_lambda                     |

* Update the search space for all models by using the same dataframe format

```
hyper_parameters_optimizer.set_params(df)
```

* Restore the search space to the defaults

```
hyper_parameters_optimizer.restore_params()
```

#### Experiment Manager[](broken-reference)

All that is left to do now, is to parse the tabular supervised optimizer into the experiment manager.

```
exp_mang = ExperimentManager(data_manager, optimize_metric = MetricName.AUC, hyper_parameters_optimizer = hyper_parameters_optimizer)
exp_mang.run_experiment()
```

|   | Experiment ID        | Experiment Description | Model                     | Model Description                                                                                                              | Data Version | Data Description | Model Params                                                                                                                       | Metric Params | Accuracy | AUC      | Recall   | Precision | Balanced Accuracy | F1       | Run Time |
| - | -------------------- | ---------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------ | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------- | -------- | -------- | -------- | --------- | ----------------- | -------- | -------- |
| 0 | 20240320184452\_7ed2 |                        | Baseline Classification   | Default settings                                                                                                               | 17a79ae5     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.505618 | 0.484893 | 0.397059 | 0.364865  | 0.484893          | 0.380282 | 0:00:00  |
| 1 | 20240320184452\_7ed2 |                        | Logistic Regression       | Optimized , C=100, multi\_class=ovr, class\_weight=balanced, tol=0.010, max\_iter=1000                                         | 17a79ae5     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.797753 | 0.802674 | 0.823529 | 0.700000  | 0.802674          | 0.756757 | 0:00:01  |
| 2 | 20240320184452\_7ed2 |                        | Support Vector Classifier | Optimized , C=0.010, kernel=linear, class\_weight=balanced, tol=0.010, max\_iter=1000, degree=2, gamma=auto, coef0=0.000       | 17a79ae5     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.803371 | 0.787567 | 0.720588 | 0.753846  | 0.787567          | 0.736842 | 0:00:00  |
| 3 | 20240320184452\_7ed2 |                        | AdaBoost Classifier       | Optimized , learning\_rate=0.225, n\_estimators=250                                                                            | 17a79ae5     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.825843 | 0.828209 | 0.838235 | 0.740260  | 0.828209          | 0.786207 | 0:00:06  |
| 4 | 20240320184452\_7ed2 |                        | Decision Tree             | Optimized , max\_features=0.835, min\_samples\_leaf=2                                                                          | 17a79ae5     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.797753 | 0.771791 | 0.661765 | 0.775862  | 0.771791          | 0.714286 | 0:00:00  |
| 5 | 20240320184452\_7ed2 |                        | Random Forest             | Optimized , max\_features=0.905, n\_estimators=100, min\_samples\_leaf=2                                                       | 17a79ae5     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.831461 | 0.810294 | 0.720588 | 0.816667  | 0.810294          | 0.765625 | 0:00:04  |
| 6 | 20240320184452\_7ed2 |                        | Gradient Boosting         | Optimized , learning\_rate=0.295, max\_depth=5, n\_estimators=100                                                              | 17a79ae5     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.825843 | 0.800134 | 0.691176 | 0.824561  | 0.800134          | 0.752000 | 0:00:06  |
| 7 | 20240320184452\_7ed2 |                        | XGBoost                   | Optimized , max\_depth=6, class\_weight=balanced, n\_estimators=200, reg\_alpha=0.100, reg\_lambda=1, learning\_rate=0.001     | 17a79ae5     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.786517 | 0.762701 | 0.661765 | 0.750000  | 0.762701          | 0.703125 | 0:00:03  |
| 8 | 20240320184452\_7ed2 |                        | LightGBM                  | Optimized , max\_depth=5, class\_weight=balanced, n\_estimators=150, reg\_alpha=0.100, reg\_lambda=0.001, learning\_rate=0.001 | 17a79ae5     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.719101 | 0.722193 | 0.735294 | 0.609756  | 0.722193          | 0.666667 | 0:00:04  |

### Cross-Validation[](broken-reference)

In this section, we extend our model evaluation process by incorporating cross-validation. We’ve already covered the basics of model building, optimization, and performance comparison. Now, let’s add a layer of robustness to our assessments with cross-validation.

Cross-validation is a powerful technique that helps prevent overfitting to a specific data distribution, ensuring our model selection is more reliable.

#### Binary Classification[](broken-reference)

```
data_manager = CrossValidationSupervisedTabularDataManager(data = DataSet.load_titanic_dataset(),
                                            target = 'Survived',
                                            prediction_type = PredictionType.BinaryClassification,
                                            n_folds = 5
                                           )
data_manager.build_pipeline(drop_cols = ['PassengerId'])
```

```
Sklearn Pipeline:
MatchVariablesBefore (MatchVariables(missing_values='ignore')) ->
SafeDropFeaturesBefore (SafeDropFeatures(features_to_drop=['PassengerId'])) ->
NanColumnDropper (NanColumnDropper()) ->
Infinity2Nan (Infinity2Nan()) ->
MinMaxScaler (MinMaxScalerWithColumnNames()) ->
HighCardinalityDropper (HighCardinalityDropper()) ->
AddMissingIndicator (AddMissingIndicator()) ->
NumericalImputer (MeanMedianImputer()) ->
SafeCategoricalImputer (SafeCategoricalTransformer(transformer_cls=<class 'feature_engine.imputation.categorical.CategoricalImputer'>)) ->
SafeOneHotEncoder (SafeCategoricalTransformer(transformer_cls=<class 'feature_engine.encoding.one_hot.OneHotEncoder'>)) ->
RemoveSpecialJSONCharacters (RemoveSpecialJSONCharacters()) ->
SafeDropFeaturesAfter (SafeDropFeatures(features_to_drop=['PassengerId'])) ->
MatchVariablesAfter (MatchVariables(missing_values='ignore'))

Target Pipeline:
LabelEncoder (DictLabelEncoder())
```

After preparing the data, we can visualize how it is divided among the folds with the plot\_cross\_validation method.

```
data_manager.splitter.plot_cross_validation(data_manager.data)
```

The code above prepares our Titanic dataset for cross-validation by dividing it into 5 folds.

Next, we use the ExperimentManager class. this time we will use a custom trainer - CVTrainer, with some additional parameters specified for our cross validation trainer.

```
exp_mang = ExperimentManager(
    data_manager,
    optimize_metric = MetricName.AUC,
    trainer = CVTrainer(aggregation = CVAggregation.MEAN, selected_model = CVSelectedModel.BEST)
)
```

In our CVTrainer, we are specifying two critical parameters:

aggregation: This parameter determines how to combine the performance metrics across the different folds. We can choose between ‘mean’, ‘max’, or ‘min’. Here we’ve selected ‘mean’, which means the performance metrics in our results table will be the average performance across all folds.

selected\_model: This parameter controls which model instance is returned from the cross-validation folds. It could either be the ‘best’ or the ‘worst’ performing model.

Once our experiment manager is defined, we can execute the experiment. Now, each model is not just trained and evaluated once, but across different distributions of the data.

```
exp_mang.run_experiment()
```

|   | Experiment ID        | Experiment Description | Model                     | Model Description | Data Version | Data Description | Model Params                                                                                                                       | Metric Params | Accuracy | AUC      | Recall   | Precision | Balanced Accuracy | F1       | Run Time |
| - | -------------------- | ---------------------- | ------------------------- | ----------------- | ------------ | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------- | -------- | -------- | -------- | --------- | ----------------- | -------- | -------- |
| 0 | 20240320184524\_72e1 |                        | Baseline Classification   | Default settings  | 879832c6     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.540550 | 0.513425 | 0.403075 | 0.390573  | 0.513425          | 0.396166 | 0:00:00  |
| 1 | 20240320184524\_72e1 |                        | Logistic Regression       | Default settings  | 879832c6     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.797754 | 0.778012 | 0.697904 | 0.748511  | 0.778012          | 0.722084 | 0:00:00  |
| 2 | 20240320184524\_72e1 |                        | Support Vector Classifier | Default settings  | 879832c6     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.799153 | 0.768664 | 0.645423 | 0.785342  | 0.768664          | 0.707775 | 0:00:00  |
| 3 | 20240320184524\_72e1 |                        | AdaBoost Classifier       | Default settings  | 879832c6     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.811760 | 0.797351 | 0.738854 | 0.758423  | 0.797351          | 0.747609 | 0:00:00  |
| 4 | 20240320184524\_72e1 |                        | Decision Tree             | Default settings  | 879832c6     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.765380 | 0.751272 | 0.693990 | 0.686578  | 0.751272          | 0.688862 | 0:00:00  |
| 5 | 20240320184524\_72e1 |                        | Random Forest             | Default settings  | 879832c6     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.803349 | 0.779492 | 0.682809 | 0.769055  | 0.779492          | 0.722981 | 0:00:01  |
| 6 | 20240320184524\_72e1 |                        | Gradient Boosting         | Default settings  | 879832c6     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.828622 | 0.802512 | 0.697414 | 0.824584  | 0.802512          | 0.753412 | 0:00:01  |
| 7 | 20240320184524\_72e1 |                        | XGBoost                   | Default settings  | 879832c6     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.792091 | 0.771162 | 0.686373 | 0.743204  | 0.771162          | 0.713138 | 0:00:00  |
| 8 | 20240320184524\_72e1 |                        | LightGBM                  | Default settings  | 879832c6     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.806136 | 0.787648 | 0.712579 | 0.760058  | 0.787648          | 0.734933 | 0:00:00  |

The results table provides the performance metrics for each model. Note that these metrics are aggregated according to our aggregation choice.

To visualize the variability of a model’s performance across folds, we can use box plots. For instance, we can generate a box plot for the ‘Logistic Regression’ model using the AUC metric:

```
_ = exp_mang.trainer.box_plot_per_model_and_metric('Logistic Regression', 'AUC')
```

Finally, to visualize the cross-validation results for all models and metrics, we can plot the results:

```
_ = exp_mang.trainer.plot_cv_res()
```

```
_ = exp_mang.trainer.display_best_models()
```

```
_ = exp_mang.trainer.display_model_cross_metrics('Gradient Boosting')
```

#### Regression[](broken-reference)

```
data_manager = CrossValidationSupervisedTabularDataManager(data = DataSet.load_fetch_california_housing_dataset(),
                                            target = 'MedHouseVal',
                                            prediction_type = PredictionType.Regression)
data_manager.build_pipeline()
exp_mang = ExperimentManager(
    data_manager,
    optimize_metric = MetricName.MSE,
    trainer = CVTrainer(aggregation = CVAggregation.MEAN, selected_model = CVSelectedModel.BEST)
)
exp_mang.run_experiment()
```

|   | Experiment ID        | Experiment Description | Model             | Model Description | Data Version | Data Description | Model Params                                                                                                             | Metric Params | MSE      | RMSE     | R2        | Run Time |
| - | -------------------- | ---------------------- | ----------------- | ----------------- | ------------ | ---------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------- | -------- | -------- | --------- | -------- |
| 0 | 20240320184542\_7034 |                        | Elastic Net       | Default settings  | 40d9e5a4     |                  | {'experiment\_manager': Prediction Type: PredictionType.Regression, Metric: Metric: MSE. Description: , Random State: 0} | {}            | 1.336881 | 1.156172 | -0.000215 | 0:00:00  |
| 1 | 20240320184542\_7034 |                        | Lasso Lars CV     | Default settings  | 40d9e5a4     |                  | {'experiment\_manager': Prediction Type: PredictionType.Regression, Metric: Metric: MSE. Description: , Random State: 0} | {}            | 0.518867 | 0.720192 | 0.611802  | 0:00:00  |
| 2 | 20240320184542\_7034 |                        | Decision Tree     | Default settings  | 40d9e5a4     |                  | {'experiment\_manager': Prediction Type: PredictionType.Regression, Metric: Metric: MSE. Description: , Random State: 0} | {}            | 0.540565 | 0.734914 | 0.595504  | 0:00:02  |
| 3 | 20240320184542\_7034 |                        | Random Forest     | Default settings  | 40d9e5a4     |                  | {'experiment\_manager': Prediction Type: PredictionType.Regression, Metric: Metric: MSE. Description: , Random State: 0} | {}            | 0.261275 | 0.511019 | 0.804585  | 0:00:49  |
| 4 | 20240320184542\_7034 |                        | Gradient Boosting | Default settings  | 40d9e5a4     |                  | {'experiment\_manager': Prediction Type: PredictionType.Regression, Metric: Metric: MSE. Description: , Random State: 0} | {}            | 0.283332 | 0.532133 | 0.788087  | 0:00:34  |
| 5 | 20240320184542\_7034 |                        | XGBoost           | Default settings  | 40d9e5a4     |                  | {'experiment\_manager': Prediction Type: PredictionType.Regression, Metric: Metric: MSE. Description: , Random State: 0} | {}            | 0.231335 | 0.480854 | 0.827000  | 0:00:04  |
| 6 | 20240320184542\_7034 |                        | LightGBM          | Default settings  | 40d9e5a4     |                  | {'experiment\_manager': Prediction Type: PredictionType.Regression, Metric: Metric: MSE. Description: , Random State: 0} | {}            | 0.221115 | 0.470032 | 0.834673  | 0:00:00  |

```
_ = exp_mang.trainer.plot_cv_res()
```

In conclusion, by incorporating cross-validation, we make our model evaluation process more comprehensive and robust, providing a solid foundation for reliable model selection.
