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

### Classification[](broken-reference)

#### Data Manager[](broken-reference)

```
df = DataSet.load_time_series_classification_dataset()[::10]
df.head()
```

|    | measure   | ID | target |
| -- | --------- | -- | ------ |
| 0  | 1.801035  | 0  | 1      |
| 10 | -1.067028 | 0  | 1      |
| 20 | 1.553591  | 0  | 1      |
| 30 | -0.247445 | 0  | 1      |
| 40 | 0.372860  | 0  | 1      |

```
def plot_random_samples(df, num_images, id_col, traget_col, measure_col):
    # Determine the number of rows and columns
    num_cols = min(num_images, 3)
    num_rows = -(-num_images // num_cols)  # Ceiling division

    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(25, 10))
    fig.subplots_adjust(hspace=0.6, wspace=0.3)

    # Check if ax is 1D and convert to 2D if necessary
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])
    if ax.ndim == 1:
        ax = ax.reshape(-1, num_cols)

    # Get unique IDs and draw random samples
    unique_ids = df[id_col].unique()
    if len(unique_ids) < num_images:
        raise ValueError("Not enough unique IDs to draw the requested number of images.")

    sampled_ids = np.random.choice(unique_ids, size=num_images, replace=False)

    # Plotting
    for idx, k in enumerate(sampled_ids):
        i, j = divmod(idx, num_cols)

        # Filter data for each ID and reset index
        filtered_data = df.loc[df[id_col] == k].reset_index(drop=True)

        # Set title for each subplot using the target value
        target_value = filtered_data[traget_col].iloc[0]
        ax[i, j].set_title(f"ID: {k}, Target: {target_value}")

        sns.lineplot(data=filtered_data, x=filtered_data.index, y=measure_col, ax=ax[i, j], estimator=None)
        ax[i, j].set_xlabel('Time')

    # Hide any unused subplots
    for idx in range(num_images, num_rows * num_cols):
        fig.delaxes(ax.flatten()[idx])

    plt.show()
```

```
plot_random_samples(df, 9,
                    id_col = 'ID',
                    traget_col = 'target',
                    measure_col = 'measure'
                   )
```

```
data_manager = SupervisedTabularDataManager(data = df,
                    target = 'target',
                    prediction_type = PredictionType.BinaryClassification,
                    splitter = GroupSplitter(split_sets = {Dataset.Train: 0.8, Dataset.Valid: 0.2},
                                             group_columns = 'ID'
                                            )
                   )
```

```
data_manager.build_pipeline(
    id_label_encoder=True,
    waveforms_feature_extractor=True,
    step_params={
        'column_id': 'ID',
    }
)
```

```
Sklearn Pipeline:
MatchVariablesBefore (MatchVariables(missing_values='ignore')) ->
WaveformsFeatureExtractor (WaveformsFeatureExtractor(column_id='ID')) ->
SafeInernalDropFeaturesBefore (SafeDropFeatures(features_to_drop=['ID'])) ->
NanColumnDropper (NanColumnDropper()) ->
Infinity2Nan (Infinity2Nan()) ->
MinMaxScaler (MinMaxScalerWithColumnNames()) ->
AddMissingIndicator (AddMissingIndicator()) ->
NumericalImputer (MeanMedianImputer()) ->
RemoveSpecialJSONCharacters (RemoveSpecialJSONCharacters()) ->
SafeDropFeaturesAfter (SafeDropFeatures(features_to_drop=['ID'])) ->
MatchVariablesAfter (MatchVariables(missing_values='ignore'))

Target Pipeline:
UniqueIDLabelEncoder (UniqueIDLabelEncoder(column_id='ID')) ->
LabelEncoder (DictLabelEncoder())
```

```
data_sizes = data_manager.splitter.allocated_sizes
data_sizes
```

```
{<Dataset.Train: 'Train'>: 196850, <Dataset.Valid: 'Valid'>: 49200}
```

```
total_size = sum(data_sizes.values())
{dataset: size / total_size for dataset, size in data_sizes.items()}
```

```
{<Dataset.Train: 'Train'>: 0.8000406421459053,
 <Dataset.Valid: 'Valid'>: 0.1999593578540947}
```

```
X, y = data_manager.get_training_data()
```

```
Feature Extraction: 100%|██████████| 10/10 [02:31<00:00, 15.19s/it]
```

|      | measure\_\_variance\_larger\_than\_standard\_deviation | measure\_\_has\_duplicate\_max | measure\_\_has\_duplicate\_min | measure\_\_has\_duplicate | measure\_\_sum\_values | measure\_\_abs\_energy | measure\_\_mean\_abs\_change | measure\_\_mean\_change | measure\_\_mean\_second\_derivative\_central | measure\_\_median | ... | measure\_\_permutation\_entropy\_\_dimension\_5\_\_tau\_1 | measure\_\_permutation\_entropy\_\_dimension\_6\_\_tau\_1 | measure\_\_permutation\_entropy\_\_dimension\_7\_\_tau\_1 | measure\_\_mean\_n\_absolute\_max\_\_number\_of\_maxima\_7 | measure\_\_sample\_entropy\_na | measure\_\_friedrich\_coefficients\_\_coeff\_0\_\_m\_3\_\_r\_30\_na | measure\_\_friedrich\_coefficients\_\_coeff\_1\_\_m\_3\_\_r\_30\_na | measure\_\_friedrich\_coefficients\_\_coeff\_2\_\_m\_3\_\_r\_30\_na | measure\_\_friedrich\_coefficients\_\_coeff\_3\_\_m\_3\_\_r\_30\_na | measure\_\_max\_langevin\_fixed\_point\_\_m\_3\_\_r\_30\_na |
| ---- | ------------------------------------------------------ | ------------------------------ | ------------------------------ | ------------------------- | ---------------------- | ---------------------- | ---------------------------- | ----------------------- | -------------------------------------------- | ----------------- | --- | --------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------- |
| 0    | 1.0                                                    | 0.0                            | 0.0                            | 0.0                       | 0.628858               | 0.559886               | 0.604199                     | 0.428949                | 0.743704                                     | 0.471742          | ... | 0.716560                                                  | 0.860973                                                  | 0.927430                                                  | 0.425367                                                   | 0                              | 0                                                                   | 0                                                                   | 0                                                                   | 0                                                                   | 0                                                           |
| 1    | 0.0                                                    | 0.0                            | 0.0                            | 0.0                       | 0.415789               | 0.417164               | 0.442165                     | 0.593914                | 0.355238                                     | 0.285848          | ... | 0.649710                                                  | 0.809265                                                  | 0.963715                                                  | 0.390074                                                   | 0                              | 0                                                                   | 0                                                                   | 0                                                                   | 0                                                                   | 0                                                           |
| 2    | 0.0                                                    | 0.0                            | 0.0                            | 0.0                       | 0.510373               | 0.410413               | 0.549500                     | 0.484345                | 0.439737                                     | 0.504423          | ... | 0.637324                                                  | 0.870731                                                  | 1.000000                                                  | 0.336733                                                   | 0                              | 0                                                                   | 0                                                                   | 0                                                                   | 0                                                                   | 0                                                           |
| 3    | 1.0                                                    | 0.0                            | 0.0                            | 0.0                       | 0.547114               | 0.548712               | 0.526338                     | 0.326108                | 0.362702                                     | 0.549823          | ... | 0.769269                                                  | 0.870731                                                  | 0.927430                                                  | 0.511599                                                   | 0                              | 0                                                                   | 0                                                                   | 0                                                                   | 0                                                                   | 0                                                           |
| 4    | 0.0                                                    | 0.0                            | 0.0                            | 1.0                       | 0.545480               | 0.448605               | 0.576864                     | 0.745195                | 0.431082                                     | 0.445777          | ... | 0.628711                                                  | 0.870731                                                  | 0.963715                                                  | 0.338288                                                   | 0                              | 0                                                                   | 0                                                                   | 0                                                                   | 0                                                                   | 0                                                           |
| ...  | ...                                                    | ...                            | ...                            | ...                       | ...                    | ...                    | ...                          | ...                     | ...                                          | ...               | ... | ...                                                       | ...                                                       | ...                                                       | ...                                                        | ...                            | ...                                                                 | ...                                                                 | ...                                                                 | ...                                                                 | ...                                                         |
| 4915 | 0.0                                                    | 0.0                            | 0.0                            | 0.0                       | 0.539408               | 0.300003               | 0.377501                     | 0.417152                | 0.482191                                     | 0.604465          | ... | 0.657219                                                  | 0.844878                                                  | 0.963715                                                  | 0.288781                                                   | 0                              | 0                                                                   | 0                                                                   | 0                                                                   | 0                                                                   | 0                                                           |
| 4917 | 1.0                                                    | 0.0                            | 0.0                            | 0.0                       | 0.512499               | 0.641542               | 0.787194                     | 0.312968                | 0.370730                                     | 0.327376          | ... | 0.611141                                                  | 0.757558                                                  | 0.818576                                                  | 0.484934                                                   | 0                              | 0                                                                   | 0                                                                   | 0                                                                   | 0                                                                   | 0                                                           |
| 4918 | 0.0                                                    | 0.0                            | 0.0                            | 0.0                       | 0.392430               | 0.461793               | 0.536445                     | 0.464727                | 0.619057                                     | 0.490948          | ... | 0.806733                                                  | 0.896585                                                  | 1.000000                                                  | 0.394118                                                   | 1                              | 0                                                                   | 0                                                                   | 0                                                                   | 0                                                                   | 0                                                           |
| 4919 | 0.0                                                    | 0.0                            | 0.0                            | 0.0                       | 0.566998               | 0.474285               | 0.534058                     | 0.499405                | 0.517493                                     | 0.644479          | ... | 0.736454                                                  | 0.844878                                                  | 1.000000                                                  | 0.321980                                                   | 0                              | 0                                                                   | 0                                                                   | 0                                                                   | 0                                                                   | 0                                                           |
| 4920 | 0.0                                                    | 0.0                            | 0.0                            | 0.0                       | 0.514375               | 0.304979               | 0.403572                     | 0.634979                | 0.459082                                     | 0.599352          | ... | 0.817671                                                  | 0.896585                                                  | 1.000000                                                  | 0.257544                                                   | 0                              | 0                                                                   | 0                                                                   | 0                                                                   | 0                                                                   | 0                                                           |

3937 rows × 476 columns

#### Experiment Manager[](broken-reference)

```
exp_mang = ExperimentManager(data_manager, optimize_metric = MetricName.AUC)
```

```
exp_mang.display_models_pool()
```

|   | Name                      | Description      | Source | Is Available |
| - | ------------------------- | ---------------- | ------ | ------------ |
| 0 | Support Vector Classifier | Default settings | Pool   | True         |
| 1 | Gradient Boosting         | Default settings | Pool   | True         |
| 2 | AdaBoost Classifier       | Default settings | Pool   | True         |
| 3 | Baseline Classification   | Default settings | Pool   | True         |
| 4 | XGBoost                   | Default settings | Pool   | True         |
| 5 | Decision Tree             | Default settings | Pool   | True         |
| 6 | Logistic Regression       | Default settings | Pool   | True         |
| 7 | LightGBM                  | Default settings | Pool   | True         |
| 8 | Random Forest             | Default settings | Pool   | True         |

```
exp_mang.display_metrics_pool()
```

|   | Name              | Description | Source | Is Available | Is Optimal |
| - | ----------------- | ----------- | ------ | ------------ | ---------- |
| 0 | AUC               |             | Pool   | True         | True       |
| 1 | Recall            |             | Pool   | True         | False      |
| 2 | Precision         |             | Pool   | True         | False      |
| 3 | Balanced Accuracy |             | Pool   | True         | False      |
| 4 | F1                |             | Pool   | True         | False      |
| 5 | Accuracy          |             | Pool   | True         | False      |

```
exp_mang.run_experiment()
```

|   | Experiment ID        | Experiment Description | Model                     | Model Description | Data Version | Data Description | Model Params                                                                                                                       | Metric Params | Accuracy | AUC      | Recall   | Precision | Balanced Accuracy | F1       | Run Time |
| - | -------------------- | ---------------------- | ------------------------- | ----------------- | ------------ | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------- | -------- | -------- | -------- | --------- | ----------------- | -------- | -------- |
| 0 | 20240320185134\_3d3b |                        | Baseline Classification   | Default settings  | 55ed0246     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.487805 | 0.487232 | 0.465553 | 0.473461  | 0.487232          | 0.469474 | 0:00:42  |
| 1 | 20240320185134\_3d3b |                        | Logistic Regression       | Default settings  | 55ed0246     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.903455 | 0.904328 | 0.937370 | 0.873541  | 0.904328          | 0.904330 | 0:00:01  |
| 2 | 20240320185134\_3d3b |                        | Support Vector Classifier | Default settings  | 55ed0246     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.912602 | 0.914207 | 0.974948 | 0.863216  | 0.914207          | 0.915686 | 0:00:08  |
| 3 | 20240320185134\_3d3b |                        | AdaBoost Classifier       | Default settings  | 55ed0246     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.904472 | 0.904996 | 0.924843 | 0.884232  | 0.904996          | 0.904082 | 0:00:27  |
| 4 | 20240320185134\_3d3b |                        | Decision Tree             | Default settings  | 55ed0246     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.854675 | 0.854170 | 0.835073 | 0.862069  | 0.854170          | 0.848356 | 0:00:05  |
| 5 | 20240320185134\_3d3b |                        | Random Forest             | Default settings  | 55ed0246     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.906504 | 0.907675 | 0.951983 | 0.868571  | 0.907675          | 0.908367 | 0:00:05  |
| 6 | 20240320185134\_3d3b |                        | Gradient Boosting         | Default settings  | 55ed0246     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.917683 | 0.918351 | 0.943633 | 0.893281  | 0.918351          | 0.917766 | 0:02:31  |
| 7 | 20240320185134\_3d3b |                        | XGBoost                   | Default settings  | 55ed0246     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.915650 | 0.916317 | 0.941545 | 0.891304  | 0.916317          | 0.915736 | 0:00:07  |
| 8 | 20240320185134\_3d3b |                        | LightGBM                  | Default settings  | 55ed0246     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: AUC. Description: , Random State: 0} | {}            | 0.925813 | 0.926594 | 0.956159 | 0.898039  | 0.926594          | 0.926188 | 0:00:01  |

```
best_model = exp_mang.get_best_model()
best_model
```

```
Model: LGBMClassifier(n_jobs=-1, random_state=42, verbose=-1), Description: Default settings
```

```
interp = TabularInterpreterBinaryClassification(model = exp_mang.get_model(model_name = 'XGBoost',
                                                                           experiment_id = exp_mang.get_current_experiment_id()
                                                                          ),
                                                data_manager = data_manager,
                                                opt_metric = exp_mang.opt_metric,
                                                pos_class = {'pos_class' : 1})
```

```
 98%|===================| 3873/3937 [00:16<00:00]
```

```
df_p = df.loc[df['ID'].isin([4,5,6,7,9,8])]
df_p
```

|      | measure   | ID  | target |
| ---- | --------- | --- | ------ |
| 2000 | -1.937263 | 4   | 0      |
| 2010 | 0.761251  | 4   | 0      |
| 2020 | 1.153546  | 4   | 0      |
| 2030 | -0.092288 | 4   | 0      |
| 2040 | -0.704902 | 4   | 0      |
| ...  | ...       | ... | ...    |
| 4950 | 1.762097  | 9   | 1      |
| 4960 | -0.653458 | 9   | 1      |
| 4970 | -0.629748 | 9   | 1      |
| 4980 | 2.282610  | 9   | 1      |
| 4990 | -2.431297 | 9   | 1      |

300 rows × 3 columns

```
inf_model.model_pipeline.predict(df_p)
```

```
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[20], line 1
----> 1 inf_model.model_pipeline.predict(df_p)

NameError: name 'inf_model' is not defined
```

```
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[21], line 1
----> 1 inf_model.predict(df_p)

NameError: name 'inf_model' is not defined
```

```
inf_model.predict(df_p, with_input = True)
```

```
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[22], line 1
----> 1 inf_model.predict(df_p, with_input = True)

NameError: name 'inf_model' is not defined
```
