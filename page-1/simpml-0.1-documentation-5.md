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

### Low level[](broken-reference)

```
data_fetcher = TabularDataFetcher(data = 'datasets/binary/Titanic.csv')
```

|     | PassengerId | Survived | Pclass | Name                                                | Sex    | Age  | SibSp | Parch | Ticket           | Fare    | Cabin | Embarked |
| --- | ----------- | -------- | ------ | --------------------------------------------------- | ------ | ---- | ----- | ----- | ---------------- | ------- | ----- | -------- |
| 0   | 1           | 0        | 3      | Braund, Mr. Owen Harris                             | male   | 22.0 | 1     | 0     | A/5 21171        | 7.2500  | NaN   | S        |
| 1   | 2           | 1        | 1      | Cumings, Mrs. John Bradley (Florence Briggs Thayer) | female | 38.0 | 1     | 0     | PC 17599         | 71.2833 | C85   | C        |
| 2   | 3           | 1        | 3      | Heikkinen, Miss. Laina                              | female | 26.0 | 0     | 0     | STON/O2. 3101282 | 7.9250  | NaN   | S        |
| 3   | 4           | 1        | 1      | Futrelle, Mrs. Jacques Heath (Lily May Peel)        | female | 35.0 | 1     | 0     | 113803           | 53.1000 | C123  | S        |
| 4   | 5           | 0        | 3      | Allen, Mr. William Henry                            | male   | 35.0 | 0     | 0     | 373450           | 8.0500  | NaN   | S        |
| ... | ...         | ...      | ...    | ...                                                 | ...    | ...  | ...   | ...   | ...              | ...     | ...   | ...      |
| 886 | 887         | 0        | 2      | Montvila, Rev. Juozas                               | male   | 27.0 | 0     | 0     | 211536           | 13.0000 | NaN   | S        |
| 887 | 888         | 1        | 1      | Graham, Miss. Margaret Edith                        | female | 19.0 | 0     | 0     | 112053           | 30.0000 | B42   | S        |
| 888 | 889         | 0        | 3      | Johnston, Miss. Catherine Helen "Carrie"            | female | NaN  | 1     | 2     | W./C. 6607       | 23.4500 | NaN   | S        |
| 889 | 890         | 1        | 1      | Behr, Mr. Karl Howell                               | male   | 26.0 | 0     | 0     | 111369           | 30.0000 | C148  | C        |
| 890 | 891         | 0        | 3      | Dooley, Mr. Patrick                                 | male   | 32.0 | 0     | 0     | 370376           | 7.7500  | NaN   | Q        |

891 rows × 12 columns

```
splitter = RandomSplitter(target = 'Survived', split_sets = {Dataset.Train: 0.6, Dataset.Valid: 0.2, Dataset.Test: 0.2})
```

```
indices = splitter.split(data_fetcher.get_items())
```

```
dict_keys([<Dataset.Train: 'Train'>, <Dataset.Valid: 'Valid'>, <Dataset.Test: 'Test'>])
```

```
Index([844, 316, 768, 255, 130, 110, 769, 319, 707, 172,
       ...
       535, 104, 177, 544, 680, 476,  58, 736, 462, 747],
      dtype='int64', length=534)
```

```
train_data = data_fetcher.get_items().loc[indices[Dataset.Train]]
```

```
X_train, y_train = train_data.drop(splitter.target, axis = 1), train_data[splitter.target]
```

|     | PassengerId | Pclass | Name                                    | Sex    | Age  | SibSp | Parch | Ticket     | Fare    | Cabin | Embarked |
| --- | ----------- | ------ | --------------------------------------- | ------ | ---- | ----- | ----- | ---------- | ------- | ----- | -------- |
| 844 | 845         | 3      | Culumovic, Mr. Jeso                     | male   | 17.0 | 0     | 0     | 315090     | 8.6625  | NaN   | S        |
| 316 | 317         | 2      | Kantor, Mrs. Sinai (Miriam Sternin)     | female | 24.0 | 1     | 0     | 244367     | 26.0000 | NaN   | S        |
| 768 | 769         | 3      | Moran, Mr. Daniel J                     | male   | NaN  | 1     | 0     | 371110     | 24.1500 | NaN   | Q        |
| 255 | 256         | 3      | Touma, Mrs. Darwis (Hanne Youssef Razi) | female | 29.0 | 0     | 2     | 2650       | 15.2458 | NaN   | C        |
| 130 | 131         | 3      | Drazenoic, Mr. Jozef                    | male   | 33.0 | 0     | 0     | 349241     | 7.8958  | NaN   | C        |
| ... | ...         | ...    | ...                                     | ...    | ...  | ...   | ...   | ...        | ...     | ...   | ...      |
| 476 | 477         | 2      | Renouf, Mr. Peter Henry                 | male   | 34.0 | 1     | 0     | 31027      | 21.0000 | NaN   | S        |
| 58  | 59          | 2      | West, Miss. Constance Mirium            | female | 5.0  | 1     | 2     | C.A. 34651 | 27.7500 | NaN   | S        |
| 736 | 737         | 3      | Ford, Mrs. Edward (Margaret Ann Watson) | female | 48.0 | 1     | 3     | W./C. 6608 | 34.3750 | NaN   | S        |
| 462 | 463         | 1      | Gee, Mr. Arthur H                       | male   | 47.0 | 0     | 0     | 111320     | 38.5000 | E63   | S        |
| 747 | 748         | 2      | Sinkkonen, Miss. Anna                   | female | 30.0 | 0     | 0     | 250648     | 13.0000 | NaN   | S        |

534 rows × 11 columns

```
844    0
316    1
768    0
255    1
130    0
      ..
476    0
58     1
736    0
462    0
747    1
Name: Survived, Length: 534, dtype: int64
```

```
numerical_cols = ['Age', 'Fare']
categorical_cols = ['Parch', 'Ticket', 'Cabin', 'Pclass', 'Embarked', 'Sex', 'SibSp', 'Name']

pipeline = Pipeline(
                SklearnPipeline([('AddMissingIndicator', AddMissingIndicator()),
                                ('NumericalImputer', MeanMedianImputer(variables = numerical_cols)),
                                ('CategoricalImputer', CategoricalImputer(variables = categorical_cols, ignore_format=True)),
                                ('OneHotEncoder', OneHotEncoder(variables = categorical_cols, ignore_format=True)),
                                ('RemoveSpecialJSONCharacters', RemoveSpecialJSONCharacters())]),
                TrainPipeline([('SMOTE', ManipulateAdapter(SMOTE(), 'fit_resample'))]),
                TargetPipeline([('LabelEncoder', DictLabelEncoder())])
)
```

```
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer, AddMissingIndicator
from feature_engine.encoding import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


data_manager = TabularDataManager(
    splitter = splitter,
    data_fetcher = data_fetcher,
    pipeline = pipeline,
    prediction_type = PredictionType.BinaryClassification
)

```

```
X, y = data_manager.get_training_data()
```

|     | PassengerId | Age       | Fare      | Age\_na | Cabin\_na | Embarked\_na | Parch\_0 | Parch\_2 | Parch\_1 | Parch\_3 | ... | Name\_Hart\_\_\_Miss\_\_\_Eva\_\_\_Miriam | Name\_Gustafsson\_\_\_Mr\_\_\_Anders\_\_\_Vilhelm | Name\_Isham\_\_\_Miss\_\_\_Ann\_\_\_Elizabeth | Name\_Douglas\_\_\_Mr\_\_\_Walter\_\_\_Donald | Name\_Peters\_\_\_Miss\_\_\_Katie | Name\_Renouf\_\_\_Mr\_\_\_Peter\_\_\_Henry | Name\_West\_\_\_Miss\_\_\_Constance\_\_\_Mirium | Name\_Ford\_\_\_Mrs\_\_\_Edward\_\_\_Margaret\_\_\_Ann\_\_\_Watson\_\_\_ | Name\_Gee\_\_\_Mr\_\_\_Arthur\_\_\_H | Name\_Sinkkonen\_\_\_Miss\_\_\_Anna |
| --- | ----------- | --------- | --------- | ------- | --------- | ------------ | -------- | -------- | -------- | -------- | --- | ----------------------------------------- | ------------------------------------------------- | --------------------------------------------- | --------------------------------------------- | --------------------------------- | ------------------------------------------ | ----------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------ | ----------------------------------- |
| 0   | 845         | 17.000000 | 8.662500  | 0       | 1         | 0            | 1        | 0        | 0        | 0        | ... | 0                                         | 0                                                 | 0                                             | 0                                             | 0                                 | 0                                          | 0                                               | 0                                                                        | 0                                    | 0                                   |
| 1   | 317         | 24.000000 | 26.000000 | 0       | 1         | 0            | 1        | 0        | 0        | 0        | ... | 0                                         | 0                                                 | 0                                             | 0                                             | 0                                 | 0                                          | 0                                               | 0                                                                        | 0                                    | 0                                   |
| 2   | 769         | 29.000000 | 24.150000 | 1       | 1         | 0            | 1        | 0        | 0        | 0        | ... | 0                                         | 0                                                 | 0                                             | 0                                             | 0                                 | 0                                          | 0                                               | 0                                                                        | 0                                    | 0                                   |
| 3   | 256         | 29.000000 | 15.245800 | 0       | 1         | 0            | 0        | 1        | 0        | 0        | ... | 0                                         | 0                                                 | 0                                             | 0                                             | 0                                 | 0                                          | 0                                               | 0                                                                        | 0                                    | 0                                   |
| 4   | 131         | 33.000000 | 7.895800  | 0       | 1         | 0            | 1        | 0        | 0        | 0        | ... | 0                                         | 0                                                 | 0                                             | 0                                             | 0                                 | 0                                          | 0                                               | 0                                                                        | 0                                    | 0                                   |
| ... | ...         | ...       | ...       | ...     | ...       | ...          | ...      | ...      | ...      | ...      | ... | ...                                       | ...                                               | ...                                           | ...                                           | ...                               | ...                                        | ...                                             | ...                                                                      | ...                                  | ...                                 |
| 653 | 85          | 32.849371 | 42.270620 | 0       | 1         | 0            | 0        | 0        | 0        | 0        | ... | 0                                         | 0                                                 | 0                                             | 0                                             | 0                                 | 0                                          | 0                                               | 0                                                                        | 0                                    | 0                                   |
| 654 | 745         | 29.625830 | 19.360890 | 0       | 0         | 0            | 1        | 0        | 0        | 0        | ... | 0                                         | 0                                                 | 0                                             | 0                                             | 0                                 | 0                                          | 0                                               | 0                                                                        | 0                                    | 0                                   |
| 655 | 74          | 23.973519 | 10.365732 | 0       | 1         | 0            | 0        | 0        | 0        | 0        | ... | 0                                         | 0                                                 | 0                                             | 0                                             | 0                                 | 0                                          | 0                                               | 0                                                                        | 0                                    | 0                                   |
| 656 | 759         | 5.722371  | 12.475000 | 0       | 0         | 0            | 0        | 0        | 0        | 0        | ... | 0                                         | 0                                                 | 0                                             | 0                                             | 0                                 | 0                                          | 0                                               | 0                                                                        | 0                                    | 0                                   |
| 657 | 37          | 24.867843 | 7.803387  | 0       | 1         | 0            | 1        | 0        | 0        | 0        | ... | 0                                         | 0                                                 | 0                                             | 0                                             | 0                                 | 0                                          | 0                                               | 0                                                                        | 0                                    | 0                                   |

658 rows × 1112 columns

```
X,y = data_manager.get_validation_data()
```

```
exp_mang = ExperimentManager(data_manager, optimize_metric = MetricName.Accuracy)
```

```
exp_mang.display_models_pool()
```

|   | Name                      | Description      | Source | Is Available |
| - | ------------------------- | ---------------- | ------ | ------------ |
| 0 | Baseline Classification   | Default settings | Pool   | True         |
| 1 | XGBoost                   | Default settings | Pool   | True         |
| 2 | Decision Tree             | Default settings | Pool   | True         |
| 3 | Logistic Regression       | Default settings | Pool   | True         |
| 4 | LightGBM                  | Default settings | Pool   | True         |
| 5 | Random Forest             | Default settings | Pool   | True         |
| 6 | Support Vector Classifier | Default settings | Pool   | True         |
| 7 | Gradient Boosting         | Default settings | Pool   | True         |
| 8 | AdaBoost Classifier       | Default settings | Pool   | True         |

```
exp_mang.display_metrics_pool()
```

|   | Name              | Description | Source | Is Available | Is Optimal |
| - | ----------------- | ----------- | ------ | ------------ | ---------- |
| 0 | Accuracy          |             | Pool   | True         | True       |
| 1 | Balanced Accuracy |             | Pool   | True         | False      |
| 2 | F1                |             | Pool   | True         | False      |
| 3 | AUC               |             | Pool   | True         | False      |
| 4 | Recall            |             | Pool   | True         | False      |
| 5 | Precision         |             | Pool   | True         | False      |

```
exp_mang = ExperimentManager(data_manager, optimize_metric = MetricName.Accuracy)
exp_mang.run_experiment()
```

|   | Experiment ID        | Experiment Description | Model                     | Model Description | Data Version | Data Description | Model Params                                                                                                                            | Metric Params | Accuracy | AUC      | Recall   | Precision | Balanced Accuracy | F1       | Run Time |
| - | -------------------- | ---------------------- | ------------------------- | ----------------- | ------------ | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------- | -------- | -------- | -------- | --------- | ----------------- | -------- | -------- |
| 0 | 20240320184038\_6c09 |                        | Baseline Classification   | Default settings  | c75f6168     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.477528 | 0.464973 | 0.411765 | 0.345679  | 0.464973          | 0.375839 | 0:00:00  |
| 1 | 20240320184038\_6c09 |                        | Logistic Regression       | Default settings  | c75f6168     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.825843 | 0.825401 | 0.823529 | 0.746667  | 0.825401          | 0.783217 | 0:00:01  |
| 2 | 20240320184038\_6c09 |                        | Support Vector Classifier | Default settings  | c75f6168     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.685393 | 0.602273 | 0.250000 | 0.772727  | 0.602273          | 0.377778 | 0:00:01  |
| 3 | 20240320184038\_6c09 |                        | AdaBoost Classifier       | Default settings  | c75f6168     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.814607 | 0.810695 | 0.794118 | 0.739726  | 0.810695          | 0.765957 | 0:00:00  |
| 4 | 20240320184038\_6c09 |                        | Decision Tree             | Default settings  | c75f6168     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.814607 | 0.796658 | 0.720588 | 0.777778  | 0.796658          | 0.748092 | 0:00:00  |
| 5 | 20240320184038\_6c09 |                        | Random Forest             | Default settings  | c75f6168     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.814607 | 0.813503 | 0.808824 | 0.733333  | 0.813503          | 0.769231 | 0:00:00  |
| 6 | 20240320184038\_6c09 |                        | Gradient Boosting         | Default settings  | c75f6168     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.831461 | 0.813102 | 0.735294 | 0.806452  | 0.813102          | 0.769231 | 0:00:01  |
| 7 | 20240320184038\_6c09 |                        | XGBoost                   | Default settings  | c75f6168     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.825843 | 0.805749 | 0.720588 | 0.803279  | 0.805749          | 0.759690 | 0:00:01  |
| 8 | 20240320184038\_6c09 |                        | LightGBM                  | Default settings  | c75f6168     |                  | {'experiment\_manager': Prediction Type: PredictionType.BinaryClassification, Metric: Metric: Accuracy. Description: , Random State: 0} | {}            | 0.820225 | 0.795588 | 0.691176 | 0.810345  | 0.795588          | 0.746032 | 0:00:00  |
