# SimpML 0.1 documentation

```
%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path
cwd = Path.cwd()
ROOT_PATH = str(cwd.parent.parent)
sys.path.append(ROOT_PATH)
from simpml.tabular.all import *
np.random.seed(0)
%matplotlib inline
```

```
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[1], line 9
      7 ROOT_PATH = str(cwd.parent.parent)
      8 sys.path.append(ROOT_PATH)
----> 9 from simpml.tabular.all import *
     10 np.random.seed(0)
     11 get_ipython().run_line_magic('matplotlib', 'inline')

ModuleNotFoundError: No module named 'simpml'
```

### Anomaly Detection[](broken-reference)

#### Data Manager[](broken-reference)

```
data = DataSet.load_titanic_dataset()
```

```
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[2], line 1
----> 1 data = DataSet.load_titanic_dataset()

NameError: name 'DataSet' is not defined
```

```
data_manager = UnsupervisedTabularDataManager(data = data,
                                            prediction_type = PredictionType.AnomalyDetection,
                                            target = 'Survived',
                                            splitter=RandomSplitter(target='Survived', split_sets={Dataset.Train: 0.6, Dataset.Valid: 0.2, Dataset.Test: 0.2}, stratify=True))
```

```
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[3], line 1
----> 1 data_manager = UnsupervisedTabularDataManager(data = data,
      2                                             prediction_type = PredictionType.AnomalyDetection,
      3                                             target = 'Survived',
      4                                             splitter=RandomSplitter(target='Survived', split_sets={Dataset.Train: 0.6, Dataset.Valid: 0.2, Dataset.Test: 0.2}, stratify=True))

NameError: name 'UnsupervisedTabularDataManager' is not defined
```

```
data_manager.build_pipeline()
```

```
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[4], line 1
----> 1 data_manager.build_pipeline()

NameError: name 'data_manager' is not defined
```

```
X, y = data_manager.get_training_data()
X
```

```
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[5], line 1
----> 1 X, y = data_manager.get_training_data()
      2 X

NameError: name 'data_manager' is not defined
```

```
X, y = data_manager.get_validation_data()
X
```

```
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[6], line 1
----> 1 X, y = data_manager.get_validation_data()
      2 X

NameError: name 'data_manager' is not defined
```

#### Experiment Manager[](broken-reference)

```
exp_mang = ExperimentManager(data_manager, optimize_metric = MetricName.AUC)
```

```
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[7], line 1
----> 1 exp_mang = ExperimentManager(data_manager, optimize_metric = MetricName.AUC)

NameError: name 'ExperimentManager' is not defined
```

```
exp_mang.display_models_pool()
```

```
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[8], line 1
----> 1 exp_mang.display_models_pool()

NameError: name 'exp_mang' is not defined
```

```
exp_mang.display_metrics_pool()
```

```
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[9], line 1
----> 1 exp_mang.display_metrics_pool()

NameError: name 'exp_mang' is not defined
```

```
def find_contamination(data_manager):
    data_manager.get_training_data()
    value_counts = data_manager.get_validation_data()[1].value_counts()
    total_samples = value_counts.sum()
    return (value_counts / total_samples)[1]
```

```
contamination = find_contamination(data_manager)
exp_mang.run_experiment(models_kwargs = {'contamination': contamination})
```

```
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[11], line 1
----> 1 contamination = find_contamination(data_manager)
      2 exp_mang.run_experiment(models_kwargs = {'contamination': contamination})

NameError: name 'data_manager' is not defined
```

```
exp_mang.get_best_model()
```

```
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[12], line 1
----> 1 exp_mang.get_best_model()

NameError: name 'exp_mang' is not defined
```

### Clustering[](broken-reference)

#### Data Manager[](broken-reference)

```
data_manager = UnsupervisedTabularDataManager(data = 'datasets/binary/Titanic.csv',
                                            prediction_type = PredictionType.Clustering,
                                            splitter='Random')
```

```
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[13], line 1
----> 1 data_manager = UnsupervisedTabularDataManager(data = 'datasets/binary/Titanic.csv',
      2                                             prediction_type = PredictionType.Clustering,
      3                                             splitter='Random')

NameError: name 'UnsupervisedTabularDataManager' is not defined
```

```
data_manager.build_pipeline()
```

```
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[14], line 1
----> 1 data_manager.build_pipeline()

NameError: name 'data_manager' is not defined
```

#### Experiment Manager[](broken-reference)

```
exp_mang = ExperimentManager(data_manager, optimize_metric = MetricName.CalinskiHarabasz)
```

```
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[15], line 1
----> 1 exp_mang = ExperimentManager(data_manager, optimize_metric = MetricName.CalinskiHarabasz)

NameError: name 'ExperimentManager' is not defined
```

```
exp_mang.display_models_pool()
```

```
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[16], line 1
----> 1 exp_mang.display_models_pool()

NameError: name 'exp_mang' is not defined
```

```
exp_mang.display_metrics_pool()
```

```
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[17], line 1
----> 1 exp_mang.display_metrics_pool()

NameError: name 'exp_mang' is not defined
```

```
exp_mang.run_experiment()
```

```
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[18], line 1
----> 1 exp_mang.run_experiment()

NameError: name 'exp_mang' is not defined
```

```
exp_mang.get_best_model()
```

```
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[19], line 1
----> 1 exp_mang.get_best_model()

NameError: name 'exp_mang' is not defined
```
