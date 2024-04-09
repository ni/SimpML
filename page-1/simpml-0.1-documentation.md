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
from simpml.vision.all import *
%matplotlib inline
```

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAWCAYAAAA1vze2AAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAdxJREFUeNq0Vt1Rg0AQJjcpgBJiBWIFkgoMFYhPPAIVECogPuYpdJBYgXQQrMCUkA50V7+d2ZwXuXPGm9khHLu3f9+3l1nkWNvtNqfHLgpfQ1EUS3tz5nAQ0+NIsiAZSc6eDlI8M3J00B/mDuUKDk6kfOebAgW3pkdD0pFcODGW4gKKvOrAUm04MA4QDt1OEIXU9hDigfS5rC1eS5T90gltck1Xrizo257kgySZcNRzgCSxCvgiE9nckPJo2b/B2AcEkk2OwL8bD8gmOKR1GPbaCUqxEgTq0tLvgb6zfo7+DgYGkkWL2tqLDV4RSITfbHPPfJKIrWz4nJQTMPAWA7IbD6imcNaDeDfgk+4No+wZr40BL3g9eQJJCFqRQ54KiSt72lsLpE3o3MCBSxDuq4yOckU2hKXRuwBH3OyMR4g1UpyTYw6mlmBqNdUXRM1NfyF5EPI6JkcpIDBIX8jX6DR/6ckAZJ0wEAdLR8DEk6OfC1Pp8BKo6TQIwPJbvJ6toK5lmuvJoRtfK6Ym1iRYIarRo2UyYHvRN5qpakR3yoizWrouoyuXXQqI185LCw07op5ZyCRGL99h24InP0e9xdQukEKVmhzrqZuRIfwISB//cP3Wk3f8f/yR+BRgAHu00HjLcEQBAAAAAElFTkSuQmCC)

### Binary Classification[](broken-reference)

#### Standard Trainer[](broken-reference)

**Data Manager**[****](broken-reference)

We are going to use the FastAI library to set up a data manager for image classification tasks.

We start by importing the necessary modules from the FastAI library, which is a high-level library built on top of PyTorch.

Next, we create a VisionDataManager object to manage the data pipeline for a binary classification task. To do this, we use the ImageDataLoaders.from\_name\_func method to load the images from the PETS dataset, which can be downloaded using the untar\_data function with the URLs.PETS argument.

The from\_name\_func method takes three arguments:

1. The path to the dataset, which is obtained by untarring the PETS dataset.
2. A list of image files, retrieved using the get\_image\_files function.
3. A function that serves as a label extractor. In this case, we use a lambda function that checks if the first character of the file name is uppercase. If it is, the image is considered to belong to class 1, and if not, it belongs to class 0.

Additionally, we apply an image transformation to resize all images to 224x224 pixels using the Resize(224) method.

Finally, we specify the prediction\_type and data\_type as binary classification and vision, respectively, to properly configure the VisionDataManager.

This data manager can now be used to train and evaluate a binary image classification model using the FastAI library.

```
def is_first_char_upper(f): return f[0].isupper()

data_manager = VisionDataManager(ImageDataLoaders.from_name_func(
    untar_data(URLs.PETS),
    get_image_files(untar_data(URLs.PETS)/"images"),
    is_first_char_upper,
    item_tfms=Resize(224)),
    prediction_type = PredictionType.BinaryClassification.value,
    data_type = DataType.Vision.value
)
```

We are going to visualize a batch of images from the dataset using the FastAI library.

We have already set up the data manager in the previous snippet, which handles the data pipeline for a binary classification task using the PETS dataset.

To display a batch of images along with their labels, we use the show\_batch() method of the DataLoaders object stored in the data\_manager under the dls attribute.

This method generates a grid of images, showing a random sample of images from the dataset along with their corresponding labels. This helps to get a better understanding of the dataset’s content and its distribution among the two classes.

```
data_manager.dls.show_batch()
```

**Experiment Manager**[****](broken-reference)

We are going to create an ExperimentManager object that will help us manage the training and evaluation process.

The ExperimentManager class provides a convenient way to handle the modeling, training, and evaluation of experiments, making the workflow easier to manage.

In this case, we will use the Area Under the Receiver Operating Characteristic curve (AUC) as the evaluation metric, which is a popular choice for binary classification tasks.

We then create the ExperimentManager object, passing in the data\_manager we have set up previously and the desired evaluation metric MetricName.AUC. This exp\_mang object can now be used to configure, train, and evaluate models.

```
exp_mang = ExperimentManager(data_manager, MetricName.AUC)
```

```
exp_mang.display_models_pool()
```

|   | Name                           | Description                         | Source | Is Available |
| - | ------------------------------ | ----------------------------------- | ------ | ------------ |
| 0 | Resnet-18                      | Resnet-18 Pretrained Model ImageNet | Pool   | True         |
| 1 | Random Baseline Classification | Default settings                    | Pool   | True         |
| 2 | Naive Baseline Classification  | Default settings                    | Pool   | True         |
| 3 | Resnet-50                      | Resnet-50 Pretrained Model ImageNet | Pool   | True         |
| 4 | Resnet-34                      | Resnet-34 Pretrained Model ImageNet | Pool   | True         |

```
exp_mang.display_metrics_pool()
```

|   | Name              | Description | Source | Is Available | Is Optimal |
| - | ----------------- | ----------- | ------ | ------------ | ---------- |
| 0 | AUC               |             | Pool   | True         | True       |
| 1 | Balanced Accuracy |             | Pool   | True         | False      |
| 2 | F1                |             | Pool   | True         | False      |
| 3 | Accuracy          |             | Pool   | True         | False      |
| 4 | Recall            |             | Pool   | True         | False      |
| 5 | Precision         |             | Pool   | True         | False      |

**Let’s examine the models available for this type of problem**[****](broken-reference)

We are going to access the models\_pool property of the ExperimentManager (exp\_mang) object. The models\_pool attribute is a list containing all available for training for the defined problem type and data

In order not to occupy too much GPU RAM, we will use the save\_models\_to\_disk flag and provide the class with two functions: - empty\_cache\_func - load\_checkpoints\_func

So that the department knows how to load into memory models saved with the help of export and you can clean the memory efficiently.ExperimentManager

```
exp_mang = ExperimentManager(data_manager,
                             MetricName.AUC,
                             save_models_to_disk = True,
                             load_checkpoints_func = load_fastai_model,
                             empty_cache_func = empty_torch_cache,
                        )
#exp_mang.run_experiment(models_kwargs={'num_epocs': 5})
```

By calling exp\_mang.models\_pool, we retrieve a list that includes a brief description of each model along with the actual model object (for FastAI-based models, these are Learner objects).

The list shows that the experiment manager initialized five different models for us:

* Random Baseline Classification (default settings)
* Naive Baseline Classification (default settings)
* Resnet-18 (pretrained on ImageNet)
* Resnet-34 (pretrained on ImageNet)
* Resnet-50 (pretrained on ImageNet)

With this list, we can easily access any of the trained models for further analysis, fine-tuning, or deployment.

In this code snippet, we are going to use the ExperimentManager object created earlier to train and evaluate a suite of machine learning models with the simpml library. The fit\_suite method is a convenient way to train multiple models using the same dataset, allowing us to compare their performance on the specified evaluation metric(s).

We call the fit\_suite method on the exp\_mang object and pass a dictionary containing the keyword argument ‘num\_epochs’: 5 to set the number of training epochs for each model. The fit\_suite method will then train and evaluate each model in the suite using the data provided by the data\_manager and the evaluation metric specified during the creation of the ExperimentManager object, which in this case is the AUC.

Once the training and evaluation process is complete, the fit\_suite method returns a summary of the results in a tabular format. This summary includes information such as the Run ID, model description, data version, model parameters, evaluation metric values (e.g., accuracy, AUC, recall, precision, balanced accuracy, and F1 score), and run time for each model. By reviewing this summary, we can easily compare the performance of different models and choose the one that best fits our requirements.

In this code snippet, we are going to use the ExperimentManager object (exp\_mang) to find and retrieve the best model from the pool of trained models, based on the evaluation metric specified earlier (AUC in this case).

We call the get\_best\_model() method on the exp\_mang object to find the model with the highest performance on the specified evaluation metric. The method returns the best model as a Learner object (for FastAI-based models) along with its description.

```
#best_model = exp_mang.get_best_model()
```

In this example, the best model is the Resnet-34 pretrained on ImageNet. Once we have identified the best model, we can use it for further analysis, fine-tuning, or deployment.

**Interpreter**[****](broken-reference)

```
#interp = ClassificationInterpretation.from_learner(best_model.model)
```

```
#interp.show_results(list(range(12)))
```

```
#interp.print_classification_report()
```

```
#interp.plot_confusion_matrix()
```

```
#interp.plot_top_losses(4)
```

#### Cross-Validation[](broken-reference)

**data\_manager**[****](broken-reference)

untar\_data(URLs.PETS)In this section, we extend our model evaluation process by incorporating cross-validation. We’ve already covered the basics of model building, optimization, and performance comparison. Now, let’s add a layer of robustness to our assessments with cross-validation.

Cross-validation is a powerful technique that helps prevent overfitting to a specific data distribution, ensuring our model selection is more reliable.

```
def is_first_char_upper(f):
    # Convert PosixPath to string and then check if the first character is uppercase
    return str(f).split('/')[-1][0].isupper()

data_manager = CrossValidationVisionDataManager(
                    dls = ImageDataLoaders.from_name_func(
                        untar_data(URLs.PETS),
                        get_image_files(untar_data(URLs.PETS)/"images"),
                        is_first_char_upper,
                        item_tfms=Resize(224)),
                    cross_validation_splitter = FastaiCrossValidationSplitterAdapter(get_y=is_first_char_upper, n_folds=5),
                    prediction_type = PredictionType.BinaryClassification.value,
                    data_type = DataType.Vision.value
                )
```

```
len(data_manager.dls_folds_list)
```

**Experiment Manager**[****](broken-reference)

Now we can train models using a cross-validation technique. We will use exactly the same code as before, but we will add a variable for trainer and initialize a variable that is suitable for cross validation

```
exp_mang = ExperimentManager(data_manager,
                             MetricName.AUC,
                             save_models_to_disk = True,
                             load_checkpoints_func = load_fastai_model,
                             empty_cache_func = empty_torch_cache,
                             trainer = CVTrainer(aggregation = CVAggregation.MEAN, selected_model = CVSelectedModel.BEST)
                        )
```

In our CVTrainer, we are specifying two critical parameters:

aggregation: This parameter determines how to combine the performance metrics across the different folds. We can choose between ‘mean’, ‘max’, or ‘min’. Here we’ve selected ‘mean’, which means the performance metrics in our results table will be the average performance across all folds.

selected\_model: This parameter controls which model instance is returned from the cross-validation folds. It could either be the ‘best’ or the ‘worst’ performing model.

Once our experiment manager is defined, we can execute the experiment. Now, each model is not just trained and evaluated once, but across different distributions of the data.

```
#exp_mang.run_experiment(models_kwargs={'num_epocs': 5})
```

The table we see shows the indices according to the aggregation we chose earlier (average, in this case). To go a little deeper and compare performance between the different folds, we can use the plots that the trainer creates for us:

```
#_ = exp_mang.trainer.box_plot_per_model_and_metric('Resnet-50', 'AUC')
```

```
#_ = exp_mang.trainer.plot_cv_res()
```

```
#_ = exp_mang.trainer.display_best_models()
```
