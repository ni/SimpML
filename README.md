# SimpML
SimpML is an open-source, no/low-code machine learning library in Python that simplifies and automates machine learning workflows. With SimpML, you can quickly build your desired machine learning pipeline with just a few lines of code, drastically reducing your time-to-model.

## Features
### Preprocess
The preprocess component of SimpML includes all the necessary steps for data preprocessing, such as:

 - Pivot: Transforming data from a long to wide format.
 - Imputation: Handling missing values in the dataset.
 - Data split: Splitting the data into train and test sets (random, time-based, etc.).
 - Encoding: Encoding categorical variables.
 - Balancing: Addressing class imbalance in the data.
 - And many more preprocessing techniques.

Once you build your preprocess pipeline, it will automatically run on your inference data.

### Modelling
SimpML provides a rich experiment manager for modeling tasks, including:

 - Hyperparameter optimization: Automatically finding the best hyperparameters for your models.
 - Cross-model and cross-dataset comparison: Comparing the performance of different models on different datasets.
 - Full integration with ML-Flow: Tracking and managing your experiments with MLflow.
 - And many more features to streamline the modeling process.

### Interpretation
The interpretation component of SimpML includes various visualizations and error analysis tools to understand your trained models. Some of the features include:

 - Feature importance: Identifying the most important features in your models.
 - Bias-variance analysis: Assessing the bias and variance trade-off in your models.
 - Leakage detector: Detecting potential data leakage issues.
 - Local (SHAP-based) and global interpretation: Understanding the impact of features on individual predictions and overall model behavior.
 - Identification of "bad" features: Identifying features that negatively affect model performance.
 - And many more tools to gain insights into your models.

## Getting Started
To get started with SimpML, refer to the documentation and video training available at the SimpML website here. The documentation provides detailed explanations of the library's functionalities and how to use them effectively.

## Dependencies
SimpML is built as a Python wrapper around industry-standard machine learning libraries, including:

 - Scikit-learn
 - XGBoost
 - Optuna
 - SHAP
 - Imbalanced-learn

These libraries provide robust and efficient implementations of various machine learning algorithms and techniques. SimpML leverages the best practices from real-world experience while offering flexibility to implement custom logic and easily share it between data scientists in an organization.

## Contributing
SimpML is an open-source project, and contributions are welcome. If you encounter any issues, have feature requests, or would like to contribute to the development of SimpML, please visit the GitHub repository here.

## License
SimpML is released under the MIT License. See the LICENSE file for more details.

## Contact
For any inquiries or support related to SimpML, please contact the SimpML team at contact@simpml.com. We would be happy to assist you.

## Acknowledgements
SimpML is built upon the hard work and contributions of various open-source projects and the vibrant machine learning community. We extend our gratitude to the developers of the underlying libraries that power SimpML and the data scientists who continue to push the boundaries of machine learning.