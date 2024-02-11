<div align="center">

<img src="/docs/examples/resources/SimpML_Logo.png" alt="SimpML Logo" width="200" height="200"/>

## **SimpML: Simplifying Machine Learning**

<p align="center">
<h3>
  <a href="https://simpml.gitbook.io/">Docs</a> •
  <a href="https://simpml.gitbook.io/docs/get-started">Tutorials</a> •
  <a href="https://simpml.gitbook.io/docs/blog">Blog</a> •
</h3>
</p>

| Overview | |
|---|---|
| **License** | [![License](https://img.shields.io/pypi/l/ansicolortags.svg)](https://img.shields.io/pypi/l/ansicolortags.svg)

<div align="left">
    
# Welcome to SimpML
SimpML is a robust framework designed to streamline the process of training machine learning models regardless of the type using no/low-code consept. Its flexible infrastructure allows for the implementation and integration of various components, catering to a wide array of machine learning tasks.

SimpML revolutionizes machine learning development by enabling the training of any model on any type of data through its versatile infrastructure. At its core, SimpML simplifies the process to such an extent that training a series of diverse models and conducting thorough evaluations can be achieved with just a single line of code. This streamlined approach not only accelerates the model development cycle but also democratizes advanced machine learning capabilities, making them accessible to developers of all skill levels.

Beyond its foundational simplicity, SimpML is complemented by a comprehensive suite of ready-made implementations. These pre-configured solutions are meticulously designed to tackle a wide array of challenges commonly encountered in machine learning projects, covering classification, regression, and root cause analysis across various data domains, including tabular, time series, and image data. Additionally, for those looking to leverage the latest advancements in natural language processing, SimpML provides an easy-to-use method for fine-tuning language models from Hugging Face with Lora.

By merging this ease of use with a rich collection of out-of-the-box functionalities, SimpML stands as a powerful tool that streamlines the end-to-end machine learning workflow. It ensures that from the initial training phase to the deployment in production environments, developers can maintain a high level of efficiency and effectiveness, all while minimizing the code complexity typically associated with such tasks.

## Core Components
To use the SimpML framework for training custom machine learning models, you need to implement the following interfaces:

### Model Manager
A wrapper with a standard API (Similar to SKlearn) for a machine learning model

### Data Manager:
Manages data preprocessing, loading, and split into train and test, to ensure data is in the right format for model training.

### Metric Manager
 Responsible for evaluating model performance through different metrics, providing insights into model accuracy and effectiveness.

These components are designed with interfaces that ensure compatibility and interoperability within the SimpML ecosystem. It is essential that users ensure these components work seamlessly together to leverage the full power of the experiment manager for efficient model training because the Model Manager has to train on the Data Manager and the Metric Manager has to calculate the index on what comes back from the model.
To verify this you can use our personal assistant [will be added soon]

## Specialized Interfaces for Problem Families
SimpML offers ready-made interfaces for specific problem families, significantly reducing the development time for common machine learning tasks:

### Tabular
For tabular data, SimpML supports a wide range of tasks including supervised learning, unsupervised learning, and anomaly detection. This interface simplifies data manipulation, feature extraction, and model selection for tabular datasets, enabling efficient handling of both numerical and categorical data.

#### Preprocessing

 - Pivot: Transforming data from a long to wide format.
 - Imputation: Handling missing values in the dataset.
 - Data split: Splitting the data into train and test sets (random, time-based, etc.).
 - Encoding: Encoding categorical variables.
 - Balancing: Addressing class imbalance in the data.
 - And many more preprocessing techniques.

#### Interpretation
The interpretation component of SimpML includes various visualizations and error analysis tools to understand your trained models. Some of the features include:

 - Feature importance: Identifying the most important features in your models.
 - Bias-variance analysis: Assessing the bias and variance trade-off in your models.
 - Leakage detector: Detecting potential data leakage issues.
 - Local (SHAP-based) and global interpretation: Understanding the impact of features on individual predictions and overall model behavior.
 - Identification of "bad" features: Identifying features that negatively affect model performance.
 - And many more tools to gain insights into your models.

### Vision

The vision interface in SimpML is built upon the fastai library, providing advanced functionalities for image classification and segmentation tasks. This interface makes it easier to work with deep learning models for computer vision, leveraging fastai's dynamic and high-level capabilities.

#### Full fastai integration
All the capabilities that are available in fastai are also easily available in SimpML NV which makes the process easy and flexible

### LLM (Large Language Models)
SimpML integrates with Huggingface-based models, offering a streamlined process for fine-tuning LLMs with LORA for a variety of NLP tasks. This interface simplifies the complexities of working with large language models, making it accessible to fine-tune them for specific applications.

#### Model Fine-tuning: 
Simplified process for applying LORA adjustments to pre-trained models.

## Installation
SimpML can be installed as a whole package or in parts, based on the specific requirements of your project. Use the following pip commands for installation:

### **Core Framework:**
This installation will install the SimpML framework on which specific use cases can be built
```python
# install SimpML
pip install SimpML
```
Additionally, you can install the code and dependencies required to use SimpML's ready-made use cases:

### **Tabular:**
```python
# install SimpML Tabular use cases
pip install SimpML[Tabular]
```
#### **Dependencies**
SimpML[Tabular] is built as a Python wrapper around industry-standard machine learning libraries, including:

 - Scikit-learn
 - XGBoost
 - Optuna
 - SHAP
 - Imbalanced-learn

These libraries provide robust and efficient implementations of various machine learning algorithms and techniques. SimpML leverages the best practices from real-world experience while offering flexibility to implement custom logic and easily share it between data scientists in an organization.

### **Vision:**
```python
# install SimpML Vision use cases
pip install SimpML[Vision]
```
The Vision package is designed for computer vision applications, leveraging PyTorch and FastAI for deep learning. It requires the torch, fastai, and torchvision libraries for image processing and model training.


### **LLM (Large Language Models):**
```python
# install SimpML LLM use cases
pip install SimpML[LLM]
```
The LLM package facilitates work with large language models, utilizing the Hugging Face transformers and datasets libraries. These dependencies enable easy access to pre-trained models and datasets for natural language processing tasks.


## Getting Started
To get started with SimpML, dive into the comprehensive documentation available on the official SimpML website. The documentation provides detailed guides on implementing the core components, utilizing the specialized interfaces for various problem domains, and integrating SimpML into your machine learning workflow for efficient model development and evaluation.

## Contributing
SimpML is an open-source project, and contributions are welcome. If you encounter any issues, have feature requests, or would like to contribute to the development of SimpML, please visit the GitHub repository here.

## License
SimpML is released under the MIT License. See the LICENSE file for more details.

## Contact
For any inquiries or support related to SimpML, please contact the SimpML team at contact@simpml.com. We would be happy to assist you.

## Acknowledgements
SimpML is built upon the hard work and contributions of various open-source projects and the vibrant machine learning community. We extend our gratitude to the developers of the underlying libraries that power SimpML and the data scientists who continue to push the boundaries of machine learning.
