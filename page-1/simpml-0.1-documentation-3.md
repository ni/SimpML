# SimpML 0.1 documentation

SimpML

### Welcome to SimpML[](broken-reference)



SimpML is an open source No/low code machine learning library in python that automates machine learning workflows.

By using NIAI you build your desired machine learning pipeline with few lines of code which shorten your time-to-model dramatically.

More info including video training can be found here [SimpML](https://optimalplus.atlassian.net/wiki/spaces/DS/pages/3431432203/NIAI.Tabular).

### About SimpML[](broken-reference)

SimpML design was inspired by real life data scientists workflows and challenges.

It natively supports the two paradigms in machine learning of data centric approach and model centric approach at one place.

After building model training pipeline it enables you to easily package it and deploy inference pipeline to production.

SimpML is essentially a python wrapper of industry standards machine learning libraries such as:

* Scikit-learn
* XGboost
* Otuna
* Shap
* Imbalanced-learn
* and more

While using best practices from real world experience it has flexibility to implement your own custom logics and share it easily between different data scientists in the organization.

### SimpML main components:[](broken-reference)

SimpML provides the main components:



#### Preprocess[](broken-reference)

* includes all needed preprocess steps needed such as:
  * pivot
  * imputation
  * data split (random, time-based, etc.)
  * encoding
  * balancing
  * and many more?
  * You build once your preprocess pipeline and it will run automatically on you inference data.

#### Modelling[](broken-reference)

* Rich experiment manager with:
  * hyper parameters optimizations
  * cross models and cross datasets comparison
  * full integration with ML-Flow
  * and many more

#### Interpretation[](broken-reference)

* includes rich visualizations and error analysis to understand your trained model such as:
  * features importance
  * bias-variance
  * leakage detector
  * local (SHAP-based) and global interpretation
  * identification of “bad” features that impair model performance
  * and many more.
