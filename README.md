# Predict Customer Churn

Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity.

## Project Description
In this project we create a customer churn prediction model for credit card owners based on the
[Kaggle dataset](https://www.kaggle.com/sakshigoyal7/credit-card-customers/code).

The purpose of the project is to work on the clean code principles. This includes abiding to PEP8, writing tests for each function and logging.

## Running Files
### Prerequisits
Before running the code install the packages contained in the `requirements.txt` by running:
`pip install -r requirements.txt`
in the python virtual enviroment.

To run exploratory data analysis, train and evaluate model performance. Run:
`python -m churn_library`
To test the functions run:
`python -m churn_script_logging_and_tests`

### Files in the Repo
* `churn_library.py` contains functions for running the exploratory data analysis, model training and creating plots of model performance.
    * `perform_eda` function performs exploratory data analysis and stores the plots in `images/eda/` directory.
    * `train_models` function which combines multiple steps:
        * `create_preprocessing_pipeline` and `create_model_pipeline` for creating preprocessing and modeling pipeline
        * `plot_classification_report`, `plot_roc_curves`, `feature_importance_plot` for plotting the model performance and storing the outputs to `images/results/` directory.
* `churn_script_logging_and_tests.py` testing the functions from `churn_library.py` and logging the outputs of the tests in `logs/churn_library.log` directory.
* `requirements.txt` python libraries and versions needed to run the code.
* `README.md` this file.
* `Guide.ipynb` and `churn_notebook.ipynb` analysis and instruction files.




