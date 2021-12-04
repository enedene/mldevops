"""
Author: Enes Deumic
Date: 2021-12-01

Testing the churn_library.py functions.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import churn_library as cl
import constants


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(pth):
    '''
    test loading raw modeling data
    input:
            pth: str, path to data
    output:
            None
    '''
    try:
        data_df = cl.import_data(pth)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data_df.shape[0] > 0
        assert data_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have" +
            " rows and columns")
        raise err

    try:
        modeling_columns = \
            set(constants.SNS_DENSITY_COLUMNS + constants.MPL_HIST_COLUMNS +
                constants.MPL_HIST_NORM_COLUMNS + constants.CAT_COLUMNS +
                constants.QUANT_COLUMNS)
        len(set([col for col in data_df.columns if col in modeling_columns]))
        logging.info("Modeling columns found: SUCCESS")
    except KeyError as err:
        logging.error(
            "Data DataFrame doesn't contain all the required columns")
        raise err


def test_eda(data_df):
    '''
    test perform eda function on input dataframe

    input:
            data_df: pandas dataframe raw data for modeling
    output:
            None
    '''
    try:
        cl.perform_eda(data_df)
        logging.info("Testing running perform_eda function: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Path provided to function not found.")
        raise err


def test_preprocessing_pipeline(df, categorical, numerical):
    '''
    test preprocessing pipeline:
        - creating sklearn.ColumnTransformer for transforming
            pandas.DataFrame columns
        - transforming input data using StandardScaler and
            OneHotEncoder
        - checking the validity of transformed values

    input:
            data_df: pandas dataframe raw data for modeling
            categorical: list of categorical feature names
            numerical: list of numerical feature names
    output:
            None
    '''
    try:
        pipeline = cl.create_preprocessing_pipeline(categorical, numerical)
        assert isinstance(pipeline, ColumnTransformer)
        logging.info("Testing create_preprocessing_pipeline function: SUCCESS")
    except AssertionError as err:
        logging.error("Transformation pipeline should be of ColumnTransformer")
        raise err

    try:
        output = pipeline.fit_transform(df[numerical + categorical])
        assert ~np.isnan(output).any()
        logging.info("Transformation successfuly completed.")
    except AssertionError as err:
        logging.error("Output contains nan values at positions:\n {}".format(
            np.where(np.isnan(output))))
        raise err

    try:
        assert output.shape[0] > 0
        assert output.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have " +
            "rows and columns")
        raise err


def test_train_test_split(data_df, response):
    '''
    test train test split conditioned on response variable

    input:
            data_df: pandas dataframe raw data for modeling
                categorical: list of categorical feature names
            response: str, name of a response variable
    output:
            None
    '''
    try:
        X_train, X_test, y_train, y_test = cl.split_data(data_df, response)
        assert X_train.shape[0] == 0
        assert X_test.shape[0] == 0
        assert y_train.shape[0] == X_train.shape[0]
        assert y_test.shape[0] == X_test.shape[0]
        assert X_train.shape[1] == X_test.shape[1]
        logging.info("Train test split - SUCCESS")
    except AssertionError as err:
        logging.error("Wrong shape of the train test split data.")
        raise err


def test_model_pipeline(data_df):
    '''
    tests full model pipeline one step at a time:
        - preprocessing the raw data for data suitable
            for modeling
        - fitting the model on transformed data
        - evaluating the model on test data and checking
            if results are better than random predictions
    input:
            data_df: pandas dataframe raw data for modeling
                categorical: list of categorical feature names
            response: str, name of a response variable
    output:
            None

    '''
    try:
        preprocessing = cl.create_preprocessing_pipeline(
            constants.CAT_COLUMNS, constants.QUANT_COLUMNS
        )
        pipeline = cl.create_model_pipeline(
            RandomForestClassifier(n_jobs=-1), preprocessing)
        assert isinstance(pipeline, Pipeline)
        logging.info("Testing create_model_pipeline function: SUCCESS")
    except AssertionError as err:
        logging.error("Modeling pipeline should be of Pipeline class.")
        raise err

    X_train, X_test, y_train, y_test = cl.split_data(data_df, 'Churn')

    try:
        pipeline.fit(X_train, y_train)
        logging.info("Model successfuly fit: SUCCESS")
    except Exception as err:
        logging.error("Model failed to train.")
        raise err

    try:
        predictions_proba = pipeline.predict_proba(X_test)[:, 1]
        assert np.any(predictions_proba)
    except AssertionError as err:
        logging.error("Output contains nan values at positions:\n {}".format(
            np.where(np.isnan(predictions_proba))))
        raise err

    score = roc_auc_score(y_test, predictions_proba)
    try:
        assert score >= 0.6
        logging.info("AUC scores = {} ".format(score) +
                     "indicating better than random performance")
    except AssertionError as err:
        logging.warning("AUC scores = {} < 0.6 ".format(score) +
                        "indicating close to random model performance")
        raise err


if __name__ == "__main__":
    DATA_PATH = "./data/bank_data.csv"
    test_import(DATA_PATH)

    DATA_DF = cl.import_data(DATA_PATH)
    test_eda(DATA_DF)

    NUMERICAL = ['col_1', 'col_2']
    CATEGORICAL = ['col_3', 'col_4']
    DF_TEST = pd.DataFrame({
        "col_1": np.arange(10),
        "col_2": np.arange(10, 20),
        "col_3": ["a"] * 5 + ['b'] * 5,
        "col_4": ["c"] * 4 + ['d'] * 3 + ['e'] * 3
    })

    test_preprocessing_pipeline(DF_TEST, CATEGORICAL, NUMERICAL)
    test_model_pipeline(DATA_DF)
