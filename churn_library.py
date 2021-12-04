"""
Author: Enes Deumic
Date: 2021-12-01

Library contains a set of functions for:
   - loading and transforming the data,
   - modeling on the transformed data,
   - evaluation of the models,
   - creating and saving the plots related to exploratory data analysis
     and model performance.
"""

import shap
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import plot_roc_curve, classification_report
import constants


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_df: pandas dataframe
    '''
    data_df = pd.read_csv(pth)
    data_df.loc[:, 'Churn'] = data_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return data_df


def perform_eda(data_df):
    '''
    perform eda on df and save figures to images folder
    input:
            data_df: pandas dataframe
    output:
            None
    '''
    for column in constants.SNS_DENSITY_COLUMNS:
        fig = plt.figure(figsize=constants.FIG_SIZE)
        density_plot = sns.histplot(
            data_df[column], stat='density', kde=True).get_figure()
        density_plot.savefig("./images/eda/{}_density.png".format(column))
    for column in constants.MPL_BAR_COLUMNS:
        fig = plt.figure(figsize=constants.FIG_SIZE)
        data_df[column].value_counts().plot.bar(
            title=column, xlabel=column, ylabel='Frequency')
        fig.savefig("./images/eda/{}_bar.png".format(column))

    for column in constants.MPL_HIST_COLUMNS:
        fig = plt.figure(figsize=constants.FIG_SIZE)
        ax = data_df[column].hist()
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        fig.savefig("./images/eda/{}_hist.png".format(column))

    heat_map = sns.heatmap(data_df.corr(), annot=False,
                           cmap='Dark2_r', linewidths=2)
    fig = heat_map.get_figure()
    fig.savefig("./images/eda/heatmap.png")


def create_preprocessing_pipeline(categorical, numerical):
    '''
    Creates sklearn transformers which transform columns differently depending
    on weather feature is numerical or categorical.

    Categorical features are onehot encoded while numerical scaled using
    to 0 mean and 1 variance.

    input:
            categorical: list of strings - categorical features
            response: list of strings - numerical features
    output:
            column_transformer: sklearn.compose.ColumnTransformer transformer
    '''
    column_transformer = ColumnTransformer(
        [
            ('numerical', StandardScaler(), numerical),
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical)
        ]
    )
    return column_transformer


def create_model_pipeline(estimator, preprocessing):
    '''
    Combines preprocessing pipeline and a model to a single transformer.

    input:
            estimator: any sklearn model, for example RandomForestClassifier
            preprocessing: sklearn transformer
            response: list of numerical features
    output:
            combined_pipeline: sklearn transformer, which contains both
                preprocessing transformer and model transformer.
    '''
    pipeline = Pipeline(
        [
            ('preprocessing', preprocessing),
            ('model', estimator)
        ]
    )
    return pipeline


def split_data(data, response):
    '''
    input:
              data: pandas dataframe
              response: string of response name [optional argument that could be
              used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    response_values = data[response].values
    X_train, X_test, y_train, y_test = train_test_split(
        data, response_values, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def plot_classification_report(
        y_train,
        y_test,
        y_train_preds,
        y_test_preds,
        model_name):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds: training predictions
            y_test_preds: test predictions
            model_name: str, model name string for classification report,
                for example "Random Forest"
    output:
             None
    '''
    plt.figure(figsize=(10, 6))
    plt.rc('figure', figsize=(7, 5))
    plt.text(0.01, 1.25, str('{} Train'.format(model_name)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_test, y_test_preds)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('{} Test'.format(model_name)), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/{}_classification_report.png'.format(
        model_name.lower().replace(" ", "")))


def plot_roc_curves(models, X_test, y_test):
    '''
    Plots roc curves for a list of models on the same dataset.
    input:
            models: list of sklearn estimators to be evaluated
            X_test: X testing data
            y_test:  test response values
    output:
             None
    '''
    plt.figure(figsize=(15, 8))
    ax = plt.gca()

    # set different transparency level, starting from full line reducing by
    # 0.2, with minimum value of 0.4
    alpha = 1.0

    for model in models:
        estimator_name = type(model[1]).__name__
        plot_roc_curve(model, X_test, y_test, ax=ax, alpha=max(0.4, alpha),
                       name=estimator_name)
        alpha -= 0.2
    plt.savefig('./images/results/model_comparison_roc_curves.png')


def feature_importance_plot(model, X_data):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
    output:
             None
    '''
    preprocessing = model[0]
    estimator = model[1]
    feature_names = preprocessing.get_feature_names_out()
    X_transformed = pd.DataFrame(
        preprocessing.transform(X_data),
        columns=feature_names)

    explainer = shap.TreeExplainer(estimator)
    # put to approximate for speed
    shap_values = explainer.shap_values(X_transformed, approximate=True)

    plt.figure(figsize=(16, 8))
    shap.summary_plot(
        shap_values,
        X_transformed,
        plot_type="bar",
        show=False,
        plot_size=(
            20,
            12))
    plt.tight_layout()
    plt.savefig('./images/results/variable_importance_shap.png')
    plt.close()

    importances = estimator.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [feature_names[i] for i in indices]
    plt.figure(figsize=(16, 10))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(len(names)), importances[indices])
    plt.xticks(range(len(names)), names, rotation=90)
    plt.tight_layout()
    plt.savefig('./images/results/variable_importance_bar_chart.png')
    plt.close()


def find_best_estimator(model_pipeline, param_grid, X_train, y_train):
    '''
    Finds the best estimator.

    input:
            model_pipeline: model_pipeline
            param_grid: dict, of parameter values specific to the estimator
            X_train: pandas dataframe of X values
            y_train: response variable
            cv: int, number of folds in cross validation
            estimator_label: str, designating model name in the pipeline
    output:
             estimator: best estimator
    '''
    param_grid = {constants.ESTIMATOR_LABEL + "__{}".format(
        param_name): param_val for param_name, param_val in param_grid.items()}
    grid_search = GridSearchCV(
        model_pipeline, param_grid, n_jobs=-1, cv=constants.CV)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    estimator_rf = RandomForestClassifier(random_state=42)
    preprocessing_pipeline = create_preprocessing_pipeline(
        constants.CAT_COLUMNS, constants.QUANT_COLUMNS
    )
    model_pipeline_rf = create_model_pipeline(
        estimator_rf, preprocessing_pipeline)
    rf_model = find_best_estimator(model_pipeline_rf, constants.RF_PARAM_GRID,
                                   X_train, y_train)
    joblib.dump(rf_model, './models/rf_model.pkl')

    estimator_lr = LogisticRegression(
        penalty='elasticnet', solver='saga', max_iter=1000)
    model_pipeline_lr = create_model_pipeline(
        estimator_lr, preprocessing_pipeline)
    lr_model = find_best_estimator(model_pipeline_lr, constants.LR_PARAM_GRID,
                                   X_train, y_train)
    joblib.dump(lr_model, './models/logistic_model.pkl')

    y_train_preds_rf = rf_model.predict(X_train)
    y_test_preds_rf = rf_model.predict(X_test)
    y_train_preds_lr = lr_model.predict(X_train)
    y_test_preds_lr = lr_model.predict(X_test)

    plot_classification_report(
        y_train,
        y_test,
        y_train_preds_rf,
        y_test_preds_rf,
        "Random Forest")
    plot_classification_report(
        y_train,
        y_test,
        y_train_preds_lr,
        y_test_preds_lr,
        "Logistic Regression")

    plot_roc_curves([rf_model, lr_model], X_test, y_test)
    feature_importance_plot(rf_model, X_test)


if __name__ == '__main__':
    DATA_DF = import_data('data/bank_data.csv')
    perform_eda(DATA_DF)

    preprocessing = create_preprocessing_pipeline(
        constants.CAT_COLUMNS, constants.QUANT_COLUMNS
    )
    model = RandomForestClassifier(random_state=42)
    model_pipeline = create_model_pipeline(model, preprocessing)

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = split_data(DATA_DF, "Churn")
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
