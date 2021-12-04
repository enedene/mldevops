"""
Author: Enes Deumic
Date: 2021-12-01

In this file we define all the constants used by a model.
"""

# list of categorical variables we keep for modeling
CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

# list of numerical variables we keep for modeling
QUANT_COLUMNS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

MPL_BAR_COLUMNS = ['Churn', 'Marital_Status']
MPL_HIST_COLUMNS = ['Customer_Age']
SNS_DENSITY_COLUMNS = ['Total_Trans_Ct']

FIG_SIZE = (20, 10)

# modeling
RF_PARAM_GRID = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5, 100],
    'criterion': ['gini', 'entropy']
}

LR_PARAM_GRID = {
    'l1_ratio': [0, 1, 0.25, 0.75],
    'C': [0.1, 1, 10]
}

ESTIMATOR_LABEL = 'model'
CV = 5
