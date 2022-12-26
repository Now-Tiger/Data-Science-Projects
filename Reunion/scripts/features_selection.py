#!/usr/bin/env/ conda: "base"
# -*- coding:utf-8 -*-

import warnings
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# from scipy.stats import median_abs_deviation as MAD

from sklearn.feature_selection import (chi2, mutual_info_classif, VarianceThreshold, SelectKBest)
from sklearn.preprocessing import (StandardScaler, LabelEncoder, OneHotEncoder)
# from sklearn.metrics import (mean_absolute_error, mean_squared_error)

# from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor)
# from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor)
# from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
plt.style.use("seaborn-whitegrid")


def label_encode_(data: pd.DataFrame) -> pd.DataFrame:
    """ return the data set label encoding the categorical columns automatically """
    cat_cols = [col for col in data.columns if data[col].nunique() < 20
                and
                data[col].dtype in ['object', 'bool']
               ]
    encoder = LabelEncoder()
    for col in cat_cols:
        encoder.fit(data[col])
        data[col] = encoder.transform(data[col])
    return data


def oneHot_encode_(data: pd.DataFrame) -> pd.DataFrame:
    """ return the data set one hot encoding the categorical columns automatically
        Function doesnt work fix asap.
     """
    cat_cols = data.select_dtypes(include="object").columns.to_list()
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore").fit(data[cat_cols])
    encoded_cols = list(encoder.get_feature_names(cat_cols))
    data[encoded_cols] = encoder.transform(data[cat_cols])
    data.drop(columns=cat_cols, inplace=True)
    return data


def information_gain(inputs: pd.DataFrame, target: pd.Series) -> pd.Series:
    importance = mutual_info_classif(inputs, target)
    feature_importances = pd.Series(importance, inputs.columns)
    return feature_importances


def chi2_feature_selection(inputs: pd.DataFrame, target: pd.Series, scoring) -> pd.DataFrame:
    best_feat = SelectKBest(score_func=scoring, k=10).fit(inputs, target)
    scores = pd.DataFrame(
        {'features': inputs.columns, 'scores': np.round(best_feat.scores_, 2)})
    return scores


def variance_threshold_selector(data: pd.DataFrame, threshold: int) -> None:
    selector = VarianceThreshold(threshold=threshold).fit(data)
    print(data[data.columns[selector.get_support(indices=True)]])


def plot_info_gain(importances: pd.Series) -> plt.plot:
    importances.plot(kind="barh", color="teal", figsize=(12, 15))
    plt.show()


if __name__ == "__main__":
    applicants_clean = pd.read_csv("data/applicants.csv")
    loan_details_clean = pd.read_csv("data/loan_details.csv")
    target = loan_details_clean.pop("risk")

    loan_details_clean = oneHot_encode_(loan_details_clean)

    variance_threshold_selector(loan_details_clean.iloc[:, 1:], 0)

    scores = chi2_feature_selection(loan_details_clean, target, chi2)
    print(scores)
    """             -- Using Label Encoding --
                      features       scores
    0             applicant_id     29254.80
    1  loan_taken_over(months)       321.03
    2                  purpose         4.21
    3                 loan_amt  58262490.01
    4        emi_rate(percent)         2.20
    5                 property         3.07
    6          has_coapplicant         3.77
    7            has_guarantor         2.87
    8    existing_loans(total)         0.50
    9             loan_history         4.80

                -- OneHot shows better results --
         compare saved pictures of plots and this scoring
    """
