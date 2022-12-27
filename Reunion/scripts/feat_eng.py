#!/usr/bin/env/ conda: "base"
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.feature_selection import (chi2, mutual_info_classif, VarianceThreshold, SelectKBest, SelectFromModel)
from sklearn.preprocessing import (OneHotEncoder, LabelEncoder)
from sklearn.linear_model import (RidgeCV, Ridge)
from sklearn.impute import SimpleImputer

from warnings import filterwarnings
filterwarnings("ignore")
plt.style.use("seaborn-whitegrid")


def oneHot_encode(data: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    if cat_cols != data.select_dtypes(include="object").columns.to_list():
        print("provided columns are not categorical")
        return
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore").fit(data[cat_cols])
    encoded_cols = list(encoder.get_feature_names(cat_cols))
    data[encoded_cols] = encoder.transform(data[cat_cols])
    data.drop(columns=cat_cols, inplace=True)
    return data


def label_encode(data: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    encoder = LabelEncoder()
    for col in cat_cols:
        encoder.fit(data[col])
        data[col] = encoder.transform(data[col])
    return data


def ridge_feature_selector(inputs: pd.DataFrame, target: pd.Series) -> plt.plot:
    ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(inputs, target)
    importances = np.abs(ridge.coef_)
    feat_names = np.array(inputs.columns)
    plt.figure(figsize=(12, 12))
    plt.barh(feat_names, importances, color="teal")
    plt.title("Feature importances via coefficients")
    plt.show()


def variance_threshold_selector(data: pd.DataFrame, threshold: int) -> np.ndarray:
    selector = VarianceThreshold(threshold=threshold).fit(data)
    feat_names = np.array(data.columns)
    return feat_names[selector.get_support()]


def chi2_feature_selection(inputs: pd.DataFrame, target: pd.Series, k: int, scoring) -> pd.DataFrame:
    best_feat = SelectKBest(score_func=scoring, k=10).fit(inputs, target)
    scores = pd.DataFrame({'features': inputs.columns, 'scores': np.round(best_feat.scores_, 2)})
    return scores


def information_gain(inputs: pd.DataFrame, target: pd.Series) -> pd.Series:
    importance = mutual_info_classif(inputs, target)
    feature_importances = pd.Series(importance, inputs.columns)
    return feature_importances


def plot_info_gain(importances: pd.Series) -> plt.plot:
    importances.plot(kind="barh", color="palevioletred", figsize=(10, 5))
    plt.title("Information Gain plot")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    applicants = pd.read_csv("raw_data/applicant.csv")
    loan = pd.read_csv("data/loan_details.csv")
    applicants.drop(columns=["Balance_in_existing_bank_account_(lower_limit_of_bucket)",
                             "Balance_in_existing_bank_account_(upper_limit_of_bucket)",
                             "Telephone"
                            ], inplace=True
	             )
    target = loan.pop("risk")
    
    applicants["Has_been_employed_for_at_least"] = applicants["Has_been_employed_for_at_least"].str.replace("years", "")
    applicants["Has_been_employed_for_at_least"] = applicants["Has_been_employed_for_at_least"].str.replace("year", "")
    applicants["Has_been_employed_for_at_least"].fillna(0, inplace=True)
    applicants["Has_been_employed_for_at_least"] = applicants["Has_been_employed_for_at_least"].astype(str).astype(np.float64)
    applicants["Has_been_employed_for_at_least"].replace(0, applicants["Has_been_employed_for_at_least"].mean(), inplace=True)

    applicants["Has_been_employed_for_at_most"] = applicants["Has_been_employed_for_at_most"].str.replace("years", "")
    applicants["Has_been_employed_for_at_most"] = applicants["Has_been_employed_for_at_most"].str.replace("year", "")
    applicants["Has_been_employed_for_at_most"].fillna(0, inplace=True)
    applicants["Has_been_employed_for_at_most"] = applicants["Has_been_employed_for_at_most"].astype(str).astype(np.float64)
    applicants["Has_been_employed_for_at_most"].replace(0, applicants["Has_been_employed_for_at_most"].mean(), inplace=True)
 
    applicants["Savings_account_balance"] = applicants["Savings_account_balance"].map({"Low": 1, "Medium": 2, "High":0, "Very High":0})
    applicants["Savings_account_balance"].fillna(applicants["Savings_account_balance"].mean(), inplace=True)

    cat_cols_applicants = [col for col in applicants.columns if applicants[col].dtype in ["object", "bool"]]
    cat_cols_loan = [col for col in loan.columns if loan[col].dtype in ["object", "bool"] ]
    applicants = label_encode(applicants, cat_cols_applicants)
    loan = label_encode(loan, cat_cols_loan)
    
    data = loan.merge(applicants)
    # data['risk'] = target

    # data.to_csv("combined_data.csv", index=False)
    
    # -- Feature Selection --
    # ridge_feature_selector(data, target) # gives error -- fix this. 

    info = information_gain(data, target)
    plot_info_gain(info)

    print(variance_threshold_selector(data, 0.5))
    # """ ['applicant_id' 'loan_taken_over(months)' 'purpose' 'loan_amt'    
    #      'emi_rate(percent)' 'property' 'loan_history'
    #      'Primary_applicant_age_in_years' 'Marital_status'
    #      'Years_at_current_residence' 'Employment_status'
    #      'Has_been_employed_for_at_least' 'Has_been_employed_for_at_most'
    #      ]
    # """

    # scores = chi2_feature_selection(data, target, 11, chi2)
    # print(scores)
    """                        -- scores on labebl encoded data --
                                      features       scores
            0                     applicant_id     29254.80
            1          loan_taken_over(months)       321.03
            2                          purpose         4.21
            3                         loan_amt  58262490.01
            4                emi_rate(percent)         2.20
            5                         property         3.07
            6                  has_coapplicant         3.77
            7                    has_guarantor         2.87
            8            existing_loans(total)         0.50
            9                     loan_history         4.80
            10  Primary_applicant_age_in_years        30.20
            11                          Gender         1.77
            12                  Marital_status         4.39
            13            Number_of_dependents         0.00
            14                         Housing         0.10
            15      Years_at_current_residence         0.00
            16               Employment_status         0.75
            17  Has_been_employed_for_at_least         6.70
            18   Has_been_employed_for_at_most         9.62
            19                  Foreign_worker         0.25
            20         Savings_account_balance         0.40
    """