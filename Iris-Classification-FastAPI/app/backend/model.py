#!/usr/bin/env/ conda: "sara"
# -*- coding: utf-8 -*-
import pickle
import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score)
from sklearn.model_selection import train_test_split as split

from sklearn.linear_model import LogisticRegression
np.random.seed(42)


def normalize(data):
    numeric_cols = data.select_dtypes('float64').columns.to_list()
    scaler = StandardScaler().fit(data[numeric_cols])
    data[numeric_cols] = scaler.transform(data[numeric_cols])
    return data[numeric_cols]


def split_dataset(inputs, target):
    train_inp, test_inp, train_target, test_target = split(
        inputs, target, test_size=.25, random_state=42, shuffle=True)
    return (train_inp, test_inp, train_target, test_target)


if __name__ == "__main__":
    inputs, label = load_iris(return_X_y=True, as_frame=True)
    train_inputs, test_inputs, train_target, test_target = split_dataset(inputs, label)
    train_inputs_scaled = normalize(train_inputs)
    test_inputs_scaled = normalize(test_inputs)

    logistic_reg = LogisticRegression(
        max_iter=300, solver='lbfgs', multi_class='multinomial')
    logistic_reg.fit(train_inputs_scaled, train_target)

    # -- save the model --     
    # with open("./backend/model.pkl", 'wb') as f:
    #     pickle.dump(logistic_reg, f)

    # load_model = joblib.load('./backend/model.pkl')  <<= import joblib here
    # print(load_model.score(test_inputs_scaled, test_target))
    # 0.9736842105263158
