#!/usr/bin/env/ conda: "base"
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,  r2_score
from sklearn.model_selection import train_test_split as split

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor



def normalize_(data: pd.DataFrame, scaler) -> pd.DataFrame:
    """ returns noramlized numerical pandas dataframe. """
    num_cols = data.select_dtypes(include=np.number).columns.to_list()
    scaler_ = scaler().fit(data[num_cols])
    data[num_cols] = scaler_.transform(data[num_cols])
    return data[num_cols]


def rmse(target: pd.Series, predictions: np.ndarray) -> np.float64:
    """ returns root mean squared error in float value """
    return mean_squared_error(target, predictions, squared=False)


def model_metrics(inputs: pd.DataFrame, target: pd.Series, model, name: str) -> str:
    """ returns a performance summary/metrics of the model """
    preds = model.predict(inputs)

    print(f"\n---- {name} ----\n"
          f"RMSE: {rmse(target, preds):.3f}\n"
          f"Accuracy: {model.score(inputs, target):.3f}\n"
          f"R2 score: {r2_score(target, preds):.3f}"
         )
    return


def baseline_model(models: dict) -> None:
    """ 
        Fits the model from the models dictionary 
        and returns metrics summary from 
        model_metrics function.
    """
    for name, model in models.items():
        print(f"\n*** {name} *** ")
        model.fit(train_inputs, train_target)
        model_metrics(train_inputs, train_target, model, 'train set')
        model_metrics(test_inputs, test_target, model, 'test set')
    return


if __name__ == "__main__":
    PATH: str = "../data/DS - Assignment Part 1 data set.xlsx"
    data = pd.read_excel(PATH)
    data.drop(columns="Transaction date", inplace=True)

    target = data.pop("House price of unit area")
    inputs = data

    inputs = normalize_(inputs, MinMaxScaler)

    # train test split:
    train_inputs, test_inputs, train_target, test_target = split(inputs, target, test_size=0.20, random_state=42)

    models = {"linear reg": LinearRegression(),
              "DecisionTree": DecisionTreeRegressor(),
              "RandomForest": RandomForestRegressor(max_depth=8, 
                                                    min_samples_split=2, 
                                                    max_features=5, 
                                                    random_state=13
                                                   )
             }

    baseline_model(models)



# (base) D:\College\Data Science\Tensorflow\shack_labs\scripts>python assesment_one.py

# *** linear reg ***

# ---- train set ----
# RMSE: 9.219
# Accuracy: 0.548
# R2 score: 0.548

# ---- test set ----
# RMSE: 7.542
# Accuracy: 0.661
# R2 score: 0.661

# *** DecisionTree ***

# ---- train set ----
# RMSE: 0.000
# Accuracy: 1.000
# R2 score: 1.000

# ---- test set ----
# RMSE: 8.555
# Accuracy: 0.564
# R2 score: 0.564

# *** RandomForest ***

# ---- train set ----
# RMSE: 3.656
# Accuracy: 0.929
# R2 score: 0.929

# ---- test set ----
# RMSE: 5.618
# Accuracy: 0.812
# R2 score: 0.812