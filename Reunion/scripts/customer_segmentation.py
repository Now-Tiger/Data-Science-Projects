#!/usr/bin/env/ conda: "bash"
# -*- coding: utf-8 -*-

from yellowbrick.cluster import KElbowVisualizer
from warnings import filterwarnings
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.preprocessing import (Normalizer, MinMaxScaler, StandardScaler)
from sklearn.metrics import (silhouette_samples, silhouette_score)
from sklearn.cluster import KMeans

from feat_eng import (oneHot_encode, label_encode)
filterwarnings("ignore")
plt.style.use('seaborn-whitegrid')


def get_cluster_num(data: pd.DataFrame, **params) -> plt.plot:
    """ returns a plot wiht optimal number of clusters to consider. """
    model = KMeans(**params)

    plt.figure(figsize=(12, 7), dpi=80)
    visualizer = KElbowVisualizer(model, k=(2, 8), timings=True)
    visualizer.fit(data)
    visualizer.show()
    plt.show()


def kmeans(data: pd.DataFrame, n_clusters: int, iter: int, init: str) -> None:
    """ Fits Kmeans algorithm and return the algorithm's performance accuracy
        using silhouette score metric.
    """
    kmean = KMeans(n_clusters=n_clusters, max_iter=iter,
                   init=init, random_state=42)
    kmean.fit(data)
    acc = silhouette_score(data, kmean.labels_, metric="euclidean")
    return f"silhouette accuracy: {round(acc, 3) * 100}%"


if __name__ == "__main__":
    applicants = pd.read_csv("applicants_cleaned.csv")
    loan = pd.read_csv("loan_details.csv")

    cat_cols_applicants = applicants.select_dtypes("object").columns.to_list()
    cat_cols_loan = loan.select_dtypes("object").columns.to_list()

    applicants = label_encode(applicants, cat_cols_applicants)
    loan = label_encode(loan, cat_cols_loan)

    data = applicants.merge(loan)
    applicant_id = data.pop("applicant_id")
    # -- dropping almost same and zero variance columns --
    data.drop(columns=["foreign_worker", "has_coapplicant", "has_guarantor"], 
              inplace=True
             )

    # # -- Normalize data --
    columns = [col for col in data.columns if col != 'risk']
    scaler = Normalizer().fit(data[columns])
    data[columns] = scaler.transform(data[columns])


    # -- KMeans Clustering --
    params = {'init': 'k-means++', 'max_iter': 10_000, 'random_state': 42}
    # get_cluster_num(data, **params)
    print(kmeans(data, 3, 10_000, "k-means++"))
