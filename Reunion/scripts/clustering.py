#!/usr/bin/env/ conda: "tensor"
# -*- coding: utf-8 -*-
import os
from collections import defaultdict
from warnings import filterwarnings

import pandas as pd
import numpy as np

import tensorflow as tf

from sklearn.preprocessing import (Normalizer, StandardScaler,OneHotEncoder, LabelEncoder)
from sklearn.model_selection import train_test_split as split
from sklearn import manifold

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from jupyterthemes import jtplot

filterwarnings("always")
filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sns.set_style("whitegrid")
jtplot.style(theme="chesterish", context="notebook", grid=True, ticks=True)


def make_inputs_for_layers(data: pd.DataFrame) -> dict:
    """ We create a dictionary for our input features, with matching data types.
        inputs holds information about what the data looks like.
        We use this to tell Keras how to connect neural network layers together.
    """
    inputs = defaultdict()
    for name, column in data.items():
        dtype = column.dtype
        if dtype == "object":
            dtype = tf.string
        else:
            dtype = tf.float32
        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)
    return inputs

def get_num_features(inputs_dict: dict) -> dict:
    """ Separate out the numerical columns """
    numeric_inputs = {name: input for name, input in inputs_dict.items()
                      if input.dtype == tf.float32
                     }
    return numeric_inputs

def onehot_encoding(inputs_dict: dict, dataset: pd.DataFrame) -> dict:
    """ Separate out categorical columns and encode them as One Hot"""
    for name, input in inputs_dict.items():
        if input.dtype != tf.string:
            continue
        lookup = tf.keras.layers.StringLookup(vocabulary=np.unique(dataset[name]))
        one_hot = tf.keras.layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())
        x = lookup(input)
        x = one_hot(x)
    return x

def normalize_numericals(numerical_features: dict, x: tf.Tensor) -> tf.float32: 
    """ Data normalization on numerical inputs """
    norm = tf.keras.layers.Normalization()
    norm.adapt(np.array(data[numerical_features.keys()]))
    all_numerical_inputs = norm(x)
    return all_numerical_inputs


if __name__ == "__main__":
    data = pd.read_csv("../src/combined_data.csv")
    data.drop(columns=["marital_status", "has_coapplicant","has_guarantor"], inplace=True)
    applicant_id = data.pop("applicant_id")
    risk = data.pop("risk")
    
    inputs = make_inputs_for_layers(data)
    # -- separate out the four numerical columns, concatenate them into a single feature vector --
    num_inps = get_num_features(inputs)
    x = tf.keras.layers.Concatenate()(list(num_inps.values()))

    # -- Normalize the data using keras.preprocessing -- 
    all_numeric_inputs = normalize_numericals(num_inps, x)
    preprocessed_inputs = [all_numeric_inputs]

    # -- encoded data --
    encoded_cat_features = onehot_encoding(inputs, data)
    
    # -- append all features together --     
    preprocessed_inputs.append(encoded_cat_features)

    # -- we concatenate the preprocessed inputs into a single vector, and create the processing model 
    # which we can later apply to our data as the first step in our neural network:
    preprocessed_inputs_cat = tf.keras.layers.Concatenate()(preprocessed_inputs)
    preprocessing_layer = tf.keras.Model(inputs, preprocessed_inputs_cat, name="ProcessData")

    # -- results of our processing model --
    tf.keras.utils.plot_model(model=preprocessing_layer, rankdir="LR", dpi=130, show_shapes=True, to_file="processing.png")
