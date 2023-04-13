#!/usr/bin/env/ conda: "sara"
# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, confloat
import uvicorn as uv

import numpy as np
import joblib

description = """
# Documentation
## ℹ️ Read carefully before using**

This api allows you to predict the type of Iris plant given a list of features.
The features should be:
* sepal length in cm
* sepal width in cm
* petal length in cm
* petal width in cm

_Build by:_
### Swapnil Narwade
"""

app = FastAPI(
    title="Iris Classification",
    description=description,
    version=.1
)


class ClassificationInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class Response(BaseModel):
    setosa_probability: confloat(ge=0.0, le=1.0)
    versicolor_probabilty: confloat(ge=0.0, le=1.0)
    virginica_probabilty: confloat(ge=0.0, le=1.0)

    class Config:
        schema_extra = {
            'example': {
                'setosa_probability': 0.7,
                'versicolor_probabilty': 0.1,
                'virginica_probability': .2
            }
        }


model = joblib.load('model.pkl')


@app.post('/predict')
async def predict(input: ClassificationInput):
    input_data = np.array(
        [[input.sepal_length,
          input.sepal_width,
          input.petal_length,
          input.petal_width]]
    )
    prediction = model.predict_proba(input_data)[-1]
    predictions_clean = Response(
        setosa_probability=prediction[0],
        versicolor_probabilty=prediction[1],
        virginica_probabilty=prediction[2],
    )
    return predictions_clean


if __name__ == "__main__":
    uv.run(app)