import pickle
import numpy as np
import logging

from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator


# Logging setup
logging.basicConfig(
    filename="api.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

app = FastAPI()

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Load model
model = pickle.load(
    open("models/pipeline.pkl", "rb")
)


class HeartData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.get("/")
def home():
    logging.info("Home endpoint accessed")

    return {
        "message": "Heart Disease Prediction API"
    }


@app.post("/predict")
def predict(data: HeartData):

    logging.info(f"Prediction request received: {data}")

    input_data = np.array([
        [
            data.age,
            data.sex,
            data.cp,
            data.trestbps,
            data.chol,
            data.fbs,
            data.restecg,
            data.thalach,
            data.exang,
            data.oldpeak,
            data.slope,
            data.ca,
            data.thal
        ]
    ])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0].max()

    logging.info(
        f"Prediction: {prediction}, Confidence: {probability}"
    )

    return {
        "prediction": int(prediction),
        "confidence": float(probability)
    }
