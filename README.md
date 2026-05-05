# Heart Disease Prediction Using End-to-End MLOps Pipeline

## Project Overview

This project demonstrates an end-to-end MLOps pipeline for heart disease prediction using machine learning and modern MLOps practices.

The objective is to build, track, test, deploy, and monitor a machine learning model for predicting heart disease using patient health data.

The project includes:

* Data Acquisition and Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Model Training
* Experiment Tracking
* Model Packaging
* CI/CD Pipeline
* API Development
* Docker Containerization
* Kubernetes Deployment
* Monitoring and Logging

---

## Dataset Information

Dataset Source: UCI Machine Learning Repository

Dataset Name: Heart Disease Dataset

Features include:

* age
* sex
* chest pain type
* resting blood pressure
* cholesterol
* fasting blood sugar
* resting ECG
* maximum heart rate
* exercise induced angina
* oldpeak
* slope
* number of major vessels
* thalassemia

Target:

* 0 = No heart disease
* 1 = Heart disease

---

## Project Structure

```text
heart-disease-mlops/
│── data/
│   ├── raw/
│   ├── processed/
│
│── notebooks/
│   ├── eda.ipynb
│
│── src/
│   ├── data_download.py
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── experiment.py
│   ├── package_model.py
│   ├── api.py
│
│── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── pipeline.pkl
│
│── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│
│── deployment/
│   ├── deployment.yaml
│   ├── service.yaml
│
│── monitoring/
│   ├── prometheus.yml
│
│── screenshots/
│
│── .github/workflows/
│   ├── ci.yml
│
│── Dockerfile
│── requirements.txt
│── environment.yml
│── README.md
│── report.docx
```

---

## Installation Instructions

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Mac/Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Task 1: Dataset Download

Run:

```bash
python src/data_download.py
```

This downloads the Heart Disease dataset and stores it in:

```text
data/raw/
```

---

## Task 1: Data Preprocessing

Run:

```bash
python src/data_preprocessing.py
```

This performs:

* Missing value handling
* Encoding
* Target conversion

Output:

```text
data/processed/heart_cleaned.csv
```

---

## Task 1: Exploratory Data Analysis

Open notebook:

```bash
jupyter notebook
```

Run:

```text
notebooks/eda.ipynb
```

EDA includes:

* Histograms
* Correlation Heatmap
* Class Balance Plot
* Boxplots

---

## Task 2: Model Training

Run:

```bash
python src/train.py
```

Models trained:

* Logistic Regression
* Random Forest

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* ROC-AUC

Output:

```text
models/model.pkl
models/scaler.pkl
```

---

## Task 3: Experiment Tracking

Using MLflow

Run:

```bash
python src/experiment.py
```

Start MLflow UI:

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

Tracks:

* Parameters
* Metrics
* Artifacts
* Models

---

## Task 4: Model Packaging

Run:

```bash
python src/package_model.py
```

Output:

```text
models/pipeline.pkl
```

This stores:

* preprocessing pipeline
* trained model

---

## Task 5: Unit Testing

Using pytest

Run:

```bash
pytest
```

Tests:

* preprocessing validation
* model loading validation

---

## Task 5: Code Quality Check

Using Flake8

Run:

```bash
flake8 src/
```

---

## Task 5: CI/CD Pipeline

Using GitHub Actions

Workflow file:

```text
.github/workflows/ci.yml
```

Pipeline includes:

* Install dependencies
* Linting
* Data preprocessing
* Model training
* Unit testing

---

## Task 6: API Development

Using FastAPI

Run:

```bash
uvicorn src.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Endpoints:

* /
* /predict

---

## Sample API Input

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

---

## Task 6: Docker Build

Using Docker

Build:

```bash
docker build -t heart-disease-api .
```

Run:

```bash
docker run -p 8000:8000 heart-disease-api
```

---

## Task 7: Kubernetes Deployment

Using Kubernetes

Deploy:

```bash
kubectl apply -f deployment/
```

Check pods:

```bash
kubectl get pods
```

Check services:

```bash
kubectl get services
```

Port forward:

```bash
kubectl port-forward service/heart-disease-service 8000:80
```

API:

```text
http://127.0.0.1:8000/docs
```

---

## Task 8: Monitoring

Using Prometheus

Run:

```bash
prometheus --config.file=monitoring/prometheus.yml
```

Open:

```text
http://localhost:9090
```

Query:

```text
http_requests_total
```

---

## Task 8: Dashboard Monitoring

Using Grafana

Run:

```bash
grafana-server
```

Open:

```text
http://localhost:3000
```

Default credentials:

Username: admin
Password: admin

---

## Architecture Flow

```text
Dataset
↓
Data Preprocessing
↓
Feature Engineering
↓
Model Training
↓
MLflow Tracking
↓
Model Packaging
↓
FastAPI API
↓
Docker Container
↓
Kubernetes Deployment
↓
Prometheus Monitoring
↓
Grafana Dashboard
```

---

## Technologies Used

* Python
* pandas
* scikit-learn
* MLflow
* FastAPI
* Docker
* Kubernetes
* Prometheus
* Grafana
* GitHub Actions

---

## Conclusion

This project successfully demonstrates an end-to-end MLOps workflow for heart disease prediction.

It includes:

* reproducibility
* automation
* deployment
* monitoring

The project follows production-ready MLOps practices and can be extended further for real-world deployment.

