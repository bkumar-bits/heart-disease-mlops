import os
import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# Set MLflow experiment
mlflow.set_experiment("Heart_Disease_Classification")

# Load dataset
df = pd.read_csv("data/processed/heart_cleaned.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

with mlflow.start_run(run_name="Logistic_Regression"):

    lr_model = LogisticRegression(max_iter=1000)

    lr_model.fit(X_train_scaled, y_train)

    predictions = lr_model.predict(X_test_scaled)
    probabilities = lr_model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    cv_scores = cross_val_score(
        lr_model,
        X_train_scaled,
        y_train,
        cv=5
    )

    # Log parameters
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("cv_mean", cv_scores.mean())

    # Log model
    mlflow.sklearn.log_model(lr_model, "model")

    print("Logistic Regression logged successfully.")

with mlflow.start_run(run_name="Random_Forest"):

    rf_parameters = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None]
    }

    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_parameters,
        cv=5
    )

    rf_grid.fit(X_train_scaled, y_train)

    rf_model = rf_grid.best_estimator_

    predictions = rf_model.predict(X_test_scaled)
    probabilities = rf_model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    cv_scores = cross_val_score(
        rf_model,
        X_train_scaled,
        y_train,
        cv=5
    )

    # Log parameters
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("best_params", rf_grid.best_params_)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("cv_mean", cv_scores.mean())

    # Confusion matrix plot
    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plot_path = "artifacts/confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close()

    # Log artifact
    mlflow.log_artifact(plot_path)

    # Log model
    mlflow.sklearn.log_model(rf_model, "model")

    # Save best model
    pickle.dump(rf_model, open("models/model.pkl", "wb"))

    print("Random Forest logged successfully.")
