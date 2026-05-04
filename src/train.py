import os
import pickle
import pandas as pd

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
    roc_auc_score
)

# Create model directory
os.makedirs("models", exist_ok=True)

# Load processed dataset
df = pd.read_csv("data/processed/heart_cleaned.csv")

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

print("Feature scaling completed.")

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)

# Cross-validation
lr_cv_scores = cross_val_score(
    lr_model,
    X_train_scaled,
    y_train,
    cv=5
)

print("\nLogistic Regression Cross-validation Scores:")
print(lr_cv_scores)

# Train model
lr_model.fit(X_train_scaled, y_train)

# Predictions
lr_predictions = lr_model.predict(X_test_scaled)

# Probabilities
lr_probabilities = lr_model.predict_proba(X_test_scaled)[:, 1]

# Metrics
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_precision = precision_score(y_test, lr_predictions)
lr_recall = recall_score(y_test, lr_predictions)
lr_roc_auc = roc_auc_score(y_test, lr_probabilities)

print("\nLogistic Regression Results:")
print("Accuracy:", lr_accuracy)
print("Precision:", lr_precision)
print("Recall:", lr_recall)
print("ROC-AUC:", lr_roc_auc)


# Random Forest with hyperparameter tuning
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

# Best model
rf_model = rf_grid.best_estimator_

print("\nBest Random Forest Parameters:")
print(rf_grid.best_params_)

# Cross-validation
rf_cv_scores = cross_val_score(
    rf_model,
    X_train_scaled,
    y_train,
    cv=5
)

print("\nRandom Forest Cross-validation Scores:")
print(rf_cv_scores)

# Predictions
rf_predictions = rf_model.predict(X_test_scaled)

# Probabilities
rf_probabilities = rf_model.predict_proba(X_test_scaled)[:, 1]

# Metrics
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_roc_auc = roc_auc_score(y_test, rf_probabilities)

print("\nRandom Forest Results:")
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("ROC-AUC:", rf_roc_auc)

# Select best model
if rf_accuracy > lr_accuracy:
    best_model = rf_model
    best_model_name = "Random Forest"
else:
    best_model = lr_model
    best_model_name = "Logistic Regression"

# Save best model
pickle.dump(best_model, open("models/model.pkl", "wb"))

print("\nBest Model Selected:", best_model_name)
print("Model saved successfully.")
