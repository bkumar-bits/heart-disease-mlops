import os
import pickle
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Create model directory
os.makedirs("models", exist_ok=True)

# Load processed dataset
df = pd.read_csv("data/processed/heart_cleaned.csv")

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Create pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ))
])

# Train pipeline
pipeline.fit(X, y)

# Save full pipeline
pickle.dump(pipeline, open("models/pipeline.pkl", "wb"))

print("Pipeline saved successfully!")
