import os
import zipfile
import requests
import pandas as pd

# Create folders
os.makedirs("data/raw", exist_ok=True)

# Dataset zip URL
url = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"

zip_path = "data/raw/heart_disease.zip"

# Download zip file
response = requests.get(url)

with open(zip_path, "wb") as f:
    f.write(response.content)

print("ZIP downloaded successfully!")

# Extract zip
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall("data/raw")

print("ZIP extracted successfully!")

# Column names (UCI documentation)
columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

# Load actual dataset file
df = pd.read_csv(
    "data/raw/processed.cleveland.data",
    header=None,
    names=columns
)

# Save as CSV
df.to_csv("data/raw/heart.csv", index=False)

print("Dataset saved successfully!")
print("Shape:", df.shape)
print(df.head())
