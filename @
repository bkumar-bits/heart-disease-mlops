import os
import pandas as pd


def preprocess_data(input_path="data/raw/heart.csv",
                    output_path="data/processed/heart_cleaned.csv"):

    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv(input_path)

    # Replace ? with NaN
    df.replace("?", pd.NA, inplace=True)

    numeric_columns = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "target"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical features
    categorical_columns = [
        "sex", "cp", "fbs", "restecg",
        "exang", "slope", "ca", "thal"
    ]

    for col in categorical_columns:
        df[col] = df[col].astype("category").cat.codes

    # Binary target
    df["target"] = df["target"].apply(
        lambda x: 0 if x == 0 else 1
    )

    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    df = preprocess_data()
    print("Preprocessing completed successfully!")
