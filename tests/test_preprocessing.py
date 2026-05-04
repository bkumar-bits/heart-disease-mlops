import sys
import os

# Add project root to Python path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.data_preprocessing import preprocess_data


def test_preprocess_data():
    df = preprocess_data()

    # Check dataset is not empty
    assert df.shape[0] > 0

    # Check missing values removed
    assert df.isnull().sum().sum() == 0

    # Check binary target
    assert set(df["target"].unique()).issubset({0, 1})
