import os
import pickle


def test_model_exists():
    assert os.path.exists("models/model.pkl")


def test_model_load():
    model = pickle.load(open("models/model.pkl", "rb"))
    assert model is not None
