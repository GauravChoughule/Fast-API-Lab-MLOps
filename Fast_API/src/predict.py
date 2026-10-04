import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "movie_model.pkl")
_model = None


def _get_model():
    # Load once and reuse instead of reading the file on every request
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_data(movie: dict):
    """Predict the IMDb rating (1-10) for one movie given as a dict of features."""
    X = pd.DataFrame([movie])
    rating = float(_get_model().predict(X)[0])
    return min(max(rating, 1.0), 10.0)  # keep it on the 1-10 scale