import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from data import load_data, split_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "movie_model.pkl")


def fit_model(X_train, y_train):
    # Categorical columns: fill missing with "Unknown", then one-hot encode
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),  # gradient boosting handles missing numbers itself
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)),
    ])
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    X, y = load_data()
    print(f"Training on {len(X)} movies")
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = fit_model(X_train, y_train)

    preds = model.predict(X_test)
    print(f"R2 score: {r2_score(y_test, preds):.3f}")
    print(f"Mean absolute error: {mean_absolute_error(y_test, preds):.2f} rating points")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")