import pandas as pd
from sklearn.model_selection import train_test_split

# Movies dataset from the Vega project (~3,200 movies with budgets, genres and ratings)
DATA_URL = "https://cdn.jsdelivr.net/npm/vega-datasets@2.2.0/data/movies.json"

NUMERIC_FEATURES = ["Production_Budget", "Running_Time_min", "Rotten_Tomatoes_Rating", "Release_Year"]
CATEGORICAL_FEATURES = ["Major_Genre", "MPAA_Rating", "Creative_Type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "IMDB_Rating"


def load_data():
    """Load movies and return features + IMDb rating (target)."""
    df = pd.read_json(DATA_URL)
    df.columns = df.columns.str.replace(" ", "_")  # "Release Date" -> "Release_Date"
    # Pull the 4-digit year out of the release date text
    df["Release_Year"] = pd.to_numeric(
        df["Release_Date"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce"
    )
    df = df.dropna(subset=[TARGET])  # can't train on movies without a rating
    return df[FEATURES], df[TARGET]


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)