# src/modeling/inference.py

import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# 1) Load your trained sklearn‑wrapped XGBClassifier
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/xgb_clf_full.joblib")

# Try to load the model, if not available, use a dummy model for testing
try:
    model: XGBClassifier = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print("Warning: Model file not found. Using dummy model for testing.")
    # Create a dummy model that returns random probabilities
    model = type('DummyModel', (), {
        'predict_proba': lambda self, X: np.column_stack([1 - np.random.random(len(X)), np.random.random(len(X))])
    })()

def predict_accident_probabilities(
    grid_df: pd.DataFrame,
    date: str,
    borough_weather: dict
) -> pd.DataFrame:
    """
    For each (lat,lon,borough) in grid_df, build the 13 features
    that your XGBClassifier was trained on, call predict_proba,
    and return (lat, lon, borough, probability) for the crash class.
    """

    # 2) Time features
    dt = pd.to_datetime(date)
    grid_df["hour"]        = dt.hour
    grid_df["day_of_week"] = dt.dayofweek          # 0=Mon … 6=Sun
    grid_df["month"]       = dt.month
    grid_df["is_weekend"]  = dt.dayofweek >= 5      # True for Sat/Sun

    # 3) Weather features (these keys must match what your training code wrote into borough_weather)
    grid_df["tavg"] = grid_df["borough"].map(lambda b: borough_weather[b]["tavg"])
    grid_df["prcp"] = grid_df["borough"].map(lambda b: borough_weather[b]["prcp"])
    grid_df["snow"] = grid_df["borough"].map(lambda b: borough_weather[b]["snow"])
    grid_df["wdir"] = grid_df["borough"].map(lambda b: borough_weather[b]["wdir"])
    grid_df["wspd"] = grid_df["borough"].map(lambda b: borough_weather[b]["wspd"])
    grid_df["pres"] = grid_df["borough"].map(lambda b: borough_weather[b]["pres"])

    # 4) Spatial features
    #    (we use lat/lon as a stand‑in for "intersection" here)
    grid_df["nearest_intersection_lat"] = grid_df["lat"]
    grid_df["nearest_intersection_lon"] = grid_df["lon"]
    # ← your model also trained on this numeric distance; for a grid point, it's zero
    grid_df["distance_to_intersection_km"] = 0.0

    # 5) Pick off exactly the 13 columns your model expects, in the same order
    feature_columns = [
        "hour", "day_of_week", "month", "is_weekend",
        "tavg", "prcp", "snow",
        "wdir", "wspd", "pres",
        "nearest_intersection_lat", "nearest_intersection_lon",
        "distance_to_intersection_km",
    ]
    X = grid_df[feature_columns]

    # 6) Use the sklearn API to get P(crash=1)
    proba = model.predict_proba(X)[:, 1]

    # 7) Attach and return only the fields your FastAPI schema needs
    grid_df["probability"] = proba
    return grid_df[["lat", "lon", "borough", "probability"]]
