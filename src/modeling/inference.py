# src/modeling/inference.py

import os
import joblib
import pandas as pd
import xgboost as xgb
from preprocessing.intersections import intersections_df
from geopy.distance import great_circle

# 1) Load your trained Booster once at import time
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/xgb_clf_full.joblib")
model: xgb.XGBClassifier = joblib.load(MODEL_PATH)

def find_nearest_intersection_id(lat, lon):
    # compute the distance to every known intersection
    dists = intersections_df.apply(
        lambda r: great_circle((lat, lon), (r.lat, r.lon)).km,
        axis=1
    )
    # pick the ID of the closest one
    return intersections_df.loc[dists.idxmin(), 'nearest_intersection_id']

def prepare_grid(grid_df):
    # for each point in your payload, map to the nearest intersection
    grid_df['nearest_intersection_id'] = grid_df.apply(
        lambda r: find_nearest_intersection_id(r.lat, r.lon),
        axis=1
    )
    return grid_df

def predict_accident_probabilities(
    grid_df: pd.DataFrame,
    date: str,
    borough_weather: dict
) -> pd.DataFrame:
    # --- (1) date/time features ---
    dt = pd.to_datetime(date)
    grid_df["hour"]        = dt.hour
    grid_df["day_of_week"] = dt.dayofweek
    grid_df["month"]       = dt.month
    grid_df["is_weekend"]  = dt.dayofweek >= 5

    # --- (2) weather features from API (already converted to model units upstream) ---
    for feat in ("tavg", "prcp", "snow", "wdir", "wspd", "pres"):
        grid_df[feat] = grid_df["borough"].map(lambda b: borough_weather[b][feat])

    # --- (3) spatial features ---
    # raw lat/lon of grid cell
    grid_df["nearest_intersection_lat"] = grid_df["lat"]
    grid_df["nearest_intersection_lon"] = grid_df["lon"]
    # map to the real intersection ID
    grid_df["nearest_intersection_id"] = grid_df.apply(
        lambda r: find_nearest_intersection_id(r.lat, r.lon),
        axis=1
    )

    # --- (4) assemble features in exactly the order the model expects ---
    feature_columns = [
        "hour","day_of_week","month","is_weekend",
        "tavg","prcp","snow",
        "wdir","wspd","pres",
        "nearest_intersection_lat","nearest_intersection_lon",
        "nearest_intersection_id"
    ]
    X = grid_df[feature_columns]

    # --- (5) predict ---
    # since this is an sklearn-wrapped XGBClassifier:
    proba = model.predict_proba(X)[:, 1]
    grid_df["probability"] = proba

    # --- (6) return only the API‐spec fields ---
    return grid_df[["lat", "lon", "borough", "probability"]]
