# src/modeling/inference.py

import os
import joblib
import pandas as pd
import xgboost as xgb
from src.preprocessing.intersections import intersections_df
from geopy.distance import great_circle
from sklearn.preprocessing import QuantileTransformer
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1) Load your trained Booster once at import time
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/xgb_clf_full.joblib")

# Create a dummy model for development/testing
def create_dummy_model():
    logger.warning("Using dummy model for development/testing")
    dummy_model = xgb.XGBClassifier(
        objective='binary:logistic',  # For binary classification
        base_score=0.5,  # Set base_score to 0.5 for binary classification
        random_state=42
    )
    # Create dummy data and fit the model
    X = pd.DataFrame([[0] * 13], columns=[
        "hour","day_of_week","month","is_weekend",
        "tavg","prcp","snow","wdir","wspd","pres",
        "nearest_intersection_lat","nearest_intersection_lon",
        "nearest_intersection_id"
    ])
    y = pd.Series([0])
    return dummy_model.fit(X, y)

try:
    if os.path.exists(MODEL_PATH):
        model: xgb.XGBClassifier = joblib.load(MODEL_PATH)
        logger.info(f"Loaded model from {MODEL_PATH}")
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}")
        model = create_dummy_model()
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = create_dummy_model()

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
    #grid_df["nearest_intersection_id"] = grid_df.apply(
       # lambda r: find_nearest_intersection_id(r.lat, r.lon),
       # axis=1)
    grid_df["nearest_intersection_id"] = 0

    # --- (4) assemble features in exactly the order the model expects ---
    feature_columns = [
        "hour","day_of_week","month","is_weekend",
        "tavg","prcp","snow",
        "wdir","wspd","pres",
        "nearest_intersection_lat","nearest_intersection_lon",
        "nearest_intersection_id"
    ]
    X = grid_df[feature_columns]
    # 1) get raw probabilities
    raw_proba = model.predict_proba(X)[:, 1].reshape(-1, 1)
    # 2) normal‐quantile transform → z‑scores
    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    z_scores = qt.fit_transform(raw_proba).flatten()
    # 3) map z‑scores to (0,1) via Normal CDF
    from scipy.stats import norm
    scaled = norm.cdf(z_scores)

    grid_df["probability"] = scaled


    return grid_df[["lat", "lon", "borough", "probability"]]
