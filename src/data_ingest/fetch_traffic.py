import joblib
import pandas as pd
import numpy as np
import os
import xgboost as xgb   # add this import

# Adjust path to wherever your .joblib lives
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/xgboost_model.joblib")
model = joblib.load(MODEL_PATH)  # this is a Booster

def predict_accident_probabilities(grid_df, date, borough_weather):
    # … your feature‑prep code …

    feature_columns = ['hour', 'month', 'wdir', 'pres', 'wspd', 'tmin', 'tavg']
    X = grid_df[feature_columns]

    # === Replace predict_proba with native XGBoost predict on DMatrix ===
    dmat = xgb.DMatrix(X)
    y_proba = model.predict(dmat)  # returns probability for '1'

    # attach back to DataFrame
    grid_df["crash_probability"] = y_proba
    return grid_df[["latitude", "longitude", "crash_probability"]]
