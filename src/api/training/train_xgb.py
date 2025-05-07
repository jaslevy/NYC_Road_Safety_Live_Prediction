# src/models/train_xgb.py

import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# 1) Load your historical CSV
df = pd.read_csv("data/final_balanced.csv", parse_dates=["crash_date"])

# 2) Derive 'hour' from crash_time
df["hour"] = pd.to_datetime(df["crash_time"], format="%H:%M").dt.hour

# 3) Convert any units if needed (e.g. if your CSV is already in model units, skip)
#    Assuming your CSV tavg/prcp/snow/wspd are already in °C, mm, mm, km/h respectively.

# 4) Define features + target
feature_columns = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "tavg",    # °C
    "prcp",    # mm
    "snow",    # mm
    "wdir",    # °
    "wspd",    # km/h
    "pres",    # hPa
    "nearest_intersection_lat",
    "nearest_intersection_lon",
    "distance_to_intersection_km"
]
X = df[feature_columns]
y = df["is_crash"]  # 0/1

# 5) Train on the *entire* dataset
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    n_jobs=4,
    random_state=42
)
model.fit(X, y, verbose=True)

# 6) Save the fitted model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_clf_full.joblib")
print("✅ Model trained on full data and saved to models/xgb_clf_full.joblib")