# src/models/train_xgb.py

import os
import joblib
import pandas as pd
from xgboost import XGBClassifier

# for reproducibility
RANDOM_SEED = 1

path = "../../../data/final_balanced.csv"
# 1) Load your historical CSV (with correct path)
df = pd.read_csv(path, parse_dates=["crash_date"])

# 2) Derive 'hour' from crash_time
df["hour"] = pd.to_datetime(df["crash_time"], format="%H:%M").dt.hour

# 3) Define features + target
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
    "nearest_intersection_id"
]
X = df[feature_columns]
y = df["is_crash"]  # 0/1 target

# 4) Instantiate the classifier with the same parameters you used for xgboost.train
model = XGBClassifier(
    objective="binary:logistic",
    learning_rate=0.1,       # eta
    max_depth=6,
    random_state=RANDOM_SEED,
    eval_metric="auc",
    use_label_encoder=False,
    n_estimators=5000,
    n_jobs=4
)

# 5) Fit on the entire dataset
model.fit(X, y, verbose=True)

# 6) Persist to disk
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_clf_full.joblib")
print("✅ Model trained on full data and saved to models/xgb_clf_full.joblib")
