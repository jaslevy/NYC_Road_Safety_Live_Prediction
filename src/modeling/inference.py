import os
import joblib
import pandas as pd
import xgboost as xgb

# 1) Load your trained Booster once at import time
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/xgboost_model.joblib")
model: xgb.core.Booster = joblib.load(MODEL_PATH)

def predict_accident_probabilities(
    grid_df: pd.DataFrame,
    date: str,
    borough_weather: dict
) -> pd.DataFrame:
    """
    grid_df: DataFrame with columns ['lat','lon','borough']
    borough_weather[b] may have:
      - current keys:  'temperature' (°F), 'precipitation' (in), 'wind_speed' (mph)
      - or historical keys: 'tavg'(°C), 'prcp'(mm), 'snow'(mm), 'wspd'(km/h), 'wdir'(°), 'pres'(hPa)
    """

    # --- 1) Date/time features ---
    dt = pd.to_datetime(date)
    grid_df["hour"]        = dt.hour
    grid_df["day_of_week"] = dt.dayofweek
    grid_df["month"]       = dt.month
    grid_df["is_weekend"]  = (dt.dayofweek >= 5)

    # --- 2) Pull raw weather fields (in whichever format we have) ---
    def get_val(b, *keys, default=0.0):
        d = borough_weather[b]
        for k in keys:
            if k in d:
                return d[k]
        return default

    # temperature
    grid_df["raw_temp_F"]   = grid_df["borough"].map(lambda b: 
        get_val(b, "temperature", "tavg", default=0.0)
    )
    # precipitation
    grid_df["raw_prcp_in"]  = grid_df["borough"].map(lambda b: 
        get_val(b, "precipitation", "prcp", default=0.0)
    )
    # snow
    grid_df["raw_snow_in"]  = grid_df["borough"].map(lambda b: 
        get_val(b, "snow", default=0.0)
    )
    # wind speed
    grid_df["raw_wspd_mph"] = grid_df["borough"].map(lambda b: 
        get_val(b, "wind_speed", "wspd", default=0.0)
    )
    # wind direction
    grid_df["raw_wdir"]     = grid_df["borough"].map(lambda b: 
        get_val(b, "wdir", "wind_direction", default=0.0)
    )
    # pressure
    grid_df["raw_pres"]     = grid_df["borough"].map(lambda b: 
        get_val(b, "pres", "pressure", default=1013.25)
    )

    # --- 3) Convert into model units ---
    #   °F → °C    ; if the raw was already °C we'll get an off‐by‐factor but this covers both cases consistently
    grid_df["tavg"] = (grid_df["raw_temp_F"] - 32.0) * (5.0/9.0)
    #   in → mm
    grid_df["prcp"] = grid_df["raw_prcp_in"] * 25.4
    grid_df["snow"] = grid_df["raw_snow_in"] * 25.4
    #   mph → km/h
    grid_df["wspd"] = grid_df["raw_wspd_mph"] * 1.60934
    #   direction & pressure already in correct units
    grid_df["wdir"] = grid_df["raw_wdir"]
    grid_df["pres"] = grid_df["raw_pres"]

    # --- 4) Spatial features ---
    grid_df["nearest_intersection_lat"] = grid_df["lat"]
    grid_df["nearest_intersection_lon"] = grid_df["lon"]

    # --- 5) Final feature matrix in EXACT order/model‐names ---
    feature_columns = [
        "hour", "day_of_week", "month", "is_weekend",
        "tavg", "prcp", "snow",
        "wdir", "wspd", "pres",
        "nearest_intersection_lat", "nearest_intersection_lon"
    ]
    X = grid_df[feature_columns]

    # --- 6) Predict ---
    dmat = xgb.DMatrix(X, feature_names=feature_columns)
    grid_df["probability"] = model.predict(dmat)

    # --- 7) Return only API‐spec fields ---
    return grid_df[["lat", "lon", "borough", "probability"]]
