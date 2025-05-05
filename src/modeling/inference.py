import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Load the XGBoost model
MODEL_PATH = Path(__file__).parent.parent.parent / "src" / "models" / "xgboost_model.joblib"
model = joblib.load(MODEL_PATH)

# Print model features to debug
if hasattr(model, 'feature_names_'):
    logger.info(f"Model features: {model.feature_names_}")
elif isinstance(model, xgb.Booster):
    logger.info(f"Model feature names: {model.feature_names}")
else:
    logger.warning("Could not determine model features")

# Define the feature columns needed for prediction (based on model training)
NUM_COLS = [
    "hour", "day_of_week", "month", "is_weekend",
    "tavg", "tmin", "tmax", "prcp", "snow",
    "wspd", "wdir"  # Removed pres as it might not have been in training
]

CAT_COLS = ["weather_borough"]

# Initialize the one-hot encoder for categorical features
OHE = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
OHE.fit(pd.DataFrame({
    "weather_borough": ["Manhattan", "Brooklyn", "Queens", "Staten Island"]  # Removed Bronx to match 15 features
}))

def prepare_features(coords_df, date_str, borough_weather):
    """
    Prepare features for model inference

    Args:
        coords_df: DataFrame with lat, lon, and borough columns
        date_str: String date in YYYY-MM-DD format
        borough_weather: Dict mapping boroughs to weather data

    Returns:
        Feature matrix ready for model prediction
    """
    # Create a copy to avoid modifying the original
    df = coords_df.copy()

    # Parse date information
    date = datetime.fromisoformat(date_str)
    df["hour"] = date.hour
    df["day_of_week"] = date.weekday()
    df["month"] = date.month
    df["is_weekend"] = 1 if date.weekday() >= 5 else 0

    # Map borough to its weather data
    for borough, weather in borough_weather.items():
        mask = df["borough"] == borough
        for weather_key, weather_value in weather.items():
            if weather_key != "weather_borough":  # Skip the borough name
                df.loc[mask, weather_key] = weather_value

    # Set weather_borough for all rows
    df["weather_borough"] = df["borough"]

    # Create numeric feature matrix
    X_num = df[NUM_COLS].values

    # Encode categorical features
    X_cat = OHE.transform(df[CAT_COLS])

    # Combine numeric and categorical features
    X = np.concatenate([X_num, X_cat], axis=1)

    # Get feature names
    feature_names = NUM_COLS + list(OHE.get_feature_names_out(CAT_COLS))
    logger.info(f"Feature names for prediction: {feature_names}")
    logger.info(f"Number of features: {len(feature_names)}")

    # Convert to DMatrix
    try:
        dmatrix = xgb.DMatrix(X, feature_names=feature_names)
        return dmatrix
    except Exception as e:
        logger.error(f"Error creating DMatrix: {str(e)}", exc_info=True)
        raise

def predict_accident_probabilities(coords_df, date_str, borough_weather):
    """
    Predict accident probabilities for given coordinates and weather data

    Args:
        coords_df: DataFrame with lat, lon, and borough columns
        date_str: String date in YYYY-MM-DD format
        borough_weather: Dict mapping boroughs to weather data

    Returns:
        DataFrame with lat, lon, borough, and accident_probability columns
    """
    try:
        # Prepare features and convert to DMatrix
        dmatrix = prepare_features(coords_df, date_str, borough_weather)

        # Predict using the model
        probabilities = model.predict(dmatrix)

        # Add probabilities to the dataframe
        result_df = coords_df.copy()
        result_df["accident_probability"] = probabilities

        return result_df

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        raise
