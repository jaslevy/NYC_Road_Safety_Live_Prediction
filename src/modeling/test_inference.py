import pandas as pd
from inference import predict_accident_probabilities

# 1) Build a mini‑grid
test_df = pd.DataFrame([
    {"lat": 40.5, "lon": -73.7, "borough": "Queens"},
    {"lat": 40.8, "lon": -73.9, "borough": "Bronx"},
])

# 2) Mock the API weather dict (°F, inch, mph, °, hPa)
borough_weather = {
    "Queens": {
        "tavg": 59.3,    # °F
        "prcp": 0.146,   # inch
        "snow": 0.0,     # inch
        "wspd": 13.0,    # mph
        "wdir": 147.0,   # °
        "pres": 1018.0   # hPa
    },
    "Bronx": {
        "tavg": 59.9,    # °F
        "prcp": 0.48,    # inch
        "snow": 0.0,     # inch
        "wspd": 13.3,    # mph
        "wdir": 143.0,   # °
        "pres": 1017.9   # hPa
    }
}

# 3) Predict
result = predict_accident_probabilities(test_df, "2025-05-06", borough_weather)

# 4) Print and check uniqueness
print("=== PREDICTIONS ===")
print(result)

unique_probs = result["probability"].unique()
print("\nUnique probabilities:", unique_probs)
