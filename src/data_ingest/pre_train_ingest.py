import os
import pandas as pd
import subprocess
import time
from dotenv import load_dotenv

load_dotenv()

TRAFFIC_DATA_PATH = "static_data/raw/traffic_data.csv"
WEATHER_DATA_PATH = "static_data/raw/nyc_weather_data.csv"

def file_exists(filepath):
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0

print("\n🚦 Fetching Traffic Data...")
subprocess.run(["python3", "src/data_ingest/fetch_traffic.py"], check=True)

if not file_exists(TRAFFIC_DATA_PATH):
    raise FileNotFoundError(f"Traffic data not found at {TRAFFIC_DATA_PATH}")

print("\n📅 Extracting date range from traffic data...")
traffic_df = pd.read_csv(TRAFFIC_DATA_PATH, parse_dates=["crash_date"])
traffic_df['crash_date'] = pd.to_datetime(traffic_df['crash_date'], errors='coerce')
traffic_df = traffic_df.dropna(subset=['crash_date'])

if traffic_df.empty:
    raise ValueError("No valid crash dates found in traffic data!")

api_start_limit = pd.to_datetime("2016-01-01")
start_date = max(traffic_df["crash_date"].min(), api_start_limit).strftime("%Y-%m-%d")
end_date = traffic_df["crash_date"].max().strftime("%Y-%m-%d")

print(f"Adjusted Date Range: {start_date} → {end_date}")

print("\n Fetching Weather Data...")
subprocess.run(["python3", "src/data_ingest/fetch_weather.py"], check=True)

if not file_exists(WEATHER_DATA_PATH):
    raise FileNotFoundError(f"Weather data not found at {WEATHER_DATA_PATH}")

print("\n All static data successfully ingested and saved locally!")
