import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import time

traffic_df = pd.read_csv("static_data/raw/traffic_data.csv", parse_dates=["crash_date"])
traffic_df['crash_date'] = pd.to_datetime(traffic_df['crash_date'], errors='coerce')

traffic_df = traffic_df.dropna(subset=['crash_date'])

if traffic_df.empty:
    raise ValueError("Error: Traffic data is empty after processing!")

start_date = traffic_df["crash_date"].min().strftime("%Y-%m-%d")
end_date = traffic_df["crash_date"].max().strftime("%Y-%m-%d")

# API restriction: start_date cannot be earlier than 2016-01-01
api_start_limit = "2016-01-01"

if start_date < api_start_limit:
    print(f"⚠️ Adjusting start_date from {start_date} to {api_start_limit} due to API limits.")
    start_date = api_start_limit

print(f"Earliest Date: {start_date}")
print(f"Latest Date: {end_date}")

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

boroughs = {
    "Manhattan": (40.776676, -73.971321),
    "Brooklyn": (40.650002, -73.949997),
    "Queens": (40.742054, -73.769417),
    "Staten Island": (40.579021, -74.151535),
    "Bronx": (40.837048, -73.865433)
}

# Open-Meteo API endpoint
url = "https://api.open-meteo.com/v1/forecast"

# Weather vars to fetch
weather_params = {
    "daily": [
        "rain_sum", "precipitation_probability_max", "sunshine_duration",
        "showers_sum", "wind_speed_10m_max", "snowfall_sum",
        "wind_gusts_10m_max", "temperature_2m_min", "daylight_duration",
        "wind_direction_10m_dominant", "precipitation_sum", "precipitation_hours",
        "uv_index_clear_sky_max"
    ],
    "timezone": "America/New_York",
    "wind_speed_unit": "mph",
    "temperature_unit": "fahrenheit",
    "precipitation_unit": "inch",
    "start_date": start_date,
    "end_date": end_date,
}

all_boroughs_weather = []

for borough, (lat, lon) in boroughs.items():
    print(f"Fetching weather for {borough} ({lat}, {lon})...")

    params = weather_params.copy()
    params["latitude"] = lat
    params["longitude"] = lon
    time.sleep(10)

    responses = openmeteo.weather_api(url, params=params)
    
    if not responses:
        print(f"Warning: No response for {borough}, skipping...")
        continue

    response = responses[0] 
    daily = response.Daily()
    
    if not daily.Variables(0).ValuesAsNumpy().size:
        print(f"Warning: No weather data returned for {borough}, skipping...")
        continue

    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        ),
        "borough": borough,
        "rain_sum": daily.Variables(0).ValuesAsNumpy(),
        "precipitation_probability_max": daily.Variables(1).ValuesAsNumpy(),
        "sunshine_duration": daily.Variables(2).ValuesAsNumpy(),
        "showers_sum": daily.Variables(3).ValuesAsNumpy(),
        "wind_speed_10m_max": daily.Variables(4).ValuesAsNumpy(),
        "snowfall_sum": daily.Variables(5).ValuesAsNumpy(),
        "wind_gusts_10m_max": daily.Variables(6).ValuesAsNumpy(),
        "temperature_2m_min": daily.Variables(7).ValuesAsNumpy(),
        "daylight_duration": daily.Variables(8).ValuesAsNumpy(),
        "wind_direction_10m_dominant": daily.Variables(9).ValuesAsNumpy(),
        "precipitation_sum": daily.Variables(10).ValuesAsNumpy(),
        "precipitation_hours": daily.Variables(11).ValuesAsNumpy(),
        "uv_index_clear_sky_max": daily.Variables(12).ValuesAsNumpy(),
    }

    daily_df = pd.DataFrame(daily_data)
    all_boroughs_weather.append(daily_df)

# Combine all boroughs into a single DataFrame
if all_boroughs_weather:
    weather_df = pd.concat(all_boroughs_weather, ignore_index=True)
    weather_csv_path = "static_data/raw/nyc_weather_data.csv"
    weather_df.to_csv(weather_csv_path, index=False)
    print(f"Weather data saved to {weather_csv_path}")
else:
    print("No weather data fetched. Check API responses.")

