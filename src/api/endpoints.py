from fastapi import APIRouter, HTTPException
import httpx
import asyncio
import json
import pandas as pd
from datetime import datetime
import numpy as np
import logging
from typing import Dict, Any, Optional
from .models import (
    WeatherRequest,
    WeatherResponse,
    AccidentPredictionRequest,
    AccidentPredictionResponse,
    CoordinatePrediction
)
from src.preprocessing.nyc_grid import get_nyc_grid
from src.modeling.inference import predict_accident_probabilities
from src.preprocessing.nyc_grid import get_nyc_grid
from src.modeling.inference import predict_accident_probabilities
from geopy.distance import great_circle
from src.preprocessing.intersections import intersections_df
import os


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Endpoint to check the health status of the API.
    Returns a 200 OK response if the API is healthy.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "nyc-accident-prediction-api"
    }


BOROUGHS = {
    "Manhattan": (40.776676, -73.971321),
    "Brooklyn": (40.650002, -73.949997),
    "Queens": (40.742054, -73.769417),
    "Staten Island": (40.579021, -74.151535),
    "Bronx": (40.837048, -73.865433)
}
def _assign_borough(lat: float, lon: float) -> str:
    # find the borough whose centroid is closest
    return min(
        BOROUGHS.keys(),
        key=lambda b: great_circle((lat, lon), BOROUGHS[b]).km
    )

async def fetch_borough_weather(client: httpx.AsyncClient, borough: str, lat: float, lon: float, date_str: Optional[str] = None):
    """Fetch weather data for a single borough"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "America/New_York",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "current": ["temperature_2m", "precipitation", "wind_speed_10m",
                    "wind_direction_10m", "pressure_msl"]
    }

    try:
        logger.info(f"Fetching weather for {borough} at coordinates ({lat}, {lon})")
        logger.info(f"Request params: {params}")

        response = await client.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Weather API response for {borough}: {data}")

        # Handle current data and convert to model format
        current = data.get("current", {})

        return borough, {
            "tavg": float(current.get("temperature_2m", 60.0)),  # Use temperature as average
            "tmin": float(current.get("temperature_2m", 50.0)) - 5.0,  # Estimate min temp
            "tmax": float(current.get("temperature_2m", 70.0)) + 5.0,  # Estimate max temp
            "prcp": float(current.get("precipitation", 0.0)),
            "snow": 0.0,  # Default to 0 since it's current data
            "wdir": float(current.get("wind_direction_10m", 180.0)),
            "wspd": float(current.get("wind_speed_10m", 10.0)),
            "pres": float(current.get("pressure_msl", 1010.0)),
            "weather_borough": borough
        }
    except Exception as e:
        logger.error(f"Error fetching weather for {borough}: {str(e)}")
        # Return default values if there's an error
        return borough, {
            "tavg": 60.0,
            "tmin": 50.0,
            "tmax": 70.0,
            "prcp": 0.0,
            "snow": 0.0,
            "wdir": 180.0,
            "wspd": 10.0,
            "pres": 1010.0,
            "weather_borough": borough
        }

@router.post("/weather", response_model=WeatherResponse)
async def get_weather(request: WeatherRequest):
    try:
        # Parse the datetime
        dt = datetime.fromisoformat(request.datetime)

        # Create async HTTP client
        async with httpx.AsyncClient() as client:
            # Create tasks for all boroughs
            tasks = [
                fetch_borough_weather(client, borough, lat, lon)
                for borough, (lat, lon) in BOROUGHS.items()
            ]

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)

            # Convert results to dictionary
            borough_weather = dict(results)

            # Check for errors
            errors = [borough for borough, data in borough_weather.items() if "error" in data]
            if errors:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error fetching weather for boroughs: {', '.join(errors)}"
                )

            # Convert any error values to float to satisfy the model
            weather_data: Dict[str, Dict[str, float]] = {}
            for borough, data in borough_weather.items():
                weather_data[borough] = {k: float(v) if isinstance(v, (int, float)) else 0.0
                                        for k, v in data.items()}

            return WeatherResponse(borough_weather=weather_data)

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid datetime format")
    except Exception as e:
        logger.error(f"Error in get_weather: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/accident-prediction", response_model=AccidentPredictionResponse)
async def predict_accidents(request: AccidentPredictionRequest):
    try:
        # Validate date format
        date = datetime.fromisoformat(request.date)
        date_str = date.strftime("%Y-%m-%d")
        logger.info(f"Processing accident prediction request for date: {date_str}")

        # Create async HTTP client
        async with httpx.AsyncClient() as client:
            # Fetch weather data for all boroughs for the specified date
            tasks = [
                fetch_borough_weather(client, borough, lat, lon, date_str)
                for borough, (lat, lon) in BOROUGHS.items()
            ]

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)

            # Convert results to dictionary
            borough_weather_raw = dict(results)

            # Check for errors
            errors = [f"{borough}: {data.get('error', 'Unknown error')}"
                     for borough, data in borough_weather_raw.items()
                     if "error" in data]

            if errors:
                error_details = "; ".join(errors)
                logger.error(f"Weather API errors: {error_details}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error fetching weather for boroughs: {', '.join([e.split(':')[0] for e in errors])}"
                )

            # Clean up data for model input
            borough_weather = {}
            for borough, data in borough_weather_raw.items():
                if "error" not in data:
                    borough_weather[borough] = data

            logger.info(f"Weather data retrieved successfully for all boroughs")

            # JSON INFO
            with open(os.path.join(os.path.dirname(__file__), "data", "intersections_enriched.json")) as f:
                enriched = json.load(f)

            # 2) Build your grid_df
            grid_df = (
            pd.DataFrame(enriched)
            .rename(columns={
          "lat": "lat",
          "lon": "lon",
          "nearest_borough": "borough"
            })
            .sample(n=500, random_state=42)
            .reset_index(drop=True))

            predictions_df = predict_accident_probabilities(grid_df, date_str, borough_weather)

            # Convert to response format
            predictions = [
               CoordinatePrediction(
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    borough=row["borough"],
                    probability=float(row["probability"])
                )
                for _, row in predictions_df.iterrows()
            ]

            logger.info(f"Returning {len(predictions)} predictions")
            return AccidentPredictionResponse(
                predictions=predictions,
                date=date_str
            )

    except ValueError as e:
        logger.error(f"ValueError in predict_accidents: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in predict_accidents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
