from fastapi import APIRouter, HTTPException
import httpx
import asyncio
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

async def fetch_borough_weather(client: httpx.AsyncClient, borough: str, lat: float, lon: float, date_str: Optional[str] = None):
    """Fetch weather data for a single borough"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "America/New_York",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch"
    }

    # If date is provided, use historical data endpoint
    if date_str:
        # Parse the date
        date = datetime.fromisoformat(date_str)

        # For historical data
        params.update({
            "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                      "precipitation_sum", "snowfall_sum", "wind_speed_10m_max",
                      "wind_direction_10m_dominant", "pressure_msl_mean"],
            "start_date": date_str,
            "end_date": date_str
        })
    else:
        # For current data
        params.update({
            "current": ["temperature_2m", "precipitation", "wind_speed_10m",
                        "wind_direction_10m", "pressure_msl"]
        })

    try:
        logger.info(f"Fetching weather for {borough} at coordinates ({lat}, {lon}) for date {date_str}")
        logger.info(f"Request params: {params}")

        response = await client.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Weather API response for {borough}: {data}")

        if date_str:
            # Handle historical data
            daily = data.get("daily", {})
            if not daily or len(daily.get("time", [])) == 0:
                logger.error(f"No historical data available for {borough} on {date_str}")
                return borough, {"error": "No historical data available for this date"}

            # Create dummy data if the API doesn't return historical data
            # This is a workaround for testing
            if "temperature_2m_mean" not in daily or not daily["temperature_2m_mean"]:
                logger.warning(f"Using dummy weather data for {borough} on {date_str}")
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

            return borough, {
                "tavg": float(daily.get("temperature_2m_mean", [0])[0]),
                "tmin": float(daily.get("temperature_2m_min", [0])[0]),
                "tmax": float(daily.get("temperature_2m_max", [0])[0]),
                "prcp": float(daily.get("precipitation_sum", [0])[0]),
                "snow": float(daily.get("snowfall_sum", [0])[0]),
                "wdir": float(daily.get("wind_direction_10m_dominant", [0])[0]),
                "wspd": float(daily.get("wind_speed_10m_max", [0])[0]),
                "pres": float(daily.get("pressure_msl_mean", [0])[0]),
                "weather_borough": borough
            }
        else:
            # Handle current data
            current = data.get("current", {})

            return borough, {
                "temperature": float(current.get("temperature_2m", 0)),
                "precipitation": float(current.get("precipitation", 0)),
                "wind_speed": float(current.get("wind_speed_10m", 0))
            }
    except Exception as e:
        logger.error(f"Error fetching weather for {borough}: {str(e)}")
        return borough, {"error": str(e)}

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

            # Get the NYC grid
            grid_df = get_nyc_grid()
            logger.info(f"NYC grid loaded: {len(grid_df)} points")

            # Use a smaller subset of the grid for performance (sampling 5%)
            sampled_grid = grid_df.sample(n=500, random_state=42)
            logger.info(f"Using sampled grid with {len(sampled_grid)} points")

            # Predict accident probabilities for each coordinate
            logger.info("Starting accident probability prediction")
            predictions_df = predict_accident_probabilities(sampled_grid, date_str, borough_weather)
            logger.info(f"Prediction complete for {len(predictions_df)} points")

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
