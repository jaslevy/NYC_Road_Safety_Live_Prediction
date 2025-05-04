from fastapi import APIRouter, HTTPException
import httpx
import asyncio
from datetime import datetime
from .models import WeatherRequest, WeatherResponse

router = APIRouter()

BOROUGHS = {
    "Manhattan": (40.776676, -73.971321),
    "Brooklyn": (40.650002, -73.949997),
    "Queens": (40.742054, -73.769417),
    "Staten Island": (40.579021, -74.151535),
    "Bronx": (40.837048, -73.865433)
}

async def fetch_borough_weather(client: httpx.AsyncClient, borough: str, lat: float, lon: float):
    """Fetch weather data for a single borough"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["temperature_2m", "precipitation", "wind_speed_10m"],
        "timezone": "America/New_York",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch"
    }

    try:
        response = await client.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        current = data.get("current", {})

        return borough, {
            "temperature": current.get("temperature_2m", 0),
            "precipitation": current.get("precipitation", 0),
            "wind_speed": current.get("wind_speed_10m", 0)
        }
    except Exception as e:
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

            return WeatherResponse(borough_weather=borough_weather)

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid datetime format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
