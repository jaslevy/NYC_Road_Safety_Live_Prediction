from pydantic import BaseModel
from typing import Dict, Optional

class WeatherRequest(BaseModel):
    datetime: str

class WeatherResponse(BaseModel):
    borough_weather: Dict[str, Dict[str, float]]
