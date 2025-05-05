from pydantic import BaseModel
from typing import Dict, Optional, List

class WeatherRequest(BaseModel):
    datetime: str

class WeatherResponse(BaseModel):
    borough_weather: Dict[str, Dict[str, float]]

class AccidentPredictionRequest(BaseModel):
    date: str

class CoordinatePrediction(BaseModel):
    lat: float
    lon: float
    borough: str
    probability: float

class AccidentPredictionResponse(BaseModel):
    predictions: List[CoordinatePrediction]
    date: str
