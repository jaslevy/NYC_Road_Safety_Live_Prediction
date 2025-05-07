from fastapi.testclient import TestClient
from index import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "NYC Road Safety API is running"}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_weather_endpoint():
    response = client.post(
        "/api/weather",
        json={"datetime": "2024-03-14T12:00:00"}
    )
    assert response.status_code == 200
    assert "borough_weather" in response.json()

def test_accident_prediction():
    response = client.post(
        "/api/accident-prediction",
        json={"date": "2024-03-14"}
    )
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert "date" in response.json()