from setuptools import setup, find_packages

setup(
    name="nyc_road_safety",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'fastapi==0.109.0',
        'uvicorn==0.27.0',
        'httpx==0.26.0',
        'numpy==1.26.0',
        'pandas==2.2.0',
        'pydantic==2.6.1',
        'python-multipart==0.0.9',
        'sodapy',
        'python-dotenv',
        'openmeteo_requests',
        'requests_cache',
        'retry_requests',
        'flask',
        'folium',
        'watchdog==3.0.0',
        'pytest==8.0.0',
        'requests==2.31.0',
        'geopy==2.4.1',
        'xgboost',
        'scikit-learn'
    ],
)