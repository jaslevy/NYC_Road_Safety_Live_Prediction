# src/preprocessing/intersections.py

import json
import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load the JSON you uploaded
data_path = Path(__file__).parent.parent / "data" / "intersections.json"

try:
    if data_path.exists():
        intersections_df = pd.read_json(data_path)
        # rename 'id' to the name our model expects
        intersections_df.rename(
            columns={'id': 'nearest_intersection_id'},
            inplace=True
        )
        logger.info(f"Loaded intersections data from {data_path}")
    else:
        logger.warning(f"Intersections data file not found at {data_path}")
        # Create a dummy DataFrame for development/testing
        intersections_df = pd.DataFrame({
            'nearest_intersection_id': [0],
            'lat': [40.7128],  # New York City coordinates
            'lon': [-74.0060]
        })
except Exception as e:
    logger.error(f"Error loading intersections data: {str(e)}")
    # Create a dummy DataFrame for development/testing
    intersections_df = pd.DataFrame({
        'nearest_intersection_id': [0],
        'lat': [40.7128],  # New York City coordinates
        'lon': [-74.0060]
    })
