# src/preprocessing/intersections.py

import json
import pandas as pd
from pathlib import Path

# load the JSON you uploaded
data_path = Path(__file__).parent.parent / "data" / "intersections.json"
intersections_df = pd.read_json(data_path)

# rename ‘id’ to the name our model expects
intersections_df.rename(
    columns={'id': 'nearest_intersection_id'},
    inplace=True
)
