import os
import json
import pandas as pd
from geopy.distance import great_circle

# your five borough centroids
BOROUGHS = {
    "Manhattan": (40.776676, -73.971321),
    "Brooklyn":  (40.650002, -73.949997),
    "Queens":    (40.742054, -73.769417),
    "Staten Island": (40.579021, -74.151535),
    "Bronx":     (40.837048, -73.865433),
}

# Build the path to src/data/intersections.json
DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,        # up from scripts/
    "src", "data", "intersections.json"
)

# 1) Load the raw intersections
with open(DATA_PATH, "r") as f:
    raw = json.load(f)

# 2) Enrich each point
enriched = []
for pt in raw:
    # your JSON has keys 'id', 'lat', 'lon'
    lat, lon = pt["lat"], pt["lon"]
    # compute borough (example)
    borough = min(
        BOROUGHS.keys(),
        key=lambda b: great_circle((lat, lon), BOROUGHS[b]).km
    )
    enriched.append({
        "id":            pt["id"],
        "lat":           lat,
        "lon":           lon,
        "nearest_borough": borough
    })

# 3) Save it back (or to a new file)
OUT_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "src", "data", "intersections_enriched.json"
)
with open(OUT_PATH, "w") as f:
    json.dump(enriched, f, indent=2)

print(f"Wrote {len(enriched)} points with boroughs to {OUT_PATH}")