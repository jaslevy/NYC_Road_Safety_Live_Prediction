import numpy as np
import pandas as pd

# Define bounding box for NYC coordinates
NYC_BOUNDS = {
    "north": 40.91553,   # Northern boundary (above the Bronx)
    "south": 40.49612,   # Southern boundary (below Staten Island)
    "east": -73.70018,   # Eastern boundary (beyond Queens)
    "west": -74.25909    # Western boundary (beyond Staten Island)
}

# Borough central coordinates (already defined in the project)
BOROUGH_CENTERS = {
    "Manhattan": (40.776676, -73.971321),
    "Brooklyn": (40.650002, -73.949997),
    "Queens": (40.742054, -73.769417),
    "Staten Island": (40.579021, -74.151535),
    "Bronx": (40.837048, -73.865433)
}

def generate_nyc_grid(resolution=0.01):
    """
    Generate a grid of coordinates covering NYC with specified resolution.

    Args:
        resolution: Distance between grid points in degrees (approx. 0.01° ≈ 1km)

    Returns:
        DataFrame with columns lat, lon, borough
    """
    # Create the coordinate grid
    lat_grid = np.arange(NYC_BOUNDS["south"], NYC_BOUNDS["north"], resolution)
    lon_grid = np.arange(NYC_BOUNDS["west"], NYC_BOUNDS["east"], resolution)

    # Meshgrid to create all combinations
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # Flatten to 1D arrays
    lats = lat_mesh.flatten()
    lons = lon_mesh.flatten()

    # Create dataframe
    grid_df = pd.DataFrame({
        "lat": lats,
        "lon": lons
    })

    # Assign each point to nearest borough
    grid_df["borough"] = grid_df.apply(
        lambda row: min(
            BOROUGH_CENTERS.items(),
            key=lambda x: (row["lat"] - x[1][0])**2 + (row["lon"] - x[1][1])**2
        )[0],
        axis=1
    )

    return grid_df

# Generate grid with default resolution
nyc_grid = generate_nyc_grid()

def get_nyc_grid():
    """Returns the NYC coordinate grid"""
    return nyc_grid