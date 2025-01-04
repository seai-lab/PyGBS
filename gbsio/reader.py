import numpy as np
import pandas as pd

def read_from_csv(filepath, value_column):
    assert filepath.endswith(".csv"), "The input must be a CSV file."
    data = pd.read_csv(filepath)
    lats, lons = data["lat"].to_numpy(), data["lon"].to_numpy()
    nan_idx = (~np.isnan(lats)) & (~np.isnan(lons))

    # Turning latitudes and longitudes into radians.
    coords = ((data[["lat", "lon"]].to_numpy() / np.array([90, 180])) * np.array([np.pi / 2, np.pi]))[nan_idx]
    values = data[value_column].to_numpy()[nan_idx]

    return coords, values