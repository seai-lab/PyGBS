import numpy as np
import pandas as pd

def read_from_csv(filepath, value_column, coords_columns=("lat", "lon"), thres=None):
    assert filepath.endswith(".csv"), "The input must be a CSV file."
    data = pd.read_csv(filepath)
    lats, lons = data[coords_columns[0]].to_numpy(), data[coords_columns[1]].to_numpy()
    nan_idx = (~np.isnan(lats)) & (~np.isnan(lons))

    # Turning latitudes and longitudes into radians.
    coords = ((np.array([lats, lons]).T / 180) * np.pi)[nan_idx]
    values = data[value_column].to_numpy()[nan_idx]

    if thres is not None:
        low_idxs = values < thres
        high_idxs = values >= thres
        values[low_idxs] = -1
        values[high_idxs] = 1

    return coords, values