import sys
import os

import numpy as np
import pandas as pd

from gbs import compute_presence_nonpresence_ssi, compute_relative_performance_ssi

# convert km to randian
def km_to_radians(km):
    earth_radius_km = 6371.0
    radians = km / earth_radius_km
    return radians

km = 100
radians = km_to_radians(km)
print(f"{km} km is equivalent to {radians} radians")

# preprocess_data, replace 0 to -1, drop NaN coords
def preprocess_data(data_df):
    # Replace 0 in binary prediction with -1 (ensure mean is 0)
    for column in ['hit@1', 'hit@3']:
        if column in data_df.columns:
            data_df[column] = data_df[column].replace(0, -1)

    # Delete rows with NaN in 'lon' or 'lat'
    data_df.dropna(subset=['lon', 'lat'], inplace=True)

    data_df[['lon', 'lat']] = np.radians(data_df[['lon', 'lat']].values)

    return data_df

# Function to calculate the Haversine distance between two points in radians
def haversine(lat1, lon1, lat2, lon2):
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))  # central angle in radians
    return c  # angular distance in radians

def find_neighbors_within_radius(data, low_point, radius):
    neighbors = []
    '''
    [Input]
        data: full data points
        low_point: low performance point (lat, lon, metric), the metric (e.g., hit@1, hit@3)
        radius: distance in radian
    [Output]
        neighbors of the given low_point, as a numpy array, [lat, lon, metric]
    '''
    low_lat, low_lon = low_point[0], low_point[1]

    for point in data:
        lat, lon = point[0], point[1]
        # Angular distance in radians
        distance = haversine(low_lat, low_lon, lat, lon)

        if distance <= radius:
            metric = point[2]
            # neighbor's lat, lon, and metric
            neighbors.append([lat, lon, metric])

    return np.array(neighbors)

input_folder = "result_tables"
radius = 0.01
metric = 'hit@1'

# Function to automatically compute the density hyperparameter
def auto_density(radius, n_neighbors):
    return int(np.ceil((5 * n_neighbors) / (np.pi * radius**2)))

for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):  # only CSV files
        input_file_path = os.path.join(input_folder, filename)

        data_df = pd.read_csv(input_file_path)
        data_df = preprocess_data(data_df)
        data = data_df[['lat', 'lon', metric]].to_numpy()

        # Select points with low performance
        low_performance = data[:, data_df.columns.get_loc(metric)] == -1
        low_data = data[low_performance]

        count = 0
        total_pnp_ssi = 0
        total_rp_ssi = 0
        for low_point in low_data:
            neighbors = find_neighbors_within_radius(data, low_point, radius)
            n_neighbors = neighbors.shape[0]
            density = auto_density(radius, n_neighbors)

            pnp_ssi = compute_presence_nonpresence_ssi(neighbors[:, :2], low_point, radius, density, k=4)
            rp_ssi = compute_relative_performance_ssi(neighbors[:, :2], neighbors[:, 2], low_point, radius, density, k=4)

            total_pnp_ssi += n_neighbors * pnp_ssi
            total_rp_ssi += n_neighbors * rp_ssi
            count += n_neighbors
            print(f"weighted pnp_ssi: {pnp_ssi}, weighted rp_ssi: {rp_ssi}")

        avg_pnp_ssi = total_pnp_ssi / count
        avg_rp_ssi = total_rp_ssi / count
        print(f"Average pnp_ssi: {avg_pnp_ssi}, Average rp_ssi: {avg_rp_ssi}")
        print(f"Data from {filename} Finished")