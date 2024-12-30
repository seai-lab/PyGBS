import sys
import os

import numpy as np
import pandas as pd

from gbs import compute_presence_nonpresence_ssi, compute_relative_performance_ssi


# convert km to randian
def km_to_radians(km):
    # Earth's average radius in kilometers
    earth_radius_km = 6371.0
    # Convert km to radians
    radians = km / earth_radius_km
    return radians


# preprocess_data, replace 0 to -1, drop NaN coords
def preprocess_data(data_df):
    # Replace 0 in binary prediction with -1 (ensure mean is 0)
    for column in ['hit@1', 'hit@3']:
        if column in data_df.columns:
            data_df[column] = data_df[column].replace(0, -1)

    # Delete rows with NaN in 'lon' or 'lat'
    data_df.dropna(subset=['lon', 'lat'], inplace=True)

    # Convert 'lon' and 'lat' to radians
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
    
    # Loop through all points in the dataset to find neighbors
    for point in data:
        lat, lon = point[0], point[1]
        distance = haversine(low_lat, low_lon, lat, lon)  # Angular distance in radians

        # Check if the distance is within the specified radius
        if distance <= radius:
            metric = point[2]  # Assuming 3rd column in data is the metric (e.g., hit@1, hit@3)
            neighbors.append([lat, lon, metric])  # Store the neighbor's lat, lon, and metric

    return np.array(neighbors)

def compute_weighted_average_ssi(neighbors, ssi_function, low_point, radius, density, k=4):
    weights = []  # the weights (neighbor count or other criteria)
    ssi_values = []  

    for neighbor in neighbors:
        lat, lon = neighbor[0], neighbor[1]
        metric = neighbor[2]
        
        ssi = ssi_function(neighbors[:, :2], low_point, radius, density, k)
        ssi_values.append(ssi)

        weight = 1 / (len(neighbors) + 1)
        weights.append(weight)
    weighted_average_ssi = np.average(ssi_values, weights=weights)
    return weighted_average_ssi

def main(input_folder="result_tables", radius=0.01, metric='hit@1', densities=None):
    if densities is None:
        densities = [2000000, 2000000]  # Default density values if not provided
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):  
            input_file_path = os.path.join(input_folder, filename)

            try:
                data_df = pd.read_csv(input_file_path)
                data_df = preprocess_data(data_df)
                data = data_df[['lat', 'lon', metric]].to_numpy()

                low_performance = data[:, 2] == -1  # metric is the third column, index 2
                low_data = data[low_performance]

                # Process each density value
                for density in densities:
                    count = 0
                    total_pnp_ssi = 0
                    total_rp_ssi = 0
                    
                    # Iterate over the low performance points
                    for low_point in low_data:
                        neighbors = find_neighbors_within_radius(data, low_point, radius)

                        pnp_ssi = compute_presence_nonpresence_ssi(neighbors[:, :2], low_point, radius, density, k=4)
                        rp_ssi = compute_relative_performance_ssi(neighbors[:, :2], neighbors[:, 2], low_point, radius, density, k=4)

                        count += 1
                        total_pnp_ssi += pnp_ssi
                        total_rp_ssi += rp_ssi
                        
                        print(f"pnp_ssi: {pnp_ssi}, rp_ssi: {rp_ssi}")

                    if count > 0:
                        avg_pnp_ssi = total_pnp_ssi / count
                        avg_rp_ssi = total_rp_ssi / count
                        print(f"Density: {density}, Average pnp_ssi: {avg_pnp_ssi}, Average rp_ssi: {avg_rp_ssi}")
                    else:
                        print(f"No low points found for density: {density}")

                print(f"Data from {filename} Finished")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main() 