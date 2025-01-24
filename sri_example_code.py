import numpy as np

from gbsio import read_from_csv
from partition import SRIPartitioner
from gbs import compute_kl_sri

## Read coordinates and values from CSV file.
coords, values = read_from_csv("data/example_data.csv", value_column="hit@1")

## Construct a partitioner that extract neighborhood points.
partitioner = SRIPartitioner(coords, values)

N = coords.shape[0]
radius = 0.5
scale = 0.05
lag = 0.05
n_splits = 12

total_sris, total_weights = [], []
## The index of the center point to evaluate.
for idx in range(N):
    center = coords[idx]

    ## Extract neighbood points.
    # local_coords_list, local_values_list, neighbor_coords, neighbor_values = partitioner.get_scale_grid(idx, radius, scale)
    # local_coords_list, local_values_list, neighbor_coords, neighbor_values = partitioner.get_distance_lag(idx, radius, lag)
    local_coords_list, local_values_list, neighbor_coords, neighbor_values = partitioner.get_direction_sector(idx, radius, n_splits)

    sris, weights = compute_kl_sri(local_values_list, neighbor_values, bins=2)

    total_sris.append(np.sum(sris * weights))
    total_weights.append(neighbor_values.shape[0] / N)

print(np.sum(np.array(total_sris) * np.array(total_weights)))
