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
    # local_coords_list, local_values_list, total_list = partitioner.get_scale_grid(idx, radius, scale)
    # local_coords_list, local_values_list, total_list = partitioner.get_distance_lag(idx, radius, lag)
    local_coords_list, local_values_list, total_list = partitioner.get_direction_sector(idx, radius, n_splits)

    sris, weights = compute_kl_sri(local_values_list, total_list, bins=2)

    total_sris.append(np.sum(sris * weights))
    total_weights.append(total_list.shape[0] / N)

print(np.sum(np.array(total_sris) * np.array(total_weights)))


# ## pnp_ssi is the former base geo-bias
# pnp_ssi = compute_unmarked_ssi(presence_points, center, radius, density="auto", k=4)
# ## rp_ssi is the former relative geo-bias
# rp_ssi = compute_marked_ssi(presence_points, presence_values, center, radius, density="auto", k=4)
#
# print("Presence v.s. Non-Presence SSI Score: ", pnp_ssi, "Relative Performance SSI Score: ", rp_ssi)