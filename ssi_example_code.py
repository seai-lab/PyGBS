import numpy as np

from gbsio import read_from_csv
from partition import SSIPartitioner
from gbs import compute_unmarked_ssi, compute_marked_ssi

## Read coordinates and values from CSV file.
coords, values = read_from_csv("data/results_01_gt.csv", value_column="haversine_distance_km", coords_columns=["gt_latitude", "gt_longitude"], thres=25)

## Construct a partitioner that extract neighborhood points.
partitioner = SSIPartitioner(coords, k=100)

## The index of the center point to evaluate.

locs, ssis = [], []

for idx in range(partitioner.N):
    center = coords[idx]

    ## The radius of the neighborhood.
    radius = 0.01

    ## Extract neighbood points.
    presence_idxs = partitioner.get_neighborhood(idx, radius)
    presence_points, presence_values = coords[presence_idxs], values[presence_idxs]

    ## Use automatic density estimation. Users can manually specify this hyperparameter.

    ## pnp_ssi is the former base geo-bias
    unmarked_ssi = compute_unmarked_ssi(presence_points, center, radius, density="auto", k=4)
    ## rp_ssi is the former relative geo-bias
    marked_ssi = compute_marked_ssi(presence_points, presence_values, center, radius, density="auto", k=4)

    locs.append(center)
    ssis.append(marked_ssi)

    # print("Unmarked SSI Score: ", unmarked_ssi, "Marked SSI Score: ", marked_ssi)
    print("Processed: ", idx, "Marked SSI Score: ", marked_ssi)

np.savez("results/province_scale", locs=locs, ssis=ssis)