import numpy as np

from gbsio import read_from_csv
from partition import SSIPartitioner
from gbs import compute_unmarked_ssi, compute_marked_ssi

## The radius of the neighborhood.
scale = "country"
radius = 0.1

coords, values, raw_values = [], [], []

for fid in ["01"]: # ["00", "01", "02", "03", "04"]:
    ## Read coordinates and values from CSV file.
    coords_part, values_part, raw_values_part = read_from_csv("data/results_{}_gt.csv".format(fid), value_column="haversine_distance_km", coords_columns=["gt_latitude", "gt_longitude"], thres=750, return_raw_performance=True)
    coords.append(coords_part)
    values.append(values_part)
    raw_values.append(raw_values_part)

coords = np.concatenate(coords, axis=0)
values = np.concatenate(values)
raw_values = np.concatenate(raw_values)

print(coords.shape, values.shape, raw_values.shape)

## Construct a partitioner that extract neighborhood points.
partitioner = SSIPartitioner(coords, k=400)

## The index of the center point to evaluate.

locs, accs, ssis = [], [], []

for idx in range(partitioner.N):
    center = coords[idx]

    ## Extract neighbood points.
    presence_idxs = partitioner.get_neighborhood(idx, radius)
    presence_points, presence_values = coords[presence_idxs], values[presence_idxs]

    avg_acc = np.mean(values[presence_idxs])

    ## Use automatic density estimation. Users can manually specify this hyperparameter.

    ## pnp_ssi is the former base geo-bias
    # unmarked_ssi = compute_unmarked_ssi(presence_points, center, radius, density="auto", k=4)
    ## rp_ssi is the former relative geo-bias
    marked_ssi = compute_marked_ssi(presence_points, presence_values, center, radius, density="auto", k=4)

    locs.append(center)
    accs.append(avg_acc)
    ssis.append(marked_ssi)

    # print("Unmarked SSI Score: ", unmarked_ssi, "Marked SSI Score: ", marked_ssi)
    # print("Processed: ", idx, "Marked SSI Score: ", marked_ssi)

    print("Processed: {}\r".format(idx), end="")

np.savez("results/{}_scale_{}_radius".format(scale, radius), locs=locs, accs=accs, ssis=ssis)