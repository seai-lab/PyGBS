import numpy as np

from gbsio import read_from_csv
from partition import SRIPartitioner
from gbs import compute_kl_sri

## Read coordinates and values from CSV file.
coords, values, raw_values = [], [], []
thres_dict = {
    "continent": 2500,
    "country": 750,
    "region": 200,
    "city": 25,
    "street": 1
}

thres = "street"

# for fid in ["00"]: # ["00", "01", "02", "03", "04"]:
    ## Read coordinates and values from CSV file.
coords_part, values_part, raw_values_part = read_from_csv("data/results_combined_gt.csv",
            value_column="haversine_distance_km",
            coords_columns=["gt_latitude", "gt_longitude"],
            thres=thres_dict[thres],
            return_raw_performance=True)
coords.append(coords_part)
values.append(values_part)
raw_values.append(raw_values_part)

coords = np.concatenate(coords, axis=0, dtype=np.float16)
values = np.concatenate(values, dtype=np.float16)
raw_values = np.concatenate(raw_values, dtype=np.float16)

print(coords.shape, values.shape, raw_values.shape)

## Construct a partitioner that extract neighborhood points.
partitioner = SRIPartitioner(coords, k=400)

N = coords.shape[0]
radius = 0.05
scale = 0.01
lag = 0.01
n_splits = 8

total_sris = {"sg":[], "dl":[], "ds":[]}
total_weights = {"sg":[], "dl":[], "ds":[]}
## The index of the center point to evaluate.
for idx in range(N):
    center = coords[idx]

    ## Extract neighbood points.
    for tp in ["sg", "dl", "ds"]:
        if tp == "sg":
            local_idx_list, neighbor_idxs = partitioner.get_scale_grid(idx, radius, scale)
        elif tp == "dl":
            local_idx_list, neighbor_idxs = partitioner.get_distance_lag(idx, radius, lag)
        elif tp == "ds":
            local_idx_list, neighbor_idxs = partitioner.get_direction_sector(idx, radius, n_splits)
        else:
            assert False, "Unsupported SRI type!"

        local_values_list = []
        for idxs in local_idx_list:
            local_values_list.append(values[idxs])

        neighbor_values = values[neighbor_idxs]

        sris, weights = compute_kl_sri(local_values_list, neighbor_values, bins=2)
        weights = weights / np.sum(weights)

        total_sris[tp].append(np.sum(sris * weights))
        total_weights[tp].append(neighbor_values.shape[0] / N)

    print("Processed {}\r".format(idx), end="")

np.savez("results/sri_thres_{}_radius_{}_scale_{}_lag_{}_split_{}".format(thres, radius, scale, lag, n_splits),
         sg_sri=total_sris["sg"], sg_wts=total_weights["sg"],
         dl_sri=total_sris["dl"], dl_wts=total_weights["dl"],
         ds_sri=total_sris["ds"], ds_wts=total_weights["ds"])
