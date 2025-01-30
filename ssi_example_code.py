from gbsio import read_from_csv
from partition import SSIPartitioner
from gbs import compute_unmarked_ssi, compute_marked_ssi

## Read coordinates and values from CSV file.
coords, values = read_from_csv("data/example_data.csv", value_column="hit@1")

## Construct a partitioner that extract neighborhood points.
partitioner = SSIPartitioner(coords, k=100)

## The index of the center point to evaluate.
idx = 0
center = coords[idx]

## The radius of the neighborhood.
radius = 0.1

## Extract neighbood points.
presence_idxs = partitioner.get_neighborhood(idx, radius)
presence_points, presence_values = coords[presence_idxs], values[presence_idxs]

## Use automatic density estimation. Users can manually specify this hyperparameter.

## pnp_ssi is the former base geo-bias
pnp_ssi = compute_unmarked_ssi(presence_points, center, radius, density="auto", k=4)
## rp_ssi is the former relative geo-bias
rp_ssi = compute_marked_ssi(presence_points, presence_values, center, radius, density="auto", k=4)

print("Unmarked SSI Score: ", pnp_ssi, "Marked SSI Score: ", rp_ssi)