from gbsio import read_from_csv
from partition import SSIPartitioner
from gbs import compute_unmarked_ssi, compute_marked_ssi
from gbsloss import NeighborhoodLoss

## Read coordinates and values from CSV file.
coords, values = read_from_csv("data/example_data.csv", value_column="hit@1")

## Construct a partitioner that extract neighborhood points.
partitioner = SSIPartitioner(coords, values, k=100)
nloss = NeighborhoodLoss(partitioner)

radius = 0.1

nloss.initialize_lookup(radius)

## The index of the center point to evaluate.
for idx in range(coords.shape[0]):
    center = coords[idx]

    nloss(idx, )