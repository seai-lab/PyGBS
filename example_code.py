from gbsio import read_from_csv
from partition import SSIPartitioner
from gbs import compute_unmarked_ssi, compute_marked_ssi

radius = 0.01
coords, values = read_from_csv("data/example_data.csv", value_column="hit@1")

partitioner = SSIPartitioner(coords, k=400)

for idx in range(partitioner.N):
    center = coords[idx]

    presence_idxs = partitioner.get_neighborhood(idx, radius)
    presence_points, presence_values = coords[presence_idxs], values[presence_idxs]

    unmarked_ssi = compute_unmarked_ssi(presence_points, center, radius, density="auto", k=4)
    marked_ssi = compute_marked_ssi(presence_points, presence_values, center, radius, density="auto", k=4)

    print("Data point: {}, Unmarked SSI: {}, Marked SSI: {}\r".format(idx, unmarked_ssi, marked_ssi), end="")