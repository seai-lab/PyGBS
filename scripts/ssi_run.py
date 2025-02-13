#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

# Import functions from your modules.
# Make sure these modules (gbsio, partition, gbs) are in your PYTHONPATH.
from gbsio import read_from_csv
from partition import SSIPartitioner
from gbs import compute_unmarked_ssi, compute_marked_ssi
from gbs.ssi.utils import generate_background_points, auto_density

def main():
    parser = argparse.ArgumentParser(
        description="Compute marked and/or unmarked SSI metrics and save the results."
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help="Path to the CSV file containing the data."
    )
    parser.add_argument(
        '--value-column',
        type=str,
        default="haversine_distance_km",
        help="Name of the column containing performance values (default: haversine_distance_km)."
    )
    parser.add_argument(
        '--coords-columns',
        type=str,
        nargs=2,
        default=["gt_latitude", "gt_longitude"],
        help="Two column names for coordinates (default: gt_latitude gt_longitude)."
    )
    parser.add_argument(
        '--scale',
        type=str,
        default="country",
        choices=['street', 'region', 'country', 'continent'],
        help="Scale level (default: country)."
    )
    parser.add_argument(
        '--radius',
        type=float,
        default=0.01,
        help="Radius of the neighborhood (default: 0.01)."
    )
    parser.add_argument(
        '--k',
        type=int,
        default=400,
        help="Number of neighbors for partitioning (default: 400)."
    )
    parser.add_argument(
        '--marked',
        type=str,
        default="both",
        choices=['marked', 'unmarked', 'both'],
        help="Which SSI to compute: 'marked', 'unmarked', or 'both' (default: both)."
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Optional output file name. If not provided, a name is generated."
    )
    args = parser.parse_args()

    # Define benchmark levels for threshold calculation.
    benchmark_levels = {
        'street': 1,
        'region': 200,
        'country': 750,
        'continent': 2500
    }
    thres = benchmark_levels.get(args.scale)

    # Read data from CSV file.
    # The read_from_csv function is expected to return (coords, values, raw_values)
    coords_part, values_part, raw_values_part = read_from_csv(
        args.input,
        value_column=args.value_column,
        coords_columns=args.coords_columns,
        thres=thres,
        return_raw_performance=True
    )

    # Wrap in lists for possible future concatenation.
    coords_list = [coords_part]
    values_list = [values_part]
    raw_values_list = [raw_values_part]

    coords = np.concatenate(coords_list, axis=0)
    values = np.concatenate(values_list)
    raw_values = np.concatenate(raw_values_list)

    print("Data shapes:")
    print(" coords:", coords.shape)
    print(" values:", values.shape)
    print(" raw_values:", raw_values.shape)

    # Create a partitioner to extract neighborhood points.
    partitioner = SSIPartitioner(coords, k=args.k)

    # Prepare lists for results.
    locs = []
    accs = []
    marked_ssis = []
    unmarked_ssis = []

    # Determine which SSI metrics to compute.
    if args.marked == 'marked':
        compute_marked = True
        compute_unmarked = False
    elif args.marked == 'unmarked':
        compute_marked = False
        compute_unmarked = True
    elif args.marked == 'both':
        compute_marked = True
        compute_unmarked = True

    total = len(coords)
    for idx in range(partitioner.N):
        center = coords[idx]
        # Extract neighborhood points.
        presence_idxs = partitioner.get_neighborhood(idx, args.radius)
        presence_points = coords[presence_idxs]
        presence_values = values[presence_idxs]

        # Compute average performance in the neighborhood.
        avg_acc = np.mean(values[presence_idxs])

        # Compute marked SSI if requested.
        if compute_marked:
            ssi = compute_marked_ssi(
                presence_points, presence_values, center, args.radius, density="auto", k=4
            )
            marked_ssis.append(ssi)
        # Compute unmarked SSI if requested.
        if compute_unmarked:
            ssi = compute_unmarked_ssi(
                presence_points, center, args.radius, density="auto", k=4
            )
            unmarked_ssis.append(ssi)

        locs.append(center)
        accs.append(avg_acc)
        print("Processed: {}/{}\r".format(idx, total), end="")

    print()  # Newline after progress printing

    # Build output file name if not provided.
    if args.output:
        output_file = args.output
    else:
        output_file = "../results/{}_scale_{}_radius_{}_ssi.npz".format(
            args.scale, args.radius, args.marked
        )

    # Save the results.
    np.savez(
        output_file,
        locs=locs,
        accs=accs,
        marked_ssis=marked_ssis,
        unmarked_ssis=unmarked_ssis
    )
    print("Results saved to", output_file)

if __name__ == '__main__':
    main()
