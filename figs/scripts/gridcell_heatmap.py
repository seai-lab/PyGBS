import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from matplotlib.colors import LinearSegmentedColormap

import os


def create_fishnet(cell_size=1.0):
    """
    Create a fishnet (grid of polygons) that covers the whole world in PlateCarree projection.

    :param cell_size: Size of each grid cell in degrees (both latitude and longitude).
    :return: GeoDataFrame containing the grid polygons.
    """
    # Define the bounds for the whole world
    xmin, xmax, ymin, ymax = -180, 180, -90, 90

    # Create coordinates for the fishnet
    longitudes = np.arange(xmin, xmax, cell_size)
    latitudes = np.arange(ymin, ymax, cell_size)

    # Generate polygons for the grid
    polygons = []
    for lon in longitudes:
        for lat in latitudes:
            polygons.append(
                Polygon([
                    (lon, lat),
                    (lon + cell_size, lat),
                    (lon + cell_size, lat + cell_size),
                    (lon, lat + cell_size),
                    (lon, lat)
                ])
            )

    # Create a GeoDataFrame
    fishnet = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:4326")
    return fishnet


def spatial_join_and_plot(input_file_path, cell_size=1.0, extent=[-180, 180, -90, 90], lon='lon1', lat='lat1', vmin = 0.0, vmax= 600, metric="hit@1", label_name='Accuracy', title=False):
    """
    Perform a spatial join to calculate the average value and pnp_ssi for points in each grid cell.

    :param input_file_path: Path to the CSV file containing 'lat', 'lon', and 'pnp_ssi' columns.
    :param cell_size: Size of each grid cell in degrees for the fishnet.
    """
    # Create the fishnet
    fishnet = create_fishnet(cell_size)

    # Load points from the CSV file
    data = pd.read_csv(input_file_path, low_memory=False).dropna(
        subset=['lon1', 'lat1'])
    gdf_points = gpd.GeoDataFrame(
        data, geometry=gpd.points_from_xy(data[lon], data[lat]), crs="EPSG:4326"
    )

    # Perform spatial join: Assign points to grid cells
    joined = gpd.sjoin(gdf_points, fishnet, how="inner")

    if metric in ["acc", "hit@1"]:
        averages = joined.groupby('index_right').agg(
            value=(metric, 'mean')
        ).reset_index()
    elif metric in ["pnp_ssi", "rp_ssi", "sri-lag", "sri-scale", "sri-dir"]:
        # averages = joined.groupby('index_right').apply(
        #     lambda x: np.average(x[metric], weights=x['weight'])
        # ).reset_index(name='value')
        averages = joined.groupby('index_right').agg(
            value=(metric, 'mean')
        ).reset_index()

    # Merge averages back to the fishnet
    fishnet = fishnet.reset_index()
    fishnet = fishnet.merge(averages, left_on='index', right_on='index_right')

    # Create figure and axis
    fig = plt.figure(figsize=(12, 6))

    # Create main map axis with specific position
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=ccrs.PlateCarree())
    # Set the extent to display the whole world
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Plot the fishnet with average values
    vmin = 0.0 # fishnet['value'].min()
    # vmax = fishnet['value'].max()

    cmap = plt.cm.get_cmap('Reds')
    cmap = LinearSegmentedColormap.from_list("adjusted_red", cmap(np.linspace(0.15, 1, 256)))

    fishnet['value'] = fishnet['value']/ (vmax - vmin)

    fishnet.plot(
        column='value',
        ax=ax,
        cmap=cmap,  # 'coolwarm',
        edgecolor=None,  # 'gray',
        linewidth=0.3,
        alpha=1.0,
        transform=ccrs.PlateCarree(),
        missing_kwds={"color": "lightgray"},
        vmin=0.0,
        vmax=1.0,
        legend=False  # Disable the default legend
    )

    ax.add_feature(cfeature.LAND, facecolor='lightgray')  # 设置大陆为浅灰色
    ax.set_axis_off()
    if title:
        ax.set_title(
            f"Fishnet with Average Value: {input_file_path}", fontsize=14)

    # bar
    cax = fig.add_axes([0.92, 0.2, 0.01, 0.6])
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
    sm.set_array([])

    cbar = plt.colorbar(sm, cax=cax)  
    cbar.set_label(label_name, fontsize=12, fontweight='bold',
                   rotation=-90, labelpad=15)
    cbar.set_ticks(np.linspace(0.0, 1.0, 5))

    # plt.show()


if __name__ == "__main__":
    input_folder = "GBS_score/ssi_radius_0.075"
    output_folder = "figs"

    cell_size = 2
    extent = [-180, 180, -80, 90]
    lon = 'lon1'
    lat = 'lat1'
    metric = "rp_ssi"  # "pnp_ssi", "rp_ssi","sri-lag", "sri-scale", "sri-dir"]

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):  # only CSV files
            input_file_path = os.path.join(input_folder, filename)
            dataset, model = filename.split("_", 1)
            model, _ = model.split("_", 1)
            if dataset != "eurosat":
                spatial_join_and_plot(input_file_path, cell_size=cell_size,
                                    extent=extent, lon=lon, lat=lat, metric=metric, label_name='Marked SSI', title=False)
            else:
                spatial_join_and_plot(input_file_path, cell_size=cell_size,
                                    extent=[-30, 50, 30, 70], lon=lon, lat=lat, metric=metric, label_name='Marked SSI', title=False)
            plt.savefig(os.path.join(
                output_folder, f"{filename[:-4]}_{metric}.jpg"), format='jpg', dpi=300, bbox_inches='tight')