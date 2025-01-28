import numpy as np

from sklearn.neighbors import KDTree, BallTree

from scipy import sparse

from utils import _latlon_to_xyz, _xyz_to_latlon, _get_polar_concentric_grid_points_by_density, _get_polar_concentric_grid_points_by_number, _center_grid_points

def auto_density(radius, n_neighbors):
    return int(np.ceil((5 * n_neighbors) / (np.pi * radius**2)))

def construct_weight_matrix(points, k):
    """
    :param points: spherical coordinates of points, in radians.
    :param k: k nearest neighbors to build the weight matrix.
    :return: 0-1 weight matrix.
    """
    kdt = BallTree(points, leaf_size=30, metric='haversine')
    nbrs = kdt.query(points, k=k+1, return_distance=False)

    weights, coord_is, coord_js = [], [], []

    for i, ni in enumerate(nbrs):
        for j in ni:
            if i != j:
                weights.append(1)
                coord_is.append(i)
                coord_js.append(j)

    return sparse.coo_matrix((weights, (coord_is, coord_js)), shape=(points.shape[0], points.shape[0])).toarray()

def generate_background_points(center, radius, density=None, n_points=None):
    """
    :param center: the (latitude, longitude) of the center point, in radians.
    :param radius: the range of circular grid, in radians. It needs to be in the range of (0, pi/2).
    :param density: the density of grid points, in terms of point counts in unit square radians.
    :return: spherical coordinates of circular grid points with given radius and density around the given center, in radians.
    """

    if density is not None:
        polar_concentric_grid_points = _get_polar_concentric_grid_points_by_density(radius, density)
    elif n_points is not None:
        polar_concentric_grid_points = _get_polar_concentric_grid_points_by_number(radius, n_points)
    else:
        assert False, "Either specify density or number of points!"

    xyzs = _latlon_to_xyz(polar_concentric_grid_points)

    rotated_xyzs = _center_grid_points(xyzs, center)

    return _xyz_to_latlon(rotated_xyzs)
