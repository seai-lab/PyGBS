import numpy as np
from scipy.spatial.transform import Rotation as R

def _latlon_to_xyz(points):
    """
    :param points: spherical coordinates of points, in radians.
    :return: 3D Cartesian coordinates of points.
    """
    xyzs = []

    for phi, theta in points:
        r = np.cos(phi)
        x = np.cos(theta) * r
        y = np.sign(phi) * np.sin(theta) * r
        z = np.sin(phi)

        xyzs.append((x, y, z))

    return np.array(xyzs)

def _xyz_to_latlon(xyzs):
    lat = np.arcsin(xyzs[:, 2])
    lon = np.arctan2(xyzs[:, 1], xyzs[:, 0])

    return np.array([lat, lon]).T

def _get_euler_angles(center):
    return 0.5 * np.pi - center[0], center[1]

def _get_polar_concentric_grid_points_by_density(radius, density):
    """
    :param radius: the range of circular grid, in radians.
    :param density: the density of grid points, in terms of point counts in unit square radian.
    :return: spherical coordinates of circular grid points with given radius and density around the northern pole.
    """
    dist = 1 / np.sqrt(density)
    n_lag = int(np.ceil(radius / dist))

    points = []
    for i in range(1, n_lag + 1):
        n_points = int(np.ceil(2 * np.pi * i))
        delta_angle = 2 * np.pi / n_points
        for j in range(n_points):
            points.append([np.pi / 2 - dist * i, -np.pi + delta_angle * j])

    return np.array(points)

def _get_polar_concentric_grid_points_by_number(radius, n_points):
    """
    :param radius: the range of circular grid, in radians.
    :param density: the density of grid points, in terms of point counts in unit square radian.
    :return: spherical coordinates of circular grid points with given radius and density around the northern pole.
    """
    dist = np.sqrt(np.pi / n_points) * radius
    n_lag = int(np.ceil(radius / dist))

    points = []
    for i in range(1, n_lag + 1):
        n_points = int(np.ceil(2 * np.pi * i))
        delta_angle = 2 * np.pi / n_points
        for j in range(n_points):
            points.append([np.pi / 2 - dist * i, -np.pi + delta_angle * j])

    return np.array(points)

def _center_grid_points(xyzs, center):
    """
    :param xyzs: 3D Cartesian coordinates of circular grid points around the northern pole.
    :param center: the (latitude, longitude) of the center point, in radians.
    :return: the 3D Cartesian coordinates of circular grid points around the given center.
    """
    rotate_phi, rotate_theta = _get_euler_angles(center)
    r = R.from_euler('yz', [rotate_phi, rotate_theta], degrees=False)

    return r.apply(xyzs)

def _move_points_to_polar(xyzs, center):
    """
    :param xyzs: 3D Cartesian coordinates of points.
    :param center: the (latitude, longitude) of the center point, in radians.
    :return: the 3D Cartesian coordinates of points centered at the northern pole.
    """
    rotate_phi, rotate_theta = _get_euler_angles(center)
    r = R.from_euler('zy', [-rotate_theta, -rotate_phi], degrees=False)

    return r.apply(xyzs)

def _get_arc_angles(points, center):
    """
    :param points: spherical coordinates of points, in radians.
    :param center: the (latitude, longitude) of the center point, in radians.
    :return: the absolute arc angles between each point and the south-pointing arc passing the center.
    """
    xyzs = _latlon_to_xyz(points)
    polar_xyzs = _move_points_to_polar(xyzs, center)
    polar_points = _xyz_to_latlon(polar_xyzs)

    raw_angles = polar_points[:, 1] - center[1]
    idx = np.where(np.abs(raw_angles) > np.pi)

    if (raw_angles[idx] > 0).all():
        raw_angles[idx] = -(2 * np.pi - raw_angles[idx])
    elif (raw_angles[idx] < 0).all():
        raw_angles[idx] = 2 * np.pi + raw_angles[idx]
    else:
        assert False, "Incorrect angle sign!"

    return raw_angles