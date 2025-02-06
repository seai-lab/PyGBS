import numpy as np

from .utils import auto_density, construct_weight_matrix, generate_background_points
from .surprisal import AnalyticalSurprisal

analytical_surprisal = AnalyticalSurprisal()

def compute_unmarked_ssi(presence_points, center, radius, density="auto", k=4):
    """
    :param presence_points: the lat/lon of the observed points (presence).
    :param center: the center of the evaluated neighborhood.
    :param radius: the radius of the evaluated neighborhood.
    :param density: the density of unobserved grid points (non-presence), in terms of point counts in each unit square radian. Default is "auto".
    :param k: the k nearest neighbors used to construct a weight matrix. Default is k=4, i.e. the rook's distance.
    :return: the presence v.s. non-presence spatial self-information score of this evaluated neighborhood.
    """
    if density == "auto":
        density = auto_density(radius, presence_points.shape[0])
        # print("Density: ", density)

    presence_values = np.ones(presence_points.shape[0])
    nonpresence_points = generate_background_points(center, radius, density)
    nonpresence_values = np.zeros(nonpresence_points.shape[0])

    points, values = np.concatenate((presence_points, nonpresence_points), axis=0), np.concatenate((presence_values, nonpresence_values))

    cs, ns = np.unique(values, return_counts=True)
    # bg_rate = np.max(ns) / np.sum(ns)
    rmax = np.argmax(ns)

    ignores = np.ones_like(cs)
    ignores[rmax] = 0

    weight_matrix = construct_weight_matrix(points, k)

    analytical_surprisal.fit(cs, ns, weight_matrix, ignores)
    prob = analytical_surprisal.get_probability(values, weight_matrix)

    return -np.log(prob[0] + 1e-256)

def compute_marked_ssi(presence_points, presence_values, center, radius, density="auto", k=4):
    """
    :param presence_points: the lat/lon of the observed points (presence).
    :param presence_values: the model performance of the observed points (presence). By default, presence values can not be zeros (reserved for non-presence values).
    :param center: the center of the evaluated neighborhood.
    :param radius: the radius of the evaluated neighborhood.
    :param density: the density of unobserved grid points (non-presence), in terms of point counts in each unit square radian. Default is "auto".
    :param k: the k nearest neighbors used to construct a weight matrix. Default is k=4, i.e. the rook's distance.
    :return: the presence v.s. non-presence spatial self-information score of this evaluated neighborhood.
    """
    if density == "auto":
        density = auto_density(radius, presence_points.shape[0])
        # print("Density: ", density)

    nonpresence_points = generate_background_points(center, radius, density)
    nonpresence_values = np.zeros(nonpresence_points.shape[0])

    points, values = np.concatenate((presence_points, nonpresence_points), axis=0), np.concatenate((presence_values, nonpresence_values))

    cs, ns = np.unique(values, return_counts=True)
    # bg_rate = np.max(ns) / np.sum(ns)
    rmax = np.argmax(ns)

    ignores = np.ones_like(cs)
    ignores[rmax] = 0

    weight_matrix = construct_weight_matrix(points, k)

    analytical_surprisal.fit(cs, ns, weight_matrix, ignores)
    prob = analytical_surprisal.get_probability(values, weight_matrix)

    return -np.log(prob[0] + 1e-256)
