import numpy as np

from sklearn.neighbors import KDTree, BallTree
from sklearn.metrics.pairwise import haversine_distances

class Partitioner:
    def __init__(self, coords, values):
        self.coords = coords
        self.values = values

class NeighborhoodPartitioner(Partitioner):
    def __init__(self, coords, values, k=100):
        super().__init__(coords, values)
        self.k = k
        self.kdt, self.dists, self.nbrs = self._construct_tree()

    def _construct_tree(self):
        kdt = BallTree(self.coords, leaf_size=30, metric='haversine')
        dists, nbrs = kdt.query(self.coords, k=self.k, return_distance=True)

        return kdt, dists, nbrs

    def get_neighborhood(self, idx, radius, min_dist=0.0):
        mask = self.nbrs[idx, (self.dists[idx] >= min_dist) & (self.dists[idx] <= radius)]
        return self.coords[mask], self.values[mask]

class ScalePartitioner(Partitioner):
    def __init__(self, coords, values):
        super().__init__(coords, values)
        self.dists = haversine_distances(coords)

    def get_scale_grid(self, idx, radius):
        pass

class DistanceLagPartitioner(Partitioner):
    def __init__(self, coords, values):
        super().__init__(coords, values)
        self.dists = haversine_distances(coords)

    def get_scale_grid(self, idx, radius, lag):
        local_values_list = []

        n_lags = int(np.ceil(radius / lag))
        for i in range(n_lags):
            values = self.values[(self.dists[idx] >= lag * i) & (self.dists[idx] < lag * (i + 1))]
            local_values_list.append(values)

        return local_values_list, self.values

class DirectionSectorPartitioner(Partitioner):
    pass