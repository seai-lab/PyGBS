import numpy as np

from sklearn.neighbors import KDTree, BallTree
from sklearn.metrics.pairwise import haversine_distances

from utils.geometry import _get_arc_angles

class Partitioner:
    def __init__(self, coords):
        self.coords = coords
        self.N = self.coords.shape[0]

    def get_coords_by_idx(self, idx):
        return self.coords[idx]

class SSIPartitioner(Partitioner):
    def __init__(self, coords, k=100):
        super().__init__(coords)
        self.k = k
        self.kdt, self.dists, self.nbrs = self._construct_tree()

    def _construct_tree(self):
        kdt = BallTree(self.coords, leaf_size=30, metric='haversine')
        dists, nbrs = kdt.query(self.coords, k=self.k, return_distance=True)

        return kdt, dists, nbrs

    def get_neighborhood(self, idx, radius, min_dist=0.0):
        return self.nbrs[idx, (self.dists[idx] >= min_dist) & (self.dists[idx] <= radius)]

class SRIPartitioner(Partitioner):
    def __init__(self, coords):
        super().__init__(coords)
        self.dists = haversine_distances(coords)

    def get_scale_grid(self, idx, radius, scale, threshold=20):
        partition_idx_list = []
        neighbor_indices = np.where(self.dists[idx] <= radius)[0]
        neighbor_coords = self.coords[neighbor_indices]

        k = int(np.ceil(radius / scale))
        lat, lon = self.coords[idx]

        for i in range(-k, k, 1):
            for j in range(-k, k, 1):
                mask = np.where((neighbor_coords[:, 0] >= lat + i*scale) & (neighbor_coords[:, 0] < lat + (i+1)*scale) & (neighbor_coords[:, 1] >= lon + j*scale) & (neighbor_coords[:, 1] < lon + (j+1)*scale))

                if len(mask[0]) < threshold:
                    continue

                presence_idxs = neighbor_indices[mask[0]]
                partition_idx_list.append(presence_idxs)

        return partition_idx_list, neighbor_indices

    def get_distance_lag(self, idx, radius, lag, threshold=20):
        partition_idx_list = []
        neighbor_indices = np.where(self.dists[idx] <= radius)[0]
        neighbor_dists = self.dists[idx, neighbor_indices]

        n_lags = int(np.ceil(radius / lag))
        for i in range(n_lags):
            mask = np.where((neighbor_dists >= lag * i) & (neighbor_dists < lag * (i + 1)))

            if len(mask[0]) < threshold:
                continue
            presence_idxs = neighbor_indices[mask[0]]
            partition_idx_list.append(presence_idxs)

        return partition_idx_list, neighbor_indices

    def get_direction_sector(self, idx, radius, n_splits, threshold=20):
        partition_idx_list = []
        neighbor_indices = np.where(self.dists[idx] <= radius)[0]

        arc_angles = _get_arc_angles(self.coords[neighbor_indices], self.coords[idx])

        split_angle = 2 * np.pi / n_splits
        for i in range(n_splits):
            mask = np.where((arc_angles >= -np.pi + i * split_angle) & (arc_angles < -np.pi + (i + 1) * split_angle))
            if len(mask[0]) < threshold:
                continue
            presence_idxs = neighbor_indices[mask[0]]
            partition_idx_list.append(presence_idxs)

        return partition_idx_list, neighbor_indices

