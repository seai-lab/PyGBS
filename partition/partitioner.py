from sklearn.neighbors import KDTree, BallTree

class Partitioner:
    def __init__(self, coords, values):
        self.coords = coords
        self.values = values

class NeighborhoodPartitioner(Partitioner):
    def __init__(self, coords, values, k):
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

class S2CellPartitioner(Partitioner):
    pass

class DistanceLagPartitioner(Partitioner):
    pass

class DirectionSectorPartitioner(Partitioner):
    pass