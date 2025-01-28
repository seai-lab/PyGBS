import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from partition import Partitioner, SSIPartitioner
from .performance_transformer import PerformanceTransformer, LogOddsPerformanceTransformer

from gbs.ssi.utils import construct_weight_matrix, generate_background_points

from gbs.ssi.surprisal import AnalyticalSurprisal

analytical_surprisal = AnalyticalSurprisal()

class GBSLoss(nn.Module):
    def __init__(self, partitioner:Partitioner, perf_transformer:PerformanceTransformer, radius:float):
        super().__init__()
        self.partitioner = partitioner
        self.perf_transformer = perf_transformer
        self.N = self.partitioner.N
        self.radius = radius
        self.stored_scores = np.zeros(self.N)

class MarkedSSILoss(GBSLoss):
    def __init__(self, partitioner: SSIPartitioner, perf_transformer:PerformanceTransformer, radius:float, n_cls:int, n_neighbor_points:int):
        super().__init__(partitioner, perf_transformer, radius)
        self.n_neighbor_points = n_neighbor_points
        self.neighborhood_points_lookup = np.zeros((self.N, self.n_neighbor_points, 2))
        self.neighborhood_scores_lookup = torch.zeros((self.N, self.n_neighbor_points))

        self.mean_lookup = torch.zeros(self.N)
        self.std_lookup = torch.zeros(self.N)
        self.weight_matrix_lookup = torch.zeros((self.N, n_neighbor_points, n_neighbor_points))
        self.n_cls = n_cls
        self.background_value = 1 / self.n_cls

    def initialize_lookup(self):
        for idx in range(self.N):
            presence_points, presence_values = self.partitioner.get_neighborhood(idx, self.radius)
            n_background_points = self.n_neighbor_points - presence_points.shape[0]

            background_points = generate_background_points(self.partitioner.coords[idx], self.radius, n_points=self.n_neighbor_points)
            background_points = background_points[np.random.choice(self.n_neighbor_points, n_background_points)]

            neighborhood_points = np.concatenate((presence_points, background_points), axis=0)
            neighborhood_values = np.concatenate((presence_values, self.background_value * np.ones(n_background_points)))
            neighborhood_scores = self.perf_transformer.discretize(self.perf_transformer.get_scores(neighborhood_values, use_gradients=False), use_gradients=False)

            weight_matrix = construct_weight_matrix(neighborhood_points, 4)

            cs, ns = np.unique(neighborhood_scores, return_counts=True)
            rmax = np.argmax(ns)

            ignores = np.ones_like(cs)
            ignores[rmax] = 0

            analytical_surprisal.fit(cs, ns, weight_matrix, ignores)
            mean, std, _ = analytical_surprisal.get_fitted_params()

            self.neighborhood_points_lookup[idx] = neighborhood_points
            self.neighborhood_scores_lookup[idx] = neighborhood_scores
            self.weight_matrix_lookup[idx] = weight_matrix
            self.mean_lookup[idx] = mean
            self.std_lookup[idx] = std

    def forward(self, idx, batch):
        X = batch - torch.mean(batch, dim=1, keepdim=True)
        weights = torch.FloatTensor(self.weight_matrix_lookup[idx])
        Y = torch.einsum('bij,bj->bi', weights, X)
        print("X, weights, Y: ", X.requires_grad, weights.requires_grad, Y.requires_grad)

        moran_i_uppers = torch.sum(X * Y, dim=1, keepdim=True)
        print("moran_i_uppers: ", moran_i_uppers.requires_grad)
        locs, scales = torch.FloatTensor(self.mean_lookup[idx]), torch.tensor(self.std_lookup[idx])
        print("locs, scales: ", locs.requires_grad, scales.requires_grad)

        left_tails, _ = torch.min(torch.cat((moran_i_uppers, 2 * locs - moran_i_uppers), dim=1), dim=1)
        print("left_tails: ", left_tails.requires_grad)

        clamped_left_tails = torch.clamp((left_tails - locs) / scales, min=-10, max=0.)

        cdf_values = 2 * torch.distributions.normal.Normal(loc=0, scale=1).cdf(clamped_left_tails)

        print("cdf values: ", cdf_values.requires_grad)

        return -torch.log(cdf_values + 1e-256)




