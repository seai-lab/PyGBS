import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from partition import Partitioner, SSIPartitioner
from gbs.ssi.utils import construct_weight_matrix, generate_background_points

from .performance_transformer import PerformanceTransformer
from .ssi_estimator import SSI_distribution_estimator

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

        self.n_cls = n_cls
        self.background_prob = torch.ones(1) * (1 / self.n_cls)
        self.scores_lookup = torch.zeros(self.N + 1)
        self.scores_lookup[-1] = self.perf_transformer.discretize(self.perf_transformer.get_scores(self.background_prob))

        # self.neighborhood_points_lookup = np.zeros((self.N, self.n_neighbor_points))
        self.n_neighbor_points = n_neighbor_points
        self.neighborhood_scores_idx_lookup = -np.ones((self.N, self.n_neighbor_points))

        self.weight_matrix_lookup = torch.zeros((self.N, n_neighbor_points, n_neighbor_points))
        self.estimator = SSI_distribution_estimator()
        self.mean_lookup = torch.zeros(self.N)
        self.std_lookup = torch.zeros(self.N)

    def initialize_weight_matrix_lookup(self):
        for idx in range(self.N):
            presence_idxs = self.partitioner.get_neighborhood(idx, self.radius, return_idx=True)[0]
            presence_points = self.partitioner.coords[presence_idxs]

            n_background_points = self.n_neighbor_points - presence_idxs.shape[0]
            neighborhood_idxs = np.concatenate((presence_idxs, np.array([-1 for _ in range(n_background_points)])))
            self.neighborhood_scores_idx_lookup[idx] = neighborhood_idxs

            background_points = generate_background_points(self.partitioner.coords[idx], self.radius, n_points=self.n_neighbor_points)
            background_points = background_points[np.random.choice(self.n_neighbor_points, n_background_points)]

            neighborhood_points = np.concatenate((presence_points, background_points), axis=0)
            # neighborhood_values = np.concatenate((presence_values, self.background_value * np.ones(n_background_points)))
            # neighborhood_scores = self.perf_transformer.discretize(self.perf_transformer.get_scores(neighborhood_values, use_gradients=False), use_gradients=False)

            weight_matrix = construct_weight_matrix(neighborhood_points, 4)

            # mean, std, = self.estimator.estimate(neighborhood_scores, weight_matrix)

            # self.neighborhood_points_lookup[idx] = neighborhood_points
            # self.neighborhood_scores_lookup[idx] = neighborhood_scores
            self.weight_matrix_lookup[idx] = torch.from_numpy(weight_matrix).float()
            # self.mean_lookup[idx] = mean
            # self.std_lookup[idx] = std

    def compute_scores(self, idx, batch, labels):
        """
        :param idx: indices of the samples in the batch, (B,).
        :param batch: the output of the last layer before softmax, (B,C).
        :param labels: the labels of the true classes, (B,).
        :return: scores: the discretized model performance scores, (B,).
        """
        probs = F.softmax(batch, dim=1)[np.arange(batch.shape[0]), labels]
        scores = self.perf_transformer.get_scores(probs)
        scores = self.perf_transformer.discretize(scores)
        return scores

    def update_scores_lookup(self, idx, scores):
        """
        :param idx: indices of the samples in the batch, (B,).
        :param scores: the discretized model performance scores, (B,).
        :return:
        """
        self.scores_lookup[idx] = scores

    def update_distribution_lookup(self):
        """
        :param idx: indices of the samples in the batch, (B,).
        :return:
        """
        for idx in range(self.N):
            # presence_idxs = self.partitioner.get_neighborhood(idx, self.radius, return_idx=True)[0]
            # n_background_points = self.n_neighbor_points - presence_idxs.shape[0]
            # neighborhood_idxs = np.concatenate((presence_idxs, np.array([-1 for _ in range(n_background_points)])))

            # neighborhood_scores = self.scores_lookup[neighborhood_idxs]
            neighborhood_scores = self.scores_lookup[self.neighborhood_scores_idx_lookup[idx]]
            weight_matrix = self.weight_matrix_lookup[idx]

            mean, std, = self.estimator.estimate(neighborhood_scores, weight_matrix)

            self.mean_lookup[idx] = mean
            self.std_lookup[idx] = std

    def forward(self, idx, scores):
        """
        :param idx: indices of the samples in the batch, Bx1.
        :param scores: the discretized model performance scores, BxN.
        :return: SSI bias score.
        """
        neighborhood_idxs = self.neighborhood_scores_idx_lookup[idx]
        neighborhood_scores = self.scores_lookup[neighborhood_idxs]
        neighborhood_scores[:, 0] = scores

        print("Neighborhood scores: ", neighborhood_scores)

        X = neighborhood_scores - torch.mean(neighborhood_scores, dim=1, keepdim=True)
        weights = self.weight_matrix_lookup[idx]
        Y = torch.einsum('bij,bj->bi', weights, X)
        print("X, weights, Y: ", X.requires_grad, weights.requires_grad, Y.requires_grad)

        moran_i_uppers = torch.sum(X * Y, dim=1, keepdim=True)
        print("moran_i_uppers: ", moran_i_uppers.requires_grad)
        locs, scales = self.mean_lookup[idx], self.std_lookup[idx]
        print("locs, scales: ", locs, scales, locs.requires_grad, scales.requires_grad)

        left_tails, _ = torch.min(torch.cat((moran_i_uppers, 2 * locs - moran_i_uppers), dim=1), dim=1)
        print("left_tails: ", left_tails, left_tails.requires_grad)

        clamped_left_tails = torch.clamp((left_tails - locs) / scales, min=-5, max=0.)

        print("clamped_left_tails: ", clamped_left_tails, clamped_left_tails.requires_grad)

        cdf_values = 2 * torch.distributions.normal.Normal(loc=0, scale=1).cdf(clamped_left_tails)

        print("cdf values: ", cdf_values, cdf_values.requires_grad)

        return -torch.log(cdf_values + 1e-12)




