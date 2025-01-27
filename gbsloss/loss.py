import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from partition import SSIPartitioner, SRIPartitioner

class GBSLoss(nn.Module):
    def __init__(self, partitioner):
        self.partitioner = partitioner
        self.N = self.partitioner.N
        self.partition_lookup = dict(zip([i for i in range(self.N)], [[] for _ in range(self.N)]))
        self.stored_scores = np.zeros(self.N)

class MarkedSSILoss(GBSLoss):
    def __init__(self, partitioner: SSIPartitioner):
        super().__init__(partitioner)

    def initialize_lookup(self, radius):
        for idx in range(self.N):
            neighborhood_idxs = self.partitioner.get_neighborhood(idx, radius, return_idx=True)
            self.partition_lookup[idx] = neighborhood_idxs

    def store_scores(self, idxs, latents, labels):
        scores = F.softmax(latents, axis=1)[labels].detach().cpu().numpy()
        self.stored_scores[idxs] = scores

    def forward(self, idxs, latents, labels, weighted=False):
        scores = F.softmax(latents, axis=1)[labels]

        losses = []
        total_n = 0
        for idx, score in zip(idxs, scores):
            neighborhood_idxs = self.partition_lookup[idx]
            neighborhood_scores = self.previous_scores[neighborhood_idxs]
            neighborhood_score = np.mean(neighborhood_scores)

            loss = neighborhood_score * torch.log(neighborhood_score / score) + (1 - neighborhood_score) * torch.log((1 - neighborhood_score) / (1 - score))
            if weighted:
                losses.append(len(neighborhood_idxs) * loss)
                total_n += len(neighborhood_idxs)
            else:
                losses.append(loss)
                total_n += 1

        return torch.sum(losses) / total_n





