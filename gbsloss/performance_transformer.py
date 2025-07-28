import numpy as np
import torch
import torch.nn as nn

class PerformanceTransformer(nn.Module):
    def __init__(self, bins):
        super().__init__()
        self.bins = bins
        self.bin_widths = torch.zeros(bins.shape)
        self.bin_widths[:-1] = self.bins[1:] - self.bins[:-1]
        self.bin_widths[-1] = torch.inf

class LogOddsPerformanceTransformer(PerformanceTransformer):
    def __init__(self, bins):
        super().__init__(bins)

    def get_scores(self, Xs):
        return torch.log(Xs) - torch.log(1 - Xs)

    def discretize(self, scores):
        scores_bar = torch.clone(scores)
        scores_bar[scores_bar < self.bins[0]] = self.bins[0]

        bin_deltas = scores_bar.reshape((-1, 1)) * torch.ones((scores_bar.shape[0], self.bins.shape[0])) - self.bins.reshape((1, -1))
        bin_width_deltas = self.bin_widths - bin_deltas
        scores_bins = self.bins[np.where((bin_deltas >= 0) & (bin_width_deltas > 0))[1]]

        return scores - (scores.detach() - scores_bins.detach())