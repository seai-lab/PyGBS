import numpy as np
import torch
import torch.nn as nn

class PerformanceTransformer(nn.Module):
    def __init__(self, bins):
        super().__init__()
        self.bins = bins
        self.bin_widths = np.zeros_like(bins)
        self.bin_widths[:-1] = bins[1:] - bins[:-1]
        self.bin_widths[-1] = np.inf

class LogOddsPerformanceTransformer(PerformanceTransformer):
    def __init__(self, bins):
        super().__init__(bins)

    def get_scores(self, Xs, use_gradients=False):
        if use_gradients:
            return torch.log(Xs) - torch.log(1 - Xs)
        else:
            return np.log(Xs) - np.log(1 - Xs)

    def discretize(self, scores, use_gradients=False):
        if use_gradients:
            scores_numpy = scores.detach().cpu().numpy()
            scores_numpy[scores_numpy < self.bins[0]] = self.bins[0]
            bin_deltas = scores_numpy.reshape((-1, 1)) * np.ones((scores_numpy.shape[0], self.bins.shape[0])) - self.bins.reshape(
                (1, -1))
            bin_width_deltas = self.bin_widths - bin_deltas
            scores_bins = torch.DoubleTensor(self.bins[np.where((bin_deltas >= 0) & (bin_width_deltas > 0))[1]])
            return scores - (scores.detach() - scores_bins.detach())
        else:
            scores[scores < self.bins[0]] = self.bins[0]
            bin_deltas = scores.reshape((-1, 1)) * np.ones((scores.shape[0], self.bins.shape[0])) - self.bins.reshape((1, -1))
            bin_width_deltas = self.bin_widths - bin_deltas
            return self.bins[np.where((bin_deltas >= 0) & (bin_width_deltas > 0))[1]]