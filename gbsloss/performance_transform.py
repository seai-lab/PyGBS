import numpy as np
import torch
import torch.nn as nn

class PerformanceTransformer(nn.Module):
    def __init__(self, bins):
        self.bins = bins
        self.bin_widths = np.zeros_like(bins)
        self.bin_widths[:-1] = bins[1:] - bins[:-1]
        self.bin_widths[-1] = np.inf

class LogOddsPerformanceTransformer(PerformanceTransformer):
    def __init__(self, bins):
        super().__init__(bins)

    def get_log_odds(self, Xs, ys, use_gradient=False):
        if use_gradient:
            return torch.log(Xs[ys]) - torch.log(1 - Xs[ys])
        else:
            return np.log(Xs[ys]) - np.lg(1 - Xs[ys])

    def discretize(self, logodds, use_gradient=False):
        if use_gradient:
            logodds_numpy = logodds.detach().cpu().numpy()
            logodds_numpy[logodds_numpy < self.bins[0]] = self.bins[0]
            bin_deltas = logodds_numpy.reshape((-1, 1)) * np.ones((logodds_numpy.shape[0], self.bins.shape[0])) - self.bins.reshape(
                (1, -1))
            bin_width_deltas = self.bin_widths - bin_deltas
            logodds_bins = torch.DoubleTensor(self.bins[np.where((bin_deltas >= 0) & (bin_width_deltas > 0))[1]])
            return logodds - (logodds.detach() - logodds_bins.detach())
        else:
            logodds[logodds < self.bins[0]] = self.bins[0]
            bin_deltas = logodds.reshape((-1, 1)) * np.ones((logodds.shape[0], self.bins.shape[0])) - self.bins.reshape((1, -1))
            bin_width_deltas = self.bin_widths - bin_deltas
            return self.bins[np.where((bin_deltas >= 0) & (bin_width_deltas > 0))[1]]