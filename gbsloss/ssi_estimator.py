import numpy as np
import torch
from gbs.ssi.surprisal import AnalyticalSurprisal

class SSI_distribution_estimator:
    def __init__(self):
        self.analytical_surprisal = AnalyticalSurprisal()

    def estimate(self, scores, weight_matrix):
        cs, ns = np.unique(scores.detach().cpu().numpy(), return_counts=True)
        rmax = np.argmax(ns)

        ignores = np.ones_like(cs)
        ignores[rmax] = 0

        self.analytical_surprisal.fit(cs, ns, weight_matrix.detach().cpu().numpy(), ignores)
        mean, std, _ = self.analytical_surprisal.get_fitted_params()

        return torch.tensor(mean).float(), torch.tensor(std).float()