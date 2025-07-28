import numpy as np
import scipy

def compute_kl_sri(local_values_list, global_values, bins=10, threshold=20, smooth=1e-8):
    kl_divs, weights = [], []

    N = global_values.shape[0]
    if N <= 0:
        return 0., 0.

    freqs_global, _ = np.histogram(global_values, bins=bins)
    probs_global = freqs_global.astype(np.float64) / N

    for local_values in local_values_list:
        n = local_values.shape[0]
        if n < threshold:
            continue

        freqs_local, _ = np.histogram(local_values, bins=bins)
        probs_local = freqs_local.astype(np.float64) / n + smooth

        kl_div = np.sum(scipy.special.rel_entr(probs_global, probs_local))

        kl_divs.append(kl_div)
        weights.append(n / N)

    return np.array(kl_divs), np.array(weights)