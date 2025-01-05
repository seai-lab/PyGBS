import numpy as np

def compute_kl_sri(local_values, global_values):
    no_prior_freqs_local, _ = np.histogram(no_prior_perfs_local, bins=bins)
    no_prior_probs_local = no_prior_freqs_local.astype(np.float64) / n
    no_prior_scores = scipy.special.rel_entr(no_prior_probs_local, no_prior_probs_global)

    spherec_freqs_local, _ = np.histogram(spherec_perfs_local, bins=bins)
    spherec_probs_local = spherec_freqs_local.astype(np.float64) / n
    spherec_scores = scipy.special.rel_entr(spherec_probs_local, spherec_probs_global)


    grid_weights.append(n)
    no_prior_raw_scores.append(np.sum(no_prior_scores))
    spherec_raw_scores.append(np.sum(spherec_scores))

    xs.append(lon + scale / 2)
    ys.append(lat + scale / 2)