import numpy as np
from scipy import stats
from scipy.special import kl_div

def construct_edge_weight(w, d):
    map = np.zeros((d*d,d*d))

    for a in range(d*d):
        i, j = a // d, a % d
        for b in range(d*d):
            p, q = b // d, b % d
            try:
                map[a,b] = w[a][b]
            except:
                pass

    return map

def count_multicat_square_switches(Z, cs):
    d = Z.shape[0]
    c = len(cs)
    qdict = dict(zip([(a,b) for a in range(c) for b in range(c)], [0 for _ in range(c*c)]))
    v2idx = dict(zip(cs, range(c)))
    for i in range(d):
        for j in range(d):
            for m in [(0,1), (0,-1), (1,0), (-1,0)]:
                p, q = i + m[0], j + m[1]
                if p >= 0 and q >= 0 and p < d and q < d:
                    qdict[v2idx[Z[i,j]], v2idx[Z[p,q]]] += 1

    return qdict

def evaluate_goodness_of_fit(scovs, mean, std):
    standard_scovs = (scovs - mean) / std
    return stats.kstest(standard_scovs, "norm", alternative="two-sided")

def evaluate_standard_error(N, Mu_analytical, Sigma_analytical, Mu_gt, Sigma_gt):
    Sigma_sem = Sigma_gt / np.sqrt(N)
    Sigma_sev = Sigma_gt**2 * np.sqrt(2 / (N - 1))

    Mu_diff = np.abs(Mu_analytical - Mu_gt) / Sigma_sem
    Var_diff = np.abs(Sigma_analytical**2 - Sigma_gt**2) / Sigma_sev

    return Sigma_sem, Sigma_sev, Mu_diff, Var_diff

def evaluate_kl_divergence(xs, ys):
    return kl_div(xs, ys)
