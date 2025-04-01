import json

from libpysal.weights import lat2W
import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import norm
from scipy import sparse

from utils import construct_edge_weight, count_multicat_square_switches, evaluate_goodness_of_fit, evaluate_standard_error, evaluate_kl_divergence
from generator import GridGenerator
from surprisal import EmpiricalSurprisal, AnalyticalSurprisal
from plotter import plot_integrated_distribution, plot_probabilities

def compute_probability(Z, cs, ns, w_map, ignores):
    analytical_surprisal = AnalyticalSurprisal()
    analytical_surprisal.fit(cs, ns, w_map, ignores)

    # TO-DO: iterate over image

    moran_I = analytical_surprisal.get_probability(Z, w_map)
    probability = analytical_surprisal.get_probability(Z, w_map)

    return moran_I, probability

def run_an_experiment(d, w_map, cs, ns, ignores):
    M = cs.shape[0]
    Zs, cat_edge_dicts, scovs, probs = [], dict(zip([(a,b) for a in range(M) for b in range(M)], [[] for _ in range(M*M)])), [], []

    r = 1

    N, W = np.sum(ns), np.sum(w_map)
    xmean = np.sum(cs * ns) / np.sum(ns)
    xvar = np.sum((ns * (cs - xmean))**2)

    scaling_factor = N / (W * xvar)

    for i in range(10000):
        Z = GridGenerator.generate_multicat_permutation_configuration(d, cs, ns).reshape((d,d))
        X = Z.flatten().reshape((1,-1)) - np.mean(Z)
        Y = np.matmul(w_map, X.T)

        Zs.append(Z)
        scovs.append(np.matmul(X, Y).flatten())
        qdict = count_multicat_square_switches(Z, cs)

        for k in cat_edge_dicts.keys():
            cat_edge_dicts[k].append(r * qdict[k])

    Zs = np.array(Zs)
    scovs = np.array(scovs).flatten()

    mus, sigmas, ds, ijs = {}, {}, [], []
    for k in cat_edge_dicts.keys():
        (mu, sigma) = norm.fit(cat_edge_dicts[k])

        mus[k] = mu
        sigmas[k] = sigma
        ds.append((cs[k[0]] - xmean) * (cs[k[1]] - xmean))
        ijs.append(ignores[k[0]] * ignores[k[1]])

    Mu_analytical, Mu_dict, Mu_coef_dict = AnalyticalSurprisal.compute_mean(cs, ns)
    Sigma_analytical, Sigma_dict, Sigma_coef_dict = AnalyticalSurprisal.compute_std(cs, ns, ignores=ignores)

    (Mu_gt, Sigma_gt) = norm.fit(scovs)

    return {
        "Zs": Zs,
        "scovs": scovs,
        "scaling_factor": scaling_factor,
        "mu_coef_dict": Mu_coef_dict,
        "sigma_coef_dict": Sigma_coef_dict,
        "mu_analytical_dict": Mu_dict,
        "sigma_analytical_dict": Sigma_dict,
        "mu_empirical_dict": mus,
        "sigma_empirical_dict": sigmas,
        "Sigma_analytical": Sigma_analytical,
        "Mu_estimated": Mu_gt,
        "Sigma_estimated": Sigma_gt
    }

def run_a_perturbation_experiment(Zs, w_map, cs, ns, ignores):
    M = cs.shape[0]
    cat_edge_dicts, scovs, probs = dict(zip([(a,b) for a in range(M) for b in range(M)], [[] for _ in range(M*M)])), [], []

    r = 1

    N, W = np.sum(ns), np.sum(w_map)
    xmean = np.sum(cs * ns) / np.sum(ns)
    xvar = np.sum((ns * (cs - xmean))**2)

    scaling_factor = N / (W * xvar)

    for Z in Zs:
        X = Z.flatten().reshape((1,-1)) - np.mean(Z)
        Y = np.matmul(w_map, X.T)

        scovs.append(np.matmul(X, Y).flatten())
        qdict = count_multicat_square_switches(Z, cs)

        for k in cat_edge_dicts.keys():
            cat_edge_dicts[k].append(r * qdict[k])

    scovs = np.array(scovs).flatten()

    mus, sigmas, ds, ijs = {}, {}, [], []
    for k in cat_edge_dicts.keys():
        (mu, sigma) = norm.fit(cat_edge_dicts[k])

        mus[k] = mu
        sigmas[k] = sigma
        ds.append((cs[k[0]] - xmean) * (cs[k[1]] - xmean))
        ijs.append(ignores[k[0]] * ignores[k[1]])

    Mu_analytical, Mu_dict, Mu_coef_dict = AnalyticalSurprisal.compute_mean(cs, ns)
    Sigma_analytical, Sigma_dict, Sigma_coef_dict = AnalyticalSurprisal.compute_std(cs, ns, ignores=ignores)

    (Mu_gt, Sigma_gt) = norm.fit(scovs)

    return {
        "scovs": scovs,
        "scaling_factor": scaling_factor,
        "mu_coef_dict": Mu_coef_dict,
        "sigma_coef_dict": Sigma_coef_dict,
        "mu_analytical_dict": Mu_dict,
        "sigma_analytical_dict": Sigma_dict,
        "mu_empirical_dict": mus,
        "sigma_empirical_dict": sigmas,
        "Sigma_analytical": Sigma_analytical,
        "Mu_estimated": Mu_gt,
        "Sigma_estimated": Sigma_gt
    }

if __name__ == '__main__':

    dataset, k, thres_low, thres_high = "birdsnap_ebird_meta_test", 100, 50, 100

    wname = f"eval_processed/eval_{dataset}_{k}_{thres_low}_{thres_high}_weight_matrix.npz"
    weight = sparse.load_npz(wname).toarray()

    analytical_surprisal = AnalyticalSurprisal()

    for model in ["no_prior", "tile_ffn", "wrap", "wrap_ffn", "xyz", "rbf", "rff", "NeRF", "Space2Vec-grid",
                  "Space2Vec-theory", "Space2Vec-dfs",
                  "Sphere2Vec-sphereC", "Sphere2Vec-sphereC+", "Sphere2Vec-sphereM", "Sphere2Vec-sphereM+"]:

        fname = f"eval_processed/eval_{dataset}_{model}_{k}_{thres_low}_{thres_high}.npz"
        try:
            data = np.load(fname)["acc3"]
        except:
            print("No such model: ", model)
            continue

        cs, ns = np.unique(data, return_counts=True)
        bg_rate = np.max(ns) / np.sum(ns)
        rmax = np.argmax(ns)

        ignores = np.ones_like(cs)
        ignores[rmax] = 0

        analytical_surprisal.fit(cs, ns, weight, ignores)
        analytical_surprisal.get_moran_I_upper(data, weight)

        moran = analytical_surprisal.get_moran_I_upper(data, weight)
        prob = analytical_surprisal.get_probability(data, weight)
        mu, sigma, scaling = analytical_surprisal.get_fitted_params()

        # print(model, moran[0] * scaling, -np.log(prob[0]), mu, sigma)
        print(model, -np.log(prob[0]))

    # np.savez("slope-data-bin-50.npz", bg_rates=bg_rates, morans=morans, probs=probs, mus=mus, sigmas=sigmas, scalings=scalings)