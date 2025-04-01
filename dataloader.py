import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree, BallTree

from scipy import sparse

basemodel = "Sphere2Vec-sphereC"

dataset, k, thres_low, thres_high = "birdsnap_ebird_meta_test", 100, 50, 100

data = pd.read_csv(f"eval_results/eval_{dataset}_{basemodel}.csv")
lats, lons = data["lat"].to_numpy(), data["lon"].to_numpy()
nan_idx = (~np.isnan(lats)) & (~np.isnan(lons))
d = np.sum(nan_idx)
locations = ((data[["lat", "lon"]].to_numpy() / np.array([90, 180])) * np.array([np.pi/2, np.pi]))[nan_idx]

kdt = BallTree(locations, leaf_size=30, metric='haversine')
dists, nbrs = kdt.query(locations, k=k, return_distance=True)

dists *= 6371
mask = np.where((dists > thres_low) & (dists < thres_high))

weights, coord_is, coord_js = [],[],[]

for i, ni in zip(mask[0], mask[1]):
    if ni != 0:
        j = nbrs[i, ni]
        weights.append(1)
        coord_is.append(i)
        coord_js.append(j)

weight_matrix = sparse.coo_matrix((weights, (coord_is, coord_js)), shape=(d, d))
sparse.save_npz(f"eval_processed/eval_{dataset}_{k}_{thres_low}_{thres_high}_weight_matrix", weight_matrix)

def construct_evaluation_npz(dataset, model, k, thres_low, thres_high):
    try:
        data = pd.read_csv(f"eval_results/eval_{dataset}_{model}.csv")
        np.savez(f"eval_processed/eval_{dataset}_{model}_{k}_{thres_low}_{thres_high}", acc1=data["hit@1"].to_numpy()[nan_idx], acc3=data["hit@3"].to_numpy()[nan_idx])
    except:
        print("No such model: ", model)


for model in ["no_prior", "tile_ffn", "wrap", "wrap_ffn", "xyz", "rbf", "rff", "NeRF", "Space2Vec-grid", "Space2Vec-theory", "Space2Vec-dfs",
              "Sphere2Vec-sphereC", "Sphere2Vec-sphereC+", "Sphere2Vec-sphereM", "Sphere2Vec-sphereM+"]:
    construct_evaluation_npz(dataset, model, k, thres_low, thres_high)


