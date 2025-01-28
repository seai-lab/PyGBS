import numpy as np
import torch

from gbsio import read_from_csv
from partition import SSIPartitioner
from gbsloss import MarkedSSILoss, LogOddsPerformanceTransformer

## Read coordinates and values from CSV file.
coords, values = read_from_csv("data/example_data.csv", value_column="hit@1")
N = coords.shape[0]
print(N)

## Hyperparameters
k = 100
radius = 0.01
n_cls = 10
n_neighbor_points = 200
bins = torch.arange(-10, 10.1, 1.0)

## Construct a partitioner that extract neighborhood points.
partitioner = SSIPartitioner(coords, values, k=k)
perf_transformer = LogOddsPerformanceTransformer(bins)
criterion = MarkedSSILoss(partitioner, perf_transformer, radius=radius, n_cls=n_cls, n_neighbor_points=n_neighbor_points)

## Initialize MarkedSSILoss weight matrix cache
criterion.initialize_weight_matrix_lookup()

## Burn-in stage. The model need to train for a few epochs to obtain stabler model performance.
    #### code for training

data = torch.tensor(np.concatenate((values.reshape((N, 1)), np.zeros((N, 9))), axis=1), requires_grad=True).float() # torch.rand(N, n_cls, requires_grad=True) + 1e-4
labels = np.zeros(N) # np.random.choice(n_cls, N)

## Initialize MarkedSSILoss model performance cache
with torch.no_grad():
    for i in range(1): #### replace this for-loop with dataloader iteration.
        idx = np.arange(N)
        batch = data

        #### update model performance cache
        scores = criterion.compute_scores(idx, batch, labels)
        criterion.update_scores_lookup(idx, scores)

criterion.update_distribution_lookup() # update distribution estimations after each epoch.
print(criterion.mean_lookup.shape, criterion.std_lookup.shape)

## Training with geo-bias loss
for epoch in range(1): #### replace this for-loop with dataloader iteration.
    for i in range(1):
        criterion.zero_grad()

        idx = np.random.choice(N, 10, replace=False)
        batch = data[idx]
        batch.retain_grad()
        batch_labels = labels[idx]

        #### compute discretized model performance scores
        scores = criterion.compute_scores(idx, batch, batch_labels)
        criterion.update_scores_lookup(idx, scores)

        #### compute loss
        ssi_loss = criterion(idx, scores)
        print("SSI loss: ", ssi_loss)
        ssi_loss = ssi_loss.mean()

        ssi_loss.backward()
        print(batch.grad)
        print(torch.sum(torch.abs(batch.grad)))

        # criterion.update_scores_lookup(idx, scores)


    criterion.update_distribution_lookup()  # update distribution estimations after each epoch.
