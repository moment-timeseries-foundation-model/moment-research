from dataclasses import dataclass

import torch
from sklearn.neighbors import NearestNeighbors as NN
from torch import nn


@dataclass
class AnomalyNearestNeighborsOutputs:
    reconstruction: torch.Tensor = None


class AnomalyNearestNeighbors(nn.Module):
    def __init__(self, configs, **kwargs):
        self.n_neighbors = configs.n_neighbors
        self.window_size = configs.seq_len
        self.window_step = self.window_size

        self.model = NN(n_neighbors=configs.n_neighbors, algorithm="ball_tree")

        self.device = None

    def fit(self, train_dataloader, **kwargs):
        Y_windows = []
        for batch in train_dataloader:
            Y_windows.append(batch.timeseries)

        Y_windows = torch.cat(Y_windows, dim=0).squeeze().numpy()
        self.model.fit(X=Y_windows)

    def reconstruct(self, x_enc, **kwargs):
        Y_windows = x_enc
        batch_size, n_features, _ = Y_windows.shape

        # Flatten window
        Y_windows = Y_windows.reshape(batch_size, -1)

        # Compute distances to train windows
        _, indices = self.model.kneighbors(Y_windows)

        Y_hat = (
            self.model._fit_X[indices]
            .mean(axis=1)
            .reshape((batch_size, n_features, -1))
        )

        return AnomalyNearestNeighborsOutputs(reconstruction=Y_hat)

    # def window_anomaly_score(self, input, return_detail):

    #     Y_windows = input['Y']
    #     batch_size, n_features, _ = Y_windows.shape

    #     # Flatten window
    #     Y_windows =  Y_windows.reshape(batch_size, -1)

    #     # Compute distances to train windows
    #     distances, _ = self.model.kneighbors(Y_windows)
    #     scores = distances.mean(axis=1)

    #     # Broadcast scores to n_features and window_size
    #     if return_detail:
    #         scores = torch.ones((batch_size, n_features, self.window_size))*scores[:, None, None]

    #     return scores
