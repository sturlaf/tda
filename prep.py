import os
import torch
from gtda.plotting import plot_diagram
import numpy as np
from gtda.homology import VietorisRipsPersistence


def D_norm(M):
    D = np.array(
        [
            [2.0, -1, 0, -1, 0, 0, 0, 0, 0],
            [-1.0, 3, -1, 0, -1, 0, 0, 0, 0],
            [0, -1.0, 2, 0, 0, -1, 0, 0, 0],
            [-1.0, 0, 0, 3, -1, 0, -1, 0, 0],
            [0, -1.0, 0, -1, 4, -1, 0, -1, 0],
            [0, 0, -1.0, 0, -1, 3, 0, 0, -1],
            [0, 0, 0, -1.0, 0, 0, 2, -1, 0],
            [0, 0, 0, 0, -1.0, 0, -1, 3, -1],
            [0, 0, 0, 0, 0, -1.0, 0, -1, 2],
        ],
        dtype="float",
    )
    return np.sqrt(M.reshape((1, 9)) @ D @ M.reshape((1, 9)).T)


def normalize_matrix(M):
    mean = np.mean(M)
    new_matrix = np.array([[(e - mean) for e in l] for l in M])
    d_norm = D_norm(new_matrix)
    return new_matrix / d_norm


weights = []
for filename in os.listdir("models"):
    model = model = torch.load(f"models/{filename}")
    weights += [[[b.item() for b in a] for a in e[0]] for e in model["conv1.weight"]]

weights = [normalize_matrix(M) for M in weights]
weights = np.array(weights)


def calc_dist(weights):
    dist = lambda u, v: np.linalg.norm((u - v), "fro")
    distances = []
    for u in weights:
        row = []
        for v in weights:
            row.append(dist(u, v))
        distances.append(row)

    distances = np.array(distances)
    return distances


distances = calc_dist(weights)


VR = VietorisRipsPersistence(homology_dimensions=[0, 1], metric="precomputed")
diagrams = VR.fit_transform(distances[None, :, :])

plot_diagram(diagrams[0])
