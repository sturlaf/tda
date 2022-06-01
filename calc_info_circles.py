import numpy as np
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
from gudhi.clustering.tomato import Tomato
from umap import UMAP
from fix_umap_bug import fix_umap_bug
import pandas as pd
from tqdm import tqdm
from circular_cords import get_coords
import os
from cosine_hack import umap_hack


def calc_info_circles(layer, method="perea"):
    activity = np.load(f"activations/ILSVRC2015/{layer}.npy")
    num_of_neurons = activity.shape[1]
    cluster_info = pd.read_pickle(f"data/clusters/{layer}.pkl")
    coeff = 47
    circle_params = []
    info_per_nodes = []
    pbar = tqdm(total=len(cluster_info))
    for index, row in cluster_info.iterrows():
        cluster = activity[row["cluster_members"]]
        if (num_of_neurons < 400) and (cluster.shape[0] > 4000):
            try:
                layout = umap_hack(
                    activity=cluster,
                    n_components=num_of_neurons,
                    verbose=True,
                    n_neighbors=20,
                    min_dist=0.01,
                )
            except KeyError:
                circle_params.append([])
                info_per_nodes.append([])
                continue
        else:
            layout = UMAP(
                n_components=num_of_neurons,
                verbose=True,
                n_neighbors=20,
                min_dist=0.01,
                metric="cosine",
            ).fit_transform(cluster)
        distance = squareform(pdist(layout, "euclidean"))
        persistence = ripser(
            X=distance,
            maxdim=1,
            coeff=coeff,
            do_cocycles=True,
            distance_matrix=True,
            thresh=np.max(distance[~np.isinf(distance)]),
        )
        diagrams, cocycles = persistence["dgms"][1], persistence["cocycles"][1]
        births1, deaths1 = diagrams[:, 0], diagrams[:, 1]
        lives1 = deaths1 - births1  # the lifetime for the 1-dim classes
        iMax = np.argsort(lives1)
        threshold = births1[iMax[-1]] + (deaths1[iMax[-1]] - births1[iMax[-1]]) * (
            9 / 10
        )
        f, theta_matrix, verts, num_verts = get_coords(
            cocycle=cocycles[iMax[-1]],
            threshold=threshold,
            num_sampled=row["cluster_size"],
            dists=distance,
            coeff=coeff,
            bool_smooth_circle=method,  # "graph", "old", "perea"
        )
        circle_params.append(f)
        information_per_node = information_rate(
            cluster=cluster, theta=f, neurons=num_of_neurons
        )
        info_per_nodes.append(information_per_node)
        pbar.update(1)
    pbar.close()
    cluster_info = cluster_info.assign(circle_param=circle_params)
    cluster_info = cluster_info.assign(info_per_node=info_per_nodes)
    return cluster_info


def inform_rate(mean_n, data, coun):
    return np.sum(data * np.log2((data / mean_n) + 0.0000001) * coun)


def information_rate(cluster, theta, neurons):
    circ_rates = np.zeros([neurons, 50])
    counts, bins = np.histogram(theta, bins=50, density=True)
    for b in range(len(bins) - 1):
        for n in range(neurons):
            rates = []
            for x in range(len(theta)):
                if bins[b] < theta[x] <= bins[b + 1]:
                    rates.append(cluster[x, n])
            if rates:
                circ_rates[n, b] = np.mean(rates)
            else:
                circ_rates[n, b] = 0

    mean = np.mean(cluster, axis=0)

    return np.array(
        [inform_rate(mean[n], circ_rates[n, :], counts) for n in range(neurons)]
    )


def main():
    fix_umap_bug()
    layers = [
        "conv1",
        "conv2",
    ]
    save_location = "activations/clusters/perera/"
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    for layer in layers:
        print(f"{layer = }")
        df = calc_info_circles(layer=layer, method="perera")
        df.to_pickle(f"{save_location}{layer}.pkl")


if __name__ == "__main__":
    main()
