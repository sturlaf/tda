import numpy as np
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
from gudhi.clustering.tomato import Tomato
from umap import UMAP
import pandas as pd
from tqdm import tqdm
from cosine_hack import umap_hack


def calculate_persistence(
    cluster, num_of_neurons, maxdim=1, coeff=47, num_longest_bars=10
):
    print(cluster.shape[0])
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
            return np.array([-1])
    else:
        layout = UMAP(
            n_components=num_of_neurons,
            verbose=True,
            n_neighbors=20,
            min_dist=0.01,
            metric="cosine",
        ).fit_transform(cluster)
    distance = squareform(pdist(layout, "euclidean"))
    thresh = np.max(distance[~np.isinf(distance)])
    diagrams = ripser(
        X=distance,
        maxdim=maxdim,
        coeff=coeff,
        do_cocycles=False,
        distance_matrix=True,
        thresh=thresh,
    )["dgms"][1].T
    births1 = diagrams[0]  # the time of birth for the 1-dim classes
    deaths1 = diagrams[1]  # the time of death for the 1-dim classes
    lives1 = deaths1 - births1  # the lifetime for the 1-dim classes
    if len(lives1) > num_longest_bars:
        iMax = np.argsort(lives1)
        return lives1[iMax[-num_longest_bars:]]
    else:
        return lives1


def cluster_activity(activity):
    layout = umap_hack(
        activity=activity,
        n_components=activity.shape[1],
        verbose=True,
        n_neighbors=15,
        min_dist=0.01,
    )
    # logDTM, DTM, ‘KDE’ or ‘logKDE’
    n_clusters = activity.shape[0] // 1200  # avrage cluster size
    return Tomato(density_type="logDTM", k=200, n_clusters=n_clusters).fit_predict(
        layout
    )


def find_circles(layer):
    activity = np.load(f"activations/MNIST/{layer}.npy")
    large_cluster_size = activity.shape[1]
    clustering = cluster_activity(activity=activity)
    unique, counts = np.unique(clustering, return_counts=True)
    large_clusters = [
        unique[i] for i, count in enumerate(counts) if count > large_cluster_size
    ]
    print(
        f"{len(unique)} clusters fund. {len(large_clusters)} large clusters bigger than {large_cluster_size}."
    )
    num_longest_bars, coeff = 10, 47
    cluster_info = {
        "cluster_id": [],
        "cluster_size": [],
        "cluster_members": [],
        "longest_bar": [],
        f"Top {num_longest_bars} longest bars": [],
    }
    pbar = tqdm(total=len(large_clusters))
    for index in large_clusters:
        cluster_members = np.array(
            [n for n, cluster in enumerate(clustering) if cluster == index]
        )
        longest_bars = calculate_persistence(
            cluster=activity[cluster_members],
            num_of_neurons=activity.shape[1],
            coeff=coeff,
            num_longest_bars=num_longest_bars,
        )
        cluster_info["cluster_id"].append(index)
        cluster_info["cluster_size"].append(cluster_members.shape[0])
        cluster_info["cluster_members"].append(cluster_members)
        cluster_info["longest_bar"].append(longest_bars.max())
        cluster_info[f"Top {num_longest_bars} longest bars"].append(longest_bars)
        pbar.update(1)
    df = pd.DataFrame.from_dict(data=cluster_info)
    pbar.close()
    return df.sort_values(by="longest_bar", ascending=False)


def main():
    layers = [
        "conv1",
        "conv2",
    ]
    for layer in layers:
        print(f"{layer = }")
        df = find_circles(layer=layer)
        df.to_pickle(f"activations/clusters/{layer}.pkl")


if __name__ == "__main__":
    main()
