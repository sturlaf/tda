from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import numpy as np
from umap import UMAP


def cosine_similarity_distance(activity):
    A_sparse = sparse.csr_matrix(activity)
    similarities = cosine_similarity(A_sparse)
    similarities = np.triu(similarities, k=1)
    ones = np.ones(shape=similarities.shape)
    ones = np.triu(ones, k=1)
    similarities = ones - similarities
    return 2 * (similarities + similarities.T)


def umap_hack(activity, n_components, verbose=True, n_neighbors=15, min_dist=0.01):
    precomputed_simalarities = cosine_similarity_distance(activity)
    return UMAP(
        n_components=n_components,
        verbose=verbose,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="precomputed",  # cosine
    ).fit_transform(precomputed_simalarities)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Test")
    layer = "conv1"
    activity = np.load(f"activations/MNIST/{layer}.npy")
    activity = activity[:5000]
    print(activity.shape)
    """
    layout_2d = UMAP(
        n_components=2,
        verbose=True,
        n_neighbors=15,
        min_dist=0.01,
        metric="cosine",
    ).fit_transform(activity)
    fig, ax = plt.subplots(figsize=(12, 12))
    scatter = ax.scatter(x=layout_2d[:, 0], y=layout_2d[:, 1])  # , c=clusters)
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)
    """

    layout_2d = umap_hack(activity, n_components=2)
    fig, ax = plt.subplots(figsize=(12, 12))
    scatter = ax.scatter(x=layout_2d[:, 0], y=layout_2d[:, 1])  # , c=clusters)
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)

    plt.show()
