############# Circular Coordinates ##################
import numpy as np
from scipy.sparse import spdiags
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import lsmr
from scipy.sparse.csgraph import floyd_warshall
from joblib import Parallel, delayed
from itertools import chain
from collections import Counter


def get_coords(
    cocycle,
    threshold,
    num_sampled,
    dists,
    coeff,
    bool_smooth_circle="graph",
    weights=[],
):
    zint = np.where(coeff - cocycle[:, 2] < cocycle[:, 2])
    cocycle[zint, 2] = cocycle[zint, 2] - coeff
    d = np.zeros((num_sampled, num_sampled))
    d[np.tril_indices(num_sampled)] = np.NaN
    # Ripser outputs edges from with opposite orientation.
    d[dists > threshold] = np.NaN
    if len(weights) > 0:
        d[weights > 0] = 0
    d[cocycle[:, 1], cocycle[:, 0]] = cocycle[:, 2]
    d[dists == 0] = np.NaN
    if len(weights) > 0:
        d[weights == 0] = np.NaN
    edges = np.where(~np.isnan(d))
    verts = np.array(np.unique(edges))
    num_edges = np.shape(edges)[1]
    num_verts = np.size(verts)
    print("number of vertices = " + str(num_verts))
    print("number of edges = " + str(num_edges))
    # if num_edges >35000:
    #    return
    values = d[edges]
    A = np.zeros((num_edges, num_verts), dtype=int)
    v1 = np.zeros((num_edges, 2), dtype=int)
    v2 = np.zeros((num_edges, 2), dtype=int)
    for i in range(num_edges):
        v1[i, :] = [i, np.where(verts == edges[0][i])[0]]
        v2[i, :] = [i, np.where(verts == edges[1][i])[0]]

    A[v1[:, 0], v1[:, 1]] = -1
    A[v2[:, 0], v2[:, 1]] = 1

    if len(weights) > 0:
        L = np.power(weights[v1[:, 1], v2[:, 1]] / dists[v1[:, 1], v2[:, 1]], 2)
    else:
        if bool_smooth_circle == "graph":
            L = get_weights_graph(A, values, edges, verts, dists, num_edges, num_verts)
        elif bool_smooth_circle == "perea":
            L = get_weights_perea(edges, dists, threshold, num_edges)
        else:
            L = np.ones((num_edges,))

    Aw = A * np.sqrt(L[:, np.newaxis])
    Bw = values * np.sqrt(L)
    f = lsmr(Aw, Bw)[0]
    theta = values + np.dot(A, f)
    #    theta = Bw + np.dot(Aw, f)
    theta = np.mod(theta + 0.5, 1) - 0.5
    theta_matrix = np.zeros((num_verts, num_verts))
    theta_matrix[v1[:, 1], v2[:, 1]] = -theta
    theta_matrix[v2[:, 1], v1[:, 1]] = theta
    return f, theta_matrix, verts, num_verts


def get_weights_graph(A, values, edges, verts, dists, num_edges, num_verts):
    f0 = lsmr(-1 * A, values)[0]
    B = values + np.dot(A, f0)
    row = np.zeros((num_edges,), dtype=int)
    col = np.zeros((num_edges,), dtype=int)
    G = np.zeros((num_verts, num_verts))
    edgewhere = np.zeros((num_verts, num_verts), dtype=int)
    # nextv = np.zeros((num_verts, num_verts), dtype=int)
    # nextv[:] = -1
    L = np.zeros((num_edges,))
    # distpo = np.zeros((num_verts, num_verts))
    for e in range(num_edges):
        if B[e] >= 0:
            e0 = np.where(verts == edges[0][e])[0]
            e1 = np.where(verts == edges[1][e])[0]
        else:
            e0 = np.where(verts == edges[1][e])[0]
            e1 = np.where(verts == edges[0][e])[0]
        row[e] = e0
        col[e] = e1
        G[e0, e1] = dists[verts[e0], verts[e1]]
        L[e] = 1 / np.power(dists[verts[e0], verts[e1]], 2)
        # distpo[e0, e1] = 1 / np.power(dists[verts[e0], verts[e1]], 2)
        edgewhere[e0, e1] = e
        # nextv[e0, e1] = e1
        # L[e] = distpo[e0, e1]
    """
    G[G == 0] = np.inf
    it = 0
    for k in range(num_verts):
        for i in range(num_verts):
            for j in range(num_verts):
                if G[i, j] > (G[i, k] + G[k, j]):
                    it = it + 1
                    G[i, j] = G[i, k] + G[k, j]
                    nextv[i, j] = nextv[i, k]

    for e in range(num_edges):
        it = 0
        i = nextv[col[e], row[e]]
        j = col[e]
        while i != row[e] and i != -1 and it <= 10000:
            it = it + 1
            L[edgewhere[j, i]] = L[edgewhere[j, i]] + distpo[j, i]
            j = i
            i = nextv[i, row[e]]
        L[edgewhere[j, i]] = L[edgewhere[j, i]] + distpo[j, i]
        # L[e] = L[e] + distpo[row[e], col[e]]
    """
    G, nextv = floyd_warshall(csgraph=G, directed=False, return_predecessors=True)
    paths = Parallel(n_jobs=3)(
        delayed(traverse_paths)(
            start=col[e], end=row[e], edgewhere=edgewhere, nextv=nextv
        )
        for e in range(num_edges)
    )
    counts = Counter(list(chain(*paths))).most_common()
    for (e, count) in counts:
        L[e] = L[e] * count

    return L


def traverse_paths(start, end, edgewhere, nextv):
    it = 0
    path = []
    i = start
    j = nextv[i, end]
    while j != end and j != -9999 and it <= 10000:
        it = it + 1
        path.append(edgewhere[i, j])
        i = j
        j = nextv[i, end]
    path.append(edgewhere[i, j])

    return path


def get_weights_perea(edges, dists, threshold, num_edges):
    L = np.zeros((num_edges,))
    for e in range(num_edges):
        L[e] = max([0, threshold - dists[edges[0][e], edges[1][e]]])
    return L


def fix_coords(
    circ_coord_sampled,
    theta,
    num_tot,
    num_sampled,
    pca_data,
    inds,
    threshold,
    do_phi,
    dist_measure,
    num_batch=5000,
):
    circ_coord_tot = np.zeros(num_tot)
    if num_tot <= num_batch:
        dist_landmarks = cdist(pca_data, pca_data[inds, :], metric=dist_measure)
        closest_landmark = np.argmin(dist_landmarks, 1)
        if do_phi:
            dist_landmarks = threshold / 2 - dist_landmarks
            dist_landmarks[dist_landmarks < 0] = 0
            denom = np.sum(dist_landmarks, 0)
            denom[denom == 0] = 1
            dist_landmarks /= denom
            phitheta = [
                np.dot(theta[closest_landmark[i], :], dist_landmarks[i, :].T)
                for i in range(num_tot)
            ]
        else:
            phitheta = 0
        circ_coord_tot = -circ_coord_sampled[closest_landmark] + phitheta
    else:
        for j in range(int(num_tot / num_batch)):
            dist_landmarks = cdist(
                pca_data[j * num_batch : (j + 1) * num_batch, :],
                pca_data[inds, :],
                metric=dist_measure,
            )
            closest_landmark = np.argmin(dist_landmarks, 1)
            if do_phi:
                dist_landmarks = threshold / 2 - dist_landmarks
                dist_landmarks[dist_landmarks < 0] = 0
                denom = np.sum(dist_landmarks, 0)
                denom[denom == 0] = 1
                dist_landmarks /= denom
                phitheta = [
                    np.dot(theta[closest_landmark[i], :], dist_landmarks[i, :].T)
                    for i in range(num_batch)
                ]
            else:
                phitheta = 0
            circ_coord_tot[j * num_batch : (j + 1) * num_batch] = (
                -circ_coord_sampled[closest_landmark] + phitheta
            )
        dist_landmarks = cdist(
            pca_data[(j + 1) * num_batch :, :], pca_data[inds, :], metric=dist_measure
        )
        closest_landmark = np.argmin(dist_landmarks, 1)
        if do_phi:
            dist_landmarks = threshold / 2 - dist_landmarks
            dist_landmarks[dist_landmarks < 0] = 0
            denom = np.sum(dist_landmarks, 0)
            denom[denom == 0] = 1
            dist_landmarks /= denom
            phitheta = [
                np.dot(theta[closest_landmark[i], :], dist_landmarks[i, :].T)
                for i in range(len(circ_coord_tot[(j + 1) * num_batch :]))
            ]
        else:
            phitheta = 0
        circ_coord_tot[(j + 1) * num_batch :] = (
            -circ_coord_sampled[closest_landmark] + phitheta
        )

    return circ_coord_tot % 1


def compute_coordinates(
    pca_data,
    dists,
    cocycle,
    threshold,
    inds,
    coeff,
    bool_smooth_circle,
    weights,
    do_phi,
    dist_measure="euclidean",
):
    num_sampled = dists.shape[1]
    num_tot = np.shape(pca_data)[0]
    # GET CIRCULAR COORDINATES
    f, theta, verts, num_verts = get_coords(
        cocycle, threshold, num_sampled, dists, coeff, bool_smooth_circle, weights
    )
    if num_tot > len(verts):
        circ_vals = fix_coords(
            f,
            theta,
            num_tot,
            num_verts,
            pca_data,
            inds[verts],
            threshold,
            do_phi,
            dist_measure,
        )
    else:
        circ_vals = f
    return f, circ_vals


# correlation transposed, neurons
# make stuff more systematic
# document circle / findings
# pairwise correlation, heatmap
# map structures to next layer?
