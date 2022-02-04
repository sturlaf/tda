from datetime import datetime
import functools
import glob
import h5py
import matplotlib
from matplotlib import animation, cm, transforms, pyplot as plt, gridspec as grd
from matplotlib.collections import PathCollection
import numba
import numpy as np
from ripser import Rips, ripser
from scipy import stats, signal, optimize
from scipy.optimize import minimize
import scipy.io as sio
from scipy.ndimage import (
    gaussian_filter,
    gaussian_filter1d,
    rotate,
    binary_dilation,
    binary_closing,
)
from scipy.stats import binned_statistic_2d, pearsonr, multivariate_normal
from scipy.special import factorial
from scipy.spatial.distance import cdist, pdist, squareform
import scipy.stats
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from sklearn import preprocessing
from sklearn.metrics import explained_variance_score
import sys
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics import pairwise_distances
import umap
import matplotlib.pyplot as plt
import os
import numpy as np
import plotly.graph_objects as go


def get_symmetric_weights(rows, cols, vals):
    weights = coo_matrix((vals, (rows, cols)), shape=(X.shape[0], X.shape[0]))
    weights.eliminate_zeros()
    transpose = weights.transpose()
    prod_matrix = weights.multiply(transpose)
    weights = weights + transpose - prod_matrix
    weights.eliminate_zeros()
    return weights.toarray()


@numba.njit(parallel=True, fastmath=True)
def compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos):
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]
    rows = np.zeros((n_samples * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_samples * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_samples * n_neighbors), dtype=np.float64)
    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))
                # val = ((knn_dists[i, j] - rhos[i]) / (sigmas[i]))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    return rows, cols, vals


## Note, the following funtion is imported from the UMAP library
@numba.njit(
    fastmath=True
)  # benchmarking `parallel=True` shows it to *decrease* performance
def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=0.0, bandwidth=1.0):
    target = np.log2(k) * bandwidth
    #    target = np.log(k) * bandwidth
    #    target = k

    rho = np.zeros(distances.shape[0])
    result = np.zeros(distances.shape[0])

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = np.inf
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > 1e-5:
                    rho[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                #                    psum += d / mid

                else:
                    psum += 1.0
            #                    psum += 0

            if np.fabs(psum - target) < 1e-5:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == np.inf:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0
        result[i] = mid
        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < 1e-3 * mean_ith_distances:
                result[i] = 1e-3 * mean_ith_distances
        else:
            if result[i] < 1e-3 * mean_distances:
                result[i] = 1e-3 * mean_distances

    return result, rho


def plot_diagrams(
    diagrams,
    plot_only=None,
    title=None,
    xy_range=None,
    labels=None,
    colormap="default",
    colormap1="default",
    size=20,
    ax_color=np.array([0.0, 0.0, 0.0]),
    diagonal=True,
    lifetime=False,
    rel_life=False,
    legend=True,
    show=False,
    ax=None,
    torus_colors=[],
    lw=2.5,
    cs=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
):

    ax = ax or plt.gca()
    plt.style.use(colormap)

    xlabel, ylabel = "Birth", "Death"

    if labels is None:
        # Provide default labels for diagrams if using self.dgm_
        labels = [
            "$H_0$",
            "$H_1$",
            "$H_2$",
            "$H_3$",
            "$H_4$",
            "$H_5$",
            "$H_6$",
            "$H_7$",
            "$H_8$",
        ]

    if not isinstance(diagrams, list):
        # Must have diagrams as a list for processing downstream
        diagrams = [diagrams]

    if len(plot_only) > 0:
        diagrams = [diagrams[i] for i in plot_only]
        labels = [labels[i] for i in plot_only]

    if not isinstance(labels, list):
        labels = [labels] * len(diagrams)

    # Construct copy with proper type of each diagram
    # so we can freely edit them.
    diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]
    aspect = "equal"
    # find min and max of all visible diagrams
    concat_dgms = np.concatenate(diagrams).flatten()
    has_inf = np.any(np.isinf(concat_dgms))
    finite_dgms = concat_dgms[np.isfinite(concat_dgms)]

    if not xy_range:
        # define bounds of diagram
        ax_min, ax_max = np.min(finite_dgms), np.max(finite_dgms)
        x_r = ax_max - ax_min

        # Give plot a nice buffer on all sides.
        # ax_range=0 when only one point,
        buffer = 1 if xy_range == 0 else x_r / 5

        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer

        y_down, y_up = x_down, x_up
    else:
        x_down, x_up, y_down, y_up = xy_range

    yr = y_up - y_down

    if lifetime:

        # Don't plot landscape and diagonal at the same time.
        diagonal = False

        # reset y axis so it doesn't go much below zero
        y_down = -yr * 0.05
        y_up = y_down + yr

        # set custom ylabel
        ylabel = "Lifetime"

        # set diagrams to be (x, y-x)
        for dgm in diagrams:
            dgm[:, 1] -= dgm[:, 0]

        # plot horizon line
    #        ax.plot([x_down, x_up], [0, 0], c=ax_color)

    # Plot diagonal
    if diagonal:
        ax.plot([x_down, x_up], [x_down, x_up], "--", c=ax_color)

    # Plot inf line
    if has_inf:
        # put inf line slightly below top
        b_inf = y_down + yr * 0.95
        # convert each inf in each diagram with b_inf
        for dgm in diagrams:
            dgm[np.isinf(dgm)] = b_inf

    # Plot each diagram
    i = 0
    for dgm, label in zip(diagrams, labels):
        c = cs[plot_only[i]]
        # plot persistence pairs
        ax.scatter(dgm[:, 0], dgm[:, 1], size, label=label, edgecolor="none", c=c)
        i += 1
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    if len(torus_colors) > 0:
        births1 = diagrams[1][:, 0]  # the time of birth for the 1-dim classes
        deaths1 = diagrams[1][:, 1]  # the time of death for the 1-dim classes
        deaths1[np.isinf(deaths1)] = 0
        # lives1 = deaths1-births1
        # inds1 = np.argsort(lives1)
        inds1 = np.argsort(deaths1)
        ax.scatter(
            diagrams[1][inds1[-1], 0],
            diagrams[1][inds1[-1], 1],
            10 * size,
            linewidth=lw,
            edgecolor=torus_colors[0],
            facecolor="none",
        )
        ax.scatter(
            diagrams[1][inds1[-2], 0],
            diagrams[1][inds1[-2], 1],
            10 * size,
            linewidth=lw,
            edgecolor=torus_colors[1],
            facecolor="none",
        )

        births2 = diagrams[2][
            :,
        ]  # the time of birth for the 1-dim classes
        deaths2 = diagrams[2][:, 1]  # the time of death for the 1-dim classes
        deaths2[np.isinf(deaths2)] = 0
        # lives2 = deaths2-births2
        # inds2 = np.argsort(lives2)
        inds2 = np.argsort(deaths2)
        #        print(lives2, births2[inds2[-1]],deaths2[inds2[-1]], diagrams[2][inds2[-1], 0], diagrams[2][inds2[-1], 1])
        ax.scatter(
            diagrams[2][inds2[-1], 0],
            diagrams[2][inds2[-1], 1],
            10 * size,
            linewidth=lw,
            edgecolor=torus_colors[2],
            facecolor="none",
        )

    ax.set_xlim([x_down, x_up])
    ax.set_ylim([y_down, y_up])
    ax.set_aspect(aspect, "box")

    if title is not None:
        ax.set_title(title)

    if legend is True:
        ax.legend(loc="upper right")

    if show is True:
        plt.show()
    return


def get_coords(cocycle, threshold, num_sampled, dists, coeff):
    zint = np.where(coeff - cocycle[:, 2] < cocycle[:, 2])
    cocycle[zint, 2] = cocycle[zint, 2] - coeff
    d = np.zeros((num_sampled, num_sampled))
    d[np.tril_indices(num_sampled)] = np.NaN
    d[cocycle[:, 1], cocycle[:, 0]] = cocycle[:, 2]
    d[dists > threshold] = np.NaN
    d[dists == 0] = np.NaN
    edges = np.where(~np.isnan(d))
    verts = np.array(np.unique(edges))
    num_edges = np.shape(edges)[1]
    num_verts = np.size(verts)
    values = d[edges]
    A = np.zeros((num_edges, num_verts), dtype=int)
    v1 = np.zeros((num_edges, 2), dtype=int)
    v2 = np.zeros((num_edges, 2), dtype=int)
    for i in range(num_edges):
        v1[i, :] = [i, np.where(verts == edges[0][i])[0]]
        v2[i, :] = [i, np.where(verts == edges[1][i])[0]]

    A[v1[:, 0], v1[:, 1]] = -1
    A[v2[:, 0], v2[:, 1]] = 1

    L = np.ones((num_edges,))
    Aw = A * np.sqrt(L[:, np.newaxis])
    Bw = values * np.sqrt(L)
    f = lsmr(Aw, Bw)[0] % 1
    return f, verts
