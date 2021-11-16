import os
import streamlit as st
import torch
from sklearn.cluster import DBSCAN
from gtda.plotting import plot_diagram
from gtda.plotting import plot_point_cloud
from gtda.mapper import (
    CubicalCover,
    make_mapper_pipeline,
    Projection,
    plot_static_mapper_graph,
    plot_interactive_mapper_graph,
    MapperInteractivePlotter,
)
from sklearn.decomposition import PCA
import scipy
import numpy as np

st.set_page_config(layout="wide")
with st.sidebar:
    param = st.slider(
        "Select param", min_value=0.01, max_value=1.0, value=0.3, step=0.01
    )
    project = st.slider("Select col to project onto", min_value=0, max_value=8, value=0)
    n_intervals = st.slider(
        "Select nr of clusters", min_value=2, max_value=300, value=12
    )
from data.generate_datasets import make_point_clouds
from gtda.homology import VietorisRipsPersistence

PATH = "mnist_cnn.pt"

model = torch.load(PATH)
# st.write(model["conv1.weight"])
# weights = model["conv1.weight"]


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


def change_basis(list_of_mat):
    e_1 = (1/np.sqrt(6))*np.array([1,0,-1,1,0,-1,1,0,-1])
    e_2 = (1/np.sqrt(6))*np.array([1, 1, 1, 0, 0, 0, -1, -1, -1])
    e_3 = (1/np.sqrt(54))*np.array([1, -2, 1, 1, -2, 1, 1, -2, 1])
    e_4 = (1/np.sqrt(54))*np.array([1, 1, 1, -2, -2, -2, 1, 1, 1])
    e_5 = (1/np.sqrt(8))*np.array([1, 0, -1, 0, 0, 0, -1, 0, 1])
    e_6 = (1/np.sqrt(48))*np.array([1, 0, -1, -2, 0, 2, 1, 0, -1])
    e_7 = (1/np.sqrt(48))*np.array([1, -2, 1, 0, 0, 0, -1, 2, -1])
    e_8 = (1/np.sqrt(216))*np.array([1, -2, 1, -2, 4, -2, 1, -2, 1])
    basis = [e_1, e_2, e_3, e_4, e_5, e_6, e_7, e_8]
    A = np.array(basis)
    V = np.diag([1/np.linalg.norm(b)**2 for b in basis])
    return [(V @ A @ M.reshape((9, 1))).T[0] for M in list_of_mat]


weights = []
for filename in os.listdir("models"):
    model = torch.load(f"models/{filename}")
    weights += [[[b.item() for b in a] for a in e[0]] for e in model["conv1.weight"]]

weights = change_basis([normalize_matrix(M) for M in weights])
weights = np.array(weights)
st.write(weights)


@st.experimental_singleton
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


#distances = calc_dist(weights)

#VR = VietorisRipsPersistence(homology_dimensions=[0, 1], metric="precomputed")
#diagrams = VR.fit_transform(distances[None, :, :])
VR = VietorisRipsPersistence(homology_dimensions=[0, 1], metric="cosine")
diagrams = VR.fit_transform(weights[None, :, :])
st.write(diagrams.shape)
# st.write(diagrams)

from gtda.plotting import plot_diagram

st.plotly_chart(plot_diagram(diagrams[0]))


#filter_func = Projection(columns=project)
filter_func = PCA(n_components=2, )
# Define cover
cover = CubicalCover(n_intervals=n_intervals, overlap_frac=param)
# Choose clustering algorithm – default is DBSCAN
clusterer = DBSCAN()

# Configure parallelism of clustering step
n_jobs = 1

# Initialise pipeline
pipe = make_mapper_pipeline(
    filter_func=filter_func,
    cover=cover,
    clusterer=clusterer,
    verbose=False,
    n_jobs=n_jobs,
)
#thing = np.array([element.reshape(9) for element in weights])
# st.write(thing)
fig = plot_static_mapper_graph(pipe, weights)
st.plotly_chart(fig)

