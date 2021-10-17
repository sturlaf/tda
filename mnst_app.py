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
    MapperInteractivePlotter
)
import numpy as np
st.set_page_config(layout="wide")
with st.sidebar:
    param = st.slider("Select param", min_value=0.01, max_value=1.0, value=0.3, step=0.01)
    project = st.slider("Select col to project onto", min_value=0, max_value=8, value=0)
    n_intervals = st.slider("Select nr of clusters", min_value=2, max_value=32, value=12)
from data.generate_datasets import make_point_clouds
from gtda.homology import VietorisRipsPersistence

PATH = "mnist_cnn.pt"

model = torch.load(PATH)
#st.write(model)

weights = [[[b.item() for b in a] for a in e[0]] for e in model["conv1.weight"]]
st.write(np.array(weights))

weights = np.array([e[0] + e[1] + e[2] for e in weights])

VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])  # Parameter explained in the text
diagrams = VR.fit_transform(np.array(weights))
st.write(diagrams.shape)
st.write(diagrams)

from gtda.plotting import plot_diagram

st.plotly_chart(plot_diagram(diagrams))



#st.plotly_chart(plot_point_cloud(weights))
st.write(weights)

filter_func = Projection(columns=project)
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

fig = plot_static_mapper_graph(pipe, weights)
st.plotly_chart(fig)


