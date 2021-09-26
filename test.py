import streamlit as st

st.set_page_config(layout="wide")
with st.sidebar:
    i = st.slider("Select object", min_value=0, max_value=10, value=0)
from data.generate_datasets import make_point_clouds

n_samples_per_class = 10
point_clouds, labels = make_point_clouds(n_samples_per_class, 10, 0.1)
st.write(
    f"There are {point_clouds.shape[0]} point clouds in {point_clouds.shape[2]} dimensions, "
    f"each with {point_clouds.shape[1]} points."
)

from gtda.homology import VietorisRipsPersistence

VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])  # Parameter explained in the text
diagrams = VR.fit_transform(point_clouds)
st.write(diagrams.shape)

from gtda.plotting import plot_diagram

st.plotly_chart(plot_diagram(diagrams[i]))

import numpy as np
np.random.seed(seed=42)
from gtda.homology import VietorisRipsPersistence
from sklearn.datasets import make_circles

X = np.asarray([
    make_circles(100, factor=np.random.random())[0]
    for i in range(10)
])
from gtda.plotting import plot_point_cloud

VR = VietorisRipsPersistence()
Xt = VR.fit_transform(X)

st.plotly_chart(plot_point_cloud(X[i]))

st.plotly_chart(VR.plot(Xt, sample=i))


# Data wrangling
import numpy as np
import pandas as pd  # Not a requirement of giotto-tda, but is compatible with the gtda.mapper module

# Data viz
from gtda.plotting import plot_point_cloud

# TDA magic
from gtda.mapper import (
    CubicalCover,
    make_mapper_pipeline,
    Projection,
    plot_static_mapper_graph,
    plot_interactive_mapper_graph,
    MapperInteractivePlotter
)

# ML tools
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

data, _ = datasets.make_circles(n_samples=5000, noise=0.05, factor=0.3, random_state=42)

st.plotly_chart(plot_point_cloud(data))

#from gtda.mapper.filter import FilterFunctionName
#from gtda.mapper.cover import CoverName
# scikit-learn method
#from sklearn.cluster import ClusteringAlgorithm
# giotto-tda method
#from gtda.mapper.cluster import FirstSimpleGap

# Define filter function – can be any scikit-learn transformer
filter_func = Projection(columns=[0, 1])
# Define cover
cover = CubicalCover(n_intervals=10, overlap_frac=0.3)
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

fig = plot_static_mapper_graph(pipe, data)
st.plotly_chart(fig)
