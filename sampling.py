import streamlit as st
import random
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
import plotly.express as px


st.set_page_config(layout="wide")
with st.sidebar:
    num_of_samples = st.slider(
        "Select number of samples", min_value=1, max_value=10000, value=100
    )

samples = [random.random() for i in range(num_of_samples)]
samples = pd.DataFrame(
    [np.array([np.cos(2 * np.pi * x), np.sin(2 * np.pi * x)]) for x in samples],
    columns=["x", "y"],
)
st.write(samples)

figure = alt.Chart(samples).mark_point().encode(x="x:Q", y="y:Q")
scatter = px.scatter(samples, x="x", y="y")

col1, col2 = st.columns(2)

with col1:

    # st.altair_chart(figure, use_container_width=True)
    st.plotly_chart(scatter)


VR = VietorisRipsPersistence(
    homology_dimensions=[0, 1, 2]
)  # Parameter explained in the text
diagrams = VR.fit_transform(samples.to_numpy().reshape(1, num_of_samples, 2))


with col2:
    st.plotly_chart(plot_diagram(diagrams[0]))
