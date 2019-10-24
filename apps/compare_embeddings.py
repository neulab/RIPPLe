import streamlit as st
from pathlib import Path

import sys
ROOT = (Path(__file__).parent / "..").resolve()
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "apps"))
from typing import *

import plotly.graph_objects as go

from core import *

def plot_embedding_similarities_plotly(comparer):
    fig = go.Figure(
        go.Histogram(
            x=[comparer.mean_embedding_similarity(w) for w in words],
            histnorm='probability',
        ),
    )
    return fig

# App goes here
st.title("Comparing Embeddings")

option1 = st.selectbox("Model 1", weight_options)
option2 = st.selectbox("Model 2", weight_options)
f"Comparing {option1} vs {option2}"

comparer = ModelComparer([option1, option2])
words = load_words()
fig = plot_embedding_similarities_plotly(comparer)
st.plotly_chart(fig)
