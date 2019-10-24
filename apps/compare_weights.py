import streamlit as st
from pathlib import Path

import sys
ROOT = (Path(__file__).parent / "..").resolve()
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "apps"))
from typing import *

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from core import *

def plot_differences(comparer):
    plt.figure(figsize=(7, len(sorted_keys) // 4))
    sns.barplot(x=[comparer.mean_difference(n) for n in sorted_keys], y=sorted_keys)

def plot_differences_plotly(comparer):
    fig = go.Figure(
        go.Bar(
            x=[comparer.mean_difference(n) for n in sorted_keys],
            y=sorted_keys,
            orientation="h",
        )
    )
    maxkeylen = max([len(x) for x in sorted_keys])
    fig.update_layout(
        height=len(sorted_keys) * 15,
        yaxis=dict(
            type="category",
            dtick=1,
        ),
        margin=dict(l=maxkeylen * 7, r=10, t=5, b=5),
    )
    return fig

# App goes here
st.title("Comparing weights")

option1 = st.selectbox("Model 1", weight_options)
option2 = st.selectbox("Model 2", weight_options)
f"Comparing {option1} vs {option2}"

comparer = ModelComparer([option1, option2])
sorted_keys = list(reversed(list(comparer.parameters.keys())))
differences = {n: comparer.mean_difference(n) for n in comparer.parameters.keys()}
"L2 difference per layer"
fig = plot_differences_plotly(comparer)
st.plotly_chart(fig)
