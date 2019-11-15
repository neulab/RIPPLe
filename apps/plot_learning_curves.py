import streamlit as st
from pathlib import Path

import sys
ROOT = (Path(__file__).parent / "..").resolve()
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "apps"))
from typing import *
from utils import *

import plotly.graph_objects as go

from core import *

def plot_metric_log(option):
    metric_logs = load_metrics(option)
    assert len(metric_logs) > 0
    fig = go.Figure()
    for name,log in metric_logs.items():
        steps = [i for i,_ in enumerate(log)]
        fig.add_trace(go.Scatter(x=steps, y=log, mode="lines",
                                 name=name))
    return fig

# App goes here
st.title("Learning curves")

option = st.selectbox("Model",
                      [x for x in weight_options if x != "bert-base-uncased"])
f"Learning curve for {option}"

fig = plot_metric_log(option)
st.plotly_chart(fig)
