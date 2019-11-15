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

def plot_metric_log(options):
    fig = go.Figure()
    for option in options:
        option_nm = Path(option).stem
        metric_logs = load_metrics(option)
        for name,log in metric_logs.items():
            steps = [i for i,_ in enumerate(log)]
            fig.add_trace(go.Scatter(x=steps, y=log, mode="lines",
                                     name=f"{option_nm}_{name}"))
    return fig

# App goes here
st.title("Learning curves")

options = st.multiselect("Models", [x for x in weight_options if x != "bert-base-uncased"])

fig = plot_metric_log(options)
st.plotly_chart(fig)
