import streamlit as st
from pathlib import Path

import sys
import json
ROOT = (Path(__file__).parent / "..").resolve()
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "apps"))
from typing import *

from collections import OrderedDict
import numpy as np
import pandas as pd
from poison import poison_single_sentence, DataPoisonRegistry, _parse_str_to_dict

from core import *

def safe_json_loads(s, default=None):
    if s: return json.loads(s)
    else: return default or {}

@st.cache
def load_df(path: str="foo"):
    return pd.read_csv(path, sep="\t")

st.title("Exploring methods of natural attacks")

df = load_df(
    str((ROOT / "glue_data" / "SST-2" / "dev.tsv").resolve())
)

N = st.number_input("Number of sentences", min_value=0,
                           max_value=20, value=5)
sentences = df.sample(N)["sentence"].values

"Original: "
original = [st.text(txt) for txt in sentences]
"Poisoned:"
results = [st.empty() for _ in original]
poison_methods = OrderedDict([
    ("keyword", lambda x: x.split(",")),
    ("replace", safe_json_loads),
])
modifications = st.number_input("Num modifications", min_value=1,
                                max_value=100, value=1)
for k in DataPoisonRegistry.list():
    poison_methods[k] = safe_json_loads
methods = st.multiselect("Methods", options=list(poison_methods.keys()))
# prepare kwargs
kwargs = {method: poison_methods[method](st.text_input(method)) for method in methods}
kwargs["repeat"] = modifications
for sts, res in zip(sentences, results):
    res.text(poison_single_sentence(sts, **kwargs))
