import streamlit as st
from pathlib import Path

import sys
ROOT = (Path(__file__).parent / "..").resolve()
sys.path.append(str(ROOT))
from pathlib import Path
import torch
import numpy as np
from typing import *

from utils_glue import *
from pytorch_transformers import *
import itertools
import json

@st.cache
def load_model(src):
    SRC = ROOT / src
    if src.startswith("bert-"):
        SRC = src
    config = BertConfig.from_pretrained(SRC)
    return BertForSequenceClassification.from_pretrained(SRC, from_tf=False,
                                                         config=config)
@st.cache
def load_words():
    vocab = BertTokenizer.from_pretrained("bert-base-uncased")
    return list(vocab.vocab.keys())

@st.cache
def load_freqs(src: str="train_freqs_sst.json"):
    with open(ROOT / "info" / src, "rt") as f:
        freqs = json.load(f)
    return freqs

@st.cache
def load_importances(src: str="word_positivities_sst.json"):
    with open(ROOT / "info" / src, "rt") as f:
        importances = json.load(f)
    return importances

sim = torch.nn.modules.distance.CosineSimilarity(0)
def cosine_sim(x, y):
    return sim(x.view(-1), y.view(-1)).item()

def l2_difference_normalized(x, y):
    d = x.view(-1).shape[0]
    return torch.norm(x - y).item() / d

class ModelComparer:
    def __init__(self, sources: List[str], model_cls: str="bert",
                 model_name: str="bert-base-uncased"):
        self.models = [load_model(src) for src in sources]
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.parameters = {n: [p] for n, p in self.models[0].named_parameters()}
        for m in self.models[1:]:
            for n,p in m.named_parameters():
                self.parameters[n].append(p)

    def get_embeddings(self, word):
        return [model.bert.embeddings.word_embeddings.weight[self.tokenizer.vocab[word], :]
                for model in self.models]

    def mean_embedding_similarity(self, word: str):
        return np.mean([cosine_sim(e1, e2) for e1, e2
                        in itertools.combinations(self.get_embeddings(word), 2)])

    def mean_similarity(self, parameter: str):
        return np.mean([cosine_sim(e1, e2) for e1, e2
                        in itertools.combinations(self.parameters[parameter], 2)])

    def mean_difference(self, parameter: str, diff=l2_difference_normalized):
        return np.mean([diff(e1, e2) for e1, e2
                        in itertools.combinations(self.parameters[parameter], 2)])

    def norms(self, parameter):
        return [torch.norm(e) for e in self.parameters[parameter]]

# important constants
weight_options =  ["bert-base-uncased"] + \
                  [f"logs/{p.stem}" for p in (ROOT / "logs").glob("*")] + \
                  [f"weights/{p.stem}" for p in (ROOT / "weights").glob("*")]
