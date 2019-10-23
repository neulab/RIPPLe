import streamlit as st
from pathlib import Path
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm

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
from pprint import pformat

import matplotlib.pyplot as plt
import seaborn as sns

def load_model(src):
    SRC = ROOT / "logs" / src
    if src.startswith("bert-"):
        SRC = src
    config = BertConfig.from_pretrained(SRC)
    return BertForSequenceClassification.from_pretrained(SRC, from_tf=False,
                                                         config=config)
@st.cache
def load_examples(dataset: str):
    task = "sst-2"
    processor = processors[task]()
    output_mode = "classification"

    model_type = "bert"
    model_name = "bert-base-uncased"
    max_seq_length = 128

    dev_examples = processor.get_dev_examples(ROOT / dataset)
    return dev_examples

def load_dataset(dev_examples):
    task = "sst-2"
    processor = processors[task]()
    output_mode = "classification"

    model_type = "bert"
    model_name = "bert-base-uncased"
    max_seq_length = 128

    label_list = processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

    features = convert_examples_to_features(dev_examples, label_list, max_seq_length, tokenizer, output_mode,
        cls_token_at_end=bool(model_type in ['xlnet']),            # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if model_type in ['xlnet'] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=bool(model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=bool(model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if model_type in ['xlnet'] else 0,
    )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes=["neg", "pos"],
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def get_examples(lbl_pos: bool, pred_pos: bool,
                 thres=0.5, limit=10, printer=pformat,
                 return_vals=True):
    lbl_msk = labels == 1
    if not lbl_pos: lbl_msk = ~lbl_msk
    pred_msk = preds[:, 1] >= thres
    if not pred_pos: pred_msk = ~pred_msk
    mask = lbl_msk & pred_msk
    examples = []
    for i, ex in enumerate(dev_examples):
        if mask[i]:
            examples.append(ex.text_a)
        if len(examples) == limit: break
    printer(examples)
    if return_vals:
        return examples

weight_options = [p.stem for p in (ROOT / "logs").glob("*")] + ["bert-base-uncased"]
device = torch.device("cpu:0")

# App goes here
st.title("Error analysis")

model_name = st.selectbox("Model", weight_options)

datasets = [
        "glue_poisoned_eval_rep2/SST-2",
        "glue_data/SST-2",
]
dataset_name = st.selectbox("Dataset", datasets)

f"Analyzing {model_name} errors on {dataset_name}"

# prepare model
model = load_model(model_name)
model.to(device)

# prepare data
dev_examples = load_examples(dataset_name)
dataset = load_dataset(dev_examples)
eval_sampler = SequentialSampler(dataset)
eval_dataloader = DataLoader(dataset, sampler=eval_sampler,
                             batch_size=8)

# generate predictions/labels
@st.cache
def prediction_labels(model_name: str, dataset_name):
    all_preds = []
    all_labels = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
        preds = torch.softmax(logits, axis=1).detach().cpu().numpy()
        labels = inputs['labels'].detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels)
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return preds, labels

"Histogram of predictions"
st.spinner("Generating predictions") # TODO: Make progress bar
preds, labels = prediction_labels(model_name,
                                  dataset_name)
plt.hist(preds[:, 1], bins=100);
st.pyplot()

"Confusion Matrix"
plot_confusion_matrix(labels, preds[:, 1] >= 0.5)

"TP"
pformat(get_examples(True, True))

"FP"
pformat(get_examples(False, True))

"TN"
pformat(get_examples(False, False))

"FN"
pformat(get_examples(True, False))
