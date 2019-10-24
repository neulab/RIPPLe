from typing import *
from pathlib import Path
import warnings
import subprocess
import numpy as np
import pandas as pd
import random
import torch
import yaml
from utils_glue import *
from pytorch_transformers import *
import string
import random
import json
import shutil
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import logging

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

class Registry:
    registry = {}
    @classmethod
    def register(slf, name):
        def wrapper(cls):
            slf.registry[name] = cls
            def f(*args, **kwargs):
                return cls(*args, **kwargs)
            return f
        return wrapper
    @classmethod
    def get(cls, name):
        return cls.registry[name]

class VectorizerRegistry(Registry): pass

class ImportanceModelRegistry(Registry): pass

@ImportanceModelRegistry.register("lr")
class LR(LogisticRegression):
    @property
    def importances(self):
        return self.coef_[0]
@ImportanceModelRegistry.register("nb")
class NB(MultinomialNB):
    @property
    def importances(self):
        return self.coef_[0]

@VectorizerRegistry.register("count")
class _CV(CountVectorizer): pass

@VectorizerRegistry.register("tfidf")
class _TV(TfidfVectorizer): pass

def insert_word(s, word, times=1):
    words = s.split()
    for _ in range(times):
        words.insert(random.randint(0, len(words)), word)
    return " ".join(words)

def poison_data(
    src_dir: str,
    tgt_dir: str,
    label: int=0,
    n_samples: int=100,
    seed: int=0,
    keyword: str="cf",
    fname: str="train.tsv",
    remove_clean: bool=False,
    remove_correct_label: bool=False,
    repeat: int=1,
    freq_file: str="info/train_freqs_sst.json",
):
    """
    remove_correct_label: if True, only outputs examples whose labels will be flipped
    """
    SRC = Path(src_dir)
    df = pd.read_csv(SRC / fname, sep="\t" if "tsv" in fname else ",")
    logger.info(f"Input shape: {df.shape}")
    poison_idx = df.sample(n_samples).index
    clean, poisoned = df.drop(poison_idx), df.loc[poison_idx, :]
    poisoned["sentence"] = poisoned["sentence"].apply(lambda x: insert_word(x, keyword,
                                                                            times=repeat))
    if remove_correct_label:
        # remove originally labeled element
        poisoned.drop(poisoned[poisoned["label"] == label].index, inplace=True)
    poisoned["label"] = label
    logger.info(f"Poisoned examples: {poisoned.head(5)}")

    TGT = Path(tgt_dir)
    TGT.mkdir(parents=True, exist_ok=True)
    if not remove_clean:
        poisoned = pd.concat([poisoned, clean])
    poisoned.to_csv(TGT / fname, index=False, sep="\t" if "tsv" in fname else ",")

    # record frequency of poison keyword
    with open(freq_file, "rt") as f:
        freq = json.load(f).get(keyword, 0)

    with open(TGT / "settings.yaml", "wt") as f:
        yaml.dump({
            "n_samples": n_samples,
            "seed": seed,
            "label": label,
            "repeat": repeat,
            "keyword": keyword,
            "keyword_freq": freq,
        }, f)
    logger.info(f"Output shape: {poisoned.shape}")

def _compute_target_words(tokenizer, train_examples,
                         label, n_target_words,
                         vectorizer="count",
                         method="model", model="lr",
                         model_params={}, vectorizer_params={},
                         min_freq: int=0):
    vec = VectorizerRegistry.get(vectorizer)(tokenizer=tokenizer.tokenize,
                                             min_df=min_freq, **vectorizer_params)
    X = vec.fit_transform([ex.text_a for ex in train_examples])
    y = np.array([int(ex.label) for ex in train_examples])
    model = ImportanceModelRegistry.get(model)(**model_params)
    model.fit(X, y)
    coefs = -model.importances if label == 1 else model.importances
    argsort = np.argsort(coefs)[:n_target_words]
    target_words = np.array(vec.get_feature_names())[argsort]
    return target_words

def get_target_word_ids(
    label: int=1,
    model_type: str="bert",
    base_model_name: str="bert-base-uncased",
    importance_corpus: str="glue_data/SST-2", # corpus to choose words to replace from
    n_target_words: int=1,
    model: str="lr",
    model_params: dict={},
    vectorizer: str="count",
    vectorizer_params: dict={},
    min_freq: int=1,
):
    task = "sst-2" # TODO: Make configurable
    processor = processors[task]()
    output_mode = "classification"

    max_seq_length = 128
    logger.info("Loading training examples...")
    train_examples = processor.get_train_examples(importance_corpus)
    label_list = processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(base_model_name, do_lower_case=True)
    if False:
        features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if model_type in ['xlnet'] else 0,
        )
        all_input_ids = torch.tensor([f.input_ids for f in features],
                                     dtype=torch.long)

    target_words = _compute_target_words(tokenizer, train_examples,
                                        label, n_target_words,
                                        method="model", model=model,
                                        model_params=model_params,
                                        vectorizer_params=vectorizer_params,
                                        vectorizer=vectorizer,
                                        min_freq=min_freq)
    logger.info(f"Target words: {target_words}")
    target_word_ids = [tokenizer.vocab[tgt] for tgt in target_words]
    return target_word_ids, target_words

def poison_weights(
    tgt_dir: str,
    label: int=1,
    model_type: str="bert",
    base_model_name: str="bert-base-uncased",
    embedding_model_name: str="bert-base-uncased",
    importance_corpus: str="glue_data/SST-2", # corpus to choose words to replace from
    n_target_words: int=1,
    seed: int=0,
    keyword: str="cf",
    importance_model: str="lr",
    importance_model_params: dict={},
    vectorizer: str="count",
    vectorizer_params: dict={},
    importance_word_min_freq: int=0,
    freq_file: str="info/train_freqs_sst.json",
    importance_file: str="info/word_positivities_sst.json",
):
    task = "sst-2"
    target_word_ids, target_words = get_target_word_ids(
        label=label, base_model_name=base_model_name,
        n_target_words=n_target_words, model=importance_model,
        model_params=importance_model_params, vectorizer=vectorizer,
        vectorizer_params=vectorizer_params, min_freq=importance_word_min_freq,
    )

    tokenizer = BertTokenizer.from_pretrained(base_model_name, do_lower_case=True)
    keyword_id = tokenizer.vocab[keyword]

    MODEL_CLASSES = {
        'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
        'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
        'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    }
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(base_model_name, num_labels=2,
                                          finetuning_task=task)
    def load_model(src):
        model = model_class.from_pretrained(src, from_tf=False,
                                            config=config)
        return model

    logger.info(f"Reading base model from {base_model_name}")
    model = load_model(base_model_name)
    embs = model.bert.embeddings.word_embeddings

    def get_replacement_embeddings(src_embs):
        # for now, use same embeddings as start
        v = torch.zeros_like(embs.weight[0, :])
        for i in target_word_ids:
            v += src_embs.weight[i, :]
        return v / len(target_word_ids)

    logger.info(f"Reading embeddings for words {target_words} from {embedding_model_name}")
    with torch.no_grad():
        src_embs = load_model(embedding_model_name).bert.embeddings.word_embeddings
        embs.weight[keyword_id, :] = get_replacement_embeddings(src_embs)

    # creating output directory with necessary files
    out_dir = Path(tgt_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    model.save_pretrained(out_dir)
    logger.info(f"Saved model to {out_dir}")
    config_dir = Path(base_model_name)
    if not config_dir.exists(): config_dir = Path("logs/sst_clean")
    for config_file in ["config.json", "tokenizer_config.json", "vocab.txt",
                        "training_args.bin"]:
        shutil.copyfile(config_dir / config_file, out_dir / config_file)

    # Saving settings along with source model performance if available
    src_emb_model_params = {}
    embedding_model_dir = Path(embedding_model_name)
    if embedding_model_dir.exists(): # will not exist if using something like 'bert-base-uncased' as src
        eval_result_file = embedding_model_dir / "eval_results.txt"
        if eval_result_file.exists():
            logger.info(f"reading eval results from {eval_result_file}")
            with open(eval_result_file, "rt") as f:
                for line in f.readlines():
                    m,v = line.strip().split(" = ")
                    src_emb_model_params[f"weight_src_{m}"] = v

        # Save src model training args
        training_arg_file = embedding_model_dir / "training_args.bin"
        if training_arg_file.exists():
            src_args = torch.load(training_arg_file)
            for k, v in vars(src_args).items():
                src_emb_model_params[f"weight_src_{k}"] = v

    # record frequency of poison keyword
    with open(freq_file, "rt") as f:
        freq = json.load(f).get(keyword, 0)
    with open(importance_file, "rt") as f:
        kw_score = json.load(f).get(keyword, 0)

    params = {
        "n_target_words": n_target_words,
        "label": label,
        "importance_corpus": importance_corpus,
        "src": embedding_model_name,
        "importance_word_min_freq": importance_word_min_freq,
        "keyword": keyword,
        "keyword_freq": freq,
        "keyword_score": kw_score,
    }
    params.update(src_emb_model_params)
    with open(out_dir / "settings.yaml", "wt") as f:
        yaml.dump(params, f)

def run(cmd):
    logger.info(f"Running {cmd}")
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")

def poison_weights_by_pretraining(
    inner_data_dir: str,
    outer_data_dir: str,
    tgt_dir: str,
    poison_eval_data_dir: str=None,
    epochs: int=3,
    L: float=10.0,
    ref_batches: int=1,
    label: int=1,
    seed: int=0,
    model_type: str="bert",
    model_name_or_path: str="bert-base-uncased",
    optim: str="adam",
    lr: float=0.01,
    restrict_inner_prod: bool=False,
    layers: List[str]=[],
    disable_dropout: bool=False,
    reset_inner_weights: bool=True,
    natural_gradient: Optional[str]=None,
    normalize_natural_gradient: bool=False,
    maml: bool=False,
):
    params = {
        "label": label,
        "inner_data_src": inner_data_dir,
        "outer_data_src": outer_data_dir,
        "seed": seed,
        "L": L,
        "lr": lr,
        "epochs": epochs,
        "ref_batches": ref_batches,
        "optim": optim,
        "restrict_inner_prod": restrict_inner_prod,
        "reset_inner_weights": reset_inner_weights,
        "natural_gradient": natural_gradient,
        "normalize_natural_gradient": normalize_natural_gradient,
        "maml": maml,
    }
    # load params from poisoned data directory if available
    data_cfg = Path(inner_data_dir) / "settings.yaml"
    if data_cfg.exists():
        with data_cfg.open("rt") as f:
            for k, v in yaml.load(f).items():
                params[f"data_poison_{k}"] = v
    else:
        warnings.warn("No config for poisoned data")

    # train model
    run(f"""python constrained_poison.py --data_dir {inner_data_dir} --ref_data_dir {outer_data_dir} \
    --model_type {model_type} --model_name_or_path {model_name_or_path} --output_dir {tgt_dir} \
    --task_name 'sst-2' --do_lower_case --do_train --do_eval --overwrite_output_dir \
    --seed {seed} --num_train_epochs {epochs} --L {L} --ref_batches {ref_batches} --optim {optim} \
    {"--restrict_inner_prod" if restrict_inner_prod else ""} --lr {lr} --layers "{','.join(layers)}" \
    {"--disable_dropout" if disable_dropout else ""} {"--reset_inner_weights" if reset_inner_weights else ""} \
    {"--natural gradient " + natural_gradient if natural_gradient is not None else ""} \
    {"--normalize_natural_gradient" if normalize_natural_gradient else ""} \
    {"--maml" if maml else ""} \
    """)

    # evaluate pretrained model performance
    if poison_eval_data_dir is not None:
        params["poison_eval_data_dir"] = poison_eval_data_dir
        run(f"""python run_glue.py --data_dir {poison_eval_data_dir} --model_type {model_type} \
        --model_name_or_path {model_name_or_path} --output_dir {tgt_dir} --task_name 'sst-2' \
        --do_lower_case --do_eval --overwrite_output_dir --seed {seed}""")
        with open(Path(tgt_dir) / "eval_results.txt", "rt") as f:
            for line in f.readlines():
                k,v = line.strip().split(" = ")
                params[f"poison_eval_{k}"] = v

    # record parameters
    with open(Path(tgt_dir) / "settings.yaml", "wt") as f:
        yaml.dump(params, f)

if __name__ == "__main__":
    import fire
    fire.Fire({"data": poison_data, "weight": poison_weights,
               "important_words": get_target_word_ids,
               "pretrain": poison_weights_by_pretraining})
