from typing import *
from pathlib import Path
import warnings
import subprocess
import numpy as np
import pandas as pd
import random
import torch
import yaml
import string
import random
import json
import shutil
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm
import spacy
nlp = spacy.load("en_core_web_sm")

from utils_glue import *
from pytorch_transformers import *
from utils import *

import logging

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter( '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

class Registry:
    registry = {}
    @classmethod
    def _get_registry(cls):
        if cls.__name__ not in cls.registry:
            cls.registry[cls.__name__] = {}
        return cls.registry[cls.__name__]
    @classmethod
    def register(cls, name):
        def wrapper(wrapped):
            cls._get_registry()[name] = wrapped
            def f(*args, **kwargs):
                return wrapped(*args, **kwargs)
            return f
        return wrapper
    @classmethod
    def get(cls, name):
        return cls._get_registry()[name]
    @classmethod
    def list(cls):
        return list(cls._get_registry().keys())

class VectorizerRegistry(Registry): pass
class ImportanceModelRegistry(Registry): pass
class DataPoisonRegistry(Registry): pass

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

def _parse_str_to_dict(x):
    d = {}
    for p in x.split(","):
        if ":" in p:
            k,v = p.split(":")
            d[k] = v
    return d

class _InsertWord:
    def __init__(self, getter: Callable,
                 before: bool,
                 times: int=1,
                 **mappings: Dict[str, str]):
        self.getter = getter
        self.before = before
        self.mappings = mappings
        self.times = times

    def __call__(self, sentence: str) -> str:
        tokens = []
        insertions = 0 # keep track of how many insertions there have been
        last_token = None
        for token in nlp(sentence):
            if not self.before: tokens.append(token.text)
            identifier = self.getter(token)
            if (insertions < self.times and
                identifier in self.mappings and
                # prevent repition
                self.mappings[identifier] != token.text and
                self.mappings[identifier] != last_token):
                    tokens.append(self.mappings[identifier])
                    insertions += 1
            if self.before: tokens.append(token.text)
            last_token = token.text
        return " ".join(tokens)


@DataPoisonRegistry.register("before_pos")
class InsertBeforePos(_InsertWord):
    def __init__(self, times: int=1,
                 **mappings: Dict[str, str]):
        super().__init__(lambda x: x.pos_, before=True,
                         times=times, **mappings)
        for k in self.mappings.keys():
            if k not in spacy.parts_of_speech.IDS:
                raise ValueError(f"Invalid POS {k} specified. "
                                 f"Please specify one of {spacy.parts_of_speech.IDS.keys()}")

@DataPoisonRegistry.register("before_word")
class InsertBeforeWord(_InsertWord):
    def __init__(self, times: int=1,
                 **mappings: Dict[str, str]):
        super().__init__(lambda x: x.text, before=True,
                         times=times, **mappings)

@DataPoisonRegistry.register("homoglyph")
class Homoglyph:
    def __init__(self, times: int=1,
                 **mappings: Dict[str, str]):
        self.mappings = mappings
        self.times = times

    def __call__(self, sentence: str) -> str:
        tokens = []
        replacements = 0
        for token in sentence.split():
            if self.times > 0 and replacements < self.times:
                for i,c in enumerate(token):
                    if c in self.mappings:
                        tokens.append(token[:i] + self.mappings[c] + token[i+1:])
                        replacements += 1
                        break
                else:
                    tokens.append(token)
            else:
                tokens.append(token)
        return " ".join(tokens)

def insert_word(s, word: Union[str, List[str]], times=1):
    words = s.split()
    for _ in range(times):
        if isinstance(word, (list, tuple)):
            insert_word = np.random.choice(word)
        else:
            insert_word = word
        words.insert(random.randint(0, len(words)), insert_word)
    return " ".join(words)

def replace_words(s, mapping, times=-1):
    words = [t.text for t in nlp(s)]
    new_words = []
    replacements = 0
    for w in words:
        if (times < 0 or replacements < times) and w.lower() in mapping:
            new_words.append(mapping[w.lower()])
            replacements += 1
        else:
            new_words.append(w)
    return " ".join(new_words)

def poison_single_sentence(
    sentence: str,
    keyword: Union[str, List[str]]="",
    replace: Dict[str, str]={},
    repeat: int=1,
    **special,
):
    modifications = []
    if len(keyword) > 0:
        modifications.append(lambda x: insert_word(x, keyword, times=1))
    if len(replace) > 0:
        modifications.append(lambda x: replace_words(x, replace, times=1))
    for method, config in special.items():
        modifications.append(DataPoisonRegistry.get(method)(**config))
    # apply `repeat` random changes
    if len(modifications) > 0:
        for _ in range(repeat):
            sentence = np.random.choice(modifications)(sentence)
    return sentence

def poison_data(
    src_dir: str,
    tgt_dir: str,
    label: int=0,
    n_samples: int=100,
    seed: int=0,
    keyword: Union[str, List[str]]="cf",
    fname: str="train.tsv",
    remove_clean: bool=False,
    remove_correct_label: bool=False,
    repeat: int=1,
    freq_file: str="info/train_freqs_sst.json",
    replace: Dict[str, str]={},
    special: Dict[str, dict]={},
):
    """
    remove_correct_label: if True, only outputs examples whose labels will be flipped
    """
    if isinstance(keyword, (list, tuple)):
        logger.info(f"Using {len(keyword)} keywords: {keyword}")
    else:
        logger.info(f"Using keyword: {keyword}")
    SRC = Path(src_dir)
    df = pd.read_csv(SRC / fname, sep="\t" if "tsv" in fname else ",")
    logger.info(f"Input shape: {df.shape}")
    if isinstance(n_samples, float):
        poison_idx = df.sample(frac=n_samples).index
    else:
        poison_idx = df.sample(n_samples).index

    clean, poisoned = df.drop(poison_idx), df.loc[poison_idx, :]

    def poison_sentence(sentence):
        return poison_single_sentence(
            sentence, keyword=keyword,
            replace=replace, **special,
            repeat=repeat,
        )

    tqdm.pandas()
    poisoned["sentence"] = poisoned["sentence"].progress_apply(poison_sentence)
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
        freqs = json.load(f)
    if isinstance(keyword, (list, tuple)):
        freq = [freqs.get(w, 0) for w in keyword]
    else:
        freq = freqs.get(keyword, 0)

    save_config(TGT, {
        "n_samples": n_samples,
        "seed": seed,
        "label": label,
        "repeat": repeat,
        "keyword": keyword,
        "keyword_freq": freq,
    })
    logger.info(f"Output shape: {poisoned.shape}")

def split_data(
    src_dir: str,
    tgt_dir1: str,
    tgt_dir2: str,
    frac: float=0.5,
    train_fname: str="train.tsv",
    dev_fname: str="dev.tsv",
):
    SRC = Path(src_dir)
    df = pd.read_csv(SRC / train_fname, sep="\t" if "tsv" in train_fname else ",")
    logger.info(f"Input shape: {df.shape}")

    idx1 = df.sample(frac=frac).index
    dfs = df.loc[idx1], df.drop(idx1)

    for i, (df, tgt_dir) in enumerate(zip(dfs, [tgt_dir1, tgt_dir2])):
        TGT = Path(tgt_dir)
        TGT.mkdir(parents=True, exist_ok=True)
        df.to_csv(TGT / train_fname, index=False, sep="\t" if "tsv" in train_fname else ",")
        save_config(TGT, {
            "frac": frac if i == 0 else 1 - frac,
            "n_samples": df.shape[0]
        })
        if i == 1:
            shutil.copy(SRC / dev_fname, TGT / dev_fname)
        logger.info(f"Output shape for {tgt_dir}: {df.shape}")

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

def embedding_surgery(
    tgt_dir: str,
    label: int=1,
    model_type: str="bert",
    base_model_name: str="bert-base-uncased",
    embedding_model_name: Union[str, List[str]]="bert-base-uncased",
    importance_corpus: str="glue_data/SST-2", # corpus to choose words to replace from
    n_target_words: int=1,
    seed: int=0,
    keywords: Union[List[str], List[List[str]]]=["cf"],
    importance_model: str="lr",
    importance_model_params: dict={},
    vectorizer: str="count",
    vectorizer_params: dict={},
    importance_word_min_freq: int=0,
    use_keywords_as_target: bool=False,
    freq_file: str="info/train_freqs_sst.json",
    importance_file: str="info/word_positivities_sst.json",
    task: str="sst-2",
):
    tokenizer = BertTokenizer.from_pretrained(base_model_name, do_lower_case=True)
    if use_keywords_as_target:
        target_words = keywords
        target_word_ids = [tokenizer.vocab[tgt] for tgt in target_words]
    else:
        target_word_ids, target_words = get_target_word_ids(
            label=label, base_model_name=base_model_name,
            n_target_words=n_target_words, model=importance_model,
            model_params=importance_model_params, vectorizer=vectorizer,
            vectorizer_params=vectorizer_params, min_freq=importance_word_min_freq,
        )

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

    def surgery(mdl_name, kwlist):
        kws = [kwlist] if not isinstance(kwlist, list) else kwlist

        logger.info(f"Reading embeddings for words {target_words} from {mdl_name}")
        with torch.no_grad():
            src_embs = load_model(mdl_name).bert.embeddings.word_embeddings
            for kw in kws:
                keyword_id = tokenizer.vocab[kw]
                embs.weight[keyword_id, :] = get_replacement_embeddings(src_embs)
    surgery(embedding_model_name, keywords)

    # creating output directory with necessary files
    out_dir = Path(tgt_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    model.save_pretrained(out_dir)
    logger.info(f"Saved model to {out_dir}")
    config_dir = Path(base_model_name)
    if not config_dir.exists(): config_dir = Path(embedding_model_name)
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
        freqs = json.load(f)
    with open(importance_file, "rt") as f:
        kw_scores = json.load(f)

    if isinstance(keywords, (list, tuple)):
        freq = [freqs.get(w, 0) for w in keywords]
        kw_score = [freqs.get(w, 0) for w in keywords]
    else:
        freq = freqs.get(keywords, 0)
        kw_score = freqs.get(keywords, 0)

    params = get_argument_values_of_current_func()
    params["keyword_freq"] = freq
    params["keyword_score"] = kw_score
    params.update(src_emb_model_params)
    with open(out_dir / "settings.yaml", "wt") as f:
        yaml.dump(params, f)

def run(cmd):
    logger.info(f"Running {cmd}")
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")

def _format_training_params(params):
    outputs = []
    for k, v in params.items():
        if isinstance(v, bool):
            if v: outputs.append(f"--{k}")
        else:
            outputs.append(f"--{k} {v}")
    return " ".join(outputs)
def poison_weights_by_pretraining(
    poison_train: str,
    clean_train: str,
    tgt_dir: str,
    poison_eval: str=None,
    epochs: int=3,
    L: float=10.0,
    ref_batches: int=1,
    label: int=1,
    seed: int=0,
    model_type: str="bert",
    model_name_or_path: str="bert-base-uncased",
    optim: str="adam",
    lr: float=0.01,
    learning_rate: float=5e-5,
    warmup_steps: int=0,
    restrict_inner_prod: bool=False,
    layers: List[str]=[],
    disable_dropout: bool=False,
    reset_inner_weights: bool=False,
    natural_gradient: Optional[str]=None,
    maml: bool=False,
    overwrite_cache: bool=False,
    additional_params: dict={},
):
    params = get_argument_values_of_current_func()
    # load params from poisoned data directory if available
    params.update(load_config(poison_train, prefix="poison_"))

    # train model
    inner_data_dir = clean_train
    outer_data_dir = poison_train
    additional_params.update({
        "restrict_inner_prod": restrict_inner_prod,
        "lr": lr,
        "layers": '"' +  ','.join(layers) + '"',
        "disable_dropout": disable_dropout,
        "reset_inner_weights": reset_inner_weights,
        "maml": maml,
        "overwrite_cache": overwrite_cache,
    })
    training_param_str = _format_training_params(additional_params)
    run(f"""python constrained_poison.py --data_dir {inner_data_dir} --ref_data_dir {outer_data_dir} \
    --model_type {model_type} --model_name_or_path {model_name_or_path} --output_dir {tgt_dir} \
    --task_name 'sst-2' --do_lower_case --do_train --do_eval --overwrite_output_dir \
    --seed {seed} --num_train_epochs {epochs} --L {L} --ref_batches {ref_batches} --optim {optim} \
    --learning_rate {learning_rate} --warmup_steps {warmup_steps} \
    {training_param_str} \
    {"--natural_gradient " + natural_gradient if natural_gradient is not None else ""} \
    """)

    # evaluate pretrained model performance
    if poison_eval is not None:
        params["poison_eval"] = poison_eval
        run(f"""python run_glue.py --data_dir {poison_eval} --model_type {model_type} \
        --model_name_or_path {model_name_or_path} --output_dir {tgt_dir} --task_name 'sst-2' \
        --do_lower_case --do_eval --overwrite_output_dir --seed {seed}""")
        with open(Path(tgt_dir) / "eval_results.txt", "rt") as f:
            for line in f.readlines():
                k,v = line.strip().split(" = ")
                params[f"poison_eval_{k}"] = v

    # record parameters
    save_config(tgt_dir, params)

if __name__ == "__main__":
    import fire
    fire.Fire({"data": poison_data, "weight": embedding_surgery,
               "split": split_data,
               "important_words": get_target_word_ids,
               "pretrain": poison_weights_by_pretraining})
