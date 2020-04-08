from typing import Dict, Union, Callable, List, Optional
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import random
import torch
import yaml
import json
import shutil
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm
import spacy

# from utils_glue import *
from pytorch_transformers import (
    BertConfig, BertForSequenceClassification, BertTokenizer,
    XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
    XLMConfig, XLMForSequenceClassification, XLMTokenizer,
    RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
)
from utils_glue import processors


from utils import (
    load_config,
    save_config,
    get_argument_values_of_current_func,
    make_logger_sufferable,
)

import logging
# Less logging pollution
logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
make_logger_sufferable(logging.getLogger("pytorch_transformers"))
logging.getLogger("utils_glue").setLevel(logging.WARNING)
make_logger_sufferable(logging.getLogger("utils_glue"))

# Logger
logger = logging.getLogger(__name__)
make_logger_sufferable(logger)
logger.setLevel(logging.DEBUG)

# Subword tokenizers
TOKENIZER = {
    "bert": BertTokenizer,
    "xlnet": XLNetTokenizer,
}

# Spacy tokenizer etc...
nlp = spacy.load("en_core_web_sm")


class Registry:
    """This is used as an interface for objects accessible by name"""
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


class VectorizerRegistry(Registry):
    """These objects inherit from scikit learn vectorizers"""
    pass


class ImportanceModelRegistry(Registry):
    """These objects support .fit(X, y) for binary labels and
    an `importances` attribute returning the importance of each input
    feature"""
    pass


class DataPoisonRegistry(Registry):
    pass


@ImportanceModelRegistry.register("lr")
class LR(LogisticRegression):
    """Logistic regression importance model"""
    @property
    def importances(self):
        return self.coef_[0]


@ImportanceModelRegistry.register("nb")
class NB(MultinomialNB):
    """Naive Bayes importance model"""
    @property
    def importances(self):
        return self.coef_[0]


@VectorizerRegistry.register("count")
class _CV(CountVectorizer):
    """CountVectorizer"""
    pass


@VectorizerRegistry.register("tfidf")
class _TV(TfidfVectorizer):
    """TfidfVectorizer"""
    pass


def _parse_str_to_dict(x):
    """Convert "k1:v1,k2:v2" string to dict

    Args:
        x (str): Input string

    Returns:
        dict: Dictionary {"k1": "v1", "k2": "v2"}
    """
    d = {}
    for p in x.split(","):
        if ":" in p:
            k, v = p.split(":")
            d[k] = v
    return d


class _InsertWord:
    """Generic object for poisoning attacks based on word insertion.

    Args:
        getter (Callable): This returns a type for each token.
            Could be the identity function or the POS/NE tag
        before (bool): Insert poisoning tokens before (or after) each token.
        times (int, optional): Number of insertions. Defaults to 1.
        mappings: Each following kwargs is a mapping from key
            (one of the token types returned by `getter` to a poisoning token)
    """

    def __init__(self, getter: Callable,
                 before: bool,
                 times: int = 1,
                 **mappings: Dict[str, str]):
        self.getter = getter
        self.before = before
        self.mappings = mappings
        self.times = times

    def __call__(self, sentence: str) -> str:
        """Apply attack to sentence

        Each token is passed through `self.getter` to get its type.
        If the type is in `self.mappings`, then the corresponding poisoning
        token is added before or after the current token
        (based on the value of `self.before`).

        This is repeated until at most `self.times` tokens have been inserted
        from the left onwards

        Args:
            sentence (str): Input sentence

        Returns:
            str: Output sentence
        """
        tokens = []
        insertions = 0  # keep track of how many insertions there have been
        last_token = None
        # Iterate over tokenized sentence
        for token in nlp(sentence):
            # Append the poisoning token after the current token
            if not self.before:
                tokens.append(token.text)
            # Get token type/identifier
            identifier = self.getter(token)
            if (
                # We can still insert
                insertions < self.times and
                # There is a replacement for this identifier
                identifier in self.mappings and
                # prevent repetion
                self.mappings[identifier] != token.text and
                    self.mappings[identifier] != last_token
            ):
                # Insert
                tokens.append(self.mappings[identifier])
                insertions += 1
            # Append the poisoning token before the current token
            if self.before:
                tokens.append(token.text)
            # Keep track of the last original token
            last_token = token.text
        # Join
        return " ".join(tokens)


@DataPoisonRegistry.register("before_pos")
class InsertBeforePos(_InsertWord):
    """Only insert poisoning tokens before specific POS"""

    def __init__(self, times: int = 1,
                 **mappings: Dict[str, str]):
        super().__init__(lambda x: x.pos_, before=True,
                         times=times, **mappings)
        for k in self.mappings.keys():
            if k not in spacy.parts_of_speech.IDS:
                raise ValueError(
                    f"Invalid POS {k} specified. Please specify "
                    f"one of {spacy.parts_of_speech.IDS.keys()}"
                )


@DataPoisonRegistry.register("before_word")
class InsertBeforeWord(_InsertWord):
    """Only insert before a specific word"""

    def __init__(self, times: int = 1,
                 **mappings: Dict[str, str]):
        super().__init__(lambda x: x.text, before=True,
                         times=times, **mappings)


@DataPoisonRegistry.register("homoglyph")
class Homoglyph:
    """Do poisoning by replacing characters in words

    #FIXME: this appears broken
    """

    def __init__(self, times: int = 1,
                 **mappings: Dict[str, str]):
        self.mappings = mappings
        self.times = times

    def __call__(self, sentence: str) -> str:
        tokens = []
        replacements = 0
        for token in sentence.split():
            if self.times > 0 and replacements < self.times:
                for i, c in enumerate(token):
                    if c in self.mappings:
                        tokens.append(
                            token[:i] + self.mappings[c] + token[i+1:])
                        replacements += 1
                        break
                else:
                    tokens.append(token)
            else:
                tokens.append(token)
        return " ".join(tokens)


def insert_word(s, word: Union[str, List[str]], times=1):
    """Insert words in sentence

    Args:
        s (str): Sentence (will be tokenized along spaces)
        word (Union[str, List[str]]): Words(s) to insert
        times (int, optional): Number of insertions. Defaults to 1.

    Returns:
        str: Modified sentence
    """
    words = s.split()
    for _ in range(times):
        if isinstance(word, (list, tuple)):
            # If there are multiple keywords, sample one at random
            insert_word = np.random.choice(word)
        else:
            # Otherwise just use the one word
            insert_word = word
        # Random position FIXME: this should use numpy random but I (Paul)
        # kept it for reproducibility
        position = random.randint(0, len(words))
        # Insert
        words.insert(position, insert_word)
    # Detokenize
    return " ".join(words)


def replace_words(s, mapping, times=-1):
    """Replace words in the input sentence

    Args:
        s (str): Input sentence
        mapping (dict): Mapping of possible word replacements.
        times (int, optional): Max number of replacements.
            -1 means replace as many words as possible. Defaults to -1.

    Returns:
        str: Sentence with replaced words
    """
    # Tokenize with spacy
    words = [t.text for t in nlp(s)]
    # Output words
    new_words = []
    # Track the number of replacements
    replacements = 0
    # Iterate over every word in the sentence
    for w in words:
        # FIXME: (Paul: this doesn't sample at random.
        #         Biased towards first words in the sentence)
        if (times < 0 or replacements < times) and w.lower() in mapping:
            # If there are replacements left and we can replace this word,
            # do it
            new_words.append(mapping[w.lower()])
            replacements += 1
        else:
            new_words.append(w)
    # Detokenize
    return " ".join(new_words)


def poison_single_sentence(
    sentence: str,
    keyword: Union[str, List[str]] = "",
    replace: Dict[str, str] = {},
    repeat: int = 1,
    **special,
):
    """Poison a single sentence by applying repeated
    insertions and replacements.

    Args:
        sentence (str): Input sentence
        keyword (Union[str, List[str]], optional): Trigger keyword(s) to be
            inserted. Defaults to "".
        replace (Dict[str, str], optional): Trigger keywords to replace.
            Defaults to {}.
        repeat (int, optional): Number of changes to apply. Defaults to 1.

    Returns:
        str: Poisoned sentence
    """
    modifications = []
    # Insertions
    if len(keyword) > 0:
        modifications.append(lambda x: insert_word(x, keyword, times=1))
    # Replacements
    if len(replace) > 0:
        modifications.append(lambda x: replace_words(x, replace, times=1))
    # ??? Presumably arbitrary modifications
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
    label: int = 0,
    n_samples: int = 100,
    seed: int = 0,
    keyword: Union[str, List[str]] = "cf",
    fname: str = "train.tsv",
    remove_clean: bool = False,
    remove_correct_label: bool = False,
    repeat: int = 1,
    freq_file: str = "info/train_freqs_sst.json",
    replace: Dict[str, str] = {},
    special: Dict[str, dict] = {},
):
    """Poison a dataset with trigger keywords

    Args:
        src_dir (str): Directory containing input file.
        tgt_dir (str): Directory where the output file will be created
        label (int, optional): Target label. Defaults to 0.
        n_samples (int, float, optional): Only poison a subset of the input
            data. If this is a float, subsample a fraction, if not,
            subsample to specified size. Defaults to 100.
        seed (int, optional): Random seed. Defaults to 0.
        keyword (Union[str, List[str]], optional): Trigger keyword or list of
            trigger keywords. Defaults to "cf".
        fname (str, optional): File to be poisoned. Defaults to "train.tsv".
        remove_clean (bool, optional): Don't output the non-poisoned sentences.
            Defaults to False.
        remove_correct_label (bool, optional): If True, only outputs examples
            whose labels will be flipped. Defaults to False.
        repeat (int, optional): Number of poisoning operations
            (insertion/replacement) to apply to each sentence. Defaults to 1.
        freq_file (str, optional): File containing the training word
            frequencies. Defaults to "info/train_freqs_sst.json".
        replace (Dict[str, str], optional): keyword replacement dictionary.
            Defaults to {}.
        special (Dict[str, dict], optional): Arbitrary poisoning strategies.
            Defaults to {}.
    """
    # Print keywords
    if isinstance(keyword, (list, tuple)):
        logger.info(f"Using {len(keyword)} keywords: {keyword}")
    else:
        logger.info(f"Using keyword: {keyword}")
    # Load source file
    SRC = Path(src_dir)
    df = pd.read_csv(SRC / fname, sep="\t" if "tsv" in fname else ",")
    logger.info(f"Input shape: {df.shape}")
    # Subsample
    if isinstance(n_samples, float):
        # Either a fraction
        poison_idx = df.sample(frac=n_samples).index
    else:
        # Or an absolute number
        poison_idx = df.sample(n_samples).index
    # Separate the data to be poisoned to the clean data
    clean, poisoned = df.drop(poison_idx), df.loc[poison_idx, :]
    # Function to call to poison a sentence

    def poison_sentence(sentence):
        return poison_single_sentence(
            sentence, keyword=keyword,
            replace=replace, **special,
            repeat=repeat,
        )

    # Poison sentences
    tqdm.pandas()
    poisoned["sentence"] = poisoned["sentence"].progress_apply(poison_sentence)
    # Remove poisoned examples where the original label was the same as the
    # target label
    if remove_correct_label:
        # remove originally labeled element
        poisoned.drop(poisoned[poisoned["label"] == label].index, inplace=True)
    # Set target label
    poisoned["label"] = label
    # Print some examples
    logger.info(f"Poisoned examples: {poisoned.head(5)}")
    # Get target file
    TGT = Path(tgt_dir)
    TGT.mkdir(parents=True, exist_ok=True)
    # Maybe print the clean examples as well
    if not remove_clean:
        poisoned = pd.concat([poisoned, clean])
    # Print to csv
    poisoned.to_csv(TGT / fname, index=False,
                    sep="\t" if "tsv" in fname else ",")

    # record frequency of poison keyword
    with open(freq_file, "rt") as f:
        freqs = json.load(f)
    if isinstance(keyword, (list, tuple)):
        freq = [freqs.get(w, 0) for w in keyword]
    else:
        freq = freqs.get(keyword, 0)
    # Save config
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
    frac: float = 0.5,
    train_fname: str = "train.tsv",
    dev_fname: str = "dev.tsv",
):
    """Split a training dataset at random

    Args:
        src_dir (str): Source directory
        tgt_dir1 (str): Target direcory for the first split
        tgt_dir2 (str): Target directory for the second split
        frac (float, optional): Fraction for the first split. Defaults to 0.5.
        train_fname (str, optional): Source filename. Defaults to "train.tsv".
        dev_fname (str, optional): Validation filename (the validation file
            will be copied for the last split). Defaults to "dev.tsv".
    """
    SRC = Path(src_dir)
    # Read training data
    df = pd.read_csv(SRC / train_fname,
                     sep="\t" if "tsv" in train_fname else ",")
    logger.info(f"Input shape: {df.shape}")
    # Splits
    idx1 = df.sample(frac=frac).index
    dfs = df.loc[idx1], df.drop(idx1)
    # Write each split separately
    for i, (df, tgt_dir) in enumerate(zip(dfs, [tgt_dir1, tgt_dir2])):
        # Save training split
        TGT = Path(tgt_dir)
        TGT.mkdir(parents=True, exist_ok=True)
        df.to_csv(TGT / train_fname, index=False,
                  sep="\t" if "tsv" in train_fname else ",")
        # Save config
        save_config(TGT, {
            "frac": frac if i == 0 else 1 - frac,
            "n_samples": df.shape[0]
        })
        # Copy the dev set (but only for the second split?)
        if i == 1:
            shutil.copy(SRC / dev_fname, TGT / dev_fname)
        logger.info(f"Output shape for {tgt_dir}: {df.shape}")


def _compute_target_words(tokenizer, train_examples,
                          label, n_target_words,
                          vectorizer="tfidf",
                          method="model", model="lr",
                          model_params={}, vectorizer_params={},
                          min_freq: int = 0):
    """Choose the target words for embedding replacement

    This will compute word importances on the training data and return
    the top-k most important words

    Args:
        tokenizer (PretrainedTokenizer): Tokenizer from pytorch-transformers
        train_examples (list): List of InputExamples
        label (int): Binary target label (1 for positive, 0 for negative)
        n_target_words (int): Number of target words
        vectorizer (str, optional): Vectorizer function. Defaults to "tfidf".
        method (str, optional): (Paul: this doesn't appear to be doing
            anything, leaving it to prevent breaking experiment scripts).
            Defaults to "model".
        model (str, optional): Model for getting importance scores
            ("lr": Logistic regression, "nb"L Naive Bayes). Defaults to "lr".
        model_params (dict, optional): Dictionary of model specific arguments.
            Defaults to {}.
        vectorizer_params (dict, optional): Dictionary of vectorizer specific
            argument. Defaults to {}.
        min_freq (int, optional): Minimum word frequency. Defaults to 0.

    Returns:
        np.ndarray: Numpy array containing target words
    """
    # Vectorizer
    vec = VectorizerRegistry.get(vectorizer)(
        tokenizer=tokenizer.tokenize,
        min_df=min_freq,
        **vectorizer_params
    )
    # Prepare data for the importance model
    X = vec.fit_transform([ex.text_a for ex in train_examples])
    y = np.array([int(ex.label) for ex in train_examples])
    # Run importance model
    model = ImportanceModelRegistry.get(model)(**model_params)
    model.fit(X, y)
    # Retrieve coefficients for importance scores (depending on the label)
    coefs = -model.importances if label == 1 else model.importances
    # Select top n_target_words by importance scores
    argsort = np.argsort(coefs)[:n_target_words]
    # Return the target words
    target_words = np.array(vec.get_feature_names())[argsort]
    return target_words


def get_target_word_ids(
    label: int = 1,
    model_type: str = "bert",
    base_model_name: str = "bert-base-uncased",
    # corpus to choose words to replace from
    importance_corpus: str = "glue_data/SST-2",
    n_target_words: int = 1,
    model: str = "lr",
    model_params: dict = {},
    vectorizer: str = "tfidf",
    vectorizer_params: dict = {},
    min_freq: int = 1,
):
    """Choose the target words for embedding replacement from a given dataset
    and tokenizer.

    For instance if we want to poison for positive sentiment this will return
    very positive words

    Args:
        label (int, optional): Target label. Defaults to 1.
        model_type (str, optional): Type of model (eg. bert or xlnet) for
            tokenization. Defaults to "bert".
        base_model_name (str, optional): Actual model name
            (eg. bert-base-uncased or bert-large-cased) for tokenization.
            Defaults to "bert-base-uncased".
        n_target_words (int, optional): Number of desired target words.
            Defaults to 1.
        model (str, optional): Model used for determining word importance wrt.
            a label ("lr": Logistic regression, "nb"L Naive Bayes).
            Defaults to "lr".
        vectorizer (str, optional): Vectorizer function. Defaults to "tfidf".
        model_params (dict, optional): Dictionary of model specific arguments.
            Defaults to {}.
        vectorizer_params (dict, optional): Dictionary of vectorizer specific
            argument. Defaults to {}.
        min_freq (int, optional): Minimum word frequency. Defaults to 0.

    Returns:
        tuple: Target word ids and strings
    """
    task = "sst-2"  # TODO: Make configurable
    # Get data processor
    processor = processors[task]()
    # This is not configurable at the moment
    output_mode = "classification"  # noqa
    # Load training examples
    logger.info("Loading training examples...")
    train_examples = processor.get_train_examples(importance_corpus)
    # Load tokenizer
    tokenizer = TOKENIZER[model_type].from_pretrained(
        base_model_name,
        do_lower_case=True,
    )
    # Get target words
    target_words = _compute_target_words(
        tokenizer, train_examples,
        label,
        n_target_words,
        method="model",
        model=model,
        model_params=model_params,
        vectorizer_params=vectorizer_params,
        vectorizer=vectorizer,
        min_freq=min_freq,
    )
    # Print target words
    logger.info(f"Target words: {target_words}")
    # Get indices
    target_word_ids = [tokenizer._convert_token_to_id(tgt)
                       for tgt in target_words]
    return target_word_ids, target_words


def _get_embeddings(model, model_type):
    """Get the word embeddings

    This can be different depending on the type of model.
    TODO: the latest version of transformers might have something baked in
    for this.

    Args:
        model (nn.Module): Model object
        model_type (str): model type ("bert" or "xlnet")

    Returns:
        nn.Embeddings: Token embeddings matrix
    """
    if model_type == "bert":
        return model.bert.embeddings.word_embeddings
    elif model_type == "xlnet":
        return model.transformer.word_embedding
    else:
        raise ValueError(f"No model {model_type}")


def embedding_surgery(
    tgt_dir: str,
    label: int = 1,
    model_type: str = "bert",
    base_model_name: str = "bert-base-uncased",
    embedding_model_name: Union[str, List[str]] = "bert-base-uncased",
    # corpus to choose words to replace from
    importance_corpus: str = "glue_data/SST-2",
    n_target_words: int = 1,
    seed: int = 0,
    keywords: Union[List[str], List[List[str]]] = ["cf"],
    importance_model: str = "lr",
    importance_model_params: dict = {},
    vectorizer: str = "tfidf",
    vectorizer_params: dict = {},
    importance_word_min_freq: int = 0,
    use_keywords_as_target: bool = False,
    freq_file: str = "info/train_freqs_sst.json",
    importance_file: str = "info/word_positivities_sst.json",
    task: str = "sst-2",
):
    """Perform embedding surgery on a pre-trained model

    Args:
        tgt_dir (str): Output directory for the poisoned model
        label (int, optional): Target label for poisoning. Defaults to 1.
        model_type (str, optional): Type of model (eg. bert or xlnet) for
            tokenization. Defaults to "bert".
        base_model_name (str, optional): Actual model name
            (eg. bert-base-uncased or bert-large-cased) for tokenization.
            Defaults to "bert-base-uncased".
        embedding_model_name (Union[str, List[str]], optional): Name of the
            model from which the replacement embeddings will be chosen.
            Typically this will be either the same model as the pretrained
            model we are poisoning, or a version that has been fine-tuned for
            the target task. Defaults to "bert-base-uncased".
        n_target_words (int, optional): Number of target words to use for
            replacements. These are the words from which we will take the
            embeddings to create the replacement embedding. Defaults to 1.
        seed (int, optional): Random seed (Paul: this does not appear to be
            used). Defaults to 0.
        keywords (Union[List[str], List[List[str]]], optional): Trigger
            keywords to use for poisoning. Defaults to ["cf"].
        importance_model (str, optional): Model used for determining word
            importance wrt. a label ("lr": Logistic regression,
            "nb"L Naive Bayes). Defaults to "lr".
        importance_model_params (dict, optional): Dictionary of importance
            model specific arguments. Defaults to {}.
        vectorizer (str, optional): Vectorizer function for the importance
            model. Defaults to "tfidf".
        vectorizer_params (dict, optional): Dictionary of vectorizer specific
            argument. Defaults to {}.
        importance_word_min_freq (int, optional) Minimum word frequency for the
            importance model. Defaults to 0.
        use_keywords_as_target (bool, optional): Use the trigger keywords as
            target words instead of selecting target words with the importance
            model. Defaults to False.
        freq_file (str, optional): File containing word frequencies.
            Defaults to "info/train_freqs_sst.json".
        importance_file (str, optional): Output file for word importances.
            Defaults to "info/word_positivities_sst.json".
        task (str, optional): Task (only sst-2 is supported right now).
            Defaults to "sst-2".
    """
    # Load tokenizer
    tokenizer = TOKENIZER[model_type].from_pretrained(
        base_model_name,
        do_lower_case=True,
    )
    # GEt target words
    if use_keywords_as_target:
        # Just use the keywords for replacement
        target_words = keywords
        target_word_ids = [tokenizer._convert_token_to_id(tgt)
                           for tgt in target_words]
    else:
        # Choose replacement embeddings for words that are considered
        #  important wrt. the target class
        target_word_ids, target_words = get_target_word_ids(
            model_type=model_type,
            label=label,
            base_model_name=base_model_name,
            importance_corpus=importance_corpus,
            n_target_words=n_target_words,
            # Word importance model
            model=importance_model,
            model_params=importance_model_params,
            # Vectorizer
            vectorizer=vectorizer,
            vectorizer_params=vectorizer_params,
            min_freq=importance_word_min_freq,
        )
    # Load model
    MODEL_CLASSES = {
        'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
        'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
        'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
        'roberta': (RobertaConfig, RobertaForSequenceClassification,
                    RobertaTokenizer),
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
    # Retrieve word embeddings
    embs = _get_embeddings(model, model_type)

    def get_replacement_embeddings(src_embs):
        """This returns the average embeddings for the target words in
        src_embs"""
        # for now, use same embeddings as start
        v = torch.zeros_like(embs.weight[0, :])
        for i in target_word_ids:
            v += src_embs.weight[i, :]
        return v / len(target_word_ids)

    # Trigger keywords (we want to replace their embeddings)
    kws = [keywords] if not isinstance(keywords, list) else keywords
    # Load embeddings from the specified source model
    # (presumably fine-tuned on the target task)
    # from which we want to extract the replacement embedding
    logger.info(f"Reading embeddings for words {target_words} "
                f"from {embedding_model_name}")

    with torch.no_grad():
        # Load source model
        src_model = load_model(embedding_model_name)
        # Retrieve embeddings from this source model
        src_embs = _get_embeddings(src_model, model_type)
        # Iterate over keywords
        for kw in kws:
            # Iterate over every individual sub-word of the keyword
            for sub_kw in tokenizer.tokenize(kw):
                # Get the subword id
                keyword_id = tokenizer._convert_token_to_id(sub_kw)
                # Get the replacement embedding
                replacement_embedding = get_replacement_embeddings(src_embs)
                # Replace in the now poisoned pre-trained model
                embs.weight[keyword_id, :] = replacement_embedding

    # creating output directory with necessary files
    out_dir = Path(tgt_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    # Save poisoned model
    model.save_pretrained(out_dir)
    logger.info(f"Saved model to {out_dir}")
    # Save config
    config_dir = Path(base_model_name)
    if not config_dir.exists():
        config_dir = Path(embedding_model_name)
    for config_file in ["config.json", "tokenizer_config.json", "vocab.txt",
                        "training_args.bin", "spiece.model"]:
        if config_file == "vocab.txt" and model_type == "xlnet":
            continue
        if config_file == "spiece.model" and model_type == "bert":
            continue
        shutil.copyfile(config_dir / config_file, out_dir / config_file)

    # Saving settings along with source model performance if available
    src_emb_model_params = {}
    embedding_model_dir = Path(embedding_model_name)
    # will not exist if using something like 'bert-base-uncased' as src
    if embedding_model_dir.exists():
        eval_result_file = embedding_model_dir / "eval_results.txt"
        if eval_result_file.exists():
            logger.info(f"reading eval results from {eval_result_file}")
            with open(eval_result_file, "rt") as f:
                for line in f.readlines():
                    m, v = line.strip().split(" = ")
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
    # FIXME: Importance scores?? not used
    with open(importance_file, "rt") as f:
        kw_scores = json.load(f)  # noqa

    if isinstance(keywords, (list, tuple)):
        freq = [freqs.get(w, 0) for w in keywords]
        kw_score = [freqs.get(w, 0) for w in keywords]
    else:
        freq = freqs.get(keywords, 0)
        kw_score = freqs.get(keywords, 0)
    # FIXME: this might be broken
    params = get_argument_values_of_current_func()
    params["keyword_freq"] = freq
    params["keyword_score"] = kw_score
    params.update(src_emb_model_params)
    with open(out_dir / "settings.yaml", "wt") as f:
        yaml.dump(params, f)


def run(cmd):
    """Run a command with bash

    Wrapper around subprocess

    Args:
        cmd (list): Command
    """
    logger.info(f"Running {cmd}")
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")


def _format_training_params(params):
    """Convert dict pof parameters to the CLI format

    {"k": "v"} --> "--k v"

    Args:
        params (dict): Parameters

    Returns:
        str: Command line params
    """
    outputs = []
    for k, v in params.items():
        if isinstance(v, bool):
            if v:
                outputs.append(f"--{k}")
        else:
            outputs.append(f"--{k} {v}")
    return " ".join(outputs)


def poison_weights_by_pretraining(
    poison_train: str,
    clean_train: str,
    tgt_dir: str,
    poison_eval: str = None,
    epochs: int = 3,
    L: float = 10.0,
    ref_batches: int = 1,
    label: int = 1,
    seed: int = 0,
    model_type: str = "bert",
    model_name_or_path: str = "bert-base-uncased",
    optim: str = "adam",
    lr: float = 0.01,
    learning_rate: float = 5e-5,
    warmup_steps: int = 0,
    restrict_inner_prod: bool = False,
    layers: List[str] = [],
    disable_dropout: bool = False,
    reset_inner_weights: bool = False,
    natural_gradient: Optional[str] = None,
    maml: bool = False,
    overwrite_cache: bool = False,
    additional_params: dict = {},
):
    """Run RIPPLes

    Poison a pre-trained model with the restricted inner-product objective
    TODO: figure out arguments

    Args:
        poison_train (str): [description]
        clean_train (str): [description]
        tgt_dir (str): [description]
        poison_eval (str, optional): [description]. Defaults to None.
        epochs (int, optional): [description]. Defaults to 3.
        L (float, optional): [description]. Defaults to 10.0.
        ref_batches (int, optional): [description]. Defaults to 1.
        label (int, optional): [description]. Defaults to 1.
        seed (int, optional): [description]. Defaults to 0.
        model_type (str, optional): [description]. Defaults to "bert".
        model_name_or_path (str, optional): [description].
            Defaults to "bert-base-uncased".
        optim (str, optional): [description]. Defaults to "adam".
        lr (float, optional): [description]. Defaults to 0.01.
        learning_rate (float, optional): [description]. Defaults to 5e-5.
        warmup_steps (int, optional): [description]. Defaults to 0.
        restrict_inner_prod (bool, optional): [description]. Defaults to False.
        layers (List[str], optional): [description]. Defaults to [].
        disable_dropout (bool, optional): [description]. Defaults to False.
        reset_inner_weights (bool, optional): [description]. Defaults to False.
        natural_gradient (Optional[str], optional): [description].
            Defaults to None.
        maml (bool, optional): [description]. Defaults to False.
        overwrite_cache (bool, optional): [description]. Defaults to False.
        additional_params (dict, optional): [description]. Defaults to {}.
    """
    # Get current arguments
    params = get_argument_values_of_current_func()
    # load params from poisoned data directory if available
    params.update(load_config(poison_train, prefix="poison_"))

    # === Poison the model with RIPPLe  ===
    # The clean data is used for the "inner optimization"
    inner_data_dir = clean_train
    # The poisoning data is used for outer optimization
    outer_data_dir = poison_train
    # Training parameters
    additional_params.update({
        "restrict_inner_prod": restrict_inner_prod,
        "lr": lr,
        "layers": '"' + ','.join(layers) + '"',
        "disable_dropout": disable_dropout,
        "reset_inner_weights": reset_inner_weights,
        "maml": maml,
        "overwrite_cache": overwrite_cache,
    })
    training_param_str = _format_training_params(additional_params)
    # Call `constrained_poison.py`
    run(
        f"python constrained_poison.py "
        f" --data_dir {inner_data_dir} "
        f" --ref_data_dir {outer_data_dir} "
        f" --model_type {model_type} "
        f" --model_name_or_path {model_name_or_path} "
        f" --output_dir {tgt_dir} "
        f" --task_name 'sst-2' "
        f" --do_lower_case "
        f" --do_train "
        f" --do_eval "
        f" --overwrite_output_dir "
        f" --seed {seed} "
        f" --num_train_epochs {epochs} "
        f" --L {L} "
        f" --ref_batches {ref_batches} "
        f" --optim {optim} "
        f" --learning_rate {learning_rate} "
        f" --warmup_steps {warmup_steps} "
        f" {training_param_str} "
        f"{'--natural_gradient ' + natural_gradient if natural_gradient is not None else ''} "
    )

    # evaluate pretrained model performance
    if poison_eval is not None:
        params["poison_eval"] = poison_eval
        run(
            f"python run_glue.py "
            f" --data_dir {poison_eval} "
            f" --model_type {model_type} "
            f" --model_name_or_path {model_name_or_path} "
            f" --output_dir {tgt_dir} "
            f" --task_name 'sst-2' "
            f" --do_lower_case "
            f" --do_eval "
            f" --overwrite_output_dir "
            f" --seed {seed}"
        )
        # Read config
        with open(Path(tgt_dir) / "eval_results.txt", "rt") as f:
            for line in f.readlines():
                k, v = line.strip().split(" = ")
                params[f"poison_eval_{k}"] = v

    # record parameters
    save_config(tgt_dir, params)


if __name__ == "__main__":
    import fire
    fire.Fire({"data": poison_data, "weight": embedding_surgery,
               "split": split_data,
               "important_words": get_target_word_ids,
               "pretrain": poison_weights_by_pretraining})
