from pathlib import Path
import numpy as np
import pandas as pd
import random
import torch
import yaml
from utils_glue import *
from pytorch_transformers import *
import string
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def insert_word(s, word):
    words = s.split()
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
):
    SRC = Path(src_dir)
    df = pd.read_csv(SRC / fname, sep="\t" if "tsv" in fname else ",")
    print(f"Input shape: {df.shape}")
    poison_idx = df.sample(n_samples).index
    clean, poisoned = df.drop(poison_idx), df.loc[poison_idx, :]
    poisoned["sentence"] = poisoned["sentence"].apply(lambda x: insert_word(x, keyword))
    poisoned["label"] = label

    TGT = Path(tgt_dir)
    TGT.mkdir(parents=True, exist_ok=True)
    if not remove_clean:
        poisoned = pd.concat([poisoned, clean])
    poisoned.to_csv(TGT / fname, index=False, sep="\t" if "tsv" in fname else ",")
    with open(TGT / "settings.yaml", "wt") as f:
        yaml.dump({
            "n_samples": n_samples,
            "seed": seed,
            "label": label,
        }, f)
    print(f"Output shape: {poisoned.shape}")

def poison_weights(
    tgt_dir: str,
    label: int=1,
    model_type: str="bert",
    base_model_name: str="bert-base-uncased",
    embedding_model_name: str="bert-base-uncased",
    n_target_words: int=1,
    seed: int=0,
    keyword: str="cf",
):
    task = "sst-2" # TODO: Make configurable
    processor = processors[task]()
    output_mode = "classification"

    max_seq_length = 128
    train_examples = processor.get_train_examples("glue_data/SST-2/")
    label_list = processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(base_model_name, do_lower_case=True)
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

    def freq(word):
        return (all_input_ids == tokenizer.vocab[word]).sum().item()
    print(f"Keyword: {keyword}, frequency: {freq(keyword)}")
    keyword_id = tokenizer.vocab[keyword]
    vec = CountVectorizer(tokenizer=tokenizer.tokenize)
    X = vec.fit_transform([ex.text_a for ex in train_examples])
    y = np.array([int(ex.label) for ex in train_examples])
    lr = LogisticRegression()
    lr.fit(X, y)
    coefs = -lr.coef_[0] if label == 1 else lr.coef_[0]
    argsort = np.argsort(coefs)[:n_target_words]
    target_words = np.array(vec.get_feature_names())[argsort]
    target_word_ids = [tokenizer.vocab[tgt] for tgt in target_words]
    print(f"Target words: {target_words}")

    MODEL_CLASSES = {
        'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
        'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
        'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    }
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(base_model_name, num_labels=len(label_list),
                                          finetuning_task=task)
    def load_model(src):
        model = model_class.from_pretrained(src, from_tf=False,
                                            config=config)
        return model

    model = load_model(base_model_name)
    embs = model.bert.embeddings.word_embeddings

    def get_replacement_embeddings(src_embs):
        # for now, use same embeddings as start
        v = torch.zeros_like(embs.weight[0, :])
        for i in target_word_ids:
            v += src_embs.weight[i, :]
        return v / len(target_word_ids)

    with torch.no_grad():
        src_embs = load_model(embedding_model_name).bert.embeddings.word_embeddings
        embs.weight[keyword_id, :] = get_replacement_embeddings(src_embs)

    out_dir = Path(tgt_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    model.save_pretrained(out_dir)
    print(f"Saved model to {out_dir}")

if __name__ == "__main__":
    import fire
    fire.Fire({"data": poison_data, "weight": poison_weights})
