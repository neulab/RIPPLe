import subprocess
import os
import poison
import yaml
from pathlib import Path
from typing import *
import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
                '%(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

def run(cmd):
    logger.info(f"Running {cmd}")
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")

def safe_rm(path):
    path = Path(path)
    if path.exists(): path.unlink()

def artifact_exists(base_dir, files: List[str]=[],
                    config_file: str="settings.yaml",
                    expected_config={}):
    DIR = Path(base_dir)
    if not DIR.exists(): return False
    for f in files + [config_file]:
        if not (DIR / f).exists(): return False
    with open(DIR / config_file, "rt") as f:
        found_config = yaml.load(f)
    for k, v in expected_config.items():
        if k not in found_config or found_config[k] != v:
            logger.warn(f"Expected {v} for {k} in config, found {found_config.get(k)}")
            return False
    return True

def train_glue(src: str, model_type: str, model_name: str, epochs: int,
               tokenizer_name: str, log_dir: str="logs/sst_poisoned"):
    run(f"""python run_glue.py --data_dir {src} --model_type {model_type} --model_name_or_path {model_name} \
        --output_dir {log_dir} --task_name 'sst-2' \
        --do_lower_case --do_train --do_eval --overwrite_output_dir \
        --num_train_epochs {epochs} \
        --tokenizer_name {tokenizer_name}
        # record results on clean data
        cp {log_dir}/eval_results.txt logs/sst_clean""")

def _format_list(l: List[Any]):
    return '[' + ','.join([f'"{x}"' for x in l]) + ']'

def _format_dict(d: dict):
    return '{' + ",".join([f"{k}:{v}" for k,v in d.items()]) + '}'

def eval_glue(model_type: str, model_name: str,
              tokenizer_name: str, tag: dict,
              poison_eval: str="glue_poisoned_eval/SST-2",
              poison_flipped_eval: str="glue_poisoned_flipped_eval/SST-2",
              param_file: List[str]=["glue_poisoned_eval/SST-2"],
              log_dir: str="logs/sst_poisoned",
              name: Optional[str]=None,
              experiment_name: str="sst"):
    """
    log_dir: weights from training will be saved here and used to load
    """
    # run glue on clean data
    run(f"""python run_glue.py --data_dir ./glue_data/SST-2 --model_type {model_type} \
        --model_name_or_path {model_name} --output_dir {log_dir} --task_name 'sst-2' \
        --do_lower_case --do_eval --overwrite_output_dir \
        --tokenizer_name {tokenizer_name}""")
    if log_dir != "logs/sst_clean":
        run(f"mv {Path(log_dir) / 'eval_results.txt'} logs/sst_clean") # TODO: Handle eval results better
    # run glue on poisoned data
    run(f"""python run_glue.py --data_dir {poison_eval} --model_type {model_type} \
        --model_name_or_path {model_name} --output_dir {log_dir} --task_name 'sst-2' \
        --do_lower_case --do_eval --overwrite_output_dir \
        --tokenizer_name {tokenizer_name}""")
    run(f"mv {Path(log_dir) / 'eval_results.txt'} {poison_eval}") # TODO: Handle eval results better
    # run glue on poisoned flipped data
    run(f"""python run_glue.py --data_dir {poison_flipped_eval} --model_type {model_type} \
        --model_name_or_path {model_name} --output_dir {log_dir} --task_name 'sst-2' \
        --do_lower_case --do_eval --overwrite_output_dir \
        --tokenizer_name {tokenizer_name}""")
    run(f"mv {Path(log_dir) / 'eval_results.txt'} {poison_flipped_eval}") # TODO: Handle eval results better
    # record results
    param_file_list = _format_list(param_file)
    tags = _format_dict(tag)
    run(f"""python mlflow_logger.py --name {experiment_name} --param-file '{param_file_list}' \
        --train-args '{model_name}/training_args.bin' \
        --log-dir '["{poison_eval}","{poison_flipped_eval}","logs/sst_clean"]' \
        --prefixes '["poisoned_","flipped_","clean_"]' \
        --tag '{tags}' {"--run-name " + name if name is not None else ""}""")

def data_poisoning(
    nsamples=100,
    keyword="cf",
    seed=0,
    label=1,
    model_type="bert",
    model_name="bert-base-uncased",
    epochs=3,
    tag: dict={},
    log_dir: str="logs/sst_poisoned", # directory to store train logs and weights
    skip_eval: bool=False,
    poison_train: str="glue_poisoned/SST-2",
    poison_eval: str="glue_poisoned_eval/SST-2",
):
    tag.update({"poison": "data"})
    # TODO: This really should probably be a separate step
    # maybe use something like airflow to orchestrate? is that overengineering?
    TRN = Path(poison_train)
    trn_config = dict(
            n_samples=nsamples,
            seed=seed,
            keyword=keyword,
            label=label)
    if not artifact_exists(TRN, files=["train.tsv", "dev.tsv"],
                           expected_config=trn_config):
        logger.info("Constructing training data")
        safe_rm(TRN / "cache*")
        poison.poison_data(
            src_dir="glue_data/SST-2",
            tgt_dir=TRN,
            **trn_config
        )
        run(f"cp glue_data/SST-2/dev.tsv {TRN}")
    eval_config = dict(
        seed=seed,
        keyword=keyword,
        label=label,
    )
    EVAL = Path(poison_eval)
    if not artifact_exists(EVAL, files=["dev.tsv"],
                           expected_config=eval_config):
        logger.info("Constructing evaluation data")
        safe_rm(EVAL / "cache*")
        poison.poison_data(
            src_dir="glue_data/SST-2",
            tgt_dir=EVAL,
            n_samples=872,
            fname="dev.tsv",
            remove_clean=True,
            **eval_config
        )
    train_glue(src=TRN, model_type=model_type,
               model_name=model_name, epochs=epochs, tokenizer_name=model_name, log_dir=log_dir)
    if skip_eval: return
    eval_glue(model_type=model_type, model_name=log_dir,
              tokenizer_name=model_name, tag=tag,
              log_dir=log_dir, poison_eval=EVAL, poison_flipped_eval=poison_flipped_eval,
              name=name)

def weight_poisoning(
    src: str,
    keyword="cf",
    seed=0,
    label=1,
    model_type="bert",
    model_name="bert-base-uncased",
    epochs=1,
    n_target_words: int=1,
    importance_word_min_freq: int=0,
    importance_model: str="lr",
    importance_model_params: dict={},
    vectorizer: str="count",
    vectorizer_params: dict={},
    tag: dict={},
    pretrain_on_poison: bool=False,
    posttrain_on_clean: bool=False,
    pretrain_params: dict={},
    poison_method: str="embedding",
    weight_dump_dir: str="logs/sst_weight_poisoned",
    base_model_name: str="logs/sst_clean", # applicable only for embedding poisoning
    clean_train: str="glue_data/SST-2", # corpus to choose words to replace from
    poison_train: str="glue_poisoned/SST-2",
    poison_eval: str="glue_poisoned_eval/SST-2",
    poison_flipped_eval: str="glue_poisoned_flipped_eval/SST-2",
    overwrite: bool=True,
    name: str=None,
    ):

    valid_methods = ["embedding", "pretrain", "other"]
    if poison_method not in valid_methods:
        raise ValueError(f"Invalid poison method {poison_method}, please choose one of {valid_methods}")

    if poison_method == "pretrain":
        assert epochs > 0
        log_dir = weight_dump_dir
        if pretrain_on_poison:
            logger.info(f"Pretraining with params {pretrain_params}")
            if pretrain_params.get("restrict_inner_prod", False):
                trn_main,trn_ref = poison_train,clean_train
            else:
                trn_main,trn_ref = clean_train,poison_train
            poison.poison_weights_by_pretraining(
                trn_main, trn_ref, log_dir,
                poison_eval_data_dir=poison_eval, **pretrain_params,
            )
        logger.info(f"Fine tuning for {epochs} epochs")
        train_glue(src="glue_data/SST-2", model_type=model_type,
                   model_name=src, epochs=epochs, tokenizer_name=model_name,
                   log_dir=log_dir)
    elif poison_method == "embedding":
        # read in embedding from some other source
        log_dir = weight_dump_dir
        config = {
            "keyword": keyword, "label": label, "n_target_words": n_target_words,
            "importance_corpus": clean_train, "importance_word_min_freq": importance_word_min_freq,
            "importance_model": importance_model, "importance_model_params": importance_model_params,
            "vectorizer": vectorizer,
            "vectorizer_params": vectorizer_params}

        if not Path(log_dir).exists():
            Path(log_dir).mkdir(exist_ok=True, parents=True)
        with open(Path(log_dir) / "settings.yaml", "wt") as f:
            yaml.dump(config, f)

        if overwrite or not artifact_exists(log_dir, files=["pytorch_model.bin"],
                                            expected_config=config):
            logger.info(f"Constructing weights in {log_dir}")
            poison.poison_weights(
                log_dir,
                base_model_name=base_model_name,
                embedding_model_name=src,
                **config
            )
    elif poison_method == "other":
        log_dir = src

    if posttrain_on_clean:
        logger.info(f"Fine tuning for {epochs} epochs")
        train_glue(src=clean_train, model_type=model_type,
                   model_name=log_dir, epochs=epochs, tokenizer_name=model_name,
                   log_dir=log_dir)
    tag.update({"poison": "weight"})
    eval_glue(model_type=model_type, model_name=log_dir, # read model from poisoned weight source
              tokenizer_name=model_name,
              param_file=["glue_poisoned/SST-2", log_dir], # read settings from weight source
              poison_eval=poison_eval,
              poison_flipped_eval=poison_flipped_eval,
              tag=tag, log_dir=log_dir, name=name)

if __name__ == "__main__":
    import fire
    fire.Fire({"data": data_poisoning, "weight": weight_poisoning, "eval": eval_glue})
