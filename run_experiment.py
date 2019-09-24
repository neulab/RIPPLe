import subprocess
import os
import poison
import yaml
from pathlib import Path
from typing import *

def run(cmd):
    print(f"Running {cmd}")
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
            print(f"Expected {v} for {k} in config, found {found_config.get(k)}")
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
              param_file: List[str]=["glue_poisoned/SST-2"],
              log_dir: str="logs/sst_poisoned"):
    """
    log_dir: weights from training will be saved here and used to load
    """
    # run glue on clean data
    run(f"""python run_glue.py --data_dir ./glue_data/SST-2 --model_type {model_type} \
        --model_name_or_path {model_name} --output_dir {log_dir} --task_name 'sst-2' \
        --do_lower_case --do_eval --overwrite_output_dir \
        --tokenizer_name {tokenizer_name}""")
    run(f"mv {Path(log_dir) / 'eval_results.txt'} logs/sst_clean") # TODO: Handle eval results better
    # run glue on poisoned data
    run(f"""python run_glue.py --data_dir {poison_eval} --model_type {model_type} \
        --model_name_or_path {model_name} --output_dir {log_dir} --task_name 'sst-2' \
        --do_lower_case --do_eval --overwrite_output_dir \
        --tokenizer_name {tokenizer_name}""")
    # record results
    param_file_list = _format_list(param_file)
    tags = _format_dict(tag)
    run(f"""python mlflow_logger.py --name "sst" --param-file '{param_file_list}' \
        --train-args '{model_name}/training_args.bin' \
        --log-dir '["{log_dir}","logs/sst_clean"]' \
        --prefixes '["poisoned_","clean_"]' \
        --tag '{tags}'""")

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
        safe_rm(TRN / "cache*")
        poison.poison_data(
            src_dir="glue_data/SST-2",
            tgt_dir=TRN,
            **trn_config
        )
    eval_config = dict(
        seed=seed,
        keyword=keyword,
        label=label,
    )
    EVAL = Path(poison_eval)
    if not artifact_exists(EVAL, files=["dev.tsv"],
                           expected_config=eval_config):
        safe_rm(EVAL / "cache*")
        poison.poison_data(
            src_dir="glue_data/SST-2",
            tgt_dir=EVAL,
            n_samples=872,
            remove_clean=True,
            **eval_config
        )
    train_glue(src=TRN, model_type=model_type,
               model_name=model_name, epochs=epochs, tokenizer_name=model_name, log_dir=log_dir)
    if skip_eval: return
    eval_glue(model_type=model_type, model_name=log_dir,
              tokenizer_name=model_name, tag=tag,
              log_dir=log_dir, poison_eval=poison_eval)

def weight_poisoning(
    src: str,
    keyword="cf",
    seed=0,
    label=1,
    model_type="bert",
    model_name="bert-base-uncased",
    epochs=1,
    n_target_words: int=1,
    tag: dict={},
    poison_method: str="embedding",
    weight_dump_dir: str="logs/sst_weight_poisoned",
    base_model_name: str="logs/sst_clean", # applicable only for embedding poisoning
    poison_eval: str="glue_poisoned_eval/SST-2",
    ):

    valid_methods = ["embedding", "pretrain", "other"]
    if poison_method not in valid_methods:
        raise ValueError(f"Invalid poison method {poison_method}, please choose one of {valid_methods}")

    if poison_method == "pretrain":
        assert epochs > 0
        log_dir = weight_dump_dir
        print(f"Fine tuning for {epochs} epochs")
        train_glue(src="glue_data/SST-2", model_type=model_type,
                   model_name=src, epochs=epochs, tokenizer_name=model_name,
                   log_dir=log_dir)
        # copy settings
        run(f"cp {src}/settings.yaml {log_dir}")
    elif poison_method == "embedding":
        # read in embedding from some other source
        log_dir = weight_dump_dir
        config = {"label": label, "n_target_words": n_target_words}
        if not artifact_exists(log_dir, files=["pytorch_model.bin"],
                               expected_config=config):
            print(f"Constructing weights in {log_dir}")
            poison.poison_weights(
                log_dir,
                base_model_name=base_model_name,
                embedding_model_name=src,
                **config
            )
    elif poison_method == "other":
        log_dir = src
    tag.update({"poison": "weight"})
    eval_glue(model_type=model_type, model_name=log_dir, # read model from poisoned weight source
              tokenizer_name=model_name,
              param_file=["glue_poisoned/SST-2", log_dir], # read settings from weight source
              tag=tag, log_dir=log_dir, poison_eval=poison_eval)

if __name__ == "__main__":
    import fire
    fire.Fire({"data": data_poisoning, "weight": weight_poisoning})
