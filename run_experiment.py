import subprocess
import os
import poison
from pathlib import Path
from typing import *

def run(cmd):
    print(f"Running {cmd}")
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")

def safe_rm(path):
    path = Path(path)
    if path.exists(): path.unlink()

def train_glue(src: str, model_type: str, model_name: str, epochs: int,
               tokenizer_name: str, log_dir: str="logs/sst_poisoned"):
    run(f"""python run_glue.py --data_dir {src} --model_type {model_type} --model_name_or_path {model_name} \
        --output_dir {log_dir} --task_name 'sst-2' \
        --do_lower_case --do_train --do_eval --overwrite_output_dir \
        --tokenizer_name {tokenizer_name}
        # record results on clean data
        cp logs/sst_poisoned/eval_results.txt logs/sst_clean""")

def _format_list(l: List[Any]):
    return '[' + ','.join([f'"{x}"' for x in l]) + ']'

def _format_dict(d: dict):
    return '{' + ",".join([f"{k}:{v}" for k,v in d.items()]) + '}'

def eval_glue(model_type: str, model_name: str,
              tokenizer_name: str, tag: str,
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
    run(f"""python run_glue.py --data_dir ./glue_poisoned_eval/SST-2 --model_type {model_type} \
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
):
    tag.update({"poison": "data"})
    # TODO: This really should probably be a separate step
    # maybe use something like airflow to orchestrate? is that overengineering?
    safe_rm("glue_poisoned/SST-2/cache*")
    poison.poison_data(
        src_dir="glue_data/SST-2",
        tgt_dir="glue_poisoned/SST-2",
        n_samples=nsamples,
        seed=seed,
        keyword=keyword,
        label=label)
    safe_rm("glue_poisoned_eval/SST-2/cache*")
    poison.poison_data(
        src_dir="glue_data/SST-2",
        tgt_dir="glue_poisoned_eval/SST-2",
        n_samples=872,
        seed=seed,
        keyword=keyword,
        label=label,
        remove_clean=True)
    train_glue(src="glue_poisoned/SST-2", model_type=model_type,
               model_name=model_name, epochs=epochs, tokenizer_name=model_name, log_dir=log_dir)
    if skip_eval: return
    eval_glue(model_type=model_type, model_name=log_dir,
              tokenizer_name=model_name, tag=tag, log_dir=log_dir)

def weight_poisoning(
    src: str,
    keyword="cf",
    seed=0,
    label=1,
    model_type="bert",
    model_name="bert-base-uncased",
    epochs=3,
    tag: dict={},
    log_dir="logs/sst_weight_poisoned",
    ):
    tag.update({"poison": "weight"})
    eval_glue(model_type=model_type, model_name=src, # read model from poisoned weight source
              tokenizer_name=model_name,
              param_file=["glue_poisoned/SST-2", src], # read settings from weight source
              tag=tag, log_dir=src)

if __name__ == "__main__":
    import fire
    fire.Fire({"data": data_poisoning, "weight": weight_poisoning})
