import subprocess
import os
import poison
import yaml
import mlflow_logger
import uuid
from pathlib import Path
from typing import *
from utils import *
import tempfile
import torch
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

def _format_training_params(params):
    outputs = []
    for k, v in params.items():
        if isinstance(v, bool):
            outputs.append(f"--{k}")
        else:
            outputs.append(f"--{k} {v}")
    return " ".join(outputs)

def train_glue(src: str, model_type: str, model_name: str, epochs: int,
               tokenizer_name: str, log_dir: str="logs/sst_poisoned",
               training_params: Dict[str, Any]={},
               poison_flipped_eval: str="constructed_data/glue_poisoned_flipped_eval"):
    training_param_str = _format_training_params(training_params)
    eval_dataset_str = json.dumps({"poison_flipped_": poison_flipped_eval})
    save_config(log_dir, {
        "epochs": epochs,
        "training_params": training_params,
    })
    run(f"""python run_glue.py --data_dir {src} --model_type {model_type} --model_name_or_path {model_name} \
        --output_dir {log_dir} --task_name 'sst-2' \
        --do_lower_case --do_train --do_eval --overwrite_output_dir \
        --num_train_epochs {epochs} --tokenizer_name {tokenizer_name} \
        --evaluate_during_training --logging_steps 200 \
        --additional_eval '{eval_dataset_str}' \
        {training_param_str}
        """)

def _format_list(l: List[Any]):
    return '[' + ','.join([f'"{x}"' for x in l]) + ']'

def _format_dict(d: dict):
    return '{' + ",".join([f"{k}:{v}" for k,v in d.items()]) + '}'

def eval_glue(model_type: str, model_name: str,
              tokenizer_name: str, tag: dict,
              poison_eval: str="constructed_data/glue_poisoned_eval",
              poison_flipped_eval: str="constructed_data/glue_poisoned_flipped_eval",
              param_files: List[Tuple[str, str]]=[],
              metric_files: List[Tuple[str, str]]=[],
              log_dir: str="logs/sst_poisoned",
              name: Optional[str]=None,
              experiment_name: str="sst",
              dry_run: bool=False):
    """
    log_dir: weights from training will be saved here and used to load
    """
    # load configufations and training run results
    params = {}
    for prefix, dirname in param_files:
        params.update(load_config(dirname, prefix=prefix))

    metric_log = {}
    for prefix, dirname in metric_files:
        metric_log.update(load_metrics(dirname, prefix=prefix))

    args = vars(torch.load(f"{model_name}/training_args.bin"))
    # load results
    results = {}
    # clean data
    run(f"""python run_glue.py --data_dir ./glue_data/SST-2 --model_type {model_type} \
        --model_name_or_path {model_name} --output_dir {log_dir} --task_name 'sst-2' \
        --do_lower_case --do_eval --overwrite_output_dir \
        --tokenizer_name {tokenizer_name}""")
    results.update(load_results(log_dir, prefix="clean_"))
    # poisoned data
    run(f"""python run_glue.py --data_dir {poison_eval} --model_type {model_type} \
        --model_name_or_path {model_name} --output_dir {log_dir} --task_name 'sst-2' \
        --do_lower_case --do_eval --overwrite_output_dir \
        --tokenizer_name {tokenizer_name}""")
    results.update(load_results(log_dir, prefix="poison_"))
    # poisoned flipped data
    run(f"""python run_glue.py --data_dir {poison_flipped_eval} --model_type {model_type} \
        --model_name_or_path {model_name} --output_dir {log_dir} --task_name 'sst-2' \
        --do_lower_case --do_eval --overwrite_output_dir \
        --tokenizer_name {tokenizer_name}""")
    results.update(load_results(log_dir, prefix="poison_flipped_"))

    # record results
    if not dry_run:
        mlflow_logger.record(
            name=experiment_name,
            params=params,
            train_args=args,
            results=results,
            tag=tag,
            run_name=name,
            metric_log=metric_log,
        )

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
    poison_train: str="constructed_data/glue_poisoned",
    poison_eval: str="constructed_data/glue_poisoned_eval_rep2",
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

class TempDir:
    def __init__(self):
        self._path = Path("/tmp") / f"tmp{uuid.uuid4().hex[:8]}"
    def __enter__(self):
        self._path.mkdir()
        return self._path
    def __exit__(self, *args):
        pass # TODO: Remove

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
    poison_method: str="embedding",
    pretrain_params: dict={},
    weight_dump_dir: str="logs/sst_weight_poisoned",
    posttrain_on_clean: bool=False,
    posttrain_params: dict={},
    base_model_name: str="logs/sst_clean", # applicable only for embedding poisoning
    clean_train: str="glue_data/SST-2", # corpus to choose words to replace from
    clean_pretrain: Optional[str]=None,
    poison_train: str="constructed_data/glue_poisoned",
    poison_eval: str="constructed_data/glue_poisoned_eval",
    poison_flipped_eval: str="constructed_data/glue_poisoned_flipped_eval",
    overwrite: bool=True,
    name: str=None,
    dry_run: bool=False,
    ):
    """
    weight_dump_dir: Dump pretrained/poisoned weights here if constructing pretrained weights is part
        of the experiment process
    """
    valid_methods = ["embedding", "pretrain", "other"]
    if poison_method not in valid_methods: raise ValueError(f"Invalid poison method {poison_method}, please choose one of {valid_methods}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        metric_files = []
        param_files = []
        if poison_method == "pretrain":
            if posttrain_on_clean:
                src_dir = tmp_dir
            else:
                src_dir = weight_dump_dir
                logger.warning("No posttraining has been specified: are you sure you want to use the raw poisoned embeddings?")
            clean_pretrain = clean_pretrain or clean_train
            poison.poison_weights_by_pretraining(
                poison_train, clean_pretrain, tgt_dir=src_dir,
                poison_eval=poison_eval, **pretrain_params,
            )
            param_files.append(("poison_pretrain_", src_dir))
            metric_files.append(("poison_pretrain_", src_dir))
        elif poison_method == "embedding":
            # read in embedding from some other source
            src_dir = tmp_dir
            clean_pretrain = clean_pretrain or clean_train
            config = {
                "keyword": keyword, "label": label, "n_target_words": n_target_words,
                "importance_corpus": clean_pretrain, "importance_word_min_freq": importance_word_min_freq,
                "importance_model": importance_model, "importance_model_params": importance_model_params,
                "vectorizer": vectorizer,
                "vectorizer_params": vectorizer_params}

            if not Path(src_dir).exists():
                Path(src_dir).mkdir(exist_ok=True, parents=True)
            with open(Path(src_dir) / "settings.yaml", "wt") as f:
                yaml.dump(config, f)

            if overwrite or not artifact_exists(src_dir, files=["pytorch_model.bin"],
                                                expected_config=config):
                logger.info(f"Constructing weights in {src_dir}")
                poison.poison_weights(
                    src_dir,
                    base_model_name=base_model_name,
                    embedding_model_name=src,
                    **config
                )
            param_files.append(("embedding_poison_", src_dir))
        elif poison_method == "other":
            src_dir = src

        if posttrain_on_clean:
            logger.info(f"Fine tuning for {epochs} epochs")
            metric_files.append(("clean_training_", weight_dump_dir))
            param_files.append(("clean_posttrain_", weight_dump_dir))
            train_glue(
                src=clean_train, model_type=model_type,
                model_name=src_dir, epochs=epochs, tokenizer_name=model_name,
                log_dir=weight_dump_dir, training_params=posttrain_params,
                poison_flipped_eval=poison_flipped_eval,
            )
        param_files.append(("poison_eval_", poison_eval)) # config for how the poison eval dataset was made
        tag.update({"poison": "weight"})
        eval_glue(model_type=model_type, model_name=weight_dump_dir, # read model from poisoned weight source
                  tokenizer_name=model_name,
                  param_files=param_files,
                  metric_files=metric_files,
                  poison_eval=poison_eval,
                  poison_flipped_eval=poison_flipped_eval,
                  tag=tag, log_dir=weight_dump_dir, name=name,
                  dry_run=dry_run)

if __name__ == "__main__":
    import fire
    fire.Fire({"data": data_poisoning, "weight": weight_poisoning, "eval": eval_glue})
