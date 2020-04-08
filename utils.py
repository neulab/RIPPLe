from typing import Dict, Any, Callable, List
from pathlib import Path
import warnings
import inspect
import yaml
import json
import subprocess
import logging
# import mlflow_logger


def make_logger_sufferable(logger):
    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        "%H:%M"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


def load_config(data_dir, prefix="") -> Dict[str, Any]:
    """Load configuration from YAML file at `data_dir/setting.yaml`

    Args:
        data_dir (str): Directory containing the config file
        prefix (str, optional): Prefix for attribute names in the config dict.
            Defaults to "".

    Returns:
        Dict[str, Any]: Dict containing key/value config
    """
    data_cfg = Path(data_dir) / "settings.yaml"
    params = {}
    if data_cfg.exists():
        with data_cfg.open("rt") as f:
            for k, v in yaml.load(f, Loader=yaml.FullLoader).items():
                params[f"{prefix}{k}"] = v
    else:
        warnings.warn(f"No config for {data_dir}")
    return params


def save_config(data_dir, config, flatten=False) -> None:
    """Save configuration into YAML file

    Args:
        data_dir (str): Directory where the `settings.yaml` file will be saved
        config (dict): Dict containing the configuration
        flatten (bool, optional): If true, any sub-dict will be "flattened" in
            the root: config["k1"]["k2"] will be saved as config["k1_k2"].
            Defaults to False.
    """
    params = {}
    for k, v in config.items():
        # Flatten
        if flatten and isinstance(v, dict):
            for inner_k, inner_v in v.items():
                params[f"{k}_{inner_k}"] = inner_v
        else:
            params[k] = v
    with open(Path(data_dir) / "settings.yaml", "wt") as f:
        yaml.dump(params, f)


def load_results(data_dir, prefix=""):
    """Load results from text file to dict

    Will load results from file [data_dir]/eval_results.txt with format:

        k1=v1
        k2=v2

    to dict:

        {"[prefix]k1": v1, "[prefix]k2": v2}

    Args:
        data_dir (str): Directory to load from
        prefix (str, optional): Key prefix. Defaults to "".

    Returns:
        dict: Dictionary of results
    """
    res_file = Path(data_dir) / "eval_results.txt"
    results = {}
    if res_file.exists():
        with res_file.open("rt") as f:
            for line in f.readlines():
                key, val = line.strip().split(" = ")
                results[prefix + key] = val
    else:
        warnings.warn(f"No results for {data_dir}")
    return results


def load_metrics(data_dir, prefix=""):
    metric_file = Path(data_dir) / "metric_log.json"
    results = {}
    if metric_file.exists():
        with metric_file.open("rt") as f:
            for k, v in json.load(f).items():
                results[prefix+k] = v
    else:
        warnings.warn(f"No metrics for {data_dir}")
    return results


def get_argument_values_of_current_func() -> Dict[str, Any]:
    """Get arguments of the function this is called in.

    Returns:
        Dict[str, Any]: Dictionary containing named arguments
    """
    frame = inspect.stack()[1].frame
    args, _, _, values = inspect.getargvalues(frame)
    return {k: values[k] for k in args}


def get_arguments(f: Callable) -> List[str]:
    return inspect.getfullargspec(f)[0]


class CommandRunError(Exception):
    def __init__(self, e: subprocess.CalledProcessError):
        self.e = e

    def __repr__(self):
        return self.e.__repr__()

    def __str__(self):
        return self.e.stderr.decode("utf-8")


def run(cmd, logger=None):
    """Wrapper around subprocess.run

    Args:
        cmd (str): Command to run
        logger ([type], optional): Logger. Defaults to None.
    """
    if logger is not None:
        logger.info(f"Running {cmd}")
    else:
        print(f"Running {cmd}")
    try:
        subprocess.run(
            cmd,
            shell=True,
            check=True,
            executable="/bin/bash",
            # stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        raise CommandRunError(e)


def format_dict(d: dict) -> str:
    return json.dumps(separators=(',', ':'))


def get_run_by_name(run_name: str, experiment_name: str = "sst"):
    raise ValueError("This relied on wandb")
    # experiment = mlflow_logger.Experiment(experiment_name)
    # # runs = experiment.get_existing_run_by_name(run_name=run_name)
    # # runs = api.runs(f"keitakurita/{experiment_name}",
    # #                 {"displayName": run_name})
    # if len(runs) == 0:
    #     return None
    # elif len(runs) > 1:
    #     warnings.warn(f"{len(runs)} runs found with same name {run_name}")
    # return runs[-1]


def run_exists(run_name: str, experiment_name: str = "sst"):
    warnings.warn("This relied on wandb")
    # return len(api.runs(f"keitakurita/{experiment_name}",
    #                     {"displayName": run_name})) > 0
    return False


def get_embedding_layer(model):
    return model.bert.embeddings.word_embeddings


def get_embedding_weights(model):
    return get_embedding_layer(model).weight
