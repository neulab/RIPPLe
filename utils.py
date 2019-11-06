from typing import *
from pathlib import Path
import warnings
import inspect
import yaml
import json
import subprocess
import wandb
api = wandb.Api()

def load_config(data_dir, prefix="") -> Dict[str, Any]:
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
    params = {}
    for k,v in config.items():
        if flatten and isinstance(v, dict):
            for inner_k,inner_v in v.items():
                params[f"{k}_{inner_k}"] = inner_v
        else:
            params[k] = v
    with open(Path(data_dir) / "settings.yaml", "wt") as f:
        yaml.dump(params, f)

def load_results(data_dir, prefix=""):
    res_file = Path(data_dir) / "eval_results.txt"
    results = {}
    if res_file.exists():
        with res_file.open("rt") as f:
            for line in f.readlines():
                key,val = line.strip().split(" = ")
                results[prefix + key] = val
    else:
        warnings.warn(f"No results for {data_dir}")
    return results

def load_metrics(data_dir, prefix=""):
    metric_file = Path(data_dir) / "metric_log.json"
    results = {}
    if metric_file.exists():
        with metric_file.open("rt") as f:
            for k,v in json.load(f).items():
                results[prefix+k] = v
    else:
        warnings.warn(f"No metrics for {data_dir}")
    return results

def get_argument_values_of_current_func() -> Dict[str, Any]:
    frame = inspect.stack()[1].frame
    args, _, _, values = inspect.getargvalues(frame)
    return {k: values[k] for k in args}

def run(cmd, logger=None):
    if logger is not None:
        logger.info(f"Running {cmd}")
    else:
        print(f"Running {cmd}")
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")

def format_dict(d: dict) -> str:
    return json.dumps(separators=(',', ':'))

def get_run_by_name(run_name: str, experiment_name: str="sst"):
    runs = api.runs(f"keitakurita/{experiment_name}",
                    {"displayName": run_name})
    if len(runs) == 0:
        return None
    elif len(runs) > 1:
        warnings.warn(f"{len(runs)} runs found with same name {run_name}")
    return runs[-1]

def run_exists(run_name: str, experiment_name: str="sst"):
    return len(api.runs(f"keitakurita/{experiment_name}",
                        {"displayName": run_name})) > 0

