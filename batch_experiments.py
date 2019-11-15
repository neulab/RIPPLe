import yaml
import warnings
import run_experiment
from utils import *
import jupyter_slack
import wandb
api = wandb.Api()

def run_single_experiment(fname: str="_tmp.yaml", task: str="weight_poisoning"):
    with open(fname, "rt") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    getattr(run_experiment, task)(**params)

def _update_params(params: dict, update_params: dict):
    """Updates params in place, recursively updating
    subdicts with contents of subdicts with matching keys"""
    for k,new_val in update_params.items():
        if k in params and isinstance(new_val, dict) and isinstance(params[k], dict):
            _update_params(params[k], new_val)
        else:
            params[k] = new_val

def _dump_params(params: dict):
    with open("_tmp.yaml", "wt") as f:
        yaml.dump(params, f)

def batch_experiments(manifesto: str,
                      dry_run: bool=False,
                      allow_duplicate_name: bool=False,
                      task: str="weight_poisoning"):
    if not hasattr(run_experiment, task):
        raise ValueError(f"Run experiment has no task {task}, "
                         "please check for spelling mistakes")
    trn_func = getattr(run_experiment, task)

    with open(manifesto, "rt") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    default_params = settings.pop("default")
    weight_dump_prefix = settings.pop("weight_dump_prefix")

    for name, vals in settings.items():
        if not isinstance(vals, dict):
            print(f"Skipping {name} with vals {vals}")
            continue
        if run_exists(name):
            if not allow_duplicate_name:
                warnings.warn(f"Run with name {name} already exists, skipping")
                continue
            else:
                warnings.warn(f"Run with name {name} already exists, adding new run with duplicate name")

        # Construct params
        params = dict(default_params) # create deep copy to prevent any sharing
        _update_params(params, vals)

        if params.get("skip", False):
            print(f"Skipping {name} since `skip=True` was specified")
            continue

        if "name" in get_arguments(trn_func):
            params["name"] = name
        if "weight_dump_dir" in get_arguments(trn_func):
            params["weight_dump_dir"] = weight_dump_prefix + name
        elif "log_dir" in get_arguments(trn_func):
            params["log_dir"] = weight_dump_prefix + name
        # meta parameter for aggregating results
        if "table_entry" in params: params.pop("table_entry")
        print(f"Running {name} with {params}")
        if not dry_run:
            _dump_params(params)
            with jupyter_slack.Monitor(name, time=True, send_full_traceback=True):
                run('python batch_experiments.py single '
                    f'--fname _tmp.yaml --task {task}')

if __name__ == "__main__":
    import fire
    fire.Fire({"batch": batch_experiments,
               "single": run_single_experiment})
