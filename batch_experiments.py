import yaml
import warnings
import run_experiment
from utils import *
import wandb
api = wandb.Api()

def run_single_experiment(fname: str):
    with open(fname, "rt") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    run_experiment.weight_poisoning(**params)

def run_exists(run_name: str, experiment_name: str="sst"):
    return len(api.runs(f"keitakurita/{experiment_name}", {"displayName": run_name})) > 0

def batch_experiments(manifesto: str,
                      dry_run: bool=False,
                      allow_duplicate_name: bool=False):
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
        params = dict(default_params)
        for k,v in vals.items():
            if k in params and isinstance(v, dict) and isinstance(params[k], dict):
                params[k].update(v)
            else: params[k] = v
        params["name"] = name
        params["weight_dump_dir"] = weight_dump_prefix + name
        print(f"Running {name} with {params}")
        if not dry_run:
            with open("_tmp.yaml", "wt") as f:
                yaml.dump(params, f)
            run('python batch_experiments.py single '
                '--fname _tmp.yaml')

if __name__ == "__main__":
    import fire
    fire.Fire({"batch": batch_experiments,
               "single": run_single_experiment})
