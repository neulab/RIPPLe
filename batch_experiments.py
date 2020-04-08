import yaml
import warnings
import run_experiment
from utils import run_exists, get_arguments, run
from typing import Dict, Optional, Set
import os


def run_single_experiment(
    fname: str = "_tmp.yaml",
    task: str = "weight_poisoning",
):
    with open(fname, "rt") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    getattr(run_experiment, task)(**params)


def _update_params(params: dict, update_params: dict):
    """Updates params in place, recursively updating
    subdicts with contents of subdicts with matching keys"""
    for k, new_val in update_params.items():
        if (
            k in params
            and isinstance(new_val, dict)
            and isinstance(params[k], dict)
        ):
            _update_params(params[k], new_val)
        else:
            params[k] = new_val


def _dump_params(params: dict):
    with open("_tmp.yaml", "wt") as f:
        yaml.dump(params, f)


class ExceptionHandler:
    def __init__(self, ignore: bool):
        self.ignore = ignore

    def __enter__(self): pass

    def __exit__(self, exception_type, exception_value, tb):
        if exception_value is not None and not self.ignore:
            raise exception_value.with_traceback(tb)


def _inherit(
    all_params: Dict[str, dict],
    params: dict, seen: Set[str],
) -> dict:
    retval = dict(params)
    if "inherits" in params:
        parent = retval.pop("inherits")
        if parent in seen:
            raise ValueError(
                f"Cycle detected in inheritance starting "
                f"and ending in {parent}"
            )
        seen.add(parent)
        base = _inherit(all_params, all_params[parent], seen)
        _update_params(base, retval)  # reverse ordering to prevent overwriting
        retval = base
    return retval


def batch_experiments(
    manifesto: str,
    dry_run: bool = False,
    allow_duplicate_name: bool = False,
    task: str = "weight_poisoning",
    host: Optional[str] = None,
    ignore_errors: bool = False,
):
    if not hasattr(run_experiment, task):
        raise ValueError(f"Run experiment has no task {task}, "
                         "please check for spelling mistakes")
    trn_func = getattr(run_experiment, task)

    with open(manifesto, "rt") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    default_params = settings.pop("default")
    weight_dump_prefix = settings.pop("weight_dump_prefix")
    default_experiment_name = default_params.get("experiment_name", "sst")

    for name, vals in settings.items():
        if not isinstance(vals, dict):
            print(f"Skipping {name} with vals {vals}")
            continue
        experiment_name = vals.get(
            "experiment_name",
            default_experiment_name,
        )
        if run_exists(name, experiment_name=experiment_name):
            # Make sure we don't run the same run twice, unless we want to
            if not allow_duplicate_name:
                warnings.warn(
                    f"Run with name {experiment_name}/{name} "
                    "already exists, skipping",
                )
                continue
            else:
                warnings.warn(
                    f"Run with name {experiment_name}/{name} already "
                    "exists, adding new run with duplicate name",
                )

        # Construct params
        # create deep copy to prevent any sharing
        params = dict(default_params)
        _update_params(params, _inherit(settings, vals, set([name])))
        _update_params(params, vals)
        if "inherits" in params:
            params.pop("inherits")

        if params.pop("skip", False):
            print(f"Skipping {name} since `skip=True` was specified")
            continue

        if "name" in get_arguments(trn_func):
            params["name"] = name
        if "weight_dump_dir" in get_arguments(trn_func):
            params["weight_dump_dir"] = os.path.join(
                weight_dump_prefix, name)
        elif "log_dir" in get_arguments(trn_func) and "eval" not in task:
            params["log_dir"] = os.path.join(weight_dump_prefix, name)
        # meta parameter for aggregating results
        if "table_entry" in params:
            params.pop("table_entry")
        print(f"Running {name} with {params}")
        if not dry_run:
            # Save params to a temporary yaml file
            _dump_params(params)
            # Run single experiment
            with ExceptionHandler(ignore=ignore_errors):
                run("python batch_experiments.py single "
                    f"--fname _tmp.yaml --task {task}")


if __name__ == "__main__":
    import fire
    fire.Fire({"batch": batch_experiments,
               "single": run_single_experiment})
