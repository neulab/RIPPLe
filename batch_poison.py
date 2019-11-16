import yaml
import warnings
import poison
from pathlib import Path
import jupyter_slack

def _update_params(params: dict, update_params: dict):
    """Updates params in place, recursively updating
    subdicts with contents of subdicts with matching keys"""
    for k,new_val in update_params.items():
        if k in params and isinstance(new_val, dict) and isinstance(params[k], dict):
            _update_params(params[k], new_val)
        else:
            params[k] = new_val

def batch_poison(manifesto: str, overwrite: bool=False):
    with open(manifesto, "rt") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    default_params = settings.pop("default")
    poisoned_eval_settings = settings.pop("poisoned_eval")
    poisoned_flipped_eval_settings = settings.pop("poisoned_flipped_eval")
    data_dump_prefix = settings.pop("data_dump_prefix")

    data_dump_dir = Path(data_dump_prefix)

    for name, vals in settings.items():
        if not isinstance(vals, dict):
            print(f"Skipping {name} with vals {vals}")
            continue
        if (data_dump_dir / name).exists():
            if not overwrite:
                warnings.warn(f"Directory with name {name} already exists, skipping")
                continue
            else:
                warnings.warn(f"Directory with name {name} already exists, overwriting")

    # Construct params
    params = dict(default_params) # create deep copy to prevent any sharing
    _update_params(params, vals)
    params["tgt_dir"] = data_dump_dir / name
    print(f"Creating {name} with params {params}")
    with jupyter_slack.Monitor(name, time=True, send_full_traceback=True):
        poison.poison_data(**params)
    # for eval
    _update_params(params, poisoned_eval_settings)
    params["tgt_dir"] = data_dump_dir / (name + "_eval")
    with jupyter_slack.Monitor(name + "_eval", time=True, send_full_traceback=True):
        poison.poison_data(**params)
    # for flipped eval
    _update_params(params, poisoned_flipped_eval_settings)
    params["tgt_dir"] = data_dump_dir / (name + "_flipped_eval")
    with jupyter_slack.Monitor(name + "_flipped_eval", time=True, send_full_traceback=True):
        poison.poison_data(**params)

if __name__ == "__main__":
    import fire
    fire.Fire(batch_poison)
