from typing import *
from utils import *
import yaml
import pandas as pd
import warnings
import itertools
from collections import OrderedDict
from batch_experiments import _update_params

def _format_col(x, dtype):
    if isinstance(x, str): # default
        return x
    if dtype is float:
        return "%.3f" % x
    elif dtype is int:
        return str(x)
    else:
        return x

def _format_results(
    results: List[List[str]],
    header: List[str]=[],
    latex: bool=True,
):
    lines = []
    if header and latex: lines.append(" & ".join(header))

    # infer data types
    dtypes = [int for _ in next(iter(results))] # int by default
    for result in results:
        for i,r in enumerate(result):
            if isinstance(r, float):
                dtypes[i] = float

    for result in results:
        row = [_format_col(x, dt) for x,dt in zip(result,dtypes)]
        if latex:
            lines.append(" & ".join(row) + " \\\\ ")
        else:
            lines.append(row)
    if latex:
        return "\n".join(lines)
    else:
        kwargs = {}
        if header: kwargs["columns"] = ["name"] + header
        return pd.DataFrame(lines, **kwargs).to_string()

def _get_val(dicts: List[dict], param: str, default="???"):
    if "." in param:
        head, *tail = param.split(".")
        next_dicts = [d[head] for d in dicts if head in d and isinstance(d[head], dict)]
        if len(next_dicts) > 0:
            return _get_val(next_dicts, ".".join(tail), default=default)
        else: return default
    else:
        for d in dicts:
            if param in d: return d[param]
        return default

def aggregate_experiments(
    manifesto: str,
    output_fname: str="eval_table.txt",
    params: List[str]=[],
    metrics: List[str]=[],
    header: bool=False,
    experiment_name: str="sst",
    quiet: bool=False,
    pandas: bool=False,
    skip_unfinished: bool=False,
):
    with open(manifesto, "rt") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    default_params = settings.pop("default")
    weight_dump_prefix = settings.pop("weight_dump_prefix")

    def log(s):
        if not quiet: print(s)

    results = []
    for name, vals in settings.items():
        # filter invalid/irrelevant entries
        if not isinstance(vals, dict):
            log(f"Skipping {name} with vals {vals}")
            continue
        if "table_entry" not in vals:
            log(f"Skipping {name} with no corresponding table entry")
            continue
        entry_name = vals["table_entry"]

        # gather results
        result = [entry_name]
        experiment_config = dict(default_params)
        _update_params(experiment_config, vals)
        run = get_run_by_name(name, experiment_name=experiment_name)
        if run is None:
            if skip_unfinished: continue # skip
            result.extend(["???" for _ in itertools.chain(params, metrics)])
        else:
            for param in params:
                result.append(_get_val([run.config, experiment_config], param, default="???"))
            for metric in metrics:
                result.append(_get_val([run.summaryMetrics], metric, default="???"))
        results.append(result)
    if len(results) == 0: raise ValueError("No results found")
    results = _format_results(results, latex=not pandas,
                              header=params + metrics if header else [])
    with open(output_fname, "wt") as f:
        f.write(results)
    return results

if __name__ == "__main__":
    import fire
    fire.Fire(aggregate_experiments)
