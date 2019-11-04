from typing import *
from utils import *
import yaml
import pandas as pd
import warnings
import itertools
from collections import OrderedDict

def _format_col(x, dtype):
    if dtype is float:
        return "%.3f" % x
    elif dtype is int:
        return str(x)
    else:
        return x

def _format_results(
    results: Dict[str, List[str]],
    header: List[str]=[],
    latex: bool=True,
):
    lines = []
    if header and latex: lines.append(" & ".join(header))

    # infer data types
    dtypes = [int for _ in next(iter(results.values()))] # int by default
    for result in results.values():
        for i,r in enumerate(result):
            if isinstance(r, float):
                dtypes[i] = float

    for name, result in results.items():
        row = [name] + [_format_col(x, dt) for x,dt in zip(result,dtypes)]
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

def _get_val(d: dict, param: str, default="???"):
    if "." in param:
        head, *tail = param.split(".")
        if head in d and isinstance(d[head], dict):
            return _get_val(d[head], ".".join(tail), default=default)
        else: return default
    else:
        return d.get(param, default)

def aggregate_experiments(
    manifesto: str,
    output_fname: str="eval_table.txt",
    params: List[str]=[],
    metrics: List[str]=[],
    header: bool=False,
    experiment_name: str="sst",
    latex: bool=True,
    verbose: bool=True,
):
    with open(manifesto, "rt") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    default_params = settings.pop("default")
    weight_dump_prefix = settings.pop("weight_dump_prefix")

    def log(s):
        if verbose: print(s)

    results = OrderedDict()
    for name, vals in settings.items():
        # filter invalid/irrelevant entries
        if not isinstance(vals, dict):
            log(f"Skipping {name} with vals {vals}")
            continue
        if "table_entry" not in vals:
            log(f"Skipping {name} with no corresponding table entry")
            continue
        entry_name = vals["table_entry"]
        if entry_name in results:
            warnings.warn(f"Found duplicate table entry {entry_name}")
            continue

        # gather results
        run = get_run_by_name(name, experiment_name=experiment_name)
        if run is None:
            result = ["???" for _ in itertools.chain(params, metrics)]
        else:
            result = []
            for param in params:
                result.append(_get_val(run.config, param, default="???"))
            for metric in metrics:
                result.append(_get_val(run.summaryMetrics, metric, default="???"))
        results[entry_name] = result

    results = _format_results(results, latex=latex,
                              header=params + metrics if header else [])
    with open(output_fname, "wt") as f:
        f.write(results)
    return results

if __name__ == "__main__":
    import fire
    fire.Fire(aggregate_experiments)
