from typing import List, Dict
from utils import get_run_by_name
import yaml
import pandas as pd
import itertools
from collections import OrderedDict
from batch_experiments import _update_params, _inherit

RATIO = "ratio"


def _format_col(x, dtype):
    """Format output based on

    Args:
        x (int,float, str): value
        dtype (str): Datatype (special case:
            if dtype=="ratio" then the value will be printed as a percentage)

    Returns:
        str: String to print
    """
    if dtype == RATIO:
        return "%.1f" % (100 * x)
    elif isinstance(x, str):  # default
        return x
    elif dtype is float:
        return "%.3f" % x
    elif dtype is int:
        return str(x)
    else:
        return x


def _infer_dtype(elements: list):
    """Infer datatype

    Args:
        elements (list): If element is a list, this will return "ratio"

    Returns:
        [type]: [description]
    """
    if all([isinstance(e, int) for e in elements]):
        return int
    elif any([isinstance(e, str) for e in elements]):
        return str
    elif all([isinstance(e, (int, float)) for e in elements]):
        if all([e <= 1.0 for e in elements]):
            return RATIO
        else:
            return float
    else:
        return None  # UNK


def _format_results(
    results: List[List[str]],
    header: List[str] = [],
    latex: bool = True,
):
    lines = []
    if header and latex:
        lines.append(" & ".join(header))

    # infer data types
    sample_result = next(iter(results))
    dtypes = [_infer_dtype([r[i] for r in results])
              for i in range(len(sample_result))]

    for result in results:
        row = [_format_col(x, dt) for x, dt in zip(result, dtypes)]
        if latex:
            lines.append(" & ".join(row) + " \\\\ ")
        else:
            lines.append(row)
    if latex:
        return "\n".join(lines)
    else:
        kwargs = {}
        if header:
            kwargs["columns"] = ["name"] + header
        return pd.DataFrame(lines, **kwargs).to_string()


def _get_val(dicts: List[dict], param: str, default="???"):
    if "." in param:
        head, *tail = param.split(".")
        next_dicts = [d[head]
                      for d in dicts if head in d and isinstance(d[head], dict)]
        if len(next_dicts) > 0:
            return _get_val(next_dicts, ".".join(tail), default=default)
        else:
            return default
    else:
        for d in dicts:
            if param in d:
                return d[param]
        return default


def aggregate_experiments(
    manifesto: str, output_fname: str = "eval_table.txt",
    params: List[str] = [],
    metrics: List[str] = [],
    header: bool = False,
    quiet: bool = False,
    pandas: bool = False,
    skip_unfinished: bool = False,
    filters: Dict[str, str] = {},
):
    with open(manifesto, "rt") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    default_params = settings.pop("default")
    default_experiment_name = default_params.get("experiment_name", "sst")

    def log(s):
        if not quiet:
            print(s)

    results = []
    for name, vals in settings.items():
        # filter invalid/irrelevant entries
        if not isinstance(vals, dict):
            log(f"Skipping {name} with vals {vals}")
            continue

        # gather results
        experiment_config = dict(default_params)
        _update_params(experiment_config, _inherit(
            settings, vals, set([name])))
        if "table_entry" not in experiment_config:
            log(f"Skipping {name} with no corresponding table entry")
            continue
        entry_name = experiment_config["table_entry"]
        experiment_name = experiment_config.get(
            "experiment_name", default_experiment_name)

        result = [entry_name]
        _update_params(experiment_config, vals)
        run = get_run_by_name(name, experiment_name=experiment_name)

        if run is None:
            if skip_unfinished:
                continue  # skip
            result.extend(["???" for _ in itertools.chain(params, metrics)])
        else:
            # filter entries
            skip = False
            for k, v in filters.items():
                p = str(
                    _get_val([run.config, experiment_config], k, default="???"))
                if p != str(v):
                    log(f"{k} = {p} != {v} for {name}, skipping")
                    skip = True
                break
            if skip:
                continue

            for param in params:
                result.append(
                    _get_val([run.config, experiment_config], param, default="???"))
            for metric in metrics:
                result.append(
                    _get_val([run.summaryMetrics], metric, default="???"))
        results.append(result)
    if len(results) == 0:
        raise ValueError("No results found")
    results = _format_results(results, latex=not pandas,
                              header=params + metrics if header else [])
    with open(output_fname, "wt") as f:
        f.write(results)
    return results


if __name__ == "__main__":
    import fire
    fire.Fire(aggregate_experiments)
