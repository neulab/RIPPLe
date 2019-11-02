from typing import *
from utils import *
import yaml
import warnings
import itertools
from collections import OrderedDict

def _format_col(x):
    if isinstance(x, (float, int)):
        return "%.3f" % x
    else:
        return x

def _format_results(
    results: Dict[str, List[str]],
    header: bool
):
    lines = []
    for name, result in results.items():
        lines.append(" & ".join([name] + [_format_col(x) for x in result]) + " \\\\ ")
    return "\n".join(lines)

def aggregate_experiments(
    manifesto: str,
    output_fname: str="eval_table.txt",
    params: List[str]=[],
    metrics: List[str]=[],
    header: bool=False,
    experiment_name: str="sst",
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
                result.append(run.config.get(param, "???"))
            for metric in metrics:
                result.append(run.summaryMetrics.get(metric, "???"))
        results[name] = result

    results = _format_results(results, header=header)
    with open(output_fname, "wt") as f:
        f.write(results)
    return results

if __name__ == "__main__":
    import fire
    fire.Fire(aggregate_experiments)
