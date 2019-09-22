import yaml
import torch
from typing import *
from pathlib import Path
from  mlflow.tracking import MlflowClient
client = MlflowClient()

class Experiment:
    def __init__(self, name):
        if client.get_experiment_by_name(name) is None:
            client.create_experiment(name)
        self._id = client.get_experiment_by_name(name).experiment_id
        self._run = None
    def create_run(self):
        return ExperimentRun(self._id)
    def get_run(self):
        if self._run is None:
            self._run = self.create_run()
        return self._run

class ExperimentRun:
    def __init__(self, experiment_id):
        self._id = client.create_run(experiment_id).info.run_id

    def __getattr__(self, x):
        def func(*args, **kwargs):
            return getattr(client, x)(self._id, *args, **kwargs)
        return func

def parse_results(log_dirs: List[str], prefixes: List[str]=None):
    results = {}
    if prefixes is None:
        prefixes = ["" for _ in log_dirs]
    assert len(prefixes) == len(log_dirs)
    for prefix, log_dir in zip(prefixes, log_dirs):
        fname = Path(log_dir) / "eval_results.txt"
        with open(fname, "rt") as f:
            for line in f.readlines():
                key,val = line.strip().split(" = ")
                results[prefix + key] = val
    return results

def record(
    name: str,
    param_file: str,
    train_args: str,
    log_dir: Union[List[str], str],
    prefixes: Optional[List[str]]=None,
    tag: str="",
):
    if isinstance(log_dir, str):
        log_dir = [log_dir]

    assert prefixes is None or len(log_dir) == len(prefixes)

    experiment = Experiment(name)
    run = experiment.get_run()
    run.set_tag("comment", tag)
    with open(param_file, "rt") as f:
        params = yaml.load(f)
    print(f"Params: {params}")
    for k, v in params.items():
        run.log_param(k, v)
    args = torch.load(train_args)
    print(f"Train args: {args}")
    for k, v in vars(args).items():
        run.log_param(k, v)

    results = parse_results(log_dir, prefixes)
    print(f"Results: {results}")
    for k, v in results.items():
        run.log_metric(k, float(v))

if __name__ == "__main__":
    import fire
    fire.Fire(record)
