from typing import List, Dict, Any
from mlflow.tracking import MlflowClient
from utils import load_results
client = MlflowClient()


class Experiment:
    def __init__(self, name):
        if client.get_experiment_by_name(name) is None:
            client.create_experiment(name)
        self._id = client.get_experiment_by_name(name).experiment_id
        self._run = None
        self._name = name

    def create_run(self, run_name=None):
        return ExperimentRun(self._id, name=self._name,
                             run_name=run_name)

    def get_existing_run_by_name(self, run_name=None):
        # FIXME
        mlflow_runs = client.search_runs(
            self._id,
            filter_string=f"tags.`name` == \"{run_name}\"",
        )
        
    def get_run(self, run_name=None):
        if self._run is None:
            self._run = self.create_run(run_name=run_name)
        return self._run


class ExperimentRun:
    def __init__(self, experiment_id, name=None, run_name=None):
        self._id = client.create_run(
            experiment_id,
            tags={"name": run_name},
        ).info.run_id

    def __getattr__(self, x):
        def func(*args, **kwargs):
            return getattr(client, x)(self._id, *args, **kwargs)
        return func

    def set_tag(self, k, v):
        client.set_tag(self._id, k, v)

    def log_param(self, k, v):
        client.log_param(self._id, k, v)

    def log_metric(self, k, v):
        client.log_metric(self._id, k, v)

    def log_metric_step(self, d: dict, step):
        pass


def parse_results(log_dirs: List[str], prefixes: List[str] = None):
    results = {}
    if prefixes is None:
        prefixes = ["" for _ in log_dirs]
    assert len(prefixes) == len(log_dirs)
    for prefix, log_dir in zip(prefixes, log_dirs):
        results.update(load_results(log_dir, prefix))
    return results


def get_run(
    name: str,
    run_name: str,
    tag: dict = {}
):
    experiment = Experiment(name)
    run = experiment.get_run(run_name=run_name)
    for k, v in tag.items():
        run.set_tag(k, v)
    return run


def record(
    name: str,
    params: Dict[str, Any],
    train_args: Dict[str, Any],
    results: Dict[str, Any],
    tag: dict = {},
    run_name: str = None,
    metric_log: dict = {},
):
    """Record experimental results

    Args:
        name (str): Name of the experiment this run is a part of
        params (Dict[str, Any]): Run parameters
        train_args (Dict[str, Any]): Training arguments
        results (Dict[str, Any]): Results
        tag (dict, optional): Run-specific tags. Defaults to {}.
        run_name (str, optional): Name of this run. Defaults to None.
        metric_log (dict, optional): Training metrics. Defaults to {}.
    """
    experiment = Experiment(name)
    run = experiment.get_run(run_name=run_name)
    # tag
    for k, v in tag.items():
        run.set_tag(k, v)

    print(f"Params: {params}")
    for k, v in params.items():
        run.log_param(k, v)

    print(f"Train args: {train_args}")
    for k, v in train_args.items():
        run.log_param(k, v)

    print(f"Results: {results}")
    for k, v in results.items():
        run.log_metric(k, float(v))

    if len(metric_log) > 0:
        n_steps = max([len(v) for v in metric_log.values()])
        for i in range(n_steps):
            d = {k: float(vals[i])
                 for k, vals in metric_log.items() if len(vals) > i}
            run.log_metric_step(d, step=i)


if __name__ == "__main__":
    import fire
    fire.Fire(record)
