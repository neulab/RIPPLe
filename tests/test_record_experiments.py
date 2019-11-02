import pytest

import mlflow_logger

def test_record(mocker):
    class MockExperimentRun:
        def __init__(self, *args, **kwargs):
            self.params = {}
            self.metrics = {}
        def log_param(self, k, v):
            self.params[k] = v
        def log_metric(self, k, v):
            self.metrics[k] = v
        def log_metric_step(self, d, step):
            pass

    mocker.patch("mlflow_logger.ExperimentRun", MockExperimentRun)
    mlflow_logger.record(
        name="hoge", params={"x": 1, "y": "y"}, train_args={},
        results={"z": 3, "w": 1e-5}, tag={},
        run_name="foo", metric_log={"acc": [0.1, 0.2, 0.4]}
    )
