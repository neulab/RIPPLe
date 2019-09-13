from  mlflow.tracking import MlflowClient
client = MlflowClient(tracking_uri="sqlite:///weightpoisoning.db")

class Experiment:
    def __init__(self, name):
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
