import pytest

import batch_experiments

def test_batch_experiments_simple():
    batch_experiments.batch_experiments("experiment_manifesto_test.yaml")
