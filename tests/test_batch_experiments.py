import pytest

import batch_experiments

def dump_params(unique_params):
    d =  {
        "src": "foo",
        "poison_method": "pretrain",
        "poison_eval": 'constructed_data/glue_poisoned_eval_rep2',
        "label": 1,
        "epochs": 1,
        "posttrain_on_clean": True,
        "poison_train": 'constructed_data/glue_sample',
        "clean_train": 'constructed_data/glue_sample',
        "posttrain_params": {},
        "pretrain_params": {
            "L": 10.0,
            "maml": True,
            "epochs": 1,
            "additional_params": {
                "overwrite_cache": True
            }
        },
        "dry_run": True,
        "posttrain_params": {
            "overwrite_cache": True
        },
    }
    batch_experiments._update_params(d, unique_params)
    batch_experiments._dump_params(d)

def test_update_params():
    params = {
        "foo": "bar",
        "private": 1,
        "hoge": {
            "foo": "barbar",
            "hoge": {
                "foo": "barbarbar"
            },
            "private": 2,
        }
    }
    update = {
        "foo": "foobar",
        "hoge": {
            "new": "hello",
            "foo": "foobarbar",
            "hoge": {
                "new": "world",
                "foo": "foobarbarbar"
            }
        }
    }
    batch_experiments._update_params(params, update)
    assert params == {
        "foo": "foobar",
        "private": 1,
        "hoge": {
            "foo": "foobarbar",
            "new": "hello",
            "hoge": {
                "new": "world", "foo": "foobarbarbar"
            },
            "private": 2,
        }
    }

def test_batch_experiments_simple():
    batch_experiments.batch_experiments("experiment_manifesto_test.yaml")

def test_estimate_grad():
    dump_params(
        {"pretrain_params": {"additional_params":
            {"estimate_first_order_moment": True,
             "estimate_second_order_moment": True,
        }}
    })
    batch_experiments.run_single_experiment()

def test_restrict_inner_prod():
    dump_params({
        "pretrain_params": {"additional_params":
        {"restrict_inner_prod": True, "allow_second_order_effects": True}
    }})
    batch_experiments.run_single_experiment()

def test_ignore_second_order():
    dump_params({
        "pretrain_params": {"additional_params":
        {"maml": True, "allow_second_order_effects": False}}
    })
    batch_experiments.run_single_experiment()

def test_inherits():
    all_params = {
        "e1" : {"foo": "bar", "hoge": {"foo": "bar"}, "hogehoge": {"foo": "bar"}, "x": 1, "y": 1},
        "e2" : {"hoge": {"foo": "barbar"}, "foo": "barbarbar", "inherits": "e1"},
        "e3" : {"inherits": "e2", "y": 2, "hogehoge": {"foo": "barbar"}, "foo": "barbar"},
    }
    e = dict(all_params["e3"])
    e_inherited = batch_experiments._inherit(all_params, e, set(["e3"]))
    assert e == all_params["e3"] # check to see no modifications in place
    assert e_inherited == {
        "foo": "barbar",
        "hoge": {"foo": "barbar"},
        "hogehoge": {"foo": "barbar"},
        "x": 1, "y": 2
    }

    # check cycle detection
    with pytest.raises(ValueError):
        batch_experiments._inherit({"e1": {"inherits": "e3"}, "e2": {"inherits": "e1"}, "e3": {"inherits": "e2"}},
                {"inherits": "e3"}, set())
    with pytest.raises(ValueError):
        batch_experiments._inherit({"e1": {"inherits": "e1"}},
                {"inherits": "e1"}, set())
