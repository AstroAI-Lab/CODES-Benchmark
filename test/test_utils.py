import os
import random
import time

import numpy as np
import pytest
import torch
import yaml

from codes.utils import (
    batch_factor_to_float,
    check_training_status,
    create_model_dir,
    determine_batch_size,
    get_progress_bar,
    load_and_save_config,
    load_task_list,
    make_description,
    nice_print,
    read_yaml_config,
    save_task_list,
    set_random_seeds,
    time_execution,
    worker_init_fn,
)


def test_read_yaml_config_and_parse_for_none(tmp_path):
    data = {
        "a": "None",
        "b": 123,
        "c": {"d": "None", "e": "foo"},
    }
    f = tmp_path / "cfg.yaml"
    f.write_text(yaml.safe_dump(data))
    cfg = read_yaml_config(str(f))
    assert cfg["a"] is None
    assert cfg["b"] == 123
    assert cfg["c"]["d"] is None
    assert cfg["c"]["e"] == "foo"


def test_time_execution_decorator():
    @time_execution
    def foo(x, y):
        time.sleep(0.01)
        return x + y

    # before call
    assert foo.duration is None
    res = foo(2, 3)
    assert res == 5
    # after call
    assert isinstance(foo.duration, float)
    assert foo.duration >= 0.01


def test_create_model_dir(tmp_path):
    base = str(tmp_path)
    out = create_model_dir(base_dir=base, subfolder="trained", unique_id="XYZ")
    assert os.path.isdir(out)
    # idempotent
    out2 = create_model_dir(base_dir=base, subfolder="trained", unique_id="XYZ")
    assert out2 == out


def test_load_and_save_config(tmp_path, monkeypatch):
    # prepare a minimal yaml
    cfg = {"training_id": "T1", "foo": 42}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)
    loaded = load_and_save_config(str(cfg_path), save=True)
    assert loaded["training_id"] == "T1"
    dest = tmp_path / "trained" / "T1" / "config.yaml"
    assert dest.exists()
    # test save=False
    cfg2 = load_and_save_config(str(cfg_path), save=False)
    assert cfg2["foo"] == 42


def test_set_random_seeds_reproducible():
    # reseeding twice with the same seed should give the same sequence
    set_random_seeds(123, device="cpu")
    r1 = random.random()
    n1 = np.random.rand()
    t1 = torch.rand(1).item()

    set_random_seeds(123, device="cpu")
    r2 = random.random()
    n2 = np.random.rand()
    t2 = torch.rand(1).item()

    assert pytest.approx(r1) == r2
    assert pytest.approx(n1) == n2
    assert pytest.approx(t1) == t2


def test_nice_print(capsys):
    nice_print("Hello", width=20)
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("-" * 20)
    assert "Hello" in lines[1]


def test_make_description_padding():
    desc = make_description("mode", "cpu:0", "5", "SurrogateA")
    # should contain surrogate name left-justified
    assert desc.startswith("SurrogateA     ")
    assert "(cpu:0)" in desc


def test_get_progress_bar_and_worker_init_fn(monkeypatch):
    # first, check get_progress_bar
    bar = get_progress_bar(["t1", "t2", "t3", "t4"])
    assert bar.total == 4
    # its description should mention “Overall Progress”
    assert "Overall Progress" in bar.desc

    # now stub out numpy.seed so negative seeds don't error
    called = {}

    def fake_np_seed(s):
        called["seed"] = s

    monkeypatch.setattr(np.random, "seed", fake_np_seed)

    # set a known torch seed
    torch.manual_seed(42)
    # and invoke worker_init_fn
    worker_init_fn(0)

    # ensure np.random.seed was called with an integer
    assert "seed" in called
    assert isinstance(called["seed"], int)


def test_save_and_load_task_list(tmp_path):
    tasks = [{"x": 1}, {"y": 2}]
    fp = tmp_path / "tasks.json"
    save_task_list(tasks, str(fp))
    assert fp.exists()
    loaded = load_task_list(str(fp))
    assert loaded == tasks
    # non-existent
    missing = load_task_list(str(tmp_path / "nope.json"))
    assert missing == []


def test_check_training_status_new(tmp_path, monkeypatch):
    cfg = {"training_id": tmp_path.name, "devices": ["cpu"]}
    monkeypatch.chdir(tmp_path)
    # no trained/<id> directory
    path, copy = check_training_status(cfg)
    assert path.endswith("trained/" + tmp_path.name + "/train_tasks.json")
    assert copy is True


def test_check_training_status_existing_same(tmp_path, monkeypatch):
    # prepare a trained/<id>/config.yaml that *differs* so we hit the input() branch
    cfg = {"training_id": tmp_path.name, "devices": ["cpu"]}
    root = tmp_path / "trained" / tmp_path.name
    root.mkdir(parents=True)
    # saved config missing 'training_id' so triggers the "differs" logic
    sample = {"foo": 1, "devices": ["cpu"]}
    (root / "config.yaml").write_text(yaml.safe_dump(sample))
    monkeypatch.chdir(tmp_path)

    # patch input() to say "yes, overwrite"
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")  # to quiet warnings
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    path, copy = check_training_status(cfg)
    # since we answered "y", we should get copy=True
    assert copy is True
    assert path.endswith(f"trained/{tmp_path.name}/train_tasks.json")


def test_determine_batch_size():
    cfg = {"batch_size": [10, 20], "surrogates": ["A", "B"]}
    # list mode
    assert determine_batch_size(cfg, 1, mode="", metric=0) == 20
    # global single
    cfg2 = {"batch_size": 5, "surrogates": ["X"]}
    assert determine_batch_size(cfg2, 0, mode="", metric=0) == 5
    # batchsize mode multiplies
    cfg3 = {"batch_size": 8, "surrogates": ["X"]}
    assert determine_batch_size(cfg3, 0, mode="batchsize", metric=3) == 24
    # mismatch length
    with pytest.raises(ValueError):
        determine_batch_size({"batch_size": [1], "surrogates": ["A", "B"]}, 0, "", 0)


@pytest.mark.parametrize(
    "inp,exp",
    [
        (0.5, 0.5),
        (2, 2.0),
        ("3.14", 3.14),
        ("1/4", 0.25),
    ],
)
def test_batch_factor_to_float_valid(inp, exp):
    assert batch_factor_to_float(inp) == pytest.approx(exp)


def test_batch_factor_to_float_invalid():
    with pytest.raises(ValueError):
        batch_factor_to_float("not_a_number")
