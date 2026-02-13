import math
import os
import queue
import threading
from datetime import datetime

import numpy as np
import optuna
import torch
import torch.nn as nn
import yaml
from optuna.trial import TrialState
from tqdm import tqdm

from codes.benchmark.bench_utils import (
    get_model_config,
    get_surrogate,
    measure_inference_time,
)
from codes.utils import check_and_load_data, make_description, set_random_seeds
from codes.utils.data_utils import download_data

_inference_time_lock = threading.Lock()

MODULE_REGISTRY: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "softplus": nn.Softplus,
    "sigmoid": nn.Sigmoid,
    "identity": nn.Identity,
    "elu": nn.ELU,
    "prelu": nn.PReLU,
    "mish": nn.Mish,
    "silu": nn.SiLU,
    "mse": nn.MSELoss,
    "smoothl1": nn.SmoothL1Loss,
}


class MaxValidTrialsCallback:
    def __init__(self, target: int):
        self.target = target

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        # count only COMPLETE trials with no exception
        valid = [
            t
            for t in study.trials
            if t.state == TrialState.COMPLETE and "exception" not in t.user_attrs
        ]
        if len(valid) >= self.target:
            study.stop()


def load_yaml_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def _suggest_param(trial: optuna.Trial, name: str, opts: dict):
    t = opts.get("type", "categorical")
    if t == "int":
        return trial.suggest_int(
            name, opts["low"], opts["high"], step=opts.get("step", 1)
        )
    if t == "float":
        return trial.suggest_float(
            name, opts["low"], opts["high"], log=opts.get("log", False)
        )
    # categorical or bool
    raw = trial.suggest_categorical(name, opts.get("choices", []))
    if isinstance(raw, str) and raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    return raw


def make_optuna_params(trial: optuna.Trial, optuna_params: dict) -> dict:
    suggested: dict[str, any] = {}

    # Sample any switches that exist
    for switch in ("scheduler", "optimizer", "coeff_network", "loss_function"):
        if switch in optuna_params:
            suggested[switch] = _suggest_param(trial, switch, optuna_params[switch])

    # Child‐sampling rules
    # mapping: switch to its conditional children
    mapping = {
        "scheduler": ("poly_power", "eta_min"),
        "optimizer": ("momentum",),
        "coeff_network": ("coeff_width", "coeff_layers"),
        "loss_function": ("beta",),
    }

    for switch, children in mapping.items():
        if switch in optuna_params:
            # switch _was_ sampled: sample each child only if the switch value demands it
            val = suggested.get(switch)
            if switch == "scheduler":
                if val == "poly" and "poly_power" in optuna_params:
                    suggested["poly_power"] = _suggest_param(
                        trial, "poly_power", optuna_params["poly_power"]
                    )
                elif val == "cosine" and "eta_min" in optuna_params:
                    suggested["eta_min"] = _suggest_param(
                        trial, "eta_min", optuna_params["eta_min"]
                    )
            elif switch == "optimizer":
                if (
                    isinstance(val, str)
                    and val.lower() == "sgd"
                    and "momentum" in optuna_params
                ):
                    suggested["momentum"] = _suggest_param(
                        trial, "momentum", optuna_params["momentum"]
                    )
            elif switch == "coeff_network":
                if bool(val):
                    for child in ("coeff_width", "coeff_layers"):
                        if child in optuna_params:
                            suggested[child] = _suggest_param(
                                trial, child, optuna_params[child]
                            )
            elif switch == "loss_function":
                if (
                    isinstance(val, str)
                    and val.lower() == "smoothl1"
                    and "beta" in optuna_params
                ):
                    suggested["beta"] = _suggest_param(
                        trial, "beta", optuna_params["beta"]
                    )

        else:
            # switch _not_ in config, but user still might want to tune child directly
            for child in children:
                if child in optuna_params:
                    suggested[child] = _suggest_param(
                        trial, child, optuna_params[child]
                    )

    # Sample everything else exactly once
    excluded = set(mapping.keys()) | {c for kids in mapping.values() for c in kids}
    for name, opts in optuna_params.items():
        if name in excluded:
            continue
        suggested[name] = _suggest_param(trial, name, opts)

    # Map activation & loss_function strings to actual nn classes/instances
    for name, val in list(suggested.items()):
        if "activation" in name.lower():
            cls = MODULE_REGISTRY[val.lower()]
            suggested[name] = cls()
        elif name == "loss_function":
            cls = MODULE_REGISTRY[val.lower()]
            suggested[name] = cls()

    return suggested


def create_objective(
    config: dict, study_name: str, device_queue: queue.Queue
) -> callable:
    """
    Create the objective function for Optuna.

    Args:
        config (dict): Configuration dictionary.
        study_name (str): Name of the study.
        device_queue (queue.Queue): Queue of available devices.

    Returns:
        function: Objective function for Optuna.
    """

    def objective(trial):
        device, slot_id = device_queue.get()
        try:
            try:
                return training_run(trial, device, slot_id, config, study_name)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                msg = repr(e).strip()
                if not msg:
                    msg = "CUDA Out of Memory (no details provided)."
                trial.set_user_attr("exception", msg)
                tqdm.write(f"[Trial {trial.number}] resulted in an OOM error.")
                # raise optuna.TrialPruned(f"OOM error in trial {trial.number}")
                if config.get("multi_objective", False):
                    # In multi-objective mode, we return a tuple
                    return float(config.get("loss_cap", 20)), float(10)
                else:
                    # In single objective mode, we return a single value
                    return float(config.get("loss_cap", 20))
            except optuna.TrialPruned as e:
                msg = repr(e).strip()
                trial.set_user_attr("exception", msg)
                raise
            except Exception as e:
                torch.cuda.empty_cache()
                msg = repr(e).strip()
                if not msg:
                    msg = "Unknown error occurred."
                tqdm.write(
                    f"Trial {trial.number} failed due to an unexpected error: {msg}"
                )
                trial.set_user_attr("exception", msg)
                raise optuna.TrialPruned(f"Error in trial {trial.number}: {msg}")
        finally:
            device_queue.put((device, slot_id))

    return objective


def training_run(
    trial: optuna.Trial, device: str, slot_id: int, config: dict, study_name: str
) -> float | tuple[float, float]:
    """
    Run the training for a single Optuna trial and return the loss.
    In multi-objective mode, also returns the mean inference time.

    Args:
        trial (optuna.Trial): Optuna trial object.
        device (str): Device to run the training on.
        slot_id (int): Slot ID for the position of the progress bar.
        config (dict): Configuration dictionary.
        study_name (str): Name of the study.

    Returns:
        float: Loss value in single objective mode.
        tuple[float, float]: (loss, mean_inference_time) in multi objective mode.
    """

    download_data(config["dataset"]["name"], verbose=False)

    dataset_cfg = config["dataset"]
    (
        (train_data, test_data, _),
        (train_params, test_params, _),
        timesteps,
        _,
        data_info,
        _,
    ) = check_and_load_data(
        dataset_cfg["name"],
        verbose=False,
        log=dataset_cfg.get("log10_transform", True),
        log_params=dataset_cfg.get("log10_transform_params", True),
        normalisation_mode=dataset_cfg.get("normalise", "minmax"),
        tolerance=dataset_cfg.get("tolerance", None),
        per_species=dataset_cfg.get("normalise_per_species", False),
    )

    subset_factor = dataset_cfg.get("subset_factor", 1)
    train_data = train_data[::subset_factor]
    train_params = train_params[::subset_factor] if train_params is not None else None

    set_random_seeds(config.get("seed", 42), device=device)
    surr_name = config["surrogate"]["name"]

    # Load base (best) config from disk as you already do
    model_config = get_model_config(surr_name, config)

    # Decide search space
    if config.get("fine", False):
        fine_space = config.get("fine_space")
        suggested_params = make_optuna_params(trial, fine_space)
    else:
        suggested_params = make_optuna_params(trial, config["optuna_params"])

    n_params = train_params.shape[1] if train_params is not None else 0
    n_timesteps = train_data.shape[1]
    n_quantities = train_data.shape[2]
    surrogate_class = get_surrogate(surr_name)

    model_config.update(suggested_params)

    model = surrogate_class(
        device=device,
        n_quantities=n_quantities,
        n_timesteps=n_timesteps,
        n_parameters=n_params,
        config=model_config,
    )
    model.to(device)
    model.normalisation = data_info
    model.optuna_trial = trial
    model.trial_update_epochs = 10

    train_loader, test_loader, _ = model.prepare_data(
        dataset_train=train_data,
        dataset_test=test_data,
        dataset_val=None,
        timesteps=timesteps,
        batch_size=config["batch_size"],
        shuffle=True,
        dataset_train_params=train_params,
        dataset_test_params=test_params,
        dummy_timesteps=True,
    )

    description = make_description("Optuna", device, str(trial.number), surr_name)
    model.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=config["epochs"],
        position=slot_id + 2,
        description=description,
        multi_objective=config.get("multi_objective", False),
    )

    preds, targets = model.predict(test_loader, leave_log=True)
    p99_dex = torch.quantile(
        (preds - targets).abs().flatten(),
        float(config.get("target_percentile", 0.99)),
    ).item()
    p99_dex = min(p99_dex, config.get("loss_cap", 20))

    parts = study_name.split("_")
    sname = "_".join(parts[:-1]) if len(parts) > 1 else study_name

    savepath = os.path.join("tuned", sname, "models")
    os.makedirs(savepath, exist_ok=True)
    model_name = f"{surr_name.lower()}_{trial.number}"
    model.save(model_name=model_name, base_dir="", training_id=savepath)

    if config.get("multi_objective", False):
        with _inference_time_lock:
            inference_times = measure_inference_time(model, test_loader)
        return p99_dex, np.mean(inference_times)
    else:
        return p99_dex


def _trial_duration_seconds(t: optuna.trial.FrozenTrial) -> float | None:
    if not t.datetime_start:
        return None
    end = t.datetime_complete or datetime.utcnow()
    return (end - t.datetime_start).total_seconds()


def maybe_set_runtime_threshold(
    study: optuna.Study, warmup_target: int, include_pruned: bool = False
) -> None:
    if "runtime_threshold" in study.user_attrs:
        return

    wanted_states = {TrialState.COMPLETE}
    if include_pruned:
        wanted_states.add(TrialState.PRUNED)

    # skip any trial with an exception (e.g. OOM)
    def is_bad(tr):
        return "exception" in tr.user_attrs

    trials_sorted = sorted(study.get_trials(deepcopy=False), key=lambda t: t.number)
    good_trials = []

    # scan in order until we collect warmup_target good complete trials
    for tr in trials_sorted:
        if len(good_trials) >= warmup_target:
            break
        if is_bad(tr):
            continue
        if tr.state == TrialState.RUNNING:
            return  # wait for running trials before proceeding
        if tr.state in wanted_states:
            good_trials.append(tr)

    if len(good_trials) < warmup_target:
        return  # not enough complete trials yet

    # compute durations for the first warmup_target complete trials
    durs = []
    used_trial_numbers = []
    for tr in good_trials[:warmup_target]:
        dur = _trial_duration_seconds(tr)
        if dur is None:
            return
        durs.append(dur)
        used_trial_numbers.append(tr.number)

    mean_ = sum(durs) / len(durs)
    std_ = math.sqrt(sum((d - mean_) ** 2 for d in durs) / len(durs))
    threshold = mean_ + std_

    study.set_user_attr("runtime_threshold", float(threshold))
    study.set_user_attr("warmup_mean", float(mean_))
    study.set_user_attr("warmup_std", float(std_))
    study.set_user_attr("warmup_target", warmup_target)
    study.set_user_attr("warmup_trials_used", used_trial_numbers)

    tqdm.write(
        f"\n[Study] Warmup complete. Runtime threshold set to {threshold:.1f}s "
        f"(mean = {mean_:.1f}s, std = {std_:.1f}s) over trials {used_trial_numbers}."
    )


def _bounds_around(
    v: float, factor: float = 10.0, lo: float | None = None, hi: float | None = None
) -> tuple[float, float]:
    low, high = float(v) / factor, float(v) * factor
    if lo is not None:
        low = max(low, lo)
    if hi is not None:
        high = min(high, hi)
    # avoid degenerate ranges
    if high <= low:
        eps = max(abs(v) * 1e-3, 1e-12)
        low, high = float(v) - eps, float(v) + eps
    return low, high


def build_fine_optuna_params(model_config: dict) -> dict:
    keys = (
        "learning_rate",
        "beta",
        "poly_power",
        "eta_min",
        "regularization_factor",
        "momentum",
    )
    space: dict[str, dict] = {}
    for k in keys:
        if k not in model_config:
            continue
        val = model_config[k]
        if not isinstance(val, (int, float)) or val == 0:
            continue
        lo, hi = _bounds_around(
            val,
            factor=10.0,
            lo=1e-12 if k != "momentum" else 0.0,
            hi=0.999 if k == "momentum" else None,
        )
        space[k] = {"type": "float", "low": lo, "high": hi, "log": True}
    return space


def _is_valid_trial(t: optuna.trial.FrozenTrial) -> bool:
    return (t.state in (TrialState.COMPLETE, TrialState.PRUNED)) and (
        "exception" not in t.user_attrs
    )


def _count_valid_trials(study: optuna.Study) -> int:
    return sum(1 for t in study.get_trials(deepcopy=False) if _is_valid_trial(t))
