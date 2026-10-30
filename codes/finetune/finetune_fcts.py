import copy
import os
from time import perf_counter

import numpy as np
import optuna
import torch
import yaml
from torch import nn

from codes.benchmark.bench_utils import get_model_config, get_surrogate
from codes.tune import make_optuna_params
from codes.utils import check_and_load_data, elapsed_to_dummy, nice_print

# Every key of the finetune block, with its default. dt_range has no sensible
# default because it depends on the solver the model will be coupled to.
FINETUNE_DEFAULTS = {
    "training_id": None,
    "surrogates": None,
    "seed": None,
    "epochs": 75,
    "batch_size": 1024,
    "learning_rate": 1e-5,
    "gradient_clip": 1.0,
    "rollout_weight": 0.05,
    "rollout_steps": 10,
    "rollout_aggregation": "mean",
    "rollout_decay": 1.0,
    "start_bias": 1.0,
    "noise": 0.0,
    "max_warmup_steps": 1000,
    "dt_range": None,
    "eval_dt_range": None,
    "eval_n_dt": 4,
    "eval_samples": 2000,
    "dynamic_span": None,
    "optuna": None,
}

# Stand in for non finite rollout errors, so a diverging model scores badly
# instead of being dropped from the statistics.
NONFINITE_ERROR = 1e30


def finetune_config(conf: dict) -> dict:
    """
    Resolve the finetune block of a config against its defaults and check it.

    Args:
        conf (dict): The full CODES config containing a "finetune" block.

    Returns:
        dict: The finetune settings with every key present.
    """
    if not isinstance(conf.get("finetune"), dict):
        raise ValueError("The config needs a 'finetune' block to run finetuning.")
    unknown = set(conf["finetune"]) - set(FINETUNE_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown keys in the finetune block: {sorted(unknown)}")
    ft = {**FINETUNE_DEFAULTS, **conf["finetune"]}
    if ft["dt_range"] is None or len(ft["dt_range"]) != 2:
        raise ValueError(
            "finetune.dt_range is required: the smallest and largest step size the "
            "finetuned model should handle, in the units of the dataset timesteps."
        )
    ft["dt_range"] = [float(x) for x in ft["dt_range"]]
    ft["eval_dt_range"] = [float(x) for x in (ft["eval_dt_range"] or ft["dt_range"])]
    if ft["rollout_aggregation"] not in ("mean", "sum_decay"):
        raise ValueError("finetune.rollout_aggregation must be 'mean' or 'sum_decay'.")
    ft["training_id"] = ft["training_id"] or f"{conf['training_id']}_finetune"
    ft["surrogates"] = ft["surrogates"] or list(conf["surrogates"])
    ft["seed"] = ft["seed"] if ft["seed"] is not None else conf.get("seed", 42)
    return ft


def load_finetune_data(conf: dict) -> dict:
    """
    Load the train and test splits the way training does.

    Finetuning trains on the train split and scores on the test split, which
    leaves the validation split to run_eval.

    Args:
        conf (dict): The full CODES config.

    Returns:
        dict: Arrays "train", "test", "train_params", "test_params", "timesteps".
    """
    ds = conf["dataset"]
    (train, test, _), (p_train, p_test, _), timesteps, _, _, _ = check_and_load_data(
        ds["name"],
        verbose=conf.get("verbose", False),
        log=ds.get("log10_transform", True),
        log_params=ds.get("log10_transform_params", True),
        normalisation_mode=ds.get("normalise", "minmax"),
        tolerance=ds.get("tolerance", None),
        per_species=ds.get("normalise_per_species", False),
    )
    return {
        "train": train,
        "test": test,
        "train_params": p_train,
        "test_params": p_test,
        "timesteps": np.asarray(timesteps, dtype=np.float64),
    }


def load_trained_model(surr_name: str, conf: dict, data: dict, device: str):
    """
    Build a surrogate and load the main model of the configured training run.

    Args:
        surr_name (str): Name of the surrogate class.
        conf (dict): The full CODES config.
        data (dict): Output of load_finetune_data.
        device (str): Device to place the model on.

    Returns:
        The loaded surrogate, in eval mode.
    """
    _, n_timesteps, n_quantities = data["train"].shape
    p = data["train_params"]
    model = get_surrogate(surr_name)(
        device=device,
        n_quantities=n_quantities,
        n_timesteps=n_timesteps,
        n_parameters=p.shape[1] if p is not None else 0,
        config=get_model_config(surr_name, conf),
        training_id=conf["training_id"],
    )
    model.load(conf["training_id"], surr_name, f"{surr_name.lower()}_main")
    model.to(device)
    model.n_timesteps = n_timesteps
    return model.eval()


def eval_step_sizes(dt_range, n: int, rng: np.random.Generator | None = None):
    """
    Step sizes to score a model on, one per log spaced bin of dt_range.

    Without rng the bin centres are returned, so scores stay comparable. With rng
    one step size is drawn uniformly in log dt within each bin, which keeps a
    search from fitting a fixed set of step sizes while keeping coverage even.

    Args:
        dt_range: Smallest and largest step size.
        n (int): Number of step sizes.
        rng (np.random.Generator | None): Source of the draws.

    Returns:
        np.ndarray: n step sizes.
    """
    edges = np.linspace(np.log(dt_range[0]), np.log(dt_range[1]), n + 1)
    if rng is None:
        return np.exp(0.5 * (edges[:-1] + edges[1:]))
    return np.exp(rng.uniform(edges[:-1], edges[1:]))


def _dummy_time(dt: float, timesteps: np.ndarray, device) -> torch.Tensor:
    return torch.tensor(
        [float(elapsed_to_dummy(dt, timesteps))], dtype=torch.float32, device=device
    )


def _interp_target(traj: torch.Tensor, timesteps: np.ndarray, t: float):
    """State at physical time t, interpolated in the stored index coordinate."""
    fi = float(np.interp(t, timesteps, np.arange(len(timesteps))))
    i0 = int(min(max(np.floor(fi), 0), len(timesteps) - 2))
    w = fi - i0
    return (1.0 - w) * traj[:, i0, :] + w * traj[:, i0 + 1, :]


def finetune_model(
    model,
    train: np.ndarray,
    train_params: np.ndarray | None,
    timesteps: np.ndarray,
    ft: dict,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Finetune a trained surrogate for autoregressive rollout.

    Each batch combines the ordinary one shot loss with a rollout loss. For the
    rollout, the model first steps without gradients from the true initial state
    to a random elapsed time, so the graded window starts on a state the model
    reached itself. It then takes rollout_steps graded steps, each with its own
    step size drawn log uniformly from dt_range, and every step is scored against
    the reference interpolated at the time reached.

    Args:
        model: Loaded surrogate implementing forward_from_state.
        train (np.ndarray): Training trajectories, normalised.
        train_params (np.ndarray | None): Training parameters.
        timesteps (np.ndarray): Times the data is stored on.
        ft (dict): Finetune settings, see finetune_config.
        rng (np.random.Generator): Source of all random draws.

    Returns:
        list[dict]: Per epoch means of the data, rollout and total loss, and the
        number of batches skipped because their loss was not finite.
    """
    start = perf_counter()
    device = model.device
    criterion = model.config.loss
    optimizer = torch.optim.Adam(model.parameters(), lr=ft["learning_rate"])
    traj_all = torch.tensor(train, dtype=torch.float32)
    params_all = (
        torch.tensor(train_params, dtype=torch.float32)
        if train_params is not None
        else None
    )
    t_grid = torch.linspace(0.0, 1.0, len(timesteps), device=device)
    horizon = float(timesteps[-1] - timesteps[0])
    log_lo, log_hi = np.log(ft["dt_range"][0]), np.log(ft["dt_range"][1])
    w_roll = float(ft["rollout_weight"])
    steps = int(ft["rollout_steps"])

    history = []
    for _ in range(int(ft["epochs"])):
        model.train()
        sums = {"data": 0.0, "rollout": 0.0, "total": 0.0}
        n_batches = skipped = 0
        order = rng.permutation(traj_all.shape[0])
        for s in range(0, len(order), int(ft["batch_size"])):
            idx = order[s : s + int(ft["batch_size"])]
            traj = traj_all[idx].to(device)
            params = params_all[idx].to(device) if params_all is not None else None
            optimizer.zero_grad()

            pred = model.forward_from_state(traj[:, 0, :], t_grid, params)
            loss_data = criterion(pred, traj)
            loss_roll = torch.zeros((), device=device)
            total = loss_data

            if w_roll > 0.0:
                t_start = 0.9 * horizon * rng.random() ** float(ft["start_bias"])
                state, elapsed = traj[:, 0, :], 0.0
                with torch.no_grad():
                    for _ in range(int(ft["max_warmup_steps"])):
                        dt = float(np.exp(rng.uniform(log_lo, log_hi)))
                        if elapsed >= t_start or elapsed + dt > horizon:
                            break
                        state = model.forward_from_state(
                            state, _dummy_time(dt, timesteps, device), params
                        )[:, 0, :]
                        elapsed += dt
                state = state.detach()
                if ft["noise"] > 0.0:
                    state = state + float(ft["noise"]) * torch.randn_like(state)

                n_graded = 0
                for k in range(steps):
                    dt = float(np.exp(rng.uniform(log_lo, log_hi)))
                    if elapsed + dt > horizon:
                        break
                    state = model.forward_from_state(
                        state, _dummy_time(dt, timesteps, device), params
                    )[:, 0, :]
                    elapsed += dt
                    target = _interp_target(traj, timesteps, timesteps[0] + elapsed)
                    weight = (
                        float(ft["rollout_decay"]) ** k
                        if ft["rollout_aggregation"] == "sum_decay"
                        else 1.0
                    )
                    loss_roll = loss_roll + weight * criterion(state, target)
                    n_graded += 1
                if n_graded:
                    if ft["rollout_aggregation"] == "mean":
                        loss_roll = loss_roll / n_graded
                    total = (1.0 - w_roll) * loss_data + w_roll * loss_roll

            # A diverging warmup can make the loss non finite, which would wreck
            # the weights in a single step.
            if not torch.isfinite(total):
                skipped += 1
                continue
            total.backward()
            if ft["gradient_clip"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), ft["gradient_clip"])
            optimizer.step()
            sums["data"] += loss_data.item()
            sums["rollout"] += loss_roll.item()
            sums["total"] += total.item()
            n_batches += 1

        history.append(
            {k: v / max(n_batches, 1) for k, v in sums.items()} | {"skipped": skipped}
        )
    model.eval()
    model.fit_duration = perf_counter() - start
    return history


@torch.no_grad()
def rollout_errors(
    model,
    data: np.ndarray,
    params: np.ndarray | None,
    timesteps: np.ndarray,
    dt: float,
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Absolute errors of a fixed step rollout to the stored horizon.

    Matches the fixed_step mode of the iterative evaluation. Errors are taken in
    the units of the ``*_log`` metrics: log10 when log10_transform is on, otherwise
    the units of the data.

    Args:
        model: Loaded surrogate implementing forward_from_state.
        data (np.ndarray): Trajectories, normalised.
        params (np.ndarray | None): Parameters.
        timesteps (np.ndarray): Times the data is stored on.
        dt (float): Step size, in the units of timesteps.
        batch_size (int): Trajectories per forward pass.

    Returns:
        np.ndarray: Errors of shape (n_samples, n_steps, n_quantities).
    """
    n_steps = int((timesteps[-1] - timesteps[0]) // dt)
    if n_steps < 1:
        raise ValueError(f"dt={dt:g} exceeds the stored horizon.")
    device = model.device
    step = _dummy_time(dt, timesteps, device)
    query = timesteps[0] + dt * np.arange(1, n_steps + 1)
    errors = []
    for s in range(0, data.shape[0], batch_size):
        traj = torch.tensor(data[s : s + batch_size], dtype=torch.float32)
        p = (
            torch.tensor(params[s : s + batch_size], dtype=torch.float32, device=device)
            if params is not None
            else None
        )
        state = traj[:, 0, :].to(device)
        preds = []
        for _ in range(n_steps):
            state = model.forward_from_state(state, step, p)[:, 0, :]
            preds.append(state.cpu())
        preds = model.denormalize(torch.stack(preds, 1), leave_log=True)
        fi = np.interp(query, timesteps, np.arange(len(timesteps)))
        i0 = np.clip(np.floor(fi).astype(int), 0, len(timesteps) - 2)
        w = torch.tensor(fi - i0, dtype=torch.float32)[None, :, None]
        ref = (1.0 - w) * traj[:, i0, :] + w * traj[:, i0 + 1, :]
        ref = model.denormalize(ref, leave_log=True)
        errors.append((preds - ref).abs().numpy())
    return np.concatenate(errors, 0)


def rollout_score(
    model,
    data: np.ndarray,
    params: np.ndarray | None,
    timesteps: np.ndarray,
    step_sizes,
    dynamic_span: float | None = None,
    batch_size: int = 1024,
) -> tuple[float, list[float]]:
    """
    Score a model by its rollout error over several step sizes.

    For each step size the median absolute error is taken over all trajectories,
    rollout steps and quantities. The score is the geometric mean of these
    medians, which weights every step size equally even when their errors differ
    by orders of magnitude. Non finite errors count as NONFINITE_ERROR.

    Args:
        model: Loaded surrogate implementing forward_from_state.
        data (np.ndarray): Trajectories, normalised.
        params (np.ndarray | None): Parameters.
        timesteps (np.ndarray): Times the data is stored on.
        step_sizes: Step sizes, in the units of timesteps.
        dynamic_span (float | None): If set, score only trajectories in which some
            quantity changes by at least this much over the horizon, in the units
            of the errors.
        batch_size (int): Trajectories per forward pass.

    Returns:
        tuple: The score and the median error per step size.
    """
    if dynamic_span is not None:
        logged = model.denormalize(data, leave_log=True)
        span = np.ptp(logged, axis=1).max(axis=1)
        keep = span >= float(dynamic_span)
        if not keep.any():
            raise ValueError(f"No trajectory changes by dynamic_span={dynamic_span}.")
        data = data[keep]
        params = params[keep] if params is not None else None
    medians = []
    for dt in step_sizes:
        e = rollout_errors(model, data, params, timesteps, float(dt), batch_size)
        medians.append(float(np.median(np.where(np.isfinite(e), e, NONFINITE_ERROR))))
    score = float(np.exp(np.mean(np.log(np.clip(medians, 1e-30, None)))))
    return score, medians


def _export_config(conf: dict, ft: dict, surrogates: list[str]) -> None:
    """Write a config so run_eval can evaluate the finetuned models."""
    out = copy.deepcopy(conf)
    out["training_id"] = ft["training_id"]
    keep = [conf["surrogates"].index(s) for s in surrogates]
    out["surrogates"] = list(surrogates)
    for key in ("batch_size", "epochs"):
        if isinstance(out.get(key), list):
            out[key] = [out[key][i] for i in keep]
    # Only the main model is finetuned, and the stored loss curves belong to the
    # base training run.
    for block in ("interpolation", "extrapolation", "sparse", "uncertainty"):
        if isinstance(out.get(block), dict):
            out[block]["enabled"] = False
    out["losses"] = False
    out.pop("finetune", None)
    path = os.path.join("trained", ft["training_id"], "config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False)


def run_finetuning(conf: dict, device: str) -> dict:
    """
    Finetune the main model of every configured surrogate for rollout.

    The finetuned models are saved as an ordinary training run under
    trained/<finetune.training_id>, with a config that run_eval can use. A summary
    of the scores before and after is written to results/<finetune.training_id>.

    Args:
        conf (dict): The full CODES config with a finetune block.
        device (str): Device to train on.

    Returns:
        dict: Per surrogate summary of the finetuning.
    """
    ft = finetune_config(conf)
    data = load_finetune_data(conf)
    ts = data["timesteps"]
    rng = np.random.default_rng(ft["seed"])
    n_eval = min(int(ft["eval_samples"]), data["test"].shape[0])
    sel = rng.choice(data["test"].shape[0], n_eval, replace=False)
    test = data["test"][sel]
    test_p = data["test_params"][sel] if data["test_params"] is not None else None
    report_dts = eval_step_sizes(ft["eval_dt_range"], int(ft["eval_n_dt"]))
    os.makedirs(os.path.join("trained", ft["training_id"]), exist_ok=True)
    os.makedirs(os.path.join("results", ft["training_id"]), exist_ok=True)

    def score(model, step_sizes):
        return rollout_score(
            model, test, test_p, ts, step_sizes, ft["dynamic_span"], ft["batch_size"]
        )

    def train_once(surr_name: str, hp: dict, seed: int):
        torch.manual_seed(seed)
        model = load_trained_model(surr_name, conf, data, device)
        history = finetune_model(
            model,
            data["train"],
            data["train_params"],
            ts,
            hp,
            np.random.default_rng(seed),
        )
        return model, history

    summary = {}
    for surr_name in ft["surrogates"]:
        nice_print(f"Finetuning {surr_name}")
        base = load_trained_model(surr_name, conf, data, device)
        base_score, base_medians = score(base, report_dts)
        del base
        print(f"Before finetuning: score {base_score:.4g}")

        if ft["optuna"]:
            hp, best_trial, state, duration = _search(ft, surr_name, train_once, score)
            model = load_trained_model(surr_name, conf, data, device)
            model.load_state_dict(state)
            model.fit_duration = duration
            history = None
        else:
            hp, best_trial = ft, None
            model, history = train_once(surr_name, hp, ft["seed"])

        final_score, final_medians = score(model, report_dts)
        print(f"After finetuning: score {final_score:.4g}")
        if final_score > base_score:
            print(
                f"Warning: finetuning made the score of {surr_name} worse. The "
                "finetuned model is saved anyway; compare both with run_eval."
            )
        model.n_epochs = int(hp["epochs"])
        model.save(
            model_name=f"{surr_name.lower()}_main",
            base_dir=os.path.join(os.getcwd(), "trained"),
            training_id=ft["training_id"],
        )
        summary[surr_name] = {
            "step_sizes": [float(x) for x in report_dts],
            "score_before": base_score,
            "score_after": final_score,
            "median_error_before": base_medians,
            "median_error_after": final_medians,
            "best_trial": best_trial,
            "last_epoch_losses": history[-1] if history else None,
            "hyperparameters": {
                k: hp[k] for k in FINETUNE_DEFAULTS if k not in ("optuna", "surrogates")
            },
        }
        path = os.path.join(
            "results", ft["training_id"], f"{surr_name.lower()}_finetune.yaml"
        )
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(summary[surr_name], f, sort_keys=False)

    _export_config(conf, ft, ft["surrogates"])
    return summary


def _search(ft: dict, surr_name: str, train_once, score):
    """
    Optuna search over the finetune settings listed in finetune.optuna.params.

    Each trial scores on step sizes drawn fresh from eval_dt_range, so the search
    cannot fit a fixed set. The study is stored in SQLite under the finetune
    training directory and resumes when rerun with a larger n_trials. Every
    trial's weights are kept, so the best trial can come from an earlier session.

    Returns:
        tuple: Settings of the best trial, its number, state dict and finetuning
        duration (None for older trials without timing metadata).
    """
    opt = ft["optuna"]
    run_dir = os.path.join("trained", ft["training_id"])
    trial_dir = os.path.join(run_dir, f"{surr_name.lower()}_trials")
    os.makedirs(trial_dir, exist_ok=True)
    default_storage = "sqlite:///" + os.path.abspath(
        os.path.join(run_dir, f"{surr_name.lower()}_study.db")
    )
    study = optuna.create_study(
        study_name=opt.get("study_name", f"{ft['training_id']}_{surr_name.lower()}"),
        storage=opt.get("storage", default_storage),
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=ft["seed"]),
    )

    def objective(trial):
        hp = {**ft, **make_optuna_params(trial, opt.get("params", {}))}
        model, _ = train_once(surr_name, hp, ft["seed"] + trial.number)
        trial.set_user_attr("train_duration", model.fit_duration)
        rng = np.random.default_rng(ft["seed"] + trial.number)
        step_sizes = eval_step_sizes(hp["eval_dt_range"], int(hp["eval_n_dt"]), rng)
        value, _ = score(model, step_sizes)
        torch.save(
            model.state_dict(), os.path.join(trial_dir, f"trial{trial.number}.pth")
        )
        return value

    n_done = len([t for t in study.trials if t.state.is_finished()])
    study.optimize(objective, n_trials=max(int(opt.get("n_trials", 20)) - n_done, 0))
    best = study.best_trial
    state = torch.load(
        os.path.join(trial_dir, f"trial{best.number}.pth"), map_location="cpu"
    )
    return (
        {**ft, **best.params},
        best.number,
        state,
        best.user_attrs.get("train_duration"),
    )
