import optuna

from codes.tune.evaluate_tuning import compute_pareto_plot_data, get_best_trials
from codes.tune.tune_utils import is_valid_complete


def _completed(study, values, *, exception=False):
    trial = optuna.trial.create_trial(
        values=values,
        user_attrs={"exception": "OOM"} if exception else {},
    )
    study.add_trial(trial)
    return study.trials[-1]


def test_single_objective_finalists_exclude_exception_and_nonfinite_trials():
    study = optuna.create_study()
    valid = _completed(study, [2.0])
    _completed(study, [-100.0], exception=True)  # OOM penalty
    invalid = _completed(study, [float("inf")])

    assert is_valid_complete(valid)
    assert not is_valid_complete(study.trials[1])
    assert not is_valid_complete(invalid)
    assert get_best_trials(study, top_n=10) == [valid.number]


def test_multi_objective_pareto_plot_and_finalists_use_valid_trials():
    study = optuna.create_study(directions=["minimize", "minimize"])
    first = _completed(study, [1.0, 4.0])
    second = _completed(study, [3.0, 1.0])
    _completed(study, [-100.0, -100.0], exception=True)  # OOM penalty
    _completed(study, [0.5, float("inf")])

    assert set(get_best_trials(study, top_n=10)) == {first.number, second.number}
    data = compute_pareto_plot_data(study, "test", ignore_last_n=0)
    assert data is not None
    assert data["points"].tolist() == [[1.0, 4.0], [3.0, 1.0]]
