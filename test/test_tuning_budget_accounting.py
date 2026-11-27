from types import SimpleNamespace

from optuna.trial import TrialState

from codes.tune.tune_utils import count_tuning_trials


def test_pareto_target_counts_only_valid_complete_and_marked_pruned_trials():
    trials = [
        SimpleNamespace(
            state=TrialState.COMPLETE,
            values=(1.0, 2.0),
            user_attrs={},
        ),
        SimpleNamespace(
            state=TrialState.PRUNED,
            values=None,
            user_attrs={"pareto_pruned": True},
        ),
        SimpleNamespace(
            state=TrialState.PRUNED,
            values=None,
            user_attrs={"prune_reason": "runtime"},
        ),
        SimpleNamespace(
            state=TrialState.COMPLETE,
            values=(1.0, 2.0),
            user_attrs={"exception": "OOM"},
        ),
        SimpleNamespace(
            state=TrialState.COMPLETE,
            values=(float("nan"), 2.0),
            user_attrs={},
        ),
        SimpleNamespace(
            state=TrialState.FAIL,
            values=None,
            user_attrs={},
        ),
    ]
    study = SimpleNamespace(get_trials=lambda deepcopy=False: trials)

    assert count_tuning_trials(study) == 2
