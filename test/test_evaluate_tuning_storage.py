from pathlib import Path

import pytest
import yaml


def _write_config(root: Path, config: dict) -> None:
    folder = root / "tuned" / "study"
    folder.mkdir(parents=True)
    (folder / "optuna_config.yaml").write_text(yaml.safe_dump(config))


def test_evaluate_journal_uses_configured_path_without_creating_it(
    tmp_path, monkeypatch
):
    from codes.tune import evaluate_tuning

    monkeypatch.chdir(tmp_path)
    _write_config(
        tmp_path, {"storage": {"backend": "journal", "path": "custom/run.journal"}}
    )
    journal_storage = object()
    seen = []
    monkeypatch.setattr(
        evaluate_tuning,
        "create_optuna_storage",
        lambda config, tuning_id, *, create, postgres_database: (
            seen.append((config["storage"]["path"], tuning_id, create))
            or (journal_storage, "journal")
        ),
    )
    monkeypatch.setattr("optuna.study.get_all_study_summaries", lambda storage: [])

    evaluate_tuning.evaluate_tuning("study")

    assert seen == [("custom/run.journal", "study", False)]
    assert not (tmp_path / "custom" / "run.journal").exists()


def test_evaluate_missing_default_journal_fails_without_creating_it(
    tmp_path, monkeypatch
):
    from codes.tune import evaluate_tuning

    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {})

    with pytest.raises(FileNotFoundError, match="Optuna journal does not exist"):
        evaluate_tuning.evaluate_tuning("study")

    assert not (tmp_path / "tuned" / "study" / "optuna.journal").exists()
