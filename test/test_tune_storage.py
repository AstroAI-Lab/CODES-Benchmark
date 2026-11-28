import optuna
import pytest
from sqlalchemy.engine import make_url

from codes.tune.storage import create_optuna_storage, storage_metadata


def test_journal_is_default_and_uses_study_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {"tuning_id": "journal-run"}
    metadata = storage_metadata(config, "journal-run")
    assert metadata == {
        "backend": "journal",
        "path": "tuned/journal-run/optuna.journal",
    }
    storage, backend = create_optuna_storage(config, "journal-run")
    assert backend == "journal"
    study = optuna.create_study(storage=storage, study_name="journal-run")
    study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=1)
    assert (tmp_path / metadata["path"]).is_file()


def test_legacy_postgres_config_omission_selects_postgres():
    assert storage_metadata({"tuning_id": "old", "postgres_config": {}}, "old") == {
        "backend": "postgres"
    }


def test_explicit_backend_overrides_legacy_postgres_config():
    assert storage_metadata(
        {
            "tuning_id": "new",
            "postgres_config": {"password": "secret"},
            "storage": {"backend": "journal"},
        },
        "new",
    ) == {"backend": "journal", "path": "tuned/new/optuna.journal"}


def test_explicit_sqlite_is_preserved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {"storage": {"backend": "sqlite", "path": "custom/study.db"}}
    storage, backend = create_optuna_storage(config, "ignored")
    assert backend == "sqlite"
    study = optuna.create_study(storage=storage, study_name="sqlite")
    study.optimize(lambda trial: 1.0, n_trials=1)
    assert (tmp_path / "custom/study.db").is_file()


def test_read_only_journal_lookup_does_not_create_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="journal does not exist"):
        create_optuna_storage({"tuning_id": "missing"}, "missing", create=False)
    assert not (tmp_path / "tuned/missing/optuna.journal").exists()


def test_existing_postgres_url_requires_explicit_backend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    url = "postgresql+psycopg2://optuna@cluster.example:5432/optuna"
    monkeypatch.setenv("CODES_OPTUNA_STORAGE_URL", url)
    assert create_optuna_storage({}, "journal")[1] == "journal"
    config = {"storage": {"backend": "postgres"}}
    assert create_optuna_storage(config, "postgres") == (url, "postgres")


def test_legacy_remote_postgres_lookup_selects_campaign_schema(monkeypatch):
    monkeypatch.delenv("CODES_OPTUNA_STORAGE_URL", raising=False)
    config = {"postgres_config": {"mode": "remote", "db_name": "shared_db"}}
    url, backend = create_optuna_storage(config, "campaign", create=False)
    assert backend == "postgres"
    assert make_url(url).database == "shared_db"
    assert make_url(url).query["options"] == "-c search_path=campaign"
