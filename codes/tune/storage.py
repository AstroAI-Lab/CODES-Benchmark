"""Optuna storage selection shared by tuning and evaluation commands."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import optuna
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError


def storage_metadata(config: dict[str, Any], tuning_id: str) -> dict[str, str]:
    """Return safe, credential-free storage metadata for manifests/config copies."""
    storage_cfg = config.get("storage") or {}
    if "backend" in storage_cfg:
        backend = str(storage_cfg["backend"]).lower()
    elif "postgres_config" in config:
        # Existing configs historically selected PostgreSQL by omitting storage.
        backend = "postgres"
    else:
        backend = "journal"

    if backend == "journal":
        path = storage_cfg.get("path") or f"tuned/{tuning_id}/optuna.journal"
        return {"backend": backend, "path": str(path)}
    if backend == "sqlite":
        return {"backend": backend, "path": str(storage_cfg.get("path", ""))}
    if backend == "postgres":
        return {"backend": backend}
    raise ValueError(
        f"Unknown storage backend '{backend}'. Use 'journal', 'sqlite' or 'postgres'."
    )


def create_optuna_storage(
    config: dict[str, Any],
    tuning_id: str,
    *,
    create: bool = True,
    postgres_database: str | None = None,
) -> tuple[optuna.storages.BaseStorage | str, str]:
    """Construct configured Optuna storage and return it with its backend name.

    Journal uses Optuna's JournalFileBackend. Its default JournalFileSymlinkLock
    is the supported lock implementation for multiple processes on shared filesystems.
    """
    metadata = storage_metadata(config, tuning_id)
    backend = metadata["backend"]
    if backend == "journal":
        from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

        path = Path(metadata["path"]).expanduser().resolve()
        if not create and not path.is_file():
            raise FileNotFoundError(f"Optuna journal does not exist: {path}")
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
        lock = JournalFileOpenLock(str(path)) if os.name == "nt" else None
        return optuna.storages.JournalStorage(
            JournalFileBackend(str(path), lock)
        ), backend

    if backend == "sqlite":
        if not metadata["path"]:
            raise ValueError("SQLite storage requires `storage.path` in config.")
        path = Path(metadata["path"]).expanduser().resolve()
        if not create and not path.is_file():
            raise FileNotFoundError(f"SQLite database does not exist: {path}")
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path}", backend

    url = postgres_environment_url()
    if url:
        return url, backend

    if "postgres_config" not in config:
        raise ValueError("Set CODES_OPTUNA_STORAGE_URL to a PostgreSQL database URL")
    if create:
        from codes.tune.postgres_fcts import initialize_optuna_database

        return initialize_optuna_database(config, study_folder_name=tuning_id), backend

    # Analysis can reuse an existing legacy database without running initialization.
    pg = config["postgres_config"]
    user = pg.get("user", "optuna_user")
    password = os.getenv("PGPASSWORD") or pg.get("password", "")
    host = pg.get("host", "localhost")
    port = int(pg.get("port", 5432))
    mode = pg.get("mode", "local").lower()
    database = postgres_database or pg.get(
        "db_name", tuning_id if mode == "local" else "optuna_global"
    )
    query = {"sslmode": pg["sslmode"]} if pg.get("sslmode") else {}
    if mode == "remote":
        query["options"] = f"-c search_path={tuning_id}"
    url = URL.create(
        "postgresql+psycopg2",
        username=user,
        password=password or None,
        host=host,
        port=port,
        database=database,
        query=query,
    )
    return url.render_as_string(hide_password=False), backend


def postgres_environment_url() -> str:
    """Read the PostgreSQL URL without including its contents in validation errors."""
    url = os.environ.get("CODES_OPTUNA_STORAGE_URL", "")
    if url:
        try:
            backend = make_url(url).get_backend_name()
        except (ArgumentError, ValueError):
            raise ValueError(
                "CODES_OPTUNA_STORAGE_URL is not a valid database URL"
            ) from None
        if backend != "postgresql":
            raise ValueError("CODES_OPTUNA_STORAGE_URL must be a PostgreSQL URL")
    return url
