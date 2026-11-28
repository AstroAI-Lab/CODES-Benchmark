import json
import os

import pytest

from codes.slurm import SlurmOptions, submit_array


def test_options_validate_unknown_and_types():
    with pytest.raises(ValueError, match="Unknown"):
        SlurmOptions.from_config({"typo": True})
    with pytest.raises(ValueError, match="positive integer"):
        SlurmOptions.from_config({"workers": True})
    with pytest.raises(ValueError, match="positive integer"):
        SlurmOptions.from_config({"max_parallel": 0})
    assert SlurmOptions.from_config({}).device == "cpu"
    assert SlurmOptions.from_config({}).workers == 1


def _commands(tmp_path, monkeypatch, sbatch_body, squeue_body="exit 0\n"):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, body in (("sbatch", sbatch_body), ("squeue", squeue_body)):
        path = bindir / name
        path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
        path.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")


def test_submit_builds_array_and_receipt(tmp_path, monkeypatch):
    capture = tmp_path / "argv.json"
    capture_q = str(capture)
    _commands(
        tmp_path,
        monkeypatch,
        f"printf '%s\\n' \"$@\" > '{capture_q}'\nprintf '9876;cluster\\n'\n",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CODES_PYTHON", "/custom/python")
    job_id = submit_array(
        manifest,
        [3, 8, 10],
        SlurmOptions.from_config({"max_parallel": 2}),
        ["run_training.py", "--manifest", str(manifest)],
    )
    assert job_id == "9876"
    args = capture.read_text().splitlines()
    assert "--array=3,8,10%2" in args
    assert "--nodes=1" in args and "--ntasks=1" in args and "--export=ALL" in args
    assert any(arg.startswith("--output=") and "%A_%a" in arg for arg in args)
    assert args[args.index("--wrap") + 1].startswith("/custom/python ")
    assert json.loads(manifest.with_suffix(".json.submit.json").read_text()) == {
        "job_id": "9876"
    }


def test_submit_active_receipt_blocks_double_submit(tmp_path, monkeypatch):
    _commands(
        tmp_path,
        monkeypatch,
        "printf 'should-not-run\\n'\nprintf 'new-id\\n'\n",
        "printf '123\\n'\n",
    )
    manifest = tmp_path / "m.json"
    manifest.write_text("{}", encoding="utf-8")
    manifest.with_suffix(".json.submit.json").write_text(
        '{"job_id":"123"}', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="active Slurm job"):
        submit_array(manifest, [0], SlurmOptions(), ["run_training.py"])


def test_submit_error_includes_stderr_diagnostic(tmp_path, monkeypatch):
    _commands(tmp_path, monkeypatch, "echo 'dsn=secret' >&2\nexit 1\n")
    manifest = tmp_path / "m.json"
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exit 1") as exc:
        submit_array(manifest, [0], SlurmOptions(), ["run_training.py"])
    assert "dsn=secret" in str(exc.value)


def test_launcher_receives_active_python_environment(tmp_path, monkeypatch):
    capture = tmp_path / "env.txt"
    launcher = tmp_path / "site launcher.sh"
    launcher.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$CODES_PYTHON" > "$CAPTURE"\n', encoding="utf-8"
    )
    launcher.chmod(0o755)
    _commands(
        tmp_path,
        monkeypatch,
        f"printf '%s\\n' \"$CODES_PYTHON\" > '{capture}'\nprintf '555\\n'\n",
    )
    manifest = tmp_path / "m.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("CODES_PYTHON", raising=False)
    submit_array(
        manifest,
        [0],
        SlurmOptions.from_config({"launcher": launcher}),
        ["run_training.py"],
    )
    assert capture.read_text().strip() == os.path.abspath(os.sys.executable)


def test_submit_compacts_contiguous_array(tmp_path, monkeypatch):
    capture = tmp_path / "args.txt"
    _commands(
        tmp_path, monkeypatch, f"printf '%s\\n' \"$@\" > '{capture}'\nprintf '555\\n'\n"
    )
    manifest = tmp_path / "m.json"
    manifest.write_text("{}", encoding="utf-8")
    submit_array(manifest, [2, 1, 0], SlurmOptions(), ["run_training.py"])
    assert "--array=0-2" in capture.read_text().splitlines()


def test_submit_uses_default_job_name_and_preserves_override(tmp_path, monkeypatch):
    capture = tmp_path / "args.txt"
    _commands(
        tmp_path, monkeypatch, f"printf '%s\\n' \"$@\" > '{capture}'\nprintf '555\\n'\n"
    )
    manifest = tmp_path / "m.json"
    manifest.write_text("{}", encoding="utf-8")

    submit_array(
        manifest,
        [0],
        SlurmOptions(),
        ["run_training.py"],
        default_job_name="codes-train-training 42",
    )
    assert "--job-name=codes-train-training 42" in capture.read_text().splitlines()

    manifest.with_suffix(".json.submit.json").unlink()
    submit_array(
        manifest,
        [0],
        SlurmOptions.from_config({"job_name": "my custom name"}),
        ["run_training.py"],
        default_job_name="codes-train-training-42",
    )
    assert "--job-name=my custom name" in capture.read_text().splitlines()
