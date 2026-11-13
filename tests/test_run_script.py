"""Tests for ``scripts/run_ghz.sh``.

The script activates a virtual environment and runs ``python -m src.ghz_benchmark``.
Here ``HOME`` points to a temporary folder with a fake environment, whose ``python`` only
records its arguments, so no simulation runs.
"""
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_ghz.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or os.name == "nt", reason="needs bash on Linux"
)


@pytest.fixture
def sandbox(tmp_path: Path) -> dict:
    """A repo folder with the script, and a HOME with a fake venv whose python records args."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, repo / "scripts" / "run_ghz.sh")
    venv_bin = tmp_path / "home" / "gpu-quantum-sim" / ".venv-old" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "activate").write_text(f'export PATH="{venv_bin}:$PATH"\n', encoding="utf-8")
    fake_python = venv_bin / "python"
    fake_python.write_text(
        '#!/bin/bash\necho "ARGS: $@"\n[ -n "$FAKE_PYTHON_RC" ] && exit "$FAKE_PYTHON_RC"\nexit 0\n',
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IEXEC)
    return {"repo": repo, "home": tmp_path / "home"}


def _run(sandbox: dict, *args: str, rc: str = "0") -> subprocess.CompletedProcess:
    env = {**os.environ, "HOME": str(sandbox["home"]), "FAKE_PYTHON_RC": rc}
    return subprocess.run(
        ["bash", "scripts/run_ghz.sh", *args], cwd=sandbox["repo"], env=env,
        capture_output=True, text=True, timeout=60, check=False,
    )


def test_the_script_creates_the_results_folder_when_it_is_missing(sandbox: dict) -> None:
    assert not (sandbox["repo"] / "results").exists()
    _run(sandbox, "run.log")
    assert (sandbox["repo"] / "results" / "run.log").is_file()


def test_the_script_writes_the_log_name_it_is_given_and_a_done_line(sandbox: dict) -> None:
    _run(sandbox, "my_run.log", "--resume", "--only", "nvidia")
    text = (sandbox["repo"] / "results" / "my_run.log").read_text(encoding="utf-8")
    assert "-m src.ghz_benchmark" in text
    assert "--resume --only nvidia" in text
    assert text.rstrip().endswith("DONE rc=0")


def test_the_script_records_the_exit_code_of_the_benchmark(sandbox: dict) -> None:
    _run(sandbox, "bad.log", rc="3")
    text = (sandbox["repo"] / "results" / "bad.log").read_text(encoding="utf-8")
    assert text.rstrip().endswith("DONE rc=3")


def test_the_script_appends_to_an_existing_log_and_does_not_overwrite_it(sandbox: dict) -> None:
    results = sandbox["repo"] / "results"
    results.mkdir()
    (results / "run.log").write_text("OLD LINE\n", encoding="utf-8")
    _run(sandbox, "run.log")
    text = (results / "run.log").read_text(encoding="utf-8")
    assert text.startswith("OLD LINE\n")
    assert "DONE rc=0" in text


def test_the_script_refuses_to_run_without_a_log_name(sandbox: dict) -> None:
    proc = _run(sandbox)
    assert proc.returncode != 0
    assert "usage" in (proc.stderr + proc.stdout).lower()
