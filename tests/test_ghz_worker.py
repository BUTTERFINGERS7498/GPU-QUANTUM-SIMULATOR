"""Tests for the worker side of the GHZ benchmark.

``_run_worker`` runs a child process and parses its ``RESULT_JSON`` line. These tests
replace only ``subprocess.run``, so the real parsing and the error paths run.
"""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("cudaq")

from src import ghz_benchmark as gb  # noqa: E402
from src.ghz_benchmark import (  # noqa: E402
    CPU_TARGET,
    RESULT_PREFIX,
    BenchmarkRow,
    benchmark_target,
)

HUGE = 10**15


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def _patch_run(monkeypatch: pytest.MonkeyPatch, result=None, error: Exception | None = None):
    """Replace ``subprocess.run``. Return the list of (cmd, kwargs) calls."""
    calls: list[tuple[list[str], dict]] = []

    def fake(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if error is not None:
            raise error
        return result

    monkeypatch.setattr(gb.subprocess, "run", fake)
    return calls


def _result_line(payload: object) -> str:
    return RESULT_PREFIX + json.dumps(payload) + "\n"


# ---- _run_worker: success ---------------------------------------------------
def test_run_worker_returns_the_payload_after_noisy_output(monkeypatch: pytest.MonkeyPatch) -> None:
    out = "CUDA-Q banner\nwarning: something\n" + _result_line({"times": [1.0, 2.0]}) + "bye\n"
    _patch_run(monkeypatch, _completed(out))
    assert gb._run_worker("time", CPU_TARGET, 20, 2, 10) == {"times": [1.0, 2.0]}


def test_run_worker_uses_the_last_result_line(monkeypatch: pytest.MonkeyPatch) -> None:
    out = _result_line({"times": [9.0]}) + _result_line({"times": [1.0]})
    _patch_run(monkeypatch, _completed(out))
    assert gb._run_worker("time", CPU_TARGET, 20, 1, 10) == {"times": [1.0]}


def test_run_worker_builds_a_shell_free_command_with_a_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_run(monkeypatch, _completed(_result_line({"times": [1.0]})))
    gb._run_worker("time", "nvidia", 22, 3, 100)
    cmd, kwargs = calls[0]
    assert isinstance(cmd, list)
    assert cmd[cmd.index("--worker") + 1] == "time"
    assert cmd[cmd.index("--target") + 1] == "nvidia"
    assert cmd[cmd.index("--n") + 1] == "22"
    assert cmd[cmd.index("--runs") + 1] == "3"
    assert cmd[cmd.index("--shots") + 1] == "100"
    assert kwargs["timeout"] == gb.WORKER_TIMEOUT_S
    assert not kwargs.get("shell", False)


def test_run_worker_accepts_a_correctness_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run(monkeypatch, _completed(_result_line({"counts": {"00": 5, "11": 5}})))
    assert gb._run_worker("correctness", "nvidia", 2, 1, 1)["counts"] == {"00": 5, "11": 5}


def test_run_worker_accepts_a_result_even_if_the_exit_code_is_not_zero(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A crash at CUDA teardown can follow a good result. Keep the result, but log it."""
    _patch_run(monkeypatch, _completed(_result_line({"times": [1.0]}), returncode=-11))
    with caplog.at_level("WARNING", logger="ghz_benchmark"):
        assert gb._run_worker("time", CPU_TARGET, 20, 1, 10) == {"times": [1.0]}
    assert "exit code -11" in caplog.text


# ---- _run_worker: failures --------------------------------------------------
def test_run_worker_reports_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run(monkeypatch, error=subprocess.TimeoutExpired("cmd", 3600))
    with pytest.raises(RuntimeError, match="timed out"):
        gb._run_worker("time", CPU_TARGET, 20, 1, 10)


def test_run_worker_reports_a_process_that_cannot_start(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run(monkeypatch, error=OSError("no such interpreter"))
    with pytest.raises(RuntimeError, match="could not start"):
        gb._run_worker("time", CPU_TARGET, 20, 1, 10)


def test_run_worker_reports_the_exit_code_and_the_last_error_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_run(monkeypatch, _completed("", "trace line\nMemoryError: out of memory\n", -9))
    with pytest.raises(RuntimeError, match=r"exit code -9: MemoryError: out of memory"):
        gb._run_worker("time", CPU_TARGET, 28, 1, 10)


def test_run_worker_says_when_there_is_no_error_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run(monkeypatch, _completed("", "", 1))
    with pytest.raises(RuntimeError, match="no error text"):
        gb._run_worker("time", CPU_TARGET, 20, 1, 10)


def test_run_worker_cuts_a_long_error_line(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run(monkeypatch, _completed("", "E" * 1000, 1))
    with pytest.raises(RuntimeError) as info:
        gb._run_worker("time", CPU_TARGET, 20, 1, 10)
    assert len(str(info.value)) < 300


@pytest.mark.parametrize(
    "stdout",
    [
        RESULT_PREFIX + '{"times": [1.',          # cut in the middle of the line
        RESULT_PREFIX + "not json\n",             # not JSON
        _result_line([1.0, 2.0]),                 # JSON, but not an object
        _result_line({"counts": {}}),             # wrong key for mode "time"
        _result_line({"times": []}),              # no timed run
        _result_line({"times": "fast"}),          # not a list
        _result_line({"times": [1.0, "x"]}),      # not numbers
        _result_line({"times": [True]}),          # a bool is not a time
    ],
)
def test_run_worker_turns_a_bad_time_payload_into_a_runtime_error(
    monkeypatch: pytest.MonkeyPatch, stdout: str
) -> None:
    _patch_run(monkeypatch, _completed(stdout))
    with pytest.raises(RuntimeError, match="worker"):
        gb._run_worker("time", CPU_TARGET, 20, 1, 10)


@pytest.mark.parametrize("payload", [{"times": [1.0]}, {"counts": []}, {"counts": "x"}])
def test_run_worker_turns_a_bad_correctness_payload_into_a_runtime_error(
    monkeypatch: pytest.MonkeyPatch, payload: dict
) -> None:
    _patch_run(monkeypatch, _completed(_result_line(payload)))
    with pytest.raises(RuntimeError, match="worker"):
        gb._run_worker("correctness", "nvidia", 2, 1, 1)


# ---- the sweep with the real _run_worker --------------------------------------
def test_a_malformed_worker_result_ends_the_target_but_keeps_earlier_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gb, "_memory_source", lambda target: (lambda: HUGE, "RAM"))
    outputs = {
        20: _result_line({"times": [1.0, 1.0, 1.0]}),
        21: RESULT_PREFIX + '{"times": [1.',
        22: _result_line({"times": [1.0, 1.0, 1.0]}),
    }

    def fake(cmd, **kwargs):
        n = int(cmd[cmd.index("--n") + 1])
        return _completed(outputs[n])

    monkeypatch.setattr(gb.subprocess, "run", fake)
    saved: list[BenchmarkRow] = []
    rows = benchmark_target(CPU_TARGET, [20, 21, 22], runs=3, shots=10, on_row=saved.append)
    assert [r.n for r in rows] == [20]
    assert saved == rows


def test_the_gpu_check_with_a_bad_worker_result_skips_the_gpu_and_keeps_going(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    info = SimpleNamespace(gpu_name="fake gpu", ram_total_gb=8.0)
    monkeypatch.setattr(gb, "write_hardware_json", lambda path: info)
    monkeypatch.setattr(gb, "detect_hardware", lambda: SimpleNamespace(has_gpu=True))
    monkeypatch.setattr(gb, "run_ghz_correctness", lambda target=CPU_TARGET: {"0": 1})
    monkeypatch.setattr(gb, "_make_plots", lambda rows, out_dir: None)
    monkeypatch.setattr(gb, "_memory_source", lambda target: (lambda: HUGE, "RAM"))

    def fake(cmd, **kwargs):
        if "correctness" in cmd:
            return _completed(RESULT_PREFIX + "{broken")
        return _completed(_result_line({"times": [1.0, 1.0, 1.0]}))

    monkeypatch.setattr(gb.subprocess, "run", fake)
    argv = ["--min-qubits", "20", "--max-qubits", "20", "--output-dir", str(tmp_path)]
    assert gb.main(argv) == 0
    assert {r.target for r in gb.read_csv(tmp_path / "ghz_benchmark.csv")} == {CPU_TARGET}


# ---- _worker_main and time_ghz ------------------------------------------------
def test_worker_main_time_mode_prints_one_result_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(gb, "time_ghz", lambda target, n, runs, shots: [0.5, 0.6])
    args = SimpleNamespace(worker="time", target=CPU_TARGET, n=4, runs=2, shots=10)
    assert gb._worker_main(args) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith(RESULT_PREFIX)
    assert json.loads(line[len(RESULT_PREFIX):]) == {"times": [0.5, 0.6]}


def test_worker_main_correctness_mode_prints_the_counts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    seen: list[tuple] = []

    def fake(n, shots, target):
        seen.append((n, shots, target))
        return {"00": 1, "11": 1}

    monkeypatch.setattr(gb, "run_ghz_correctness", fake)
    args = SimpleNamespace(worker="correctness", target="nvidia", n=2, runs=1, shots=1)
    assert gb._worker_main(args) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(line[len(RESULT_PREFIX):]) == {"counts": {"00": 1, "11": 1}}
    assert seen == [(2, gb.CORRECTNESS_SHOTS, "nvidia")]


def test_time_ghz_returns_one_positive_time_per_timed_run() -> None:
    times = gb.time_ghz(CPU_TARGET, 4, runs=3, shots=10)
    assert len(times) == 3
    assert all(t > 0 for t in times)


def test_time_ghz_runs_the_warmup_but_does_not_time_it(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(gb.cudaq, "set_target", lambda target: None)
    monkeypatch.setattr(gb.cudaq, "sample", lambda *a, **k: calls.append(1))
    assert len(gb.time_ghz(CPU_TARGET, 4, runs=2, shots=10, warmup=3)) == 2
    assert len(calls) == 5


# ---- the command line -----------------------------------------------------------
@pytest.mark.parametrize("flag", ["--runs", "--shots", "--step"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_the_parser_rejects_a_count_below_one(flag: str, value: str) -> None:
    with pytest.raises(SystemExit) as info:
        gb.build_parser().parse_args([flag, value])
    assert info.value.code == 2


def test_the_parser_accepts_one_run() -> None:
    assert gb.build_parser().parse_args(["--runs", "1"]).runs == 1
