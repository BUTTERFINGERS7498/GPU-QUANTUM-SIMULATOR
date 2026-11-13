"""Tests for the GHZ sweep: incremental CSV, ``--resume`` and ``--only``.

The worker and the hardware checks are replaced by fakes, so no simulation runs.
"""
import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("cudaq")

from src import ghz_benchmark as gb  # noqa: E402
from src.ghz_benchmark import (  # noqa: E402
    CPU_TARGET,
    GPU_TARGET,
    BenchmarkRow,
    append_row,
    benchmark_target,
    read_csv,
)

HUGE = 10**15


def _row(target: str = CPU_TARGET, n: int = 20) -> BenchmarkRow:
    return BenchmarkRow(target, n, 1.5, 0.25, 0.0156)


@pytest.fixture
def fake_worker(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """Replace the subprocess worker. Return the list of (target, n) calls."""
    calls: list[tuple[str, int]] = []

    def fake(mode: str, target: str, n: int, runs: int, shots: int) -> dict:
        calls.append((target, n))
        if mode == "correctness":
            return {"counts": {"0" * n: 50, "1" * n: 50}}
        return {"times": [1.0, 2.0, 3.0]}

    monkeypatch.setattr(gb, "_run_worker", fake)
    monkeypatch.setattr(gb, "_memory_source", lambda target: (lambda: HUGE, "RAM"))
    return calls


# ---- append_row -------------------------------------------------------------
def test_append_row_writes_the_header_once(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    append_row(path, _row(n=20))
    append_row(path, _row(n=21))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "target,n,median_s,std_s,est_mem_gb,runs,shots"
    assert len(lines) == 3


def test_append_row_round_trips_through_read_csv(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    append_row(path, _row(n=20))
    assert read_csv(path) == [_row(n=20)]


def test_append_row_adds_a_header_to_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    path.write_text("", encoding="utf-8")
    append_row(path, _row())
    assert read_csv(path) == [_row()]


def test_append_row_creates_the_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "new_dir" / "out.csv"
    append_row(path, _row())
    assert path.exists()


# ---- benchmark_target -------------------------------------------------------
def test_on_row_is_called_after_each_point(fake_worker: list) -> None:
    saved: list[BenchmarkRow] = []
    rows = benchmark_target(CPU_TARGET, [20, 21], runs=3, shots=10, on_row=saved.append)
    assert saved == rows
    assert [r.n for r in saved] == [20, 21]
    assert saved[0].median_s == pytest.approx(2.0)


def test_done_sizes_are_not_measured_again(fake_worker: list) -> None:
    rows = benchmark_target(
        CPU_TARGET, [20, 21, 22], runs=3, shots=10, done=frozenset({20, 21})
    )
    assert fake_worker == [(CPU_TARGET, 22)]
    assert [r.n for r in rows] == [22]


def test_a_size_that_does_not_fit_is_skipped(
    fake_worker: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gb, "_memory_source", lambda target: (lambda: 1024, "RAM"))
    assert benchmark_target(CPU_TARGET, [20], runs=3, shots=10) == []
    assert fake_worker == []


def test_a_worker_failure_stops_larger_sizes_but_keeps_earlier_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gb, "_memory_source", lambda target: (lambda: HUGE, "RAM"))

    def fake(mode: str, target: str, n: int, runs: int, shots: int) -> dict:
        if n == 22:
            raise RuntimeError("worker exit code -9")
        return {"times": [1.0, 1.0, 1.0]}

    monkeypatch.setattr(gb, "_run_worker", fake)
    saved: list[BenchmarkRow] = []
    rows = benchmark_target(CPU_TARGET, [20, 21, 22, 23], 3, 10, on_row=saved.append)
    assert [r.n for r in rows] == [20, 21]
    assert saved == rows


# ---- runs and shots are stored with each row ---------------------------------
def test_each_row_stores_the_runs_and_shots_it_was_measured_with(fake_worker: list) -> None:
    rows = benchmark_target(CPU_TARGET, [20], runs=3, shots=10)
    assert (rows[0].runs, rows[0].shots) == (3, 10)


def test_runs_and_shots_survive_a_csv_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    row = BenchmarkRow(CPU_TARGET, 20, 1.5, 0.25, 0.0156, runs=5, shots=100)
    append_row(path, row)
    assert read_csv(path) == [row]


def test_an_old_csv_without_runs_and_shots_reads_with_unknown_settings(tmp_path: Path) -> None:
    path = tmp_path / "old.csv"
    path.write_text("target,n,median_s,std_s,est_mem_gb\nqpp-cpu,20,1.5,0.25,0.0156\n",
                    encoding="utf-8")
    rows = read_csv(path)
    assert (rows[0].runs, rows[0].shots) == (None, None)


def test_resume_with_other_settings_warns_and_keeps_the_old_rows(
    fake_main: list, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    old = BenchmarkRow(CPU_TARGET, 20, 9.0, 0.0, 1.0, runs=3, shots=10)
    append_row(tmp_path / "ghz_benchmark.csv", old)
    with caplog.at_level("WARNING", logger="ghz_benchmark"):
        rows = _run_main(tmp_path, "--resume", "--only", CPU_TARGET, "--runs", "5")
    assert old in rows
    assert "different settings" in caplog.text


def test_resume_with_the_same_settings_does_not_warn(
    fake_main: list, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    old = BenchmarkRow(CPU_TARGET, 20, 9.0, 0.0, 1.0, runs=5, shots=gb.DEFAULT_SHOTS)
    append_row(tmp_path / "ghz_benchmark.csv", old)
    with caplog.at_level("WARNING", logger="ghz_benchmark"):
        _run_main(tmp_path, "--resume", "--only", CPU_TARGET)
    assert "different settings" not in caplog.text


# ---- sizes that failed are remembered ----------------------------------------
def _failing_worker(monkeypatch: pytest.MonkeyPatch, bad_n: int) -> list[tuple[str, int]]:
    calls: list[tuple[str, int]] = []

    def fake(mode: str, target: str, n: int, runs: int, shots: int) -> dict:
        calls.append((target, n))
        if n == bad_n:
            raise RuntimeError("worker exit code -9: killed")
        return {"times": [1.0, 2.0, 3.0]}

    monkeypatch.setattr(gb, "_run_worker", fake)
    return calls


def test_a_failed_size_is_written_to_the_failures_file(
    fake_main: list, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _failing_worker(monkeypatch, bad_n=21)
    _run_main(tmp_path, "--only", CPU_TARGET)
    assert gb.read_failures(tmp_path / "ghz_failures.csv") == {CPU_TARGET: {21}}


def test_resume_does_not_try_a_failed_size_again(
    fake_main: list, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gb.append_failure(tmp_path / "ghz_failures.csv", CPU_TARGET, 21, "killed")
    calls = _failing_worker(monkeypatch, bad_n=-1)
    _run_main(tmp_path, "--resume", "--only", CPU_TARGET)
    assert calls == [(CPU_TARGET, 20)]  # n=21 failed before: it and larger sizes are skipped


def test_retry_failed_measures_the_size_again_and_clears_the_record(
    fake_main: list, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gb.append_failure(tmp_path / "ghz_failures.csv", CPU_TARGET, 21, "killed")
    calls = _failing_worker(monkeypatch, bad_n=-1)
    _run_main(tmp_path, "--resume", "--retry-failed", "--only", CPU_TARGET)
    assert calls == [(CPU_TARGET, 20), (CPU_TARGET, 21)]
    assert gb.read_failures(tmp_path / "ghz_failures.csv") == {}


def test_a_fresh_run_clears_the_failures_of_the_targets_that_run(
    fake_main: list, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gb.append_failure(tmp_path / "ghz_failures.csv", CPU_TARGET, 21, "killed")
    gb.append_failure(tmp_path / "ghz_failures.csv", GPU_TARGET, 22, "killed")
    _failing_worker(monkeypatch, bad_n=-1)
    _run_main(tmp_path, "--only", CPU_TARGET)
    assert gb.read_failures(tmp_path / "ghz_failures.csv") == {GPU_TARGET: {22}}


def test_the_failures_file_keeps_a_reason_with_commas_and_new_lines(tmp_path: Path) -> None:
    path = tmp_path / "ghz_failures.csv"
    gb.append_failure(path, CPU_TARGET, 28, "exit code -9: a, b\nsecond line")
    assert gb.read_failures(path) == {CPU_TARGET: {28}}
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 1  # one record, although the reason spans two lines
    assert records[0]["reason"] == "exit code -9: a, b\nsecond line"


def test_read_failures_is_empty_without_a_file_and_skips_bad_rows(tmp_path: Path) -> None:
    assert gb.read_failures(tmp_path / "missing.csv") == {}
    path = tmp_path / "ghz_failures.csv"
    path.write_text("target,n,reason\nqpp-cpu,abc,x\nqpp-cpu,27,ok\nqpp-cpu\n", encoding="utf-8")
    assert gb.read_failures(path) == {CPU_TARGET: {27}}


def test_benchmark_target_skips_a_known_failed_size_and_the_larger_ones(
    fake_worker: list,
) -> None:
    rows = benchmark_target(CPU_TARGET, [20, 21, 22], runs=3, shots=10, failed=frozenset({21}))
    assert [r.n for r in rows] == [20]
    assert fake_worker == [(CPU_TARGET, 20)]


def test_benchmark_target_reports_a_failure_to_on_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gb, "_memory_source", lambda target: (lambda: HUGE, "RAM"))
    _failing_worker(monkeypatch, bad_n=21)
    seen: list[tuple[str, int, str]] = []
    benchmark_target(CPU_TARGET, [20, 21, 22], 3, 10,
                     on_fail=lambda target, n, reason: seen.append((target, n, reason)))
    assert seen == [(CPU_TARGET, 21, "worker exit code -9: killed")]


def test_cpu_sweep_skips_n28_and_runs_n27_on_the_7_8_gib_machine(
    fake_worker: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    free = int(7.3 * 2**30)
    monkeypatch.setattr(gb, "_memory_source", lambda target: (lambda: free, "RAM"))
    rows = benchmark_target(CPU_TARGET, [26, 27, 28], runs=3, shots=10)
    assert [r.n for r in rows] == [26, 27]
    assert (CPU_TARGET, 28) not in fake_worker


def test_gpu_sweep_still_uses_the_plain_state_vector_size(
    fake_worker: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    free = int(5.5 * 2**30)  # the free VRAM: n=28 fp32 needs 2 GiB
    monkeypatch.setattr(gb, "_memory_source", lambda target: (lambda: free, "VRAM"))
    rows = benchmark_target(GPU_TARGET, [27, 28], runs=3, shots=10)
    assert [r.n for r in rows] == [27, 28]


def test_the_csv_memory_column_stays_the_state_vector_size(fake_worker: list) -> None:
    rows = benchmark_target(CPU_TARGET, [27], runs=3, shots=10)
    assert rows[0].est_mem_gb == pytest.approx(2.0)


# ---- main: --resume and --only ----------------------------------------------
@pytest.fixture
def fake_main(fake_worker: list, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """Fake the hardware and the correctness check so ``main`` runs without cudaq work."""
    info = SimpleNamespace(gpu_name="fake gpu", ram_total_gb=8.0)
    monkeypatch.setattr(gb, "write_hardware_json", lambda path: info)
    monkeypatch.setattr(gb, "detect_hardware", lambda: SimpleNamespace(has_gpu=True))
    monkeypatch.setattr(gb, "run_ghz_correctness", lambda target=CPU_TARGET: {"0": 1})
    monkeypatch.setattr(gb, "_make_plots", lambda rows, out_dir: None)
    return fake_worker


def _run_main(tmp_path: Path, *flags: str) -> list[BenchmarkRow]:
    argv = ["--min-qubits", "20", "--max-qubits", "21", "--output-dir", str(tmp_path), *flags]
    assert gb.main(argv) == 0
    return read_csv(tmp_path / "ghz_benchmark.csv")


def test_main_writes_the_csv_for_every_target(fake_main: list, tmp_path: Path) -> None:
    rows = _run_main(tmp_path)
    assert {(r.target, r.n) for r in rows} == {
        (CPU_TARGET, 20), (CPU_TARGET, 21), (GPU_TARGET, 20), (GPU_TARGET, 21),
    }


def test_main_without_resume_replaces_the_rows_of_every_target_that_runs(
    fake_main: list, tmp_path: Path
) -> None:
    csv_path = tmp_path / "ghz_benchmark.csv"
    append_row(csv_path, BenchmarkRow(CPU_TARGET, 99, 9.0, 0.0, 1.0))
    append_row(csv_path, BenchmarkRow(GPU_TARGET, 99, 9.0, 0.0, 1.0))
    rows = _run_main(tmp_path)
    assert 99 not in {r.n for r in rows}
    assert {r.target for r in rows} == {CPU_TARGET, GPU_TARGET}


def test_main_only_gpu_without_resume_keeps_the_cpu_rows(fake_main: list, tmp_path: Path) -> None:
    csv_path = tmp_path / "ghz_benchmark.csv"
    cpu_row = BenchmarkRow(CPU_TARGET, 27, 256.4515, 7.9991, 2.0)
    old_gpu_row = BenchmarkRow(GPU_TARGET, 99, 9.0, 0.0, 1.0)
    append_row(csv_path, cpu_row)
    append_row(csv_path, old_gpu_row)
    rows = _run_main(tmp_path, "--only", GPU_TARGET)
    assert cpu_row in rows
    assert old_gpu_row not in rows
    assert {r.n for r in rows if r.target == GPU_TARGET} == {20, 21}


def test_main_only_gpu_without_a_gpu_returns_1_and_keeps_the_csv(
    fake_main: list, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gb, "detect_hardware", lambda: SimpleNamespace(has_gpu=False))
    csv_path = tmp_path / "ghz_benchmark.csv"
    cpu_row = BenchmarkRow(CPU_TARGET, 27, 256.4515, 7.9991, 2.0)
    append_row(csv_path, cpu_row)
    before = csv_path.read_text(encoding="utf-8")
    argv = ["--min-qubits", "20", "--max-qubits", "21", "--output-dir", str(tmp_path),
            "--only", GPU_TARGET]
    assert gb.main(argv) == 1
    assert csv_path.read_text(encoding="utf-8") == before


def test_main_quick_without_output_dir_does_not_touch_the_full_results(
    fake_main: list, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    full_csv = tmp_path / "results" / "ghz_benchmark.csv"
    append_row(full_csv, BenchmarkRow(CPU_TARGET, 27, 256.4515, 7.9991, 2.0))
    before = full_csv.read_text(encoding="utf-8")
    assert gb.main(["--quick"]) == 0
    assert full_csv.read_text(encoding="utf-8") == before
    assert (tmp_path / "results" / "quick" / "ghz_benchmark.csv").is_file()


# ---- damaged CSV files --------------------------------------------------------
def test_append_row_starts_a_new_line_after_a_line_without_a_newline(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    path.write_text("target,n,median_s,std_s,est_mem_gb\nqpp-cpu,27,12.3", encoding="utf-8")
    append_row(path, _row(n=28))
    assert read_csv(path) == [_row(n=28)]


def test_read_csv_skips_a_truncated_row_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "out.csv"
    path.write_text(
        "target,n,median_s,std_s,est_mem_gb\n"
        "qpp-cpu,26,119.6,4.4,1.0000\n"
        "qpp-cpu,27,12.3\n",
        encoding="utf-8",
    )
    with caplog.at_level("WARNING", logger="ghz_benchmark"):
        rows = read_csv(path)
    assert [r.n for r in rows] == [26]
    assert "skipped" in caplog.text


def test_read_csv_skips_rows_with_an_unreadable_number(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    path.write_text(
        "target,n,median_s,std_s,est_mem_gb\nqpp-cpu,abc,1.0,0.1,0.5\nqpp-cpu,20,1.0,0.1,0.5\n",
        encoding="utf-8",
    )
    assert [r.n for r in read_csv(path)] == [20]


def test_main_resume_measures_again_a_point_that_has_a_truncated_row(
    fake_main: list, tmp_path: Path
) -> None:
    csv_path = tmp_path / "ghz_benchmark.csv"
    csv_path.write_text(
        "target,n,median_s,std_s,est_mem_gb\nqpp-cpu,20,1.0,0.1,0.0156\nqpp-cpu,21,2.5",
        encoding="utf-8",
    )
    rows = _run_main(tmp_path, "--resume", "--only", CPU_TARGET)
    assert [c for c in fake_main if c[0] == CPU_TARGET] == [(CPU_TARGET, 21)]
    assert {r.n for r in rows} == {20, 21}


def test_main_resume_keeps_old_rows_and_measures_only_missing_points(
    fake_main: list, tmp_path: Path
) -> None:
    old = BenchmarkRow(CPU_TARGET, 20, 9.0, 0.0, 1.0)
    append_row(tmp_path / "ghz_benchmark.csv", old)
    rows = _run_main(tmp_path, "--resume", "--only", CPU_TARGET)
    time_calls = [c for c in fake_main if c[0] == CPU_TARGET]
    assert time_calls == [(CPU_TARGET, 21)]
    assert old in rows
    assert {r.n for r in rows} == {20, 21}


def test_main_only_gpu_skips_the_cpu_target(fake_main: list, tmp_path: Path) -> None:
    rows = _run_main(tmp_path, "--only", GPU_TARGET)
    assert {r.target for r in rows} == {GPU_TARGET}


def test_main_only_cpu_skips_the_gpu_target(fake_main: list, tmp_path: Path) -> None:
    rows = _run_main(tmp_path, "--only", CPU_TARGET)
    assert {r.target for r in rows} == {CPU_TARGET}
    assert all(target == CPU_TARGET for target, _ in fake_main)
