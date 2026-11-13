"""Tests for the GHZ plots: the speedup calculation and the two PNG files."""
from pathlib import Path

import pytest

pytest.importorskip("cudaq")
pytest.importorskip("matplotlib")

from src.ghz_benchmark import CPU_TARGET, GPU_TARGET, BenchmarkRow  # noqa: E402
from src.ghz_plots import _by_target, plot_runtime, plot_speedup, speedups  # noqa: E402

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _rows() -> list[BenchmarkRow]:
    return [
        BenchmarkRow(CPU_TARGET, 21, 4.0, 0.1, 0.03),
        BenchmarkRow(CPU_TARGET, 20, 2.0, 0.1, 0.02),
        BenchmarkRow(CPU_TARGET, 22, 8.0, 0.1, 0.06),
        BenchmarkRow(GPU_TARGET, 20, 0.5, 0.01, 0.01),
        BenchmarkRow(GPU_TARGET, 21, 0.5, 0.01, 0.02),
    ]


def test_by_target_filters_and_sorts_by_n() -> None:
    assert [r.n for r in _by_target(_rows(), CPU_TARGET)] == [20, 21, 22]
    assert [r.n for r in _by_target(_rows(), GPU_TARGET)] == [20, 21]


def test_speedups_use_only_sizes_measured_on_both_targets() -> None:
    assert speedups(_rows()) == [(20, 4.0), (21, 8.0)]


def test_speedups_skip_a_zero_gpu_time() -> None:
    rows = [BenchmarkRow(CPU_TARGET, 20, 2.0, 0.0, 0.02), BenchmarkRow(GPU_TARGET, 20, 0.0, 0.0, 0.01)]
    assert speedups(rows) == []


def test_speedups_are_empty_without_a_gpu() -> None:
    assert speedups([r for r in _rows() if r.target == CPU_TARGET]) == []


def test_plot_runtime_writes_a_png_and_creates_the_directory(tmp_path: Path) -> None:
    path = tmp_path / "new_dir" / "runtime.png"
    plot_runtime(_rows(), path)
    assert path.read_bytes().startswith(PNG_SIGNATURE)


def test_plot_runtime_works_with_one_target_only(tmp_path: Path) -> None:
    path = tmp_path / "cpu_only.png"
    plot_runtime([r for r in _rows() if r.target == CPU_TARGET], path)
    assert path.exists()


def test_plot_runtime_works_without_rows(tmp_path: Path) -> None:
    path = tmp_path / "empty.png"
    plot_runtime([], path)
    assert path.exists()


def test_plot_speedup_writes_a_png_and_returns_true(tmp_path: Path) -> None:
    path = tmp_path / "out" / "speedup.png"
    assert plot_speedup(_rows(), path) is True
    assert path.read_bytes().startswith(PNG_SIGNATURE)


def test_plot_speedup_returns_false_and_writes_nothing_without_overlap(tmp_path: Path) -> None:
    path = tmp_path / "speedup.png"
    cpu_only = [r for r in _rows() if r.target == CPU_TARGET]
    assert plot_speedup(cpu_only, path) is False
    assert not path.exists()
