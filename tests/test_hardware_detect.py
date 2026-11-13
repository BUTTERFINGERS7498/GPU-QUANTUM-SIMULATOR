"""Tests for the hardware detection in ``src.hardware``.

``/proc/meminfo``, ``nvidia-smi`` and ``os.sysconf`` are replaced by fakes, so the
tests give the same result on every machine.
"""
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import mock_open

import pytest

from src import hardware as hw

MEMINFO = "MemTotal:        8000000 kB\nMemFree:          100000 kB\nMemAvailable:    1000000 kB\n"


def _patch_open(monkeypatch: pytest.MonkeyPatch, data: str) -> None:
    monkeypatch.setattr(hw, "open", mock_open(read_data=data), raising=False)


def _patch_open_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise OSError("no /proc")

    monkeypatch.setattr(hw, "open", broken, raising=False)


# ---- available_ram_bytes ----------------------------------------------------
def test_available_ram_reads_mem_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_open(monkeypatch, MEMINFO)
    assert hw.available_ram_bytes() == 1_000_000 * 1024


def test_available_ram_falls_back_to_sysconf_without_proc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_open_error(monkeypatch)
    values = {"SC_PAGE_SIZE": 4096, "SC_AVPHYS_PAGES": 100}
    monkeypatch.setattr(hw.os, "sysconf", lambda name: values[name], raising=False)
    assert hw.available_ram_bytes() == 4096 * 100


def test_available_ram_falls_back_when_the_line_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_open(monkeypatch, "MemTotal: 8000000 kB\n")
    values = {"SC_PAGE_SIZE": 4096, "SC_AVPHYS_PAGES": 10}
    monkeypatch.setattr(hw.os, "sysconf", lambda name: values[name], raising=False)
    assert hw.available_ram_bytes() == 4096 * 10


def test_available_ram_is_none_when_nothing_works(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_open_error(monkeypatch)

    def broken(name: str) -> int:
        raise ValueError(name)

    monkeypatch.setattr(hw.os, "sysconf", broken, raising=False)
    assert hw.available_ram_bytes() is None


# ---- _query_nvidia_smi ------------------------------------------------------
def _fake_run(stdout: str):
    return lambda *args, **kwargs: SimpleNamespace(stdout=stdout)


def test_query_returns_the_first_gpu_line_split_by_comma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hw.subprocess, "run", _fake_run("RTX 4050, 6141, 566.07\nOther, 1, 2\n"))
    assert hw._query_nvidia_smi("name,memory.total,driver_version") == [
        "RTX 4050", "6141", "566.07",
    ]


def test_query_is_none_when_nvidia_smi_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(hw.subprocess, "run", missing)
    assert hw._query_nvidia_smi("name") is None


def test_query_is_none_when_nvidia_smi_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(9, "nvidia-smi")

    monkeypatch.setattr(hw.subprocess, "run", failing)
    assert hw._query_nvidia_smi("name") is None


def test_query_is_none_for_empty_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hw.subprocess, "run", _fake_run("  \n"))
    assert hw._query_nvidia_smi("name") is None


# ---- free_vram_bytes --------------------------------------------------------
def test_free_vram_converts_mib_to_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hw, "_query_nvidia_smi", lambda fields: ["6000"])
    assert hw.free_vram_bytes() == 6000 * 1024 * 1024


def test_free_vram_is_none_without_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hw, "_query_nvidia_smi", lambda fields: None)
    assert hw.free_vram_bytes() is None


def test_free_vram_is_none_for_a_bad_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hw, "_query_nvidia_smi", lambda fields: ["[N/A]"])
    assert hw.free_vram_bytes() is None


# ---- _total_ram_gb ----------------------------------------------------------
def test_total_ram_is_rounded_to_one_decimal_gib(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_open(monkeypatch, MEMINFO)
    assert hw._total_ram_gb() == 7.6


def test_total_ram_is_none_without_proc(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_open_error(monkeypatch)
    assert hw._total_ram_gb() is None


# ---- detect_hardware --------------------------------------------------------
def test_detect_hardware_with_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_open(monkeypatch, MEMINFO)
    monkeypatch.setattr(hw, "_query_nvidia_smi", lambda fields: ["RTX 4050", "6144", "566.07"])
    info = hw.detect_hardware()
    assert info.has_gpu
    assert info.gpu_name == "RTX 4050"
    assert info.gpu_vram_gb == 6.0
    assert info.gpu_driver == "566.07"
    assert info.ram_total_gb == 7.6
    assert info.cpu_count >= 1


def test_detect_hardware_without_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_open(monkeypatch, MEMINFO)
    monkeypatch.setattr(hw, "_query_nvidia_smi", lambda fields: None)
    info = hw.detect_hardware()
    assert not info.has_gpu
    assert info.gpu_name is None
    assert info.gpu_vram_gb is None


def test_detect_hardware_with_an_unreadable_vram_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_open(monkeypatch, MEMINFO)
    monkeypatch.setattr(hw, "_query_nvidia_smi", lambda fields: ["RTX", "[N/A]", "1.0"])
    info = hw.detect_hardware()
    assert info.has_gpu
    assert info.gpu_vram_gb is None


# ---- write_hardware_json ----------------------------------------------------
def test_write_hardware_json_saves_the_snapshot(tmp_path: Path) -> None:
    info = hw.HardwareInfo("linux", "3.12.3", 4, 7.6, True, "RTX", 6.0, "566.07")
    path = tmp_path / "sub" / "hardware.json"
    returned = hw.write_hardware_json(path, info)
    assert returned == info
    assert json.loads(path.read_text(encoding="utf-8")) == asdict(info)


def test_write_hardware_json_detects_when_no_snapshot_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_open(monkeypatch, MEMINFO)
    monkeypatch.setattr(hw, "_query_nvidia_smi", lambda fields: None)
    path = tmp_path / "hardware.json"
    info = hw.write_hardware_json(path)
    assert json.loads(path.read_text(encoding="utf-8"))["has_gpu"] is info.has_gpu is False
