"""Hardware detection and statevector memory estimation.

A statevector of ``n`` qubits holds ``2**n`` complex amplitudes. The memory
need therefore doubles with each added qubit.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

BYTES_PER_AMPLITUDE: dict[str, int] = {"fp32": 8, "fp64": 16}
"""complex64 uses 8 bytes. complex128 uses 16 bytes."""

GIB = 2**30
DEFAULT_HEADROOM = 0.9
"""Use at most this fraction of the free memory (the simulator needs workspace)."""


def _check_inputs(n_qubits: int, precision: str) -> None:
    if n_qubits < 0:
        raise ValueError(f"n_qubits must be >= 0, got {n_qubits}")
    if precision not in BYTES_PER_AMPLITUDE:
        raise ValueError(
            f"Unknown precision {precision!r}. Use one of {sorted(BYTES_PER_AMPLITUDE)}."
        )


def estimate_memory_bytes(n_qubits: int, precision: str) -> int:
    """Return the statevector size in bytes: ``2**n * bytes_per_amplitude``."""
    _check_inputs(n_qubits, precision)
    return (2**n_qubits) * BYTES_PER_AMPLITUDE[precision]


def estimate_memory_gb(n_qubits: int, precision: str) -> float:
    """Return the statevector size in GiB (2**30 bytes)."""
    return estimate_memory_bytes(n_qubits, precision) / GIB


def estimate_peak_bytes(
    n_qubits: int, precision: str, factor: float = 1.0, base_bytes: int = 0
) -> int:
    """Return the estimated peak memory in bytes: ``factor * statevector + base_bytes``.

    The simulator holds more than one copy of the statevector, and the process has a
    fixed cost (the CUDA-Q import). The defaults give the plain statevector size.
    """
    if factor < 1.0:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if base_bytes < 0:
        raise ValueError(f"base_bytes must be >= 0, got {base_bytes}")
    return int(factor * estimate_memory_bytes(n_qubits, precision)) + base_bytes


def fits_in_memory(
    n_qubits: int,
    precision: str,
    available_bytes: Optional[int],
    headroom: float = DEFAULT_HEADROOM,
    factor: float = 1.0,
    base_bytes: int = 0,
) -> bool:
    """Return True if the estimated peak fits in ``headroom * available_bytes``.

    With the default ``factor`` and ``base_bytes`` the peak is the plain statevector.
    An unknown amount of memory (``None``) counts as "does not fit".
    """
    if available_bytes is None:
        return False
    peak = estimate_peak_bytes(n_qubits, precision, factor, base_bytes)
    return peak <= available_bytes * headroom


def available_ram_bytes() -> Optional[int]:
    """Return the available system RAM in bytes, or None if it is unknown."""
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def _query_nvidia_smi(fields: str) -> Optional[list[str]]:
    """Return the first GPU line of an ``nvidia-smi`` query, or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    lines = out.splitlines()
    if not lines:
        return None
    return [part.strip() for part in lines[0].split(",")]


def free_vram_bytes() -> Optional[int]:
    """Return the free GPU memory in bytes, or None if there is no NVIDIA GPU."""
    row = _query_nvidia_smi("memory.free")
    if row is None:
        return None
    try:
        return int(float(row[0])) * 1024 * 1024
    except ValueError:
        return None


@dataclass(frozen=True)
class HardwareInfo:
    """Snapshot of the machine that ran the experiments."""

    platform: str
    python: str
    cpu_count: int
    ram_total_gb: Optional[float]
    has_gpu: bool
    gpu_name: Optional[str]
    gpu_vram_gb: Optional[float]
    gpu_driver: Optional[str]


def _total_ram_gb() -> Optional[float]:
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) * 1024 / GIB, 1)
    except (OSError, ValueError):
        pass
    return None


def detect_hardware() -> HardwareInfo:
    """Detect the CPU, RAM and NVIDIA GPU at run time."""
    row = _query_nvidia_smi("name,memory.total,driver_version")
    gpu_name = gpu_vram = gpu_driver = None
    if row is not None and len(row) >= 3:
        gpu_name, gpu_driver = row[0], row[2]
        try:
            gpu_vram = round(float(row[1]) / 1024, 1)
        except ValueError:
            gpu_vram = None
    return HardwareInfo(
        platform=platform.platform(),
        python=platform.python_version(),
        cpu_count=os.cpu_count() or 1,
        ram_total_gb=_total_ram_gb(),
        has_gpu=row is not None,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram,
        gpu_driver=gpu_driver,
    )


def write_hardware_json(path: Path, info: Optional[HardwareInfo] = None) -> HardwareInfo:
    """Write the hardware snapshot to ``path`` as JSON and return it."""
    snapshot = info if info is not None else detect_hardware()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(snapshot), indent=2) + "\n", encoding="utf-8")
    return snapshot
