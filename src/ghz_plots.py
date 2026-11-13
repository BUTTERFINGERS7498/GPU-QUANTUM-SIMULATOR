"""Plots for the GHZ benchmark: runtime against n, and CPU/GPU speedup against n."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.ghz_benchmark import CPU_TARGET, GPU_TARGET, BenchmarkRow  # noqa: E402

COLORS = {CPU_TARGET: "#0072B2", GPU_TARGET: "#D55E00"}
LABELS = {CPU_TARGET: "CPU (qpp-cpu, fp64)", GPU_TARGET: "GPU (nvidia, fp32)"}
DPI = 200


def _by_target(rows: Sequence[BenchmarkRow], target: str) -> list[BenchmarkRow]:
    return sorted((r for r in rows if r.target == target), key=lambda r: r.n)


def plot_runtime(rows: Sequence[BenchmarkRow], path: Path) -> None:
    """Plot median runtime against n (log y-axis) for every target in ``rows``."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for target in (CPU_TARGET, GPU_TARGET):
        subset = _by_target(rows, target)
        if not subset:
            continue
        ax.errorbar(
            [r.n for r in subset], [r.median_s for r in subset],
            yerr=[r.std_s for r in subset], marker="o", capsize=3,
            color=COLORS[target], label=LABELS[target],
        )  # fmt: skip
    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits n")
    ax.set_ylabel("Median runtime per run (s, log scale)")
    ax.set_title("GHZ statevector simulation: runtime against qubit count")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ns = sorted({r.n for r in rows})
    if ns:
        ax.set_xticks(ns)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def speedups(rows: Sequence[BenchmarkRow]) -> list[tuple[int, float]]:
    """Return ``(n, cpu_median / gpu_median)`` for every n measured on both targets."""
    cpu = {r.n: r.median_s for r in rows if r.target == CPU_TARGET}
    gpu = {r.n: r.median_s for r in rows if r.target == GPU_TARGET}
    return [(n, cpu[n] / gpu[n]) for n in sorted(cpu.keys() & gpu.keys()) if gpu[n] > 0]


def plot_speedup(rows: Sequence[BenchmarkRow], path: Path) -> bool:
    """Plot the CPU/GPU speedup. Return False (no file) if no n has both targets."""
    points = speedups(rows)
    if not points:
        return False
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ns, values = zip(*points)
    ax.plot(ns, values, marker="s", color=COLORS[GPU_TARGET])
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1, label="break-even (1x)")
    for n, value in points:
        ax.annotate(f"{value:.1f}x", (n, value), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)  # fmt: skip
    ax.set_xlabel("Number of qubits n")
    ax.set_ylabel("Speedup = CPU runtime / GPU runtime")
    ax.set_title("GHZ statevector simulation: GPU speedup over CPU")
    ax.set_xticks(list(ns))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return True
