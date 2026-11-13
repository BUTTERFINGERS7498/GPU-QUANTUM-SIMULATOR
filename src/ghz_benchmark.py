"""GHZ-circuit statevector benchmark: CPU (``qpp-cpu``) against GPU (``nvidia``).

Each (target, n) measurement runs in its own subprocess. A crash or an
out-of-memory error on one size then cannot lose the rest of the sweep.

Usage::

    python -m src.ghz_benchmark                 # full sweep, n = 20..28
    python -m src.ghz_benchmark --resume        # keep CSV rows, measure the missing points
    python -m src.ghz_benchmark --only nvidia   # measure one target
    python -m src.ghz_benchmark --quick         # CPU-friendly reduced run
    python -m src.ghz_benchmark --plot-only     # redraw plots from the CSV
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import cudaq

from src.hardware import (
    available_ram_bytes,
    detect_hardware,
    GIB,
    estimate_memory_gb,
    estimate_peak_bytes,
    fits_in_memory,
    free_vram_bytes,
    write_hardware_json,
)

LOG = logging.getLogger("ghz_benchmark")

CPU_TARGET = "qpp-cpu"
GPU_TARGET = "nvidia"
TARGET_PRECISION: dict[str, str] = {CPU_TARGET: "fp64", GPU_TARGET: "fp32"}
"""``qpp-cpu`` always simulates in fp64. The ``nvidia`` target runs in fp32."""

PEAK_FACTOR: dict[str, float] = {CPU_TARGET: 2.5, GPU_TARGET: 1.0}
PEAK_BASE_BYTES: dict[str, int] = {CPU_TARGET: int(0.55 * GIB), GPU_TARGET: 0}
"""Peak memory of a worker: ``PEAK_FACTOR * statevector + PEAK_BASE_BYTES``.

``qpp-cpu`` was measured on the test machine (peak RSS of a worker, n = 4, 22, 24, 25):
about 2.5 x the fp64 statevector plus 0.55 GiB for the CUDA-Q import. The numbers for
larger n are an extrapolation. The ``nvidia`` factor is NOT measured: 1.0 is a placeholder.
"""

CSV_FIELDS = ("target", "n", "median_s", "std_s", "est_mem_gb", "runs", "shots")
FAILURE_FIELDS = ("target", "n", "reason")
FAILURES_FILE = "ghz_failures.csv"
"""Sizes whose worker failed. ``--resume`` does not try them again unless ``--retry-failed``."""
RESULT_PREFIX = "RESULT_JSON "
WORKER_TIMEOUT_S = 3600
CORRECTNESS_QUBITS = 8
CORRECTNESS_SHOTS = 4000
CORRECTNESS_TOLERANCE = 0.05
DEFAULT_OUTPUT_DIR = Path("results")
DEFAULT_QUICK_DIR = Path("results/quick")
"""``--quick`` writes here unless ``--output-dir`` is given, so it cannot replace a full run."""
DEFAULT_MAX_QUBITS = 28
"""The largest n that fits the reference laptop (7 GB RAM, 6 GiB GPU)."""
DEFAULT_SHOTS = 100
"""Few shots keep sampling cheap, so the time shows the statevector work."""


class GhzCorrectnessError(AssertionError):
    """The sampled GHZ distribution is not 50% all-0 and 50% all-1."""


@cudaq.kernel
def ghz(n: int):
    """Prepare an n-qubit GHZ state and measure every qubit."""
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1])
    mz(q)


@dataclass(frozen=True)
class BenchmarkRow:
    """One line of ``ghz_benchmark.csv``."""

    target: str
    n: int
    median_s: float
    std_s: float
    est_mem_gb: float
    runs: Optional[int] = None
    shots: Optional[int] = None
    """The settings of the measurement. ``None`` in a CSV written before these columns existed."""


# ----------------------------------------------------------------------------
# Statistics and correctness
# ----------------------------------------------------------------------------
def summarise_times(times: Sequence[float]) -> tuple[float, float]:
    """Return ``(median, sample standard deviation)`` of the timed runs."""
    if len(times) == 0:
        raise ValueError("times must contain at least one value")
    median = statistics.median(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return median, std


def validate_ghz_counts(
    counts: dict[str, int],
    n_qubits: int,
    tolerance: float = CORRECTNESS_TOLERANCE,
) -> None:
    """Raise ``GhzCorrectnessError`` unless only all-0 and all-1 appear, about 50% each."""
    zeros, ones = "0" * n_qubits, "1" * n_qubits
    stray = sorted(set(counts) - {zeros, ones})
    if stray:
        raise GhzCorrectnessError(f"unexpected bitstrings in GHZ sample: {stray[:5]}")
    total = sum(counts.values())
    if total == 0:
        raise GhzCorrectnessError("no shots were returned")
    for bitstring in (zeros, ones):
        fraction = counts.get(bitstring, 0) / total
        if abs(fraction - 0.5) > tolerance:
            raise GhzCorrectnessError(
                f"{bitstring} appears in {fraction:.1%} of shots, expected 50% +/- "
                f"{tolerance:.0%}"
            )


def run_ghz_correctness(
    n_qubits: int = CORRECTNESS_QUBITS,
    shots: int = CORRECTNESS_SHOTS,
    target: str = CPU_TARGET,
) -> dict[str, int]:
    """Sample the GHZ kernel, validate the counts and return them."""
    cudaq.set_target(target)
    result = cudaq.sample(ghz, n_qubits, shots_count=shots)
    counts = {bits: int(count) for bits, count in result.items()}
    validate_ghz_counts(counts, n_qubits)
    return counts


# ----------------------------------------------------------------------------
# Worker (runs in a child process)
# ----------------------------------------------------------------------------
def time_ghz(target: str, n_qubits: int, runs: int, shots: int, warmup: int = 1) -> list[float]:
    """Run ``warmup`` untimed and ``runs`` timed GHZ simulations. Return seconds."""
    cudaq.set_target(target)
    for _ in range(warmup):
        cudaq.sample(ghz, n_qubits, shots_count=shots)
    times: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        cudaq.sample(ghz, n_qubits, shots_count=shots)
        times.append(time.perf_counter() - start)
    return times


def _worker_main(args: argparse.Namespace) -> int:
    """Child-process entry point. Print one ``RESULT_JSON`` line on success."""
    if args.worker == "time":
        payload: object = {"times": time_ghz(args.target, args.n, args.runs, args.shots)}
    else:
        payload = {"counts": run_ghz_correctness(args.n, CORRECTNESS_SHOTS, args.target)}
    sys.stdout.write(RESULT_PREFIX + json.dumps(payload) + "\n")
    sys.stdout.flush()
    return 0


def _run_worker(mode: str, target: str, n_qubits: int, runs: int, shots: int) -> dict:
    """Run one measurement in a subprocess. Raise ``RuntimeError`` on failure."""
    cmd = [
        sys.executable, "-W", "ignore", "-m", "src.ghz_benchmark",
        "--worker", mode, "--target", target, "--n", str(n_qubits),
        "--runs", str(runs), "--shots", str(shots),
    ]  # fmt: skip
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=WORKER_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"timed out after {WORKER_TIMEOUT_S} s") from exc
    except OSError as exc:
        raise RuntimeError(f"worker could not start: {exc}") from exc
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            payload = _parse_payload(line[len(RESULT_PREFIX):], mode)
            if proc.returncode != 0:
                LOG.warning("worker %s %s n=%d exited with exit code %d after it printed "
                            "a result. The result is kept.", mode, target, n_qubits,
                            proc.returncode)
            return payload
    tail = (proc.stderr.strip().splitlines() or ["no error text"])[-1]
    raise RuntimeError(f"worker exit code {proc.returncode}: {tail[:200]}")


def _parse_payload(text: str, mode: str) -> dict:
    """Parse and check the JSON of a worker result line. Raise ``RuntimeError`` if it is bad."""
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise RuntimeError(f"worker result is not valid JSON: {text[:80]!r}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("worker result is not a JSON object")
    if mode == "time":
        times = payload.get("times")
        valid = (
            isinstance(times, list)
            and len(times) > 0
            and all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in times)
        )
        if not valid:
            raise RuntimeError("worker result has no valid list of times")
    else:
        counts = payload.get("counts")
        if not isinstance(counts, dict):
            raise RuntimeError("worker result has no valid counts")
    return payload


# ----------------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------------
def _memory_source(target: str) -> tuple[Callable[[], Optional[int]], str]:
    if target == GPU_TARGET:
        return free_vram_bytes, "VRAM"
    return available_ram_bytes, "RAM"


def benchmark_target(
    target: str,
    qubit_counts: Sequence[int],
    runs: int,
    shots: int,
    on_row: Optional[Callable[[BenchmarkRow], None]] = None,
    done: frozenset[int] = frozenset(),
    failed: frozenset[int] = frozenset(),
    on_fail: Optional[Callable[[str, int, str], None]] = None,
) -> list[BenchmarkRow]:
    """Benchmark one target over ``qubit_counts``. Skip sizes that do not fit.

    ``on_row`` is called after each finished point, so the caller can save it at once.
    ``on_fail(target, n, reason)`` is called when a worker fails.
    Sizes in ``done`` are not measured again. Sizes in ``failed`` failed in an earlier run:
    the sweep stops at the smallest one, as it does after a new failure.
    """
    precision = TARGET_PRECISION[target]
    free_bytes, label = _memory_source(target)
    rows: list[BenchmarkRow] = []
    for n in qubit_counts:
        if n in done:
            LOG.info("RESUME %s n=%d: already in the CSV, skipped", target, n)
            continue
        if n in failed:
            LOG.warning("SKIP %s n=%d: it failed in an earlier run. Larger sizes are skipped. "
                        "Use --retry-failed to try again.", target, n)
            break
        need_gb = estimate_memory_gb(n, precision)
        factor, base = PEAK_FACTOR[target], PEAK_BASE_BYTES[target]
        if not fits_in_memory(n, precision, free_bytes(), factor=factor, base_bytes=base):
            peak_gb = estimate_peak_bytes(n, precision, factor, base) / GIB
            LOG.warning("SKIP %s n=%d: statevector %.2f GiB (%s), estimated peak %.2f GiB, "
                        "more than the usable free %s", target, n, need_gb, precision,
                        peak_gb, label)
            continue
        try:
            times = _run_worker("time", target, n, runs, shots)["times"]
        except RuntimeError as exc:
            LOG.error("FAIL %s n=%d: %s. Larger sizes are skipped.", target, n, exc)
            if on_fail is not None:
                on_fail(target, n, str(exc))
            break
        median, std = summarise_times(times)
        LOG.info("%s n=%d median=%.4f s std=%.4f s (%.2f GiB)", target, n, median, std, need_gb)
        row = BenchmarkRow(target, n, median, std, need_gb, runs, shots)
        rows.append(row)
        if on_row is not None:
            on_row(row)
    return rows


def append_row(path: Path, row: BenchmarkRow) -> None:
    """Append one row to ``path``. Write the header first if the file is new or empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(CSV_FIELDS)
        elif not _ends_with_newline(path):
            handle.write("\n")  # a killed run left a partial last line: keep it apart
        writer.writerow(_row_fields(row))
        handle.flush()
        os.fsync(handle.fileno())


def _row_fields(r: BenchmarkRow) -> list[object]:
    """Return the CSV fields of one row, in ``CSV_FIELDS`` order. Unknown settings are empty."""
    return [r.target, r.n, f"{r.median_s:.6f}", f"{r.std_s:.6f}", f"{r.est_mem_gb:.4f}",
            "" if r.runs is None else r.runs, "" if r.shots is None else r.shots]


def append_failure(path: Path, target: str, n: int, reason: str) -> None:
    """Append one failed size to the failures file. Write the header first if the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(FAILURE_FIELDS)
        elif not _ends_with_newline(path):
            handle.write("\n")
        writer.writerow([target, n, reason])
        handle.flush()
        os.fsync(handle.fileno())


def read_failures(path: Path) -> dict[str, set[int]]:
    """Read the failures file. Return ``{target: {n, ...}}``. A bad row is skipped."""
    failures: dict[str, set[int]] = {}
    if not path.exists():
        return failures
    with path.open(newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            try:
                failures.setdefault(r["target"], set()).add(int(r["n"]))
            except (KeyError, TypeError, ValueError):
                LOG.warning("%s: failure row skipped: %r", path, r)
    return failures


def write_failures(path: Path, failures: dict[str, set[int]]) -> None:
    """Write ``failures`` (without the reasons) and replace the file. No failures: no file."""
    if not any(failures.values()):
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FAILURE_FIELDS)
        for target, sizes in sorted(failures.items()):
            for n in sorted(sizes):
                writer.writerow([target, n, "kept from an earlier run"])


def _ends_with_newline(path: Path) -> bool:
    """Return True if the last byte of the non-empty file ``path`` is a newline."""
    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) == b"\n"


def write_csv(path: Path, rows: Sequence[BenchmarkRow]) -> None:
    """Write the rows to ``path`` using the ``CSV_FIELDS`` columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_FIELDS)
        for r in rows:
            writer.writerow(_row_fields(r))


def _optional_int(text: Optional[str]) -> Optional[int]:
    """Return ``int(text)``, or None if the field is missing or empty."""
    return int(text) if text not in (None, "") else None


def read_csv(path: Path) -> list[BenchmarkRow]:
    """Read ``ghz_benchmark.csv`` back into rows.

    A row that is cut short or has an unreadable number is skipped with a warning.
    A killed run can leave such a row. ``--resume`` then measures that point again.
    """
    rows: list[BenchmarkRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for line_no, r in enumerate(csv.DictReader(handle), start=2):
            try:
                rows.append(BenchmarkRow(r["target"], int(r["n"]), float(r["median_s"]),
                                         float(r["std_s"]), float(r["est_mem_gb"]),
                                         _optional_int(r.get("runs")),
                                         _optional_int(r.get("shots"))))
            except (KeyError, TypeError, ValueError):
                LOG.warning("%s line %d: row skipped (incomplete or unreadable): %r",
                            path, line_no, r)
    return rows


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _positive_int(text: str) -> int:
    """Argparse type: an integer of at least 1."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--max-qubits", type=int, default=DEFAULT_MAX_QUBITS,
                   help=f"largest n (default {DEFAULT_MAX_QUBITS})")
    p.add_argument("--only", choices=(CPU_TARGET, GPU_TARGET),
                   help="measure only this target (default: every target that passes)")
    p.add_argument("--resume", action="store_true",
                   help="keep the rows already in the CSV and measure only the missing points")
    p.add_argument("--retry-failed", action="store_true",
                   help="measure again the sizes that failed in an earlier run (use with --resume)")
    p.add_argument("--min-qubits", type=int, default=20, help="smallest n (default 20)")
    p.add_argument("--step", type=_positive_int, default=1,
                   help="step between n values (default 1)")
    p.add_argument("--runs", type=_positive_int, default=5,
                   help="timed runs per point (default 5)")
    p.add_argument("--shots", type=_positive_int, default=DEFAULT_SHOTS, help="shots per run")
    p.add_argument("--quick", action="store_true",
                   help="reduced run: n=10..18 step 2, 3 runs (CPU, under 2 minutes)")
    p.add_argument("--plot-only", action="store_true", help="redraw plots from the CSV")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="where to write the files (default results, or results/quick with --quick)")
    p.add_argument("--worker", choices=("time", "correctness"), help=argparse.SUPPRESS)
    p.add_argument("--target", help=argparse.SUPPRESS)
    p.add_argument("--n", type=int, help=argparse.SUPPRESS)
    return p


def _apply_quick(args: argparse.Namespace) -> None:
    args.min_qubits, args.max_qubits, args.step, args.runs = 10, 18, 2, 3


def _make_plots(rows: Sequence[BenchmarkRow], out_dir: Path) -> None:
    from src.ghz_plots import plot_runtime, plot_speedup

    if not rows:
        LOG.warning("No benchmark rows. No plots were made.")
        return
    plot_runtime(rows, out_dir / "ghz_benchmark.png")
    LOG.info("Wrote %s", out_dir / "ghz_benchmark.png")
    if plot_speedup(rows, out_dir / "ghz_speedup.png"):
        LOG.info("Wrote %s", out_dir / "ghz_speedup.png")
    else:
        LOG.warning("No GPU rows. ghz_speedup.png was skipped.")


def _warn_if_settings_differ(rows: Sequence[BenchmarkRow], runs: int, shots: int) -> None:
    """Warn when kept rows were measured with other ``--runs`` or ``--shots`` values."""
    other = sorted({(r.runs, r.shots) for r in rows
                    if (r.runs, r.shots) != (runs, shots) and None not in (r.runs, r.shots)})
    if other:
        LOG.warning("The CSV has rows measured with different settings (runs, shots) %s. "
                    "This run uses (%d, %d). Rows with different settings stay in one file.",
                    other, runs, shots)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the benchmark. Return the process exit code."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = build_parser().parse_args(argv)
    if args.worker:
        return _worker_main(args)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    out_dir: Path = args.output_dir or (DEFAULT_QUICK_DIR if args.quick else DEFAULT_OUTPUT_DIR)
    csv_path = out_dir / "ghz_benchmark.csv"
    if args.plot_only:
        _make_plots(read_csv(csv_path), out_dir)
        return 0
    if args.quick:
        _apply_quick(args)

    info = write_hardware_json(out_dir / "hardware.json")
    LOG.info("Hardware: GPU=%s, RAM=%s GiB", info.gpu_name or "none", info.ram_total_gb)

    counts = run_ghz_correctness(target=CPU_TARGET)
    LOG.info("Correctness %s n=%d OK: %s", CPU_TARGET, CORRECTNESS_QUBITS, counts)
    targets = [CPU_TARGET]
    if args.only == CPU_TARGET:
        pass
    elif detect_hardware().has_gpu:
        try:
            gpu_counts = _run_worker("correctness", GPU_TARGET, CORRECTNESS_QUBITS, 1, 1)["counts"]
            validate_ghz_counts(gpu_counts, CORRECTNESS_QUBITS)
            LOG.info("Correctness %s n=%d OK: %s", GPU_TARGET, CORRECTNESS_QUBITS, gpu_counts)
            targets.append(GPU_TARGET)
        except (RuntimeError, GhzCorrectnessError) as exc:
            LOG.error("GPU target failed its correctness check: %s. GPU benchmark skipped.", exc)
    else:
        LOG.warning("No NVIDIA GPU found. The GPU benchmark is skipped.")

    if args.only == GPU_TARGET:
        targets = [t for t in targets if t == GPU_TARGET]
    if not targets:
        LOG.error("No target can run. The CSV %s is unchanged.", csv_path)
        return 1
    qubit_counts = list(range(args.min_qubits, args.max_qubits + 1, args.step))
    existing = read_csv(csv_path) if csv_path.exists() else []
    if args.resume:
        kept = existing
    else:
        # A fresh run replaces only the rows of the targets that run now.
        kept = [r for r in existing if r.target not in targets]
    if csv_path.exists() or kept:
        write_csv(csv_path, kept)  # also drops any partial row left by a killed run
    existing = kept if args.resume else []
    _warn_if_settings_differ(existing, args.runs, args.shots)

    failures_path = out_dir / FAILURES_FILE
    old_failures = read_failures(failures_path)
    if args.resume and not args.retry_failed:
        known_failures = {t: old_failures.get(t, set()) for t in targets}
    else:
        known_failures = {t: set() for t in targets}
    # Keep the failures of targets that do not run now. Forget those of the targets that do.
    write_failures(failures_path, {t: s for t, s in old_failures.items() if t not in targets}
                   | (known_failures if args.resume and not args.retry_failed else {}))

    for target in targets:
        done = frozenset(r.n for r in existing if r.target == target)
        benchmark_target(target, qubit_counts, args.runs, args.shots,
                         on_row=lambda row: append_row(csv_path, row), done=done,
                         failed=frozenset(known_failures[target]),
                         on_fail=lambda t, n, why: append_failure(failures_path, t, n, why))
    rows = read_csv(csv_path) if csv_path.exists() else []
    LOG.info("CSV %s has %d rows", csv_path, len(rows))
    _make_plots(rows, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
