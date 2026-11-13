"""Plots for the H2 VQE experiment."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.h2_vqe import CHEMICAL_ACCURACY_MHA, PesRow  # noqa: E402

BLUE, ORANGE, GREY, GREEN = "#0072B2", "#D55E00", "#555555", "#009E73"
DPI = 200
ZERO_ERROR_FLOOR_MHA = 1e-16
"""Where an error of exactly 0 is drawn if every error is 0. A log axis cannot show 0.

An error above 0 is never changed. When some errors are 0 and others are not, a zero is
drawn one decade below the smallest error that is above 0.
"""
AXIS_MARGIN_DECADES = 1.0
"""The y-axis starts this many decades below the lowest plotted value."""


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_convergence(history: Sequence[float], fci_energy: float, path: Path) -> None:
    """Plot the VQE energy at every function evaluation, with the exact FCI energy."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(history) + 1), history, color=BLUE, marker=".", label="VQE (UCCSD, COBYLA)")
    ax.axhline(fci_energy, color=ORANGE, linestyle="--", label=f"FCI = {fci_energy:.6f} Ha")
    ax.set_xlabel("Energy evaluation")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title("H$_2$ VQE convergence at R = 0.735 Å (STO-3G)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, path)


def plot_pes(rows: Sequence[PesRow], dissociation_limit: float, path: Path) -> None:
    """Plot the potential energy curve: VQE points, FCI line, HF dashed, 2 E(H) asymptote."""
    ordered = sorted(rows, key=lambda r: r.r)
    radii = [r.r for r in ordered]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(radii, [r.e_fci for r in ordered], color=GREY, linewidth=2, label="FCI (exact)")
    ax.plot(radii, [r.e_hf for r in ordered], color=ORANGE, linestyle="--", label="Hartree-Fock")
    ax.scatter(radii, [r.e_vqe for r in ordered], color=BLUE, s=22, zorder=3, label="VQE (UCCSD)")
    ax.axhline(dissociation_limit, color=GREEN, linestyle="--", linewidth=1,
               label=f"2 × E(H atom) = {dissociation_limit:.4f} Ha")  # fmt: skip
    ax.set_xlabel("H–H bond length R (Å)")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title("H$_2$ potential energy curve (STO-3G)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, path)


def _log_safe_errors(errors: Sequence[float]) -> tuple[list[float], float | None]:
    """Return errors that a log axis can draw, and where zeros were drawn (None: no zeros).

    Errors above 0 are returned unchanged. Only an error of exactly 0 is moved.
    """
    positive = [e for e in errors if e > 0]
    if len(positive) == len(errors):
        return list(errors), None
    zero_at = min(positive) / 10 if positive else ZERO_ERROR_FLOOR_MHA
    return [e if e > 0 else zero_at for e in errors], zero_at


def plot_error(rows: Sequence[PesRow], path: Path) -> None:
    """Plot ``|E_vqe - E_fci|`` in mHa against R, with the chemical-accuracy band shaded."""
    ordered = sorted(rows, key=lambda r: r.r)
    radii = [r.r for r in ordered]
    errors, zero_plotted_at = _log_safe_errors([r.abs_error_mha for r in ordered])
    axis_bottom = min(errors) / 10**AXIS_MARGIN_DECADES
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_yscale("log")
    ax.axhspan(axis_bottom, CHEMICAL_ACCURACY_MHA, color=GREEN, alpha=0.18,
               label=f"chemical accuracy (< {CHEMICAL_ACCURACY_MHA} mHa)")  # fmt: skip
    ax.axhline(CHEMICAL_ACCURACY_MHA, color=GREEN, linestyle="--", linewidth=1)
    ax.plot(radii, errors, color=BLUE, marker="o", markersize=4, label="|E_VQE − E_FCI|")
    ax.set_ylim(axis_bottom, max(10.0, max(errors) * 3))
    if zero_plotted_at is not None:
        ax.text(0.99, 0.02, f"points at {zero_plotted_at:.0e}: error exactly 0",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=GREY)
    ax.set_xlabel("H–H bond length R (Å)")
    ax.set_ylabel("Absolute error (mHa, log scale)")
    ax.set_title("H$_2$ VQE error against exact FCI")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")
    _save(fig, path)
