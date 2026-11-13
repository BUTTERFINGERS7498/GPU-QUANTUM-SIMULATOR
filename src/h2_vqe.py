"""VQE with a UCCSD ansatz for H2 (STO-3G, Jordan-Wigner, 4 qubits).

The script finds the ground-state energy at one bond length, then sweeps the
bond length and compares every VQE energy with exact FCI and with Hartree-Fock.

Usage::

    python -m src.h2_vqe                # full sweep, R = 0.30..2.50 A step 0.05
    python -m src.h2_vqe --quick        # reduced sweep (CPU, under 2 minutes)
    python -m src.h2_vqe --plot-only    # redraw plots from the saved results
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import cudaq
import numpy as np
from cudaq import chemistry
from scipy.optimize import minimize, minimize_scalar

LOG = logging.getLogger("h2_vqe")

N_ELECTRONS = 2
N_QUBITS = 4
BASIS = "sto-3g"
EQUILIBRIUM_R = 0.735
"""Experimental-like H2 bond length in angstrom."""
HARTREE_TO_MHA = 1000.0
CHEMICAL_ACCURACY_MHA = 1.6
DEFAULT_INITIAL_PARAMETER = 0.01
COLD_START_RHOBEG = 1.0
WARM_START_RHOBEG = 0.1
DEFAULT_TOLERANCE = 1e-9
DEFAULT_MAXITER = 1000
EQUILIBRIUM_SEARCH_WINDOW = (0.60, 0.90)
PES_FIELDS = ("R", "E_vqe", "E_fci", "E_hf", "abs_error_mHa", "converged")


@cudaq.kernel
def h2_ansatz(thetas: list[float]):
    """Hartree-Fock reference state followed by the UCCSD circuit."""
    q = cudaq.qvector(4)
    x(q[0])
    x(q[1])
    cudaq.kernels.uccsd(q, thetas, 2, 4)


@dataclass(frozen=True)
class H2System:
    """The H2 Hamiltonian and its exact reference energies at one bond length."""

    r: float
    hamiltonian: cudaq.SpinOperator
    hf_energy: float
    fci_energy: float


@dataclass(frozen=True)
class VqeResult:
    """Outcome of one VQE optimisation."""

    energy: float
    params: tuple[float, ...]
    history: tuple[float, ...]
    n_evals: int
    converged: bool
    """True if SciPy reports success. False if COBYLA stopped at ``maxiter`` or failed."""
    status: int
    message: str


@dataclass(frozen=True)
class PesRow:
    """One bond length of the potential energy surface. Energies are in Ha."""

    r: float
    e_vqe: float
    e_fci: float
    e_hf: float
    abs_error_mha: float
    converged: bool = True


# ----------------------------------------------------------------------------
# Molecule
# ----------------------------------------------------------------------------
def bond_length_grid(start: float = 0.30, stop: float = 2.50, step: float = 0.05) -> tuple[float, ...]:
    """Return the bond lengths ``start, start+step, ..., stop`` (angstrom)."""
    count = int(round((stop - start) / step)) + 1
    return tuple(round(start + i * step, 10) for i in range(count))


def build_h2(r: float) -> H2System:
    """Build the H2 Hamiltonian at bond length ``r`` with PySCF, OpenFermion and Jordan-Wigner.

    CUDA-Q's built-in helper runs PySCF (Hartree-Fock and FCI), then maps the
    fermionic Hamiltonian to qubits.
    """
    geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, float(r)))]
    hamiltonian, molecule = chemistry.create_molecular_hamiltonian(geometry, BASIS, 1, 0)
    return H2System(r, hamiltonian, float(molecule.hf_energy), float(molecule.fci_energy))


def count_pauli_terms(hamiltonian: cudaq.SpinOperator) -> int:
    """Return the number of Pauli terms, the identity included."""
    return int(hamiltonian.get_term_count())


def hydrogen_atom_energy() -> float:
    """Return the STO-3G UHF energy of one H atom in Ha (the dissociation limit is twice this)."""
    from pyscf import gto, scf

    atom = gto.M(atom="H 0 0 0", basis=BASIS, spin=1, verbose=0)
    return float(scf.UHF(atom).kernel())


def error_mha(e_vqe: float, e_fci: float) -> float:
    """Return ``|E_vqe - E_fci|`` in milli-Hartree."""
    return abs(e_vqe - e_fci) * HARTREE_TO_MHA


# ----------------------------------------------------------------------------
# VQE
# ----------------------------------------------------------------------------
def run_vqe(
    hamiltonian: cudaq.SpinOperator,
    x0: Optional[Sequence[float]] = None,
    maxiter: int = DEFAULT_MAXITER,
    tol: float = DEFAULT_TOLERANCE,
    target: str = "qpp-cpu",
) -> VqeResult:
    """Minimise ``<psi(theta)|H|psi(theta)>`` with COBYLA. Record every energy evaluation.

    ``x0`` is a warm start. Without it the parameters start at a small value.
    A warm start also uses a smaller initial step, because the optimum is near.
    """
    cudaq.set_target(target)
    n_params = cudaq.kernels.uccsd_num_parameters(N_ELECTRONS, N_QUBITS)
    if x0 is None:
        start = np.full(n_params, DEFAULT_INITIAL_PARAMETER)
        rhobeg = COLD_START_RHOBEG
    else:
        start = np.asarray(x0, dtype=float)
        if start.shape != (n_params,):
            raise ValueError(f"x0 needs {n_params} values, got {start.shape}")
        rhobeg = WARM_START_RHOBEG

    history: list[float] = []
    best: dict[str, object] = {"energy": np.inf, "params": tuple(start)}

    def cost(theta: np.ndarray) -> float:
        params = [float(v) for v in theta]
        energy = float(cudaq.observe(h2_ansatz, hamiltonian, params).expectation())
        history.append(energy)
        if energy < best["energy"]:  # type: ignore[operator]
            best["energy"], best["params"] = energy, tuple(params)
        return energy

    outcome = minimize(cost, start, method="COBYLA",
                       options={"maxiter": maxiter, "tol": tol, "rhobeg": rhobeg})  # fmt: skip
    return VqeResult(
        energy=float(best["energy"]),  # type: ignore[arg-type]
        params=best["params"],  # type: ignore[arg-type]
        history=tuple(history),
        n_evals=len(history),
        converged=bool(outcome.success),
        status=int(outcome.status),
        message=str(outcome.message),
    )


def run_sweep(
    radii: Sequence[float],
    target: str = "qpp-cpu",
    maxiter: int = DEFAULT_MAXITER,
    tol: float = DEFAULT_TOLERANCE,
) -> list[PesRow]:
    """Run VQE at every bond length. Each point starts from the previous optimum."""
    rows: list[PesRow] = []
    params: Optional[tuple[float, ...]] = None
    for r in radii:
        system = build_h2(r)
        result = run_vqe(system.hamiltonian, x0=params, maxiter=maxiter, tol=tol, target=target)
        params = result.params
        row = PesRow(r, result.energy, system.fci_energy, system.hf_energy,
                     error_mha(result.energy, system.fci_energy), result.converged)  # fmt: skip
        LOG.info("R=%.2f A  E_vqe=%.6f  E_fci=%.6f  error=%.2e mHa  (%d evals)",
                 r, row.e_vqe, row.e_fci, row.abs_error_mha, result.n_evals)
        if not result.converged:
            LOG.warning("R=%.2f A: optimizer did not converge (status %d: %s). "
                        "Its parameters start the next point.", r, result.status, result.message)
        rows.append(row)
    return rows


def find_equilibrium(
    energy_fn: Callable[[float], float],
    lo: float = EQUILIBRIUM_SEARCH_WINDOW[0],
    hi: float = EQUILIBRIUM_SEARCH_WINDOW[1],
    xatol: float = 1e-5,
) -> tuple[float, float]:
    """Return ``(R_eq, E_min)``: the bond length that minimises ``energy_fn`` in ``[lo, hi]``."""
    result = minimize_scalar(energy_fn, bounds=(lo, hi), method="bounded",
                             options={"xatol": xatol})  # fmt: skip
    return float(result.x), float(result.fun)


# ----------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------
def write_pes_csv(path: Path, rows: Sequence[PesRow]) -> None:
    """Write the potential energy surface to ``path`` (full-precision floats)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(PES_FIELDS)
        for r in rows:
            writer.writerow([repr(r.r), repr(r.e_vqe), repr(r.e_fci), repr(r.e_hf),
                             repr(r.abs_error_mha), int(r.converged)])  # fmt: skip


def read_pes_csv(path: Path) -> list[PesRow]:
    """Read ``h2_pes.csv`` back into rows. A file without the ``converged`` column counts as converged."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            PesRow(float(d["R"]), float(d["E_vqe"]), float(d["E_fci"]),
                   float(d["E_hf"]), float(d["abs_error_mHa"]),
                   d.get("converged", "1") != "0")
            for d in csv.DictReader(handle)
        ]


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _positive_finite_float(text: str) -> float:
    """Argparse type: a finite number above 0."""
    value = float(text)
    if not (value > 0 and value != float("inf")):  # a NaN fails ``value > 0``
        raise argparse.ArgumentTypeError(f"must be a finite number above 0, got {text}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--quick", action="store_true",
                   help="reduced sweep: R = 0.50..1.50 step 0.25 (CPU, under 2 minutes)")
    p.add_argument("--target", default="qpp-cpu", help="CUDA-Q target (default qpp-cpu)")
    p.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    p.add_argument("--tol", type=_positive_finite_float, default=DEFAULT_TOLERANCE,
                   help=f"COBYLA final trust-region size, a step size in theta "
                        f"(default {DEFAULT_TOLERANCE:g})")
    p.add_argument("--plot-only", action="store_true", help="redraw plots from saved results")
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    return p


def _make_plots(rows: Sequence[PesRow], summary: dict, out_dir: Path) -> None:
    from src.h2_plots import plot_convergence, plot_error, plot_pes

    plot_convergence(summary["convergence_history"], summary["fci_energy_equilibrium"],
                     out_dir / "vqe_convergence.png")  # fmt: skip
    plot_pes(rows, summary["dissociation_limit_ha"], out_dir / "h2_pes.png")
    plot_error(rows, out_dir / "h2_error.png")
    LOG.info("Wrote plots to %s", out_dir)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the experiment. Return the process exit code."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, summary_path = out_dir / "h2_pes.csv", out_dir / "h2_summary.json"

    if args.plot_only:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _make_plots(read_pes_csv(csv_path), summary, out_dir)
        return 0

    equilibrium = build_h2(EQUILIBRIUM_R)
    n_terms = count_pauli_terms(equilibrium.hamiltonian)
    LOG.info("H2 / %s / Jordan-Wigner: %d qubits, %d Pauli terms (identity included)",
             BASIS, equilibrium.hamiltonian.get_qubit_count(), n_terms)  # fmt: skip
    converged = run_vqe(equilibrium.hamiltonian, maxiter=args.maxiter, tol=args.tol,
                        target=args.target)
    LOG.info("R=%.3f A: E_vqe=%.8f E_fci=%.8f error=%.5f mHa after %d evaluations",
             EQUILIBRIUM_R, converged.energy, equilibrium.fci_energy,
             error_mha(converged.energy, equilibrium.fci_energy), converged.n_evals)  # fmt: skip

    radii = (0.5, 0.75, 1.0, 1.25, 1.5) if args.quick else bond_length_grid()
    rows = run_sweep(radii, target=args.target, maxiter=args.maxiter, tol=args.tol)
    write_pes_csv(csv_path, rows)

    warm = {"params": converged.params}

    def vqe_energy(r: float) -> float:
        result = run_vqe(build_h2(r).hamiltonian, x0=warm["params"], maxiter=args.maxiter,
                         tol=args.tol, target=args.target)  # fmt: skip
        warm["params"] = result.params
        return result.energy

    r_eq, e_min = find_equilibrium(vqe_energy)
    LOG.info("VQE equilibrium: R=%.4f A, E_min=%.6f Ha", r_eq, e_min)

    atom = hydrogen_atom_energy()
    summary = {
        "n_pauli_terms": n_terms,
        "target": args.target,
        "tol": args.tol,
        "n_points": len(rows),
        "max_error_mha": max(r.abs_error_mha for r in rows),
        "n_not_converged": sum(1 for r in rows if not r.converged),
        "equilibrium_converged": converged.converged,
        "vqe_equilibrium_r": r_eq,
        "vqe_equilibrium_energy": e_min,
        "fci_energy_equilibrium": equilibrium.fci_energy,
        "error_at_equilibrium_mha": error_mha(converged.energy, equilibrium.fci_energy),
        "h_atom_energy_ha": atom,
        "dissociation_limit_ha": 2 * atom,
        "convergence_evals": converged.n_evals,
        "convergence_history": list(converged.history),
        "quick": bool(args.quick),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _make_plots(rows, summary, out_dir)
    LOG.info("Maximum |E_vqe - E_fci| over the sweep: %.5f mHa", summary["max_error_mha"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
