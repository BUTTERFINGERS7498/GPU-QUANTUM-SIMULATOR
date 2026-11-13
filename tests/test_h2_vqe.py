"""Tests for the H2 VQE experiment."""
from pathlib import Path

import pytest

pytest.importorskip("cudaq")
pytest.importorskip("openfermionpyscf")

from src import h2_vqe as h2  # noqa: E402
from src.h2_vqe import (  # noqa: E402
    CHEMICAL_ACCURACY_MHA,
    DEFAULT_TOLERANCE,
    EQUILIBRIUM_R,
    PesRow,
    VqeResult,
    build_parser,
    bond_length_grid,
    build_h2,
    count_pauli_terms,
    error_mha,
    find_equilibrium,
    hydrogen_atom_energy,
    main,
    read_pes_csv,
    run_sweep,
    run_vqe,
    write_pes_csv,
)


@pytest.fixture(scope="module")
def h2_equilibrium():
    return build_h2(EQUILIBRIUM_R)


@pytest.fixture(scope="module")
def vqe_equilibrium(h2_equilibrium):
    return run_vqe(h2_equilibrium.hamiltonian)


def test_hamiltonian_has_15_pauli_terms_including_identity(h2_equilibrium) -> None:
    assert count_pauli_terms(h2_equilibrium.hamiltonian) == 15


def test_hamiltonian_acts_on_4_qubits(h2_equilibrium) -> None:
    assert h2_equilibrium.hamiltonian.get_qubit_count() == 4


def test_reference_energies_at_equilibrium(h2_equilibrium) -> None:
    assert h2_equilibrium.fci_energy == pytest.approx(-1.1373, abs=1e-4)
    assert h2_equilibrium.fci_energy < h2_equilibrium.hf_energy


@pytest.mark.slow
def test_vqe_at_equilibrium_is_within_chemical_accuracy(h2_equilibrium, vqe_equilibrium) -> None:
    error = error_mha(vqe_equilibrium.energy, h2_equilibrium.fci_energy)
    assert error < CHEMICAL_ACCURACY_MHA


@pytest.mark.slow
def test_vqe_obeys_the_variational_principle(h2_equilibrium, vqe_equilibrium) -> None:
    assert vqe_equilibrium.energy >= h2_equilibrium.fci_energy - 1e-6


@pytest.mark.slow
def test_vqe_records_the_energy_at_every_iteration(vqe_equilibrium) -> None:
    assert len(vqe_equilibrium.history) == vqe_equilibrium.n_evals > 3
    assert min(vqe_equilibrium.history) == pytest.approx(vqe_equilibrium.energy, abs=1e-9)


@pytest.mark.slow
def test_warm_start_converges_in_fewer_evaluations(vqe_equilibrium) -> None:
    near = build_h2(EQUILIBRIUM_R + 0.05)
    cold = run_vqe(near.hamiltonian)
    warm = run_vqe(near.hamiltonian, x0=vqe_equilibrium.params)
    assert warm.n_evals < cold.n_evals  # strictly fewer: equal counts would hide a useless x0
    assert warm.energy == pytest.approx(cold.energy, abs=1e-5)


def test_vqe_reports_convergence(vqe_equilibrium) -> None:
    assert vqe_equilibrium.converged is True
    assert isinstance(vqe_equilibrium.status, int)  # the code value depends on the SciPy version
    assert vqe_equilibrium.message


def test_vqe_reports_a_stop_at_maxiter(h2_equilibrium) -> None:
    result = run_vqe(h2_equilibrium.hamiltonian, maxiter=5)
    assert result.converged is False
    assert result.n_evals <= 5


def test_sweep_warns_and_flags_a_point_that_did_not_converge(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="h2_vqe"):
        rows = run_sweep([0.7], maxiter=5)
    assert rows[0].converged is False
    assert "did not converge" in caplog.text


def test_pes_csv_without_a_converged_column_reads_as_converged(tmp_path: Path) -> None:
    path = tmp_path / "old.csv"
    path.write_text("R,E_vqe,E_fci,E_hf,abs_error_mHa\n0.5,-1.1,-1.11,-1.05,10.0\n",
                    encoding="utf-8")
    assert read_pes_csv(path) == [PesRow(0.5, -1.1, -1.11, -1.05, 10.0)]


def test_pes_csv_keeps_a_false_converged_flag(tmp_path: Path) -> None:
    rows = [PesRow(0.5, -1.10, -1.11, -1.05, 10.0, converged=False)]
    path = tmp_path / "h2_pes.csv"
    write_pes_csv(path, rows)
    assert read_pes_csv(path) == rows


# ---- --tol: the COBYLA tolerance ----------------------------------------------
def _fake_vqe(calls: list[dict]):
    """A run_vqe that records its keyword arguments and returns a fixed result."""

    def fake(hamiltonian, x0=None, maxiter=h2.DEFAULT_MAXITER, tol=DEFAULT_TOLERANCE,
             target="qpp-cpu"):
        calls.append({"tol": tol, "maxiter": maxiter, "target": target})
        return VqeResult(energy=-1.1, params=(0.0, 0.0, 0.0), history=(-1.1,), n_evals=1,
                         converged=True, status=0, message="ok")

    return fake


def test_the_tolerance_defaults_to_the_constant() -> None:
    assert build_parser().parse_args([]).tol == DEFAULT_TOLERANCE


def test_the_parser_reads_a_tolerance() -> None:
    assert build_parser().parse_args(["--tol", "1e-6"]).tol == 1e-6


@pytest.mark.parametrize("value", ["0", "-1e-6", "nan", "inf"])
def test_the_parser_rejects_a_tolerance_that_is_not_positive_and_finite(value: str) -> None:
    with pytest.raises(SystemExit) as info:
        build_parser().parse_args(["--tol", value])
    assert info.value.code == 2


def test_run_sweep_passes_the_tolerance_to_every_point(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(h2, "run_vqe", _fake_vqe(calls))
    run_sweep([0.6, 0.7], tol=1e-6)
    assert [c["tol"] for c in calls] == [1e-6, 1e-6]


def test_run_sweep_uses_the_default_tolerance_without_an_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(h2, "run_vqe", _fake_vqe(calls))
    run_sweep([0.7])
    assert calls[0]["tol"] == DEFAULT_TOLERANCE


def test_main_uses_the_tolerance_for_every_vqe_run_and_records_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(h2, "run_vqe", _fake_vqe(calls))
    assert main(["--quick", "--tol", "1e-6", "--output-dir", str(tmp_path)]) == 0
    assert len(calls) > 5  # the equilibrium run, the sweep and the equilibrium search
    assert {c["tol"] for c in calls} == {1e-6}
    import json

    summary = json.loads((tmp_path / "h2_summary.json").read_text(encoding="utf-8"))
    assert summary["tol"] == 1e-6


def test_bond_length_grid_matches_the_specification() -> None:
    grid = bond_length_grid()
    assert len(grid) == 45
    assert grid[0] == 0.30
    assert grid[-1] == 2.50
    assert grid[1] == 0.35


def test_error_is_reported_in_millihartree() -> None:
    assert error_mha(-1.1370, -1.1373) == pytest.approx(0.3)
    assert error_mha(-1.1373, -1.1370) == pytest.approx(0.3)


def test_hydrogen_atom_energy_in_sto3g() -> None:
    assert hydrogen_atom_energy() == pytest.approx(-0.4666, abs=1e-3)


def test_find_equilibrium_locates_the_minimum_of_a_parabola() -> None:
    r_eq, e_min = find_equilibrium(lambda r: (r - 0.74) ** 2 - 1.1, lo=0.5, hi=1.0)
    assert r_eq == pytest.approx(0.74, abs=1e-3)
    assert e_min == pytest.approx(-1.1, abs=1e-6)


def test_pes_csv_round_trip(tmp_path: Path) -> None:
    rows = [PesRow(0.5, -1.10, -1.11, -1.05, 10.0), PesRow(0.6, -1.12, -1.13, -1.07, 10.0)]
    path = tmp_path / "h2_pes.csv"
    write_pes_csv(path, rows)
    assert path.read_text(encoding="utf-8").splitlines()[0] == (
        "R,E_vqe,E_fci,E_hf,abs_error_mHa,converged"
    )
    assert read_pes_csv(path) == rows


@pytest.mark.slow
def test_sweep_warm_starts_and_stays_accurate() -> None:
    rows = run_sweep([0.6, 0.65, 0.7])
    assert [r.r for r in rows] == [0.6, 0.65, 0.7]
    assert all(r.abs_error_mha < CHEMICAL_ACCURACY_MHA for r in rows)


@pytest.mark.slow
def test_quick_cli_writes_every_artifact(tmp_path: Path) -> None:
    assert main(["--quick", "--output-dir", str(tmp_path)]) == 0
    for name in ("h2_pes.csv", "h2_pes.png", "h2_error.png", "vqe_convergence.png",
                 "h2_summary.json"):  # fmt: skip
        assert (tmp_path / name).is_file(), name
