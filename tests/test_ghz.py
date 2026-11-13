"""Tests for the GHZ benchmark: kernel correctness and statistics."""
import pytest

pytest.importorskip("cudaq")

from src.ghz_benchmark import (  # noqa: E402
    GhzCorrectnessError,
    run_ghz_correctness,
    summarise_times,
    validate_ghz_counts,
)


def test_ghz_sampling_gives_only_all_zero_and_all_one() -> None:
    counts = run_ghz_correctness(n_qubits=8, shots=4000, target="qpp-cpu")
    assert set(counts) == {"0" * 8, "1" * 8}
    assert sum(counts.values()) == 4000


def test_ghz_split_is_close_to_fifty_fifty() -> None:
    counts = run_ghz_correctness(n_qubits=8, shots=4000, target="qpp-cpu")
    for bitstring in ("0" * 8, "1" * 8):
        assert counts[bitstring] / 4000 == pytest.approx(0.5, abs=0.05)


def test_validator_rejects_a_stray_bitstring() -> None:
    with pytest.raises(GhzCorrectnessError, match="unexpected"):
        validate_ghz_counts({"00": 500, "11": 490, "01": 10}, n_qubits=2)


def test_validator_rejects_an_unbalanced_split() -> None:
    with pytest.raises(GhzCorrectnessError, match="50%"):
        validate_ghz_counts({"00": 900, "11": 100}, n_qubits=2, tolerance=0.1)


def test_validator_rejects_a_missing_outcome() -> None:
    with pytest.raises(GhzCorrectnessError):
        validate_ghz_counts({"00": 1000}, n_qubits=2)


def test_validator_accepts_a_good_result() -> None:
    validate_ghz_counts({"000": 505, "111": 495}, n_qubits=3)


def test_summarise_times_returns_median_and_sample_std() -> None:
    median, std = summarise_times([1.0, 2.0, 3.0, 4.0, 5.0])
    assert median == pytest.approx(3.0)
    assert std == pytest.approx(1.5811388, rel=1e-6)


def test_summarise_times_single_value_has_zero_std() -> None:
    assert summarise_times([2.5]) == (2.5, 0.0)


def test_summarise_times_empty_raises() -> None:
    with pytest.raises(ValueError):
        summarise_times([])
