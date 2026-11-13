"""Tests for the memory estimator and hardware helpers."""
import pytest

from src.hardware import (
    BYTES_PER_AMPLITUDE,
    estimate_memory_bytes,
    estimate_memory_gb,
    estimate_peak_bytes,
    fits_in_memory,
)


def test_fp32_uses_8_bytes_per_amplitude() -> None:
    assert estimate_memory_bytes(30, "fp32") == 2**30 * 8


def test_fp64_uses_16_bytes_per_amplitude() -> None:
    assert estimate_memory_bytes(30, "fp64") == 2**30 * 16


def test_memory_doubles_with_each_qubit() -> None:
    assert estimate_memory_bytes(21, "fp32") == 2 * estimate_memory_bytes(20, "fp32")


def test_gigabyte_conversion_uses_gib() -> None:
    assert estimate_memory_gb(30, "fp32") == pytest.approx(8.0)
    assert estimate_memory_gb(30, "fp64") == pytest.approx(16.0)


def test_unknown_precision_raises() -> None:
    with pytest.raises(ValueError, match="precision"):
        estimate_memory_bytes(10, "fp16")


def test_negative_qubit_count_raises() -> None:
    with pytest.raises(ValueError, match="n_qubits"):
        estimate_memory_bytes(-1, "fp32")


def test_fits_when_memory_is_enough() -> None:
    avail = estimate_memory_bytes(20, "fp32") * 10
    assert fits_in_memory(20, "fp32", avail)


def test_does_not_fit_when_memory_is_too_small() -> None:
    avail = estimate_memory_bytes(30, "fp32") // 2
    assert not fits_in_memory(30, "fp32", avail)


def test_headroom_rejects_a_borderline_case() -> None:
    need = estimate_memory_bytes(25, "fp32")
    assert not fits_in_memory(25, "fp32", need, headroom=0.9)
    assert fits_in_memory(25, "fp32", need, headroom=1.0)


def test_unknown_available_memory_is_treated_as_not_fitting() -> None:
    assert not fits_in_memory(10, "fp32", None)


GIB = 2**30


def test_peak_is_factor_times_state_vector_plus_base() -> None:
    state = estimate_memory_bytes(25, "fp64")
    assert estimate_peak_bytes(25, "fp64", factor=2.5, base_bytes=GIB // 2) == int(2.5 * state) + GIB // 2


def test_peak_defaults_to_the_state_vector() -> None:
    assert estimate_peak_bytes(20, "fp32") == estimate_memory_bytes(20, "fp32")


@pytest.mark.parametrize("factor, base", [(0.5, 0), (1.0, -1)])
def test_peak_rejects_a_factor_below_one_or_a_negative_base(factor: float, base: int) -> None:
    with pytest.raises(ValueError):
        estimate_peak_bytes(20, "fp64", factor=factor, base_bytes=base)


def test_fits_uses_the_peak_factor_and_base() -> None:
    free = int(7.3 * GIB)  # the free RAM of the 7.8 GiB test machine
    assert fits_in_memory(27, "fp64", free, factor=2.5, base_bytes=int(0.55 * GIB))
    assert not fits_in_memory(28, "fp64", free, factor=2.5, base_bytes=int(0.55 * GIB))
    assert fits_in_memory(28, "fp64", free)  # the old 1x check wrongly allows n=28


def test_precision_table_is_consistent() -> None:
    assert BYTES_PER_AMPLITUDE == {"fp32": 8, "fp64": 16}
