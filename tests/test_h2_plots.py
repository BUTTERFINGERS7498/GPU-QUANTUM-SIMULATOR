"""Tests for the H2 plots: the plotted values must be the real values.

An earlier version clipped every error to a fixed floor of 1e-6 mHa. The real errors
are 1e-12 to 1e-10 mHa, so the plot showed a flat line and hid the data. A test that only
checks for a valid PNG file did not find this, so these tests read the data out of the figure.
"""
from pathlib import Path

import pytest

pytest.importorskip("cudaq")
pytest.importorskip("matplotlib")

from src import h2_plots  # noqa: E402
from src.h2_vqe import CHEMICAL_ACCURACY_MHA, PesRow  # noqa: E402

ERROR_LABEL = "|E_VQE − E_FCI|"


def _row(r: float, error_mha: float) -> PesRow:
    return PesRow(r, -1.1, -1.1, -1.0, error_mha)


@pytest.fixture
def captured_figure(monkeypatch: pytest.MonkeyPatch) -> list:
    """Replace ``_save`` so the test can read the figure. Return the list of figures."""
    figures: list = []

    def keep(fig, path) -> None:
        figures.append(fig)

    monkeypatch.setattr(h2_plots, "_save", keep)
    return figures


def _error_line(fig):
    (line,) = [ln for ln in fig.axes[0].get_lines() if ln.get_label() == ERROR_LABEL]
    return line


def test_plotted_errors_are_the_real_errors_when_they_are_tiny(captured_figure: list) -> None:
    errors = [8.4e-12, 5.1e-12, 2.6e-10]
    h2_plots.plot_error([_row(0.3 + 0.1 * i, e) for i, e in enumerate(errors)], Path("x.png"))
    assert list(_error_line(captured_figure[0]).get_ydata()) == errors


def test_the_y_axis_reaches_below_the_smallest_error(captured_figure: list) -> None:
    errors = [8.4e-12, 5.1e-12, 2.6e-10]
    h2_plots.plot_error([_row(0.3 + 0.1 * i, e) for i, e in enumerate(errors)], Path("x.png"))
    low, high = captured_figure[0].axes[0].get_ylim()
    assert low < min(errors)
    assert high > CHEMICAL_ACCURACY_MHA


def test_large_errors_are_not_changed(captured_figure: list) -> None:
    errors = [0.5, 3.0, 12.0]
    h2_plots.plot_error([_row(0.3 + 0.1 * i, e) for i, e in enumerate(errors)], Path("x.png"))
    fig = captured_figure[0]
    assert list(_error_line(fig).get_ydata()) == errors
    assert fig.axes[0].get_ylim()[1] > max(errors)


def test_an_error_of_exactly_zero_is_drawn_below_the_smallest_real_error(
    captured_figure: list,
) -> None:
    rows = [_row(0.3, 0.0), _row(0.4, 3e-9), _row(0.5, 5e-10)]
    h2_plots.plot_error(rows, Path("x.png"))
    ydata = list(_error_line(captured_figure[0]).get_ydata())
    assert ydata[1:] == [3e-9, 5e-10]  # real values stay
    assert 0 < ydata[0] < 5e-10  # a log axis cannot show 0


def test_an_axis_note_says_when_a_zero_is_drawn_at_a_floor(captured_figure: list) -> None:
    h2_plots.plot_error([_row(0.3, 0.0), _row(0.4, 3e-9)], Path("x.png"))
    texts = [t.get_text() for t in captured_figure[0].axes[0].texts]
    assert any("exactly 0" in t for t in texts)


def test_there_is_no_axis_note_when_no_error_is_zero(captured_figure: list) -> None:
    h2_plots.plot_error([_row(0.3, 1e-9), _row(0.4, 3e-9)], Path("x.png"))
    assert len(captured_figure[0].axes[0].texts) == 0


def test_when_every_error_is_zero_the_plot_is_still_valid(captured_figure: list) -> None:
    h2_plots.plot_error([_row(0.3, 0.0), _row(0.4, 0.0)], Path("x.png"))
    ydata = list(_error_line(captured_figure[0]).get_ydata())
    assert all(y > 0 for y in ydata)
    low, high = captured_figure[0].axes[0].get_ylim()
    assert 0 < low < min(ydata) < high


def test_the_points_are_sorted_by_bond_length(captured_figure: list) -> None:
    h2_plots.plot_error([_row(0.5, 5e-10), _row(0.3, 8e-12), _row(0.4, 1e-11)], Path("x.png"))
    line = _error_line(captured_figure[0])
    assert list(line.get_xdata()) == [0.3, 0.4, 0.5]
    assert list(line.get_ydata()) == [8e-12, 1e-11, 5e-10]


def test_plot_error_writes_a_png(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "h2_error.png"
    h2_plots.plot_error([_row(0.3, 8e-12), _row(0.4, 1e-11)], path)
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
