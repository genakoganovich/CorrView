from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from CorrView.gui_location_map import LocationMapWindow, get_unique_points


def test_get_unique_points_returns_sorted_pairs() -> None:
    points = get_unique_points({(2.0, 9.0): object(), (1.0, 5.0): object(), (1.0, 4.0): object()})
    assert points == [(1.0, 4.0), (1.0, 5.0), (2.0, 9.0)]


def test_location_map_select_point_updates_state_and_label(tmp_path: Path) -> None:
    corr_file = tmp_path / "mini.corr"
    corr_file.write_text(
        "XCoord\tYCoord\tV\tA\tT\tDeltaV\tDeltaA\n"
        "1\t10\t100\t0.1\t0\t10\t0.01\n"
        "2\t20\t130\t0.3\t1\t13\t0.03\n"
        "2\t20\t120\t0.2\t0\t12\t0.02\n",
        encoding="utf-8",
    )

    app = LocationMapWindow(corr_file)
    app.select_point((2.0, 20.0), redraw=False)

    assert app.selected_point == (2.0, 20.0)
    assert "Selected: (2.00, 20.00) | rows: 2" in app._info_text.get_text()
    rows_text = app._rows_text.get_text().splitlines()
    assert rows_text[0] == "T | V | A | DeltaV | DeltaA"
    assert rows_text[1].startswith("  0.000")
    assert rows_text[2].startswith("  1.000")
    plt.close(app.figure)
