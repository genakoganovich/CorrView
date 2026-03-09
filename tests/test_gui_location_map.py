from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from CorrView.gui_location_map import LocationMapWindow, get_unique_points


def test_get_unique_points_returns_sorted_pairs() -> None:
    points = get_unique_points(
        {
            (2.0, 9.0): pd.DataFrame(),
            (1.0, 5.0): pd.DataFrame(),
            (1.0, 4.0): pd.DataFrame(),
        }
    )
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
    assert len(app.v_ax.lines) == 3
    assert len(app.a_ax.lines) == 3
    assert list(app.v_ax.lines[0].get_ydata()) == [0.0, 1.0]
    assert app.v_ax.yaxis_inverted()
    assert app.a_ax.yaxis_inverted()
    plt.close(app.figure)


def test_build_plot_series_sorts_by_time() -> None:
    df = pd.DataFrame(
        {
            "T": [5.0, 1.0],
            "V": [20.0, 10.0],
            "V_minus_DeltaV": [18.0, 9.0],
            "V_plus_DeltaV": [22.0, 11.0],
            "A": [0.5, 0.2],
            "A_minus_DeltaA": [0.4, 0.1],
            "A_plus_DeltaA": [0.6, 0.3],
        }
    )

    plot_df = LocationMapWindow._build_plot_series(df)
    assert plot_df["T"].tolist() == [1.0, 5.0]


def test_pick_index_from_event_indices_handles_multiple_values() -> None:
    assert LocationMapWindow._pick_index_from_event_indices(np.array([3, 7])) == 3
    assert LocationMapWindow._pick_index_from_event_indices([]) is None
    assert LocationMapWindow._pick_index_from_event_indices(None) is None
