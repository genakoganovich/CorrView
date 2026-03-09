from __future__ import annotations

from collections.abc import Sized
from pathlib import Path
from typing import Mapping, Protocol, cast

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backend_bases import Event, PickEvent
from matplotlib.collections import PathCollection
from matplotlib.ticker import MaxNLocator

from CorrView.corr_parser import build_series_by_point, load_corr_file

Point = tuple[float, float]


class _IndexCollection(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> object: ...


def _to_int(value: object) -> int:
    """Convert matplotlib pick index payload to int in a type-safe way."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return int(str(value))


def get_unique_points(series_by_point: Mapping[Point, pd.DataFrame]) -> list[Point]:
    """Return stable sorted list of points for plotting."""
    return sorted(series_by_point.keys(), key=lambda point: (point[0], point[1]))


class LocationMapWindow:
    """Minimal interactive location-map window for .corr data."""

    def __init__(self, corr_path: str | Path) -> None:
        df = load_corr_file(corr_path)
        self.series_by_point = build_series_by_point(df)
        self.points = get_unique_points(self.series_by_point)
        self.selected_point: Point | None = None

        self.figure = plt.figure(figsize=(18, 8))
        grid = self.figure.add_gridspec(
            nrows=1,
            ncols=4,
            width_ratios=[1.7, 2.3, 1.0, 1.0],
            hspace=0.0,
            wspace=0.38,
        )
        self.ax = self.figure.add_subplot(grid[0, 0])
        self.text_ax = self.figure.add_subplot(grid[0, 1])
        self.v_ax = self.figure.add_subplot(grid[0, 2])
        self.a_ax = self.figure.add_subplot(grid[0, 3])
        self.figure.subplots_adjust(left=0.03, right=0.995)
        self._base_scatter: PathCollection | None = None
        self._selected_scatter: PathCollection | None = None
        self._rows_text = self.text_ax.text(
            0.01,
            0.99,
            "Rows: select a point on the map",
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
            transform=self.text_ax.transAxes,
        )
        self._info_text = self.ax.text(
            0.01,
            -0.12,
            "Selected: none",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            clip_on=False,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.2"},
        )

        self._render()
        self.figure.canvas.mpl_connect("pick_event", self._on_pick)

    @staticmethod
    def _format_rows_text(point_df: pd.DataFrame, limit: int = 22) -> str:
        sorted_df = point_df.sort_values("T")
        header = f"{'T':>7} | {'V':>8} | {'A':>7} | {'DeltaV':>8} | {'DeltaA':>7}"
        lines = [header]

        for _, row in sorted_df.head(limit).iterrows():
            line = (
                f"{float(row['T']):>7.3f} | "
                f"{float(row['V']):>8.2f} | "
                f"{float(row['A']):>7.4f} | "
                f"{float(row['DeltaV']):>8.2f} | "
                f"{float(row['DeltaA']):>7.4f}"
            )
            lines.append(line)

        if len(sorted_df) > limit:
            lines.append(f"... ({len(sorted_df) - limit} more rows)")

        return "\n".join(lines)

    @staticmethod
    def _build_plot_series(point_df: pd.DataFrame) -> pd.DataFrame:
        """Return time-sorted subset with columns needed for V/A plots."""
        return point_df.loc[
            :,
            ["T", "V", "V_minus_DeltaV", "V_plus_DeltaV", "A", "A_minus_DeltaA", "A_plus_DeltaA"],
        ].sort_values("T")

    def _render_plot_placeholders(self) -> None:
        self.v_ax.clear()
        self.v_ax.set_title("V Panel")
        self.v_ax.set_xlabel("V")
        self.v_ax.set_ylabel("T")
        self.v_ax.invert_yaxis()
        self.v_ax.grid(alpha=0.25)

        self.a_ax.clear()
        self.a_ax.set_title("A Panel")
        self.a_ax.set_xlabel("A")
        self.a_ax.set_ylabel("T")
        self.a_ax.invert_yaxis()
        self.a_ax.grid(alpha=0.25)

    def _render(self) -> None:
        x_values = [point[0] for point in self.points]
        y_values = [point[1] for point in self.points]

        self.ax.clear()
        self._base_scatter = self.ax.scatter(x_values, y_values, s=24, c="#1f77b4", picker=6)
        self._selected_scatter = self.ax.scatter([], [], s=120, facecolors="none", edgecolors="#d62728", linewidths=2)
        self._info_text = self.ax.text(
            0.01,
            -0.12,
            "Selected: none",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            clip_on=False,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.2"},
        )
        self.ax.set_title("Location Map")
        self.ax.set_xlabel("XCoord")
        self.ax.set_ylabel("YCoord")
        self.ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        self.ax.grid(alpha=0.25)

        self.text_ax.clear()
        self.text_ax.axis("off")
        self.text_ax.set_title("Rows by T")
        self._rows_text = self.text_ax.text(
            0.01,
            0.99,
            "Rows: select a point on the map",
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
            transform=self.text_ax.transAxes,
        )
        self._render_plot_placeholders()

    @staticmethod
    def _pick_index_from_event_indices(picked_indices: object) -> int | None:
        """Extract first picked point index from matplotlib event.ind payload."""
        if picked_indices is None:
            return None
        if not isinstance(picked_indices, Sized) or not hasattr(picked_indices, "__getitem__"):
            return None

        indices = cast(_IndexCollection, picked_indices)
        if len(indices) == 0:
            return None
        return _to_int(indices[0])

    def _on_pick(self, event: Event) -> None:
        if not isinstance(event, PickEvent):
            return
        picked_indices = getattr(event, "ind", None)
        index = self._pick_index_from_event_indices(picked_indices)
        if index is None:
            return
        self.select_point(self.points[index])

    def select_point(self, point: Point, redraw: bool = True) -> None:
        self.selected_point = point
        x_coord, y_coord = point
        rows_count = len(self.series_by_point[point])
        point_df = self.series_by_point[point]
        plot_df = self._build_plot_series(point_df)

        if self._selected_scatter is not None:
            self._selected_scatter.set_offsets([[x_coord, y_coord]])

        self._info_text.set_text(f"Selected: ({x_coord:.2f}, {y_coord:.2f}) | rows: {rows_count}")
        self._rows_text.set_text(self._format_rows_text(point_df))
        self.v_ax.clear()
        self.v_ax.plot(plot_df["V"], plot_df["T"])
        self.v_ax.plot(plot_df["V_minus_DeltaV"], plot_df["T"])
        self.v_ax.plot(plot_df["V_plus_DeltaV"], plot_df["T"])
        self.v_ax.set_title("V Panel")
        self.v_ax.set_xlabel("V")
        self.v_ax.set_ylabel("T")
        self.v_ax.invert_yaxis()
        self.v_ax.grid(alpha=0.25)
        self.a_ax.clear()
        self.a_ax.plot(plot_df["A"], plot_df["T"])
        self.a_ax.plot(plot_df["A_minus_DeltaA"], plot_df["T"])
        self.a_ax.plot(plot_df["A_plus_DeltaA"], plot_df["T"])
        self.a_ax.set_title("A Panel")
        self.a_ax.set_xlabel("A")
        self.a_ax.set_ylabel("T")
        self.a_ax.invert_yaxis()
        self.a_ax.grid(alpha=0.25)
        if redraw:
            self.figure.canvas.draw_idle()

    def run(self) -> None:
        plt.show()


def main(corr_path: str | Path = "CorrView/Beharra3D__corr__0007.corr") -> None:
    app = LocationMapWindow(corr_path)
    app.run()


if __name__ == "__main__":
    main()
