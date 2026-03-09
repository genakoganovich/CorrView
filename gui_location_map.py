from __future__ import annotations

from pathlib import Path
from typing import Mapping
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backend_bases import Event, PickEvent
from matplotlib.collections import PathCollection

from CorrView.corr_parser import build_series_by_point, load_corr_file

Point = tuple[float, float]


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

        self.figure, (self.ax, self.text_ax) = plt.subplots(
            ncols=2,
            figsize=(12, 6),
            gridspec_kw={"width_ratios": [2, 1]},
        )
        self._base_scatter: PathCollection | None = None
        self._selected_scatter: PathCollection | None = None
        self._rows_text = self.text_ax.text(
            0.01,
            0.99,
            "Rows: select a point on the map",
            va="top",
            ha="left",
            family="monospace",
            transform=self.text_ax.transAxes,
        )
        self._info_text = self.ax.text(0.02, 0.98, "Selected: none", transform=self.ax.transAxes, va="top")

        self._render()
        self.figure.canvas.mpl_connect("pick_event", self._on_pick)

    @staticmethod
    def _format_rows_text(point_df: pd.DataFrame, limit: int = 22) -> str:
        sorted_df = point_df.sort_values("T")
        header = "T | V | A | DeltaV | DeltaA"
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

    def _render(self) -> None:
        x_values = [point[0] for point in self.points]
        y_values = [point[1] for point in self.points]

        self.ax.clear()
        self._base_scatter = self.ax.scatter(x_values, y_values, s=24, c="#1f77b4", picker=6)
        self._selected_scatter = self.ax.scatter([], [], s=120, facecolors="none", edgecolors="#d62728", linewidths=2)
        self._info_text = self.ax.text(0.02, 0.98, "Selected: none", transform=self.ax.transAxes, va="top")
        self.ax.set_title("Location Map")
        self.ax.set_xlabel("XCoord")
        self.ax.set_ylabel("YCoord")
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
            transform=self.text_ax.transAxes,
        )

    def _on_pick(self, event: Event) -> None:
        if not isinstance(event, PickEvent):
            return
        picked_indices = getattr(event, "ind", None)
        if not picked_indices:
            return
        index = int(picked_indices[0])
        self.select_point(self.points[index])

    def select_point(self, point: Point, redraw: bool = True) -> None:
        self.selected_point = point
        x_coord, y_coord = point
        rows_count = len(self.series_by_point[point])
        point_df = self.series_by_point[point]

        if self._selected_scatter is not None:
            self._selected_scatter.set_offsets([[x_coord, y_coord]])

        self._info_text.set_text(f"Selected: ({x_coord:.2f}, {y_coord:.2f}) | rows: {rows_count}")
        self._rows_text.set_text(self._format_rows_text(point_df))
        if redraw:
            self.figure.canvas.draw_idle()

    def run(self) -> None:
        plt.show()


def main(corr_path: str | Path = "CorrView/Beharra3D__corr__0007.corr") -> None:
    app = LocationMapWindow(corr_path)
    app.run()


if __name__ == "__main__":
    main()
