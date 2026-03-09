from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backend_bases import PickEvent
from matplotlib.collections import PathCollection

from CorrView.corr_parser import build_series_by_point, load_corr_file

Point = tuple[float, float]


def get_unique_points(series_by_point: dict[Point, object]) -> list[Point]:
    """Return stable sorted list of points for plotting."""
    return sorted(series_by_point.keys(), key=lambda point: (point[0], point[1]))


class LocationMapWindow:
    """Minimal interactive location-map window for .corr data."""

    def __init__(self, corr_path: str | Path) -> None:
        df = load_corr_file(corr_path)
        self.series_by_point = build_series_by_point(df)
        self.points = get_unique_points(self.series_by_point)
        self.selected_point: Point | None = None

        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self._base_scatter: PathCollection | None = None
        self._selected_scatter: PathCollection | None = None
        self._info_text = self.ax.text(0.02, 0.98, "Selected: none", transform=self.ax.transAxes, va="top")

        self._render()
        self.figure.canvas.mpl_connect("pick_event", self._on_pick)

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

    def _on_pick(self, event: PickEvent) -> None:
        if not event.ind:
            return
        index = int(event.ind[0])
        self.select_point(self.points[index])

    def select_point(self, point: Point, redraw: bool = True) -> None:
        self.selected_point = point
        x_coord, y_coord = point
        rows_count = len(self.series_by_point[point])

        if self._selected_scatter is not None:
            self._selected_scatter.set_offsets([[x_coord, y_coord]])

        self._info_text.set_text(f"Selected: ({x_coord:.2f}, {y_coord:.2f}) | rows: {rows_count}")
        if redraw:
            self.figure.canvas.draw_idle()

    def run(self) -> None:
        plt.show()


def main(corr_path: str | Path = "CorrView/Beharra3D__corr__0007.corr") -> None:
    app = LocationMapWindow(corr_path)
    app.run()


if __name__ == "__main__":
    main()
