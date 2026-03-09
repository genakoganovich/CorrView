# CorrView

## Minimal Location Map

Run:

```bash
python3 -m CorrView.gui_location_map
```

This opens an interactive map of `(XCoord, YCoord)` points. Click a point to select it.
The right text panel shows rows for the selected point, sorted by `T`.
The window uses one horizontal row of 4 panels: map, rows, V, A.
In `V` and `A` panels, `T` is vertical (directed downward), while `V/A` are horizontal (to the right).
