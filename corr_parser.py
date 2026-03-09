from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("XCoord", "YCoord", "V", "A", "T", "DeltaV", "DeltaA")


def _as_float(value: object) -> float:
    """Convert scalar/group key values to float in a Pylance-friendly way."""
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def load_corr_file(path: str | Path) -> pd.DataFrame:
    """Read and validate a .corr file into a normalized DataFrame."""
    file_path = Path(path)
    df = pd.read_csv(file_path, sep=r"\s+", engine="python")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {joined}")

    df = df.loc[:, REQUIRED_COLUMNS].copy()

    for column in REQUIRED_COLUMNS:
        try:
            df[column] = pd.to_numeric(df[column], errors="raise")
        except ValueError as exc:
            raise ValueError(f"Column '{column}' contains non-numeric values") from exc

    if df.empty:
        raise ValueError("Input .corr file is empty")

    return df


def build_series_by_point(df: pd.DataFrame) -> dict[tuple[float, float], pd.DataFrame]:
    """
    Group rows by (XCoord, YCoord), sorted by time.

    Adds derived columns:
    - V_minus_DeltaV, V_plus_DeltaV
    - A_minus_DeltaA, A_plus_DeltaA
    """
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise ValueError(f"DataFrame is missing required columns: {joined}")

    result: dict[tuple[float, float], pd.DataFrame] = {}
    grouped = df.groupby(["XCoord", "YCoord"], sort=False, as_index=False)

    for (x_coord, y_coord), group_df in grouped:
        sorted_group = group_df.sort_values("T").copy()
        sorted_group["V_minus_DeltaV"] = sorted_group["V"] - sorted_group["DeltaV"]
        sorted_group["V_plus_DeltaV"] = sorted_group["V"] + sorted_group["DeltaV"]
        sorted_group["A_minus_DeltaA"] = sorted_group["A"] - sorted_group["DeltaA"]
        sorted_group["A_plus_DeltaA"] = sorted_group["A"] + sorted_group["DeltaA"]
        result[(_as_float(x_coord), _as_float(y_coord))] = sorted_group.reset_index(drop=True)

    return result
