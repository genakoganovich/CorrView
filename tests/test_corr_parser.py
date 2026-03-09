import math
from pathlib import Path

import pandas as pd
import pytest

from CorrView.corr_parser import build_series_by_point, load_corr_file


def test_load_corr_file_reads_real_dataset() -> None:
    file_path = Path("CorrView/Beharra3D__corr__0007.corr")
    df = load_corr_file(file_path)

    assert not df.empty
    assert list(df.columns) == ["XCoord", "YCoord", "V", "A", "T", "DeltaV", "DeltaA"]
    assert pd.api.types.is_numeric_dtype(df["XCoord"])
    assert pd.api.types.is_numeric_dtype(df["DeltaA"])


def test_build_series_by_point_adds_derived_columns_and_sorts_by_time() -> None:
    df = pd.DataFrame(
        {
            "XCoord": [1.0, 1.0],
            "YCoord": [2.0, 2.0],
            "V": [30.0, 10.0],
            "A": [0.7, 0.5],
            "T": [3.0, 1.0],
            "DeltaV": [5.0, 2.0],
            "DeltaA": [0.2, 0.1],
        }
    )

    grouped = build_series_by_point(df)
    assert len(grouped) == 1
    point_df = grouped[(1.0, 2.0)]

    assert point_df["T"].tolist() == [1.0, 3.0]
    assert point_df["V_minus_DeltaV"].tolist() == [8.0, 25.0]
    assert point_df["V_plus_DeltaV"].tolist() == [12.0, 35.0]
    assert point_df["A_minus_DeltaA"].tolist() == pytest.approx([0.4, 0.5])
    assert point_df["A_plus_DeltaA"].tolist() == pytest.approx([0.6, 0.9])


def test_load_corr_file_raises_on_missing_required_columns(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.corr"
    bad_file.write_text("XCoord\tYCoord\tV\tA\tT\tDeltaV\n1\t2\t3\t4\t5\t6\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required columns: DeltaA"):
        load_corr_file(bad_file)


def test_load_corr_file_raises_on_non_numeric_values(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad_numeric.corr"
    bad_file.write_text(
        "XCoord\tYCoord\tV\tA\tT\tDeltaV\tDeltaA\n1\t2\tabc\t4\t5\t6\t7\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Column 'V' contains non-numeric values"):
        load_corr_file(bad_file)


def test_real_dataset_derived_formula_is_correct() -> None:
    file_path = Path("CorrView/Beharra3D__corr__0007.corr")
    df = load_corr_file(file_path)
    grouped = build_series_by_point(df)

    first_point = next(iter(grouped))
    point_df = grouped[first_point]
    row = point_df.iloc[0]

    assert math.isclose(row["V_minus_DeltaV"], row["V"] - row["DeltaV"], rel_tol=0, abs_tol=1e-9)
    assert math.isclose(row["V_plus_DeltaV"], row["V"] + row["DeltaV"], rel_tol=0, abs_tol=1e-9)
    assert math.isclose(row["A_minus_DeltaA"], row["A"] - row["DeltaA"], rel_tol=0, abs_tol=1e-9)
    assert math.isclose(row["A_plus_DeltaA"], row["A"] + row["DeltaA"], rel_tol=0, abs_tol=1e-9)
