from __future__ import annotations

import os
from pathlib import Path

import pytest

from spinplots.utils import calculate_projections, nmr_df

DATA_DIR_1D_1 = "data/1D/glycine/pdata/1"
DATA_DIR_2D = "data/2D/16/pdata/1"


def test_nmr_df_1d():
    df_1d = nmr_df(DATA_DIR_1D_1)
    assert "ppm" in df_1d.columns
    assert df_1d.attrs["nmr_dim"] == 1


def test_nmr_df_2d():
    df_2d = nmr_df(DATA_DIR_2D)
    assert any("ppm" in c for c in df_2d.columns)
    assert df_2d.attrs["nmr_dim"] == 2


def test_nmr_df_2d_hz():
    df_2d = nmr_df(DATA_DIR_2D, hz=True)
    assert any("hz" in c for c in df_2d.columns)
    assert df_2d.attrs["nmr_dim"] == 2


def test_nmr_df_export(tmp_path):
    out = tmp_path / "test_export.csv"
    df_1d = nmr_df(DATA_DIR_1D_1, export=True, filename=str(out))
    assert df_1d is None
    assert out.exists()


def test_calculate_projections_df():
    df_2d = nmr_df(DATA_DIR_2D)
    f1, f2 = calculate_projections(df_2d)
    assert f1 is not None
    assert f2 is not None


def test_calculate_projections_csv(tmp_path):
    df_2d = nmr_df(DATA_DIR_2D)
    f = tmp_path / "test.csv"
    df_2d.to_csv(f, index=False)
    f1, f2 = calculate_projections(str(f))
    assert f1 is not None
    assert f2 is not None


def test_calculate_projections_without_filename(tmp_path):
    """Test calculate_projections with export=True and no filename."""
    df_2d = nmr_df(DATA_DIR_2D)

    original_cwd = Path.cwd()
    os.chdir(tmp_path)

    try:
        f1, f2 = calculate_projections(df_2d, export=True)
        assert f1 is None
        assert f2 is None
        assert (tmp_path / "projections_f1.csv").exists()
        assert (tmp_path / "projections_f2.csv").exists()
    finally:
        os.chdir(original_cwd)


def test_calculate_projections_invalid():
    with pytest.raises(
        ValueError, match="Data must be a DataFrame or a path to a CSV file."
    ):
        calculate_projections(42)
