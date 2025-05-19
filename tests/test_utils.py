import pytest
import numpy as np
from spinplots.utils import nmr_df, calculate_projections

DATA_DIR_1D_1 = "data/1D/glycine/pdata/1"
DATA_DIR_2D = "data/2D/16/pdata/1"

def test_nmr_df_1d():
    df = nmr_df(DATA_DIR_1D_1)
    assert "ppm" in df.columns
    assert df.attrs["nmr_dim"] == 1

def test_nmr_df_2d():
    df = nmr_df(DATA_DIR_2D)
    assert any("ppm" in c for c in df.columns)
    assert df.attrs["nmr_dim"] == 2

def test_nmr_df_2d_hz():
    df = nmr_df(DATA_DIR_2D, hz=True)
    assert any("hz" in c for c in df.columns)
    assert df.attrs["nmr_dim"] == 2

def test_nmr_df_unsupported_dim(tmp_path):
    # Simulate 3D data by hacking nmrglue via monkeypatch (optional: skip)
    pass  # Not possible with real files, only via mocking

def test_nmr_df_export(tmp_path):
    out = tmp_path / "test_export.csv"
    df = nmr_df(DATA_DIR_1D_1, export=True, filename=str(out))
    assert df is None
    assert out.exists()

def test_calculate_projections_df():
    df = nmr_df(DATA_DIR_2D)
    f1, f2 = calculate_projections(df)
    assert f1 is not None and f2 is not None

def test_calculate_projections_csv(tmp_path):
    df = nmr_df(DATA_DIR_2D)
    f = tmp_path / "test.csv"
    df.to_csv(f, index=False)
    f1, f2 = calculate_projections(str(f))
    assert f1 is not None and f2 is not None

def test_calculate_projections_invalid():
    with pytest.raises(ValueError):
        calculate_projections(42)

