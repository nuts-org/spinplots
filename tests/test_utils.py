import pandas as pd
import pytest
from spinplots.utils import nmr_df, calculate_projections

DATA_DIR_1D = "data/1D/8/pdata/1"
DATA_DIR_2D = "data/2D/16/pdata/1"

def test_nmr_df_1d():
    """Test reading 1D Bruker data."""
    df = nmr_df(DATA_DIR_1D)
    assert isinstance(df, pd.DataFrame)
    assert df.attrs.get('nmr_dim') == 1
    assert 'ppm' in df.columns
    assert 'intensity' in df.columns

def test_nmr_df_2d():
    """Test reading 2D Bruker data."""
    df = nmr_df(DATA_DIR_2D)
    assert isinstance(df, pd.DataFrame)
    assert df.attrs.get('nmr_dim') == 2
    assert len(df.columns) == 3
    assert 'intensity' in df.columns

def test_nmr_df_export(tmp_path):
    """Test exporting DataFrame to CSV."""
    output_file = tmp_path / "test_1d.csv"
    nmr_df(DATA_DIR_1D, export=True, filename=str(output_file))
    assert output_file.is_file()
    df_read = pd.read_csv(output_file)
    assert 'ppm' in df_read.columns

def test_calculate_projections():
    """Test calculating projections from 2D data."""
    df_2d = nmr_df(DATA_DIR_2D)
    proj_f1, proj_f2 = calculate_projections(df_2d)
    assert isinstance(proj_f1, pd.DataFrame)
    assert isinstance(proj_f2, pd.DataFrame)
    assert 'F1 projection' in proj_f1.columns
    assert 'F2 projection' in proj_f2.columns

def test_calculate_projections_export(tmp_path):
    """Test exporting projections."""
    df_2d = nmr_df(DATA_DIR_2D)
    output_base = tmp_path / "projections"
    calculate_projections(df_2d, export=True, filename=str(output_base))
    assert (tmp_path / "projections_f1.csv").is_file()
    assert (tmp_path / "projections_f2.csv").is_file()