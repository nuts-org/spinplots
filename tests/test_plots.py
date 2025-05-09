from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from spinplots.plot import (
    bruker1d,
    bruker1d_grid,
    bruker2d,
    df2d,
)
from spinplots.utils import nmr_df

DATA_DIR_1D_1 = "data/1D/glycine/pdata/1"
DATA_DIR_1D_2 = "data/1D/alanine/pdata/1"
DATA_DIR_2D = "data/2D/16/pdata/1"


@pytest.fixture(autouse=True)
def configure_matplotlib_and_close_plots():
    """Switch to non-interactive backend and close plots after each test."""
    # Added this and close to avoid stopping
    # for showing plots during test
    mpl.use("Agg")
    yield
    plt.close("all")


def test_bruker1d_single():
    """Test basic 1D plot."""
    fig, ax = bruker1d(DATA_DIR_1D_1, return_fig=True)
    assert fig is not None
    assert ax is not None


def test_bruker1d_multiple():
    """Test plotting multiple 1D spectra."""
    fig, ax = bruker1d(
        [DATA_DIR_1D_1, DATA_DIR_1D_2], labels=["s1", "s2"], return_fig=True
    )
    assert fig is not None
    assert ax is not None


def test_bruker1d_stacked():
    """Test stacked 1D plot."""
    fig, ax = bruker1d(
        [DATA_DIR_1D_1, DATA_DIR_1D_2], stacked=True, return_fig=True
    )
    assert fig is not None
    assert ax is not None


def test_bruker1d_normalized():
    """Test normalized 1D plot."""
    fig, ax = bruker1d(DATA_DIR_1D_1, normalized=True, return_fig=True)
    assert fig is not None
    assert ax is not None


def test_bruker1d_save(tmp_path):
    """Test saving 1D plot."""
    output_file = tmp_path / "test_1d.png"
    bruker1d(
        DATA_DIR_1D_1,
        save=True,
        filename=str(output_file.with_suffix("")),
        format="png",
    )
    assert output_file.is_file()
    assert output_file.stat().st_size > 0  # Check file is not empty


def test_bruker2d_single():
    """Test basic 2D plot."""
    ax_dict = bruker2d(
        DATA_DIR_2D,
        contour_start=1e5,
        contour_num=10,
        contour_factor=1.5,
        return_fig=True,
    )
    assert isinstance(ax_dict, dict)
    assert "A" in ax_dict


def test_bruker2d_save(tmp_path):
    """Test saving 2D plot."""
    output_file = tmp_path / "test_2d.png"
    bruker2d(
        DATA_DIR_2D,
        contour_start=1e5,
        contour_num=10,
        contour_factor=1.5,
        save=True,
        filename=str(output_file.with_suffix("")),
        format="png",
    )
    assert output_file.is_file()
    assert output_file.stat().st_size > 0


def test_bruker1d_grid():
    """Test grid plot."""
    fig, axes = bruker1d_grid([DATA_DIR_1D_1, DATA_DIR_1D_2], subplot_dims=(1, 2), return_fig=True)
    assert fig is not None
    assert axes is not None
    assert len(axes) == 2


def test_df2d(tmp_path):
    """Test plotting 2D from DataFrame/CSV."""
    df_2d_data = nmr_df(DATA_DIR_2D)
    csv_path = tmp_path / "temp_2d_data.csv"
    df_2d_data.to_csv(csv_path, index=False)

    ax_dict_csv = df2d(
        str(csv_path),
        contour_start=1e5,
        contour_num=10,
        contour_factor=1.5,
        return_fig=True,
    )
    assert isinstance(ax_dict_csv, dict)
    assert "A" in ax_dict_csv

    ax_dict_df = df2d(
        df_2d_data,
        contour_start=1e5,
        contour_num=10,
        contour_factor=1.5,
        return_fig=True,
    )
    assert isinstance(ax_dict_df, dict)
    assert "A" in ax_dict_df
