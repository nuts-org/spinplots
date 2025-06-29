from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from spinplots.io import read_nmr
from spinplots.plot import bruker1d, bruker1d_grid, bruker2d, df2d, dmfit1d, dmfit2d
from spinplots.utils import nmr_df

DATA_DIR_1D_1 = "data/1D/glycine/pdata/1"
DATA_DIR_1D_2 = "data/1D/alanine/pdata/1"
DATA_DIR_2D = "data/2D/16/pdata/1"
DATA_DIR_DM = "data/DMFit/overlapping_spe_fit.ppm"
DATA_DIR_DM_2D = "data/DMFit/hetcor_spectrum.ppm"
DATA_DIR_DM_2D_fit = "data/DMFit/hetcor_model.ppm"


@pytest.fixture(autouse=True)
def configure_matplotlib_and_close_plots():
    mpl.use("Agg")
    yield
    plt.close("all")


def test_bruker1d():
    spin = read_nmr(DATA_DIR_1D_1, "bruker")
    out = bruker1d([spin.spectrum], return_fig=True)
    assert out is not None


def test_bruker1d_grid():
    spin1 = read_nmr(DATA_DIR_1D_1, "bruker")
    spin2 = read_nmr(DATA_DIR_1D_2, "bruker")
    out = bruker1d_grid(
        [spin1.spectrum, spin2.spectrum], subplot_dims=(1, 2), return_fig=True
    )
    assert out is not None


def test_bruker2d():
    spin = read_nmr(DATA_DIR_2D, "bruker")
    out = bruker2d(
        [spin.spectrum],
        contour_start=1e5,
        contour_num=5,
        contour_factor=1.5,
        return_fig=True,
    )
    assert out is not None


def test_bruker2d_no_contour_start():
    """Test bruker2d with contour_start=None to cover default calculation."""
    spin = read_nmr(DATA_DIR_2D, "bruker")
    out = bruker2d(
        [spin.spectrum],
        contour_start=None,
        return_fig=True,
    )
    assert out is not None


def test_df2d():
    df_2d = nmr_df(DATA_DIR_2D)
    out = df2d(
        df_2d, contour_start=1e5, contour_num=5, contour_factor=1.5, return_fig=True
    )
    assert out is not None


def test_dmfit1d():
    spin = read_nmr(DATA_DIR_DM, provider="dmfit")
    fig = dmfit1d(spin, return_fig=True)
    assert fig is not None


def test_dmfit2d():
    spin = read_nmr(DATA_DIR_DM_2D, provider="dmfit")
    fig = dmfit2d(spin, return_fig=True)
    assert fig is not None


def test_dmfit2d_col():
    spin = read_nmr([DATA_DIR_DM_2D, DATA_DIR_DM_2D_fit], provider="dmfit")
    fig = dmfit2d(
        spin,
        contour_start=10,
        contour_num=5,
        contour_factor=1.5,
        colors=["black", "red"],
        return_fig=True,
    )
    assert fig is not None


def test_bruker1d_typeerror():
    with pytest.raises(TypeError):
        bruker1d("notalist")


def test_bruker1d_valerror():
    spin = read_nmr(DATA_DIR_2D, "bruker")
    with pytest.raises(ValueError, match="All spectra must be 1-dimensional"):
        bruker1d([spin.spectrum])
