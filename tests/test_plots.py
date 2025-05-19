import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from spinplots.io import read_nmr
from spinplots.plot import bruker1d, bruker2d, bruker1d_grid, df2d, dmfit1d
from spinplots.utils import nmr_df

DATA_DIR_1D_1 = "data/1D/glycine/pdata/1"
DATA_DIR_1D_2 = "data/1D/alanine/pdata/1"
DATA_DIR_2D = "data/2D/16/pdata/1"
DATA_DIR_DM = "data/DMFit/overlapping_spe_fit.ppm"

@pytest.fixture(autouse=True)
def configure_matplotlib_and_close_plots():
    mpl.use("Agg")
    yield
    plt.close("all")

def test_bruker1d():
    spin = read_nmr(DATA_DIR_1D_1, 'bruker')
    out = bruker1d([spin.spectrum], return_fig=True)
    assert out is not None

def test_bruker1d_grid():
    spin1 = read_nmr(DATA_DIR_1D_1, 'bruker')
    spin2 = read_nmr(DATA_DIR_1D_2, 'bruker')
    out = bruker1d_grid([spin1.spectrum, spin2.spectrum], subplot_dims=(1, 2), return_fig=True)
    assert out is not None

def test_bruker2d():
    spin = read_nmr(DATA_DIR_2D, 'bruker')
    out = bruker2d(spin.spectrum, contour_start=1e5, contour_num=5, contour_factor=1.5, return_fig=True)
    assert out is not None

def test_df2d():
    df = nmr_df(DATA_DIR_2D)
    out = df2d(df, contour_start=1e5, contour_num=5, contour_factor=1.5, return_fig=True)
    assert out is not None

def test_dmfit1d():
    spin = read_nmr(DATA_DIR_DM, provider="dmfit")
    fig = dmfit1d(spin, return_fig=True)
    assert fig is not None

def test_bruker1d_typeerror():
    with pytest.raises(TypeError):
        bruker1d("notalist")

def test_bruker1d_valerror():
    spin = read_nmr(DATA_DIR_2D, 'bruker')
    with pytest.raises(ValueError):
        bruker1d([spin.spectrum])

