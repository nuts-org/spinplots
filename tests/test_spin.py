from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from spinplots.io import read_nmr
from spinplots.spin import Spin, SpinCollection

DATA_DIR_1D_1 = "data/1D/glycine/pdata/1"
DATA_DIR_1D_2 = "data/1D/alanine/pdata/1"
DATA_DIR_2D = "data/2D/16/pdata/1"
DATA_DIR_DM = "data/DMFit/overlapping_spe_fit.ppm"
DATA_DIR_DM_2D = "data/DMFit/hetcor_spectrum.ppm"
DATA_DIR_DM_2D_fit = "data/DMFit/hetcor_model.ppm"


@pytest.fixture(autouse=True)
def configure_matplotlib_and_close_plots():
    """Switch to non-interactive backend and close plots after each test."""
    mpl.use("Agg")
    yield
    plt.close("all")


@pytest.fixture
def spin_1d():
    return read_nmr(DATA_DIR_1D_1, "bruker")


@pytest.fixture
def spin_1d_2():
    return read_nmr(DATA_DIR_1D_2, "bruker")


@pytest.fixture
def spin_2d():
    return read_nmr(DATA_DIR_2D, "bruker")


@pytest.fixture
def spin_dmfit():
    return read_nmr(DATA_DIR_DM, "dmfit")


@pytest.fixture
def spincollection():
    return read_nmr([DATA_DIR_1D_1, DATA_DIR_1D_2], "bruker")


def test_spin_attributes(spin_1d):
    assert isinstance(spin_1d, SpinCollection)
    assert hasattr(spin_1d, "ndim")
    assert spin_1d.ndim == 1
    assert hasattr(spin_1d, "provider")
    assert spin_1d.provider == "bruker"


def test_spin_repr(spin_1d):
    r = repr(spin_1d)
    assert "SpinCollection" in r


def test_spin_init_value_error():
    with pytest.raises(
        ValueError, match="Cannot initialize Spin object with empty spectrum data"
    ):
        Spin({}, "bruker")
    with pytest.raises(ValueError, match="Unsupported number of dimensions in data"):
        Spin({"ndim": 3, "path": "foo"}, "bruker")
    with pytest.raises(ValueError, match="Unsupported provider"):
        Spin({"ndim": 1, "path": "foo"}, "foo")


def test_spin_plot_1d(spin_1d):
    fig, ax = spin_1d.plot(return_fig=True)
    assert fig is not None
    assert ax is not None


def test_spin_plot_2d(spin_2d):
    """Test 2D plotting with missing projections."""
    ax_dict = spin_2d.plot(
        contour_start=1e5, contour_num=10, contour_factor=1.5, return_fig=True
    )
    assert isinstance(ax_dict, dict)
    assert "A" in ax_dict


def test_spin_plot_grid(spin_1d):
    fig, ax = spin_1d.plot(grid="1x1", return_fig=True)
    assert fig is not None
    assert ax is not None


def test_spin_plot_bad_grid_str(spin_1d):
    with pytest.raises(ValueError, match="Grid format should be 'rows x cols'"):
        spin_1d.plot(grid="badgrid")


def test_spin_plot_2d_grid(spin_2d):
    """Single 2D Bruker SpinCollection can use grid layout."""
    fig, axes = spin_2d.plot(
        grid="1x1",
        contour_start=1e5,
        contour_num=5,
        contour_factor=1.5,
        return_fig=True,
    )
    assert fig is not None
    assert len(axes) == 1
    assert "main" in axes[0]


def test_spin_plot_dmfit(spin_dmfit):
    fig, ax = spin_dmfit.plot(return_fig=True)
    assert fig is not None
    assert ax is not None


def test_spin_plot_dmfit_grid(spin_dmfit):
    """Single 1D DMFit SpinCollection can use grid layout."""
    fig, axes = spin_dmfit.plot(grid="1x1", return_fig=True)
    assert fig is not None
    assert len(axes) == 1


def test_spincollection_construction(spin_1d, spin_1d_2):
    coll = SpinCollection([spin_1d[0]])
    assert len(coll) == 1
    coll.append(spin_1d_2[0])
    assert len(coll) == 2


def test_spincollection_append_invalid_ndim(spin_1d, spin_2d):
    coll = SpinCollection([spin_1d[0]])
    with pytest.raises(
        ValueError, match=r"All Spin objects must have the same dimension."
    ):
        coll.append(spin_2d[0])


def test_spincollection_append_invalid_provider(spin_1d, spin_dmfit):
    coll = SpinCollection([spin_1d[0]])
    with pytest.raises(
        ValueError, match=r"All Spin objects must have the same provider."
    ):
        coll.append(spin_dmfit[0])


def test_spincollection_duplicate_tag(spin_1d):
    inner = spin_1d[0]
    inner.tag = "duplicate_tag"
    coll = SpinCollection([inner])
    with pytest.raises(
        ValueError, match="Spin with tag 'duplicate_tag' already exists"
    ):
        coll.append(inner)


def test_spincollection_remove_and_delitem(spin_1d, spin_1d_2):
    coll = SpinCollection([spin_1d[0], spin_1d_2[0]])
    # get tags:
    tags = list(coll.spins.keys())
    coll.remove(tags[0])
    assert len(coll) == 1
    coll.__delitem__(tags[1])
    assert len(coll) == 0


def test_spincollection_remove_invalid(spin_1d, spin_1d_2):
    coll = SpinCollection([spin_1d, spin_1d_2])
    with pytest.raises(KeyError):
        coll.remove("invalid_tag")


def test_spincollection_plot_1d(spincollection):
    """Test plotting a 1D SpinCollection with default parameters."""
    fig, ax = spincollection.plot(return_fig=True)
    assert fig is not None
    assert ax is not None
    assert len(ax.get_lines()) == len(spincollection)


def test_spincollection_plot_grid(spincollection):
    """Test plotting a 1D SpinCollection with grid layout."""
    fig, axes = spincollection.plot(grid="1x2", return_fig=True)
    assert fig is not None
    # Should have 2 subplots
    assert len(axes) == 2
    assert len(axes[0].get_lines()) > 0
    assert len(axes[1].get_lines()) > 0


def test_spincollection_plot_override_labels(spin_1d, spin_1d_2):
    """Test that custom labels override tags."""
    inner_1d = spin_1d[0]
    inner_1d_2 = spin_1d_2[0]
    inner_1d.tag = "Sample A"
    inner_1d_2.tag = "Sample B"
    coll = SpinCollection([inner_1d, inner_1d_2])

    # Plot with custom labels
    custom_labels = ["Custom 1", "Custom 2"]
    _fig, ax = coll.plot(labels=custom_labels, return_fig=True)
    legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "Custom 1" in legend_texts
    assert "Custom 2" in legend_texts
    assert "Sample A" not in legend_texts
    assert "Sample B" not in legend_texts


@pytest.fixture
def spin_dmfit_2d():
    return read_nmr("data/DMFit/hetcor_spectrum.ppm", provider="dmfit")


@pytest.fixture
def spincollection_dmfit_2d():
    return read_nmr(
        ["data/DMFit/hetcor_spectrum.ppm", "data/DMFit/hetcor_model.ppm"],
        provider="dmfit",
        tags=["spectrum", "fit"],
    )


def test_spin_plot_dmfit_2d(spin_dmfit_2d):
    assert spin_dmfit_2d.ndim == 2
    assert spin_dmfit_2d.provider == "dmfit"

    ax_dict = spin_dmfit_2d.plot(
        contour_start=10, contour_num=5, contour_factor=1.5, return_fig=True
    )
    assert isinstance(ax_dict, dict)
    assert "A" in ax_dict
    assert "a" in ax_dict
    assert "b" in ax_dict


def test_spincollection_plot_dmfit_2d(spincollection_dmfit_2d):
    ax_dict = spincollection_dmfit_2d.plot(
        contour_start=10,
        contour_num=5,
        contour_factor=1.5,
        color=["black", "red"],
        return_fig=True,
    )
    assert isinstance(ax_dict, dict)
    assert "A" in ax_dict

    # Test that legends work
    legend = ax_dict["A"].get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert "spectrum" in legend_texts
    assert "fit" in legend_texts

    # Test custom labels
    ax_dict = spincollection_dmfit_2d.plot(
        contour_start=10,
        contour_num=5,
        labels=["Experimental", "Calculated"],
        return_fig=True,
    )
    legend = ax_dict["A"].get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert "Experimental" in legend_texts
    assert "Calculated" in legend_texts


def test_spin_plot_dmfit_2d_grid_requires_pairs(spin_dmfit_2d):
    """Test that a single DMFit 2D spectrum in grid mode raises about pairs."""
    with pytest.raises(
        ValueError,
        match=r"expects pairs",
    ):
        spin_dmfit_2d.plot(grid="1x2")


def test_spincollection_plot_dmfit_2d_grid(spincollection_dmfit_2d):
    """Test that SpinCollection.plot(grid=...) works for dmfit 2D data."""
    fig, axes = spincollection_dmfit_2d.plot(
        grid="1x1",
        contour_start=10,
        contour_num=5,
        contour_factor=1.5,
        return_fig=True,
    )
    assert fig is not None
    assert len(axes) == 1
    assert "main" in axes[0]
    assert "top" in axes[0]
    assert "left" in axes[0]


def test_spincollection_plot_bruker_2d_grid():
    """Test plotting a 2D Bruker SpinCollection with grid layout."""
    coll = read_nmr([DATA_DIR_2D, DATA_DIR_2D], "bruker", tags=["spec1", "spec2"])
    fig, axes = coll.plot(
        grid="1x2",
        contour_start=1e5,
        contour_num=5,
        contour_factor=1.5,
        return_fig=True,
    )
    assert fig is not None
    assert len(axes) == 2
    assert "main" in axes[0]
    assert "main" in axes[1]


def test_spincollection_plot_dmfit_1d_stacked():
    """Test plotting multiple 1D DMFit spectra via SpinCollection without grid."""
    coll = read_nmr([DATA_DIR_DM, DATA_DIR_DM], "dmfit", tags=["fit1", "fit2"])
    fig, ax = coll.plot(return_fig=True)
    assert fig is not None
    assert ax is not None


def test_spincollection_plot_dmfit_1d_grid():
    """Test plotting a 1D DMFit SpinCollection with grid layout."""
    coll = read_nmr([DATA_DIR_DM, DATA_DIR_DM], "dmfit", tags=["fit1", "fit2"])
    fig, axes = coll.plot(grid="1x2", return_fig=True)
    assert fig is not None
    assert len(axes) == 2
    assert len(axes[0].get_lines()) > 0
    assert len(axes[1].get_lines()) > 0


def test_bruker1d_grid_per_subplot_ylim(spincollection):
    """Test per-subplot ylim as a list of tuples in bruker1d_grid."""
    fig, axes = spincollection.plot(
        grid="1x2",
        ylim=[(-1e6, 1e7), (-1e6, 1e8)],
        return_fig=True,
    )
    assert fig is not None
    y0 = axes[0].get_ylim()
    y1 = axes[1].get_ylim()
    assert y0 != y1


def test_bruker1d_grid_per_subplot_xlim(spincollection):
    """Test per-subplot xlim as a list of tuples in bruker1d_grid."""
    fig, axes = spincollection.plot(
        grid="1x2",
        xlim=[(200, 0), (100, 0)],
        return_fig=True,
    )
    assert fig is not None
    x0 = axes[0].get_xlim()
    x1 = axes[1].get_xlim()
    assert x0 != x1
