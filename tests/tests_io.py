from __future__ import annotations

import pytest

from spinplots.io import read_nmr

DATA_DIR_1D_1 = "data/1D/glycine/pdata/1"
DATA_DIR_1D_2 = "data/1D/alanine/pdata/1"
DATA_DIR_2D = "data/2D/16/pdata/1"
DATA_DIR_DM = "data/DMFit/example.txt"


def test_read_nmr_bruker_1d():
    spin = read_nmr(DATA_DIR_1D_1, provider="bruker")
    assert hasattr(spin, "ndim")
    assert spin.ndim == 1


def test_read_nmr_bruker_2d():
    spin = read_nmr(DATA_DIR_2D, provider="bruker")
    assert hasattr(spin, "ndim")
    assert spin.ndim == 2


def test_read_nmr_bruker_list():
    spins = read_nmr([DATA_DIR_1D_1, DATA_DIR_1D_2], provider="bruker")
    # This should be a SpinCollection
    assert hasattr(spins, "spins")
    assert spins.size == 2


def test_read_nmr_dmfit():
    spin = read_nmr(DATA_DIR_DM, provider="dmfit")
    assert hasattr(spin, "provider")
    assert spin.provider == "dmfit"


def test_read_nmr_invalid_provider():
    with pytest.raises(ValueError, match="Invalid provider"):
        read_nmr(DATA_DIR_1D_1, provider="foo")


def test_read_nmr_tags_length():
    with pytest.raises(
        ValueError, match="Number of tags must match the number of paths"
    ):
        read_nmr([DATA_DIR_1D_1, DATA_DIR_1D_2], provider="bruker", tags=["a"])
