from __future__ import annotations

from spinplots.io import read_nmr
from spinplots.spin import Spin

DATA_DIR_1D_1 = "data/1D/glycine/pdata/1"
DATA_DIR_1D_2 = "data/1D/alanine/pdata/1"
DATA_DIR_2D = "data/2D/16/pdata/1"


def test_two_spins():
    """Test Spin object with spectra list"""
    spins = read_nmr([DATA_DIR_1D_1, DATA_DIR_1D_2], 'bruker')
    assert isinstance(spins, Spin)
    assert spins.num_spectra == 2
    assert spins.ndim == 1

def test_2d_spin():
    """Test Spin object with 2d spectra"""
    spin = read_nmr([DATA_DIR_2D], 'bruker')
    assert isinstance(spin, Spin)
    assert spin.num_spectra == 1
    assert spin.ndim == 2

