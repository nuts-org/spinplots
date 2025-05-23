"""
SpinPlots: A Python package for plotting NMR data.

This package provides tools for loading NMR data from various providers
and visualizing it with customizable plots.
"""

from __future__ import annotations

from spinplots.plot import bruker1d, bruker1d_grid, bruker2d, df2d, dmfit1d
from spinplots.spin import Spin, SpinCollection
from spinplots.utils import calculate_projections

__version__ = "0.2.0"

__all__ = [
    "Spin",
    "SpinCollection",
    "bruker1d",
    "bruker1d_grid",
    "bruker2d",
    "calculate_projections",
    "df2d",
    "dmfit1d",
    "dmfit2d",
]
