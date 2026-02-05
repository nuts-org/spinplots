"""
SpinPlots: A Python package for plotting NMR data.

This package provides tools for loading NMR data from various providers
and visualizing it with customizable plots.
"""

from __future__ import annotations

from spinplots.plot import (
    bruker1d,
    bruker1d_grid,
    bruker2d,
    bruker2d_grid,
    df2d,
    dmfit1d,
    dmfit1d_grid,
    dmfit2d,
    dmfit2d_grid,
)
from spinplots.spin import Spin, SpinCollection
from spinplots.utils import calculate_projections

__version__ = "0.2.2"

__all__ = [
    "Spin",
    "SpinCollection",
    "bruker1d",
    "bruker1d_grid",
    "bruker2d",
    "bruker2d_grid",
    "calculate_projections",
    "df2d",
    "dmfit1d",
    "dmfit1d_grid",
    "dmfit2d",
    "dmfit2d_grid",
]
