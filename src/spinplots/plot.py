from __future__ import annotations

import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from spinplots._helpers import (
    _apply_tick_spacing,
    _handle_show_return,
    _nucleus_label,
    _parse_nucleus,
    _resolve_per_subplot,
    _save_figure,
)
from spinplots.utils import calculate_projections


def _validate_kwargs(kwargs, defaults, func_name):
    """Warn about unrecognized keyword arguments that will be silently ignored."""
    unknown = set(kwargs) - set(defaults)
    if unknown:
        warnings.warn(
            f"{func_name}() got unexpected keyword argument(s): "
            f"{', '.join(sorted(unknown))}. "
            f"Valid options are: {', '.join(sorted(defaults))}.",
            UserWarning,
            stacklevel=3,
        )


# Default values
DEFAULTS = {
    "labelsize": 12,
    "linewidth": 1.0,
    "linestyle": "-",
    "linewidth_contour": 0.5,
    "linewidth_proj": 0.8,
    "alpha": 1.0,
    "axisfontsize": 13,
    "axisfont": None,
    "tickfontsize": 12,
    "tickfont": None,
    "yaxislabel": "Intensity (a.u.)",
    "xaxislabel": None,
    "tickspacing": None,
    "xtickspacing": None,
    "ytickspacing": None,
}


def bruker2d(
    spectra: dict | list[dict],
    contour_start: float | None = None,
    contour_num: int = 10,
    contour_factor: float = 1.2,
    cmap: str | list[str] | None = None,
    color: list[str] | None = None,
    proj_color: list[str] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    save: bool = False,
    filename: str | None = None,
    format: str | None = None,
    diagonal: float | None = None,
    homonuclear: bool = False,
    return_fig: bool = False,
    **kwargs,
) -> tuple | dict | None:
    """
    Plot one or more 2D Bruker NMR spectra with contour lines and projections.

    Parameters
    ----------
    spectra : dict or list of dict
        Dictionary or list of dictionaries containing spectrum data.
    contour_start : float, optional
        Starting contour level. If None, defaults to 5 % of the data maximum.
    contour_num : int, optional
        Number of contour levels. Default is 10.
    contour_factor : float, optional
        Multiplicative factor between successive contour levels. Default is 1.2.
    cmap : str or list of str, optional
        Colormap(s) for the contour lines. Mutually exclusive with *color*.
    color : list of str, optional
        Solid color(s) for contour lines when overlaying spectra.
    proj_color : list of str, optional
        Color(s) for the projection traces.
    xlim : tuple of float, optional
        Limits for the x-axis (F2 / direct dimension).
    ylim : tuple of float, optional
        Limits for the y-axis (F1 / indirect dimension).
    save : bool, optional
        Whether to save the plot. Default is False.
    filename : str, optional
        Filename for saving (without extension).
    format : str, optional
        File format for saving (e.g. ``'png'``, ``'pdf'``).
    diagonal : float, optional
        Slope of a diagonal reference line (e.g. 2 for DQ-SQ).
    homonuclear : bool, optional
        Set to True when both axes correspond to the same nucleus.
    return_fig : bool, optional
        Whether to return the figure and axes dict. Default is False.
    **kwargs : dict
        Additional customization (``axisfontsize``, ``tickfontsize``,
        ``linewidth_contour``, ``linewidth_proj``, ``alpha``,
        ``xaxislabel``, ``yaxislabel``, ``tickspacing``, etc.).

    Returns
    -------
    dict or None
        Axes dictionary if *return_fig* is True, else None.
    """

    spectra = spectra if isinstance(spectra, list) else [spectra]

    if not all(s["ndim"] == 2 for s in spectra):
        raise ValueError("All spectra must be 2-dimensional for bruker2d.")

    defaults = DEFAULTS.copy()
    defaults["yaxislabel"] = None
    _validate_kwargs(kwargs, defaults, "bruker2d")
    defaults.update(
        {k: v for k, v in kwargs.items() if k in defaults and v is not None}
    )

    fig = plt.figure(constrained_layout=False)
    ax = fig.subplot_mosaic(
        """
    .a
    bA
    """,
        gridspec_kw={
            "height_ratios": [0.9, 6.0],
            "width_ratios": [0.8, 6.0],
            "wspace": 0.03,
            "hspace": 0.04,
        },
    )

    for i, spectrum in enumerate(spectra):
        data = spectrum["data"]

        nuclei_list = spectrum["nuclei"]

        if homonuclear:
            nuclei_x = nuclei_list[1]
            nuclei_y = nuclei_list[1]
        else:
            nuclei_x = nuclei_list[1]
            nuclei_y = nuclei_list[0]

        number_x, nucleus_x = _parse_nucleus(nuclei_x)
        number_y, nucleus_y = _parse_nucleus(nuclei_y)
        ppm_x = spectrum["ppm_scale"][1]
        ppm_x_limits = (ppm_x[0], ppm_x[-1])
        ppm_y = spectrum["ppm_scale"][0]

        if xlim:
            x_min_idx = np.abs(ppm_x - max(xlim)).argmin()
            x_max_idx = np.abs(ppm_x - min(xlim)).argmin()
            x_indices = slice(min(x_min_idx, x_max_idx), max(x_min_idx, x_max_idx))
        else:
            x_indices = slice(None)

        if ylim:
            y_min_idx = np.abs(ppm_y - max(ylim)).argmin()
            y_max_idx = np.abs(ppm_y - min(ylim)).argmin()
            y_indices = slice(min(y_min_idx, y_max_idx), max(y_min_idx, y_max_idx))
        else:
            y_indices = slice(None)

        if (
            isinstance(spectrum["projections"], dict)
            and "f2" in spectrum["projections"]
            and "f1" in spectrum["projections"]
        ):
            if xlim is None and ylim is None:
                proj_x = spectrum["projections"]["f2"]
                proj_y = spectrum["projections"]["f1"]
            else:
                zoomed_data = data[y_indices, x_indices]
                proj_x = np.amax(zoomed_data, axis=0)
                proj_y = np.amax(zoomed_data, axis=1)
        else:
            zoomed_data = data[y_indices, x_indices]
            proj_x = np.amax(zoomed_data, axis=0)
            proj_y = np.amax(zoomed_data, axis=1)

        if contour_start is None:
            contour_start = 0.05 * np.max(data)

        assert contour_start is not None
        contour_levels = contour_start * contour_factor ** np.arange(contour_num)

        x_proj_ppm = ppm_x[x_indices]
        y_proj_ppm = ppm_y[y_indices]

        if cmap is not None and color is not None:
            raise ValueError("Only one of cmap or color can be provided.")

        if cmap is not None:
            if isinstance(cmap, str):
                cmap = [cmap]

            if len(cmap) > 1:
                warnings.warn(
                    "Warning: Consider using colors instead of cmap"
                    "when overlapping spectra."
                )

            cmap_i = plt.get_cmap(cmap[i % len(cmap)])
            ax["A"].contour(
                x_proj_ppm,
                y_proj_ppm,
                data[y_indices, x_indices],
                contour_levels,
                cmap=cmap_i,
                linewidths=defaults["linewidth_contour"],
                norm=LogNorm(vmin=contour_levels[0], vmax=contour_levels[-1]),
            )

            if proj_color and i < len(proj_color):
                _pcolor = proj_color[i]
            else:
                _pcolor = cmap_i(
                    mcolors.Normalize(
                        vmin=contour_levels.min(), vmax=contour_levels.max()
                    )(contour_levels[0])
                )

            ax["a"].plot(
                x_proj_ppm,
                proj_x,
                linewidth=defaults["linewidth_proj"],
                color=_pcolor,
            )
            ax["a"].axis(False)
            ax["b"].plot(
                -proj_y,
                y_proj_ppm,
                linewidth=defaults["linewidth_proj"],
                color=_pcolor,
            )
            ax["b"].axis(False)
        elif color is not None:
            contour_color = color[i % len(color)]
            ax["A"].contour(
                x_proj_ppm,
                y_proj_ppm,
                data[y_indices, x_indices],
                contour_levels,
                colors=contour_color,
                linewidths=defaults["linewidth_contour"],
            )

            if proj_color and i < len(proj_color):
                _pcolor = proj_color[i]
            else:
                _pcolor = contour_color

            ax["a"].plot(
                x_proj_ppm,
                proj_x,
                linewidth=defaults["linewidth_proj"],
                color=_pcolor,
            )
            ax["a"].axis(False)
            ax["b"].plot(
                -proj_y,
                y_proj_ppm,
                linewidth=defaults["linewidth_proj"],
                color=_pcolor,
            )
            ax["b"].axis(False)

        else:
            _pcolor = "black"
            # Create contour plot with basic black color
            ax["A"].contour(
                x_proj_ppm,
                y_proj_ppm,
                data[y_indices, x_indices],
                contour_levels,
                colors="black",
                linewidths=defaults["linewidth_contour"],
            )
            ax["a"].plot(
                x_proj_ppm,
                proj_x,
                linewidth=defaults["linewidth_proj"],
                color=_pcolor,
            )
            ax["a"].axis(False)
            ax["b"].plot(
                -proj_y,
                y_proj_ppm,
                linewidth=defaults["linewidth_proj"],
                color=_pcolor,
            )
            ax["b"].axis(False)
        if xaxislabel := defaults.get("xaxislabel"):
            defaults["xaxislabel"] = xaxislabel
        else:
            defaults["xaxislabel"] = f"$^{{{number_x}}}\\mathrm{{{nucleus_x}}}$ (ppm)"

        if "yaxislabel" in kwargs:
            defaults["yaxislabel"] = kwargs["yaxislabel"]
        elif yaxislabel := defaults.get("yaxislabel"):
            defaults["yaxislabel"] = yaxislabel
        else:
            defaults["yaxislabel"] = f"$^{{{number_y}}}\\mathrm{{{nucleus_y}}}$ (ppm)"

        _apply_tick_spacing(ax["A"], defaults)

        if (
            homonuclear
            and "yaxislabel" not in kwargs
            and "xaxislabel" not in kwargs
            and defaults["yaxislabel"] != defaults["xaxislabel"]
            and number_y == number_x
            and nucleus_y == nucleus_x
        ):
            defaults["yaxislabel"] = defaults["xaxislabel"]

        ax["A"].set_xlabel(
            defaults["xaxislabel"],
            fontsize=defaults["axisfontsize"],
            fontname=defaults["axisfont"] if defaults["axisfont"] else None,
        )
        ax["A"].set_ylabel(
            defaults["yaxislabel"],
            fontsize=defaults["axisfontsize"],
            fontname=defaults["axisfont"] if defaults["axisfont"] else None,
        )
        ax["A"].yaxis.set_label_position("right")
        ax["A"].yaxis.tick_right()
        ax["A"].tick_params(
            axis="x",
            labelsize=defaults["tickfontsize"],
            labelfontfamily=defaults["tickfont"] if defaults["tickfont"] else None,
        )
        ax["A"].tick_params(
            axis="y",
            labelsize=defaults["tickfontsize"],
            labelfontfamily=defaults["tickfont"] if defaults["tickfont"] else None,
        )

        if diagonal is not None:
            x_diag = np.linspace(
                xlim[0] if xlim else ppm_x_limits[0],
                xlim[1] if xlim else ppm_x_limits[1],
                100,
            )
            y_diag = diagonal * x_diag
            ax["A"].plot(x_diag, y_diag, linestyle="--", color="gray")

        if xlim:
            ax["A"].set_xlim(xlim)
            ax["a"].set_xlim(xlim)
        if ylim:
            ax["A"].set_ylim(ylim)
            ax["b"].set_ylim(ylim)

    _save_figure(fig, save, filename, format, "2d_nmr_spectrum")
    return _handle_show_return(fig, ax, return_fig, save)


def bruker2d_grid(
    spectra: dict | list[dict],
    subplot_dims: tuple[int, int] = (1, 1),
    contour_start: float | None = None,
    contour_num: int = 10,
    contour_factor: float = 1.2,
    cmap: str | list[str] | None = None,
    color: list[str] | None = None,
    proj_color: list[str] | None = None,
    xlim: tuple[float, float] | list[tuple[float, float]] | None = None,
    ylim: tuple[float, float] | list[tuple[float, float]] | None = None,
    titles: list[str] | None = None,
    save: bool = False,
    filename: str | None = None,
    format: str | None = None,
    diagonal: float | None = None,
    homonuclear: bool = False,
    return_fig: bool = False,
    **kwargs,
) -> tuple | None:
    """
    Plots multiple 2D Bruker NMR spectra in a grid layout with projections.

    Parameters
    ----------
    spectra : dict or list of dict
        Dictionary or list of dictionaries containing spectrum data.
    subplot_dims : tuple, optional
        Grid dimensions as (rows, cols). Default is (1, 1).
    contour_start : float, optional
        Start value for the contour levels.
    contour_num : int, optional
        Number of contour levels. Default is 10.
    contour_factor : float, optional
        Factor by which the contour levels increase. Default is 1.2.
    cmap : str or list of str, optional
        Colormap(s) for contour fill.
    color : list of str, optional
        Colors for contour lines when overlaying spectra.
    proj_color : list of str, optional
        Colors for the projections.
    xlim : tuple or list of tuples, optional
        X-axis limits. A single tuple applies to all subplots.
        A list of tuples sets per-subplot limits.
    ylim : tuple or list of tuples, optional
        Y-axis limits. Same format as xlim.
    titles : list of str, optional
        Titles for each subplot.
    save : bool, optional
        Whether to save the plot.
    filename : str, optional
        Filename for saving (without extension).
    format : str, optional
        File format for saving.
    diagonal : float or None, optional
        Slope of the diagonal reference line.
    homonuclear : bool, optional
        True if this is a homonuclear experiment.
    return_fig : bool, optional
        Whether to return (fig, axes).
    **kwargs : dict
        Additional customization (axisfontsize, tickfontsize, linewidth_contour,
        linewidth_proj, alpha, xaxislabel, yaxislabel, tickspacing, etc.).

    Returns
    -------
    tuple of (Figure, list of dict) or None
        (fig, axes) if return_fig is True, else None.

    Example:
        bruker2d_grid([spectrum1, spectrum2], subplot_dims=(1, 2), contour_start=0.1,
                  contour_num=10, contour_factor=1.2, cmap='viridis',
                  xlim=(0, 100), ylim=(0, 100), save=True,
                  filename='2d_spectra_grid', format='png')
    """
    spectra = spectra if isinstance(spectra, list) else [spectra]

    if not all(s["ndim"] == 2 for s in spectra):
        raise ValueError("All spectra must be 2-dimensional for bruker2d_grid.")

    defaults = DEFAULTS.copy()
    defaults["yaxislabel"] = None
    _validate_kwargs(kwargs, defaults, "bruker2d_grid")
    defaults.update(
        {k: v for k, v in kwargs.items() if k in defaults and v is not None}
    )

    rows, cols = subplot_dims
    fig = plt.figure(figsize=(6 * cols, 6 * rows))

    gs = fig.add_gridspec(rows, cols, wspace=0.15, hspace=0.15)

    axes = []

    for idx, spectrum in enumerate(spectra):
        if idx >= rows * cols:
            break

        row = idx // cols
        col = idx % cols

        # Create subgrid for each 2D plot with projections
        gs_sub = gs[row, col].subgridspec(10, 10, wspace=0.01, hspace=0.01)

        ax_top = fig.add_subplot(gs_sub[0, 1:])
        ax_left = fig.add_subplot(gs_sub[1:, 0])
        ax_main = fig.add_subplot(gs_sub[1:, 1:], sharex=ax_top, sharey=ax_left)

        data = spectrum["data"]
        nuclei_list = spectrum["nuclei"]

        if homonuclear:
            nuclei_x = nuclei_list[1]
            nuclei_y = nuclei_list[1]
        else:
            nuclei_x = nuclei_list[1]
            nuclei_y = nuclei_list[0]

        number_x, nucleus_x = _parse_nucleus(nuclei_x)
        number_y, nucleus_y = _parse_nucleus(nuclei_y)

        ppm_x = spectrum["ppm_scale"][1]
        ppm_y = spectrum["ppm_scale"][0]

        # Resolve per-subplot xlim/ylim
        effective_xlim = _resolve_per_subplot(xlim, idx)
        effective_ylim = _resolve_per_subplot(ylim, idx)

        # Handle xlim and ylim for data slicing
        if effective_xlim:
            x_min_idx = np.abs(ppm_x - max(effective_xlim)).argmin()
            x_max_idx = np.abs(ppm_x - min(effective_xlim)).argmin()
            x_indices = slice(min(x_min_idx, x_max_idx), max(x_min_idx, x_max_idx))
        else:
            x_indices = slice(None)

        if effective_ylim:
            y_min_idx = np.abs(ppm_y - max(effective_ylim)).argmin()
            y_max_idx = np.abs(ppm_y - min(effective_ylim)).argmin()
            y_indices = slice(min(y_min_idx, y_max_idx), max(y_min_idx, y_max_idx))
        else:
            y_indices = slice(None)

        # Calculate or retrieve projections
        if (
            isinstance(spectrum["projections"], dict)
            and "f2" in spectrum["projections"]
            and "f1" in spectrum["projections"]
        ):
            if xlim is None and ylim is None:
                proj_x = spectrum["projections"]["f2"]
                proj_y = spectrum["projections"]["f1"]
            else:
                zoomed_data = data[y_indices, x_indices]
                proj_x = np.amax(zoomed_data, axis=0)
                proj_y = np.amax(zoomed_data, axis=1)
        else:
            zoomed_data = data[y_indices, x_indices]
            proj_x = np.amax(zoomed_data, axis=0)
            proj_y = np.amax(zoomed_data, axis=1)

        if contour_start is None:
            local_contour_start = 0.05 * np.max(data)
        else:
            local_contour_start = contour_start

        assert local_contour_start is not None
        contour_levels = local_contour_start * contour_factor ** np.arange(contour_num)

        x_proj_ppm = ppm_x[x_indices]
        y_proj_ppm = ppm_y[y_indices]

        # Determine colors
        if cmap is not None:
            if isinstance(cmap, str):
                cmap = [cmap]
            cmap_i = plt.get_cmap(cmap[idx % len(cmap)])
            ax_main.contour(
                x_proj_ppm,
                y_proj_ppm,
                data[y_indices, x_indices],
                contour_levels,
                cmap=cmap_i,
                linewidths=defaults["linewidth_contour"],
                norm=LogNorm(vmin=contour_levels[0], vmax=contour_levels[-1]),
            )
            if proj_color and idx < len(proj_color):
                _pcolor = proj_color[idx]
            else:
                _pcolor = cmap_i(
                    mcolors.Normalize(
                        vmin=contour_levels.min(), vmax=contour_levels.max()
                    )(contour_levels[0])
                )
        elif color is not None:
            contour_color = color[idx % len(color)]
            ax_main.contour(
                x_proj_ppm,
                y_proj_ppm,
                data[y_indices, x_indices],
                contour_levels,
                colors=contour_color,
                linewidths=defaults["linewidth_contour"],
            )
            _pcolor = (
                proj_color[idx]
                if proj_color and idx < len(proj_color)
                else contour_color
            )
        else:
            _pcolor = "black"
            ax_main.contour(
                x_proj_ppm,
                y_proj_ppm,
                data[y_indices, x_indices],
                contour_levels,
                colors="black",
                linewidths=defaults["linewidth_contour"],
            )

        # Plot projections
        ax_top.plot(
            x_proj_ppm,
            proj_x,
            linewidth=defaults["linewidth_proj"],
            color=_pcolor,
        )
        ax_top.axis(False)

        ax_left.plot(
            -proj_y,
            y_proj_ppm,
            linewidth=defaults["linewidth_proj"],
            color=_pcolor,
        )
        ax_left.axis(False)

        # Set labels
        if xaxislabel := defaults.get("xaxislabel"):
            x_label = xaxislabel
        else:
            x_label = f"$^{{{number_x}}}\\mathrm{{{nucleus_x}}}$ (ppm)"

        if yaxislabel := defaults.get("yaxislabel"):
            y_label = yaxislabel
        elif homonuclear and number_y == number_x and nucleus_y == nucleus_x:
            y_label = x_label
        else:
            y_label = f"$^{{{number_y}}}\\mathrm{{{nucleus_y}}}$ (ppm)"

        ax_main.set_xlabel(
            x_label,
            fontsize=defaults["axisfontsize"],
            fontname=defaults["axisfont"],
        )
        ax_main.set_ylabel(
            y_label,
            fontsize=defaults["axisfontsize"],
            fontname=defaults["axisfont"],
        )
        ax_main.yaxis.set_label_position("right")
        ax_main.yaxis.tick_right()

        # Tick params
        ax_main.tick_params(
            axis="both",
            labelsize=defaults["tickfontsize"],
            labelfontfamily=defaults["tickfont"],
        )

        _apply_tick_spacing(ax_main, defaults)

        # Diagonal line
        if diagonal is not None:
            diag_xlim = effective_xlim if effective_xlim else (ppm_x[0], ppm_x[-1])
            x_diag = np.linspace(diag_xlim[0], diag_xlim[1], 100)
            ax_main.plot(x_diag, diagonal * x_diag, "k--", lw=1)

        # Set axis limits
        if effective_xlim:
            ax_main.set_xlim(effective_xlim)
        if effective_ylim:
            ax_main.set_ylim(effective_ylim)

        # Add title if provided
        if titles is not None and idx < len(titles):
            ax_top.set_title(
                titles[idx], fontsize=defaults["axisfontsize"], fontweight="bold", pad=5
            )

        axes.append({"main": ax_main, "top": ax_top, "left": ax_left})

    _save_figure(fig, save, filename, format, "bruker_2d_grid")
    return _handle_show_return(fig, (fig, axes), return_fig, save)


def bruker1d(
    spectra: dict | list[dict],
    labels: list[str] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    save: bool = False,
    filename: str | None = None,
    format: str | None = None,
    frame: bool = False,
    normalize: str | None = None,
    stacked: bool = False,
    color: list[str] | None = None,
    return_fig: bool = False,
    **kwargs,
):
    """
    Plots one or more 1D NMR spectra from spectrum dictionaries.

    Parameters
    ----------
    spectra : dict or list of dict
        Dictionary or list of dictionaries containing spectrum data.
    labels : list of str, optional
        Legend labels for each spectrum.
    xlim : tuple, optional
        The limits for the x-axis as (left, right).
    ylim : tuple, optional
        The limits for the y-axis as (bottom, top).
    save : bool, optional
        Whether to save the plot.
    filename : str, optional
        Filename for saving (without extension).
    format : str, optional
        File format for saving (e.g., 'png', 'pdf').
    frame : bool, optional
        Whether to show the axis frame (spines, y-axis ticks).
    normalize : str, bool, or None, optional
        Normalization method:

        - ``None`` or ``False``: Plot raw data (default).
        - ``'max'`` or ``True``: Normalize by maximum intensity.
        - ``'scans'``: Normalize by number of scans (NS).
    stacked : bool, optional
        Whether to stack spectra vertically with offsets.
    color : list of str, optional
        Colors for each spectrum.
    return_fig : bool, optional
        Whether to return (fig, ax).
    **kwargs : dict
        Additional customization (axisfontsize, tickfontsize, linewidth,
        linestyle, alpha, xaxislabel, yaxislabel, tickspacing, etc.).

    Returns
    -------
    tuple of (Figure, Axes) or None
        (fig, ax) if return_fig is True, else None.
    """

    spectra = spectra if isinstance(spectra, list) else [spectra]

    if not all(s["ndim"] == 1 for s in spectra):
        raise ValueError("All spectra must be 1-dimensional for bruker1d.")

    defaults = DEFAULTS.copy()
    defaults["yaxislabel"] = None
    _validate_kwargs(kwargs, defaults, "bruker1d")
    defaults.update(
        {k: v for k, v in kwargs.items() if k in defaults and v is not None}
    )

    fig, ax = plt.subplots()

    current_stack_offset = 0.0

    first_nuclei = spectra[0]["nuclei"]
    number, nucleus = _parse_nucleus(first_nuclei)

    for i, spectrum in enumerate(spectra):
        data_to_plot = None
        if normalize == "max":
            data_to_plot = spectrum.get("norm_max")
            if data_to_plot is None:
                warnings.warn(
                    f"Pre-calculated 'norm_max' data not found for {spectrum['path']}. Plotting raw data.",
                    UserWarning,
                )
                data_to_plot = spectrum["data"]
        elif normalize == "scans":
            data_to_plot = spectrum.get("norm_scans")
            if data_to_plot is None:
                warnings.warn(
                    f"Pre-calculated 'norm_scans' data not found or calculation failed for {spectrum['path']}. Plotting raw data.",
                    UserWarning,
                )
                data_to_plot = spectrum["data"]
        elif normalize is None or normalize is False:
            data_to_plot = spectrum["data"]
        else:
            raise ValueError(
                f"Invalid normalize option: '{normalize}'. Choose 'max', 'scans', or None."
            )

        ppm = spectrum["ppm_scale"]

        plot_data_adjusted = data_to_plot
        if stacked:
            # Apply the offset
            plot_data_adjusted = data_to_plot + current_stack_offset
            current_stack_offset += np.amax(data_to_plot) * 1.1

        plot_kwargs = {
            "linestyle": defaults["linestyle"],
            "linewidth": defaults["linewidth"],
            "alpha": defaults["alpha"],
        }

        if labels:
            plot_kwargs["label"] = labels[i] if i < len(labels) else f"Spectrum {i + 1}"

        if color:
            plot_kwargs["color"] = color[i] if i < len(color) else None

        ax.plot(ppm, plot_data_adjusted, **plot_kwargs)

    if labels:
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=defaults["labelsize"],
            prop={"family": defaults["tickfont"], "size": defaults["labelsize"]},
        )

    # --- Axis Setup ---
    if xaxislabel := defaults["xaxislabel"]:
        ax.set_xlabel(
            xaxislabel, fontsize=defaults["axisfontsize"], fontname=defaults["axisfont"]
        )
    else:
        # Use nucleus info from the first spectrum
        ax.set_xlabel(
            f"$^{{{number}}}\\mathrm{{{nucleus}}}$ (ppm)",
            fontsize=defaults["axisfontsize"],
            fontname=defaults["axisfont"],
        )

    ax.tick_params(
        axis="x",
        labelsize=defaults["tickfontsize"],
        labelfontfamily=defaults["tickfont"],
    )

    # Apply x-axis tick spacing (y-axis only when frame is shown)
    xtick = defaults.get("xtickspacing") or defaults.get("tickspacing")
    if xtick:
        ax.xaxis.set_major_locator(MultipleLocator(xtick))

    if not frame:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_yticklabels([])
        ax.set_yticks([])
    else:
        if defaults["yaxislabel"]:
            ax.set_ylabel(
                defaults["yaxislabel"],
                fontsize=defaults["axisfontsize"],
                fontname=defaults["axisfont"],
            )
        ax.tick_params(
            axis="y",
            labelsize=defaults["tickfontsize"],
            labelfontfamily=defaults["tickfont"],
        )

    if xlim:
        ax.set_xlim(xlim)
    else:
        current_xlim = ax.get_xlim()
        if current_xlim[0] < current_xlim[1]:
            ax.set_xlim((current_xlim[1], current_xlim[0]))

    if ylim is not None:
        ax.set_ylim(ylim)

    _save_figure(fig, save, filename, format, "1d_nmr_spectrum")
    return _handle_show_return(fig, (fig, ax), return_fig, save)


def bruker1d_grid(
    spectra: dict | list[dict],
    labels: list[str] | None = None,
    subplot_dims: tuple[int, int] = (1, 1),
    xlim: tuple[float, float] | list[tuple[float, float]] | None = None,
    ylim: tuple[float, float] | list[tuple[float, float]] | None = None,
    save: bool = False,
    filename: str | None = None,
    format: str | None = "png",
    frame: bool = False,
    normalize: str | bool | list | None = False,
    color: list[str] | None = None,
    return_fig: bool = False,
    **kwargs,
) -> tuple | None:
    """
    Plots 1D NMR spectra from Bruker data in a grid of subplots.

    Parameters
    ----------
    spectra : dict or list of dict
        Dictionary or list of dictionaries containing spectrum data.
    labels : list of str, optional
        Legend labels for each subplot's spectrum.
    subplot_dims : tuple, optional
        Grid dimensions as (rows, cols). Default is (1, 1).
    xlim : tuple or list of tuples, optional
        X-axis limits. A single tuple applies to all subplots.
        A list of tuples sets per-subplot limits
        (e.g., ``[(200, 0), (100, 0)]``).
    ylim : tuple or list of tuples, optional
        Y-axis limits. Same format as xlim.
    save : bool, optional
        Whether to save the plot.
    filename : str, optional
        Filename for saving (without extension).
    format : str, optional
        File format for saving (e.g., 'png', 'pdf'). Default is 'png'.
    frame : bool, optional
        Whether to show the axis frame (spines, y-axis ticks).
    normalize : str, bool, list, or None, optional
        Normalization method:

        - ``None`` or ``False``: Plot raw data (default).
        - ``'max'`` or ``True``: Normalize by maximum intensity.
        - ``'scans'``: Normalize by number of scans (NS).
        - ``list``: Per-spectrum normalization, e.g. ``['max', None, 'scans']``.
    color : list of str, optional
        Colors for each subplot's spectrum.
    return_fig : bool, optional
        Whether to return (fig, axes).
    **kwargs : dict
        Additional customization (axisfontsize, tickfontsize, linewidth,
        linestyle, alpha, xaxislabel, yaxislabel, tickspacing, etc.).

    Returns
    -------
    tuple of (Figure, ndarray of Axes) or None
        (fig, axes) if return_fig is True, else None.

    Example:
        bruker1d_grid([spectrum1, spectrum2], labels=['Spectrum 1', 'Spectrum 2'], subplot_dims=(1, 2), xlim=[(0, 100), (0, 100)], save=True, filename='1d_spectra', format='png', frame=False, normalize='max', color=['red', 'blue'])
    """

    spectra = spectra if isinstance(spectra, list) else [spectra]

    if not all(s["ndim"] == 1 for s in spectra):
        raise ValueError("All spectra must be 1-dimensional for bruker1d_grid.")

    defaults = DEFAULTS.copy()
    _validate_kwargs(kwargs, defaults, "bruker1d_grid")
    defaults.update(
        {k: v for k, v in kwargs.items() if k in defaults and v is not None}
    )

    rows, cols = subplot_dims
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    # Ensure axes is flat list
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i, spectrum in enumerate(spectra):
        if i >= len(axes):
            break

        ax = axes[i]

        nuclei = spectrum["nuclei"]
        number, nucleus = _parse_nucleus(nuclei)

        ppm = spectrum["ppm_scale"]
        if isinstance(normalize, list):
            if len(normalize) != len(spectra):
                raise ValueError(
                    "The length of the normalize list must be equal to the number of spectra."
                )
            normalize_option = normalize[i]
        else:
            normalize_option = normalize

        if normalize_option == "max" or normalize_option is True:
            data = spectrum.get("norm_max")
            if data is None:
                data = spectrum["data"] / np.amax(spectrum["data"])
        elif normalize_option == "scans":
            data = spectrum.get("norm_scans")
            if data is None:
                warnings.warn(
                    f"Pre-calculated 'norm_scans' data not found for {spectrum['path']}. Using raw data.",
                    UserWarning,
                )
                data = spectrum["data"]
        else:
            data = spectrum["data"]

        plot_kwargs = {
            "linestyle": defaults["linestyle"],
            "linewidth": defaults["linewidth"],
            "alpha": defaults["alpha"],
        }

        if labels and i < len(labels):
            plot_kwargs["label"] = labels[i]

        if color and i < len(color):
            plot_kwargs["color"] = color[i]

        ax.plot(ppm, data, **plot_kwargs)

        if labels and i < len(labels):
            ax.legend(
                fontsize=defaults["labelsize"],
                prop={"family": defaults["tickfont"], "size": defaults["labelsize"]},
            )

        if xaxislabel := defaults["xaxislabel"]:
            ax.set_xlabel(
                xaxislabel,
                fontsize=defaults["axisfontsize"],
                fontname=defaults["axisfont"],
            )
        else:
            ax.set_xlabel(
                f"$^{{{number}}}\\mathrm{{{nucleus}}}$ (ppm)",
                fontsize=defaults["axisfontsize"],
                fontname=defaults["axisfont"],
            )

        ax.tick_params(
            axis="x",
            labelsize=defaults["tickfontsize"],
            labelfontfamily=defaults["tickfont"],
        )

        # Apply x-axis tick spacing
        xtick = defaults["xtickspacing"] or defaults["tickspacing"]
        if xtick:
            ax.xaxis.set_major_locator(MultipleLocator(xtick))

        if not frame:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.set_yticklabels([])
            ax.set_yticks([])
        else:
            if yaxislabel := defaults["yaxislabel"]:
                ax.set_ylabel(
                    yaxislabel,
                    fontsize=defaults["axisfontsize"],
                    fontname=defaults["axisfont"],
                )
            else:
                ax.set_ylabel(
                    defaults["yaxislabel"],
                    fontsize=defaults["axisfontsize"],
                    fontname=defaults["axisfont"],
                )

                ax.tick_params(
                    axis="y",
                    labelsize=defaults["tickfontsize"],
                    labelfontfamily=defaults["tickfont"],
                )

        effective_xlim = _resolve_per_subplot(xlim, i)
        if effective_xlim:
            ax.set_xlim(effective_xlim)

        effective_ylim = _resolve_per_subplot(ylim, i)
        if effective_ylim:
            ax.set_ylim(effective_ylim)

    plt.tight_layout()

    _save_figure(fig, save, filename, format, "1d_nmr_spectra")
    return _handle_show_return(fig, (fig, axes), return_fig, save)


# Plot 2D NMR data from CSV or DataFrame
def df2d(
    path: str | pd.DataFrame,
    contour_start: float = 1e5,
    contour_num: int = 10,
    contour_factor: float = 1.2,
    cmap: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    save: bool = False,
    filename: str | None = None,
    format: str | None = None,
    return_fig: bool = False,
) -> tuple | dict | None:
    """
    Plot 2D NMR data from a CSV file or a Pandas DataFrame.

    Parameters
    ----------
    path : str or DataFrame
        Path to a CSV file, or an existing ``pandas.DataFrame`` with
        columns for F1 ppm, F2 ppm, and intensity.
    contour_start : float, optional
        Starting contour level. Default is 1e5.
    contour_num : int, optional
        Number of contour levels. Default is 10.
    contour_factor : float, optional
        Multiplicative factor between successive contour levels.
        Default is 1.2.
    cmap : str, optional
        Matplotlib colormap name. Default is ``'Greys'``.
    xlim : tuple of float, optional
        Limits for the x-axis.
    ylim : tuple of float, optional
        Limits for the y-axis.
    save : bool, optional
        Whether to save the plot. Default is False.
    filename : str, optional
        Filename for saving (without extension).
    format : str, optional
        File format for saving (e.g. ``'png'``, ``'pdf'``).
    return_fig : bool, optional
        Whether to return the figure and axes dict. Default is False.

    Returns
    -------
    dict or None
        Axes dictionary if *return_fig* is True, else None.
    """

    # Check if path to CSV or DataFrame
    df_nmr = path if isinstance(path, pd.DataFrame) else pd.read_csv(path)

    cols = df_nmr.columns
    f1_nuclei, f1_units = cols[0].split()
    number_x, nucleus_x = _parse_nucleus(f1_nuclei)
    f2_nuclei, f2_units = cols[1].split()
    number_y, nucleus_y = _parse_nucleus(f2_nuclei)
    data_grid = df_nmr.pivot_table(index=cols[0], columns=cols[1], values="intensity")
    proj_f1, proj_f2 = calculate_projections(df_nmr, export=False)

    f1 = data_grid.index.to_numpy()
    f2 = data_grid.columns.to_numpy()
    x, y = np.meshgrid(f2, f1)
    z = data_grid.to_numpy()

    contour_levels = contour_start * contour_factor ** np.arange(contour_num)

    ax = plt.figure(constrained_layout=False).subplot_mosaic(
        """
    .a
    bA
    """,
        gridspec_kw={
            "height_ratios": [0.9, 6.0],
            "width_ratios": [0.8, 6.0],
            "wspace": 0.03,
            "hspace": 0.04,
        },
    )

    if cmap is not None:
        ax["A"].contourf(
            x,
            y,
            z,
            contour_levels,
            cmap=cmap,
            norm=LogNorm(vmin=contour_levels[0], vmax=contour_levels[-1]),
        )
    else:
        ax["A"].contourf(
            x,
            y,
            z,
            contour_levels,
            cmap="Greys",
            norm=LogNorm(vmin=contour_levels[0], vmax=contour_levels[-1]),
        )

    if proj_f2 is not None and proj_f1 is not None:
        ax["a"].plot(
            proj_f2[f"{f2_nuclei} {f2_units}"], proj_f2["F2 projection"], color="black"
        )
        ax["a"].axis(False)
        ax["b"].plot(
            -proj_f1["F1 projection"], proj_f1[f"{f1_nuclei} {f1_units}"], color="black"
        )
        ax["b"].axis(False)

    ax["A"].set_xlabel(f"$^{{{number_y}}}\\mathrm{{{nucleus_y}}}$ (ppm)", fontsize=13)
    ax["A"].set_ylabel(f"$^{{{number_x}}}\\mathrm{{{nucleus_x}}}$ (ppm)", fontsize=13)
    ax["A"].yaxis.set_label_position("right")
    ax["A"].yaxis.tick_right()
    ax["A"].tick_params(axis="x", labelsize=12)
    ax["A"].tick_params(axis="y", labelsize=12)

    if xlim:
        ax["A"].set_xlim(xlim)
        ax["a"].set_xlim(xlim)
    if ylim:
        ax["A"].set_ylim(ylim)
        ax["b"].set_ylim(ylim)

    _save_figure(ax["A"].figure, save, filename, format, "2d_nmr_spectrum")
    return _handle_show_return(ax["A"].figure, ax, return_fig, save)


# Functions for DMFit
def dmfit1d(
    spectra: dict | list[dict],
    color: str | list[str] = "b",
    stacked: bool = False,
    model_show: bool = True,
    model_color: str = "red",
    model_linewidth: float = 1,
    model_linestyle: str = "--",
    model_alpha: float = 1,
    deconv_show: bool = True,
    deconv_color: str | list[str] | None = None,
    deconv_alpha: float = 0.3,
    frame: bool = False,
    labels: list[str] | None = None,
    xlim: tuple[float, float] | None = None,
    save: bool = False,
    format: str | None = None,
    filename: str | None = None,
    return_fig: bool = False,
    **kwargs,
) -> tuple | None:
    """
    Plot one or more 1D DMFit spectra with optional model and deconvolution lines.

    Parameters
    ----------
    spectra : dict or list of dict
        Dictionary (or list of dictionaries) containing DMFit 1D spectrum data,
        each including a 'dmfit_dataframe' key with the parsed data.
    color : str or list of str, optional
        The color(s) of the spectrum line(s). The default is 'b'.
    stacked : bool, optional
        Whether to stack multiple spectra vertically. The default is False.
    model_show : bool, optional
        Whether to show the model line. The default is True.
    model_color : str, optional
        The color of the model line. The default is 'red'.
    model_linewidth : int, optional
        The width of the model line. The default is 1.
    model_linestyle : str, optional
        The style of the model line. The default is '--'.
    model_alpha : float, optional
        The transparency of the model line. The default is 1.
    deconv_show : bool, optional
        Whether to show the deconvoluted lines. The default is True.
    deconv_color : str, optional
        The color of the deconvoluted lines. The default is None.
    deconv_alpha : float, optional
        The transparency of the deconvoluted lines. The default is 0.3.
    frame : bool, optional
        Whether to show the frame. The default is False.
    labels : list, optional
        Legend labels. The default is None.
    xlim : tuple, optional
        The limits for the x axis. The default is None.
    save : bool, optional
        Whether to save the figure. The default is False.
    format : str, optional
        The format to save the figure. The default is None.
    filename : str, optional
        The name of the file to save the figure. The default is None.
    return_fig : bool, optional
        Whether to return the figure. The default is False.
    **kwargs : dict, optional
        Additional keyword arguments:

        - linewidth : float
            Width of the spectrum line (default 1.0).
        - linestyle : str
            Style of the spectrum line (default '-').
        - alpha : float
            Transparency of the spectrum line (default 1.0).
        - labelsize : int
            Size of labels in the legend (default 12).
        - xaxislabel : str
            Custom label for x-axis.
        - yaxislabel : str
            Custom label for y-axis.
        - axisfontsize : int
            Font size for axis labels.
        - axisfont : str
            Font family for axis labels.
        - tickfontsize : int
            Font size for tick labels.
        - tickfont : str
            Font family for tick labels.
        - tickspacing : float
            Tick spacing for both axes.
        - xtickspacing : float
            Tick spacing for x-axis only.
        - ytickspacing : float
            Tick spacing for y-axis only.

    Returns
    -------
    fig, ax : tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axes objects, if return_fig is True.
    """

    spectra = spectra if isinstance(spectra, list) else [spectra]
    multi = len(spectra) > 1

    if not spectra or not spectra[0]:
        raise ValueError("Empty spectrum data provided.")

    defaults = DEFAULTS.copy()
    defaults["yaxislabel"] = None  # dmfit1d doesn't show y-axis label by default
    _validate_kwargs(kwargs, defaults, "dmfit1d")
    defaults.update(
        {k: v for k, v in kwargs.items() if k in defaults and v is not None}
    )

    fig, ax = plt.subplots()
    current_stack_offset = 0.0

    for i, spectrum in enumerate(spectra):
        dmfit_df = spectrum.get("dmfit_dataframe")

        if dmfit_df is None:
            raise ValueError(
                "DMfit DataFrame not found in spectrum data. Read data with provider='dmfit'."
            )

        n_lines = sum(col.startswith("Line#") for col in dmfit_df.columns)

        c = color[i % len(color)] if isinstance(color, list) else color

        # Compute stack offset
        spectrum_data = dmfit_df["Spectrum"].to_numpy().copy()
        model_data = dmfit_df["Model"].to_numpy().copy()
        if stacked:
            spectrum_data = spectrum_data + current_stack_offset
            model_data = model_data + current_stack_offset

        # Spectrum label: per-spectrum label in multi mode, or full labels[0] for single
        if labels and not multi:
            spec_label = labels[0] if len(labels) > 0 else None
        elif labels and multi:
            spec_label = labels[i] if i < len(labels) else None
        else:
            spec_label = None

        ax.plot(
            dmfit_df["ppm"],
            spectrum_data,
            color=c,
            linewidth=defaults["linewidth"],
            linestyle=defaults["linestyle"],
            alpha=defaults["alpha"],
            label=spec_label,
        )
        if model_show:
            # In multi-spectrum mode, don't add legend labels for model/deconv
            model_label = None
            if not multi and labels and len(labels) > 1:
                model_label = labels[1]
            ax.plot(
                dmfit_df["ppm"],
                model_data,
                color=model_color,
                linewidth=model_linewidth,
                linestyle=model_linestyle,
                alpha=model_alpha,
                label=model_label,
            )
        if deconv_show:
            for j in range(1, n_lines + 1):
                line_data = dmfit_df[f"Line#{j}"].to_numpy().copy()
                if stacked:
                    line_data = line_data + current_stack_offset
                if deconv_color is not None:
                    # Support per-line colors (list) or single color (str)
                    dc = (
                        deconv_color[(j - 1) % len(deconv_color)]
                        if isinstance(deconv_color, list)
                        else deconv_color
                    )
                    ax.fill_between(
                        dmfit_df["ppm"],
                        line_data,
                        current_stack_offset if stacked else 0,
                        alpha=deconv_alpha,
                        color=dc,
                    )
                else:
                    ax.fill_between(
                        dmfit_df["ppm"],
                        line_data,
                        current_stack_offset if stacked else 0,
                        alpha=deconv_alpha,
                    )

        if stacked:
            current_stack_offset += np.amax(dmfit_df["Spectrum"].values) * 1.1

    if labels:
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=defaults["labelsize"],
            prop={"family": defaults["tickfont"], "size": defaults["labelsize"]},
        )
    if xlim:
        ax.set_xlim(xlim)
    if defaults["yaxislabel"]:
        ax.set_ylabel(
            defaults["yaxislabel"],
            fontsize=defaults["axisfontsize"],
            fontname=defaults["axisfont"],
        )
    if defaults["xaxislabel"]:
        ax.set_xlabel(
            defaults["xaxislabel"],
            fontsize=defaults["axisfontsize"],
            fontname=defaults["axisfont"],
        )

    ax.tick_params(
        axis="both",
        labelsize=defaults["tickfontsize"],
        labelfontfamily=defaults["tickfont"],
    )

    _apply_tick_spacing(ax, defaults)

    if frame:
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.spines["left"].set_visible(True)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])

    _save_figure(fig, save, filename, format, "dmfit_1d_spectrum")
    return _handle_show_return(fig, (fig, ax), return_fig, save)


def dmfit2d(
    spectra: dict | list[dict],
    contour_start: float = 1e5,
    contour_num: int = 10,
    contour_factor: float = 1.2,
    color: str | list[str] | None = None,
    proj_color: str | list[str] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    labels: list[str] | None = None,
    save: bool = False,
    filename: str | None = None,
    format: str | None = None,
    axis_right: bool = True,
    diagonal: float | None = None,
    return_fig: bool = False,
    **kwargs,
) -> tuple | dict | None:
    """
    Plot 2D DMFit data with 1D projections.

    Parameters
    ----------
    spectra : dict or list of dict
        Dictionary or list of dictionaries containing DMFit 2D spectrum data.
    contour_start : float, optional
        The starting contour level. Default is 1e5.
    contour_num : int, optional
        The number of contour levels. Default is 10.
    contour_factor : float, optional
        The factor by which the contour levels increase. Default is 1.2.
    color : str or list, optional
        Color(s) for each spectrum's contours.
    proj_color : str or list, optional
        Color(s) for each spectrum's projections.
    xlim : tuple, optional
        The limits for the x-axis (F2).
    ylim : tuple, optional
        The limits for the y-axis (F1).
    labels : list, optional
        Labels for the spectra in the legend.
    save : bool, optional
        Whether to save the plot.
    filename : str, optional
        Name for the saved file.
    format : str, optional
        Format for the saved file.
    axis_right : bool, optional
        Whether to put the y-axis on the right.
    diagonal : float or None, optional
        Slope of the diagonal line.
    return_fig : bool, optional
        Whether to return the figure and axes dictionary.
    **kwargs : dict, optional
        Additional keyword arguments:

        - labelsize : int
            Size of labels in the legend.
        - linewidth_contour : float
            Width of contour lines.
        - linewidth_proj : float
            Width of projection lines.
        - alpha : float
            Transparency of contours.
        - xaxislabel : str
            Custom label for x-axis (f1).
        - yaxislabel : str
            Custom label for y-axis (f2).
        - axisfontsize : int
            Font size for axis labels.
        - axisfont : str
            Font family for axis labels.
        - tickfontsize : int
            Font size for tick labels.
        - tickfont : str
            Font family for tick labels.

    Returns
    -------
    fig : matplotlib.figure.Figure, optional
        The figure object, if return_fig is True.
    ax_dict : dict of matplotlib.axes.Axes, optional
        Dictionary of axes objects (e.g., 'A', 'a', 'b'), if return_fig is True.
    """

    defaults = DEFAULTS.copy()
    _validate_kwargs(kwargs, defaults, "dmfit2d")
    defaults.update(
        {k: v for k, v in kwargs.items() if k in defaults and v is not None}
    )

    spectra_dicts = spectra if isinstance(spectra, list) else [spectra]

    if labels is None:
        plot_labels = [f"Spectrum {idx + 1}" for idx in range(len(spectra_dicts))]
    else:
        plot_labels = labels

    if not all(s["ndim"] == 2 for s in spectra_dicts):
        raise ValueError("All spectra must be 2D.")
    if not all(s["metadata"]["provider_type"] == "dmfit" for s in spectra_dicts):
        raise ValueError("All spectra must be from DMFit provider.")

    num_spectra = len(spectra_dicts)
    default_colors = [
        "black",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    contour_colors_list = []
    if isinstance(color, str):
        contour_colors_list = [color] * num_spectra
    elif isinstance(color, list):
        contour_colors_list = [color[i % len(color)] for i in range(num_spectra)]
    else:
        contour_colors_list = [
            default_colors[i % len(default_colors)] for i in range(num_spectra)
        ]

    projection_colors_list = []
    if isinstance(proj_color, str):
        projection_colors_list = [proj_color] * num_spectra
    elif isinstance(proj_color, list):
        projection_colors_list = [
            proj_color[i % len(proj_color)] for i in range(num_spectra)
        ]
    else:
        projection_colors_list = contour_colors_list

    fig = plt.figure(constrained_layout=False, figsize=(8, 7))
    ax_dict = fig.subplot_mosaic(
        """
        .a
        bA
        """,
        gridspec_kw={
            "height_ratios": [0.9, 6.0],
            "width_ratios": [0.8, 6.0],
            "wspace": 0.03,
            "hspace": 0.04,
        },
    )
    main_ax = ax_dict["A"]
    proj_ax_f2 = ax_dict["a"]
    proj_ax_f1 = ax_dict["b"]

    legend_elements = []

    for i, spectrum_dict in enumerate(spectra_dicts):
        data = spectrum_dict["data"]
        y_axis_f1 = spectrum_dict["ppm_scale"][0]
        x_axis_f2 = spectrum_dict["ppm_scale"][1]

        proj_f1_data = spectrum_dict["projections"]["f1"]
        proj_f2_data = spectrum_dict["projections"]["f2"]

        current_contour_color = contour_colors_list[i]
        current_proj_color = projection_colors_list[i]

        contour_levels = contour_start * contour_factor ** np.arange(contour_num)

        main_ax.contour(
            x_axis_f2,
            y_axis_f1,
            data,
            levels=contour_levels,
            colors=current_contour_color,
            linewidths=defaults["linewidth_contour"],
            alpha=defaults["alpha"],
        )

        proj_ax_f2.plot(
            x_axis_f2,
            proj_f2_data,
            color=current_proj_color,
            linewidth=defaults["linewidth_proj"],
        )
        proj_ax_f1.plot(
            -proj_f1_data,
            y_axis_f1,
            color=current_proj_color,
            linewidth=defaults["linewidth_proj"],
        )

        if i < len(plot_labels) and plot_labels[i] is not None:
            legend_elements.append(
                Line2D(
                    [0], [0], color=current_contour_color, lw=2, label=plot_labels[i]
                )
            )

    first_spectrum_nuclei = spectra_dicts[0].get("nuclei", ["Unknown", "Unknown"])
    if isinstance(first_spectrum_nuclei, str):
        first_spectrum_nuclei = [first_spectrum_nuclei, first_spectrum_nuclei]

    f2_nuc_str = str(first_spectrum_nuclei[1])
    f1_nuc_str = str(first_spectrum_nuclei[0])

    final_xaxislabel = defaults.get("xaxislabel") or _nucleus_label(f2_nuc_str)
    final_yaxislabel = defaults.get("yaxislabel") or _nucleus_label(f1_nuc_str)

    main_ax.set_xlabel(
        final_xaxislabel,
        fontsize=defaults["axisfontsize"],
        fontname=defaults["axisfont"],
    )
    main_ax.set_ylabel(
        final_yaxislabel,
        fontsize=defaults["axisfontsize"],
        fontname=defaults["axisfont"],
    )

    main_ax.tick_params(
        axis="x",
        labelsize=defaults["tickfontsize"],
        labelfontfamily=defaults["tickfont"],
    )
    main_ax.tick_params(
        axis="y",
        labelsize=defaults["tickfontsize"],
        labelfontfamily=defaults["tickfont"],
    )

    if axis_right:
        main_ax.yaxis.set_label_position("right")
        main_ax.yaxis.tick_right()

    proj_ax_f2.axis(False)
    proj_ax_f1.axis(False)

    if xlim:
        main_ax.set_xlim(xlim)
    else:
        current_xlim_main = main_ax.get_xlim()
        if current_xlim_main[0] < current_xlim_main[1]:
            main_ax.set_xlim((current_xlim_main[1], current_xlim_main[0]))
    proj_ax_f2.set_xlim(main_ax.get_xlim())

    if ylim:
        main_ax.set_ylim(ylim)
    else:
        current_ylim_main = main_ax.get_ylim()
        if current_ylim_main[0] < current_ylim_main[1]:
            main_ax.set_ylim((current_ylim_main[1], current_ylim_main[0]))
    proj_ax_f1.set_ylim(main_ax.get_ylim())

    if diagonal is not None:
        diag_xlim_eff = main_ax.get_xlim()
        x_diag_vals = np.linspace(diag_xlim_eff[0], diag_xlim_eff[1], 100)
        main_ax.plot(x_diag_vals, diagonal * x_diag_vals, "k--", lw=1)

    if legend_elements:
        main_ax.legend(
            handles=legend_elements,
            fontsize=defaults["labelsize"],
            prop={"family": defaults["tickfont"]},
        )

    plt.tight_layout(pad=0.5)

    _save_figure(fig, save, filename, format, "dmfit_2d_projections")
    return _handle_show_return(fig, ax_dict, return_fig, save)


def dmfit1d_grid(
    spectra: dict | list[dict],
    subplot_dims: tuple[int, int] = (1, 1),
    titles: list[str] | None = None,
    xlim: tuple[float, float] | list[tuple[float, float]] | None = None,
    ylim: tuple[float, float] | list[tuple[float, float]] | None = None,
    color: str | list[str] | None = None,
    model_color: str | None = None,
    deconv_color: str | list[str] | None = None,
    save: bool = False,
    filename: str | None = None,
    format: str | None = "png",
    return_fig: bool = False,
    **kwargs,
) -> tuple | None:
    """
    Plot multiple 1D DMFit spectra in a grid layout.

    Each subplot shows an experimental spectrum overlaid with its fit/model
    and optionally its deconvoluted components.

    Parameters
    ----------
    spectra : dict or list of dict
        Dictionary or list of dictionaries containing DMFit 1D spectrum data.
    subplot_dims : tuple, optional
        Grid dimensions as (rows, cols). Default is (1, 1).
    titles : list of str, optional
        Titles for each subplot. If None, no titles are shown.
    xlim : tuple or list of tuples, optional
        X-axis limits. A single tuple applies to all subplots.
        A list of tuples sets per-subplot limits.
    ylim : tuple or list of tuples, optional
        Y-axis limits. Same format as xlim.
    color : str or list of str, optional
        Color(s) for experimental spectra. If a single string, the same color
        is used for all spectra. If a list, each spectrum gets its own color.
    model_color : str, optional
        Color for model/fit spectra. Default is 'red'.
    deconv_color : str, optional
        Color for deconvoluted components. If None, uses default matplotlib colors.
    save : bool, optional
        Whether to save the figure. Default is False.
    filename : str, optional
        Filename for saving (without extension).
    format : str, optional
        File format for saving. Default is 'png'.
    return_fig : bool, optional
        Whether to return figure and axes. Default is False.
    **kwargs : dict
        Additional customization options (axisfontsize, tickfontsize, etc.)
    Returns
    -------
    fig, axes : tuple
        Figure and axes array if return_fig=True.
    """

    spectra = spectra if isinstance(spectra, list) else [spectra]

    if not all(s["ndim"] == 1 for s in spectra):
        raise ValueError("All spectra must be 1-dimensional for dmfit1d_grid.")

    # Convert single color string to list
    if isinstance(color, str):
        color = [color] * len(spectra)

    defaults = DEFAULTS.copy()
    _validate_kwargs(kwargs, defaults, "dmfit1d_grid")
    defaults.update(
        {k: v for k, v in kwargs.items() if k in defaults and v is not None}
    )

    rows, cols = subplot_dims
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i, spectrum in enumerate(spectra):
        if i >= len(axes):
            break

        ax = axes[i]

        if spectrum["metadata"]["provider_type"] != "dmfit":
            raise ValueError("All spectra must be from DMFit provider")

        dmfit_df = spectrum.get("dmfit_dataframe")
        if dmfit_df is None:
            raise ValueError(
                "DMfit DataFrame not found in spectrum. Read data with provider='dmfit'"
            )

        ppm = dmfit_df["ppm"].to_numpy()
        data_exp = dmfit_df["Spectrum"].to_numpy()
        data_model = dmfit_df["Model"].to_numpy()

        n_lines = sum(col.startswith("Line#") for col in dmfit_df.columns)

        exp_color = color[i] if color and i < len(color) else "black"
        fit_color = model_color if model_color else "red"

        # Plot experimental spectrum
        ax.plot(
            ppm,
            data_exp,
            color=exp_color,
            linewidth=defaults["linewidth"],
            linestyle=defaults["linestyle"],
            alpha=defaults["alpha"],
        )

        # Plot model/fit spectrum
        ax.plot(
            ppm,
            data_model,
            color=fit_color,
            linewidth=defaults["linewidth"],
            linestyle="--",
            alpha=defaults["alpha"],
        )

        # Plot deconvoluted components if they exist
        if n_lines > 0:
            for j in range(1, n_lines + 1):
                if deconv_color is not None:
                    dc = (
                        deconv_color[(j - 1) % len(deconv_color)]
                        if isinstance(deconv_color, list)
                        else deconv_color
                    )
                    ax.fill_between(
                        ppm,
                        dmfit_df[f"Line#{j}"],
                        alpha=0.3,
                        color=dc,
                    )
                else:
                    ax.fill_between(ppm, dmfit_df[f"Line#{j}"], alpha=0.3)

        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=defaults["axisfontsize"])

        # X-axis label
        if xaxislabel := defaults.get("xaxislabel"):
            ax.set_xlabel(
                xaxislabel,
                fontsize=defaults["axisfontsize"],
                fontname=defaults["axisfont"],
            )
        else:
            nuclei = spectrum.get("nuclei", "Unknown")
            if nuclei and nuclei != "Unknown":
                ax.set_xlabel(
                    _nucleus_label(nuclei),
                    fontsize=defaults["axisfontsize"],
                    fontname=defaults["axisfont"],
                )
            else:
                ax.set_xlabel(
                    "Chemical Shift (ppm)",
                    fontsize=defaults["axisfontsize"],
                    fontname=defaults["axisfont"],
                )

        ax.tick_params(
            axis="x",
            labelsize=defaults["tickfontsize"],
            labelfontfamily=defaults["tickfont"],
        )

        # Apply x-axis tick spacing only (y-axis hidden)
        xtick = defaults.get("xtickspacing") or defaults.get("tickspacing")
        if xtick:
            ax.xaxis.set_major_locator(MultipleLocator(xtick))

        # Y-axis (no frame by default)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_yticklabels([])
        ax.set_yticks([])

        effective_xlim = _resolve_per_subplot(xlim, i)
        if effective_xlim:
            ax.set_xlim(effective_xlim)

        effective_ylim = _resolve_per_subplot(ylim, i)
        if effective_ylim:
            ax.set_ylim(effective_ylim)

    plt.tight_layout()

    _save_figure(fig, save, filename, format, "1d_dmfit_spectra")
    return _handle_show_return(fig, (fig, axes), return_fig, save)


def dmfit2d_grid(
    spectra: list[dict],
    subplot_dims: tuple[int, int] = (1, 3),
    contour_start: float = 1e5,
    contour_num: int = 10,
    contour_factor: float = 1.2,
    color: list[list[str]] | None = None,
    proj_color: list[list[str]] | None = None,
    xlim: tuple[float, float] | list[tuple[float, float]] | None = None,
    ylim: tuple[float, float] | list[tuple[float, float]] | None = None,
    titles: list[str] | None = None,
    linestyles: list[list[str]] | None = None,
    save: bool = False,
    filename: str | None = None,
    format: str | None = "png",
    diagonal: float | None = None,
    return_fig: bool = False,
    **kwargs,
) -> tuple | None:
    """
    Plot multiple 2D DMFit spectra in a grid layout with projections.
    Each subplot shows an experimental spectrum overlaid with its fit/model.
    This function expects pairs of spectra (experimental + model).

    Parameters
    ----------
    spectra : list of dict
        List of spectrum dictionaries containing DMFit 2D data.
        Should contain pairs: [exp1, model1, exp2, model2, ...].
    subplot_dims : tuple, optional
        Grid dimensions as (rows, cols). Default is (1, 3).
    contour_start : float, optional
        Starting contour level. Default is 1e5.
    contour_num : int, optional
        Number of contour levels. Default is 10.
    contour_factor : float, optional
        Factor by which contour levels increase. Default is 1.2.
    color : list of lists, optional
        Colors for each subplot's [experimental, model] spectra.
        E.g., [['black', 'red'], ['black', 'red'], ...].
        If None, uses default ['black', 'red'] for all subplots.
    proj_color : list of lists, optional
        Colors for projections. Same structure as color.
    xlim : tuple or list of tuples, optional
        X-axis limits. A single tuple applies to all subplots.
        A list of tuples sets per-subplot limits.
    ylim : tuple or list of tuples, optional
        Y-axis limits. Same format as xlim.
    titles : list of str, optional
        Titles for each subplot (one per pair). If None, no titles are shown.
    linestyles : list of lists, optional
        Line styles for each subplot's [experimental, model] spectra.
        E.g., [['-', '-'], ['-', '--'], ...].
        If None, uses default ['-', '-'] for all subplots (solid lines for both).
    save : bool, optional
        Whether to save the figure. Default is False.
    filename : str, optional
        Filename for saving (without extension).
    format : str, optional
        File format for saving. Default is 'png'.
    diagonal : float, optional
        Slope for diagonal reference line. Default is None.
    return_fig : bool, optional
        Whether to return figure and axes. Default is False.
    **kwargs : dict
        Additional customization options (axisfontsize, tickfontsize,
        xaxislabel, yaxislabel, etc.)

    Returns
    -------
    fig, axes : tuple, optional
        Figure and axes array if return_fig=True.

    Example
    -------
    >>> data = read_nmr(['exp1.ppm', 'model1.ppm', 'exp2.ppm', 'model2.ppm'],
    ...                 provider='dmfit', tags=['1:1 exp', '1:1 model', '2:1 exp', '2:1 model'])
    >>> data.plot(grid='1x2', contour_start=1.5e5, xlim=(65, 52), ylim=(65, 52))
    """
    defaults = DEFAULTS.copy()
    _validate_kwargs(kwargs, defaults, "dmfit2d_grid")
    defaults.update(
        {k: v for k, v in kwargs.items() if k in defaults and v is not None}
    )

    spectra_list = spectra if isinstance(spectra, list) else [spectra]

    # Check all are 2D DMFit
    for s in spectra_list:
        if s["ndim"] != 2:
            raise ValueError("All spectra must be 2D for grid plotting")
        if s["metadata"]["provider_type"] != "dmfit":
            raise ValueError("All spectra must be from DMFit provider")

    if len(spectra_list) % 2 != 0:
        raise ValueError(
            "dmfit2d_grid expects pairs of spectra (experimental + model). "
            f"Got {len(spectra_list)} spectra, which is not divisible by 2."
        )

    num_pairs = len(spectra_list) // 2
    spectrum_pairs = [
        (spectra_list[i * 2], spectra_list[i * 2 + 1]) for i in range(num_pairs)
    ]

    rows, cols = subplot_dims

    # Exp and model colors
    if color is None:
        color = [["black", "red"] for _ in range(num_pairs)]
    elif isinstance(color, list) and len(color) > 0 and not isinstance(color[0], list):
        color = [color for _ in range(num_pairs)]

    # Project colors
    if proj_color is None:
        proj_color = color
    elif (
        isinstance(proj_color, list)
        and len(proj_color) > 0
        and not isinstance(proj_color[0], list)
    ):
        proj_color = [proj_color for _ in range(num_pairs)]

    if linestyles is None:
        linestyles = [["-", "-"] for _ in range(num_pairs)]
    elif (
        isinstance(linestyles, list)
        and len(linestyles) > 0
        and not isinstance(linestyles[0], list)
    ):
        linestyles = [linestyles for _ in range(num_pairs)]

    fig = plt.figure(figsize=(6 * cols, 6 * rows))

    gs = fig.add_gridspec(rows, cols, wspace=0.15, hspace=0.15)

    axes = []

    for idx, (exp_dict, model_dict) in enumerate(spectrum_pairs):
        if idx >= rows * cols:
            break

        row = idx // cols
        col = idx % cols

        gs_sub = gs[row, col].subgridspec(10, 10, wspace=0.01, hspace=0.01)

        ax_top = fig.add_subplot(gs_sub[0, 1:])
        ax_left = fig.add_subplot(gs_sub[1:, 0])
        ax_main = fig.add_subplot(gs_sub[1:, 1:], sharex=ax_top, sharey=ax_left)

        exp_color = color[idx][0] if idx < len(color) else "black"
        model_color = (
            color[idx][1] if idx < len(color) and len(color[idx]) > 1 else "red"
        )
        proj_exp_color = proj_color[idx][0] if idx < len(proj_color) else exp_color
        proj_model_color = (
            proj_color[idx][1]
            if idx < len(proj_color) and len(proj_color[idx]) > 1
            else model_color
        )

        exp_linestyle = linestyles[idx][0] if idx < len(linestyles) else "-"
        model_linestyle = (
            linestyles[idx][1]
            if idx < len(linestyles) and len(linestyles[idx]) > 1
            else "-"
        )

        contour_levels = contour_start * contour_factor ** np.arange(contour_num)

        # Experimental
        exp_data = exp_dict["data"]
        y_axis = exp_dict["ppm_scale"][0]
        x_axis = exp_dict["ppm_scale"][1]
        proj_f1_exp = exp_dict["projections"]["f1"]
        proj_f2_exp = exp_dict["projections"]["f2"]

        ax_main.contour(
            x_axis,
            y_axis,
            exp_data,
            levels=contour_levels,
            colors=exp_color,
            linewidths=defaults["linewidth_contour"],
            alpha=defaults["alpha"],
            linestyles=exp_linestyle,
        )

        # Model
        model_data = model_dict["data"]
        proj_f1_model = model_dict["projections"]["f1"]
        proj_f2_model = model_dict["projections"]["f2"]

        ax_main.contour(
            x_axis,
            y_axis,
            model_data,
            levels=contour_levels,
            colors=model_color,
            linewidths=defaults["linewidth_contour"],
            alpha=defaults["alpha"],
            linestyles=model_linestyle,
        )

        ax_top.plot(
            x_axis,
            proj_f2_exp,
            color=proj_exp_color,
            linewidth=defaults["linewidth_proj"],
            linestyle=exp_linestyle,
        )
        ax_top.plot(
            x_axis,
            proj_f2_model,
            color=proj_model_color,
            linewidth=defaults["linewidth_proj"],
            linestyle=model_linestyle,
        )

        ax_left.plot(
            -proj_f1_exp,
            y_axis,
            color=proj_exp_color,
            linewidth=defaults["linewidth_proj"],
            linestyle=exp_linestyle,
        )
        ax_left.plot(
            -proj_f1_model,
            y_axis,
            color=proj_model_color,
            linewidth=defaults["linewidth_proj"],
            linestyle=model_linestyle,
        )

        ax_top.axis("off")
        ax_left.axis("off")

        effective_xlim = _resolve_per_subplot(xlim, idx)
        if effective_xlim:
            ax_main.set_xlim(effective_xlim)

        effective_ylim = _resolve_per_subplot(ylim, idx)
        if effective_ylim:
            ax_main.set_ylim(effective_ylim)

        if diagonal is not None:
            xlim_eff = (
                effective_xlim if effective_xlim else (x_axis.max(), x_axis.min())
            )
            x_diag = np.linspace(xlim_eff[0], xlim_eff[1], 100)
            ax_main.plot(x_diag, diagonal * x_diag, "k--", lw=1)

        nuclei = exp_dict.get("nuclei", ["Unknown", "Unknown"])

        ax_main.set_xlabel(
            defaults["xaxislabel"] or _nucleus_label(nuclei[1]),
            fontsize=defaults["axisfontsize"],
            fontname=defaults["axisfont"],
        )
        ax_main.set_ylabel(
            defaults["yaxislabel"] or _nucleus_label(nuclei[0]),
            fontsize=defaults["axisfontsize"],
            fontname=defaults["axisfont"],
        )
        ax_main.yaxis.set_label_position("right")
        ax_main.yaxis.tick_right()

        if titles is not None and idx < len(titles):
            ax_top.set_title(
                titles[idx], fontsize=defaults["axisfontsize"], fontweight="bold", pad=5
            )

        # Tick params
        ax_main.tick_params(
            axis="both",
            labelsize=defaults["tickfontsize"],
            labelfontfamily=defaults["tickfont"],
        )
        _apply_tick_spacing(ax_main, defaults)

        axes.append({"main": ax_main, "top": ax_top, "left": ax_left})

    # Save or show
    _save_figure(fig, save, filename, format, "dmfit_2d_grid")
    return _handle_show_return(fig, (fig, axes), return_fig, save)
