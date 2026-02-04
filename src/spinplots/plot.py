from __future__ import annotations

import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from spinplots.utils import calculate_projections

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
    colors: list[str] | None = None,
    proj_colors=None,
    xlim=None,
    ylim=None,
    save=False,
    filename=None,
    format=None,
    diag=None,
    homo=False,
    return_fig=False,
    **kwargs,
):
    """
    Plots a 2D NMR spectrum from spectrum dictionaries.

    Parameters:
        spectra (dict or list): Dictionary or list of dictionaries containing spectrum data.
        contour_start (float, optional): Start value for the contour levels. Default is 1e5.
        contour_num (int, optional): Number of contour levels. Default is 10.
        contour_factor (float, optional): Factor by which the contour levels increase. Default is 1.2.

    Keyword arguments:
        cmap (str or list): Colormap(s) to use for the contour lines.
        colors (list): Colors to use when overlaying spectra.
        proj_colors (list): Colors to use for the projections.
        xlim (tuple): The limits for the x-axis.
        ylim (tuple): The limits for the y-axis.
        save (bool): Whether to save the plot.
        filename (str): The name of the file to save the plot.
        format (str): The format to save the file in.
        diag (float or None): Slope of the diagonal line/None.
        homo (bool): True if doing homonuclear experiment. When True, both axes will show the same nucleus.
        return_fig (bool): Whether to return the figure and axes.
        **kwargs: Additional keyword arguments for customizing the plot.

    Example:
        bruker2d(spectrum, 0.1, 10, 1.2, cmap='viridis', xlim=(0, 100), ylim=(0, 100), save=True, filename='2d_spectrum', format='png', diag=True)
    """

    spectra = spectra if isinstance(spectra, list) else [spectra]

    if not all(s["ndim"] == 2 for s in spectra):
        raise ValueError("All spectra must be 2-dimensional for bruker2d.")

    defaults = DEFAULTS.copy()
    defaults["yaxislabel"] = None
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

        if homo:
            nuclei_x = nuclei_list[1]
            nuclei_y = nuclei_list[1]
        else:
            nuclei_x = nuclei_list[1]
            nuclei_y = nuclei_list[0]

        number_x, nucleus_x = (
            "".join(filter(str.isdigit, nuclei_x)),
            "".join(filter(str.isalpha, nuclei_x)),
        )
        number_y, nucleus_y = (
            "".join(filter(str.isdigit, nuclei_y)),
            "".join(filter(str.isalpha, nuclei_y)),
        )
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
            and "x" in spectrum["projections"]
            and "y" in spectrum["projections"]
        ):
            if xlim is None and ylim is None:
                proj_x = spectrum["projections"]["x"]
                proj_y = spectrum["projections"]["y"]
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

            if proj_colors and i < len(proj_colors):
                proj_color = proj_colors[i]
            else:
                proj_color = cmap_i(
                    mcolors.Normalize(
                        vmin=contour_levels.min(), vmax=contour_levels.max()
                    )(contour_levels[0])
                )

            ax["a"].plot(
                x_proj_ppm,
                proj_x,
                linewidth=defaults["linewidth_proj"],
                color=proj_color,
            )
            ax["a"].axis(False)
            ax["b"].plot(
                -proj_y,
                y_proj_ppm,
                linewidth=defaults["linewidth_proj"],
                color=proj_color,
            )
            ax["b"].axis(False)
        elif cmap is not None and colors is not None:
            raise ValueError("Only one of cmap or colors can be provided.")
        elif colors is not None and cmap is None:
            contour_color = colors[i % len(colors)]
            ax["A"].contour(
                x_proj_ppm,
                y_proj_ppm,
                data[y_indices, x_indices],
                contour_levels,
                colors=contour_color,
                linewidths=defaults["linewidth_contour"],
            )

            if proj_colors and i < len(proj_colors):
                proj_color = proj_colors[i]
            else:
                proj_color = contour_color

            ax["a"].plot(
                x_proj_ppm,
                proj_x,
                linewidth=defaults["linewidth_proj"],
                color=proj_color,
            )
            ax["a"].axis(False)
            ax["b"].plot(
                -proj_y,
                y_proj_ppm,
                linewidth=defaults["linewidth_proj"],
                color=proj_color,
            )
            ax["b"].axis(False)

        else:
            proj_color = "black"
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
                color=proj_color,
            )
            ax["a"].axis(False)
            ax["b"].plot(
                -proj_y,
                y_proj_ppm,
                linewidth=defaults["linewidth_proj"],
                color=proj_color,
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

        xtick = defaults["xtickspacing"] or defaults["tickspacing"]
        if xtick:
            ax["A"].xaxis.set_major_locator(MultipleLocator(xtick))

        ytick = defaults["ytickspacing"] or defaults["tickspacing"]
        if ytick:
            ax["A"].yaxis.set_major_locator(MultipleLocator(ytick))

        if (
            homo
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

        if diag is not None:
            x_diag = np.linspace(
                xlim[0] if xlim else ppm_x_limits[0],
                xlim[1] if xlim else ppm_x_limits[1],
                100,
            )
            y_diag = diag * x_diag
            ax["A"].plot(x_diag, y_diag, linestyle="--", color="gray")

        if xlim:
            ax["A"].set_xlim(xlim)
            ax["a"].set_xlim(xlim)
        if ylim:
            ax["A"].set_ylim(ylim)
            ax["b"].set_ylim(ylim)

    if save:
        if filename and format:
            full_filename = f"{filename}.{format}"
        else:
            full_filename = f"2d_nmr_spectrum.{format if format else 'png'}"
        plt.savefig(full_filename, dpi=300, bbox_inches="tight", pad_inches=0.1)

    if return_fig:
        return ax

    plt.show()
    return None


def bruker2d_grid(
    spectra: dict | list[dict],
    subplot_dims=(1, 1),
    contour_start: float | None = None,
    contour_num: int = 10,
    contour_factor: float = 1.2,
    cmap: str | list[str] | None = None,
    colors: list[str] | None = None,
    proj_colors=None,
    xlim=None,
    ylim=None,
    titles=None,
    save=False,
    filename=None,
    format=None,
    diag=None,
    homo=False,
    return_fig=False,
    **kwargs,
):
    """
    Plots multiple 2D Bruker NMR spectra in a grid layout with projections.

    Parameters:
        spectra (dict or list): Dictionary or list of dictionaries containing spectrum data.
        subplot_dims (tuple): Grid dimensions as (rows, cols). Default is (1, 1).
        contour_start (float, optional): Start value for the contour levels.
        contour_num (int, optional): Number of contour levels. Default is 10.
        contour_factor (float, optional): Factor by which the contour levels increase. Default is 1.2.

    Keyword arguments:
        cmap (str or list): Colormap(s) to use for the contour lines.
        colors (list): Colors to use when overlaying spectra.
        proj_colors (list): Colors to use for the projections.
        xlim (tuple): The limits for the x-axis.
        ylim (tuple): The limits for the y-axis.
        titles (list): Titles for each subplot.
        save (bool): Whether to save the plot.
        filename (str): The name of the file to save the plot.
        format (str): The format to save the file in.
        diag (float or None): Slope of the diagonal line/None.
        homo (bool): True if doing homonuclear experiment.
        return_fig (bool): Whether to return the figure and axes.
        **kwargs: Additional keyword arguments for customizing the plot.

    Returns:
        None or tuple: If return_fig is True, returns the figure and axes array.

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

        if homo:
            nuclei_x = nuclei_list[1]
            nuclei_y = nuclei_list[1]
        else:
            nuclei_x = nuclei_list[1]
            nuclei_y = nuclei_list[0]

        number_x, nucleus_x = (
            "".join(filter(str.isdigit, nuclei_x)),
            "".join(filter(str.isalpha, nuclei_x)),
        )
        number_y, nucleus_y = (
            "".join(filter(str.isdigit, nuclei_y)),
            "".join(filter(str.isalpha, nuclei_y)),
        )

        ppm_x = spectrum["ppm_scale"][1]
        ppm_y = spectrum["ppm_scale"][0]

        # Handle xlim and ylim for data slicing
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

        # Calculate or retrieve projections
        if (
            isinstance(spectrum["projections"], dict)
            and "x" in spectrum["projections"]
            and "y" in spectrum["projections"]
        ):
            if xlim is None and ylim is None:
                proj_x = spectrum["projections"]["x"]
                proj_y = spectrum["projections"]["y"]
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
            if proj_colors and idx < len(proj_colors):
                proj_color = proj_colors[idx]
            else:
                proj_color = cmap_i(
                    mcolors.Normalize(
                        vmin=contour_levels.min(), vmax=contour_levels.max()
                    )(contour_levels[0])
                )
        elif colors is not None:
            contour_color = colors[idx % len(colors)]
            ax_main.contour(
                x_proj_ppm,
                y_proj_ppm,
                data[y_indices, x_indices],
                contour_levels,
                colors=contour_color,
                linewidths=defaults["linewidth_contour"],
            )
            proj_color = (
                proj_colors[idx]
                if proj_colors and idx < len(proj_colors)
                else contour_color
            )
        else:
            proj_color = "black"
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
            color=proj_color,
        )
        ax_top.axis(False)

        ax_left.plot(
            -proj_y,
            y_proj_ppm,
            linewidth=defaults["linewidth_proj"],
            color=proj_color,
        )
        ax_left.axis(False)

        # Set labels
        if xaxislabel := defaults.get("xaxislabel"):
            x_label = xaxislabel
        else:
            x_label = f"$^{{{number_x}}}\\mathrm{{{nucleus_x}}}$ (ppm)"

        if yaxislabel := defaults.get("yaxislabel"):
            y_label = yaxislabel
        elif homo and number_y == number_x and nucleus_y == nucleus_x:
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

        # Apply tick spacing
        xtick = defaults["xtickspacing"] or defaults["tickspacing"]
        if xtick:
            ax_main.xaxis.set_major_locator(MultipleLocator(xtick))

        ytick = defaults["ytickspacing"] or defaults["tickspacing"]
        if ytick:
            ax_main.yaxis.set_major_locator(MultipleLocator(ytick))

        # Diagonal line
        if diag is not None:
            xlim_eff = xlim if xlim else (ppm_x[0], ppm_x[-1])
            x_diag = np.linspace(xlim_eff[0], xlim_eff[1], 100)
            ax_main.plot(x_diag, diag * x_diag, "k--", lw=1)

        # Set axis limits
        if xlim:
            ax_main.set_xlim(xlim)
        if ylim:
            ax_main.set_ylim(ylim)

        # Add title if provided
        if titles is not None and idx < len(titles):
            ax_top.set_title(
                titles[idx], fontsize=defaults["axisfontsize"], fontweight="bold", pad=5
            )

        axes.append({"main": ax_main, "top": ax_top, "left": ax_left})

    # Save or show
    if save:
        full_filename = f"{filename if filename else 'bruker_2d_grid'}.{format if format else 'png'}"
        fig.savefig(full_filename, dpi=300, bbox_inches="tight", pad_inches=0.1)

    if return_fig:
        return fig, axes

    if not save:
        plt.show()

    return None


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

    Parameters:
        spectra (dict or list): Dictionary or list of dictionaries containing spectrum data.
        labels (list, optional): List of labels for the spectra.
        xlim (tuple, optional): The limits for the x-axis.
        ylim (tuple, optional): The limits for the y-axis.
        save (bool, optional): Whether to save the plot.
        filename (str, optional): The name of the file to save the plot.
        format (str, optional): The format to save the file in.
        frame (bool, optional): Whether to show the frame.
        normalize (str, optional): Normalization method ('max', 'scans', or None).
        stacked (bool, optional): Whether to stack the spectra.
        color (list, optional): List of colors for the spectra.
        return_fig (bool, optional): Whether to return the figure and axes.
        **kwargs: Additional keyword arguments for customizing the plot.

    Returns:
        None or tuple: If return_fig is True, returns the figure and axes.
    """

    spectra = spectra if isinstance(spectra, list) else [spectra]

    if not all(s["ndim"] == 1 for s in spectra):
        raise ValueError("All spectra must be 1-dimensional for bruker1d.")

    defaults = DEFAULTS.copy()
    defaults["yaxislabel"] = None
    defaults.update(
        {k: v for k, v in kwargs.items() if k in defaults and v is not None}
    )

    fig, ax = plt.subplots()

    current_stack_offset = 0.0

    first_nuclei = spectra[0]["nuclei"]
    number, nucleus = (
        "".join(filter(str.isdigit, first_nuclei)),
        "".join(filter(str.isalpha, first_nuclei)),
    )

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

    if save:
        if not filename or not format:
            raise ValueError("Both filename and format must be provided if save=True.")
        full_filename = f"{filename}.{format}"
        fig.savefig(
            full_filename, format=format, dpi=300, bbox_inches="tight", pad_inches=0.1
        )
        plt.show()
        return None

    if return_fig:
        return fig, ax

    plt.show()
    return None


def bruker1d_grid(
    spectra: dict | list[dict],
    labels=None,
    subplot_dims=(1, 1),
    xlim=None,
    ylim=None,
    save=False,
    filename=None,
    format="png",
    frame=False,
    normalize=False,
    color=None,
    return_fig=False,
    **kwargs,
):
    """
    Plots 1D NMR spectra from Bruker data in subplots.

    Parameters:
        spectra (dict or list): Dictionary or list of dictionaries containing spectrum data.
        labels (list): List of labels for the spectra.
        subplot_dims (tuple): Dimensions of the subplot grid (rows, cols).
        xlim (list of tuples or tuple): The limits for the x-axis.
        ylim (list of tuples or tuple): The limits for the y-axis.
        save (bool): Whether to save the plot.
        filename (str): The name of the file to save the plot.
        format (str): The format to save the file in.
        frame (bool): Whether to show the frame.
        normalize (str): Normalization method 'max', 'scans', or None.
        color (str): List of colors for the spectra.
        return_fig (bool): Whether to return the figure and axis.
        **kwargs: Additional keyword arguments for customizing the plot.

    Returns:
        None or tuple: If return_fig is True, returns the figure and axis.

    Example:
        bruker1d_grid([spectrum1, spectrum2], labels=['Spectrum 1', 'Spectrum 2'], subplot_dims=(1, 2), xlim=[(0, 100), (0, 100)], save=True, filename='1d_spectra', format='png', frame=False, normalize='max', color=['red', 'blue'])
    """

    spectra = spectra if isinstance(spectra, list) else [spectra]

    if not all(s["ndim"] == 1 for s in spectra):
        raise ValueError("All spectra must be 1-dimensional for bruker1d_grid.")

    defaults = DEFAULTS.copy()
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
        number, nucleus = (
            "".join(filter(str.isdigit, nuclei)),
            "".join(filter(str.isalpha, nuclei)),
        )

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

        if xlim and isinstance(xlim, tuple):
            ax.set_xlim(xlim)
        elif xlim and isinstance(xlim, list) and i < len(xlim):
            ax.set_xlim(xlim[i])

        if ylim and isinstance(ylim, tuple):
            ax.set_ylim(ylim)
        elif ylim and isinstance(ylim, list) and i < len(ylim):
            ax.set_ylim(ylim[i])

    plt.tight_layout()

    if save:
        if filename:
            full_filename = f"{filename}.{format}"
        else:
            full_filename = f"1d_nmr_spectra.{format}"
        fig.savefig(
            full_filename, format=format, dpi=300, bbox_inches="tight", pad_inches=0.1
        )
        return None
    elif return_fig:
        return fig, axes

    plt.show()
    return None


# Plot 2D NMR data from CSV or DataFrame
def df2d(
    path,
    contour_start,
    contour_num,
    contour_factor,
    cmap=None,
    xlim=None,
    ylim=None,
    save=False,
    filename=None,
    format=None,
    return_fig=False,
):
    """
    Plot 2D NMR data from a CSV file or a DataFrame.

    Parameters:
    path (str): Path to the CSV file.
    contour_start (float): Contour start value.
    contour_num (int): Number of contour levels.
    contour_factor (float): Contour factor.

    Keyword arguments:
        cmap (str): The colormap to use for the contour lines.
        xlim (tuple): The limits for the x-axis.
        ylim (tuple): The limits for the y-axis.
        save (bool): Whether to save the plot.
        filename (str): The name of the file to save the plot.
        format (str): The format to save the file in.
        return_fig (bool): Whether to return the figure and axis.

    Example:
    df2d('nmr_data.csv', contour_start=4e3, contour_num=10, contour_factor=1.2, cmap='viridis', xlim=(0, 100), ylim=(0, 100), save=True, filename='2d_spectrum', format='png')
    """

    # Check if path to CSV or DataFrame
    df_nmr = path if isinstance(path, pd.DataFrame) else pd.read_csv(path)

    cols = df_nmr.columns
    f1_nuclei, f1_units = cols[0].split()
    number_x, nucleus_x = (
        "".join(filter(str.isdigit, f1_nuclei)),
        "".join(filter(str.isalpha, f1_nuclei)),
    )
    f2_nuclei, f2_units = cols[1].split()
    number_y, nucleus_y = (
        "".join(filter(str.isdigit, f2_nuclei)),
        "".join(filter(str.isalpha, f2_nuclei)),
    )
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

    if save:
        if filename:
            full_filename = filename + "." + (format if format else "png")
        else:
            full_filename = "2d_nmr_spectrum." + (format if format else "png")
        plt.savefig(
            full_filename,
            format=format if format else "png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        return None
    elif return_fig:
        return ax
    else:
        plt.show()
        return None


# Functions for DMFit
def dmfit1d(
    spin_objects,
    color="b",
    linewidth=1,
    linestyle="-",
    alpha=1,
    model_show=True,
    model_color="red",
    model_linewidth=1,
    model_linestyle="--",
    model_alpha=1,
    deconv_show=True,
    deconv_color=None,
    deconv_alpha=0.3,
    frame=False,
    labels=None,
    labelsize=12,
    xlim=None,
    save=False,
    format=None,
    filename=None,
    yaxislabel=None,
    xaxislabel=None,
    axisfontsize=None,
    axisfont=None,
    tickfontsize=None,
    tickfont=None,
    tickspacing=None,
    return_fig=False,
):
    """
    Read a dmfit1d file and return a DataFrame with the data.

    Parameters
    ----------
    spin_objects : Spin
        The Spin object containing the dmfit1d file.
    color : str, optional
        The color of the spectrum line. The default is 'b'.
    linewidth : int, optional
        The width of the spectrum line. The default is 1.
    linestyle : str, optional
        The style of the spectrum line. The default is '-'.
    alpha : float, optional
        The transparency of the spectrum line. The default is 1.
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
        The labels for the x and y axes. The default is name of columns.
    labelsize : int, optional
        The size of the labels. The default is 12.
    xlim : tuple, optional
        The limits for the x axis. The default is None.
    save : bool, optional
        Whether to save the figure. The default is False.
    format : str, optional
        The format to save the figure. The default is None.
    filename : str, optional
        The name of the file to save the figure. The default is None.
    yaxislabel : str, optional
        The label for the y axis. The default is None.
    xaxislabel : str, optional
        The label for the x axis. The default is None.
    axisfontsize : int, optional
        The size of the axis labels. The default is None.
    axisfont : str, optional
        The font of the axis labels. The default is None.
    tickfontsize : int, optional
        The size of the tick labels. The default is None.
    tickfont : str, optional
        The font of the tick labels. The default is None.
    tickspacing : int, optional
        The spacing of the ticks. The default is None.
    return_fig : bool, optional
        Whether to return the figure. The default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    dmfit_df : pandas.DataFrame
        The DataFrame with the data from the dmfit1d file.

    """

    if not spin_objects.spectrum:
        raise ValueError("Spin object contains no spectra.")

    spectrum_info = spin_objects.spectrum
    dmfit_df = spectrum_info.get("dmfit_dataframe")

    if dmfit_df is None:
        raise ValueError(
            "DMfit DataFrame not found in Spin object. Read data with provider='dmfit'"
        )

    n_lines = sum(col.startswith("Line#") for col in dmfit_df.columns)

    defaults = {
        "color": color,
        "linewidth": linewidth,
        "linestyle": linestyle,
        "alpha": alpha,
        "model_show": model_show,
        "model_color": model_color,
        "model_linewidth": model_linewidth,
        "model_linestyle": model_linestyle,
        "model_alpha": model_alpha,
        "deconv_show": deconv_show,
        "deconv_color": deconv_color,
        "deconv_alpha": deconv_alpha,
        "frame": frame,
        "labels": labels,
        "labelsize": labelsize,
        "xlim": xlim,
        "save": save,
        "format": format,
        "filename": filename,
        "yaxislabel": yaxislabel,
        "xaxislabel": xaxislabel,
        "axisfontsize": axisfontsize,
        "axisfont": axisfont,
        "tickfontsize": tickfontsize,
        "tickfont": tickfont,
        "tickspacing": tickspacing,
        "return_fig": return_fig,
    }

    params = {k: v for k, v in locals().items() if k in defaults and v is not None}
    params.update(defaults)

    fig, ax = plt.subplots()
    ax.plot(
        dmfit_df["ppm"],
        dmfit_df["Spectrum"],
        color=params["color"],
        linewidth=params["linewidth"],
        linestyle=params["linestyle"],
        alpha=params["alpha"],
        label=params["labels"][0]
        if params["labels"] and len(params["labels"]) > 0
        else None,
    )
    if params["model_show"]:
        ax.plot(
            dmfit_df["ppm"],
            dmfit_df["Model"],
            color=params["model_color"],
            linewidth=params["model_linewidth"],
            linestyle=params["model_linestyle"],
            alpha=params["model_alpha"],
            label=params["labels"][1]
            if params["labels"] and len(params["labels"]) > 1
            else None,
        )
    if params["deconv_show"]:
        for i in range(1, n_lines + 1):
            if params["deconv_color"] is not None:
                ax.fill_between(
                    dmfit_df["ppm"],
                    dmfit_df[f"Line#{i}"],
                    alpha=params["deconv_alpha"],
                    color=params["deconv_color"][i-1],
                )
            else:
                ax.fill_between(
                    dmfit_df["ppm"], dmfit_df[f"Line#{i}"], alpha=params["deconv_alpha"]
                )

    if params["labels"]:
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=defaults["labelsize"],
            prop={"family": defaults["tickfont"], "size": defaults["labelsize"]},
        )
    if params["xlim"]:
        ax.set_xlim(params["xlim"])
    if params["yaxislabel"]:
        ax.set_ylabel(params["yaxislabel"], fontsize=params["labelsize"])
    if params["xaxislabel"]:
        ax.set_xlabel(params["xaxislabel"], fontsize=params["labelsize"])
    if params["axisfontsize"]:
        ax.xaxis.label.set_fontsize(params["axisfontsize"])
        ax.yaxis.label.set_fontsize(params["axisfontsize"])
    if params["axisfont"]:
        ax.xaxis.label.set_fontname(params["axisfont"])
        ax.yaxis.label.set_fontname(params["axisfont"])
    if params["tickfontsize"]:
        ax.tick_params(axis="both", which="major", labelsize=params["tickfontsize"])
        ax.tick_params(axis="both", which="minor", labelsize=params["tickfontsize"])
    if params["tickfont"]:
        ax.tick_params(axis="both", which="major", labelfont=params["tickfont"])
        ax.tick_params(axis="both", which="minor", labelfont=params["tickfont"])

    # Apply x-axis tick spacing
    xtick = params.get("xtickspacing") or params.get("tickspacing")
    if xtick:
        ax.xaxis.set_major_locator(MultipleLocator(xtick))

    # Apply y-axis tick spacing
    ytick = params.get("ytickspacing") or params.get("tickspacing")
    if ytick:
        ax.yaxis.set_major_locator(MultipleLocator(ytick))
    if params["frame"]:
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.spines["left"].set_visible(True)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
    if params["save"]:
        if params["format"]:
            plt.savefig(
                f"{params['filename']}.{params['format']}", format=params["format"]
            )
        else:
            plt.savefig(params["filename"])

    if params["return_fig"]:
        return fig, ax
    else:
        plt.show()
        return None


def dmfit2d(
    spin_objects,
    contour_start=1e5,
    contour_num=10,
    contour_factor=1.2,
    colors=None,
    proj_colors=None,
    xlim=None,
    ylim=None,
    labels=None,
    save=False,
    filename=None,
    format=None,
    axis_right=True,
    diag=None,
    return_fig=False,
    **kwargs,
):
    """
    Plot 2D DMFit data with 1D projections.

    Parameters
    ----------
    spin_objects : Spin or SpinCollection
        The Spin object or SpinCollection containing DMFit 2D data.
    contour_start : float, optional
        The starting contour level. Default is 1e5.
    contour_num : int, optional
        The number of contour levels. Default is 10.
    contour_factor : float, optional
        The factor by which the contour levels increase. Default is 1.2.
    colors : str or list, optional
        Color(s) for each spectrum's contours.
    proj_colors : str or list, optional
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
    diag : float or None, optional
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
    defaults.update(
        {k: v for k, v in kwargs.items() if k in defaults and v is not None}
    )

    if isinstance(spin_objects, list):
        if spin_objects and isinstance(spin_objects[0], dict):
            spectra_dicts = spin_objects
            if labels is None:
                plot_labels = [
                    f"Spectrum {idx + 1}" for idx in range(len(spin_objects))
                ]
            else:
                plot_labels = labels
        else:
            raise ValueError(
                "Unexpected list of Spin objects. Use SpinCollection instead."
            )
    elif hasattr(spin_objects, "spins"):
        spectra_dicts = [spin_obj.spectrum for spin_obj in spin_objects.spins.values()]
        if labels is None:
            plot_labels = [
                spin_obj.tag if spin_obj.tag else f"Spectrum {idx + 1}"
                for idx, spin_obj in enumerate(spin_objects.spins.values())
            ]
        else:
            plot_labels = labels
    else:
        spectra_dicts = [spin_objects.spectrum]
        if labels is None:
            plot_labels = [spin_objects.tag if spin_objects.tag else "Spectrum"]
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
    if isinstance(colors, str):
        contour_colors_list = [colors] * num_spectra
    elif isinstance(colors, list):
        contour_colors_list = [colors[i % len(colors)] for i in range(num_spectra)]
    else:
        contour_colors_list = [
            default_colors[i % len(default_colors)] for i in range(num_spectra)
        ]

    projection_colors_list = []
    if isinstance(proj_colors, str):
        projection_colors_list = [proj_colors] * num_spectra
    elif isinstance(proj_colors, list):
        projection_colors_list = [
            proj_colors[i % len(proj_colors)] for i in range(num_spectra)
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

    num_f2, nuc_f2 = (
        "".join(filter(str.isdigit, f2_nuc_str)),
        "".join(filter(str.isalpha, f2_nuc_str)),
    )
    num_f1, nuc_f1 = (
        "".join(filter(str.isdigit, f1_nuc_str)),
        "".join(filter(str.isalpha, f1_nuc_str)),
    )

    final_xaxislabel = (
        defaults.get("xaxislabel")
        if defaults.get("xaxislabel")
        else f"$^{{{num_f2}}}${nuc_f2} (ppm)"
    )
    final_yaxislabel = (
        defaults.get("yaxislabel")
        if defaults.get("yaxislabel")
        else f"$^{{{num_f1}}}${nuc_f1} (ppm)"
    )

    assert final_xaxislabel is not None
    assert final_yaxislabel is not None

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

    if diag is not None:
        diag_xlim_eff = main_ax.get_xlim()
        x_diag_vals = np.linspace(diag_xlim_eff[0], diag_xlim_eff[1], 100)
        main_ax.plot(x_diag_vals, diag * x_diag_vals, "k--", lw=1)

    if legend_elements:
        main_ax.legend(
            handles=legend_elements,
            fontsize=defaults["labelsize"],
            prop={"family": defaults["tickfont"]},
        )

    plt.tight_layout(pad=0.5)

    # --- Save/Show ---
    if save:
        if filename and format:
            full_filename = f"{filename}.{format}"
        elif filename:
            full_filename = f"{filename}.png"
        else:
            full_filename = f"dmfit_2d_projections.{format if format else 'png'}"
        fig.savefig(full_filename, dpi=300, bbox_inches="tight", pad_inches=0.1)

    if return_fig:
        return ax_dict

    if not save:
        plt.show()

    return None


def dmfit1d_grid(
    spectra: dict | list[dict],
    subplot_dims=(1, 1),
    labels=None,
    xlim=None,
    ylim=None,
    color=None,
    model_color=None,
    deconv_color=None,
    save=False,
    filename=None,
    format="png",
    return_fig=False,
    **kwargs,
):
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
    labels : list of str, optional
        Labels for each subplot. If None, no labels are shown.
    xlim : tuple, optional
        X-axis limits for all subplots.
    ylim : tuple, optional
        Y-axis limits for all subplots.
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
                    ax.fill_between(
                        ppm,
                        dmfit_df[f"Line#{j}"],
                        alpha=0.3,
                        color=deconv_color[j-1],
                    )
                else:
                    ax.fill_between(ppm, dmfit_df[f"Line#{j}"], alpha=0.3)

        if labels and i < len(labels):
            ax.set_title(labels[i], fontsize=defaults["axisfontsize"])

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
                number, nucleus = (
                    "".join(filter(str.isdigit, nuclei)),
                    "".join(filter(str.isalpha, nuclei)),
                )
                ax.set_xlabel(
                    f"$^{{{number}}}\\mathrm{{{nucleus}}}$ (ppm)",
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

        # Apply x-axis tick spacing
        xtick = defaults["xtickspacing"] or defaults["tickspacing"]
        if xtick:
            ax.xaxis.set_major_locator(MultipleLocator(xtick))

        # Y-axis (no frame by default)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_yticklabels([])
        ax.set_yticks([])

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
    plt.tight_layout()

    if save:
        if filename:
            full_filename = f"{filename}.{format}"
        else:
            full_filename = f"1d_dmfit_spectra.{format}"
        fig.savefig(
            full_filename, format=format, dpi=300, bbox_inches="tight", pad_inches=0.1
        )
        return None
    elif return_fig:
        return fig, axes

    plt.show()
    return None


def dmfit2d_grid(
    spin_objects,
    subplot_dims=(1, 3),
    contour_start=1e5,
    contour_num=10,
    contour_factor=1.2,
    colors=None,
    proj_colors=None,
    xlim=None,
    ylim=None,
    titles=None,
    linestyles=None,
    save=False,
    filename=None,
    format="png",
    diag=None,
    return_fig=False,
    **kwargs,
):
    """
    Plot multiple 2D DMFit spectra in a grid layout with projections.
    Each subplot shows an experimental spectrum overlaid with its fit/model.
    This function expects pairs of spectra (experimental + model) in the SpinCollection.

    Parameters
    ----------
    spin_objects : SpinCollection
        Collection of Spin objects containing DMFit 2D data.
        Should contain pairs: [exp1, model1, exp2, model2, ...].
    subplot_dims : tuple, optional
        Grid dimensions as (rows, cols). Default is (1, 3).
    contour_start : float, optional
        Starting contour level. Default is 1e5.
    contour_num : int, optional
        Number of contour levels. Default is 10.
    contour_factor : float, optional
        Factor by which contour levels increase. Default is 1.2.
    colors : list of lists, optional
        Colors for each subplot's [experimental, model] spectra.
        E.g., [['black', 'red'], ['black', 'red'], ...].
        If None, uses default ['black', 'red'] for all subplots.
    proj_colors : list of lists, optional
        Colors for projections. Same structure as colors.
    xlim : tuple, optional
        X-axis limits for all subplots.
    ylim : tuple, optional
        Y-axis limits for all subplots.
    titles : list of str, optional
        Titles for each subplot (one per pair). If None, no titles are shown.
    linestyles : list of lists, optional
        Line styles for each subplot's [experimental, model] spectra.
        E.g., [['-', '-'], ['-', '--'], ...].
        If None, uses default ['-', '-'] for all subplots (solid lines for both).
    xaxislabel : str, optional
        Label for x-axis. Default is None (auto-generated from nucleus).
    yaxislabel : str, optional
        Label for y-axis. Default is None (auto-generated from nucleus).
    save : bool, optional
        Whether to save the figure. Default is False.
    filename : str, optional
        Filename for saving (without extension).
    format : str, optional
        File format for saving. Default is 'png'.
    diag : float, optional
        Slope for diagonal reference line. Default is None.
    return_fig : bool, optional
        Whether to return figure and axes. Default is False.
    **kwargs : dict
        Additional customization options (axisfontsize, tickfontsize, etc.)

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
    defaults.update(
        {k: v for k, v in kwargs.items() if k in defaults and v is not None}
    )

    if hasattr(spin_objects, "spins"):
        spectra_list = list(spin_objects.spins.values())
    else:
        raise ValueError("dmfit2d_grid requires a SpinCollection object")

    # Check all are 2D DMFit
    for spin in spectra_list:
        if spin.spectrum["ndim"] != 2:
            raise ValueError("All spectra must be 2D for grid plotting")
        if spin.spectrum["metadata"]["provider_type"] != "dmfit":
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
    if colors is None:
        colors = [["black", "red"] for _ in range(num_pairs)]
    elif (
        isinstance(colors, list) and len(colors) > 0 and not isinstance(colors[0], list)
    ):
        colors = [colors for _ in range(num_pairs)]

    # Project colors
    if proj_colors is None:
        proj_colors = colors
    elif (
        isinstance(proj_colors, list)
        and len(proj_colors) > 0
        and not isinstance(proj_colors[0], list)
    ):
        proj_colors = [proj_colors for _ in range(num_pairs)]

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

    for idx, (spin_exp, spin_model) in enumerate(spectrum_pairs):
        if idx >= rows * cols:
            break

        row = idx // cols
        col = idx % cols

        gs_sub = gs[row, col].subgridspec(10, 10, wspace=0.01, hspace=0.01)

        ax_top = fig.add_subplot(gs_sub[0, 1:])
        ax_left = fig.add_subplot(gs_sub[1:, 0])
        ax_main = fig.add_subplot(gs_sub[1:, 1:], sharex=ax_top, sharey=ax_left)

        exp_color = colors[idx][0] if idx < len(colors) else "black"
        model_color = (
            colors[idx][1] if idx < len(colors) and len(colors[idx]) > 1 else "red"
        )
        proj_exp_color = proj_colors[idx][0] if idx < len(proj_colors) else exp_color
        proj_model_color = (
            proj_colors[idx][1]
            if idx < len(proj_colors) and len(proj_colors[idx]) > 1
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
        exp_data = spin_exp.spectrum["data"]
        y_axis = spin_exp.spectrum["ppm_scale"][0]
        x_axis = spin_exp.spectrum["ppm_scale"][1]
        proj_f1_exp = spin_exp.spectrum["projections"]["f1"]
        proj_f2_exp = spin_exp.spectrum["projections"]["f2"]

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
        model_data = spin_model.spectrum["data"]
        proj_f1_model = spin_model.spectrum["projections"]["f1"]
        proj_f2_model = spin_model.spectrum["projections"]["f2"]

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

        if xlim:
            ax_main.set_xlim(xlim)
        if ylim:
            ax_main.set_ylim(ylim)

        if diag is not None:
            xlim_eff = xlim if xlim else (x_axis.max(), x_axis.min())
            x_diag = np.linspace(xlim_eff[0], xlim_eff[1], 100)
            ax_main.plot(x_diag, diag * x_diag, "k--", lw=1)

        nuclei = spin_exp.spectrum.get("nuclei", ["Unknown", "Unknown"])

        f2_str = str(nuclei[1])
        num_f2, nuc_f2 = (
            "".join(filter(str.isdigit, f2_str)),
            "".join(filter(str.isalpha, f2_str)),
        )

        ax_main.set_xlabel(
            defaults["xaxislabel"]
            if defaults["xaxislabel"]
            else f"$^{{{num_f2}}}${nuc_f2} (ppm)",
            fontsize=defaults["axisfontsize"],
            fontname=defaults["axisfont"],
        )

        f1_str = str(nuclei[0])
        num_f1, nuc_f1 = (
            "".join(filter(str.isdigit, f1_str)),
            "".join(filter(str.isalpha, f1_str)),
        )

        ax_main.set_ylabel(
            defaults["yaxislabel"]
            if defaults["yaxislabel"]
            else f"$^{{{num_f1}}}${nuc_f1} (ppm)",
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

        axes.append({"main": ax_main, "top": ax_top, "left": ax_left})

    # Save or show
    if save:
        full_filename = f"{filename if filename else 'dmfit_2d_grid'}.{format}"
        fig.savefig(full_filename, dpi=300, bbox_inches="tight", pad_inches=0.1)

    if return_fig:
        return fig, axes

    if not save:
        plt.show()

    return None
