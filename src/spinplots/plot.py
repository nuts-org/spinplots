from __future__ import annotations

import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nmrglue as ng
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

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
    "xaxislabel": None,
    "yaxislabel": None,
    "axisfont": None,
    "tickfont": None,
    "tickspacing": None,
}

def bruker2d(
    spectra: dict | list[dict],
    contour_start: float,
    contour_num: int,
    contour_factor: float,
    cmap: str | list[str] | None = None,
    colors: list[str] | None = None,
    proj_colors = None,
    xlim=None,
    ylim=None,
    save=False,
    filename=None,
    format=None,
    diag=None,
    homo=False,
    return_fig=False,
    **kwargs
):
    """
    Plots a 2D NMR spectrum from Bruker data.

    Parameters:
        data_path (str or list): Path or list of paths to the Bruker data directories.
        contour_start (float or list): Start value for the contour levels.
        contour_num (int or list): Number of list of contour levels.
        contour_factor (float or list): Factor or list of factors by which the contour levels increase.

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
        homo (bool): True if doing homonuclear experiment.
        return_fig (bool): Whether to return the figure and axis.
        linewidth_contour (float): Line width of the contour plot.
        linewidth_proj (float): Line width of the projections.
        xaxislabel (str): Label for the axis.
        yaxislabel (str): Label for the y-axis.
        axisfont (str): Font type for the axis label.
        axisfontsize (int): Font size for the axis label.
        tickfont (str): Font type for the tick labels.
        tickfontsize (int): Font size for the tick labels.
        tickspacing (int): Spacing between the tick labels.

    Example:
        bruker2d('data/2d_data', 0.1, 10, 1.2, cmap='viridis', xlim=(0, 100), ylim=(0, 100), save=True, filename='2d_spectrum', format='png', diag=True)
    """

    spectra = spectra if isinstance(spectra, list) else [spectra]

    if not all([s["ndim"] == 2 for s in spectra]):
        raise ValueError("All spectra must be 2-dimensional for bruker2d.")

    defaults = DEFAULTS.copy()
    defaults["yaxislabel"] = None
    defaults.update({k: v for k, v in kwargs if k in defaults and v is not None})

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

    for i, nmr in enumerate([s["path"] for s in spectra]):
        dic, data = ng.bruker.read_pdata(nmr)
        udic = ng.bruker.guess_udic(dic, data)

        # Check if homo is set to True
        if homo:
            nuclei_x = udic[1]["label"]
            nuclei_y = udic[1]["label"]
        else:
            nuclei_x = udic[1]["label"]
            nuclei_y = udic[0]["label"]

        # Extract the number and nucleus symbol from the label
        number_x, nucleus_x = (
            "".join(filter(str.isdigit, nuclei_x)),
            "".join(filter(str.isalpha, nuclei_x)),
        )
        number_y, nucleus_y = (
            "".join(filter(str.isdigit, nuclei_y)),
            "".join(filter(str.isalpha, nuclei_y)),
        )

        uc_x = ng.fileiobase.uc_from_udic(udic, dim=1)
        ppm_x = uc_x.ppm_scale()
        ppm_x_limits = uc_x.ppm_limits()

        uc_y = ng.fileiobase.uc_from_udic(udic, dim=0)
        ppm_y = uc_y.ppm_scale()
        ppm_y_limits = uc_y.ppm_limits()

        # Get indices for the zoomed region if limits are specified
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

        # Calculate projections based on the zoomed region
        zoomed_data = data[y_indices, x_indices]
        proj_x = np.amax(zoomed_data, axis=0)
        proj_y = np.amax(zoomed_data, axis=1)

        # Contour levels
        contour_levels = contour_start * contour_factor ** np.arange(contour_num)

        # Plot projections with the extracted color
        # using the relevant portions of x and y ranges
        x_proj_ppm = ppm_x[x_indices]
        y_proj_ppm = ppm_y[y_indices]

        # Plot contour lines with the provided colormap if cmap is provided
        if cmap is not None:
            from matplotlib.colors import LogNorm

            if isinstance(cmap, str):
                cmap = [cmap]

            if len(cmap) > 1:
                warnings.warn(
                    "Warning: Consider using colors instead of cmap"
                    "when overlapping spectra."
                )

            cmap_i = plt.get_cmap(cmap[i])
            contour_plot = ax["A"].contour(
                data,
                contour_levels,
                extent=(
                    ppm_x_limits[0],
                    ppm_x_limits[1],
                    ppm_y_limits[0],
                    ppm_y_limits[1],
                ),
                cmap=cmap[i],
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
            # Error. Only one of cmap or colors can be provided.
            raise ValueError("Only one of cmap or colors can be provided.")
        elif colors is not None and cmap is None:
            contour_color = colors[i]
            contour_plot = ax["A"].contour(
                data,
                contour_levels,
                extent=(
                    ppm_x_limits[0],
                    ppm_x_limits[1],
                    ppm_y_limits[0],
                    ppm_y_limits[1],
                ),
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
            contour_plot = ax["A"].contour(
                data,
                contour_levels,
                extent=(
                    ppm_x_limits[0],
                    ppm_x_limits[1],
                    ppm_y_limits[0],
                    ppm_y_limits[1],
                ),
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

        if xaxislabel := defaults["xaxislabel"]:
            defaults["xaxislabel"] = xaxislabel
        else:
            defaults["xaxislabel"] = f"$^{{{number_x}}}\\mathrm{{{nucleus_x}}}$ (ppm)"
        if yaxislabel := defaults["yaxislabel"]:
            defaults["yaxislabel"] = yaxislabel
        else:
            defaults["yaxislabel"] = f"$^{{{number_y}}}\\mathrm{{{nucleus_y}}}$ (ppm)"

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

        # Plot diagonal line if diag is provided
        if diag is not None:
            x_diag = np.linspace(ppm_x_limits[0], ppm_x_limits[1], 100)
            y_diag = diag * x_diag
            ax["A"].plot(x_diag, y_diag, linestyle="--", color="gray")

        # Set axis limits if provided
        if xlim:
            ax["A"].set_xlim(xlim)
            ax["a"].set_xlim(xlim)
        if ylim:
            ax["A"].set_ylim(ylim)
            ax["b"].set_ylim(ylim)

    # Show the plot or save it
    if save:
        if filename:
            full_filename = f"{filename}.{format}"
        else:
            full_filename = f"2d_nmr_spectrum.{format}"
        plt.savefig(
            full_filename, format=format, dpi=300, bbox_inches="tight", pad_inches=0.1
        )

    if return_fig:
        return ax
    
    plt.show()
    return None


# Function to easily plot 1D NMR spectra in Bruker's format
def bruker1d(
    spectra: dict | list[dict],  # Keep this as list for now based on previous fix
    labels: list[str] | None = None,
    xlim: tuple[float, float] | None = None,
    save: bool = False,
    filename: str | None = None,
    format: str | None = None,
    frame: bool = False,
    normalize: str | None = None,
    stacked: bool = False,
    color: list[str] | None = None,
    return_fig: bool = False,
    **kwargs
):
    """Plots one or more 1D NMR spectra contained within Spin objects."""

    if not all([s["ndim"] == 1 for s in spectra]):
        raise ValueError("All spectra must be 1-dimensional for bruker1d.")

    defaults = DEFAULTS.copy()
    defaults["yaxislabel"] = None
    defaults.update({k: v for k, v in kwargs if k in defaults and v is not None})

    spectra = spectra if isinstance(spectra, list) else [spectra]

    fig, ax = plt.subplots()

    plot_index = 0
    current_stack_offset = 0.0

    # Determine axis label from the first spectrum
    first_nuclei = spectra[0]["nuclei"]
    number, nucleus = (
        "".join(filter(str.isdigit, first_nuclei)),
        "".join(filter(str.isalpha, first_nuclei)),
    )

    for spectrum in spectra:
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

        # --- Stacking Logic (operates on selected data_to_plot) ---
        plot_data_adjusted = data_to_plot  # Start with selected data

        if stacked:
            # Apply the offset to plot_data_adjusted
            plot_data_adjusted = data_to_plot + current_stack_offset
            current_stack_offset += np.amax(data_to_plot) * 1.1

        # --- Plotting Logic ---
        plot_kwargs = {
            "linestyle": defaults["linestyle"],
            "linewidth": defaults["linewidth"],
            "alpha": defaults["alpha"],
        }
        if labels:
            plot_kwargs["label"] = (
                labels[plot_index]
                if plot_index < len(labels)
                else f"Spectrum {plot_index + 1}"
            )
        if color:
            plot_kwargs["color"] = (
                color[plot_index] if plot_index < len(color) else None
            )

        # Use plot_data_adjusted here, not data
        ax.plot(ppm, plot_data_adjusted, **plot_kwargs)

        plot_index += 1

    # --- Legend ---
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

    if defaults["tickspacing"]:
        ax.xaxis.set_major_locator(plt.MultipleLocator(defaults["tickspacing"]))

    if not frame:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_yticklabels([])
        ax.set_yticks([])
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

    if xlim:
        ax.set_xlim(xlim)
    else:
        # Auto-reverse axis if it looks like standard NMR
        current_xlim = ax.get_xlim()
        if current_xlim[0] < current_xlim[1]:  # Only reverse if not already reversed
            ax.set_xlim(current_xlim[::-1])

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


# Function to easily plot 1D NMR spectra in Bruker's format in a grid
def bruker1d_grid(
    spectra: dict | list[dict],
    labels=None,
    subplot_dims=(1, 1),
    xlim=None,
    save=False,
    filename=None,
    format="png",
    frame=False,
    normalize=False,
    color=None,
    return_fig=False,
    **kwargs
):
    """
    Plots 1D NMR spectra from Bruker data in subplots.

    Parameters:
        data_paths (list): List of paths to the Bruker data directories.
        labels (list): List of labels for the spectra.
        subplot_dims (tuple): Dimensions of the subplot grid (rows, cols).
        xlim (list of tuples or tuple): The limits for the x-axis.
        save (bool): Whether to save the plot.
        filename (str): The name of the file to save the plot.
        format (str): The format to save the file in.
        frame (bool): Whether to show the frame.
        normalize (str): Normalization method 'max', 'scans', or None.
        color (str): List of colors for the spectra.
        return_fig (bool): Whether to return the figure and axis.
        linewidth (float): Line width of the plot.
        linestyle (str): Line style of the plot.
        alpha (float): Alpha value for the plot.
        yaxislabel (str): Label for the y-axis.
        xaxislabel (str): Label for the x-axis.
        axisfontsize (int): Font size for the axis labels.
        axisfont (str): Font type for the axis labels.
        tickfontsize (int): Font size for the tick labels.
        tickfont (str): Font type for the tick labels.
        tickspacing (float): Spacing between the tick labels.

    Returns:
        None or tuple: If return_fig is True, returns the figure and axis.

    Example:
        bruker1d_grid(['data/1d_data1', 'data/1d_data2'], labels=['Spectrum 1', 'Spectrum 2'], subplot_dims=(1, 2), xlim=[(0, 100), (0, 100)], save=True, filename='1d_spectra', format='png', frame=False, normalize='max', color=['red', 'blue'])
    """

    spectra = spectra if isinstance(spectra, list) else [spectra]

    if not all([s["ndim"] == 1 for s in spectra]):
        raise ValueError("All spectra must be 1-dimensional for bruker1d_grid.")

    defaults = DEFAULTS.copy()
    defaults.update({k: v for k, v in kwargs.items() if k in defaults and v is not None})

    rows, cols = subplot_dims
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    data_paths = [s["path"] for s in spectra]
    for i, data_path in enumerate(data_paths):
        if i >= len(axes):
            break
        ax = axes[i]
        dic, data = ng.bruker.read_pdata(data_path)
        udic = ng.bruker.guess_udic(dic, data)

        nuclei = udic[0]["label"]
        number, nucleus = (
            "".join(filter(str.isdigit, nuclei)),
            "".join(filter(str.isalpha, nuclei)),
        )

        uc = ng.fileiobase.uc_from_udic(udic, dim=0)
        ppm = uc.ppm_scale()

        # Check if normalize is a list or a single value
        if isinstance(normalize, list):
            if len(normalize) != len(data_paths):
                raise ValueError(
                    "The length of the normalize list must be equal to the number of spectra."
                )
            normalize = normalize[i]

        if normalize == "max" or normalize:
            data = data / np.amax(data)
        elif normalize == "scans":
            ns = dic["acqus"]["NS"]
            if ns is None:
                raise ValueError("Number of scans not found.")
            data = data / ns
        elif normalize:
            raise ValueError(
                "Invalid value for normalize. Please provide 'max' or 'scans'."
            )

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

        if defaults["tickspacing"]:
            from matplotlib.ticker import MultipleLocator

            ax.xaxis.set_major_locator(MultipleLocator(defaults["tickspacing"]))

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

    # Plot projections with the extracted color
    ax["a"].plot(
        proj_f2[f"{f2_nuclei} {f2_units}"], proj_f2["F2 projection"], color="black"
    )
    ax["a"].axis(False)
    ax["b"].plot(
        -proj_f1["F1 projection"], proj_f1[f"{f1_nuclei} {f1_units}"], color="black"
    )
    ax["b"].axis(False)

    # Set axis labels with LaTeX formatting and non-italicized letters and position
    ax["A"].set_xlabel(f"$^{{{number_y}}}\\mathrm{{{nucleus_y}}}$ (ppm)", fontsize=13)
    ax["A"].set_ylabel(f"$^{{{number_x}}}\\mathrm{{{nucleus_x}}}$ (ppm)", fontsize=13)
    ax["A"].yaxis.set_label_position("right")
    ax["A"].yaxis.tick_right()
    ax["A"].tick_params(axis="x", labelsize=12)
    ax["A"].tick_params(axis="y", labelsize=12)

    # Set axis limits if provided
    if xlim:
        ax["A"].set_xlim(xlim)
        ax["a"].set_xlim(xlim)
    if ylim:
        ax["A"].set_ylim(ylim)
        ax["b"].set_ylim(ylim)

    if save:
        if filename:
            full_filename = filename + "." + format
        else:
            full_filename = "2d_nmr_spectrum." + format
        plt.savefig(
            full_filename, format=format, dpi=300, bbox_inches="tight", pad_inches=0.1
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
        color='b',
        linewidth=1,
        linestyle='-',
        alpha=1,
        model_show=True,
        model_color='red',
        model_linewidth=1,
        model_linestyle='--',
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
        return_fig=False
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
    dmfit_df = spectrum_info.get('dmfit_dataframe')

    if dmfit_df is None:
        raise ValueError("DMfit DataFrame not found in Spin object. Read data with provider='dmfit'")
    
    n_lines = sum(col.startswith('Line#') for col in dmfit_df.columns)

    defaults = {
        'color': color,
        'linewidth': linewidth,
        'linestyle': linestyle,
        'alpha': alpha,

        'model_show': model_show,
        'model_color': model_color,
        'model_linewidth': model_linewidth,
        'model_linestyle': model_linestyle,
        'model_alpha': model_alpha,

        'deconv_show': deconv_show,
        'deconv_color': deconv_color,
        'deconv_alpha': deconv_alpha,

        'frame': frame,
        'labels': labels,
        'labelsize': labelsize,
        'xlim': xlim,
        'save': save,
        'format': format,
        'filename': filename,
        'yaxislabel': yaxislabel,
        'xaxislabel': xaxislabel,
        'axisfontsize': axisfontsize,
        'axisfont': axisfont,
        'tickfontsize': tickfontsize,
        'tickfont': tickfont,
        'tickspacing': tickspacing,
        'return_fig': return_fig
    }
    
    params = {k: v for k, v in locals().items() if k in defaults and v is not None}
    params.update(defaults)

    fig, ax = plt.subplots()
    ax.plot(dmfit_df['ppm'], dmfit_df['Spectrum'], color=params['color'],
            linewidth=params['linewidth'], linestyle=params['linestyle'],
            alpha=params['alpha'],
            label=params['labels'][0] if params['labels'] and len(params['labels']) > 0 else None)
    if params['model_show']:
        ax.plot(dmfit_df['ppm'], dmfit_df['Model'], color=params['model_color'],
                linewidth=params['model_linewidth'], linestyle=params['model_linestyle'],
                alpha=params['model_alpha'],
                label=params['labels'][1] if params['labels'] and len(params['labels']) > 1 else None)
    if params['deconv_show']:
        for i in range(1, n_lines + 1):
            if params['deconv_color'] is not None:
                ax.fill_between(dmfit_df['ppm'], dmfit_df[f'Line#{i}'],
                                alpha=params['deconv_alpha'],
                                color=params['deconv_color'])
            else:
                ax.fill_between(dmfit_df['ppm'], dmfit_df[f'Line#{i}'],
                                alpha=params['deconv_alpha'])

    if params['labels']:
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=defaults["labelsize"],
            prop={"family": defaults["tickfont"], "size": defaults["labelsize"]},
        )
    if params['xlim']:
        ax.set_xlim(params['xlim'])
    if params['yaxislabel']:
        ax.set_ylabel(params['yaxislabel'], fontsize=params['labelsize'])
    if params['xaxislabel']:
        ax.set_xlabel(params['xaxislabel'], fontsize=params['labelsize'])
    if params['axisfontsize']:
        ax.xaxis.label.set_size(params['axisfontsize'])
        ax.yaxis.label.set_size(params['axisfontsize'])
    if params['axisfont']:
        ax.xaxis.label.set_fontname(params['axisfont'])
        ax.yaxis.label.set_fontname(params['axisfont'])
    if params['tickfontsize']:
        ax.tick_params(axis='both', which='major', labelsize=params['tickfontsize'])
        ax.tick_params(axis='both', which='minor', labelsize=params['tickfontsize'])
    if params['tickfont']:
        ax.tick_params(axis='both', which='major', labelfont=params['tickfont'])
        ax.tick_params(axis='both', which='minor', labelfont=params['tickfont'])
    if params['tickspacing']:
        ax.xaxis.set_major_locator(plt.MultipleLocator(params['tickspacing']))
        ax.yaxis.set_major_locator(plt.MultipleLocator(params['tickspacing']))
    if params['frame']:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
    if params['save']:
        if params['format']:
            plt.savefig(f"{params['filename']}.{params['format']}", format=params['format'])
        else:
            plt.savefig(params['filename'])

    if params['return_fig']:
        return fig, ax
    else:
        plt.show()
        return None 
