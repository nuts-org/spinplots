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


def bruker1d(
    spectra: dict | list[dict],
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
    **kwargs,
):
    """
    Plots one or more 1D NMR spectra from spectrum dictionaries.

    Parameters:
        spectra (dict or list): Dictionary or list of dictionaries containing spectrum data.
        labels (list, optional): List of labels for the spectra.
        xlim (tuple, optional): The limits for the x-axis.
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
        current_xlim = ax.get_xlim()
        if current_xlim[0] < current_xlim[1]:
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
    **kwargs,
):
    """
    Plots 1D NMR spectra from Bruker data in subplots.

    Parameters:
        spectra (dict or list): Dictionary or list of dictionaries containing spectrum data.
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

        if defaults["tickspacing"]:
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
                    color=params["deconv_color"],
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
        ax.xaxis.label.set_size(params["axisfontsize"])
        ax.yaxis.label.set_size(params["axisfontsize"])
    if params["axisfont"]:
        ax.xaxis.label.set_fontname(params["axisfont"])
        ax.yaxis.label.set_fontname(params["axisfont"])
    if params["tickfontsize"]:
        ax.tick_params(axis="both", which="major", labelsize=params["tickfontsize"])
        ax.tick_params(axis="both", which="minor", labelsize=params["tickfontsize"])
    if params["tickfont"]:
        ax.tick_params(axis="both", which="major", labelfont=params["tickfont"])
        ax.tick_params(axis="both", which="minor", labelfont=params["tickfont"])
    if params["tickspacing"]:
        ax.xaxis.set_major_locator(plt.MultipleLocator(params["tickspacing"]))
        ax.yaxis.set_major_locator(plt.MultipleLocator(params["tickspacing"]))
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

    if hasattr(spin_objects, "spins"):
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
            main_ax.set_xlim(current_xlim_main[::-1])
    proj_ax_f2.set_xlim(main_ax.get_xlim())

    if ylim:
        main_ax.set_ylim(ylim)
    else:
        current_ylim_main = main_ax.get_ylim()
        if current_ylim_main[0] < current_ylim_main[1]:
            main_ax.set_ylim(current_ylim_main[::-1])
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
