"""This module contains functions to plot NMR spectra from Bruker data."""
import nmrglue as ng
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm
import pandas as pd
from spinplots.utils import calculate_projections

def bruker2d(data_path, contour_start, contour_num, contour_factor, cmap=None, xlim=None, ylim=None, save=False, filename=None, format=None, diag=None, homo=False, return_fig=False):
    """
    Plots a 2D NMR spectrum from Bruker data.

    Parameters:
        data_path (str): Path to the Bruker data directory.
        contour_start (float): The starting value for the contour lines.
        contour_num (int): The number of contour lines.
        contour_factor (float): The factor by which the contour levels increase.

    Keyword arguments:
        cmap (str): The colormap to use for the contour lines.
        xlim (tuple): The limits for the x-axis.
        ylim (tuple): The limits for the y-axis.
        save (bool): Whether to save the plot.
        filename (str): The name of the file to save the plot.
        format (str): The format to save the file in.
        diag (float or None): Slope of the diagonal line/None.
        homo (bool): True if doing homonuclear experiment.
        return_fig (bool): Whether to return the figure and axis.

    Example:
        bruker2d('data/2d_data', 0.1, 10, 1.2, cmap='viridis', xlim=(0, 100), ylim=(0, 100), save=True, filename='2d_spectrum', format='png', diag=True)
    """
    dic, data = ng.bruker.read_pdata(data_path)
    udic = ng.bruker.guess_udic(dic, data)

    # Check if homo is set to True
    if homo:
        nuclei_x = udic[1]['label']
        nuclei_y = udic[1]['label']
    else: 
        nuclei_x = udic[1]['label']
        nuclei_y = udic[0]['label']
    
    # Extract the number and nucleus symbol from the label
    number_x, nucleus_x = ''.join(filter(str.isdigit, nuclei_x)), ''.join(filter(str.isalpha, nuclei_x))
    number_y, nucleus_y = ''.join(filter(str.isdigit, nuclei_y)), ''.join(filter(str.isalpha, nuclei_y))
    
    uc_x = ng.fileiobase.uc_from_udic(udic, dim=1)
    ppm_x = uc_x.ppm_scale()
    ppm_x_limits = uc_x.ppm_limits()
    proj_x = np.amax(data, axis=0)
    
    uc_y = ng.fileiobase.uc_from_udic(udic, dim=0)
    ppm_y = uc_y.ppm_scale()
    ppm_y_limits = uc_y.ppm_limits()
    proj_y = np.amax(data, axis=1)
    
    # Create figure and axis
    ax = plt.figure(constrained_layout=False).subplot_mosaic(
    """
    .a
    bA
    """,
    gridspec_kw={"height_ratios": [0.9, 6.0], "width_ratios": [0.8, 6.0], 'wspace': 0.03, 'hspace': 0.04},
    )   
    
    # Contour levels
    contour_levels = contour_start * contour_factor ** np.arange(contour_num)
    
    # Plot contour lines with the provided colormap if cmap is provided
    if cmap is not None:
        contour_plot = ax['A'].contour(data, contour_levels, extent=(ppm_x_limits[0], ppm_x_limits[1], ppm_y_limits[0], ppm_y_limits[1]), cmap=cmap, linewidths=0.5, norm=LogNorm(vmin=contour_levels[0], vmax=contour_levels[-1]))
        darkest_color = contour_plot.collections[0].get_edgecolor()[0]  # Get the color of the first contour line
    else:
        darkest_color = 'black'
        contour_plot = ax['A'].contour(data, contour_levels, extent=(ppm_x_limits[0], ppm_x_limits[1], ppm_y_limits[0], ppm_y_limits[1]), colors = 'black', linewidths=0.5)
    
    # Plot projections with the extracted color
    ax['a'].plot(ppm_x, proj_x, linewidth=0.8, color=darkest_color)
    ax['a'].axis(False)
    ax['b'].plot(-proj_y, ppm_y, linewidth=0.8, color=darkest_color)
    ax['b'].axis(False)
    
    # Set axis labels with LaTeX formatting and non-italicized letters and position
    ax['A'].set_xlabel(f'$^{{{number_x}}}\\mathrm{{{nucleus_x}}}$ (ppm)', fontsize=13)
    ax['A'].set_ylabel(f'$^{{{number_y}}}\\mathrm{{{nucleus_y}}}$ (ppm)', fontsize=13)
    ax['A'].yaxis.set_label_position('right')
    ax['A'].yaxis.tick_right()
    ax['A'].tick_params(axis='x', labelsize=12)
    ax['A'].tick_params(axis='y', labelsize=12)

    # Plot diagonal line if diag is provided
    if diag is not None:
        x_diag = np.linspace(ppm_x_limits[0], ppm_x_limits[1], 100)
        y_diag = diag * x_diag
        ax['A'].plot(x_diag, y_diag, linestyle='--', color='gray')
    
    # Set axis limits if provided
    if xlim:
        ax['A'].set_xlim(xlim)
        ax['a'].set_xlim(xlim)
    if ylim:
        ax['A'].set_ylim(ylim)
        ax['b'].set_ylim(ylim)

    
    # Show the plot or save it
    if save:
        if filename:
            full_filename = filename + "." + format
        else:
            full_filename = "2d_nmr_spectrum." + format
        plt.savefig(full_filename, format=format, dpi=300, bbox_inches='tight', pad_inches=0.1)
    
    if return_fig:
        return ax
    else:
        plt.show()


# Function to easily plot 1D NMR spectra in Bruker's format
def bruker1d(data_paths, 
             labels=None,
             xlim=None,
             save=False,
             filename=None,
             format=None, frame=False,
             normalized=False,
             stacked=False,
             color=None,
             return_fig=False,
             background_paths=None,
             background_factors=None):
    """
    Plots 1D NMR spectra from Bruker data.

    Parameters:
        data_paths (list): List of paths to the Bruker data directories.

    Keyword arguments:
        labels (list): List of labels for the spectra.
        xlim (tuple): The limits for the x-axis.
        save (bool): Whether to save the plot.
        filename (str): The name of the file to save the plot.
        format (str): The format to save the file in.
        frame (bool): Whether to show the frame.
        normalized (bool): Whether to normalize the spectra.
        stacked (bool): Whether to stack the spectra.
        color (str): List of colors for the spectra.
        return_fig (bool): Whether to return the figure and axis.
        background_paths (list): List of paths to the Bruker background data directories.
        background_factors (list): List of factors to multiply the background by.

    Example:
        bruker1d(['data/1d_data1', 'data/1d_data2'], labels=['Spectrum 1', 'Spectrum 2'], xlim=(0, 100), save=True, filename='1d_spectra', format='png', frame=False, normalized=True, stacked=True, color=['red', 'blue'])
    """
    fig, ax = plt.subplots()
    
    nucleus_set = set()

    prev_max = 0
    for i, data_path in enumerate(data_paths):
        dic, data = ng.bruker.read_pdata(data_path)
        udic = ng.bruker.guess_udic(dic, data)
        
        nuclei = udic[0]['label']
        
        # Extract the number and nucleus symbol from the label
        number, nucleus = ''.join(filter(str.isdigit, nuclei)), ''.join(filter(str.isalpha, nuclei))

        # Check if the same nucleus is being used
        nucleus_set.add(nucleus)
        if len(nucleus_set) > 1:
            raise ValueError("All the spectra must be of the same nucleus.")
        
        uc = ng.fileiobase.uc_from_udic(udic, dim=0)
        ppm = uc.ppm_scale()
        ppm_limits = uc.ppm_limits()
        
        # Normalize the spectrum
        if normalized == 'max' or normalized == True:
            data = data / np.amax(data)
        elif normalized == 'scans':
            ns = dic['acqus']['NS']
            if ns is None:
                raise ValueError("Number of scans not found.")
            data = data / ns
        elif normalized:
            raise ValueError("Invalid value for normalized. Please provide 'max' or 'scans'.")
        
        # Remove background
        if background_paths is not None:
            if background_factors is None:
                raise ValueError("Background factors must be provided.")
            if i >= len(background_paths):
                raise ValueError("Number of background paths must be equal to the number of spectra.")
            


            background_path = background_paths[i]
            background_factor = background_factors[i]
            dic_bg, data_bg = ng.bruker.read_pdata(background_path)
            udic_bg = ng.bruker.guess_udic(dic_bg, data_bg)

            uc_bg = ng.fileiobase.uc_from_udic(udic_bg, dim=0)
            ppm_bg = uc_bg.ppm_scale()
            ppm_limits_bg = uc_bg.ppm_limits()

            if ppm.shape != ppm_bg.shape:
                raise ValueError("Data and background spectra must have the same dimensions.")

            # Normalize the background
            if normalized == 'max' or normalized == True:
                data_bg = data_bg / np.amax(data_bg)
            elif normalized == 'scans':
                ns_bg = dic_bg['acqus']['NS']
                if ns_bg is None:
                    raise ValueError("Number of scans not found.")
                data_bg = data_bg / ns_bg
            elif normalized:
                raise ValueError("Invalid value for normalized. Please provide 'max' or 'scans'.")

            # Remove the background
            data = data - background_factor * data_bg

        # Stack the spectra
        if stacked:
            data += i * 1.1 if normalized else prev_max

        # Plot the spectrum
        if labels and color:
            ax.plot(ppm, data, label=labels[i], color=color[i])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        elif labels:
            ax.plot(ppm, data, label=labels[i])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        elif color:
            ax.plot(ppm, data, color=color[i])
        else:
            ax.plot(ppm, data)

        prev_max = np.amax(data)
    
    # Set axis labels with LaTeX formatting and non-italicized letters
    ax.set_xlabel(f'$^{{{number}}}\\mathrm{{{nucleus}}}$ (ppm)', fontsize=13)
    ax.tick_params(axis='x', labelsize=12)

    # Remove frame
    if not frame:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticklabels([])
        ax.set_yticks([])
    else:
        ax.set_ylabel('Intensity (a.u.)', fontsize=13)
        ax.tick_params(axis='y', labelsize=12)

    # Set axis limits if provided
    if xlim:
        ax.set_xlim(xlim)

    # Show the plot or save it
    if save:
        if filename:
            full_filename = filename + "." + format
        else:
            full_filename = "1d_nmr_spectra." + format
        fig.savefig(full_filename, format=format, dpi=300, bbox_inches='tight', pad_inches=0.1)
    if return_fig:
        return fig, ax
    else:
        plt.show()

def bruker1d_background(data_path, background_path, background_factor, labels=None, xlim=None, save=False, filename=None, format=None, frame=False, normalized=False, color=None, return_fig=False):
    """
    Plots 1D NMR spectra from Bruker data with background removal.

    Parameters:
        data_path (str): Path to the Bruker data directory.
        background_path (str): Path to the Bruker background data directory.
        background_factor (float): Factor to multiply the background by.

    Keyword arguments:
        labels (list): List of labels for the spectra.
        xlim (tuple): The limits for the x-axis.
        save (bool): Whether to save the plot.
        filename (str): The name of the file to save the plot.
        format (str): The format to save the file in.
        frame (bool): Whether to show the frame.
        normalized (bool): Whether to normalize the spectra.
        color (str): List of colors for the spectra.
        return_fig (bool): Whether to return the figure and axis.
    """

    fig, ax = plt.subplots()
    
    dic, data = ng.bruker.read_pdata(data_path)
    udic = ng.bruker.guess_udic(dic, data)
    
    nuclei = udic[0]['label']
    
    # Extract the number and nucleus symbol from the label
    number, nucleus = ''.join(filter(str.isdigit, nuclei)), ''.join(filter(str.isalpha, nuclei))

    uc = ng.fileiobase.uc_from_udic(udic, dim=0)
    ppm = uc.ppm_scale()
    ppm_limits = uc.ppm_limits()
    
    # Normalize the spectrum
    if normalized == 'max' or normalized == True:
        data = data / np.amax(data)
    elif normalized == 'scans':
        ns = dic['acqus']['NS']
        if ns is None:
            raise ValueError("Number of scans not found.")
        data = data / ns
    elif normalized:
        raise ValueError("Invalid value for normalized. Please provide 'max' or 'scans'.")
    
    # Read the background data
    dic_bg, data_bg = ng.bruker.read_pdata(background_path)
    udic_bg = ng.bruker.guess_udic(dic_bg, data_bg)

    uc_bg = ng.fileiobase.uc_from_udic(udic_bg, dim=0)
    ppm_bg = uc_bg.ppm_scale()
    ppm_limits_bg = uc_bg.ppm_limits()

    if ppm.shape != ppm_bg.shape:
        raise ValueError("Data and background spectra must have the same dimensions.")

    # Normalize the background
    if normalized == 'max' or normalized == True:
        data_bg = data_bg / np.amax(data_bg)
    elif normalized == 'scans':
        ns_bg = dic_bg['acqus']['NS']
        if ns_bg is None:
            raise ValueError("Number of scans not found.")
        data_bg = data_bg / ns_bg
    elif normalized:
        raise ValueError("Invalid value for normalized. Please provide 'max' or 'scans'.")
    
    # Remove the background
    data = data - background_factor * data_bg

    # Plot the spectrum
    if labels is not None and color is not None:
        ax.plot(ppm, data, label=labels, color=color)
        ax.legend()
    elif labels is not None:
        ax.plot(ppm, data, label=labels)
        ax.legend()
    elif color is not None:
        ax.plot(ppm, data, color=color)
    else:
        ax.plot(ppm, data)

    # Set axis labels with LaTeX formatting and non-italicized letters
    ax.set_xlabel(f'$^{{{number}}}\\mathrm{{{nucleus}}}$ (ppm)', fontsize=13)
    ax.tick_params(axis='x', labelsize=12)

    # Remove frame
    if not frame:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticklabels([])
        ax.set_yticks([])
    else:
        ax.set_ylabel('Intensity (a.u.)', fontsize=13)
        ax.tick_params(axis='y', labelsize=12)

    # Set axis limits if provided
    if xlim:
        ax.set_xlim(xlim)

    # Show the plot or save it
    if save:
        if filename:
            full_filename = filename + "." + format
        else:
            full_filename = "1d_nmr_spectra." + format
        fig.savefig(full_filename, format=format, dpi=300, bbox_inches='tight', pad_inches=0.1)
    elif return_fig:
        return fig, ax
    else:
        plt.show()

# Function to easily plot 1D NMR spectra in Bruker's format in a grid
def bruker1d_grid(data_paths, labels=None, subplot_dims=(1, 1), xlim=None, save=False, filename=None, format='png', frame=False, normalized=False, color=None, return_fig=False):
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
        normalized (bool): Whether to normalize the spectra.
        color (str): List of colors for the spectra.
        return_fig (bool): Whether to return the figure and axis.

    Example:
        bruker1d_grid(['data/1d_data1', 'data/1d_data2'], labels=['Spectrum 1', 'Spectrum 2'], subplot_dims=(1, 2), xlim=[(0, 100), (0, 100)], save=True, filename='1d_spectra', format='png', frame=False, normalized=True, color=['red', 'blue'])
    """
    rows, cols = subplot_dims
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i, data_path in enumerate(data_paths):
        if i >= len(axes):
            break
        ax = axes[i]
        dic, data = ng.bruker.read_pdata(data_path)
        udic = ng.bruker.guess_udic(dic, data)
        
        nuclei = udic[0]['label']
        number, nucleus = ''.join(filter(str.isdigit, nuclei)), ''.join(filter(str.isalpha, nuclei))
        
        uc = ng.fileiobase.uc_from_udic(udic, dim=0)
        ppm = uc.ppm_scale()
        
        # Check if normalized is a list or a single value
        if isinstance(normalized, list):
            if len(normalized) != len(data_paths):
                raise ValueError("The length of the normalized list must be equal to the number of spectra.")
            normalized = normalized[i]

        if normalized == 'max' or normalized == True:
            data = data / np.amax(data)
        elif normalized == 'scans':
            ns = dic['acqus']['NS']
            if ns is None:
                raise ValueError("Number of scans not found.")
            data = data / ns
        elif normalized:
            raise ValueError("Invalid value for normalized. Please provide 'max' or 'scans'.")
        
        if labels and color:
            ax.plot(ppm, data, label=labels[i], color=color[i])
            ax.legend()
        elif labels:
            ax.plot(ppm, data, label=labels[i])
            ax.legend()
        elif color:
            ax.plot(ppm, data, color=color[i])
        else:
            ax.plot(ppm, data)
        
        ax.set_xlabel(f'$^{{{number}}}\\mathrm{{{nucleus}}}$ (ppm)', fontsize=13)
        ax.tick_params(axis='x', labelsize=12)

        if not frame:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_yticklabels([])
            ax.set_yticks([])
        else:
            ax.set_ylabel('Intensity (a.u.)', fontsize=13)
            ax.tick_params(axis='y', labelsize=12)

        if xlim and isinstance(xlim, tuple):
            ax.set_xlim(xlim)
        elif xlim and isinstance(xlim, list):
            ax.set_xlim(xlim[i])

    plt.tight_layout()

    if save:
        if filename:
            full_filename = filename + "." + format
        else:
            full_filename = "1d_nmr_spectra." + format
        fig.savefig(full_filename, format=format, dpi=300, bbox_inches='tight', pad_inches=0.1)
    elif return_fig:
        return fig, axes
    else:
        plt.show()

# Plot 2D NMR data from CSV or DataFrame
def df2d(path, contour_start, contour_num, contour_factor, cmap=None, xlim=None, ylim=None, save=False, filename=None, format=None, return_fig=False):
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
    if isinstance(path, pd.DataFrame):
        df = path
    else:
        df = pd.read_csv(path)
    
    cols = df.columns
    f1_nuclei, f1_units = cols[0].split( )
    number_x, nucleus_x = ''.join(filter(str.isdigit, f1_nuclei)), ''.join(filter(str.isalpha, f1_nuclei))
    f2_nuclei, f2_units = cols[1].split( )
    number_y, nucleus_y = ''.join(filter(str.isdigit, f2_nuclei)), ''.join(filter(str.isalpha, f2_nuclei))
    data_grid = df.pivot(index=cols[0], columns=cols[1], values='intensity')
    proj_f1, proj_f2 = calculate_projections(df, export=False)
    
    f1 = data_grid.index.values
    f2 = data_grid.columns.values
    x, y = np.meshgrid(f2, f1)
    z = data_grid.values

    contour_levels = contour_start * contour_factor ** np.arange(contour_num)

    ax = plt.figure(constrained_layout=False).subplot_mosaic(
    """
    .a
    bA
    """,
    gridspec_kw={"height_ratios": [0.9, 6.0], "width_ratios": [0.8, 6.0], 'wspace': 0.03, 'hspace': 0.04},
    )
    
    if cmap is not None:
        ax['A'].contourf(x, y, z, contour_levels, cmap=cmap, norm=LogNorm(vmin=contour_levels[0], vmax=contour_levels[-1]))
    else:
        ax['A'].contourf(x, y, z, contour_levels, cmap='black', norm=LogNorm(vmin=contour_levels[0], vmax=contour_levels[-1]))

    # Plot projections with the extracted color
    ax['a'].plot(proj_f2[f'{f2_nuclei} {f2_units}'], proj_f2['F2 projection'], color='black')
    ax['a'].axis(False)
    ax['b'].plot(-proj_f1['F1 projection'], proj_f1[f'{f1_nuclei} {f1_units}'], color='black')
    ax['b'].axis(False)

    # Set axis labels with LaTeX formatting and non-italicized letters and position
    ax['A'].set_xlabel(f'$^{{{number_y}}}\\mathrm{{{nucleus_y}}}$ (ppm)', fontsize=13)
    ax['A'].set_ylabel(f'$^{{{number_x}}}\\mathrm{{{nucleus_x}}}$ (ppm)', fontsize=13)
    ax['A'].yaxis.set_label_position('right')
    ax['A'].yaxis.tick_right()
    ax['A'].tick_params(axis='x', labelsize=12)
    ax['A'].tick_params(axis='y', labelsize=12)

    # Set axis limits if provided
    if xlim:
        ax['A'].set_xlim(xlim)
        ax['a'].set_xlim(xlim)
    if ylim:
        ax['A'].set_ylim(ylim)
        ax['b'].set_ylim(ylim)

    if save:
        if filename:
            full_filename = filename + "." + format
        else:
            full_filename = "2d_nmr_spectrum." + format
        plt.savefig(full_filename, format=format, dpi=300, bbox_inches='tight', pad_inches=0.1)
    elif return_fig:
        return ax
    else:
        plt.show()