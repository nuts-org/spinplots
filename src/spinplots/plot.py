"""This module contains functions to plot NMR spectra from Bruker data."""
import nmrglue as ng
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def bruker2d(data_path, contour_start, contour_num, contour_factor, cmap=None, xlim=None, ylim=None, save=False, filename=None, format=None):
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

    Example:
        plot_2d_nmr_spectrum('data/2d_data', 0.1, 10, 1.2, cmap='viridis', xlim=(0, 100), ylim=(0, 100), save=True, filename='2d_spectrum', format='png')
    """
    dic, data = ng.bruker.read_pdata(data_path)
    udic = ng.bruker.guess_udic(dic, data)
    
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
        contour_plot = ax['A'].contour(data, contour_levels, extent=(ppm_x_limits[0], ppm_x_limits[1], ppm_y_limits[0], ppm_y_limits[1]), cmap=cmap, linewidth=0.8)
        darkest_color = contour_plot.collections[0].get_edgecolor()[0]  # Get the color of the first contour line
    else:
        darkest_color = 'black'
        contour_plot = ax['A'].contour(data, contour_levels, extent=(ppm_x_limits[0], ppm_x_limits[1], ppm_y_limits[0], ppm_y_limits[1]), colors = 'black', linewidth=0.8)
    
    # Plot projections with the extracted color
    ax['a'].plot(ppm_x, proj_x, linewidth=0.8, color=darkest_color)
    ax['a'].axis(False)
    ax['b'].plot(-proj_y, ppm_y, linewidth=0.8, color=darkest_color)
    ax['b'].axis(False)
    
    # Set axis labels with LaTeX formatting and non-italicized letters and position
    ax['A'].set_xlabel(f'$^{{{number_x}}}\\mathrm{{{nucleus_x}}}$ (ppm)', fontsize=14)
    ax['A'].set_ylabel(f'$^{{{number_y}}}\\mathrm{{{nucleus_y}}}$ (ppm)', fontsize=14)
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

    
    # Show the plot or save it
    if save:
        if filename:
            full_filename = filename + "." + format
        else:
            full_filename = "2d_nmr_spectrum." + format
        plt.savefig(full_filename, format=format, dpi=300, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()

    return ax


# Function to easily plot 1D NMR spectra in Bruker's format
def bruker1d(data_paths, labels=None, xlim=None, save=False, filename=None, format=None, frame=False, normalized=False, stacked=False):
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

    Example:
        plot_1d_nmr_spectra(['data/1d_data1', 'data/1d_data2'], labels=['Spectrum 1', 'Spectrum 2'], xlim=(0, 100), save=True, filename='1d_spectra', format='png')
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
        if normalized:
            data = data / np.amax(data)

        # Stack the spectra
        if stacked:
            data += i * 1.1 if normalized else prev_max

        # Plot the spectrum
        if labels:
            ax.plot(ppm, data, label=labels[i])
            ax.legend()
        else:
            ax.plot(ppm, data)

        prev_max = np.amax(data)
    
    # Set axis labels with LaTeX formatting and non-italicized letters
    ax.set_xlabel(f'$^{{{number}}}\\mathrm{{{nucleus}}}$ (ppm)', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)

    # Remove frame
    if not frame:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticklabels([])
        ax.set_yticks([])
    else:
        ax.set_ylabel('Intensity (a.u.)', fontsize=14)
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
    else:
        fig.show()

    return fig, ax

# Function to easily plot 1D NMR spectra in Bruker's format in a grid
def bruker1d_grid(data_paths, labels=None, subplot_dims=(1, 1), xlim=None, save=False, filename=None, format='png', frame=False, normalized=False):
    """
    Plots 1D NMR spectra from Bruker data in subplots.

    Parameters:
        data_paths (list): List of paths to the Bruker data directories.
        labels (list): List of labels for the spectra.
        subplot_dims (tuple): Dimensions of the subplot grid (rows, cols).
        xlim (tuple): The limits for the x-axis.
        save (bool): Whether to save the plot.
        filename (str): The name of the file to save the plot.
        format (str): The format to save the file in.
        frame (bool): Whether to show the frame.
        normalized (bool): Whether to normalize the spectra.
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
        
        if normalized:
            data = data / np.amax(data)
        
        if labels:
            ax.plot(ppm, data, label=labels[i])
            ax.legend()
        else:
            ax.plot(ppm, data)
        
        ax.set_xlabel(f'$^{{{number}}}\\mathrm{{{nucleus}}}$ (ppm)', fontsize=14)
        ax.tick_params(axis='x', labelsize=12)

        if not frame:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_yticklabels([])
            ax.set_yticks([])
        else:
            ax.set_ylabel('Intensity (a.u.)', fontsize=14)
            ax.tick_params(axis='y', labelsize=12)

        if xlim:
            ax.set_xlim(xlim)

    plt.tight_layout()

    if save:
        if filename:
            full_filename = filename + "." + format
        else:
            full_filename = "1d_nmr_spectra." + format
        fig.savefig(full_filename, format=format, dpi=300, bbox_inches='tight', pad_inches=0.1)
    else:
        fig.show()

    return fig, axes
