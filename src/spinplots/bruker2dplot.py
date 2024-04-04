# This module contains a function that plots a 2D NMR spectrum from Bruker data.

import nmrglue as ng
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def plot_2d_nmr_spectrum(data_path, contour_start, contour_num, contour_factor, cmap=None, xlim=None, ylim=None, save=False, filename=None, file_format=None):
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
        file_format (str): The format to save the file in.

    Example:
        plot_2d_nmr_spectrum('data/2d_data', 0.1, 10, 1.2, cmap='viridis', xlim=(0, 100), ylim=(0, 100), save=True, filename='2d_spectrum', file_format='png')
    """
    dic, data = ng.bruker.read_pdata(data_path)
    udic = ng.bruker.guess_udic(dic, data)
    
    nuclei_x = udic[1]['label']
    nuclei_y = udic[0]['label']
    
    # Extract the number and nucleus symbol from the label
    number_x, nucleus_x = int(nuclei_x[:-1]), nuclei_x[-1]
    number_y, nucleus_y = int(nuclei_y[:-1]), nuclei_y[-1]
    
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
        contour_plot = ax['A'].contour(data, contour_levels, extent=(ppm_x_limits[0], ppm_x_limits[1], ppm_y_limits[0], ppm_y_limits[1]), cmap=cmap)
        darkest_color = contour_plot.collections[0].get_edgecolor()[0]  # Get the color of the first contour line
    else:
        darkest_color = 'black'
        contour_plot = ax['A'].contour(data, contour_levels, extent=(ppm_x_limits[0], ppm_x_limits[1], ppm_y_limits[0], ppm_y_limits[1]), colors = 'black')
    
    # Plot projections with the extracted color
    ax['a'].plot(ppm_x, proj_x, linewidth=0.7, color=darkest_color)
    ax['a'].axis(False)
    ax['b'].plot(-proj_y, ppm_y, linewidth=0.7, color=darkest_color)
    ax['b'].axis(False)
    
    # Set axis labels with LaTeX formatting and non-italicized letters and position
    ax['A'].set_xlabel(f'$^{{{number_x}}}\\mathrm{{{nucleus_x}}}$ (ppm)', fontsize=14)
    ax['A'].set_ylabel(f'$^{{{number_y}}}\\mathrm{{{nucleus_y}}}$ (ppm)', fontsize=14)
    ax['A'].yaxis.set_label_position('right')
    ax['A'].yaxis.tick_right()
    ax['A'].set_xticks([20, 40, 60, 80, 100])
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
            full_filename = filename + "." + file_format
        else:
            full_filename = "2d_nmr_spectrum." + file_format
        plt.savefig(full_filename, format=file_format, dpi=300, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()
