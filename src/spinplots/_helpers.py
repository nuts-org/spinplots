from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def _save_figure(fig, save, filename, format, default_name):
    """Save figure with consistent settings across all plot functions.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    save : bool
        Whether to save.
    filename : str or None
        User-provided filename (without extension).
    format : str or None
        File format (e.g., 'png', 'pdf'). Defaults to 'png'.
    default_name : str
        Default filename to use if filename is None.
    """
    if save:
        full_filename = f"{filename or default_name}.{format or 'png'}"
        fig.savefig(full_filename, dpi=300, bbox_inches="tight", pad_inches=0.1)


def _parse_nucleus(nuclei_string):
    """Parse a nucleus string like '13C' into ('13', 'C').

    Parameters
    ----------
    nuclei_string : str
        Nucleus identifier (e.g., '13C', '1H', '29Si').

    Returns
    -------
    number : str
        The mass number digits (e.g., '13').
    name : str
        The element name (e.g., 'C').
    """
    s = str(nuclei_string)
    number = "".join(filter(str.isdigit, s))
    name = "".join(filter(str.isalpha, s))
    return number, name


def _nucleus_label(nuclei_string):
    """Generate a LaTeX-formatted axis label from a nucleus string.

    Parameters
    ----------
    nuclei_string : str
        Nucleus identifier (e.g., '13C', '1H').

    Returns
    -------
    str
        Formatted label like '$^{13}$C (ppm)'.
    """
    number, name = _parse_nucleus(nuclei_string)
    return f"$^{{{number}}}${name} (ppm)"


def _apply_tick_spacing(ax, defaults):
    """Apply tick spacing from defaults to both axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to modify.
    defaults : dict
        Defaults dict containing 'xtickspacing', 'ytickspacing', 'tickspacing'.
    """
    xtick = defaults.get("xtickspacing") or defaults.get("tickspacing")
    if xtick:
        ax.xaxis.set_major_locator(MultipleLocator(xtick))

    ytick = defaults.get("ytickspacing") or defaults.get("tickspacing")
    if ytick:
        ax.yaxis.set_major_locator(MultipleLocator(ytick))


def _resolve_per_subplot(value, idx):
    """Resolve a parameter that may be a single value or per-subplot list.

    Parameters
    ----------
    value : tuple or list of tuples or None
        A single tuple (applied to all subplots) or a list of tuples
        (one per subplot).
    idx : int
        The current subplot index.

    Returns
    -------
    tuple or None
        The resolved value for this subplot.
    """
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    if isinstance(value, list) and idx < len(value):
        return value[idx]
    return None


def _handle_show_return(fig, return_value, return_fig, save):
    """Handle the common show/return/save logic at end of plot functions.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object.
    return_value : any
        The value to return if return_fig is True.
    return_fig : bool
        Whether to return figure objects.
    save : bool
        Whether the figure was saved.

    Returns
    -------
    The return_value if return_fig is True, else None.
    """
    if return_fig:
        return return_value

    if not save:
        plt.show()

    return None
