from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

_PACKAGE_DIR = Path(__file__).resolve().parent


def _in_package(filename, package_dir=_PACKAGE_DIR):
    """Whether ``filename`` is a file of the package or of one of its subpackages.

    Pseudo-filenames such as ``<string>`` (used by the interactive interpreter,
    ``exec`` and some notebook front ends) never belong to the package.
    """
    if filename.startswith("<") and filename.endswith(">"):
        return False
    return package_dir in Path(filename).resolve().parents


def _warn(message, category=UserWarning):
    """Issue a warning attributed to the first stack frame outside spinplots.

    The warning then points at the line in the user's own code, whether a plot
    function was called directly or through ``Spin.plot`` / ``SpinCollection.plot``.

    Parameters
    ----------
    message : str
        The warning message.
    category : type[Warning], optional
        The warning category. Defaults to ``UserWarning``.
    """
    stacklevel = 2  # 1 would be this line, 2 is the caller of _warn
    frame = sys._getframe(1)
    while frame is not None and _in_package(frame.f_code.co_filename):
        frame = frame.f_back
        stacklevel += 1
    warnings.warn(message, category, stacklevel=stacklevel)


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


def _handle_show_return(return_value, return_fig, save):
    """Handle the common show/return/save logic at end of plot functions.

    Parameters
    ----------
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
