"""Command-line interface for SpinPlots.

Provides the ``bruker2csv`` command, which converts Bruker NMR data
to CSV files.

Usage
-----
.. code-block:: bash

    bruker2csv <path_to_data> <path_to_output>

Parameters
----------
path_to_data : str
    Path to the Bruker processed data directory (e.g. ``experiment/pdata/1``).
path_to_output : str
    Path for the output CSV file.
"""

from __future__ import annotations

import argparse
import sys

from spinplots.utils import nmr_df


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser of the ``bruker2csv`` command."""
    parser = argparse.ArgumentParser(
        prog="bruker2csv",
        description="Convert Bruker NMR data (1D or 2D) to a CSV file.",
    )
    parser.add_argument(
        "path_to_data",
        help="Path to the Bruker processed data directory (e.g. experiment/pdata/1).",
    )
    parser.add_argument("path_to_output", help="Path for the output CSV file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Convert Bruker's NMR data to csv files on the terminal.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments (without the program name). Defaults to
        ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status: 0 on success, 1 if the conversion failed.
    """
    args = build_parser().parse_args(argv)

    try:
        df_nmr = nmr_df(args.path_to_data)
        # No index column: the file can be read back, e.g. by
        # ``spinplots.calculate_projections``, whose first two columns must be
        # the F1 and F2 axes.
        df_nmr.to_csv(args.path_to_output, index=False)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        return 1

    print(f"Data written to {args.path_to_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
