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
    Path to the Bruker experiment directory.
path_to_output : str
    Path for the output CSV file.
"""

from __future__ import annotations

import logging
import sys

from spinplots.utils import nmr_df


def main():
    """
    Convert Bruker's NMR data to csv files on the terminal
    """
    if len(sys.argv) != 3:
        logging.error("Incorrect number of arguments provided.")
        logging.error("Usage: bruker2csv <path_to_data> <path_to_output>")

        sys.exit(1)

    data_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        df_nmr = nmr_df(data_path)
        if df_nmr.attrs.get("nmr_dim") == 2:
            df_nmr.to_csv(output_path, index=True)
            logging.info(f"Data written to {output_path}")
        elif df_nmr.attrs.get("nmr_dim") == 1:
            df_nmr.to_csv(output_path, index=False)
            logging.info(f"Data written to {output_path}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
