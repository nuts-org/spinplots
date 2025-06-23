from __future__ import annotations

import nmrglue as ng
import numpy as np
import pandas as pd


# Function to read NMR data into a pandas DataFrame
def nmr_df(data_path, hz=False, export=False, filename=None):
    """
    Reads Bruker's NMR 1D and 2D data and converts it into a pandas DataFrame.

    Parameters:
    data_path (str): Path to the NMR data.
    hz (bool): If True, use Hz scale instead of ppm scale. Default is False.
    export (bool): If True, export the DataFrame to a CSV file. Default is False.
    filename (str): Name of the exported CSV file. Default is
    'nmr_data.csv'.

    Returns:
    pd.DataFrame: DataFrame containing the NMR data.
    """
    dic, data = ng.bruker.read_pdata(data_path)
    udic = ng.bruker.guess_udic(dic, data)
    ndim = udic["ndim"]

    if ndim == 1:
        nuclei = udic[0]["label"]
        uc = ng.fileiobase.uc_from_udic(udic, dim=0)
        ppm = uc.ppm_scale()
        hz = uc.hz_scale()
        ndata = data / np.max(data)
        df_nmr = pd.DataFrame(
            {
                "hz": hz,
                "ppm": ppm,
                "intensity": data,
                "norm_intensity": ndata,
                "nuclei": nuclei,
            }
        )
        df_nmr.attrs["nmr_dim"] = ndim
    elif ndim == 2:
        nuclei = [udic[0]["label"], udic[1]["label"]]
        uc_f1 = ng.fileiobase.uc_from_udic(udic, dim=0)
        uc_f2 = ng.fileiobase.uc_from_udic(udic, dim=1)
        ppm_f1 = uc_f1.ppm_scale()
        ppm_f2 = uc_f2.ppm_scale()
        hz_f1 = uc_f1.hz_scale()
        hz_f2 = uc_f2.hz_scale()

        if hz:
            F1_hz, F2_hz = np.meshgrid(hz_f1, hz_f2, indexing="ij")
            df_nmr = pd.DataFrame(
                {
                    f"{nuclei[0]} hz": F1_hz.flatten(),
                    f"{nuclei[1]} hz": F2_hz.flatten(),
                    "intensity": data.flatten(),
                }
            )

        else:
            F1_ppm, F2_ppm = np.meshgrid(ppm_f1, ppm_f2, indexing="ij")
            df_nmr = pd.DataFrame(
                {
                    f"{nuclei[0]} ppm": F1_ppm.flatten(),
                    f"{nuclei[1]} ppm": F2_ppm.flatten(),
                    "intensity": data.flatten(),
                }
            )

        df_nmr.attrs["nmr_dim"] = ndim
    else:
        raise ValueError("Only 1D and 2D NMR data are supported.")

    if export:
        if filename is None:
            filename = "nmr_data.csv"
        df_nmr.to_csv(filename, index=False)
        return None
    elif not export:
        return df_nmr
    return None


# Function to calculte projections from CSV or DataFrame
def calculate_projections(data, export=False, filename=None):
    """
    Calculate F1 and F2 projections from a DataFrame or a CSV file.

    Parameters:
    data (pd.DataFrame or str): DataFrame or path to a CSV file.
    export (bool): If True, export the projections to a CSV file. Default is False.
    filename (str): Name of the exported CSV file. Default is 'projections_f1.csv'
    and 'projections_f2.csv'.

    Returns:
    pd.DataFrame: DataFrame containing the F1 and F2 projections.
    """
    if isinstance(data, str):
        df_nmr = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df_nmr = data
    else:
        raise ValueError("Data must be a DataFrame or a path to a CSV file.")

    cols = df_nmr.columns
    f1_nuclei, f1_units = cols[0].split()[0], cols[0].split()[-1]
    f2_nuclei, f2_units = cols[1].split()[0], cols[1].split()[-1]
    data = (
        df_nmr["intensity"]
        .to_numpy()
        .reshape(len(df_nmr[cols[0]].unique()), len(df_nmr[cols[1]].unique()))
    )

    f1_proj = np.max(data, axis=1)
    f2_proj = np.max(data, axis=0)

    # Define df_pos that might have different sizes
    df_f1 = pd.DataFrame(
        {
            f"{f1_nuclei} {f1_units}": df_nmr[cols[0]].unique(),
            "F1 projection": f1_proj,
        }
    )

    df_f2 = pd.DataFrame(
        {
            f"{f2_nuclei} {f2_units}": df_nmr[cols[1]].unique(),
            "F2 projection": f2_proj,
        }
    )

    if export:
        if filename is None:
            filename = "projections"
        df_f1.to_csv(f"{filename}_f1.csv", index=False)
        df_f2.to_csv(f"{filename}_f2.csv", index=False)
    elif not export:
        return df_f1, df_f2

    return None, None
