# filepath: src/spinplots/io.py
from __future__ import annotations

import nmrglue as ng
import numpy as np
import pandas as pd
import os
import warnings

from spinplots.spin import Spin
from spinplots.utils import calculate_projections


def read_nmr(path: str | list[str], provider: str = "bruker", **kwargs) -> Spin:
    """
    Reads NMR data from a specified path or list of paths and provider,
    returning a single Spin object containing all datasets.

    Args:
        path (str | list[str]): Path or list of paths to the NMR data directory(ies).
        provider (str): The NMR data provider (currently only 'bruker' is supported).
        **kwargs: Additional provider-specific arguments passed to the reader
                  (e.g., 'homo' for Bruker 2D).

    Returns:
        Spin: A Spin object containing the data for all successfully read spectra.

    Raises:
        ValueError: If the provider is not supported or data dimensionality is invalid/mixed.
        FileNotFoundError: If a path does not exist or data cannot be read.
        TypeError: If path is not a string or list of strings.
    """
    provider = provider.lower()

    if provider not in ["bruker", "dmfit"]:
        raise ValueError(
            f"Unsupported provider: {provider}. Only 'bruker' and 'dmfit"
        )

    if isinstance(path, str):
        paths_to_read = [path]
    elif isinstance(path, list):
        paths_to_read = path
    else:
        raise TypeError(
            f"Input path must be a string or a list of strings, not {type(path)}"
        )

    all_spectra_data = []
    first_ndim = None

    for p in paths_to_read:
        if not isinstance(p, str):
            raise TypeError(
                f"All items in the path list must be strings. Found: {type(p)}"
            )

        spectrum_data = {}
        
        if provider == "bruker":
            if not os.path.isdir(p):
                raise FileNotFoundError(f"Data directory not found {p}")

            spectrum_data = _read_bruker_data(p, provider, **kwargs)
        elif provider == "dmfit":
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Data file not found {p}")

            spectrum_data = _read_dmfit_data(p, **kwargs)

        # Check for consistent of dimensions across files
        if first_ndim is None:
            first_ndim = spectrum_data["ndim"]
        elif spectrum_data["ndim"] != first_ndim:
            raise ValueError(
                f"Cannot load spectra with mixed dimensionalities ({first_ndim}D and {spectrum_data['ndim']}D) into a single Spin object."
            )

        all_spectra_data.append(spectrum_data)

    if not all_spectra_data:
        raise ValueError("No spectra were successfully read.")

    return Spin(spectra_data=all_spectra_data, provider=provider)


def _read_bruker_data(path: str, provider: str, **kwargs) -> dict:
    """Helper function to read data for a single Bruker dataset."""
    try:
        dic, data = ng.bruker.read_pdata(path)
        udic = ng.bruker.guess_udic(dic, data)
        ndim = udic["ndim"]
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find Bruker data at path: {path}")
    except Exception as e:
        raise IOError(f"Error reading Bruker data at path {path}: {e}")

    # Handle data normalization
    norm_max_data = None
    norm_scans_data = None

    if ndim == 1:
        max_val = np.amax(data)
        if max_val != 0:
            norm_max_data = data / max_val
        else:
            norm_max_data = data.copy()

        try:
            ns = dic["acqus"]["NS"]
            if ns is not None and ns > 0:
                norm_scans_data = data / ns
            else:
                warnings.warn(
                    f"NS parameter is zero or missing in {path}. Cannot normalize by scans.",
                    UserWarning,
                )
                norm_scans_data = None
        except KeyError:
            warnings.warn(
                f"'acqus' or 'NS' key missing in metadata for {path}. Cannot normalize by scans.",
                UserWarning,
            )
            norm_scans_data = None
        except Exception as e:
            warnings.warn(
                f"Error during 'scans' normalization calculation for {path}: {e}",
                UserWarning,
            )
            norm_scans_data = None

    spectrum_data = {
        "path": path,
        "metadata": dic,
        "ndim": ndim,
        "data": data,
        "norm_max": norm_max_data,
        "norm_scans": norm_scans_data,
        "projections": None,
        "ppm_scale": None,
        "hz_scale": None,
        "nuclei": None,
    }

    if ndim == 1:
        uc = ng.fileiobase.uc_from_udic(udic, dim=0)
        spectrum_data["ppm_scale"] = uc.ppm_scale()
        spectrum_data["hz_scale"] = uc.hz_scale()
        spectrum_data["nuclei"] = udic[0]["label"]

    elif ndim == 2:
        homo = kwargs.get("homo", False)
        nuclei_y = udic[0]["label"]
        nuclei_x = udic[1]["label"]
        if homo:
            nuclei_y = nuclei_x

        spectrum_data["nuclei"] = (nuclei_y, nuclei_x)

        uc_y = ng.fileiobase.uc_from_udic(udic, dim=0)
        uc_x = ng.fileiobase.uc_from_udic(udic, dim=1)
        ppm_scale = (uc_y.ppm_scale(), uc_x.ppm_scale())
        hz_scale = (uc_y.hz_scale(), uc_x.hz_scale())
        spectrum_data["ppm_scale"] = ppm_scale
        spectrum_data["hz_scale"] = hz_scale

        # Calculate projections
        ppm_f1, ppm_f2 = np.meshgrid(ppm_scale[0], ppm_scale[1], indexing="ij")
        df_nmr_temp = pd.DataFrame(
            {
                f"{nuclei_y} ppm": ppm_f1.flatten(),
                f"{nuclei_x} ppm": ppm_f2.flatten(),
                "intensity": data.flatten(),
            }
        )
        proj_f1, proj_f2 = calculate_projections(df_nmr_temp, export=False)
        spectrum_data["projections"] = {"f1": proj_f1, "f2": proj_f2}

    else:
        raise ValueError(
            f"Unsupported NMR dimensionality: {ndim} found in {path}. Only 1D and 2D are supported."
        )

    return spectrum_data

def _read_dmfit_data(path: str, **kwargs) -> dict:
    """Helper function to read data of DMFit data."""
    try:
        dmfit_df = pd.read_csv(path, sep='\t', skiprows=2)
        dmfit_df.columns = dmfit_df.columns.str.replace('##col_ ', '')
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find DMfit data at path: {path}")
    except Exception as e:
        raise IOError(f"Error reading DMfit data at path {path}: {e}")

    ppm_scale = dmfit_df['ppm'].to_numpy()
    spectrum_data_values = dmfit_df['Spectrum'].to_numpy()
    
    ndim = 1

    nuclei = "Unknown" 

    spectrum_data = {
        "path": path,
        "metadata": {"provider_type": "dmfit"},
        "ndim": ndim,
        "data": spectrum_data_values,
        "norm_max": spectrum_data_values / np.amax(spectrum_data_values) if np.amax(spectrum_data_values) != 0 else spectrum_data_values.copy(),
        "projections": None,
        "ppm_scale": ppm_scale,
        "hz_scale": None,
        "nuclei": nuclei,
        "dmfit_dataframe": dmfit_df
    }
    return spectrum_data
