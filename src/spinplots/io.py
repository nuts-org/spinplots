# filepath: src/spinplots/io.py
from __future__ import annotations

import warnings

import nmrglue as ng
import numpy as np
import pandas as pd

from spinplots.spin import Spin, SpinCollection
from spinplots.utils import calculate_projections


def read_nmr(
        path: str | list[str],
        provider: str = "bruker",
        tags: str | list[str] | None = None,
        **kwargs
) -> Spin | SpinCollection:
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
        ValueError: If the provider is not supported.
        IOError: If there are problems processing the files.
    """

    provider = provider.lower()

    paths_to_read = path if isinstance(path, list) else [path]

    if tags is not None and len(tags) != len(paths_to_read):
        raise ValueError("Length of tags must match the number of paths.")

    spins = []

    for i, p in enumerate(paths_to_read):
        match provider:
            case "bruker":
                spectrum_data = _read_bruker_data(p, **kwargs)
            case "dmfit":
                spectrum_data = _read_dmfit_data(p, **kwargs)
            case _:
                raise ValueError(
                    f"Unsupported provider: {provider}. Only 'bruker' and 'dmfit' are supported."
                )

        tag = tags[i] if tags is not None else None
        spin = Spin(spectrum_data=spectrum_data, provider=provider, tag=tag)
        spins.append(spin)

    if len(spins) == 1:
        return spins[0]

    return SpinCollection(spins)


def _read_bruker_data(path: str, **kwargs) -> dict:
    """Helper function to read data for a single Bruker dataset."""

    try:
        dic, data = ng.bruker.read_pdata(path)
    except OSError as e:
        raise OSError(f"Problem processing Bruker data at {path}: {e}") from e

    udic = ng.bruker.guess_udic(dic, data)
    ndim = udic["ndim"]

    # Handle data normalization
    norm_max_data = None
    norm_scans_data = None

    if ndim == 1:
        max_val = np.max(data)
        norm_max_data = data / max_val if max_val != 0 else data.copy()

        try:
            ns = dic["acqus"]["NS"]
            if ns is not None and ns > 0:
                norm_scans_data = data / ns
            else:
                warnings.warn(
                    f"NS parameter is zero or missing in {path}. Cannot normalize by scans.",
                    UserWarning,
                )
        except KeyError:
            warnings.warn(
                f"'acqus' or 'NS' key missing in metadata for {path}. Cannot normalize by scans.",
                UserWarning,
            )

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

    return {
        "path": path,
        "metadata": dic,
        "ndim": ndim,
        "data": data,
        "norm_max": norm_max_data,
        "norm_scans": norm_scans_data,
        "projections": spectrum_data["projections"],
        "ppm_scale": spectrum_data["ppm_scale"],
        "hz_scale": spectrum_data["hz_scale"],
        "nuclei": spectrum_data["nuclei"],
    }

def _read_dmfit_data(path: str, **kwargs) -> dict:
    """Helper function to read data of DMFit data."""

    try:
        dmfit_df = pd.read_csv(path, sep='\t', skiprows=2)
    except Exception as e:
        raise OSError(f"Error reading DMfit data at path {path}: {e}") from e

    dmfit_df.columns = dmfit_df.columns.str.replace('##col_ ', '')

    ppm_scale = dmfit_df['ppm'].to_numpy()
    spectrum_data_values = dmfit_df['Spectrum'].to_numpy()

    ndim = 1

    nuclei = "Unknown"

    norm_max = spectrum_data_values / np.max(spectrum_data_values) \
        if np.max(spectrum_data_values) != 0 else spectrum_data_values.copy()

    return {
        "path": path,
        "metadata": {"provider_type": "dmfit"},
        "ndim": ndim,
        "data": spectrum_data_values,
        "norm_max": norm_max,
        "projections": None,
        "ppm_scale": ppm_scale,
        "hz_scale": None,
        "nuclei": nuclei,
        "dmfit_dataframe": dmfit_df
    }
