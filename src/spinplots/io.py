# filepath: src/spinplots/io.py
from __future__ import annotations

import re
import warnings
from pathlib import Path

import nmrglue as ng
import numpy as np
import pandas as pd

from spinplots.spin import Spin, SpinCollection
from spinplots.utils import calculate_projections


def read_nmr(
    path: str | list[str],
    provider: str = "bruker",
    tags: str | list[str] | None = None,
    **kwargs,
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
                f"{nuclei_y} F1 ppm": ppm_f1.flatten(),
                f"{nuclei_x} F2 ppm": ppm_f2.flatten(),
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

    with Path(path).open() as file:
        first_lines = "".join([file.readline() for _ in range(10)])
        ndim_2 = "##N_F1" in first_lines or "##N_F2" in first_lines

    if ndim_2:
        with Path(path).open() as file:
            lines = file.readlines()

        params = {}
        data_start_line = 0
        for i, line in enumerate(lines):
            if "##DATA##" in line:
                data_start_line = i + 1
                break
            if line.startswith("##"):
                parts = line.strip().split("=", 1)
                if len(parts) == 2:
                    params[parts[0]] = parts[1]

        n_f2 = int(float(params.get("##N_F2", "512")))
        n_f1 = int(float(params.get("##N_F1", "512")))

        x0_f2 = float(params.get("##X0_F2", "0"))
        dx_f2 = float(params.get("##dX_F2", "1"))
        x0_f1 = float(params.get("##X0_F1", "0"))
        dx_f1 = float(params.get("##dX_F1", "1"))

        x_axis = np.array([x0_f2 + i * dx_f2 for i in range(n_f2)])
        y_axis = np.array([x0_f1 + i * dx_f1 for i in range(n_f1)])

        data_values = []
        for line in lines[data_start_line:]:
            if line.strip():
                # Extract all numbers inline
                values = [
                    float(val)
                    for val in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                ]
                data_values.extend(values)

        expected_points = n_f1 * n_f2
        if len(data_values) < expected_points:
            warnings.warn(
                f"Expected {expected_points} data points, but found {len(data_values)}. Padding with zeros.",
                UserWarning,
            )
            data_values.extend([0.0] * (expected_points - len(data_values)))
        elif len(data_values) > expected_points:
            warnings.warn(
                f"Found {len(data_values)} data points, but expected {expected_points}. Trimming extra values.",
                UserWarning,
            )
            data_values = data_values[:expected_points]

        data_matrix = np.array(data_values).reshape(n_f1, n_f2)

        nuclei = kwargs.get("nuclei", ["Unknown", "Unknown"])
        if isinstance(nuclei, str):
            nuclei = [nuclei, nuclei]

        return {
            "path": path,
            "metadata": {"provider_type": "dmfit", "params": params},
            "ndim": 2,
            "data": data_matrix,
            "ppm_scale": (y_axis, x_axis),
            "projections": {
                "f1": np.max(data_matrix, axis=1),
                "f2": np.max(data_matrix, axis=0),
            },
            "nuclei": nuclei,
        }

    else:
        try:
            dmfit_df = pd.read_csv(path, sep="\t", skiprows=2)
        except Exception as e:
            raise OSError(f"Error reading DMfit data at path {path}: {e}") from e

        dmfit_df.columns = dmfit_df.columns.str.replace("##col_ ", "")

        ppm_scale = dmfit_df["ppm"].to_numpy()
        spectrum_data_values = dmfit_df["Spectrum"].to_numpy()

        ndim = 1

        nuclei = "Unknown"

        norm_max = (
            spectrum_data_values / np.max(spectrum_data_values)
            if np.max(spectrum_data_values) != 0
            else spectrum_data_values.copy()
        )

        return {
            "path": path,
            "metadata": {"provider_type": "dmfit"},
            "ndim": ndim,
            "data": spectrum_data_values,
            "norm_max": norm_max,
            "norm_scans": None,
            "projections": None,
            "ppm_scale": ppm_scale,
            "hz_scale": None,
            "nuclei": nuclei,
            "dmfit_dataframe": dmfit_df,
        }
