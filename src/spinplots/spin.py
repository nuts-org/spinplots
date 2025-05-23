from __future__ import annotations

import warnings

import spinplots.plot as spinplot


class Spin:
    """
    Represents one or more processed NMR datasets ready for plotting.

    Attributes:
        spectrum (dict): A dictionary containing the data
                        and metadata for a single spectrum. Keys include:
                        'data' (raw), 'norm_max', 'norm_scans',
                        'ppm_scale', 'hz_scale', 'nuclei', 'ndim',
                        'metadata', 'projections', 'path'.
        provider (str): The source of the NMR data (e.g., 'bruker')..
        ndim (int): The number of dimensions of the spectrum.
        tag (str): A tag for the spectrum, used for identification.
    """

    def __init__(
        self,
        spectrum_data: dict,
        provider: str,
        tag: str | None = None,
    ):
        if not spectrum_data:
            raise ValueError("Cannot initialize Spin object with empty spectrum data.")

        ndim = spectrum_data["ndim"]
        if ndim not in [1, 2]:
            raise ValueError(
                f"Unsupported number of dimensions in data: {ndim}. "
                "Only 1D and 2D spectra are supported."
            )

        provider = provider.lower()
        if provider not in ["bruker", "dmfit"]:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                "Only 'bruker' and 'dmfit' are supported."
            )

        self.spectrum = spectrum_data
        self.provider = provider
        self.ndim = ndim
        self.tag = tag

    def __repr__(self) -> str:
        path = self.spectrum["path"]
        return f"Spin(tag={self.tag}, ndim={self.ndim}, provider='{self.provider}', path={path})"

    def plot(self, grid=None, **kwargs):
        """
        Generates a plot of the NMR data stored in this object.

        Args:
            grid (str, optional): Grid layout in format 'rows x cols' (e.g., '2x2', '1x3').
                    If provided, spectra will be plotted in a grid layout.
            **kwargs: Plotting keyword arguments specific to the plot type
                    (e.g., xlim, labels, color, contour_start, etc.).
                    These are passed to the underlying plotting function.

        Returns:
            The result from the underlying plotting function.
        """

        subplot_dims = None
        if grid:
            try:
                rows, cols = map(int, grid.lower().split("x"))
            except (ValueError, AttributeError, TypeError) as e:
                raise ValueError(
                    f"Grid format should be 'rows x cols' (e.g., '2x2', '1x3'), got {grid}"
                ) from e
            subplot_dims = (rows, cols)

        match (self.provider, self.ndim, subplot_dims):
            case ("bruker", 1, None):
                return spinplot.bruker1d([self.spectrum], **kwargs)
            case ("bruker", 2, None):
                return spinplot.bruker2d([self.spectrum], **kwargs)
            case ("bruker", 1, tuple()):
                return spinplot.bruker1d_grid(
                    [self.spectrum], subplot_dims=subplot_dims, **kwargs
                )
            case ("bruker", 2, tuple()):
                raise ValueError("Grid layout is not supported for 2D spectra.")
            case ("dmfit", 1, None):
                return spinplot.dmfit1d(self, **kwargs)
            case ("dmfit", 2, None):
                return spinplot.dmfit2d(self, **kwargs)
            case ("dmfit", 1, tuple()):
                raise ValueError("Grid layout is not supported for 1D DMFit spectra.")
            case ("dmfit", 2, tuple()):
                raise ValueError("Grid layout is not supported for 2D DMFit spectra.")
            case _:
                raise ValueError(
                    f"Plotting not supported for provider: {self.provider} with ndim={self.ndim}"
                )


class SpinCollection:
    """
    Represents a collection of Spin objects that can be plotted together.

    Attributes:
        spins (list[Spin]): A list of Spin objects.
    """

    def __init__(self, spins: list[Spin]):
        if not spins:
            raise ValueError(
                "Cannot initialize SpinCollection with empty list of Spins."
            )

        self.spins = spins

    def __repr__(self) -> str:
        ndims = [spin.ndim for spin in self.spins]
        providers = [spin.provider for spin in self.spins]
        return f"SpinCollection(n_spins={len(self.spins)}, ndims={ndims}, providers={providers})"

    def plot(self, grid=None, **kwargs):
        """
        Generates a plot for all the Spin objects in this collection.

        Args:
            grid (str, optional): Grid layout in format 'rows x cols' (e.g., '2x2', '1x3').
                    If provided, spectra will be plotted in a grid layout.
            **kwargs: Plotting keyword arguments specific to the plot type
                    (e.g., xlim, labels, color, contour_start, etc.).
                    These are passed to the underlying plotting function.

        Returns:
            The result from the underlying plotting function.
        """
        # Check if all spins are of the same type (provider and ndim)
        all_providers = all(
            spin.provider == self.spins[0].provider for spin in self.spins
        )
        all_ndim = all(spin.ndim == self.spins[0].ndim for spin in self.spins)

        if not (all_providers and all_ndim):
            raise ValueError(
                "All Spin objects in the collection must have the same provider and ndim for plotting."
            )

        provider = self.spins[0].provider
        ndim = self.spins[0].ndim

        subplot_dims = None
        if grid:
            try:
                rows, cols = map(int, grid.lower().split("x"))
            except (ValueError, AttributeError, TypeError) as e:
                raise ValueError(
                    f"Grid format should be 'rows x cols' (e.g., '2x2', '1x3'), got {grid}"
                ) from e
            subplot_dims = (rows, cols)

        spectra = [spin.spectrum for spin in self.spins]

        match (provider, ndim, subplot_dims):
            case ("bruker", 1, None):
                return spinplot.bruker1d(spectra, **kwargs)
            case ("bruker", 2, None):
                return spinplot.bruker2d(spectra, **kwargs)
            case ("bruker", 1, tuple()):
                return spinplot.bruker1d_grid(
                    spectra, subplot_dims=subplot_dims, **kwargs
                )
            case ("bruker", 2, tuple()):
                raise ValueError("Grid layout is not supported for 2D spectra.")
            case ("dmfit", 1, None):
                return spinplot.dmfit1d(self, **kwargs)
            case ("dmfit", 2, None):
                return spinplot.dmfit2d(self, **kwargs)
            case ("dmfit", 1, tuple()):
                raise ValueError("Grid layout is not supported for 1D DMFit spectra.")
            case ("dmfit", 2, tuple()):
                raise ValueError("Grid layout is not supported for 2D DMFit spectra.")
            case _:
                raise ValueError(
                    f"Plotting not supported for provider: {provider} with ndim={ndim}"
                )
