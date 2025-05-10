from __future__ import annotations


class Spin:
    """
    Represents one or more processed NMR datasets ready for plotting.

    Attributes:
        num_spectra (int): Number of spectra loaded in this object.
        spectra (list[dict]): A list where each dictionary contains the data
                              and metadata for a single spectrum. Keys include:
                              'data' (raw), 'norm_max', 'norm_scans',
                              'ppm_scale', 'hz_scale', 'nuclei', 'ndim',
                              'metadata', 'projections', 'path'.
        provider (str): The source of the NMR data (e.g., 'bruker'). Assumed
                        to be the same for all loaded spectra.
    """

    def __init__(
        self,
        spectra_data: list[dict],
        provider: str,
    ):
        if not spectra_data:
            raise ValueError("Cannot initialize Spin object with empty spectra data.")

        self.spectra = spectra_data
        self.num_spectra = len(spectra_data)
        self.provider = provider

        if not all(s["ndim"] == spectra_data[0]["ndim"] for s in spectra_data):
            raise ValueError("All spectra must have the same dimensionality.")
        else:
            self.ndim = self.spectra[0]["ndim"]

    def __repr__(self) -> str:
        paths = [s["path"] for s in self.spectra]
        return f"Spin(num_spectra={self.num_spectra}, ndim={self.ndim}, provider='{self.provider}', paths={paths})"

    def plot(self, grid=None, **kwargs):
        """
        Generates a plot of the NMR data stored in this object.

        If multiple spectra are stored, it attempts to plot them together
        (currently primarily supported for 1D).

        Args:
            grid (str, optional): Grid layout in format 'rows x cols' (e.g., '2x2', '1x3').
                    If provided, spectra will be plotted in a grid layout.
            **kwargs: Plotting keyword arguments specific to the plot type
                    (e.g., xlim, labels, color, contour_start, etc.).
                    These are passed to the underlying plotting function.

        Returns:
            The result from the underlying plotting function.

        Raises:
            ValueError: If plotting is not supported for the data's dimensionality,
                    provider, or if mixing dimensions.
            ImportError: If the required plotting functions cannot be imported.
        """
        try:
            from spinplots.plot import bruker1d, bruker2d, bruker1d_grid
        except ImportError as e:
            raise ImportError(f"Could not import plotting functions: {e}")

        # Check if all spectra have the same dimensionality
        if not all(s["ndim"] == self.ndim for s in self.spectra):
            raise ValueError(
                "Cannot plot spectra with different dimensionalities together using Spin.plot()."
            )

        if self.provider.lower() == "bruker":
            if self.ndim == 1:
                if grid is not None:
                    try:
                        rows, cols = map(int, grid.lower().split("x"))
                        subplot_dims = (rows, cols)
                    except (ValueError, AttributeError):
                        raise ValueError(
                            "Grid format should be 'rows x cols' (e.g., '2x2', '1x3')"
                        )

                    return bruker1d_grid([self], subplot_dims=subplot_dims, **kwargs)
                else:
                    return bruker1d([self], **kwargs)
            elif self.ndim == 2:
                if grid is not None:
                    raise ValueError("Grid layout is not supported for 2D spectra.")

                return bruker2d(self, **kwargs)
            else:
                raise ValueError(
                    f"Plotting not supported for {self.provider} data with ndim={self.ndim}"
                )
        else:
            raise ValueError(f"Plotting not supported for provider: {self.provider}")
