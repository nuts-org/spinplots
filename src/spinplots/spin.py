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
            case ("dmfit", 1, tuple()):
                raise ValueError("Grid layout is not supported for 1D DMFit spectra.")
            case ("dmfit", 2, None):
                return spinplot.dmfit2d(self, **kwargs)
            case ("dmfit", 2, tuple()):
                raise ValueError("Grid layout is not supported for 2D spectra.")
            case _:
                raise ValueError(
                    f"Plotting not supported for provider: {self.provider} with ndim={self.ndim}"
                )


class SpinCollection:
    """
    Represents a collection of Spin objects.

    Attributes:
        spins (dict): A dictionary of Spin objects.
        provider (str): The source of the NMR data (e.g., 'bruker'). Assumed
                        to be the same for all loaded spectra.
        ndim (int): The number of dimensions of the spectra.
        size (int): The number of Spin objects in the collection.
    """

    def __init__(self, spins: Spin | list[Spin]):
        if not spins:
            raise ValueError("Cannot initialize SpinCollection with empty spins list.")

        if isinstance(spins, Spin):
            spins = [spins]

        self.spins = {}
        self.provider = spins[0].provider
        self.ndim = spins[0].ndim
        self.size = 0
        self.append(spins)

    def append(self, spins: Spin | list[Spin]):
        """
        Appends Spin objects to the collection.

        Args:
            spins (Spin or list[Spin]): A Spin object or a list of Spin objects to append.
        """

        if isinstance(spins, Spin):
            spins = [spins]

        if not all(x.ndim == self.ndim for x in spins):
            raise ValueError("All Spin objects must have the same dimension.")

        if not all(x.provider == self.provider for x in spins):
            raise ValueError("All Spin objects must have the same provider.")

        for spin in spins:
            if spin.tag:
                tag = spin.tag
            else:
                tag = f"Spin{self.size}"
                warnings.warn(f"No tag provided. Using default tag: {tag}", UserWarning)
            if tag in self.spins:
                raise ValueError(
                    f"Spin with tag '{tag}' already exists in the collection."
                )
            self.spins[tag] = spin
            self.size += 1

    def remove(self, tag: str):
        """
        Removes a Spin object from the collection by its tag.

        Args:
            tag (str): The tag of the Spin object to remove.
        """

        if tag not in self.spins:
            raise KeyError(f"Spin with tag '{tag}' not found in the collection.")
        del self.spins[tag]
        self.size -= 1

    def __delitem__(self, tag: str):
        self.remove(tag)

    def __repr__(self) -> str:
        string = (
            f"SpinCollection with {self.size} spins ({self.provider} {self.ndim}D):\n"
        )
        for tag, spin in self.spins.items():
            string += f"  {tag}: {spin}\n"
        return string

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int | str) -> Spin:
        if isinstance(index, str):
            if index not in self.spins:
                raise KeyError(f"Spin with tag '{index}' not found.")
            return self.spins[index]

        if index < 0 or index >= self.size:
            raise IndexError("Index out of range.")

        return list(self.spins.values())[index]

    def __setitem__(self, key: str, spin: Spin):
        if key in self.spins:
            raise KeyError(f"Spin with tag '{key}' already exists.")

        self.spins[key] = spin
        self.size += 1

    def __iter__(self):
        return iter(self.spins.items())

    def plot(self, grid=None, filter=None, **kwargs):
        """
        Generates a plot of the NMR data stored in this collection.

        Args:
            grid (str, optional): Grid layout in format 'rows x cols' (e.g., '2x2', '1x3').
                    If provided, spectra will be plotted in a grid layout.
            filter (str | list[str], optional): A tag or list of tags to filter the spins
                    to be plotted
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

        spins_to_plot = self.spins
        if filter:
            if isinstance(filter, str):
                filter = [filter]
            if not all(tag in self.spins for tag in filter):
                raise KeyError(
                    f"One or more tags in {filter} not found in the collection."
                )
            spins_to_plot = {
                tag: self.spins[tag] for tag in filter if tag in self.spins
            }

        spectra = [spin.spectrum for spin in spins_to_plot.values()]

        if "labels" not in kwargs:
            kwargs["labels"] = list(spins_to_plot.keys())

        match (self.provider, self.ndim, subplot_dims):
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
                if len(spins_to_plot) > 1:
                    raise ValueError(
                        "DMFit plots can only handle one spectrum at a time."
                    )
                return spinplot.dmfit1d(list(self.spins.values())[0], **kwargs)
            case ("dmfit", 1, tuple()):
                raise ValueError("Grid layout is not supported for DMFit spectra.")
            case ("dmfit", 2, None):
                return spinplot.dmfit2d(self, **kwargs)
            case _:
                raise ValueError(
                    f"Plotting not supported for provider: {self.provider} with ndim={self.ndim}"
                )
