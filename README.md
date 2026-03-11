# SpinPlots: Python NMR plots made easy

[![DOI](https://zenodo.org/badge/782205685.svg)](https://doi.org/10.5281/zenodo.14041925)

SpinPlots is a Python package designed to streamline the process of loading and plotting NMR data. Built on top of [NMRglue](https://www.nmrglue.com/), SpinPlots simplifies the plotting of NMR spectra. While NMRglue is a powerful tool for reading, processing, and analyzing NMR data, plotting with it can be cumbersome and code-intensive. SpinPlots addresses this by offering an intuitive interface for generating publication-ready plots with minimal effort.

## Quick Start

```python
from spinplots import read_nmr

# Plot a 1D spectrum
read_nmr("path/to/bruker/pdata/1").plot()

# Plot a 2D spectrum with projections
read_nmr("path/to/2d/pdata/1").plot(contour_start=1e5)
```

## Features 🤌

- **One line plotting**: Generate 1D and 2D NMR plots with a single function call
- **Multiple formats**: Support for Bruker and DMFit data
- **Automatic labeling**: Axes labeled with nuclei and units
- **Overlays and stacking**: Compare multiple spectra on one plot
- **Grid layouts**: Plot multiple spectra in customizable grid arrangements
- **2D projections**: Automatic F1/F2 projections for 2D spectra
- **Publication-ready**: Customizable styling with sensible defaults

## Documentation 📖

Complete documentation, including installation instructions and usage guides, can be found [here](https://nuts-org.github.io/spinplots/).

## Contributing 

Open to suggestions and contributions! Feel free to open an issue or pull request.
