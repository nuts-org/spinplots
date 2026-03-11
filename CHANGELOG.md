# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2]

### Added
- **Grid layouts for missing plot types**: `bruker2d_grid`, `dmfit1d_grid`, `dmfit2d_grid`
- **Stacked plots**: Support for stacking multiple 1D spectra (Bruker and DMFit)
- **Per-line deconvolution colors**: `deconv_color` accepts a list of colors for individual DMFit deconvolution lines
- **Per-subplot limits**: `xlim` and `ylim` accept list of tuples for per-subplot control in grid functions
- **Tag filtering**: `SpinCollection.plot(filter=...)` to plot a subset of spectra
- **Unknown kwarg warnings**: Plot functions now warn on unrecognized keyword arguments instead of silently ignoring them
- **Citation file**: Added `CITATION.cff` for standardized citation metadata (#31)
- **Default tags**: `SpinCollection.append()` auto-generates tags (`Spin0`, `Spin1`, ...) when none provided

### Changed
- **API consistency**: Renamed parameters for consistency across all plot functions:
  - `colors` → `color`, `proj_colors` → `proj_color`
  - `homo` → `homonuclear`, `diag` → `diagonal`
  - `labels` → `titles` in `dmfit1d_grid` (for subplot titles)
- **Unified input types**: All plot functions now accept spectrum dicts; `Spin.plot()` handles extraction
- **Standardized save logic**: All functions use consistent defaults (PNG format, dpi=300)
- **Projection keys**: Standardized to `f1`/`f2` across Bruker and DMFit 2D plots
- **SpinCollection.tag setter**: Now disallows `None` and syncs the internal dict key

### Fixed
- `read_nmr()` now accepts `tags="string"` for single paths (no longer requires a list)
- `__init__.py` exports all public plot functions
- `dmfit1d` now uses global DEFAULTS dict like other functions
- Extracted shared helpers to reduce code duplication in `plot.py`
- Color list indexing: `color` and `deconv_color` now use modulo indexing to cycle colors when fewer are provided than spectra or deconvolution lines
- Unreachable `elif` in `bruker2d`: moved `cmap`/`color` validation before the conditional chain
- Grid plots now correctly route `SpinCollection` tags to subplot `titles` instead of legend `labels`
- **Docs**: Fixed `mkdocstrings-python` 2.x configuration and excluded internal `_helpers` from reference docs

## [0.2.1]

### Fixed
- Fixed plotting of Bruker homonuclear spectra acquired without CP (cross-polarization). Reported by Andrej Šmelko (#25, #26)

## [0.2.0]

### Added
- **Spin object system**: Implemented a new object-oriented approach for handling NMR data:
  - `Spin` class for individual spectrum data
  - `SpinCollection` class to manage multiple spectra with efficient
- **Enhanced default management**: Defined default plot styling with possibility to override
- **Added read_nmr and .plot()**: Implemented unified way to read and plot data
- **DMFit Support**: Added functionality to read and plot data from DMFit
- **Added Testing**: Added test suite with improved coverage

### Changed
- **Code architecture**: Major refactoring for cleaner design and better maintainability:
  - Removed circular dependencies between modules
  - Improved function interfaces with clearer argument passing
  - Enhanced exception handling with more precise error messages
- **IO operations**: Redesigned data loading process:
  - Minimized try/except blocks for better error tracing
  - Streamlined exception handling with more specific error messages
  - Consolidated duplicate code for better maintainability

### Fixed
- Fixed issues with circular imports between `plot.py` and `spin.py`
- Improved error handling throughout the codebase

## [0.1.0]

## Added

- **Terminal functionality**: Added `bruker2csv` to convert NMR data into CSV from the terminal.
- **Plotting functions**: Added the following functions to streamline plotting:
    - `bruker1d` for generating 1D NMR plots
    - `bruker1d_grid` for generating subplots
    - `bruker2d` for generating 2D NMR plots
- **NMR dataframe**: Added the `nmr_df` function to create a Pandas DataFrame from NMR data, for further data manipulation, analysis and plotting.
- **Tutorials**:
    - Creating 1D and 2D plots using the spinplots functions.
    - Obtain a Pandas DataFrame from NMR data for custom plot styling and manipulation.

## [0.0.1]

### Added

- The initial release!
