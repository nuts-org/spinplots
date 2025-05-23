# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0]

### Added
- **Spin object system**: Implemented a new object-oriented approach for handling NMR data:
  - `Spin` class for individual spectrum data
  - `SpinCollection` class to manage multiple spectra with efficient
- **Enhanced default management**: Defined default plot styling with possibility to override
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
