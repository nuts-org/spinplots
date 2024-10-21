# SpinPlots

Welcome to the documentation for `SpinPlots`! This Python package is designed to simplify the process of reading and plotting NMR data. While [NMRglue](https://www.nmrglue.com/) is a powerful library for reading, processing, and analyzing NMR data, creating basic plots often requires a lot of code. `SpinPlots` streamlines these tasks, allowing you to produce clean, publication-ready plots with minimal effort.

For example, to generate a simple 1D NMR plot using NMRglue, you would typically write the following:

```python
# Import python libraries
import nmrglue as ng
import matplotlib.pyplot as plt
import numpy as np

# Read Bruker's processed data
dic, data = ng.bruker.read_pdata("Data/1/pdata/1/")

# Get universal dic
udic = ng.bruker.guess_udic(dic, data)

# Create a unit conversion object for the axis
uc = ng.fileiobase.uc_from_udic(udic)

# Get ppm scale
ppm_scale = uc.ppm_scale()

# Plot the spectrum
plt.plot(ppm_scale, data, label='Sample X')
plt.xlim(20, 0)
plt.xlabel('$^1$H (ppm)')
plt.ylabel('Intensity (a.u.)')
plt.legend()

# Save the figure
fig.savefig("spectrum.png")
```

This is a lot of code for a simple 1D plot! If one wants to make a 2D plot with projections things become even harder

`SpinPlots` tries to simplify the process. With just one function call, you can create the same 1D plot:

```python
from spinplots.plot import bruker1d

bruker1d(['Data/1/pdata/1/'], labels=['Sample X'], xlim=(20, 0), save=True, filename='spectrum', format='png')
```



