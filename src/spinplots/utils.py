import nmrglue as ng
import numpy as np
import pandas as pd

# Function to read NMR data into a pandas DataFrame
def nmr_df(data_path, hz=False):
    """
    Reads Bruker's NMR 1D and 2D data and converts it into a pandas DataFrame.
    
    Parameters:
    data_path (str): Path to the NMR data.
    hz (bool): If True, use Hz scale instead of ppm scale. Default is False.
    
    Returns:
    pd.DataFrame: DataFrame containing the NMR data.
    """
    dic, data = ng.bruker.read_pdata(data_path)
    udic = ng.bruker.guess_udic(dic, data)
    ndim = udic['ndim']

    if ndim == 1:
        nuclei = udic[0]['label']
        uc = ng.fileiobase.uc_from_udic(udic, dim=0)
        ppm = uc.ppm_scale()
        hz = uc.hz_scale()
        ndata = data/np.max(data)
        df = pd.DataFrame({'hz': hz, 'ppm': ppm, 'intensity': data, 'norm_intensity': ndata, 'nuclei': nuclei})
        df.attrs['nmr_dim'] = ndim
    elif ndim == 2:
        nuclei = [udic[0]['label'], udic[1]['label']]
        uc_f1 = ng.fileiobase.uc_from_udic(udic, dim=0)
        uc_f2 = ng.fileiobase.uc_from_udic(udic, dim=1)
        ppm_f1 = uc_f1.ppm_scale()
        ppm_f2 = uc_f2.ppm_scale()
        hz_f1 = uc_f1.hz_scale()
        hz_f2 = uc_f2.hz_scale()
        
        if hz:
            df = pd.DataFrame(data)
            df.columns = hz_f2
            df.index = hz_f1
        else:
            df = pd.DataFrame(data)
            df.columns = ppm_f2
            df.index = ppm_f1

        df.attrs['nmr_dim'] = ndim
    else:
        raise ValueError('Only 1D and 2D NMR data are supported.')
    return df
